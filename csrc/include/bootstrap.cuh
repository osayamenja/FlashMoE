//
// Created by oja7 on 11/12/24.
//

#ifndef BOOTSRAP_CUH
#define BOOTSTRAP_CUH

#include <algorithm>
#include <vector>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>
#include <torch/torch.h>

#include "topo.cuh"
#include "debug.cuh"
#include "types.cuh"
#include "moe/expert.cuh"

#define SUPPORTED = 1;
namespace aristos{
    template<typename Element>
    __host__ __forceinline__
    void estimateMemory(const InitialConfig& iC, WorkerAttribute* __restrict__ const& dWa) {
        // estimate available device memory
        size_t free = 0, total = 0;
        CHECK_ERROR_EXIT(cudaMemGetInfo(&free, &total));
        // Deduct cost for the dense case, assuming at least one expert per device
        const auto bP = iC.bytesPerParameter<Element>();
        free -= bP * (iC.numParameters + iC.p2pBuffer);
        const size_t mX = cute::ceil_div(iC.numLayers, iC.moeFrequency) * bP * 2UL * (iC.hiddenProjDim * iC.embedDim);
        dWa->memoryCapacity = free / mX;
    }

    template<
        unsigned int Arch,
        typename Element
    >
    __host__ __forceinline__
    void measureThroughput(WorkerAttribute* __restrict__ cosnt& dWa,
        const unsigned int& M, const unsigned int& N, const unsigned int& K) {
        // malloc memory for all matrices
        constexpr auto batch = 2U;
        cuda::std::byte* abc;
        constexpr auto eS = sizeof(Element);
        const auto stateSize = sizeof(uint) * (1 + M / BLOCK_M); // tileSync + dT
        const auto aSize = stateSize + eS * M * K;
        const auto abSize = aSize + eS * K * N;
        const auto abcSize = abSize + eS * M * N;
        const auto abcBSize = abcSize + eS * cute::max(N, K); // bias
        const auto abcBSSize = abcBSize + eS * M; // scale/combine weights
        CHECK_ERROR_EXIT(cudaMallocAsync(&abc, abcBSSize, aristosStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(abc, 0, abcBSSize, aristosStream));

        // memcpy to device memory

        auto* __restrict__ p = CAST_TO(uint, abc);
        const auto pS = cute::make_tuple(M, N, K);
        using Activation = cutlass::epilogue::thread::ReLU<Element>;
        constexpr auto blocks = Hardware<Arch>::blocks::value - 1U;
        auto* __restrict__ pBp = CAST_TO(Element, abc + aSize);
        auto* __restrict__ pDp = CAST_TO(Element, abc + abcSize);
        const auto pB = cuda::std::array<Element*, batch>{pBp, pBp};
        const auto pD = cuda::std::array<Element*, batch>{pDp, pDp};
        auto* __restrict__ pC = CAST_TO(Element, abc + abSize);
        auto* __restrict__ pSC = CAST_TO(Element, abc + abcBSize);
        expert<Arch, Activation, batch><<<blocks, ARISTOS_BLOCK_SIZE, 0, aristosStream>>>(pS, p,
            p + 1, CAST_TO(Element, abc + stateSize), pB, pC, pD, pSC, pSC);
        uint stage = 0;
        CHECK_ERROR_EXIT(cudaMemcpyAsync(&stage, p, sizeof(uint), cudaMemcpyDeviceToHost, aristosStream));
        CHECK_ERROR_EXIT(cudaFreeAsync(abc, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        // quantize to uint16_t, this should be safe as the value is very small: in the hundreds
        // we use uint due to the API requirement of CUDA atomicMin
        dWa->throughput = static_cast<uint16_t>(stage);
    }

    __host__ __forceinline__
    void discoverTopology(void* hAp, const uint& n, const uint& globalRank, const WorkerAttribute& lWa) {
        const auto aD = n * n;
        const size_t heapBytes = n * sizeof(flagsType) + aD * sizeof(floatPair)
        + sizeof(uint) * (n + 2) + n * BETA_BUFFER;
        auto* symHeap = nvshmem_calloc(heapBytes, sizeof(cuda::std::byte));
        // Pointer orchestration
        // Starting index of flags array
        auto* flags = CAST_TO(flagsType, symHeap);
        auto* adj = CAST_TO(floatPair, flags + n);
        // Navigate to our slice of the adjacency matrix
        auto* results = adj + globalRank * n;
        // Repurpose a subset of the symmetric heap for local storage.
        auto* attributes = CAST_TO(WorkerAttribute, adj + aD);
        static_assert(sizeof(WorkerAttribute) >= sizeof(uint));
        auto* syncArray = CAST_TO(uint, attributes + n);
        // Starting index of heap
        auto* sHeap = CAST_TO(cuda::std::byte, syncArray + 2);
        const auto remotePresent = [&n, &results] {
            for (int i = 0; i < n; ++i) {
                if (nvshmem_ptr(results, i) == nullptr) return true;
            }
            return false;
        };
        const auto isRemotePresent = remotePresent();
        const auto sharedSize = n * (sizeof(floatPair) + sizeof(unsigned int));
        topology::discover<<<ARISTOS_SUPER_BLOCK_SIZE, ARISTOS_BLOCK_SIZE, sharedSize, aristosStream>>>(n, globalRank,
            isRemotePresent, lWa, sHeap, flags, results, syncArray, attributes);
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hAp, adj, aD * sizeof(floatPair) + n * sizeof(uint),
            cudaMemcpyDeviceToHost, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        nvshmem_free(symHeap);
    }

    __host__ __forceinline__
    decltype(auto) runDecider(const InitialConfig& iC, const floatPair* __restrict__ const& aP,
        const WorkerAttribute* __restrict__ const& attributes, const uint& rank, const uint& world) {
        const auto mC = ModelConfig{
            iC.numLayers,
            iC.redAmount,
            iC.globalBatch,
            iC.miniBatch,
            iC.moeFrequency,
            iC.p2pBuffer,
            iC.gradBuffer,
        };
        std::vector<Worker> workers(world);
        for (uint i = 0; i < workers.size(); ++i) {
            const auto [t, m] = attributes[i];
            workers.emplace_back(i, t, m);
        }
        const auto adj = make_tensor(aP, make_layout(cute::make_shape(world, world), cute::LayoutRight{}));
        const auto dTg = decider::decide(adj, workers, iC.numExperts,
            iC.numExperts, mC);
        const auto peerTranslation = subsets(dTg, rank);
        std::vector<Expert> experts(iC.numExperts);
        for (uint i = 0; i < iC.numExperts; ++i) {
            // assuming homogenous experts, where each has normalized compute cost of 1
            experts.emplace_back(i, 1);
        }
        std::vector<Worker> expertParallelGroup(peerTranslation.size());
        auto epRank = 0U;
        for (uint i = 0; i < expertParallelGroup.size(); ++i) {
            const auto wRank = peerTranslation[i];
            if (wRank == rank) {
                epRank = i;
            }
            expertParallelGroup.emplace_back(i, workers[wRank].processingRate, workers[wRank].memoryCapacity);
        }
        const auto assignment = decider::assign(experts, expertParallelGroup);
        std::vector<uint> scratch(world);
        std::ranges::fill(scratch.begin(), scratch.end(), 0U);
        auto expertSlots = 0U;
        // compute expert slots for our group
        for (const auto &i : assignment) {
            const auto tally = scratch[i] + 1;
            scratch[i] = tally;
            expertSlots = cuda::std::max(tally, expertSlots);
        }
        auto numLocalExperts = scratch[epRank];
        if (peerTranslation.size() == world) {
            // everyone is in one group; thus, we end early
            // peer translation, sharding spec, expert slots
            return cuda::std::tuple{peerTranslation, assignment, epRank, expertSlots, numLocalExperts};
        }
        // Get other group ids
        std::unordered_set<uint> groups{};
        for (const auto& i: dTg) {
            groups.emplace(i);
        }
        const auto myGroup = dTg[rank];
        for (const auto& group : groups) {
            if (group != myGroup) {
                std::ranges::fill(scratch.begin(), scratch.end(), 0U);
                const auto oPt = subsets(dTg, group);
                std::vector<Worker> ePG(oPt.size());
                for (uint i = 0; i < ePG.size(); ++i) {
                    const auto wRank = oPt[i];
                    ePG.emplace_back(wRank, workers[wRank].processingRate,
                        workers[wRank].memoryCapacity);
                }
                for (const auto &i : decider::assign(experts, expertParallelGroup)) {
                    const auto tally = scratch[i] + 1;
                    scratch[i] = tally;
                    expertSlots = cuda::std::max(tally, expertSlots);
                }
            }
        }
        return cuda::std::tuple{peerTranslation, assignment, epRank, expertSlots, numLocalExperts};
    }

    template<unsigned int Arch, typename Element>
    requires(aristos::TensorValueType<Element> && SupportedArch<Arch>)
    __host__ __forceinline__
    void archSpecificInit(const InitialConfig& iC) {
        // initialize communication backend
        nvshmem_init();
        CUTE_CHECK_ERROR(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        const auto globalWorld = nvshmem_n_pes();
        const auto rank = nvshmem_my_pe();

        // Pointer to adjacency matrix and throughput of all devices
        const auto aD = globalWorld * globalWorld;
        auto* aP = std::calloc(globalWorld * globalWorld * sizeof(floatPair) +
            globalWorld * sizeof(WorkerAttribute), sizeof(cuda::std::byte));
        // measure device throughput
        auto* hA = static_cast<floatPair*>(aP);
        auto* wAp = CAST_TO(WorkerAttribute, hA + aD);
        estimateMemory<Element>(&wAp[rank]);
        measureThroughput<Arch, Element>(wAp + rank, iC.seqLen, iC.hiddenProjDim, iC.embedDim);
        discoverTopology(aP, globalWorld, globalWorld, wAp[rank]);

        // The topology adjacency matrix is ready, let's map devices to optimal cooperative process groups
        const auto [peerT, epSpec, epRank,
            expertSlots, numLocalExperts] = runDecider(iC, hA, wAp, rank, globalWorld);

        // Now allocate memory
        /// Symmetric memory
        const uint epWorld = peerT.size();
        const auto eCap = iC.shouldDrop? iC.expertCapacity(epWorld) : iC.seqLen;
        const auto heapBytes = STAGES * CELLS * epWorld * expertSlots * eCap * iC.embedDim * sizeof(Element);
        const auto tMc = Config::tiles<BLOCK_M>(eCap);
        const auto tN = Config::tiles<BLOCK_N>(iC.embedDim);
        const auto flagBytes = (epWorld * numLocalExperts + iC.numExperts * tMc * tN) * sizeof(flagsType);
        auto* sHeap = nvshmem_calloc(flagBytes + heapBytes, sizeof(cuda::std::byte));
        auto* sHb = static_cast<cuda::std::byte*>(sHeap);

        // local bookkeeping memory
        constexpr auto blocks = Hardware<Arch>::blocks::value - 1U;
        const auto bookSize = Bookkeeping::bookLength(iC.seqLen, iC.numExperts, numLocalExperts, iC.hiddenProjDim,
            iC.embedDim, eCap, blocks, epWorld, sizeof(Element));

        cuda::std::byte* book;
        cuda::barrier<cuda::thread_scope_device>* b;
        CHECK_ERROR_EXIT(cudaMallocAsync(&book, bookSize + sizeof(cuda::barrier<cuda::thread_scope_device>), aristosStream));
        // We allocate below separately due to alignment constraints
        CHECK_ERROR_EXIT(cudaMallocAsync(&b, sizeof(cuda::barrier<cuda::thread_scope_device>), aristosStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(&book, 0, bookSize, aristosStream));

        auto hB = new cuda::barrier<cuda::thread_scope_device>{blocks};
        CHECK_ERROR_EXIT(cudaMemcpyAsync(b, hB, sizeof(cuda::barrier<cuda::thread_scope_device>),
            cudaMemcpyHostToDevice, aristosStream));

        // Initialize bookkeeping
        hostBookkeeping = Bookkeeping{
            book,
            iC.seqLen * iC.miniBatch,
            iC.numExperts,
            numLocalExperts,
            iC.hiddenProjDim,
            iC.embedDim,
            eCap,
            blocks,
            epWorld,
            b,
            sizeof(Element)
        };
        CHECK_ERROR_EXIT(cudaMemcpyToSymbolAsync(bookkeeping, &hostBookkeeping, sizeof(Bookkeeping), 0,
            cudaMemcpyHostToDevice, aristosStream));

        const auto functionId = 8 * ((Arch - MIN_ARCH) / 100) + 4 * (iC.numExperts > BLOCK_N) +
            2 * iC.shouldDrop + (iC.k > 1);
        // Initialize config struct
        hostMoEConfig = Config{
            sHb + flagBytes,
            static_cast<flagsType *>(sHeap),
            functionId,
            epRank,
            iC.k,
            iC.embedDim,
            iC.numExperts,
            numLocalExperts,
            iC.seqLen * iC.miniBatch,
            epWorld,
            iC.hiddenProjDim,
            expertSlots,
            iC.capacityFactor
        };
        CHECK_ERROR_EXIT(cudaMemcpyToSymbolAsync(moeConfig, &hostMoEConfig, sizeof(Config), 0,
            cudaMemcpyHostToDevice, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        delete hB;
        std::free(aP);
    }
    // Should be called before loading the model
    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void aristosInit(const InitialConfig& iC) {
        reportError(!isInitialized, "Already Initialized");
        isInitialized = true;
        reportError(iC.embedDim % BLOCK_N == 0 && iC.hiddenProjDim % BLOCK_N == 0,
            "Must be multiple of BLOCK_N");
        reportError(iC.seqLen % BLOCK_M == 0, "Must be a multiple of BLOCK_M");
        int l2CacheSize = 0;
        int cudaDevAttribute = 0;
        int dev = 0;
        int blocks = 0;
        int arch = 0;
        CHECK_ERROR_EXIT(cudaGetDevice(&dev));
        CHECK_ERROR_EXIT(cudaDeviceGetAttribute(&cudaDevAttribute, cudaDevAttrMemoryPoolsSupported, dev));
        reportError(cudaDevAttribute, "Memory Pools support required");
        CHECK_ERROR_EXIT(cudaDeviceGetAttribute(&arch, cudaDevAttrComputeCapabilityMajor, dev));
        reportError(arch, ">= Volta is required");
        CHECK_ERROR_EXIT(cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, dev));
        CHECK_ERROR_EXIT(cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev));
        switch (arch) {
            case 7:
                archSpecificInit<700, Element>(iC);
            break;
            case 8:
                archSpecificInit<800, Element>(iC);
            break;
            default:
                archSpecificInit<900, Element>(iC);
                break;
        }
#if (__CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800)
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.base_ptr  = bookKeeping;
        stream_attribute.accessPolicyWindow.num_bytes = cuda::std::min(static_cast<size_t>(0.25 * l2CacheSize),
            sizeof(unsigned int) * numNeighbors + sizeof(unsigned int) * numExperts * 2);
        stream_attribute.accessPolicyWindow.hitRatio  = 1.0;
        stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

        //Set the attributes to a CUDA stream of type cudaStream_t
        cudaStreamSetAttribute(aristosStream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
#endif
    }

    template<typename Element>
    __host__ __forceinline__
    void dispatchForwardKernel(const torch::Tensor& activations, const torch::Tensor& moeOutput,
        const torch::Tensor& expertsUp, const torch::Tensor& expertsDown,
        const torch::Tensor& biasUp, const torch::Tensor& biasDown,
        const torch::Tensor& gateWeights, const torch::Tensor& gateOutput) {
        using ElementC = mp_t; // accumulate type
        const auto sl = hostBookkeeping.sl;
        const auto ed = hostBookkeeping.ed;
        const auto pd = hostBookkeeping.pd;
        const auto nLx = hostBookkeeping.nLx;
        const auto nx = hostBookkeeping.nx;
        const auto px = hostBookkeeping.px;
#if CHECK_TENSORS
        TORCH_CHECK(activations.scalar_type() == expertsUp.scalar_type());
        TORCH_CHECK(activations.scalar_type() == expertsDown.scalar_type());
        TORCH_CHECK(activations.scalar_type() == biasUp.scalar_type());
        TORCH_CHECK(activations.scalar_type() == biasDown.scalar_type());
        TORCH_CHECK(activations.scalar_type() == gateWeights.scalar_type());
        TORCH_CHECK(activations.scalar_type() == gateOutput.scalar_type());

        TORCH_CHECK(activations.is_contiguous())
        TORCH_CHECK(expertsUp.is_contiguous())
        TORCH_CHECK(expertsDown.is_contiguous())
        TORCH_CHECK(biasUp.is_contiguous())
        TORCH_CHECK(biasDown.is_contiguous())
        TORCH_CHECK(gateWeights.is_contiguous())
        TORCH_CHECK(gateOutput.is_contiguous())

        TORCH_CHECK(activations.is_cuda());
        TORCH_CHECK(expertsUp.is_cuda());
        TORCH_CHECK(expertsDown.is_cuda());
        TORCH_CHECK(biasUp.is_cuda());
        TORCH_CHECK(biasDown.is_cuda());
        TORCH_CHECK(gateWeights.is_cuda());
        TORCH_CHECK(gateOutput.is_cuda());

        TORCH_CHECK(activations.sizes()[0] == sl
            && activations.sizes()[1] == ed);
        TORCH_CHECK(gateWeights.size(0) == nx
            && gateWeights.size(1) == ed)
        TORCH_CHECK(gateOutput.size(0) == sl
            && gateWeights.size(1) == px)

        const auto eSzU = expertsUp.sizes();
        const auto eSzD = expertsDown.sizes();
        TORCH_CHECK(eSzU.size() == 3 && eSzD.size() == 3);
        TORCH_CHECK(eSzU[0] == nLx && eSzD[0] == nLx);
        // Weights are already transposed in memory
        TORCH_CHECK(eSzU[1] == pd && eSzD[1] == ed);
        TORCH_CHECK(eSzU[2] == ed && eSzD[3] == pd);
        const auto bSzU = biasUp.sizes();
        const auto bSzD = biasDown.sizes();
        TORCH_CHECK(bSzU.size() == 2 && bSzD.size() == 2);
        TORCH_CHECK(bSzU[0] == nLx && bSzD[0] == nLx);
        TORCH_CHECK(bSzU[1] == pd && bSzD[1] == ed);
#endif

        // Now construct cute tensors
        // Activations & Output
        const auto* __restrict__ aP = activations.const_data_ptr<Element>();
        const auto tA = cute::make_tensor(cute::make_gmem_ptr(aP),
            make_layout(cute::make_shape(sl, ed),
                cute::LayoutRight{}));
        const auto* __restrict__ oP = moeOutput.mutable_data_ptr<Element>();
        const auto tO = cute::make_tensor(cute::make_gmem_ptr(oP),
            make_layout(cute::make_shape(sl, ed),
                cute::LayoutRight{}));

        // Expert weights
        const auto* __restrict__ ePu = expertsUp.const_data_ptr<Element>();
        const auto tXu = cute::make_tensor(cute::make_gmem_ptr(ePu),
            make_layout(make_shape(nLx, cute::make_shape(pd, ed)), cute::LayoutRight{}));
        const auto* __restrict__ ePd = expertsDown.const_data_ptr<Element>();
        const auto tXd = cute::make_tensor(cute::make_gmem_ptr(ePd),
            make_layout(make_shape(nLx, cute::make_shape(ed, pd)), cute::LayoutRight{}));

        // Bias
        const auto* __restrict__ bPu = biasUp.const_data_ptr<Element>();
        const auto tBu = cute::make_tensor(cute::make_gmem_ptr(bPu),
            make_layout(make_shape(nLx, cute::make_shape(1, pd)), cute::LayoutRight{}));
        const auto* __restrict__ bPd = biasDown.const_data_ptr<Element>();
        const auto tBd = cute::make_tensor(cute::make_gmem_ptr(bPd),
            make_layout(make_shape(nLx, cute::make_shape(1, ed)), cute::LayoutRight{}));
        // Gate
        const auto* __restrict__ gP = gateWeights.const_data_ptr<Element>();
        const auto tG = cute::make_tensor(cute::make_gmem_ptr(gP),
            make_layout(cute::make_shape(nx, ed), cute::LayoutRight{}));
        const auto* __restrict__ gOp = gateWeights.const_data_ptr<Element>();
        const auto tGo = cute::make_tensor(cute::make_gmem_ptr(gOp),
            make_layout(cute::make_shape(sl, px), cute::LayoutRight{}));
        // Decode function id
        switch (hostMoEConfig.functionId) {
            case 0: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element>();
            }
            break;
            case 1: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element>();
            }
            break;
            case 2: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element>();
            }
            break;
            case 3: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element>();
            }
            break;
            case 4: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element>();
            }
            break;
            case 5: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element>();
            }
            break;
            case 6: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element>();
            }
            break;
            case 7: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<700, g, d, c, ElementC, Element>();
            }
            break;
            case 8: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element>();
            }
            break;
            case 9: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element>();
            }
            break;
            case 10: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element>();
            }
            break;
            case 11: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element>();
            }
            case 12: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element>();
            }
            break;
            case 13: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element>();
            }
            break;
            case 14: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element>();
            }
            break;
            case 15: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<800, g, d, c, ElementC, Element>();
            }
            break;
            case 16: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element>();
            }
            break;
            case 17: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element>();
            }
            break;
            case 18: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element>();
            }
            break;
            case 19: {
                constexpr auto g = GateReductionLevel::singleBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                moe::forward<900, g, d, c, ElementC, Element>();
            }
            break;
            case 20: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element>();
            }
            break;
            case 21: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::yes;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element>();
            }
            break;
            case 22: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::single;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element>();
            }
            break;
            case 23: {
                constexpr auto g = GateReductionLevel::multiBlock;
                constexpr auto d = DropTokens::no;
                constexpr auto c = CombineMode::multithreaded;
                // Call forward pass
                moe::forward<900, g, d, c, ElementC, Element>();
            }
            break;
            default:
                reportError(false, "No such function exists!");
        }
    }

    __host__ __forceinline__
    void forwardHost(const torch::Tensor& activations, const torch::Tensor& moeOutput,
        const torch::Tensor& expertsUp, const torch::Tensor& expertsDown,
        const torch::Tensor& biasUp, const torch::Tensor& biasDown,
        const torch::Tensor& gateWeights, const torch::Tensor& gateOutput){
        reportError(isInitialized, "Not initialized");
        switch (activations.scalar_type()) {
            case torch::kFloat: {
                if (at::globalContext().allowTF32CuBLAS() || at::globalContext().allowTF32CuDNN()) {
                    dispatchForwardKernel<cute::tfloat32_t>(activations, moeOutput, expertsUp, expertsDown,
                        biasUp, biasDown, gateWeights, gateOutput);
                }
                else {
                    dispatchForwardKernel<float>(activations, moeOutput, expertsUp, expertsDown,
                        biasUp, biasDown, gateWeights, gateOutput);
                }
            }
            break;
            case torch::kFloat16:
                dispatchForwardKernel<cute::half_t>(activations, moeOutput, expertsUp, expertsDown,
                    biasUp, biasDown, gateWeights, gateOutput);
            break;
            case torch::kBFloat16:
                dispatchForwardKernel<cute::bfloat16_t>(activations, moeOutput, expertsUp, expertsDown,
                    biasUp, biasDown, gateWeights, gateOutput);
            break;
            case torch::kFloat8_e4m3fn:
                dispatchForwardKernel<cute::float_e4m3_t>(activations, moeOutput, expertsUp, expertsDown,
                    biasUp, biasDown, gateWeights, gateOutput);
            break;
            case torch::kFloat8_e5m2:
                dispatchForwardKernel<cute::float_e5m2_t>(activations, moeOutput, expertsUp, expertsDown,
                    biasUp, biasDown, gateWeights, gateOutput);
            break;
            default:
                reportError(false, "Not supported!");
        }
    }

    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void backwardHost(){
        reportError(isInitialized, "Not initialized");
    }

    __host__ __forceinline__
    void aristosFinalize(){
        reportError(isInitialized, "Not initialized!");
        isInitialized = false;
        CHECK_ERROR_EXIT(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
#if (__CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800)
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.num_bytes = 0; // Setting the window size to 0 disable it
        // Overwrite the access policy attribute to a CUDA Stream
        cudaStreamSetAttribute(aristosStream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
        cudaCtxResetPersistingL2Cache();
#endif

        CHECK_ERROR_EXIT(cudaFreeAsync(hostBookkeeping.book, aristosStream));
        CHECK_ERROR_EXIT(cudaFreeAsync(hostBookkeeping.deviceBlockade, aristosStream));
        nvshmem_free(hostMoEConfig.sHeap);
        nvshmem_finalize();
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
    }
}
#endif //BOOTSTRAP_CUH
