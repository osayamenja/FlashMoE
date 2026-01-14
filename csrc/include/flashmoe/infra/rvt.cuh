//
// Created by osayamen on 1/12/26.
//

#ifndef FLASHMOE_RVT_CUH
#define FLASHMOE_RVT_CUH
namespace flashmoe
{
    constexpr int RED_MAX_ALIGNMENT = 16;
    template<typename Element, int Alignment>
    requires(Alignment > 0 && Alignment <= RED_MAX_ALIGNMENT && cutlass::is_pow2<Alignment>::value)
    struct RedAddType {
        using Type = Element;
        using Width = cute::Int<1>;
    };
    template<int Alignment>
    struct RedAddType<__half, Alignment> {
        // Alignment > sizeof(__half) means that Alignment is 2, 4, 8 or 16
        // This means we can safely promote to __half2
        using Type = cuda::std::conditional_t<(Alignment > sizeof(__half)), __half2, __half>;
        using Width = cute::Int<sizeof(Type) / sizeof(__half)>;
    };
    template<int Alignment>
    struct RedAddType<__nv_bfloat16, Alignment> {
        using Type = cuda::std::conditional_t<(Alignment > sizeof(__nv_bfloat16)), __nv_bfloat162, __nv_bfloat16>;
        using Width = cute::Int<sizeof(Type) / sizeof(__nv_bfloat16)>;
    };
    template<int Arch, typename Element, int VectorWidth>
    struct RedAdd {
        static_assert(VectorWidth >= 1 && VectorWidth <= (RED_MAX_ALIGNMENT / sizeof(Element)) &&
            cutlass::is_pow2<VectorWidth>::value);
        static_assert(Arch == 700 || Arch == 800 || Arch == 900);
        static_assert(cuda::std::is_same_v<Element, float> ||
            cuda::std::is_same_v<Element, __half> || cuda::std::is_same_v<Element, __nv_bfloat16>);
    };

    template<int MaxVectorWidth>
    struct RedAdd<700, float, MaxVectorWidth> {
        using VectorWidth = cute::Int<1>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, float>)
        void operator()(float* __restrict__ const& addr, const T& v) const {
            asm volatile("red.global.add.f32 [%0], %1;"
                     :
                     : "l"(addr), "f"(v[0])
                     : "memory");
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<700, __half, MaxVectorWidth> {
        using VectorWidth = cute::Int<1>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, __half>)
        void operator()(__half* __restrict__ const& addr, const T& v) const {
            asm volatile("red.global.add.noftz.f16 [%0], %1;"
                     :
                     : "l"(addr), "h"(v[0])
                     : "memory");
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<700, __half2, MaxVectorWidth> {
        using VectorWidth = cute::Int<1>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, __half2>)
        void operator()(__half2* __restrict__ const& addr, const T& v) const {
            // __half2 is packed 32-bit => use f16x2
            asm volatile("red.global.add.noftz.f16x2 [%0], %1;"
                         :
                         : "l"(addr), "r"(v[0])
                         : "memory");
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<800, float, MaxVectorWidth> {
        using VectorWidth = cute::Int<1>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, float>)
        void operator()(float* __restrict__ const& addr, const T& v) const {
            asm volatile("red.global.add.f32 [%0], %1;"
                     :
                     : "l"(addr), "f"(v[0])
                     : "memory");
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<800, __half, MaxVectorWidth> {
        using VectorWidth = cute::Int<1>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, __half>)
        void operator()(__half* __restrict__ const& addr, const T& v) const {
            asm volatile("red.global.add.noftz.f16 [%0], %1;"
                     :
                     : "l"(addr), "h"(v[0])
                     : "memory");
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<800, __half2, MaxVectorWidth> {
        using VectorWidth = cute::Int<1>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, __half2>)
        void operator()(__half2* __restrict__ const& addr, const T& v) const {
            asm volatile("red.global.add.noftz.f16x2 [%0], %1;"
                     :
                     : "l"(addr), "r"(v[0])
                     : "memory");
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<800, __nv_bfloat16, MaxVectorWidth> {
        using VectorWidth = cute::Int<1>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, __nv_bfloat16>)
        void operator()(__nv_bfloat16* __restrict__ const& addr, const T& v) const {
            atomicAdd(addr, v[0]);
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<800, __nv_bfloat162, MaxVectorWidth> {
        using VectorWidth = cute::Int<1>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, __nv_bfloat162>)
        void operator()(__nv_bfloat162* __restrict__ const& addr, const T& v) const {
            atomicAdd(addr, v[0]);
        }
    };

    template<int MaxVectorWidth>
    struct RedAdd<900, float, MaxVectorWidth> {
        using VectorWidth = cute::Int<cute::min(MaxVectorWidth, 4)>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, float>)
        void operator()(float* __restrict__ const& addr, const T& v) const {
            if constexpr (VectorWidth::value == 1) {
                asm volatile("red.global.add.f32 [%0], %1;"
                                         :
                                         : "l"(addr), "f"(v[0])
                                         : "memory");
            }
            else if (VectorWidth::value == 2) {
                asm volatile("red.global.v2.f32.add [%0], {%1, %2};"
                         :
                         : "l"(addr), "f"(v[0]), "f"(v[1])
                         : "memory");
            }
            else if (VectorWidth::value == 4) {
                asm volatile("red.global.v4.f32.add [%0], {%1, %2, %3, %4};"
                         :
                         : "l"(addr), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3])
                         : "memory");
            }
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<900, __half, MaxVectorWidth> {
        using VectorWidth = cute::Int<cute::min(MaxVectorWidth, 8)>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, __half>)
        void operator()(__half* __restrict__ const& addr, const T& v) const {
            if constexpr (VectorWidth::value == 1) {
                asm volatile("red.global.add.noftz.f16 [%0], %1;"
                             :
                             : "l"(addr), "h"(v[0])
                             : "memory");
            }
            else if constexpr (VectorWidth::value == 2) {
                asm volatile("red.global.v2.f16.add.noftz [%0], {%1, %2};"
                             :
                             : "l"(addr), "h"(v[0]), "h"(v[1])
                             : "memory");
            }
            else if constexpr (VectorWidth::value == 4) {
                asm volatile("red.global.v4.f16.add.noftz [%0], {%1, %2, %3, %4};"
                             :
                             : "l"(addr), "h"(v[0]), "h"(v[1]), "h"(v[2]), "h"(v[3])
                             : "memory");
            }
            else if constexpr (VectorWidth::value == 8) {
                asm volatile("red.global.v8.f16.add.noftz [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                             :
                             : "l"(addr),
                               "h"(v[0]), "h"(v[1]), "h"(v[2]), "h"(v[3]),
                               "h"(v[4]), "h"(v[5]), "h"(v[6]), "h"(v[7])
                             : "memory");
            }
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<900, __half2, MaxVectorWidth> {
        using VectorWidth = cute::Int<cute::min(MaxVectorWidth, 4)>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, __half2>)
        void operator()(__half2* __restrict__ const& addr, const T& v) const {
            if constexpr (VectorWidth::value == 1) {
                asm volatile("red.global.add.noftz.f16x2 [%0], %1;"
                             :
                             : "l"(addr), "r"(v[0])
                             : "memory");
            }
            else if constexpr (VectorWidth::value == 2) {
                asm volatile("red.global.v2.f16x2.add.noftz [%0], {%1, %2};"
                             :
                             : "l"(addr), "r"(v[0]), "r"(v[1])
                             : "memory");
            }
            else if constexpr (VectorWidth::value == 4) {
                asm volatile("red.global.v4.f16x2.add.noftz [%0], {%1, %2, %3, %4};"
                             :
                             : "l"(addr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3])
                             : "memory");
            }
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<900, __nv_bfloat16, MaxVectorWidth> {
        using VectorWidth = cute::Int<cute::min(MaxVectorWidth, 8)>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, __nv_bfloat16>)
        void operator()(__nv_bfloat16* __restrict__ const& addr, const T& v) const {
            if constexpr (VectorWidth::value == 1) {
                asm volatile("red.global.add.noftz.bf16 [%0], %1;"
                             :
                             : "l"(addr), "h"(v[0])
                             : "memory");
            }
            else if constexpr (VectorWidth::value == 2) {
                asm volatile("red.global.v2.bf16.add.noftz [%0], {%1, %2};"
                             :
                             : "l"(addr), "h"(v[0]), "h"(v[1])
                             : "memory");
            }
            else if constexpr (VectorWidth::value == 4) {
                asm volatile("red.global.v4.bf16.add.noftz [%0], {%1, %2, %3, %4};"
                             :
                             : "l"(addr), "h"(v[0]), "h"(v[1]), "h"(v[2]), "h"(v[3])
                             : "memory");
            }
            else if constexpr (VectorWidth::value == 8) {
                asm volatile("red.global.v8.bf16.add.noftz [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                             :
                             : "l"(addr),
                               "h"(v[0]), "h"(v[1]), "h"(v[2]), "h"(v[3]),
                               "h"(v[4]), "h"(v[5]), "h"(v[6]), "h"(v[7])
                             : "memory");
            }
        }
    };
    template<int MaxVectorWidth>
    struct RedAdd<900, __nv_bfloat162, MaxVectorWidth> {
        using VectorWidth = cute::Int<cute::min(MaxVectorWidth, 4)>;
        template<typename T>
        requires(cuda::std::is_same_v<typename T::value_type, __nv_bfloat162>)
        void operator()(__nv_bfloat162* __restrict__ const& addr, const T& v) const {
            if constexpr (VectorWidth::value == 1) {
                asm volatile("red.global.add.noftz.bf16x2 [%0], %1;"
                             :
                             : "l"(addr), "r"(v[0])
                             : "memory");
            }
            else if constexpr (VectorWidth::value == 2) {
                asm volatile("red.global.v2.bf16x2.add.noftz [%0], {%1, %2};"
                             :
                             : "l"(addr), "r"(v[0]), "r"(v[1])
                             : "memory");
            }
            else if constexpr (VectorWidth::value == 4) {
                asm volatile("red.global.v4.bf16x2.add.noftz [%0], {%1, %2, %3, %4};"
                             :
                             : "l"(addr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3])
                             : "memory");
            }
        }
    };
}
#endif //FLASHMOE_RVT_CUH