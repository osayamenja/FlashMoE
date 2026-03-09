from nvMatmulHeuristics import NvMatmulHeuristicsInterface, \
                            NvMatmulHeuristicsTarget, \
                            NvMatmulHeuristicsMatmulLayout, \
                            NvMatmulHeuristicsFlags, \
                            NvMatmulHeuristicsNvidiaGpu

if __name__ == "__main__":
    # Load interface
    nvmmh = NvMatmulHeuristicsInterface(NvMatmulHeuristicsTarget.CUTLASS3,
                                        precision='TST',
                                        flags=NvMatmulHeuristicsFlags.PERF_MODEL_BASED_AUTO_TUNING)

    # Create Hardware descriptor
    # hw can be "None" to use the system's GPU instead
    # hw = nvmmh.createHardwareDescriptor()
    # nvmmh.setHardwarePredefinedGpu(hw, NvMatmulHeuristicsNvidiaGpu.A100_SXM_80GB)
    hw = None

    # Select layout
    layout = NvMatmulHeuristicsMatmulLayout.TN_ROW_MAJOR

    # Load internal discovery set for improved accuracy
    assert nvmmh.loadInternalDiscoverySet(layout, hw)

    # Get best configurations
    configs = nvmmh.get_with_mnk(4096, 2048, 2048, layout, 1, hw)

    # Print results
    print(f"Found {len(configs)} configurations:")
    for i, config in enumerate(sorted(configs, key=lambda d: d['runtime']), 1):
       #print(f"Configuration {i}:")
       print(f"Kernel: {config['kernel']}")
       #print(f"  Estimated runtime: {config['runtime'] * 1000:.6f} ms")