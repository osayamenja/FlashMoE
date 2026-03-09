def get_arch(device=0):
    from cuda.bindings import runtime as cudart

    err, major = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device
    )
    assert err == cudart.cudaError_t.cudaSuccess
    err, minor = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device
    )
    assert err == cudart.cudaError_t.cudaSuccess
    return (int(major) * 10) + int(minor)

def init_nvshmem():
    import nvshmem.core as nvshmem
    
    pass