import flashmoe

from cuda.bindings import runtime as cudart

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

if __name__ == "__main__":
    s = 16
    h = 32
    i = 32
    ec = 32
    expert_map = []
    rank_map = []
    # initialize NVSHMEM

    device_id = 0
    err, = cudart.cudaSetDevice(device_id)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(err)

    err, stream = cudart.cudaStreamCreate()
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(err)

    arch = get_arch(device_id)
    print("Arch is", arch)

    arg = flashmoe.InitializeArgs(s=s, h=h, i=i, ec=ec, arch=arch,
                                  expert_map=expert_map, rank_map=rank_map, stream_ptr=int(stream))
    flashmoe.initialize(arg)
    err, = cudart.cudaStreamDestroy(stream)
    assert err == cudart.cudaError_t.cudaSuccess