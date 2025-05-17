# Run
## Requirements
- Install NVSHMEM from [here](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html). This installation also includes `nvshmrun`, which is needed for running within a node.
- Install CPM as [so](https://github.com/cpm-cmake/CPM.cmake?tab=readme-ov-file#adding-cpm). Make sure to create the `cmake` directory as they recommend.
- Install CMake.
- (Optional but recommended) Install ninja
## CMake Build
- cd `csrc`
- mkdir `cmake-build-release` && cd `cmake-build-release`
- Configure `aristos_config.json` as needed.
- Run `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=<path to ninja> -Wno-dev -G Ninja -S .. -B .`
- Run `cmake --build . --target aristos -j`
### Single Node
- Run `nvshmrun -n <number of processes> -ppn <processes per node> ./aristos`
