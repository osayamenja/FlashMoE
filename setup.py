"""
Setup script for FlashMoE
Compiles CUDA kernels and creates pip-installable package
"""
import os
import sys
import json
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Environment variables for dependencies
NVSHMEM_HOME = os.environ.get('NVSHMEM_HOME', os.path.expanduser('~/.local/nvshmem'))
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')


def check_nvshmem():
    # Check for either static library or host shared library
    nvshmem_static = Path(NVSHMEM_HOME) / 'lib' / 'libnvshmem.a'
    nvshmem_host = Path(NVSHMEM_HOME) / 'lib' / 'libnvshmem_host.so'
    
    if not (nvshmem_static.exists() or nvshmem_host.exists()):
        print(f"WARNING: NVSHMEM not found at {NVSHMEM_HOME}")
        print("Set NVSHMEM_HOME environment variable or install NVSHMEM")
        print("See: https://docs.nvidia.com/nvshmem/")
        
        # Check if running in CI/non-interactive mode
        if not sys.stdin.isatty():
            print("\nRunning in non-interactive mode. Set NVSHMEM_HOME to continue.")
            print("Example: export NVSHMEM_HOME=/path/to/nvshmem")
            sys.exit(1)
        
        # Interactive mode - ask user
        try:
            user_input = input("\nWARNING: Continuing without NVSHMEM. Multi-GPU features will not work.\nContinue anyway? [y/N]: ")
            if user_input.lower() != 'y':
                sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\nInstallation cancelled.")
            sys.exit(1)
        
        return False
    return True


def download_dependencies():
    """
    Download required dependencies (CUTLASS, CCCL, etc.) if not already cached
    This mimics what CPM does in CMake
    """
    csrc_dir = Path('csrc')
    cache_dir = csrc_dir / 'cmake' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    dependencies = {
        'cutlass': {
            'url': 'https://github.com/NVIDIA/cutlass/archive/refs/heads/main.zip',
            'name': 'cutlass-main',
        },
        'cccl': {
            'url': 'https://github.com/NVIDIA/cccl/archive/refs/tags/v2.8.4.zip',
            'name': 'cccl-2.8.4',
        },
        'fmt': {
            'url': 'https://github.com/fmtlib/fmt/archive/refs/tags/11.1.4.zip',
            'name': 'fmt-11.1.4',
        },
        'nvtx': {
            'url': 'https://github.com/NVIDIA/NVTX/archive/refs/tags/v3.1.1-c-cpp.zip',
            'name': 'NVTX-3.1.1-c-cpp',
        },
        'matx': {
            'url': 'https://github.com/NVIDIA/MatX/archive/refs/tags/v0.9.1.zip',
            'name': 'MatX-0.9.1',
        },
    }
    
    import urllib.request
    import zipfile
    
    for dep_name, dep_info in dependencies.items():
        dep_cache = cache_dir / dep_name
        
        # Check if already downloaded
        if dep_cache.exists() and any(dep_cache.iterdir()):
            print(f"✓ {dep_name} already cached")
            continue
        
        print(f"Downloading {dep_name}...")
        dep_cache.mkdir(exist_ok=True)
        
        # Download zip file
        zip_path = cache_dir / f"{dep_name}.zip"
        try:
            urllib.request.urlretrieve(dep_info['url'], zip_path)
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dep_cache)
            
            # Remove zip file
            zip_path.unlink()
            
            print(f"✓ {dep_name} downloaded and extracted")
        except Exception as e:
            print(f"WARNING: Failed to download {dep_name}: {e}")
            print(f"You can manually run CMake to download dependencies:")
            print(f"  cd csrc && mkdir -p cmake-build-release && cd cmake-build-release")
            print(f"  cmake -DCMAKE_BUILD_TYPE=Release -G Ninja -S .. -B .")


def get_cuda_extensions():
    """
    Configure CUDA extension for FlashMoE
    This compiles your existing CUDA kernels into a Python extension
    """
    
    # Find all CUDA source files in csrc/
    csrc_dir = Path('csrc')
    
    print("\nChecking dependencies...")
    download_dependencies()
    
    cpm_cache = csrc_dir / 'cmake' / 'cache'
    
    sources = [
        'csrc/python_bindings.cu',
    ]
    
    include_dirs = [
        str(csrc_dir.absolute()),  # Use absolute path
        os.path.join(str(csrc_dir.absolute()), "include"),
        os.path.join(str(csrc_dir.absolute()), "include", "kleos"),
        os.path.join(NVSHMEM_HOME, 'include'),
        os.path.join(CUDA_HOME, 'include'),
    ]
    print(f"include_dirs: {include_dirs}")
    
    # Add dependencies from cache
    if cpm_cache.exists():
        cutlass_dirs = list(cpm_cache.glob('cutlass/*/include'))
        if cutlass_dirs:
            include_dirs.append(str(cutlass_dirs[0].absolute()))
            print(f"Using CUTLASS from: {cutlass_dirs[0].absolute()}")
        
        cccl_base_dirs = list(cpm_cache.glob('cccl/*'))
        if cccl_base_dirs:
            cccl_base = cccl_base_dirs[0]
            
            # Add libcudacxx
            if (cccl_base / 'libcudacxx' / 'include').exists():
                include_dirs.append(str((cccl_base / 'libcudacxx' / 'include').absolute()))
                print(f"Using CCCL/libcudacxx from: {(cccl_base / 'libcudacxx' / 'include').absolute()}")
            
            # Add thrust
            if (cccl_base / 'thrust').exists():
                include_dirs.append(str((cccl_base / 'thrust').absolute()))
                print(f"Using CCCL/thrust from: {(cccl_base / 'thrust').absolute()}")
            
            # Add cub
            if (cccl_base / 'cub').exists():
                include_dirs.append(str((cccl_base / 'cub').absolute()))
                print(f"Using CCCL/cub from: {(cccl_base / 'cub').absolute()}")
        
        # FMT
        fmt_dirs = list(cpm_cache.glob('fmt/*/include'))
        if fmt_dirs:
            include_dirs.append(str(fmt_dirs[0].absolute()))
            print(f"Using fmt from: {fmt_dirs[0].absolute()}")
        
        # NVTX3
        nvtx_dirs = list(cpm_cache.glob('nvtx/*/c/include'))
        if nvtx_dirs:
            include_dirs.append(str(nvtx_dirs[0].absolute()))
            print(f"Using NVTX3 from: {nvtx_dirs[0].absolute()}")
    
    # Library directories
    library_dirs = [
        os.path.join(NVSHMEM_HOME, 'lib'),
        os.path.join(CUDA_HOME, 'lib64'),
    ]
    
    # Libraries to link
    libraries = [
        'nvshmem_host',
        'cudart',
        'cublas',
        'cuda',
    ]
    
    # Compiler flags
    extra_compile_args = {
        'cxx': [
            '-O3',
            '-std=c++20',
            '-fPIC',
        ],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-std=c++20',
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda',
            '-rdc=true',
            '--cudart=shared',
            '-gencode', 'arch=compute_70,code=sm_70',  # V100
            '-gencode', 'arch=compute_80,code=sm_80',  # A100
            '-gencode', 'arch=compute_90,code=sm_90',  # H100
            '-Xcompiler', '-fPIC',
            "-Xfatbin", "-compress-all"
        ],
        # NOTE(byungsoo): This solves device linking issue while enabling RDC,
        # which is required by NVSHMEM
        'nvcc_dlink': [
            '-dlink',
            f'-L{NVSHMEM_HOME}/lib',
            '-lnvshmem_device',
        ]
    }
    
    # Define macros from config (this is key for matching your CMake behavior)
    define_macros = []
    
    # Read kleos_config.json to set compile-time constants
    config_path = csrc_dir / 'kleos_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add config values as compile-time defines
        # This matches your CMake approach of exposing JSON params as static constants
        # Add config values as compile-time defines
        for key, value in config.items():
            # Add both with and without CONFIG_ prefix for compatibility
            define_macros.append((f'CONFIG_{key.upper()}', str(value)))
            define_macros.append((key.upper(), str(value)))

        # Auto-detect GPU architecture
        try:
            import subprocess
            # Get GPU compute capability
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                capture_output=True, text=True, check=True
            )
            compute_caps = [line.strip() for line in result.stdout.strip().split('\n')]
            
            # Map compute capability to architecture and SM count
            arch_map = {
                '7.0': (700, 80),   # V100
                '8.0': (800, 108),  # A100
                '9.0': (900, 132),  # H100
            }
            
            # Use the first GPU's compute capability
            compute_cap = compute_caps[0]
            if compute_cap in arch_map:
                kleos_arch, num_sms = arch_map[compute_cap]
                print(f"Detected GPU: Compute Capability {compute_cap} -> KLEOS_ARCH={kleos_arch}, NUM_SMS={num_sms}")
            else:
                # Default to A100 if unknown
                kleos_arch, num_sms = 800, 108
                print(f"Warning: Unknown compute capability {compute_cap}, defaulting to A100 (KLEOS_ARCH=800)")
            
        except Exception as e:
            # Default to A100 if detection fails
            kleos_arch, num_sms = 800, 108
            print(f"Warning: Could not detect GPU ({e}), defaulting to A100 (KLEOS_ARCH=800)")

        define_macros.extend([
            ('KLEOS_ARCH', str(kleos_arch)),
            ('NUM_SMS', str(num_sms)),
        ])

        # Add name mappings for code compatibility
        # Map JSON config names to C++ macro names expected by the code
        name_mappings = {
            'expert_top_k': 'E_TOP_K',
            'torch_dtype': 'DTYPE',
            'sequence_len': 'SEQ_LEN',
            'intermediate_size': 'I_SIZE',
            'moe_frequency': 'MOE_FREQ',
            'capacity_factor': 'CAP_FACTOR',
        }

        # Add the mapped names from config
        # FIXME(byungsoo): remove compile-time fixed macro
        for json_key, macro_key in name_mappings.items():
            if json_key in config:
                define_macros.append((macro_key, str(config[json_key])))
    
    ext_modules = [
        CUDAExtension(
            name='flashmoe._C',  # Standard naming: package._C
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            extra_link_args=[
                '-l:libnvshmem_device.a',  # Static device library
                f'-Wl,-rpath,{NVSHMEM_HOME}/lib',
                f'-Wl,-rpath,{CUDA_HOME}/lib64',
            ],
            define_macros=define_macros,
        )
    ]
    
    return ext_modules


def read_readme():
    """Read README.md for long description"""
    readme_file = Path('README.md')
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ''


# Check dependencies
print("=" * 60)
print("FlashMoE Setup")
print("=" * 60)

nvshmem_available = check_nvshmem()

print(f"CUDA_HOME: {CUDA_HOME}")
print(f"NVSHMEM_HOME: {NVSHMEM_HOME}")
print(f"NVSHMEM available: {nvshmem_available}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
print("=" * 60)


setup(
    name='flashmoe',
    version='0.1.0',
    author='Osayamen Jonathan Aimuyo',
    author_email='',
    description='FlashMoE: Fast Distributed MoE in a Single Kernel',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/osayamenja/FlashMoE',
    
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={
        'flashmoe': [],  # Add any data files if needed
    },
    
    ext_modules=get_cuda_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    },
    
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
    ],
    
    python_requires='>=3.8',
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Distributed Computing',
    ],
    
    keywords='moe, mixture-of-experts, cuda, distributed, nvshmem',
    
    project_urls={
        'Bug Reports': 'https://github.com/osayamenja/FlashMoE/issues',
        'Source': 'https://github.com/osayamenja/FlashMoE',
        'Paper': 'https://arxiv.org/abs/2506.04667',
    },
)
