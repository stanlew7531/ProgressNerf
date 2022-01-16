import os
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

raycasting_base_dir = osp.join(ROOT_DIR, "ProgressNerf", "Raycasting", "csrc")
raycasting_include_dirs = [osp.join(raycasting_base_dir, "include")]

try:
    ext_modules = [
        # Raycasting extension
        CUDAExtension(name='ProgressNerf.Raycasting.csrc', sources = [
            osp.join(raycasting_base_dir, 'raycasting.cpp'),
            osp.join(raycasting_base_dir, 'weighted_resampling.cu'),
        ],
        include_dirs=raycasting_include_dirs,
        optional=False),
    ]
except:
    import warnings
    warnings.warn("Failed to build CUDA extension")
    ext_modules = []


setup(name='ProgressNerf', 
    version='1.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},)