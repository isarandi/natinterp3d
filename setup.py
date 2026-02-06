import os
import sys
import numpy as np
from setuptools import setup, Extension

native = os.environ.get("NATINTERP3D_NATIVE", "0") == "1"

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []
    libraries = []
elif sys.platform == "darwin":
    extra_compile_args = ["-std=c11", "-O3", "-Wall", "-Wextra", "-Wno-unused-parameter",
                          "-fno-math-errno", "-flto",
                          "-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-lomp", "-flto"]
    libraries = ["m"]
else:
    extra_compile_args = ["-std=c11", "-O3", "-Wall", "-Wextra", "-Wno-unused-parameter",
                          "-fno-math-errno", "-flto",
                          "-fopenmp"]
    extra_link_args = ["-fopenmp", "-flto"]
    libraries = ["m", "pthread"]

if native and sys.platform != "win32":
    extra_compile_args.append("-march=native")

ext_modules = [
    Extension(
        'natinterp3d.natinterp3d_cython',
        sources=[
            'src/natinterp3d/natinterp3d_cython.pyx', 'src/natinterp3d/unity.c'],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        define_macros=[
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
            ('USE_LIST_NODE_ALLOCATOR', None)
        ],
        libraries=libraries,
        extra_link_args=extra_link_args,
    )
]
setup(ext_modules=ext_modules)