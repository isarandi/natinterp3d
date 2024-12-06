import numpy as np
from distutils.core import setup, Extension

ext_modules = [
    Extension(
        "natinterp3d.natinterp3d_cython",
        sources=["natinterp3d/natinterp3d_cython.pyx", 'natinterp3d/natural.c',
                 'natinterp3d/delaunay.c', 'natinterp3d/utils.c', 'natinterp3d/kdtree.c'],
        include_dirs=[np.get_include(), 'natinterp3d'],
        extra_compile_args=["-std=c11", "-O3", "-fopenmp", "-DUSE_LIST_NODE_ALLOCATOR",
                            '-lpthread', '-DNPY_NO_DEPRECATED_API', '-DNPY_1_7_API_VERSION'],
        extra_link_args=["-fopenmp", '-lpthread']
    )
]
setup(ext_modules=ext_modules)