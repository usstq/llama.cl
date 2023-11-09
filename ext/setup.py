from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='llext1',
      ext_modules=[
        cpp_extension.CppExtension(
                'llext1',
                sources=['llama_ext.cpp'],
                #extra_compile_args=["/openmp", "/arch:AVX2"],
                extra_compile_args=["-fopenmp", "-mavx2", "-mfma", "-mf16c", "-mno-avx256-split-unaligned-load", "-mno-avx256-split-unaligned-store"],
                extra_link_args=[]
                )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

