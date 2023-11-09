from setuptools import setup, Extension
from torch.utils import cpp_extension

ittapi_inc = '"C:\\Users\\tingqian\\Syncplicity\\LTQ\\Code\\ittapi-3.24.3\\include"'
ittapi_libpath = '"C:\\Users\\tingqian\\Syncplicity\\LTQ\\Code\\ittapi-3.24.3\\build_win\\64\\bin\\Release"'

iomp_libpath='C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2023.2.1\\windows\\compiler\\lib\\intel64_win'
setup(name='llext1',
      ext_modules=[
        cpp_extension.CppExtension(
                'llext1',
                sources=['llama_ext.cpp'],
                extra_compile_args=["/openmp", "/arch:AVX2", "/DEBUG", f'/I{ittapi_inc}'],
                #extra_compile_args=["-fopenmp", "-mavx2", "-mfma", "-mf16c", "-mno-avx256-split-unaligned-load", "-mno-avx256-split-unaligned-store"],
                extra_link_args=["libittnotify.lib", "libiomp5md.lib",
                                f'/LIBPATH:{ittapi_libpath}',
                                f'/LIBPATH:{iomp_libpath}', '/nodefaultlib:vcomp']
                )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

