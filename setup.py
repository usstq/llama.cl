import sys
from setuptools import setup, Extension
from torch.utils import cpp_extension

extra_link_args = []
extra_compile_args = []

if sys.platform == 'win32':
    use_intel_omp = True
    use_itt_api = True

    extra_compile_args += ["/openmp", "/arch:AVX2"]

    # replace following path to corresponding path on your local system
    #  https://github.com/intel/ittapi/releases
    ittapi_inc = 'C:\\Users\\tingqian\\Syncplicity\\LTQ\\Code\\ittapi-3.24.3\\include'
    ittapi_libpath = 'C:\\Users\\tingqian\\Syncplicity\\LTQ\\Code\\ittapi-3.24.3\\build_win\\64\\bin\\Release'
    # oneAPI
    iomp_libpath='C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2023.2.1\\windows\\compiler\\lib\\intel64_win'

    if use_intel_omp:
        extra_link_args += ["libiomp5md.lib", f'/LIBPATH:{iomp_libpath}', '/nodefaultlib:vcomp']
    if use_itt_api:
        extra_compile_args += [f'/I{ittapi_inc}', '/D', "USE_ITT_API"]
        extra_link_args += ["libittnotify.lib", f'/LIBPATH:{ittapi_libpath}']

elif sys.platform == "linux":
    use_intel_omp = False
    use_itt_api = False
    extra_compile_args += ["-fopenmp",
                           "-mavx2",
                           "-mfma",
                           "-mf16c",
                           "-mno-avx256-split-unaligned-load",
                           "-mno-avx256-split-unaligned-store"]
else:
    raise Exception(f"unsupported sys.platform : {sys.platform}")

print("\033[0;33m")
print(f"extra_compile_args = {extra_compile_args}")
print("")
print(f"extra_link_args = {extra_link_args}")
print("\033[00m")


c_ext = cpp_extension.CppExtension(
                name = 'chatllama.c_ext',
                sources=['ext/llama_ext.cpp'],
                extra_compile_args = extra_compile_args,
                extra_link_args = extra_link_args
                )

print(c_ext)
print(f"\tinclude_dirs={c_ext.include_dirs}")
print(f"\tlibraries={c_ext.libraries}")
print(f"\tlibraries={c_ext.library_dirs}")
print(f"\tlibraries={c_ext.runtime_library_dirs}")
print(f"\tlibraries={c_ext.sources}")
print(f"\tlibraries={c_ext.depends}")
print(f"\tlibraries={c_ext.extra_compile_args}")
print(f"\tlibraries={c_ext.extra_link_args}")
print(f"\tlibraries={c_ext.extra_objects}")


setup(name='chatllama',
      version='0.0.1',
      packages=['chatllama'],
      ext_modules=[c_ext],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      install_requires=[
        'torch>=2.0.1',
        'transformers>=4.31.0'
      ])

