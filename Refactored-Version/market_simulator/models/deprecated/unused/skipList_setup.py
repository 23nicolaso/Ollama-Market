from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    def build_extensions(self):
        # Detect MSVC vs GCC/Clang
        compiler_type = self.compiler.compiler_type
        
        for ext in self.extensions:
            if compiler_type == 'msvc':
                # MSVC compiler flags
                ext.extra_compile_args = ['/std:c++11']
            else:
                # GCC/Clang flags
                ext.extra_compile_args = ['-std=c++11']
        
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        "pyskiplist",
        ["skiplist_wrapper.pyx", "skipList.cpp"],
        language="c++",
        include_dirs=["."],  # Include numpy if needed
    )
]

setup(
    # your other setup parameters...
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
)