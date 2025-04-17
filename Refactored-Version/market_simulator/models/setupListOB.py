from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

setup(
    ext_modules=cythonize(
        Extension(
            "sortedListOB",  # or "market_simulator.models.diffOrderbook" if importing from a package
            ["sortedListOrderBook.pyx"],
            language="c++"  # 🔥 tells MSVC to treat this as C++
        ),
        language_level=3,
    )
)