from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

setup(
    ext_modules=cythonize(
        Extension(
            "diffOrderbook",  # or "market_simulator.models.diffOrderbook" if importing from a package
            ["diffOrderbook.pyx"],
            language="c++"  # 🔥 tells MSVC to treat this as C++
        ),
        language_level=3,
    )
)