from setuptools import setup, find_packages

setup(
    name="market_simulator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.5.0',
        'numpy>=1.21.0',
        'langchain_ollama>=0.1.0',
    ],
) 