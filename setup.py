from setuptools import setup, find_packages

setup(
    name="LRDFinder",
    version="0.1.0",
    description="A package for finding LRDs using neural networks",
    author="Zechang Sun",
    author_email="zechangsun33@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "astropy>=4.0.0",
        "numpy>=1.20.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires='>=3.10',
)