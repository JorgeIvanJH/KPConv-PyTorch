from setuptools import setup, Extension
import numpy

# List of C++ source files
SOURCES = [
    "../cpp_utils/cloud/cloud.cpp",
    "neighbors/neighbors.cpp",
    "wrapper.cpp"
]

# Define the extension module
module = Extension(
    name="radius_neighbors",
    sources=SOURCES,
    extra_compile_args=['/std:c++14'],  # Windows/MSVC compatible
    include_dirs=[numpy.get_include()]
)

# Setup configuration
setup(
    name="radius_neighbors",
    ext_modules=[module]
)
