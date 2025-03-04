from setuptools import setup
from cmake_build_extension import BuildExtension, CMakeExtension

setup(
    name="franka_control",
    version="0.1.0",
    ext_modules=[
        CMakeExtension(
            name="hardware.franka_control._franka_control",
            source_dir=".",
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
