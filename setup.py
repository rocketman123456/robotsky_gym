from setuptools import find_packages
from distutils.core import setup

setup(
    name="robotsky_gym",
    version="1.0.0",
    author="Yf Zhang",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="rocketman123456@buaa.edu.cn",
    description="Training Envs for Legged Robots",
    install_requires=[
        # "isaacgym",
        # "rsl-rl",
        # "matplotlib",
        "torch",
        "numpy",
        "gymnasium",
    ],
)
