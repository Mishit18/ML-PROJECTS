"""
Package configuration for Mini-GPT implementation.

This module defines the installation requirements and metadata for the
production-grade GPT implementation.
"""

from setuptools import setup, find_packages

setup(
    name="mini_gpt",
    version="1.0.0",
    description="Production-grade GPT implementation from scratch",
    author="Research Engineer",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.13.0",
        "pytest>=7.3.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
