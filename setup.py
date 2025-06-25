#!/usr/bin/env python
"""Setup script for DiffuGene package."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="diffugene",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive pipeline for genetic diffusion modeling",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DiffuGene",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "diffugene=DiffuGene.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "DiffuGene": [
            "config/*.yaml",
        ],
    },
    zip_safe=False,
    keywords="genetics, diffusion, machine-learning, genomics, deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/DiffuGene/issues",
        "Source": "https://github.com/yourusername/DiffuGene",
        "Documentation": "https://diffugene.readthedocs.io/",
    },
)
