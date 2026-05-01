"""Setup script for MUSE: Multimodal Unified Semantic-Enhanced Image Tokenizer."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="muse-tokenizer",
    version="1.0.0",
    author="MUSE Team",
    description="MUSE: A Unified Image Tokenizer with Topology-Aware Semantic Injection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "outputs*"]),
    python_requires=">=3.9",
    install_requires=[
        "accelerate>=1.0.0",
        "omegaconf>=2.1.0",
        "transformers>=4.37.0",
        "diffusers>=0.33.0",
        "timm>=1.0.0",
        "open-clip-torch>=2.31.0",
        "webdataset>=0.2.100",
        "braceexpand>=0.1.7",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "einops>=0.8.0",
        "tqdm>=4.60.0",
        "termcolor>=2.0.0",
        "torchinfo>=1.8.0",
    ],
    extras_require={
        "eval": [
            "torch-fidelity>=0.3.0",
            "scikit-image>=0.21.0",
        ],
        "logging": [
            "wandb>=0.15.0",
        ],
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
