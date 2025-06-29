from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="machine-learning-mamba",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Mambaアーキテクチャを使用した機械学習システム",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/machine_learning_mamba",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "torch[cuda]",
        ],
    },
    entry_points={
        "console_scripts": [
            "mamba-ml=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)