"""Package metadata for editable local installation."""

from setuptools import find_packages, setup

setup(
    name="doe-scientific-analyzer",
    version="0.1.0",
    description="Design of Experiments analysis for chemical and process optimization.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "openpyxl>=3.1.0",
    ],
)
