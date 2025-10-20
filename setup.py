#!/usr/bin/env python3
"""
Setup script for Enhanced AR(1) Trend Analysis tool
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            # Remove version constraints for basic setup
            package = line.split('>=')[0].split('==')[0].split('<')[0]
            requirements.append(package)

setup(
    name="enhanced-ar1-trend-analysis",
    version="1.0.0",
    author="Climate Analysis Team",
    author_email="your-email@example.com",
    description="A robust implementation of Enhanced AR(1) methodology for detecting trends in time series raster data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/enhanced-ar1-trend-analysis",
    project_urls={
        "Bug Reports": "https://github.com/your-username/enhanced-ar1-trend-analysis/issues",
        "Source": "https://github.com/your-username/enhanced-ar1-trend-analysis",
        "Documentation": "https://github.com/your-username/enhanced-ar1-trend-analysis/blob/main/README.md",
    },
    py_modules=[
        "enhanced_ar1_trend_analysis",
        "config",
        "examples",
        "test_enhanced_ar1"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "performance": [
            "dask>=2021.6.0",
            "bottleneck>=1.3.0",
            "netcdf4>=1.5.0",
            "h5netcdf>=0.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "enhanced-ar1=enhanced_ar1_trend_analysis:main",
        ],
    },
    keywords=[
        "climate", "trends", "time-series", "autocorrelation", 
        "statistics", "netcdf", "geospatial", "precipitation", 
        "temperature", "AR1", "trend-analysis"
    ],
    zip_safe=False,
    include_package_data=True,
)