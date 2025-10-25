"""
Setup configuration for HR-ThermalPV dataset tools
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="hr-thermalpv",
    version="1.0.0",
    author="Mohammed Yaqoob, Mohammed Yusuf Ansari, Dhanup Somasekharan Pillai, Eduardo Feo Flushing",
    author_email="yansari@andrew.cmu.edu",
    description="High-Resolution Thermal Imaging Dataset for Photovoltaic Homography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YaqoobAnsari/HR-ThermalPV",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "Pillow>=9.0.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hr-thermalpv-preprocess=preprocess:main",
            "hr-thermalpv-generate=generate_homography_pairs:main",
            "hr-thermalpv-split=split_dataset:main",
        ],
    },
)
