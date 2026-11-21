from setuptools import setup, find_packages

setup(
    name="vnts",
    version="0.5.0",
    description="Vector Nullspace Transformation System",
    author="VNTS Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0"
    ],
    python_requires=">=3.9",
)
