from setuptools import setup, find_packages

setup(
    name="diffhe-physics-lab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch>=2.0", "numpy>=1.24"],
    python_requires=">=3.9",
    author="Daniel Schmidt",
    description="Differentiable Finite Elements with ML integration",
)
