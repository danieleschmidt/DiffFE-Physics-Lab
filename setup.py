"""Setup script for DiffFE-Physics-Lab."""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read version from src/__init__.py
def get_version():
    """Get version from package __init__.py."""
    version_file = Path(__file__).parent / "src" / "__init__.py"
    with open(version_file) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"')
    raise RuntimeError("Could not find version")

# Read long description from README
def get_long_description():
    """Get long description from README.md."""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, encoding="utf-8") as f:
            return f.read()
    return ""

# Minimum Python version check
if sys.version_info < (3, 10):
    raise RuntimeError("DiffFE-Physics-Lab requires Python 3.10 or later")

# Core requirements
requirements = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    # Note: Firedrake must be installed separately
    # "firedrake>=0.13.0",  # Commented out - special installation required
    "jax>=0.4.25",
    "jaxlib>=0.4.25",
    "psutil>=5.9.0",
]

# Optional dependencies
optional_requirements = {
    "pytorch": [
        "torch>=2.4.0",
        "torchvision>=0.19.0",
    ],
    "optimization": [
        "optax>=0.1.0",
        "scikit-learn>=1.0.0",
    ],
    "data": [
        "h5py>=3.7.0",
        "netcdf4>=1.6.0",
        "xarray>=2022.3.0",
        "pandas>=1.5.0",
    ],
    "visualization": [
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
        "ipywidgets>=8.0.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-xdist>=3.0.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "flake8>=5.0.0",
        "mypy>=0.991",
        "pre-commit>=2.20.0",
        "bandit>=1.7.0",
        "bump2version>=1.0.0",
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "sphinxcontrib-bibtex>=2.5.0",
        "nbsphinx>=0.8.0",
        "jupyter>=1.0.0",
        "jupyterlab>=3.0.0",
    ],
    "performance": [
        "memory-profiler>=0.60.0",
        "line-profiler>=4.0.0",
        "py-spy>=0.3.0",
    ],
    "database": [
        "psycopg2-binary>=2.9.0",
        "sqlalchemy>=1.4.0",
        "alembic>=1.8.0",
    ],
}

# Convenience groups
optional_requirements["all"] = [
    dep for deps in optional_requirements.values() for dep in deps
]
optional_requirements["minimal"] = (
    optional_requirements["optimization"] + 
    optional_requirements["data"] +
    optional_requirements["visualization"]
)

# Platform-specific requirements
if sys.platform == "win32":
    # Windows-specific requirements
    pass
elif sys.platform == "darwin":
    # macOS-specific requirements
    pass
else:
    # Linux-specific requirements
    pass

# Entry points for CLI tools
entry_points = {
    "console_scripts": [
        "diffhe-run=src.cli.run:main",
        "diffhe-benchmark=src.cli.benchmark:main",
        "diffhe-validate=src.cli.validate:main",
        "diffhe-convert=src.cli.convert:main",
    ],
}

setup(
    name="diffhe-physics-lab",
    version=get_version(),
    author="DiffFE-Physics-Lab Team",
    author_email="team@diffhe-physics.org",
    description="Differentiable finite element framework for physics-informed machine learning",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/DiffFE-Physics-Lab",
    project_urls={
        "Bug Reports": "https://github.com/danieleschmidt/DiffFE-Physics-Lab/issues",
        "Source": "https://github.com/danieleschmidt/DiffFE-Physics-Lab",
        "Documentation": "https://diffhe-physics-lab.readthedocs.io/",
        "Changelog": "https://github.com/danieleschmidt/DiffFE-Physics-Lab/blob/main/CHANGELOG.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require=optional_requirements,
    entry_points=entry_points,
    include_package_data=True,
    package_data={
        "src": [
            "config/*.yaml",
            "config/*.json",
            "templates/*.py",
            "templates/*.md",
        ],
    },
    zip_safe=False,
    keywords=[
        "finite-element", "automatic-differentiation", "physics-informed-ml",
        "computational-physics", "optimization", "jax", "pytorch", "firedrake",
        "pde-solver", "inverse-problems", "machine-learning", "scientific-computing"
    ],
    platforms=["any"],
    license="BSD-3-Clause",
    # Custom installation message
    cmdclass={},
)

# Post-installation message
print("""
🎉 DiffFE-Physics-Lab installation complete!

📋 Next steps:
  1. Install Firedrake (required):
     curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
     python3 firedrake-install
  
  2. Activate Firedrake environment:
     source firedrake/bin/activate
  
  3. Test installation:
     python -c "import src; print('DiffFE-Physics-Lab ready!')"
  
  4. Explore examples:
     diffhe-run --help

📚 Documentation: https://diffhe-physics-lab.readthedocs.io/
🐛 Issues: https://github.com/danieleschmidt/DiffFE-Physics-Lab/issues
💬 Discussions: https://github.com/danieleschmidt/DiffFE-Physics-Lab/discussions

Happy differentiable computing! 🚀
""")
