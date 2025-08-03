#!/bin/bash

# Post-create script for DiffFE-Physics-Lab development container
set -e

echo "🚀 Setting up DiffFE-Physics-Lab development environment..."

# Activate Firedrake environment
source /home/firedrake/firedrake/bin/activate

# Install additional Python packages for development
echo "📦 Installing development dependencies..."
pip install --upgrade pip
pip install black isort flake8 mypy pytest pytest-cov pre-commit
pip install jupyter jupyterlab matplotlib seaborn plotly
pip install scipy scikit-learn optax

# Install JAX with CPU support (GPU support requires specific setup)
echo "🔧 Installing JAX..."
pip install jax jaxlib

# Try to install PyTorch (fallback for backend)
echo "🔧 Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install HDF5 and netCDF for data I/O
echo "💾 Installing data I/O libraries..."
pip install h5py netcdf4 xarray

# Install performance monitoring tools
echo "📊 Installing monitoring tools..."
pip install psutil memory_profiler line_profiler

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p /workspace/{data,examples,benchmarks,docs/guides,tests/{unit,integration,convergence},scripts}
mkdir -p /workspace/.cache/{meshes,solutions,checkpoints}

# Install the package in development mode
echo "🛠️ Installing DiffFE-Physics-Lab in development mode..."
cd /workspace
pip install -e .

# Set up pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create initial configuration files
echo "⚙️ Creating configuration files..."

# Create pytest configuration
cat > /workspace/pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes
    --cov=src
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests as requiring GPU
    convergence: marks tests as convergence studies
    integration: marks tests as integration tests
EOF

# Create mypy configuration
cat > /workspace/mypy.ini << 'EOF'
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

# Per-module options
[mypy-firedrake.*]
ignore_missing_imports = True

[mypy-jax.*]
ignore_missing_imports = True

[mypy-torch.*]
ignore_missing_imports = True

[mypy-scipy.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True
EOF

# Set up Git configuration for development
echo "🔧 Configuring Git..."
git config --global --add safe.directory /workspace
git config core.fileMode false

# Create sample environment file
echo "📝 Creating sample environment configuration..."
cat > /workspace/.env.example << 'EOF'
# DiffFE-Physics-Lab Environment Configuration

# Paths
DIFFHE_DATA_PATH=/workspace/data
DIFFHE_CACHE_PATH=/workspace/.cache
DIFFHE_LOG_PATH=/workspace/logs

# Compute settings
DIFFHE_DEFAULT_BACKEND=jax
DIFFHE_GPU_ENABLED=false
DIFFHE_NUM_THREADS=4

# Firedrake settings
FIREDRAKE_PARAMETERS_PATH=/workspace/config/firedrake.yaml

# Performance settings
DIFFHE_JIT_ENABLED=true
DIFFHE_CACHE_ENABLED=true
DIFFHE_PROFILE_ENABLED=false

# Development settings
DIFFHE_DEBUG=true
DIFFHE_LOG_LEVEL=INFO
DIFFHE_VALIDATE_INPUTS=true

# External services (optional)
WANDB_API_KEY=your_wandb_key_here
DIFFHE_TELEMETRY_ENABLED=false
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Copy .env.example to .env and customize settings"
echo "  2. Run 'pytest tests/' to verify installation"
echo "  3. Check 'examples/' directory for usage examples"
echo "  4. Start coding! 🎉"
echo ""
echo "🔗 Useful commands:"
echo "  - pytest: Run tests"
echo "  - black src/ tests/: Format code"
echo "  - flake8 src/: Lint code"
echo "  - mypy src/: Type checking"
echo "  - jupyter lab: Start Jupyter Lab"
