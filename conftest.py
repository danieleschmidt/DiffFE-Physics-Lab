"""Pytest configuration and fixtures for DiffFE-Physics-Lab."""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Set environment variables for testing
    os.environ["DIFFHE_DEBUG"] = "true"
    os.environ["DIFFHE_VALIDATE_INPUTS"] = "true"
    os.environ["DIFFHE_CACHE_ENABLED"] = "false"
    
    # Configure JAX for deterministic testing
    if HAS_JAX:
        os.environ["JAX_ENABLE_X64"] = "true"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    # Add markers based on test file location
    for item in items:
        # Unit tests
        if "tests/unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Integration tests
        elif "tests/integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Convergence tests
        elif "tests/convergence" in str(item.fspath):
            item.add_marker(pytest.mark.convergence)
            item.add_marker(pytest.mark.slow)
        
        # Skip tests based on missing dependencies
        if "firedrake" in item.keywords and not HAS_FIREDRAKE:
            item.add_marker(pytest.mark.skip(reason="Firedrake not available"))
        
        if "jax" in item.keywords and not HAS_JAX:
            item.add_marker(pytest.mark.skip(reason="JAX not available"))
        
        if "torch" in item.keywords and not HAS_TORCH:
            item.add_marker(pytest.mark.skip(reason="PyTorch not available"))
        
        # Skip GPU tests if no GPU available
        if "gpu" in item.keywords:
            skip_gpu = os.getenv("DIFFHE_SKIP_GPU_TESTS", "auto")
            if skip_gpu == "true":
                item.add_marker(pytest.mark.skip(reason="GPU tests disabled"))
            elif skip_gpu == "auto":
                # Try to detect GPU availability
                gpu_available = False
                if HAS_JAX:
                    try:
                        gpu_available = len(jax.devices("gpu")) > 0
                    except:
                        pass
                if not gpu_available:
                    item.add_marker(pytest.mark.skip(reason="No GPU available"))


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Test data directory fixture."""
    return Path(__file__).parent / "tests" / "data"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp(prefix="diffhe_test_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mesh_1d():
    """Simple 1D mesh fixture."""
    if not HAS_FIREDRAKE:
        pytest.skip("Firedrake not available")
    
    return fd.UnitIntervalMesh(10)


@pytest.fixture
def mesh_2d():
    """Simple 2D mesh fixture."""
    if not HAS_FIREDRAKE:
        pytest.skip("Firedrake not available")
    
    return fd.UnitSquareMesh(8, 8)


@pytest.fixture
def mesh_3d():
    """Simple 3D mesh fixture."""
    if not HAS_FIREDRAKE:
        pytest.skip("Firedrake not available")
    
    return fd.UnitCubeMesh(4, 4, 4)


@pytest.fixture(params=["P1", "P2"])
def function_space_scalar(request, mesh_2d):
    """Scalar function space fixture."""
    if not HAS_FIREDRAKE:
        pytest.skip("Firedrake not available")
    
    element_type = request.param
    degree = int(element_type[1])
    return fd.FunctionSpace(mesh_2d, "CG", degree)


@pytest.fixture(params=["P1", "P2"])
def function_space_vector(request, mesh_2d):
    """Vector function space fixture."""
    if not HAS_FIREDRAKE:
        pytest.skip("Firedrake not available")
    
    element_type = request.param
    degree = int(element_type[1])
    return fd.VectorFunctionSpace(mesh_2d, "CG", degree)


@pytest.fixture(params=["jax", "torch"])
def backend(request):
    """Backend fixture for testing multiple AD backends."""
    backend_name = request.param
    
    if backend_name == "jax" and not HAS_JAX:
        pytest.skip("JAX not available")
    elif backend_name == "torch" and not HAS_TORCH:
        pytest.skip("PyTorch not available")
    
    from src.backends import get_backend
    return get_backend(backend_name)


@pytest.fixture
def jax_random_key():
    """JAX random key fixture."""
    if not HAS_JAX:
        pytest.skip("JAX not available")
    
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_parameters():
    """Sample parameters for testing."""
    return {
        "diffusion_coeff": 1.0,
        "source": lambda x: 1.0,
        "E": 200e9,  # Young's modulus (Pa)
        "nu": 0.3,   # Poisson's ratio
        "body_force": [0.0, -9810.0],  # Body force (N/m^3)
    }


@pytest.fixture
def manufactured_solutions():
    """Manufactured solutions for verification."""
    import numpy as np
    
    solutions = {
        "laplacian_2d": {
            "solution": lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]),
            "source": lambda x: 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        },
        "elasticity_2d": {
            "displacement": lambda x: np.array([
                np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),
                np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
            ]),
            "body_force": lambda x: np.array([1.0, 1.0])  # Simplified
        }
    }
    
    return solutions


@pytest.fixture
def tolerance_settings():
    """Tolerance settings for numerical tests."""
    return {
        "rtol": 1e-10,
        "atol": 1e-12,
        "convergence_rtol": 1e-8,
        "optimization_tol": 1e-6
    }


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Reset any global singletons or state
    yield
    # Cleanup after test


@pytest.fixture
def mock_database(temp_dir):
    """Mock database for testing."""
    from src.database import DatabaseManager
    
    db_path = temp_dir / "test.db"
    db_url = f"sqlite:///{db_path}"
    
    db = DatabaseManager(db_url)
    yield db
    db.disconnect()


# Performance testing fixtures
@pytest.fixture
def performance_thresholds():
    """Performance thresholds for benchmarking."""
    return {
        "solve_time_2d": 1.0,      # seconds
        "gradient_time_2d": 2.0,   # seconds  
        "memory_usage_mb": 500,    # MB
        "convergence_rate": 1.8,   # minimum order
    }


@pytest.fixture
def mesh_sizes():
    """Mesh sizes for convergence studies."""
    return [8, 16, 32, 64]


# Numerical verification fixtures
@pytest.fixture
def reference_solutions():
    """Reference solutions for validation."""
    return {
        "poisson_1d": {
            "exact": lambda x: x * (1 - x) / 2,
            "l2_error_threshold": 1e-10
        },
        "elasticity_cantilever": {
            "tip_displacement": 0.0312,  # Reference value
            "relative_error_threshold": 0.05
        }
    }
