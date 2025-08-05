"""Pytest configuration and fixtures for DiffFE-Physics-Lab tests."""

import pytest
import numpy as np
import logging
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for optional dependencies
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

try:
    from flask import Flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "firedrake: marks tests as requiring Firedrake")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and requirements."""
    for item in items:
        # Add slow marker to tests that likely take longer
        if any(keyword in item.name.lower() for keyword in ['convergence', 'optimization', 'benchmark']):
            item.add_marker(pytest.mark.slow)
        
        # Add GPU marker to GPU tests
        if 'gpu' in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Add Firedrake marker to Firedrake-dependent tests
        if any(keyword in item.name.lower() for keyword in ['fem', 'mesh', 'assembly']):
            item.add_marker(pytest.mark.firedrake)
        
        # Skip Firedrake tests if not available
        if item.get_closest_marker("firedrake") and not HAS_FIREDRAKE:
            item.add_marker(pytest.mark.skip(reason="Firedrake not available"))


# Basic fixtures
@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        'mesh_size': 32,
        'dimension': 2,
        'dofs': 1000,
        'test_points': np.array([
            [0.1, 0.1],
            [0.5, 0.5],
            [0.9, 0.9]
        ])
    }


@pytest.fixture
def mock_firedrake():
    """Mock Firedrake for tests when not available."""
    if HAS_FIREDRAKE:
        return None
    
    # Create mock Firedrake module
    mock_fd = MagicMock()
    
    # Mock common classes
    mock_fd.UnitSquareMesh = Mock(return_value=Mock(
        num_cells=Mock(return_value=100),
        num_vertices=Mock(return_value=121),
        geometric_dimension=Mock(return_value=2)
    ))
    
    mock_fd.FunctionSpace = Mock(return_value=Mock(
        dim=Mock(return_value=121),
        mesh=Mock(return_value=mock_fd.UnitSquareMesh())
    ))
    
    mock_fd.Function = Mock()
    mock_fd.TestFunction = Mock()
    mock_fd.TrialFunction = Mock()
    mock_fd.Constant = Mock()
    
    # Mock operators
    mock_fd.grad = Mock()
    mock_fd.inner = Mock()
    mock_fd.dx = Mock()
    mock_fd.ds = Mock()
    mock_fd.assemble = Mock(return_value=1.0)
    mock_fd.solve = Mock()
    
    return mock_fd


@pytest.fixture
def mock_jax():
    """Mock JAX for tests when not available."""
    if HAS_JAX:
        return None
    
    mock_jax = MagicMock()
    mock_jax.grad = Mock(return_value=lambda x: np.array([2.0]))
    mock_jax.jit = Mock(side_effect=lambda f: f)
    mock_jax.vmap = Mock(side_effect=lambda f: f)
    mock_jax.random.PRNGKey = Mock(return_value=np.array([0, 0]))
    mock_jax.random.normal = Mock(return_value=np.random.normal(0, 1, (10,)))
    
    mock_jnp = MagicMock()
    mock_jnp.array = np.array
    mock_jnp.zeros = np.zeros
    mock_jnp.ones = np.ones
    mock_jnp.sqrt = np.sqrt
    mock_jnp.sin = np.sin
    mock_jnp.cos = np.cos
    
    mock_jax.numpy = mock_jnp
    
    return mock_jax


# Problem fixtures
@pytest.fixture
def simple_problem_config():
    """Configuration for a simple test problem."""
    return {
        'mesh': {
            'type': 'unit_square',
            'n_elements': 16
        },
        'function_space': {
            'family': 'CG',
            'degree': 1
        },
        'equation': {
            'type': 'laplacian',
            'parameters': {
                'diffusion_coeff': 1.0
            }
        },
        'boundary_conditions': {
            'dirichlet': {
                'boundary': 'on_boundary',
                'value': 0.0
            }
        },
        'source': lambda x: 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    }


if HAS_FIREDRAKE:
    @pytest.fixture
    def unit_square_mesh():
        """Create unit square mesh."""
        return fd.UnitSquareMesh(8, 8)
    
    @pytest.fixture
    def mesh_2d():
        """Create 2D mesh for testing."""
        return fd.UnitSquareMesh(4, 4)
    
    @pytest.fixture
    def function_space_scalar(mesh_2d):
        """Create scalar function space."""
        return fd.FunctionSpace(mesh_2d, "CG", 1)
    
    @pytest.fixture
    def function_space_vector(mesh_2d):
        """Create vector function space."""
        return fd.VectorFunctionSpace(mesh_2d, "CG", 1)
    
    @pytest.fixture
    def function_space_cg1(unit_square_mesh):
        """Create CG1 function space."""
        return fd.FunctionSpace(unit_square_mesh, "CG", 1)
    
    @pytest.fixture
    def function_space_cg2(unit_square_mesh):
        """Create CG2 function space."""
        return fd.FunctionSpace(unit_square_mesh, "CG", 2)
else:
    # Mock fixtures when Firedrake is not available
    @pytest.fixture
    def unit_square_mesh():
        """Mock unit square mesh."""
        mock_mesh = Mock()
        mock_mesh.num_cells.return_value = 64
        mock_mesh.geometric_dimension.return_value = 2
        return mock_mesh
    
    @pytest.fixture
    def mesh_2d():
        """Mock 2D mesh."""
        mock_mesh = Mock()
        mock_mesh.num_cells.return_value = 16
        mock_mesh.geometric_dimension.return_value = 2
        return mock_mesh
    
    @pytest.fixture
    def function_space_scalar(mesh_2d):
        """Mock scalar function space."""
        mock_fs = Mock()
        mock_fs.dim.return_value = 25
        mock_fs.mesh = mesh_2d
        return mock_fs
    
    @pytest.fixture
    def function_space_vector(mesh_2d):
        """Mock vector function space."""
        mock_fs = Mock()
        mock_fs.dim.return_value = 50
        mock_fs.mesh = mesh_2d
        return mock_fs
    
    @pytest.fixture
    def function_space_cg1(unit_square_mesh):
        """Mock CG1 function space."""
        mock_fs = Mock()
        mock_fs.dim.return_value = 81
        mock_fs.mesh = unit_square_mesh
        return mock_fs
    
    @pytest.fixture
    def function_space_cg2(unit_square_mesh):
        """Mock CG2 function space."""
        mock_fs = Mock()
        mock_fs.dim.return_value = 289
        mock_fs.mesh = unit_square_mesh
        return mock_fs


@pytest.fixture
def exact_solution_2d():
    """Exact solution for 2D problems."""
    def solution(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    return solution


@pytest.fixture
def source_term_2d():
    """Source term corresponding to exact solution."""
    def source(x):
        return 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    return source


# Backend fixtures
@pytest.fixture(params=['jax'] if HAS_JAX else [])
def backend_name(request):
    """Parametrize tests over available backends."""
    return request.param


@pytest.fixture
def optimization_test_data():
    """Test data for optimization problems."""
    return {
        'true_parameters': {'diffusion_coeff': 2.5, 'amplitude': 1.0},
        'initial_guess': {'diffusion_coeff': 1.0, 'amplitude': 0.5},
        'bounds': {'diffusion_coeff': (0.1, 10.0), 'amplitude': (0.1, 5.0)},
        'noise_level': 0.01,
        'tolerance': 1e-6
    }


# Flask fixtures
if HAS_FLASK:
    @pytest.fixture
    def flask_app():
        """Create Flask app for testing."""
        from src.api.app import create_app
        
        app = create_app({
            'TESTING': True,
            'DEBUG': True,
            'SECRET_KEY': 'test-secret-key'
        })
        
        return app
    
    @pytest.fixture
    def client(flask_app):
        """Create test client."""
        return flask_app.test_client()
    
    @pytest.fixture
    def app_context(flask_app):
        """Create app context."""
        with flask_app.app_context():
            yield flask_app


# Performance fixtures
@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        'assembly_time': 0.1,      # Max assembly time in seconds
        'solve_time': 1.0,         # Max solve time in seconds
        'memory_usage': 100.0,     # Max memory usage in MB
        'convergence_rate': 1.8    # Min convergence rate
    }


@pytest.fixture
def performance_timer():
    """Timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.times = []
        
        def __enter__(self):
            self.start = time.time()
            return self
        
        def __exit__(self, *args):
            self.end = time.time()
            self.duration = self.end - self.start
            self.times.append(self.duration)
    
    return Timer


@pytest.fixture
def memory_profiler():
    """Memory profiler for memory tests."""
    try:
        import psutil
        import os
        
        class MemoryProfiler:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.initial_memory = None
                self.peak_memory = None
            
            def start(self):
                self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                return self
            
            def stop(self):
                current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                self.peak_memory = current_memory
                return self.peak_memory - self.initial_memory
        
        return MemoryProfiler()
    
    except ImportError:
        # Return mock profiler if psutil not available
        class MockMemoryProfiler:
            def start(self):
                return self
            def stop(self):
                return 0.0
        
        return MockMemoryProfiler()


# Manufactured solution fixtures
@pytest.fixture
def mms_trigonometric_2d():
    """2D trigonometric manufactured solution."""
    from src.utils.manufactured_solutions import generate_manufactured_solution, SolutionType
    
    return generate_manufactured_solution(
        solution_type=SolutionType.TRIGONOMETRIC,
        dimension=2,
        parameters={'frequency': 1.0, 'amplitude': 1.0}
    )


@pytest.fixture
def mms_polynomial_2d():
    """2D polynomial manufactured solution."""
    from src.utils.manufactured_solutions import generate_manufactured_solution, SolutionType
    
    return generate_manufactured_solution(
        solution_type=SolutionType.POLYNOMIAL,
        dimension=2,
        parameters={'degree': 2}
    )


# Error computation fixtures
@pytest.fixture
def convergence_test_data():
    """Data for convergence tests."""
    mesh_sizes = [0.2, 0.1, 0.05, 0.025]
    errors_l2 = [1e-2, 2.5e-3, 6.25e-4, 1.56e-4]  # Rate ≈ 2
    errors_h1 = [1e-1, 5e-2, 2.5e-2, 1.25e-2]     # Rate ≈ 1
    
    return {
        'mesh_sizes': mesh_sizes,
        'errors_l2': errors_l2,
        'errors_h1': errors_h1,
        'expected_rate_l2': 2.0,
        'expected_rate_h1': 1.0
    }


# Test utilities
@pytest.fixture
def assert_almost_equal():
    """Enhanced almost equal assertion."""
    def _assert_almost_equal(actual, expected, rtol=1e-7, atol=1e-12, msg=""):
        if np.isscalar(actual) and np.isscalar(expected):
            assert abs(actual - expected) <= atol + rtol * abs(expected), \
                f"{msg}: {actual} != {expected} (tol={atol + rtol * abs(expected)})"
        else:
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=msg)
    
    return _assert_almost_equal


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files after tests."""
    import tempfile
    import shutil
    import os
    
    temp_dirs = []
    temp_files = []
    
    yield
    
    # Clean up temporary directories
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Clean up temporary files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


# Skip conditions
def skip_if_no_firedrake():
    """Skip test if Firedrake not available."""
    return pytest.mark.skipif(not HAS_FIREDRAKE, reason="Firedrake not available")


def skip_if_no_jax():
    """Skip test if JAX not available."""
    return pytest.mark.skipif(not HAS_JAX, reason="JAX not available")


def skip_if_no_torch():
    """Skip test if PyTorch not available."""
    return pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")


def skip_if_no_flask():
    """Skip test if Flask not available."""
    return pytest.mark.skipif(not HAS_FLASK, reason="Flask not available")


# Custom assertions
def assert_convergence_rate(mesh_sizes, errors, expected_rate, tolerance=0.2):
    """Assert that convergence rate is as expected."""
    from src.utils.error_computation import compute_convergence_rate
    
    result = compute_convergence_rate(mesh_sizes, errors)
    actual_rate = result['rate']
    
    assert abs(actual_rate - expected_rate) <= tolerance, \
        f"Convergence rate {actual_rate:.3f} not close to expected {expected_rate:.3f}"


def assert_solution_accuracy(computed_solution, exact_solution, expected_error, tolerance=0.1):
    """Assert that solution accuracy is as expected."""
    if HAS_FIREDRAKE:
        from src.utils.error_computation import compute_error
        
        actual_error = compute_error(computed_solution, exact_solution, 'L2')
        
        assert actual_error <= expected_error * (1 + tolerance), \
            f"Error {actual_error:.6e} exceeds expected {expected_error:.6e}"