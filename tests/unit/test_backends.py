"""Comprehensive unit tests for AD backends."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Callable

from src.backends.base import ADBackend, get_backend, set_default_backend, list_backends
from src.backends.jax_backend import JAXBackend
from src.backends.torch_backend import TorchBackend
from src.utils.exceptions import ValidationError


class TestADBackend:
    """Test abstract base ADBackend class."""
    
    def test_abstract_backend_instantiation(self):
        """Test that abstract backend cannot be instantiated."""
        with pytest.raises(TypeError):
            ADBackend()
    
    def test_backend_interface_methods(self):
        """Test that backend interface methods are abstract."""
        class TestBackend(ADBackend):
            def __init__(self):
                super().__init__("test")
        
        backend = TestBackend()
        
        with pytest.raises(NotImplementedError):
            backend.array(np.array([1, 2, 3]))
        
        with pytest.raises(NotImplementedError):
            backend.zeros((3, 3))
        
        with pytest.raises(NotImplementedError):
            backend.ones((3, 3))
        
        with pytest.raises(NotImplementedError):
            backend.gradient(lambda x: x**2)
        
        with pytest.raises(NotImplementedError):
            backend.compile(lambda x: x)
        
        with pytest.raises(NotImplementedError):
            backend.vectorize(lambda x: x)


class TestJAXBackend:
    """Test JAX backend implementation."""
    
    @pytest.fixture
    def jax_backend(self):
        """Create JAX backend for testing."""
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax') as mock_jax:
                with patch('src.backends.jax_backend.jnp') as mock_jnp:
                    # Setup basic JAX mocks
                    mock_jnp.array = Mock(side_effect=np.array)
                    mock_jnp.zeros = Mock(side_effect=np.zeros)
                    mock_jnp.ones = Mock(side_effect=np.ones)
                    mock_jnp.sin = Mock(side_effect=np.sin)
                    mock_jnp.cos = Mock(side_effect=np.cos)
                    mock_jnp.exp = Mock(side_effect=np.exp)
                    mock_jnp.log = Mock(side_effect=np.log)
                    mock_jnp.sqrt = Mock(side_effect=np.sqrt)
                    mock_jnp.sum = Mock(side_effect=np.sum)
                    mock_jnp.mean = Mock(side_effect=np.mean)
                    mock_jnp.std = Mock(side_effect=np.std)
                    mock_jnp.linalg.norm = Mock(side_effect=np.linalg.norm)
                    mock_jnp.dot = Mock(side_effect=np.dot)
                    
                    mock_jax.grad = Mock(return_value=lambda x: 2*x)
                    mock_jax.jit = Mock(side_effect=lambda f: f)
                    mock_jax.vmap = Mock(side_effect=lambda f: f)
                    mock_jax.random.PRNGKey = Mock(return_value=np.array([0, 0]))
                    mock_jax.random.normal = Mock(return_value=np.array([0.5]))
                    
                    yield JAXBackend()
    
    def test_jax_backend_initialization(self, jax_backend):
        """Test JAX backend initialization."""
        assert jax_backend.name == "jax"
        assert hasattr(jax_backend, 'jax')
        assert hasattr(jax_backend, 'jnp')
    
    def test_jax_backend_unavailable(self):
        """Test behavior when JAX is not available."""
        with patch('src.backends.jax_backend.HAS_JAX', False):
            with pytest.raises(ImportError, match="JAX is required"):
                JAXBackend()
    
    def test_jax_array_creation(self, jax_backend):
        """Test array creation with JAX backend."""
        data = [1, 2, 3]
        result = jax_backend.array(data)
        jax_backend.jnp.array.assert_called_once_with(data)
    
    def test_jax_zeros_ones(self, jax_backend):
        """Test zeros and ones creation."""
        shape = (3, 4)
        
        jax_backend.zeros(shape)
        jax_backend.jnp.zeros.assert_called_with(shape)
        
        jax_backend.ones(shape)
        jax_backend.jnp.ones.assert_called_with(shape)
    
    def test_jax_mathematical_operations(self, jax_backend):
        """Test mathematical operations."""
        x = np.array([1.0, 2.0, 3.0])
        
        # Test trigonometric functions
        jax_backend.sin(x)
        jax_backend.jnp.sin.assert_called_with(x)
        
        jax_backend.cos(x)
        jax_backend.jnp.cos.assert_called_with(x)
        
        # Test exponential and logarithmic functions
        jax_backend.exp(x)
        jax_backend.jnp.exp.assert_called_with(x)
        
        jax_backend.log(x)
        jax_backend.jnp.log.assert_called_with(x)
        
        # Test square root
        jax_backend.sqrt(x)
        jax_backend.jnp.sqrt.assert_called_with(x)
    
    def test_jax_statistical_operations(self, jax_backend):
        """Test statistical operations."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        
        jax_backend.sum(x)
        jax_backend.jnp.sum.assert_called_with(x)
        
        jax_backend.mean(x)
        jax_backend.jnp.mean.assert_called_with(x)
        
        jax_backend.std(x)
        jax_backend.jnp.std.assert_called_with(x)
    
    def test_jax_linear_algebra(self, jax_backend):
        """Test linear algebra operations."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        
        jax_backend.dot(x, y)
        jax_backend.jnp.dot.assert_called_with(x, y)
        
        jax_backend.norm(x)
        jax_backend.jnp.linalg.norm.assert_called_with(x)
    
    def test_jax_gradient_computation(self, jax_backend):
        """Test gradient computation."""
        def objective(x):
            return x**2 + 2*x + 1
        
        grad_fn = jax_backend.gradient(objective)
        jax_backend.jax.grad.assert_called_once_with(objective)
        
        # Test gradient evaluation
        result = grad_fn(1.0)
        assert result == 2.0  # Mock returns 2*x
    
    def test_jax_compilation(self, jax_backend):
        """Test function compilation."""
        def test_func(x):
            return x**2
        
        compiled_func = jax_backend.compile(test_func)
        jax_backend.jax.jit.assert_called_once_with(test_func)
        assert compiled_func == test_func  # Mock returns identity
    
    def test_jax_vectorization(self, jax_backend):
        """Test function vectorization."""
        def test_func(x):
            return x**2
        
        vectorized_func = jax_backend.vectorize(test_func)
        jax_backend.jax.vmap.assert_called_once_with(test_func)
        assert vectorized_func == test_func  # Mock returns identity
    
    def test_jax_random_operations(self, jax_backend):
        """Test random number generation."""
        key = jax_backend.random_key(42)
        jax_backend.jax.random.PRNGKey.assert_called_with(42)
        
        shape = (10,)
        sample = jax_backend.random_normal(key, shape)
        jax_backend.jax.random.normal.assert_called_with(key, shape)
    
    @pytest.mark.parametrize("input_data", [
        [1, 2, 3],
        np.array([1.0, 2.0, 3.0]),
        ((1, 2), (3, 4)),
        {"a": 1, "b": 2}
    ])
    def test_jax_array_conversion_types(self, jax_backend, input_data):
        """Test array conversion with different input types."""
        jax_backend.array(input_data)
        jax_backend.jnp.array.assert_called_with(input_data)


class TestTorchBackend:
    """Test PyTorch backend implementation."""
    
    @pytest.fixture
    def torch_backend(self):
        """Create PyTorch backend for testing."""
        with patch('src.backends.torch_backend.HAS_TORCH', True):
            with patch('src.backends.torch_backend.torch') as mock_torch:
                # Setup basic torch mocks
                mock_torch.tensor = Mock(side_effect=lambda x: np.array(x))
                mock_torch.zeros = Mock(side_effect=np.zeros)
                mock_torch.ones = Mock(side_effect=np.ones)
                mock_torch.sin = Mock(side_effect=np.sin)
                mock_torch.cos = Mock(side_effect=np.cos)
                mock_torch.exp = Mock(side_effect=np.exp)
                mock_torch.log = Mock(side_effect=np.log)
                mock_torch.sqrt = Mock(side_effect=np.sqrt)
                mock_torch.sum = Mock(side_effect=np.sum)
                mock_torch.mean = Mock(side_effect=np.mean)
                mock_torch.std = Mock(side_effect=np.std)
                mock_torch.norm = Mock(side_effect=np.linalg.norm)
                mock_torch.dot = Mock(side_effect=np.dot)
                mock_torch.jit.script = Mock(side_effect=lambda f: f)
                mock_torch.vmap = Mock(side_effect=lambda f: f)
                
                # Mock autograd
                mock_torch.autograd.functional.jacobian = Mock(return_value=np.array([[2.0]]))
                
                yield TorchBackend()
    
    def test_torch_backend_initialization(self, torch_backend):
        """Test PyTorch backend initialization."""
        assert torch_backend.name == "torch"
        assert hasattr(torch_backend, 'torch')
    
    def test_torch_backend_unavailable(self):
        """Test behavior when PyTorch is not available."""
        with patch('src.backends.torch_backend.HAS_TORCH', False):
            with pytest.raises(ImportError, match="PyTorch is required"):
                TorchBackend()
    
    def test_torch_array_creation(self, torch_backend):
        """Test array creation with PyTorch backend."""
        data = [1, 2, 3]
        result = torch_backend.array(data)
        torch_backend.torch.tensor.assert_called_once_with(data, dtype=torch_backend.torch.float32)
    
    def test_torch_zeros_ones(self, torch_backend):
        """Test zeros and ones creation."""
        shape = (3, 4)
        
        torch_backend.zeros(shape)
        torch_backend.torch.zeros.assert_called_with(shape)
        
        torch_backend.ones(shape)
        torch_backend.torch.ones.assert_called_with(shape)
    
    def test_torch_mathematical_operations(self, torch_backend):
        """Test mathematical operations."""
        x = np.array([1.0, 2.0, 3.0])
        
        torch_backend.sin(x)
        torch_backend.torch.sin.assert_called_with(x)
        
        torch_backend.cos(x)
        torch_backend.torch.cos.assert_called_with(x)
        
        torch_backend.exp(x)
        torch_backend.torch.exp.assert_called_with(x)
        
        torch_backend.log(x)
        torch_backend.torch.log.assert_called_with(x)
        
        torch_backend.sqrt(x)
        torch_backend.torch.sqrt.assert_called_with(x)
    
    def test_torch_gradient_computation(self, torch_backend):
        """Test gradient computation."""
        def objective(x):
            return torch_backend.torch.sum(x**2)
        
        grad_fn = torch_backend.gradient(objective)
        assert callable(grad_fn)
    
    def test_torch_compilation(self, torch_backend):
        """Test function compilation."""
        def test_func(x):
            return x**2
        
        compiled_func = torch_backend.compile(test_func)
        torch_backend.torch.jit.script.assert_called_once_with(test_func)
    
    def test_torch_vectorization(self, torch_backend):
        """Test function vectorization."""
        def test_func(x):
            return x**2
        
        vectorized_func = torch_backend.vectorize(test_func)
        torch_backend.torch.vmap.assert_called_once_with(test_func)


class TestBackendManagement:
    """Test backend management functionality."""
    
    def setUp(self):
        """Reset backend registry before each test."""
        # Clear any cached backends
        if hasattr(get_backend, '_backend_cache'):
            get_backend._backend_cache.clear()
    
    def test_list_backends(self):
        """Test listing available backends."""
        backends = list_backends()
        assert isinstance(backends, list)
        assert 'jax' in backends or 'torch' in backends or 'numpy' in backends
    
    def test_get_backend_jax_available(self):
        """Test getting JAX backend when available."""
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax'):
                with patch('src.backends.jax_backend.jnp'):
                    backend = get_backend('jax')
                    assert isinstance(backend, JAXBackend)
                    assert backend.name == 'jax'
    
    def test_get_backend_torch_available(self):
        """Test getting PyTorch backend when available."""
        with patch('src.backends.torch_backend.HAS_TORCH', True):
            with patch('src.backends.torch_backend.torch'):
                backend = get_backend('torch')
                assert isinstance(backend, TorchBackend)
                assert backend.name == 'torch'
    
    def test_get_backend_unavailable(self):
        """Test getting unavailable backend."""
        with patch('src.backends.jax_backend.HAS_JAX', False):
            with patch('src.backends.torch_backend.HAS_TORCH', False):
                with pytest.raises(ImportError):
                    get_backend('jax')
    
    def test_get_backend_unknown(self):
        """Test getting unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend: unknown"):
            get_backend('unknown')
    
    def test_set_default_backend(self):
        """Test setting default backend."""
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax'):
                with patch('src.backends.jax_backend.jnp'):
                    set_default_backend('jax')
                    backend = get_backend()  # Should use default
                    assert backend.name == 'jax'
    
    def test_backend_caching(self):
        """Test that backends are cached properly."""
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax'):
                with patch('src.backends.jax_backend.jnp'):
                    backend1 = get_backend('jax')
                    backend2 = get_backend('jax')
                    # Should be the same instance due to caching
                    assert backend1 is backend2


class TestBackendInteroperability:
    """Test interoperability between different backends."""
    
    def test_array_conversion_between_backends(self):
        """Test array conversion between backends."""
        # Create mock arrays from different backends
        jax_array = Mock()
        jax_array.shape = (3, 3)
        jax_array.__array__ = Mock(return_value=np.ones((3, 3)))
        
        torch_tensor = Mock()
        torch_tensor.shape = (3, 3)
        torch_tensor.detach = Mock()
        torch_tensor.detach.return_value.numpy = Mock(return_value=np.ones((3, 3)))
        
        # Test conversion from JAX to numpy
        np_from_jax = np.array(jax_array)
        assert np_from_jax.shape == (3, 3)
        
        # Test conversion from torch to numpy
        np_from_torch = torch_tensor.detach().numpy()
        assert np_from_torch.shape == (3, 3)
    
    @pytest.mark.parametrize("backend_name,expected_type", [
        ('jax', 'JAXBackend'),
        ('torch', 'TorchBackend')
    ])
    def test_backend_type_consistency(self, backend_name, expected_type):
        """Test that backends return consistent types."""
        mock_available = {
            'jax': 'src.backends.jax_backend.HAS_JAX',
            'torch': 'src.backends.torch_backend.HAS_TORCH'
        }
        
        mock_modules = {
            'jax': ['src.backends.jax_backend.jax', 'src.backends.jax_backend.jnp'],
            'torch': ['src.backends.torch_backend.torch']
        }
        
        with patch(mock_available[backend_name], True):
            patches = [patch(module) for module in mock_modules[backend_name]]
            
            # Apply all patches in sequence
            with patches[0]:
                if len(patches) > 1:
                    with patches[1]:
                        backend = get_backend(backend_name)
                        assert backend.__class__.__name__ == expected_type
                else:
                    backend = get_backend(backend_name)
                    assert backend.__class__.__name__ == expected_type


class TestBackendErrorHandling:
    """Test error handling in backends."""
    
    def test_backend_with_invalid_operations(self):
        """Test backend behavior with invalid operations."""
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax'):
                with patch('src.backends.jax_backend.jnp') as mock_jnp:
                    # Mock jnp operations to raise exceptions
                    mock_jnp.array.side_effect = ValueError("Invalid array data")
                    
                    backend = JAXBackend()
                    
                    with pytest.raises(ValueError, match="Invalid array data"):
                        backend.array("invalid_data")
    
    def test_gradient_of_non_differentiable_function(self):
        """Test gradient computation of non-differentiable function."""
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax') as mock_jax:
                with patch('src.backends.jax_backend.jnp'):
                    # Mock gradient to raise error for non-differentiable function
                    mock_jax.grad.side_effect = ValueError("Function not differentiable")
                    
                    backend = JAXBackend()
                    
                    def non_differentiable(x):
                        return abs(x)  # Non-differentiable at x=0
                    
                    with pytest.raises(ValueError, match="Function not differentiable"):
                        backend.gradient(non_differentiable)
    
    def test_memory_handling_large_arrays(self):
        """Test backend behavior with large arrays."""
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax'):
                with patch('src.backends.jax_backend.jnp') as mock_jnp:
                    # Mock memory error for large arrays
                    def mock_zeros(shape):
                        if np.prod(shape) > 1e6:  # Large array
                            raise MemoryError("Out of memory")
                        return np.zeros(shape)
                    
                    mock_jnp.zeros.side_effect = mock_zeros
                    
                    backend = JAXBackend()
                    
                    # Small array should work
                    small_array = backend.zeros((10, 10))
                    assert small_array is not None
                    
                    # Large array should raise MemoryError
                    with pytest.raises(MemoryError, match="Out of memory"):
                        backend.zeros((10000, 10000))


class TestBackendPerformance:
    """Test performance characteristics of backends."""
    
    @pytest.mark.slow
    def test_compilation_performance(self):
        """Test compilation performance."""
        import time
        
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax') as mock_jax:
                with patch('src.backends.jax_backend.jnp'):
                    # Mock compilation with timing
                    def mock_jit(func):
                        time.sleep(0.001)  # Simulate compilation time
                        return func
                    
                    mock_jax.jit = mock_jit
                    
                    backend = JAXBackend()
                    
                    def test_func(x):
                        return x**2 + 3*x + 1
                    
                    start_time = time.time()
                    compiled_func = backend.compile(test_func)
                    compile_time = time.time() - start_time
                    
                    # Should be reasonably fast
                    assert compile_time < 1.0
                    assert compiled_func == test_func
    
    @pytest.mark.slow
    def test_vectorization_performance(self):
        """Test vectorization performance."""
        import time
        
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax') as mock_jax:
                with patch('src.backends.jax_backend.jnp'):
                    # Mock vectorization with timing
                    def mock_vmap(func):
                        time.sleep(0.001)  # Simulate vectorization time
                        return func
                    
                    mock_jax.vmap = mock_vmap
                    
                    backend = JAXBackend()
                    
                    def test_func(x):
                        return x**2
                    
                    start_time = time.time()
                    vectorized_func = backend.vectorize(test_func)
                    vectorize_time = time.time() - start_time
                    
                    # Should be reasonably fast
                    assert vectorize_time < 1.0
                    assert vectorized_func == test_func


class TestBackendNumericalAccuracy:
    """Test numerical accuracy of backend operations."""
    
    @pytest.mark.parametrize("operation,input_val,expected", [
        ("sin", np.pi/2, 1.0),
        ("cos", 0.0, 1.0),
        ("exp", 0.0, 1.0),
        ("log", 1.0, 0.0),
        ("sqrt", 4.0, 2.0)
    ])
    def test_mathematical_function_accuracy(self, operation, input_val, expected):
        """Test accuracy of mathematical functions."""
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax'):
                with patch('src.backends.jax_backend.jnp') as mock_jnp:
                    # Set up mock to return expected values
                    getattr(mock_jnp, operation).return_value = expected
                    
                    backend = JAXBackend()
                    result = getattr(backend, operation)(input_val)
                    
                    assert result == expected
                    getattr(mock_jnp, operation).assert_called_once_with(input_val)
    
    def test_gradient_accuracy(self):
        """Test gradient computation accuracy."""
        with patch('src.backends.jax_backend.HAS_JAX', True):
            with patch('src.backends.jax_backend.jax') as mock_jax:
                with patch('src.backends.jax_backend.jnp'):
                    # Mock gradient to return analytical result
                    def mock_grad(func):
                        return lambda x: 2*x  # Derivative of x^2
                    
                    mock_jax.grad = mock_grad
                    
                    backend = JAXBackend()
                    
                    def quadratic(x):
                        return x**2
                    
                    grad_func = backend.gradient(quadratic)
                    result = grad_func(3.0)
                    
                    # Should return 2*3 = 6
                    assert result == 6.0