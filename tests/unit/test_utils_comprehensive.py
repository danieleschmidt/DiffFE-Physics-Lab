"""Comprehensive unit tests for utility functions and modules."""

import pytest
import numpy as np
import tempfile
import os
import json
import logging
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from typing import Dict, Any

from src.utils.validation import ValidationError
from src.utils.exceptions import SolverError, ConvergenceError
from src.utils.config_manager import ConfigManager
from src.utils.logging_config import get_logger
from src.utils.manufactured_solutions import generate_manufactured_solution, SolutionType

try:
    from src.utils.error_computation import compute_error, compute_convergence_rate
    HAS_ERROR_COMPUTATION = True
except ImportError:
    HAS_ERROR_COMPUTATION = False


class TestConfigManager:
    """Test configuration management utilities."""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager()
        assert hasattr(config_manager, 'config')
        assert isinstance(config_manager.config, dict)
    
    def test_config_manager_with_file(self):
        """Test ConfigManager with configuration file."""
        config_data = {
            'backend': 'jax',
            'mesh_size': 32,
            'tolerance': 1e-6,
            'max_iterations': 100
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
                config_manager = ConfigManager(config_file=temp_file)
                assert config_manager.get('backend') == 'jax'
                assert config_manager.get('mesh_size') == 32
        finally:
            os.unlink(temp_file)
    
    def test_config_manager_set_get(self):
        """Test setting and getting configuration values."""
        config_manager = ConfigManager()
        
        config_manager.set('test_key', 'test_value')
        assert config_manager.get('test_key') == 'test_value'
        
        # Test default value
        assert config_manager.get('nonexistent_key', 'default') == 'default'
        assert config_manager.get('nonexistent_key') is None
    
    def test_config_manager_nested_keys(self):
        """Test nested configuration keys."""
        config_manager = ConfigManager()
        
        config_manager.set('solver.tolerance', 1e-8)
        config_manager.set('solver.max_iterations', 200)
        
        assert config_manager.get('solver.tolerance') == 1e-8
        assert config_manager.get('solver.max_iterations') == 200
    
    def test_config_manager_update(self):
        """Test updating configuration."""
        config_manager = ConfigManager()
        
        update_data = {
            'backend': 'torch',
            'solver': {
                'tolerance': 1e-10,
                'method': 'newton'
            }
        }
        
        config_manager.update(update_data)
        assert config_manager.get('backend') == 'torch'
        assert config_manager.get('solver.tolerance') == 1e-10
        assert config_manager.get('solver.method') == 'newton'
    
    def test_config_manager_validate(self):
        """Test configuration validation."""
        config_manager = ConfigManager()
        
        # Define validation schema
        schema = {
            'backend': {'type': str, 'allowed': ['jax', 'torch', 'numpy']},
            'tolerance': {'type': float, 'min': 0.0},
            'max_iterations': {'type': int, 'min': 1}
        }
        
        # Valid configuration
        valid_config = {
            'backend': 'jax',
            'tolerance': 1e-6,
            'max_iterations': 100
        }
        
        config_manager.update(valid_config)
        
        # Mock validation (would need actual implementation)
        assert config_manager.validate(schema) is True
    
    def test_config_manager_save_load(self):
        """Test saving and loading configuration."""
        config_manager = ConfigManager()
        config_manager.set('test_param', 42)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save configuration
            config_manager.save(temp_file)
            
            # Load in new manager
            new_manager = ConfigManager(config_file=temp_file)
            assert new_manager.get('test_param') == 42
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_config_manager_environment_override(self):
        """Test environment variable override."""
        with patch.dict(os.environ, {'DIFFFE_BACKEND': 'torch', 'DIFFFE_TOLERANCE': '1e-8'}):
            config_manager = ConfigManager()
            config_manager.load_from_environment('DIFFFE_')
            
            assert config_manager.get('backend') == 'torch'
            assert config_manager.get('tolerance') == '1e-8'


class TestLoggingConfig:
    """Test logging configuration utilities."""
    
    def test_get_logger(self):
        """Test logger retrieval."""
        logger = get_logger('test_module')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_module'
    
    def test_logger_with_level(self):
        """Test logger with specific level."""
        logger = get_logger('test_module', level=logging.DEBUG)
        assert logger.level == logging.DEBUG
    
    def test_multiple_loggers(self):
        """Test multiple logger instances."""
        logger1 = get_logger('module1')
        logger2 = get_logger('module2')
        
        assert logger1.name == 'module1'
        assert logger2.name == 'module2'
        assert logger1 is not logger2
    
    def test_logger_hierarchy(self):
        """Test logger hierarchy."""
        parent_logger = get_logger('parent')
        child_logger = get_logger('parent.child')
        
        assert child_logger.parent.name == 'parent'
        assert parent_logger.name == 'parent'
    
    @patch('src.utils.logging_config.logging.basicConfig')
    def test_setup_logging(self, mock_basic_config):
        """Test logging setup."""
        from src.utils.logging_config import setup_logging
        
        setup_logging(level=logging.INFO, format='%(message)s')
        mock_basic_config.assert_called_once()


class TestExceptions:
    """Test custom exception classes."""
    
    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Invalid parameter value")
        assert str(error) == "Invalid parameter value"
        assert isinstance(error, Exception)
    
    def test_validation_error_with_context(self):
        """Test ValidationError with context."""
        context = {'parameter': 'diffusion_coeff', 'value': -1.0}
        error = ValidationError("Negative diffusion coefficient", context=context)
        
        assert "Negative diffusion coefficient" in str(error)
        assert hasattr(error, 'context')
        assert error.context == context
    
    def test_solver_error(self):
        """Test SolverError exception."""
        error = SolverError("Solver failed to converge")
        assert str(error) == "Solver failed to converge"
        assert isinstance(error, Exception)
    
    def test_solver_error_with_details(self):
        """Test SolverError with solver details."""
        details = {
            'iterations': 1000,
            'residual': 1e-3,
            'tolerance': 1e-6
        }
        error = SolverError("Convergence failed", solver_details=details)
        
        assert hasattr(error, 'solver_details')
        assert error.solver_details == details
    
    def test_convergence_error(self):
        """Test ConvergenceError exception."""
        error = ConvergenceError("Failed to achieve convergence rate")
        assert str(error) == "Failed to achieve convergence rate"
        assert isinstance(error, Exception)
    
    def test_convergence_error_with_data(self):
        """Test ConvergenceError with convergence data."""
        data = {
            'expected_rate': 2.0,
            'actual_rate': 1.2,
            'mesh_sizes': [0.1, 0.05, 0.025],
            'errors': [1e-2, 3e-3, 8e-4]
        }
        error = ConvergenceError("Poor convergence rate", convergence_data=data)
        
        assert hasattr(error, 'convergence_data')
        assert error.convergence_data == data


class TestManufacturedSolutions:
    """Test manufactured solution utilities."""
    
    def test_generate_trigonometric_solution(self):
        """Test trigonometric manufactured solution generation."""
        solution = generate_manufactured_solution(
            SolutionType.TRIGONOMETRIC,
            dimension=2,
            parameters={'frequency': 1.0, 'amplitude': 1.0}
        )
        
        assert 'solution' in solution
        assert 'source' in solution
        assert 'gradient' in solution
        
        # Test evaluation
        x = [0.5, 0.5]
        u_val = solution['solution'](x)
        f_val = solution['source'](x)
        grad_val = solution['gradient'](x)
        
        assert isinstance(u_val, (int, float))
        assert isinstance(f_val, (int, float))
        assert isinstance(grad_val, (list, tuple, np.ndarray))
        assert len(grad_val) == 2
    
    def test_generate_polynomial_solution(self):
        """Test polynomial manufactured solution generation."""
        solution = generate_manufactured_solution(
            SolutionType.POLYNOMIAL,
            dimension=2,
            parameters={'degree': 2}
        )
        
        x = [0.3, 0.7]
        u_val = solution['solution'](x)
        f_val = solution['source'](x)
        grad_val = solution['gradient'](x)
        
        assert isinstance(u_val, (int, float))
        assert isinstance(f_val, (int, float))
        assert len(grad_val) == 2
    
    def test_generate_exponential_solution(self):
        """Test exponential manufactured solution generation."""
        solution = generate_manufactured_solution(
            SolutionType.EXPONENTIAL,
            dimension=2,
            parameters={'decay_rate': 1.0}
        )
        
        x = [0.2, 0.8]
        u_val = solution['solution'](x)
        f_val = solution['source'](x)
        grad_val = solution['gradient'](x)
        
        assert isinstance(u_val, (int, float))
        assert isinstance(f_val, (int, float))
        assert len(grad_val) == 2
        assert u_val > 0  # Exponential should be positive
    
    def test_manufactured_solution_different_dimensions(self):
        """Test manufactured solutions in different dimensions."""
        # 1D solution
        solution_1d = generate_manufactured_solution(
            SolutionType.TRIGONOMETRIC,
            dimension=1,
            parameters={'frequency': 2.0}
        )
        
        x_1d = [0.5]
        u_1d = solution_1d['solution'](x_1d)
        grad_1d = solution_1d['gradient'](x_1d)
        
        assert isinstance(u_1d, (int, float))
        assert len(grad_1d) == 1
        
        # 3D solution
        solution_3d = generate_manufactured_solution(
            SolutionType.TRIGONOMETRIC,
            dimension=3,
            parameters={'frequency': 1.0}
        )
        
        x_3d = [0.3, 0.5, 0.7]
        u_3d = solution_3d['solution'](x_3d)
        grad_3d = solution_3d['gradient'](x_3d)
        
        assert isinstance(u_3d, (int, float))
        assert len(grad_3d) == 3
    
    def test_manufactured_solution_parameter_validation(self):
        """Test parameter validation in manufactured solutions."""
        # Invalid dimension
        with pytest.raises(ValueError, match="Dimension must be positive"):
            generate_manufactured_solution(
                SolutionType.TRIGONOMETRIC,
                dimension=0
            )
        
        # Invalid solution type
        with pytest.raises(ValueError, match="Unsupported solution type"):
            generate_manufactured_solution(
                "invalid_type",
                dimension=2
            )
        
        # Invalid frequency
        with pytest.raises(ValueError):
            generate_manufactured_solution(
                SolutionType.TRIGONOMETRIC,
                dimension=2,
                parameters={'frequency': -1.0}
            )
    
    def test_manufactured_solution_consistency(self):
        """Test consistency of manufactured solutions."""
        solution = generate_manufactured_solution(
            SolutionType.TRIGONOMETRIC,
            dimension=2,
            parameters={'frequency': 1.0}
        )
        
        # Test same point multiple times
        x = [0.4, 0.6]
        u1 = solution['solution'](x)
        u2 = solution['solution'](x)
        
        assert u1 == u2  # Should be deterministic
        
        # Test gradient consistency
        grad1 = solution['gradient'](x)
        grad2 = solution['gradient'](x)
        
        np.testing.assert_array_equal(grad1, grad2)


@pytest.mark.skipif(not HAS_ERROR_COMPUTATION, reason="Error computation module not available")
class TestErrorComputation:
    """Test error computation utilities."""
    
    def test_compute_l2_error_simple(self):
        """Test L2 error computation with simple functions."""
        def exact_solution(x):
            return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        
        def approximate_solution(x):
            return 0.9 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        
        # Mock function space and solutions for testing
        points = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
        
        l2_error = compute_error(approximate_solution, exact_solution, 'L2', points=points)
        assert l2_error > 0
        assert l2_error < 1.0  # Should be reasonable
    
    def test_compute_convergence_rate(self):
        """Test convergence rate computation."""
        mesh_sizes = [0.2, 0.1, 0.05, 0.025]
        errors = [4e-2, 1e-2, 2.5e-3, 6.25e-4]  # Rate ≈ 2
        
        result = compute_convergence_rate(mesh_sizes, errors)
        
        assert 'rate' in result
        assert 'r_squared' in result
        assert 'errors_fitted' in result
        
        # Should detect approximately order 2 convergence
        assert abs(result['rate'] - 2.0) < 0.5
        assert result['r_squared'] > 0.95  # Good fit
    
    def test_convergence_rate_edge_cases(self):
        """Test convergence rate computation edge cases."""
        # Too few data points
        with pytest.raises(ValueError):
            compute_convergence_rate([0.1], [1e-2])
        
        # Zero errors
        mesh_sizes = [0.2, 0.1, 0.05]
        zero_errors = [0.0, 0.0, 0.0]
        
        with pytest.raises(ValueError, match="Errors contain zeros or negative values"):
            compute_convergence_rate(mesh_sizes, zero_errors)
        
        # Negative errors
        negative_errors = [1e-2, -1e-3, 1e-4]
        with pytest.raises(ValueError, match="Errors contain zeros or negative values"):
            compute_convergence_rate(mesh_sizes, negative_errors)
    
    def test_error_norms_different_types(self):
        """Test different error norm types."""
        def exact(x):
            return x[0]**2 + x[1]**2
        
        def approx(x):
            return 0.95 * (x[0]**2 + x[1]**2)
        
        points = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]])
        
        # L2 error
        l2_error = compute_error(approx, exact, 'L2', points=points)
        assert l2_error > 0
        
        # Max error  
        max_error = compute_error(approx, exact, 'max', points=points)
        assert max_error > 0
        
        # Max error should be >= L2 error in general
        assert max_error >= l2_error


class TestUtilsIntegration:
    """Integration tests combining multiple utilities."""
    
    def test_config_logging_integration(self):
        """Test configuration and logging working together."""
        config_manager = ConfigManager()
        config_manager.set('logging.level', 'DEBUG')
        config_manager.set('logging.format', '%(levelname)s: %(message)s')
        
        # Get logger with config
        logger = get_logger('test_integration')
        assert logger.name == 'test_integration'
    
    def test_manufactured_solution_error_computation(self):
        """Test manufactured solutions with error computation."""
        if not HAS_ERROR_COMPUTATION:
            pytest.skip("Error computation not available")
        
        # Generate manufactured solution
        ms = generate_manufactured_solution(
            SolutionType.POLYNOMIAL,
            dimension=2,
            parameters={'degree': 2}
        )
        
        exact = ms['solution']
        
        # Create approximate solution (with some error)
        def approximate(x):
            return 0.95 * exact(x)
        
        # Compute error
        points = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]])
        error = compute_error(approximate, exact, 'L2', points=points)
        
        assert error > 0
        assert error < 1.0  # Should be reasonable
    
    def test_error_handling_with_config(self):
        """Test error handling with configuration."""
        config_manager = ConfigManager()
        config_manager.set('error_handling.log_level', 'ERROR')
        config_manager.set('error_handling.max_errors', 10)
        
        # Test that config affects error handling behavior
        assert config_manager.get('error_handling.log_level') == 'ERROR'


class TestUtilsPerformance:
    """Performance tests for utility functions."""
    
    @pytest.mark.slow
    def test_manufactured_solution_performance(self):
        """Test performance of manufactured solution evaluation."""
        import time
        
        solution = generate_manufactured_solution(
            SolutionType.TRIGONOMETRIC,
            dimension=2,
            parameters={'frequency': 1.0}
        )
        
        # Time multiple evaluations
        points = [[i*0.1, j*0.1] for i in range(10) for j in range(10)]
        
        start_time = time.time()
        for point in points:
            solution['solution'](point)
            solution['gradient'](point)
            solution['source'](point)
        execution_time = time.time() - start_time
        
        # Should be fast for 300 evaluations
        assert execution_time < 1.0, f"Too slow: {execution_time}s for 300 evaluations"
    
    @pytest.mark.slow
    def test_config_manager_performance(self):
        """Test performance of configuration operations."""
        import time
        
        config_manager = ConfigManager()
        
        # Time many set/get operations
        start_time = time.time()
        for i in range(1000):
            config_manager.set(f'param_{i}', i)
            config_manager.get(f'param_{i}')
        execution_time = time.time() - start_time
        
        # Should be fast for 2000 operations
        assert execution_time < 0.1, f"Config operations too slow: {execution_time}s"


class TestUtilsRobustness:
    """Test robustness and edge cases of utility functions."""
    
    def test_config_manager_with_invalid_json(self):
        """Test ConfigManager with invalid JSON."""
        invalid_json = '{"key": invalid_value}'
        
        with patch('builtins.open', mock_open(read_data=invalid_json)):
            with pytest.raises(json.JSONDecodeError):
                ConfigManager(config_file='invalid.json')
    
    def test_manufactured_solution_extreme_parameters(self):
        """Test manufactured solutions with extreme parameters."""
        # Very high frequency
        solution = generate_manufactured_solution(
            SolutionType.TRIGONOMETRIC,
            dimension=2,
            parameters={'frequency': 1000.0, 'amplitude': 1.0}
        )
        
        x = [0.001, 0.001]  # Small coordinates
        u_val = solution['solution'](x)
        
        assert isinstance(u_val, (int, float))
        assert np.isfinite(u_val)
        
        # Very small amplitude
        solution = generate_manufactured_solution(
            SolutionType.TRIGONOMETRIC,
            dimension=2,
            parameters={'frequency': 1.0, 'amplitude': 1e-15}
        )
        
        u_val = solution['solution'](x)
        assert abs(u_val) <= 1e-14  # Should be very small
    
    def test_error_computation_edge_cases(self):
        """Test error computation with edge cases."""
        if not HAS_ERROR_COMPUTATION:
            pytest.skip("Error computation not available")
        
        def zero_function(x):
            return 0.0
        
        def small_function(x):
            return 1e-15
        
        points = np.array([[0.5, 0.5]])
        
        # Error between zero functions should be zero
        error = compute_error(zero_function, zero_function, 'L2', points=points)
        assert error == 0.0
        
        # Error with very small values
        error = compute_error(small_function, zero_function, 'L2', points=points)
        assert error >= 0.0
        assert error < 1e-14
    
    def test_config_manager_circular_references(self):
        """Test ConfigManager with circular reference patterns."""
        config_manager = ConfigManager()
        
        # This shouldn't cause infinite recursion
        config_manager.set('a.b', 'value_b')
        config_manager.set('a.c', 'value_c')
        
        assert config_manager.get('a.b') == 'value_b'
        assert config_manager.get('a.c') == 'value_c'
    
    def test_exception_chaining(self):
        """Test proper exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise SolverError("Solver failed") from e
        except SolverError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original error"