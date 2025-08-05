"""Unit tests for utilities module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.utils.validation import (
    ValidationError, validate_positive, validate_range, validate_type,
    validate_function_space, validate_boundary_conditions
)
from src.utils.error_handling import (
    ErrorHandler, ErrorLevel, ErrorContext, handle_exceptions,
    log_error, create_error_response
)
from src.utils.manufactured_solutions import (
    SolutionType, generate_manufactured_solution, TrigonometricSolution,
    PolynomialSolution, ExponentialSolution
)


class TestValidation:
    """Test cases for validation utilities."""
    
    def test_validate_positive(self):
        """Test positive number validation."""
        # Valid positive numbers
        validate_positive(1.0, "test_param")
        validate_positive(0.1, "test_param")
        validate_positive(1000, "test_param")
        
        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive(0.0, "test_param")
        
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive(-1.0, "test_param")
        
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive(-0.1, "test_param")
    
    def test_validate_range(self):
        """Test range validation."""
        # Valid values in range
        validate_range(5.0, 0.0, 10.0, "test_param")
        validate_range(0.0, 0.0, 10.0, "test_param")  # Edge case
        validate_range(10.0, 0.0, 10.0, "test_param")  # Edge case
        
        # Invalid values outside range
        with pytest.raises(ValidationError, match="must be between"):
            validate_range(-1.0, 0.0, 10.0, "test_param")
        
        with pytest.raises(ValidationError, match="must be between"):
            validate_range(11.0, 0.0, 10.0, "test_param")
        
        # Invalid range specification
        with pytest.raises(ValueError, match="min_val must be less than or equal to max_val"):
            validate_range(5.0, 10.0, 0.0, "test_param")
    
    def test_validate_type(self):
        """Test type validation."""
        # Valid types
        validate_type(5, int, "test_param")
        validate_type(5.0, float, "test_param")
        validate_type("hello", str, "test_param")
        validate_type([1, 2, 3], list, "test_param")
        validate_type({'a': 1}, dict, "test_param")
        
        # Multiple allowed types
        validate_type(5, (int, float), "test_param")
        validate_type(5.0, (int, float), "test_param")
        
        # Invalid types
        with pytest.raises(ValidationError, match="must be of type"):
            validate_type("5", int, "test_param")
        
        with pytest.raises(ValidationError, match="must be of type"):
            validate_type(5, str, "test_param")
    
    def test_validate_function_space(self):
        """Test function space validation."""
        # Mock valid function space
        mock_fs = Mock()
        mock_fs.dim.return_value = 100
        mock_fs.ufl_element.return_value = Mock()
        
        # Should not raise for valid function space
        validate_function_space(mock_fs, "test_fs")
        
        # Test with None
        with pytest.raises(ValidationError, match="Function space cannot be None"):
            validate_function_space(None, "test_fs")
        
        # Test with invalid object
        with pytest.raises(ValidationError, match="must be a valid function space"):
            validate_function_space("not_a_function_space", "test_fs")
    
    def test_validate_boundary_conditions(self):
        """Test boundary condition validation."""
        # Valid boundary conditions
        valid_bcs = {
            'bc1': {
                'type': 'dirichlet',
                'boundary': 1,
                'value': 0.0
            },
            'bc2': {
                'type': 'neumann',
                'boundary': 2,
                'value': 1.0
            }
        }
        
        validate_boundary_conditions(valid_bcs)
        
        # Empty boundary conditions (should be valid)
        validate_boundary_conditions({})
        
        # Missing required fields
        invalid_bcs = {
            'bc1': {
                'type': 'dirichlet',
                # Missing 'boundary' and 'value'
            }
        }
        
        with pytest.raises(ValidationError, match="must contain"):
            validate_boundary_conditions(invalid_bcs)
        
        # Invalid boundary condition type
        invalid_type_bcs = {
            'bc1': {
                'type': 'invalid_type',
                'boundary': 1,
                'value': 0.0
            }
        }
        
        with pytest.raises(ValidationError, match="Invalid boundary condition type"):
            validate_boundary_conditions(invalid_type_bcs)


class TestErrorHandling:
    """Test cases for error handling utilities."""
    
    def test_error_handler_init(self):
        """Test error handler initialization."""
        handler = ErrorHandler()
        assert handler.errors == []
        assert handler.error_counts == {}
    
    def test_log_error(self):
        """Test error logging."""
        handler = ErrorHandler()
        
        # Log different types of errors
        handler.log_error(ErrorLevel.WARNING, "Test warning", {"context": "test"})
        handler.log_error(ErrorLevel.ERROR, "Test error", {"context": "test"})
        handler.log_error(ErrorLevel.CRITICAL, "Critical error", {"context": "test"})
        
        assert len(handler.errors) == 3
        assert handler.error_counts[ErrorLevel.WARNING] == 1
        assert handler.error_counts[ErrorLevel.ERROR] == 1
        assert handler.error_counts[ErrorLevel.CRITICAL] == 1
        
        # Check error details
        warning_error = handler.errors[0]
        assert warning_error.level == ErrorLevel.WARNING
        assert warning_error.message == "Test warning"
        assert warning_error.context["context"] == "test"
    
    def test_get_errors_by_level(self):
        """Test filtering errors by level."""
        handler = ErrorHandler()
        
        handler.log_error(ErrorLevel.WARNING, "Warning 1")
        handler.log_error(ErrorLevel.ERROR, "Error 1") 
        handler.log_error(ErrorLevel.WARNING, "Warning 2")
        handler.log_error(ErrorLevel.CRITICAL, "Critical 1")
        
        warnings = handler.get_errors_by_level(ErrorLevel.WARNING)
        assert len(warnings) == 2
        assert all(e.level == ErrorLevel.WARNING for e in warnings)
        
        errors = handler.get_errors_by_level(ErrorLevel.ERROR)
        assert len(errors) == 1
        assert errors[0].message == "Error 1"
    
    def test_has_critical_errors(self):
        """Test checking for critical errors."""
        handler = ErrorHandler()
        
        assert handler.has_critical_errors() is False
        
        handler.log_error(ErrorLevel.WARNING, "Warning")
        assert handler.has_critical_errors() is False
        
        handler.log_error(ErrorLevel.CRITICAL, "Critical error")
        assert handler.has_critical_errors() is True
    
    def test_clear_errors(self):
        """Test clearing errors."""
        handler = ErrorHandler()
        
        handler.log_error(ErrorLevel.ERROR, "Test error")
        assert len(handler.errors) == 1
        
        handler.clear_errors()
        assert len(handler.errors) == 0
        assert handler.error_counts == {}
    
    def test_generate_error_report(self):
        """Test generating error reports."""
        handler = ErrorHandler()
        
        handler.log_error(ErrorLevel.WARNING, "Warning 1")
        handler.log_error(ErrorLevel.ERROR, "Error 1")
        handler.log_error(ErrorLevel.WARNING, "Warning 2")
        
        report = handler.generate_error_report()
        
        assert 'total_errors' in report
        assert 'errors_by_level' in report
        assert 'recent_errors' in report
        
        assert report['total_errors'] == 3
        assert report['errors_by_level'][ErrorLevel.WARNING.value] == 2
        assert report['errors_by_level'][ErrorLevel.ERROR.value] == 1
    
    def test_handle_exceptions_decorator(self):
        """Test exception handling decorator."""
        @handle_exceptions(default_return="default_value")
        def test_function(should_fail=False):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Normal execution
        result = test_function(should_fail=False)
        assert result == "success"
        
        # Exception handling
        result = test_function(should_fail=True)
        assert result == "default_value"
    
    def test_log_error_function(self):
        """Test standalone log_error function."""
        with patch('src.utils.error_handling.logger') as mock_logger:
            log_error("Test error message", {"context": "test"})
            mock_logger.error.assert_called_once()
    
    def test_create_error_response(self):
        """Test error response creation."""
        response = create_error_response("Test error", 400, {"detail": "more info"})
        
        assert response['error'] is True
        assert response['message'] == "Test error"
        assert response['status_code'] == 400
        assert response['details']['detail'] == "more info"
        assert 'timestamp' in response


class TestManufacturedSolutions:
    """Test cases for manufactured solutions."""
    
    def test_trigonometric_solution(self):
        """Test trigonometric manufactured solution."""
        solution = TrigonometricSolution(dimension=2, frequency=1.0, amplitude=2.0)
        
        # Test solution evaluation
        x = [0.5, 0.5]
        u_val = solution.solution(x)
        assert isinstance(u_val, (int, float))
        assert abs(u_val) <= 2.0  # Should be within amplitude
        
        # Test source evaluation
        f_val = solution.source(x)
        assert isinstance(f_val, (int, float))
        
        # Test gradient
        grad_val = solution.gradient(x)
        assert len(grad_val) == 2  # 2D gradient
        
        # Test different dimensions
        solution_1d = TrigonometricSolution(dimension=1, frequency=2.0)
        x_1d = [0.5]
        u_1d = solution_1d.solution(x_1d)
        assert isinstance(u_1d, (int, float))
        
        solution_3d = TrigonometricSolution(dimension=3, frequency=1.0)
        x_3d = [0.5, 0.5, 0.5]
        u_3d = solution_3d.solution(x_3d)
        assert isinstance(u_3d, (int, float))
    
    def test_polynomial_solution(self):
        """Test polynomial manufactured solution."""
        solution = PolynomialSolution(dimension=2, degree=2, coefficients=[1.0, 2.0, 3.0])
        
        # Test solution evaluation
        x = [0.5, 0.5]
        u_val = solution.solution(x)
        assert isinstance(u_val, (int, float))
        
        # Test source evaluation
        f_val = solution.source(x)
        assert isinstance(f_val, (int, float))
        
        # Test gradient
        grad_val = solution.gradient(x)
        assert len(grad_val) == 2
    
    def test_exponential_solution(self):
        """Test exponential manufactured solution."""
        solution = ExponentialSolution(dimension=2, decay_rate=1.0, amplitude=1.0)
        
        # Test solution evaluation
        x = [0.5, 0.5]
        u_val = solution.solution(x)
        assert isinstance(u_val, (int, float))
        assert u_val > 0  # Exponential should be positive
        
        # Test source evaluation
        f_val = solution.source(x)
        assert isinstance(f_val, (int, float))
        
        # Test gradient
        grad_val = solution.gradient(x)
        assert len(grad_val) == 2
    
    def test_generate_manufactured_solution(self):
        """Test manufactured solution factory function."""
        # Trigonometric solution
        trig_solution = generate_manufactured_solution(
            solution_type=SolutionType.TRIGONOMETRIC,
            dimension=2,
            parameters={'frequency': 2.0, 'amplitude': 1.5}
        )
        
        assert 'type' in trig_solution
        assert 'solution' in trig_solution
        assert 'source' in trig_solution
        assert 'gradient' in trig_solution
        assert trig_solution['type'] == SolutionType.TRIGONOMETRIC.value
        
        # Test solution function
        x = [0.5, 0.5]
        u_val = trig_solution['solution'](x)
        assert isinstance(u_val, (int, float))
        
        # Polynomial solution
        poly_solution = generate_manufactured_solution(
            solution_type=SolutionType.POLYNOMIAL,
            dimension=2,
            parameters={'degree': 3, 'coefficients': [1.0, 2.0, 1.0, 0.5]}
        )
        
        assert poly_solution['type'] == SolutionType.POLYNOMIAL.value
        
        # Exponential solution
        exp_solution = generate_manufactured_solution(
            solution_type=SolutionType.EXPONENTIAL,
            dimension=2,
            parameters={'decay_rate': 0.5, 'amplitude': 2.0}
        )
        
        assert exp_solution['type'] == SolutionType.EXPONENTIAL.value
    
    def test_manufactured_solution_errors(self):
        """Test error handling in manufactured solutions."""
        # Invalid solution type
        with pytest.raises(ValueError, match="Unsupported solution type"):
            generate_manufactured_solution(
                solution_type="invalid_type",
                dimension=2
            )
        
        # Invalid dimension
        with pytest.raises(ValueError, match="Dimension must be positive"):
            TrigonometricSolution(dimension=0)
        
        with pytest.raises(ValueError, match="Dimension must be positive"):
            TrigonometricSolution(dimension=-1)
        
        # Invalid parameters
        with pytest.raises(ValueError, match="Frequency must be positive"):
            TrigonometricSolution(dimension=2, frequency=0.0)
        
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            PolynomialSolution(dimension=2, degree=-1)


class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    def test_validation_error_handling_integration(self):
        """Test validation and error handling working together."""
        handler = ErrorHandler()
        
        def validate_with_error_handling(value, param_name):
            try:
                validate_positive(value, param_name)
                return True
            except ValidationError as e:
                handler.log_error(ErrorLevel.ERROR, str(e), {"parameter": param_name, "value": value})
                return False
        
        # Valid case
        assert validate_with_error_handling(5.0, "test_param") is True
        assert len(handler.errors) == 0
        
        # Invalid case
        assert validate_with_error_handling(-1.0, "test_param") is False
        assert len(handler.errors) == 1
        assert handler.errors[0].level == ErrorLevel.ERROR
    
    def test_manufactured_solution_validation(self):
        """Test manufactured solution with validation."""
        # Valid parameters
        try:
            solution = generate_manufactured_solution(
                solution_type=SolutionType.TRIGONOMETRIC,
                dimension=2,
                parameters={'frequency': 1.0, 'amplitude': 1.0}
            )
            assert solution is not None
        except ValidationError:
            pytest.fail("Valid parameters should not raise ValidationError")
        
        # Invalid parameters should be caught by internal validation
        with pytest.raises(ValueError):  # Should raise ValueError for invalid frequency
            TrigonometricSolution(dimension=2, frequency=-1.0)
    
    def test_error_context_with_manufactured_solutions(self):
        """Test error context in manufactured solution generation."""
        handler = ErrorHandler()
        
        try:
            # This should fail
            TrigonometricSolution(dimension=2, frequency=0.0)
        except ValueError as e:
            handler.log_error(
                ErrorLevel.ERROR,
                "Failed to create manufactured solution",
                {
                    "solution_type": "trigonometric",
                    "dimension": 2,
                    "frequency": 0.0,
                    "original_error": str(e)
                }
            )
        
        assert len(handler.errors) == 1
        error = handler.errors[0]
        assert error.context["solution_type"] == "trigonometric"
        assert error.context["frequency"] == 0.0


class TestUtilsEdgeCases:
    """Test edge cases and error conditions in utilities."""
    
    def test_validation_edge_cases(self):
        """Test validation with edge cases."""
        # Very small positive number
        validate_positive(1e-10, "tiny_param")
        
        # Very large number
        validate_positive(1e10, "large_param")
        
        # Range validation with same min and max
        validate_range(5.0, 5.0, 5.0, "exact_param")
        
        # Type validation with complex types
        import numpy as np
        validate_type(np.array([1, 2, 3]), np.ndarray, "array_param")
    
    def test_error_handling_edge_cases(self):
        """Test error handling with edge cases."""
        handler = ErrorHandler()
        
        # Empty error message
        handler.log_error(ErrorLevel.INFO, "", {})
        assert len(handler.errors) == 1
        
        # Very long error message
        long_message = "x" * 10000
        handler.log_error(ErrorLevel.WARNING, long_message, {})
        assert len(handler.errors) == 2
        
        # Error with None context
        handler.log_error(ErrorLevel.ERROR, "Test error", None)
        assert len(handler.errors) == 3
        assert handler.errors[2].context == {}  # Should default to empty dict
    
    def test_manufactured_solution_edge_cases(self):
        """Test manufactured solutions with edge cases."""
        # Very high frequency
        solution = TrigonometricSolution(dimension=2, frequency=100.0)
        x = [0.1, 0.1]
        u_val = solution.solution(x)
        assert isinstance(u_val, (int, float))
        
        # Very small amplitude
        solution = TrigonometricSolution(dimension=2, amplitude=1e-10)
        u_val = solution.solution(x)
        assert abs(u_val) <= 1e-10
        
        # High degree polynomial
        solution = PolynomialSolution(dimension=2, degree=10)
        u_val = solution.solution(x)
        assert isinstance(u_val, (int, float))
        
        # Very small decay rate
        solution = ExponentialSolution(dimension=2, decay_rate=1e-5)
        u_val = solution.solution(x)
        assert u_val > 0