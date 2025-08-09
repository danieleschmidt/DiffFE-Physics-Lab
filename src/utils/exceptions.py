"""Comprehensive exception classes for robust error handling."""

import traceback
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Standardized error codes for the framework."""
    
    # General errors (1000-1999)
    UNKNOWN_ERROR = 1000
    INVALID_INPUT = 1001
    CONFIGURATION_ERROR = 1002
    DEPENDENCY_MISSING = 1003
    RESOURCE_UNAVAILABLE = 1004
    OPERATION_TIMEOUT = 1005
    
    # Validation errors (2000-2999)
    INVALID_MESH = 2000
    INVALID_FUNCTION_SPACE = 2001
    INVALID_BOUNDARY_CONDITIONS = 2002
    INVALID_PARAMETERS = 2003
    MESH_QUALITY_ERROR = 2004
    PHYSICS_CONSTRAINT_VIOLATION = 2005
    
    # Backend errors (3000-3999)
    BACKEND_UNAVAILABLE = 3000
    BACKEND_INITIALIZATION_FAILED = 3001
    COMPUTATION_ERROR = 3002
    MEMORY_ERROR = 3003
    GPU_ERROR = 3004
    
    # Solver errors (4000-4999)
    SOLVER_CONVERGENCE_FAILED = 4000
    SOLVER_SETUP_ERROR = 4001
    LINEAR_SYSTEM_ERROR = 4002
    NONLINEAR_SOLVER_ERROR = 4003
    ASSEMBLY_ERROR = 4004
    
    # Optimization errors (5000-5999)
    OPTIMIZATION_FAILED = 5000
    INVALID_OBJECTIVE = 5001
    CONSTRAINT_VIOLATION = 5002
    GRADIENT_COMPUTATION_ERROR = 5003
    
    # API errors (6000-6999)
    API_REQUEST_ERROR = 6000
    AUTHENTICATION_ERROR = 6001
    AUTHORIZATION_ERROR = 6002
    RATE_LIMIT_EXCEEDED = 6003
    MALFORMED_REQUEST = 6004
    
    # Security errors (7000-7999)
    SECURITY_VIOLATION = 7000
    INJECTION_ATTEMPT = 7001
    PATH_TRAVERSAL_ATTEMPT = 7002
    INVALID_FILE_ACCESS = 7003
    SUSPICIOUS_INPUT = 7004


class DiffFEError(Exception):
    """Base exception class for all DiffFE framework errors.
    
    Provides standardized error handling with error codes, detailed messages,
    context information, and optional suggestions for resolution.
    
    Parameters
    ----------
    message : str
        Human-readable error message
    error_code : ErrorCode, optional
        Standardized error code
    context : Dict[str, Any], optional
        Additional context information
    suggestion : str, optional
        Suggested resolution or next steps
    cause : Exception, optional
        Original exception that caused this error
    
    Examples
    --------
    >>> raise DiffFEError(
    ...     "Invalid mesh topology",
    ...     error_code=ErrorCode.INVALID_MESH,
    ...     context={'num_cells': 0, 'mesh_dim': 2},
    ...     suggestion="Ensure mesh contains at least one cell"
    ... )
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.suggestion = suggestion
        self.cause = cause
        self.traceback_info = traceback.format_exc() if cause else None
        
        # Log the error
        self._log_error()
    
    def _log_error(self):
        """Log the error with appropriate level."""
        log_data = {
            'error_code': self.error_code.name,
            'message': self.message,
            'context': self.context
        }
        
        if self.error_code.value >= 7000:  # Security errors
            logger.critical(f"Security Error [{self.error_code.name}]: {self.message}", extra=log_data)
        elif self.error_code.value >= 4000:  # Solver/computation errors
            logger.error(f"Computation Error [{self.error_code.name}]: {self.message}", extra=log_data)
        else:
            logger.warning(f"Framework Error [{self.error_code.name}]: {self.message}", extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing error information
        """
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code.name,
            'error_code_value': self.error_code.value,
            'message': self.message,
            'context': self.context,
            'suggestion': self.suggestion,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
    
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"[{self.error_code.name}] {self.message}"]
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        
        return " | ".join(parts)


class ValidationError(DiffFEError):
    """Exception for input validation and constraint violations."""
    
    def __init__(
        self,
        message: str,
        invalid_field: Optional[str] = None,
        expected_type: Optional[type] = None,
        actual_value: Any = None,
        constraint: Optional[str] = None,
        **kwargs
    ):
        # Build context from parameters
        context = kwargs.get('context', {})
        if invalid_field:
            context['invalid_field'] = invalid_field
        if expected_type:
            context['expected_type'] = expected_type.__name__
        if actual_value is not None:
            context['actual_value'] = str(actual_value)
        if constraint:
            context['constraint'] = constraint
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', ErrorCode.INVALID_INPUT)
        
        super().__init__(message, **kwargs)


class MeshValidationError(ValidationError):
    """Exception for mesh-related validation errors."""
    
    def __init__(
        self,
        message: str,
        mesh_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if mesh_info:
            context.update(mesh_info)
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', ErrorCode.INVALID_MESH)
        
        super().__init__(message, **kwargs)


class PhysicsConstraintError(ValidationError):
    """Exception for physics constraint violations."""
    
    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        parameter_value: Any = None,
        valid_range: Optional[tuple] = None,
        physics_law: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if parameter_name:
            context['parameter'] = parameter_name
        if parameter_value is not None:
            context['value'] = parameter_value
        if valid_range:
            context['valid_range'] = valid_range
        if physics_law:
            context['physics_law'] = physics_law
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', ErrorCode.PHYSICS_CONSTRAINT_VIOLATION)
        
        super().__init__(message, **kwargs)


class BackendError(DiffFEError):
    """Exception for automatic differentiation backend errors."""
    
    def __init__(
        self,
        message: str,
        backend_name: Optional[str] = None,
        available_backends: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if backend_name:
            context['backend'] = backend_name
        if available_backends:
            context['available_backends'] = available_backends
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', ErrorCode.BACKEND_UNAVAILABLE)
        
        super().__init__(message, **kwargs)


class SolverError(DiffFEError):
    """Exception for numerical solver errors."""
    
    def __init__(
        self,
        message: str,
        solver_type: Optional[str] = None,
        convergence_info: Optional[Dict[str, Any]] = None,
        iteration_count: Optional[int] = None,
        residual_norm: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if solver_type:
            context['solver_type'] = solver_type
        if convergence_info:
            context['convergence_info'] = convergence_info
        if iteration_count is not None:
            context['iterations'] = iteration_count
        if residual_norm is not None:
            context['residual_norm'] = residual_norm
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', ErrorCode.SOLVER_CONVERGENCE_FAILED)
        
        super().__init__(message, **kwargs)


class ConvergenceError(SolverError):
    """Exception for solver convergence failures."""
    
    def __init__(
        self,
        message: str = "Solver failed to converge",
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        final_residual: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if max_iterations:
            context['max_iterations'] = max_iterations
        if tolerance:
            context['tolerance'] = tolerance
        if final_residual:
            context['final_residual'] = final_residual
        
        kwargs['context'] = context
        kwargs.setdefault('suggestion', 
                         "Try increasing max_iterations, relaxing tolerance, or improving initial guess")
        
        super().__init__(message, **kwargs)


class OptimizationError(DiffFEError):
    """Exception for optimization procedure errors."""
    
    def __init__(
        self,
        message: str,
        optimization_method: Optional[str] = None,
        objective_value: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if optimization_method:
            context['method'] = optimization_method
        if objective_value is not None:
            context['objective_value'] = objective_value
        if gradient_norm is not None:
            context['gradient_norm'] = gradient_norm
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', ErrorCode.OPTIMIZATION_FAILED)
        
        super().__init__(message, **kwargs)


class SecurityError(DiffFEError):
    """Exception for security violations and malicious input detection."""
    
    def __init__(
        self,
        message: str,
        attack_type: Optional[str] = None,
        suspicious_input: Optional[str] = None,
        source_ip: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if attack_type:
            context['attack_type'] = attack_type
        if suspicious_input:
            context['suspicious_input'] = suspicious_input[:100]  # Limit length
        if source_ip:
            context['source_ip'] = source_ip
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', ErrorCode.SECURITY_VIOLATION)
        
        super().__init__(message, **kwargs)


class APIError(DiffFEError):
    """Exception for API-related errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        endpoint: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['status_code'] = status_code
        if endpoint:
            context['endpoint'] = endpoint
        if request_data:
            # Sanitize sensitive data
            safe_data = {k: v for k, v in request_data.items() 
                        if not any(sensitive in k.lower() 
                                 for sensitive in ['password', 'token', 'key', 'secret'])}
            context['request_data'] = safe_data
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', ErrorCode.API_REQUEST_ERROR)
        
        super().__init__(message, **kwargs)


class ConfigurationError(DiffFEError):
    """Exception for configuration and environment setup errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_value: Optional[str] = None,
        actual_value: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if expected_value:
            context['expected_value'] = expected_value
        if actual_value:
            context['actual_value'] = actual_value
        if config_file:
            context['config_file'] = config_file
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', ErrorCode.CONFIGURATION_ERROR)
        
        super().__init__(message, **kwargs)


class ResourceError(DiffFEError):
    """Exception for resource availability and management errors."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        required_amount: Optional[Union[int, float]] = None,
        available_amount: Optional[Union[int, float]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if resource_type:
            context['resource_type'] = resource_type
        if required_amount is not None:
            context['required'] = required_amount
        if available_amount is not None:
            context['available'] = available_amount
        
        kwargs['context'] = context
        kwargs.setdefault('error_code', ErrorCode.RESOURCE_UNAVAILABLE)
        
        super().__init__(message, **kwargs)


# Exception handling utilities

def handle_exception(
    func: callable,
    exception_map: Optional[Dict[type, ErrorCode]] = None,
    context: Optional[Dict[str, Any]] = None,
    reraise_as: Optional[type] = None
):
    """Decorator for standardized exception handling.
    
    Parameters
    ----------
    func : callable
        Function to wrap
    exception_map : Dict[type, ErrorCode], optional
        Mapping of exception types to error codes
    context : Dict[str, Any], optional
        Additional context to include
    reraise_as : type, optional
        Exception type to reraise as
    
    Returns
    -------
    callable
        Wrapped function with exception handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DiffFEError:
            # Re-raise DiffFE errors as-is
            raise
        except Exception as e:
            # Map to DiffFE error
            error_code = ErrorCode.UNKNOWN_ERROR
            if exception_map and type(e) in exception_map:
                error_code = exception_map[type(e)]
            
            error_context = context or {}
            error_context['original_exception'] = str(e)
            error_context['function'] = func.__name__
            
            if reraise_as:
                raise reraise_as(
                    message=f"Error in {func.__name__}: {str(e)}",
                    error_code=error_code,
                    context=error_context,
                    cause=e
                )
            else:
                raise DiffFEError(
                    message=f"Error in {func.__name__}: {str(e)}",
                    error_code=error_code,
                    context=error_context,
                    cause=e
                )
    
    return wrapper


def create_error_response(error: Exception, include_traceback: bool = False) -> Dict[str, Any]:
    """Create standardized error response dictionary.
    
    Parameters
    ----------
    error : Exception
        Exception to convert
    include_traceback : bool, optional
        Whether to include traceback information
    
    Returns
    -------
    Dict[str, Any]
        Standardized error response
    """
    if isinstance(error, DiffFEError):
        response = error.to_dict()
    else:
        response = {
            'error_type': type(error).__name__,
            'error_code': ErrorCode.UNKNOWN_ERROR.name,
            'error_code_value': ErrorCode.UNKNOWN_ERROR.value,
            'message': str(error),
            'context': {},
            'suggestion': None,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
    
    if include_traceback:
        response['traceback'] = traceback.format_exc()
    
    return response


def validate_and_raise(
    condition: bool,
    message: str,
    error_class: type = ValidationError,
    **kwargs
):
    """Validate condition and raise error if false.
    
    Parameters
    ----------
    condition : bool
        Condition to validate
    message : str
        Error message if condition is false
    error_class : type, optional
        Exception class to raise
    **kwargs
        Additional arguments for exception
    """
    if not condition:
        raise error_class(message, **kwargs)


# Pre-configured exception handlers for common scenarios

def handle_import_error(package_name: str, install_command: str = None):
    """Handle missing package import errors.
    
    Parameters
    ----------
    package_name : str
        Name of the missing package
    install_command : str, optional
        Command to install the package
    
    Raises
    ------
    ConfigurationError
        If package is not available
    """
    suggestion = f"Install {package_name}"
    if install_command:
        suggestion += f" with: {install_command}"
    
    raise ConfigurationError(
        message=f"Required package '{package_name}' is not available",
        error_code=ErrorCode.DEPENDENCY_MISSING,
        context={'package': package_name},
        suggestion=suggestion
    )


def handle_mesh_error(mesh, issue: str):
    """Handle mesh validation errors.
    
    Parameters
    ----------
    mesh : Any
        Mesh object
    issue : str
        Description of the issue
    
    Raises
    ------
    MeshValidationError
        If mesh is invalid
    """
    mesh_info = {}
    try:
        if hasattr(mesh, 'num_cells'):
            mesh_info['num_cells'] = mesh.num_cells()
        if hasattr(mesh, 'num_vertices'):
            mesh_info['num_vertices'] = mesh.num_vertices()
        if hasattr(mesh, 'geometric_dimension'):
            mesh_info['dimension'] = mesh.geometric_dimension()
    except:
        pass
    
    raise MeshValidationError(
        message=f"Mesh validation failed: {issue}",
        mesh_info=mesh_info,
        suggestion="Verify mesh topology and geometry"
    )


def handle_solver_failure(
    solver_type: str,
    iterations: int,
    residual: float,
    tolerance: float
):
    """Handle solver convergence failures.
    
    Parameters
    ----------
    solver_type : str
        Type of solver
    iterations : int
        Number of iterations performed
    residual : float
        Final residual norm
    tolerance : float
        Required tolerance
    
    Raises
    ------
    ConvergenceError
        If solver failed to converge
    """
    raise ConvergenceError(
        message=f"{solver_type} solver failed to converge after {iterations} iterations",
        max_iterations=iterations,
        tolerance=tolerance,
        final_residual=residual,
        suggestion="Try adjusting solver parameters, improving initial guess, or checking problem setup"
    )