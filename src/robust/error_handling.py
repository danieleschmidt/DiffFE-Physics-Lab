"""Comprehensive error handling for DiffFE-Physics-Lab."""

import logging
import time
import functools
from typing import Any, Callable, Dict, Optional, Type, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DiffFEError(Exception):
    """Base exception for DiffFE-Physics-Lab."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """Initialize DiffFE error.
        
        Args:
            message: Error message
            error_code: Optional error code
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "DIFFHE_UNKNOWN"
        self.context = context or {}
        self.timestamp = time.time()
        
        # Log error when created
        logger.error(f"DiffFE Error ({self.error_code}): {message}")
        if context:
            logger.debug(f"Error context: {context}")


class ValidationError(DiffFEError):
    """Error in input validation or problem setup."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Any = None, **kwargs):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Field name that failed validation
            value: Invalid value
            **kwargs: Additional context
        """
        context = kwargs
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)
        
        super().__init__(message, "DIFFHE_VALIDATION", context)
        self.field = field
        self.value = value


class ConvergenceError(DiffFEError):
    """Error in numerical convergence."""
    
    def __init__(self, message: str, iterations: Optional[int] = None,
                 residual: Optional[float] = None, **kwargs):
        """Initialize convergence error.
        
        Args:
            message: Error message
            iterations: Number of iterations reached
            residual: Final residual value
            **kwargs: Additional context
        """
        context = kwargs
        if iterations is not None:
            context["iterations"] = iterations
        if residual is not None:
            context["residual"] = residual
        
        super().__init__(message, "DIFFHE_CONVERGENCE", context)
        self.iterations = iterations
        self.residual = residual


class BackendError(DiffFEError):
    """Error in computational backend."""
    
    def __init__(self, message: str, backend: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        """Initialize backend error.
        
        Args:
            message: Error message
            backend: Backend name
            operation: Operation that failed
            **kwargs: Additional context
        """
        context = kwargs
        if backend:
            context["backend"] = backend
        if operation:
            context["operation"] = operation
        
        super().__init__(message, "DIFFHE_BACKEND", context)
        self.backend = backend
        self.operation = operation


class MemoryError(DiffFEError):
    """Memory-related errors."""
    
    def __init__(self, message: str, memory_used: Optional[int] = None,
                 memory_limit: Optional[int] = None, **kwargs):
        """Initialize memory error.
        
        Args:
            message: Error message
            memory_used: Memory used in bytes
            memory_limit: Memory limit in bytes
            **kwargs: Additional context
        """
        context = kwargs
        if memory_used is not None:
            context["memory_used_mb"] = memory_used // (1024 * 1024)
        if memory_limit is not None:
            context["memory_limit_mb"] = memory_limit // (1024 * 1024)
        
        super().__init__(message, "DIFFHE_MEMORY", context)
        self.memory_used = memory_used
        self.memory_limit = memory_limit


def robust_execute(func: Callable, *args, max_retries: int = 3, 
                  backoff_factor: float = 2.0, 
                  expected_exceptions: tuple = (Exception,),
                  **kwargs) -> Any:
    """Execute function with robust error handling and retries.
    
    Args:
        func: Function to execute
        *args: Function arguments
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor
        expected_exceptions: Exceptions to catch and retry
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Executing {func.__name__}, attempt {attempt + 1}")
            result = func(*args, **kwargs)
            
            if attempt > 0:
                logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
            
            return result
            
        except expected_exceptions as e:
            last_exception = e
            
            if attempt < max_retries:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}, "
                             f"retrying in {wait_time:.2f} seconds: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")
                break
    
    # Re-raise the last exception
    raise last_exception


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2.0,
                      expected_exceptions: tuple = (Exception,)):
    """Decorator for adding retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor
        expected_exceptions: Exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return robust_execute(
                func, *args, 
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                expected_exceptions=expected_exceptions,
                **kwargs
            )
        return wrapper
    return decorator


@contextmanager
def error_context(operation: str, **context):
    """Context manager for adding context to errors.
    
    Args:
        operation: Operation being performed
        **context: Additional context information
    """
    try:
        logger.debug(f"Starting operation: {operation}")
        yield
        logger.debug(f"Completed operation: {operation}")
        
    except DiffFEError:
        # Re-raise DiffFE errors as-is
        logger.error(f"DiffFE error in operation: {operation}")
        raise
        
    except Exception as e:
        # Wrap other exceptions in DiffFEError
        error_message = f"Error in {operation}: {str(e)}"
        logger.error(error_message)
        
        # Add operation context
        full_context = {"operation": operation}
        full_context.update(context)
        
        raise DiffFEError(error_message, "DIFFHE_OPERATION", full_context) from e


class ErrorRecovery:
    """Error recovery and fallback mechanisms."""
    
    def __init__(self):
        """Initialize error recovery system."""
        self.fallback_strategies = {}
        self.recovery_stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "fallback_usage": {}
        }
    
    def register_fallback(self, error_type: Type[Exception], 
                         fallback_func: Callable):
        """Register fallback strategy for error type.
        
        Args:
            error_type: Exception type to handle
            fallback_func: Fallback function to execute
        """
        self.fallback_strategies[error_type] = fallback_func
        logger.info(f"Registered fallback for {error_type.__name__}")
    
    def execute_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with error recovery.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        try:
            return func(*args, **kwargs)
            
        except Exception as e:
            self.recovery_stats["total_errors"] += 1
            
            # Look for applicable fallback strategy
            for error_type, fallback_func in self.fallback_strategies.items():
                if isinstance(e, error_type):
                    logger.warning(f"Error {type(e).__name__} occurred, using fallback")
                    
                    try:
                        result = fallback_func(*args, **kwargs)
                        self.recovery_stats["recovered_errors"] += 1
                        
                        fallback_name = fallback_func.__name__
                        self.recovery_stats["fallback_usage"][fallback_name] = \
                            self.recovery_stats["fallback_usage"].get(fallback_name, 0) + 1
                        
                        logger.info(f"Successfully recovered using {fallback_name}")
                        return result
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback strategy failed: {fallback_error}")
                        break
            
            # No recovery possible, re-raise original exception
            logger.error(f"No recovery strategy available for {type(e).__name__}")
            raise
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get error recovery statistics."""
        total = self.recovery_stats["total_errors"]
        recovered = self.recovery_stats["recovered_errors"]
        recovery_rate = (recovered / total * 100) if total > 0 else 0
        
        return {
            "total_errors": total,
            "recovered_errors": recovered,
            "recovery_rate_percent": recovery_rate,
            "fallback_usage": self.recovery_stats["fallback_usage"].copy()
        }


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        logger.info(f"Circuit breaker initialized with threshold {failure_threshold}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            DiffFEError: If circuit is open
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise DiffFEError("Circuit breaker is OPEN", "DIFFHE_CIRCUIT_OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker reset to CLOSED state")
        
        self.failure_count = 0
        self.last_failure_time = None
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }


# Global error recovery instance
global_error_recovery = ErrorRecovery()


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with default value for zero division."""
    try:
        return a / b
    except ZeroDivisionError:
        logger.warning(f"Division by zero: {a} / {b}, returning default {default}")
        return default


def safe_sqrt(x: float, default: float = 0.0) -> float:
    """Safe square root with default for negative values."""
    if x < 0:
        logger.warning(f"Square root of negative number: {x}, returning default {default}")
        return default
    return x ** 0.5


def validate_positive(value: float, name: str = "value") -> float:
    """Validate that value is positive.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive", field=name, value=value)
    return value


def validate_range(value: float, min_val: float, max_val: float, 
                  name: str = "value") -> float:
    """Validate that value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Parameter name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value is outside range
    """
    if not min_val <= value <= max_val:
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}",
            field=name, value=value, min_val=min_val, max_val=max_val
        )
    return value