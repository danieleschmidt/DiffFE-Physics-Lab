"""Robust error handling and recovery mechanisms."""

import functools
import logging
import threading
import time
import traceback
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, Union

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""

    FAIL_FAST = "fail_fast"
    RETRY = "retry"
    FALLBACK = "fallback"
    IGNORE = "ignore"
    LOG_AND_CONTINUE = "log_and_continue"


class RobustError(Exception):
    """Base class for robust error handling."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.FAIL_FAST,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.original_error = original_error
        self.context = context or {}
        self.timestamp = time.time()

    def __str__(self):
        return f"[{self.severity.value.upper()}] {self.message}"


class ErrorRecoveryManager:
    """Manages error recovery strategies and fallbacks."""

    def __init__(self):
        self.error_count = 0
        self.error_history = []
        self.recovery_handlers = {}
        self.fallback_handlers = {}
        self.max_history = 1000
        self._lock = threading.Lock()

    def register_recovery_handler(
        self, error_type: Type[Exception], handler: Callable[[Exception], Any]
    ):
        """Register a recovery handler for specific error types."""
        self.recovery_handlers[error_type] = handler

    def register_fallback_handler(
        self, operation_name: str, handler: Callable[[], Any]
    ):
        """Register a fallback handler for operations."""
        self.fallback_handlers[operation_name] = handler

    def handle_error(
        self,
        error: Exception,
        operation_name: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Handle error according to registered strategies."""
        with self._lock:
            self.error_count += 1

            # Add to history
            error_record = {
                "timestamp": time.time(),
                "error_type": type(error).__name__,
                "message": str(error),
                "operation": operation_name,
                "context": context or {},
            }
            self.error_history.append(error_record)

            # Limit history size
            if len(self.error_history) > self.max_history:
                self.error_history = self.error_history[-self.max_history :]

        # Try registered recovery handlers
        error_type = type(error)
        if error_type in self.recovery_handlers:
            try:
                return self.recovery_handlers[error_type](error)
            except Exception as recovery_error:
                logger.warning(f"Recovery handler failed: {recovery_error}")

        # Try fallback handlers
        if operation_name in self.fallback_handlers:
            try:
                return self.fallback_handlers[operation_name]()
            except Exception as fallback_error:
                logger.warning(f"Fallback handler failed: {fallback_error}")

        # Default handling
        if isinstance(error, RobustError):
            if error.recovery_strategy == RecoveryStrategy.IGNORE:
                logger.debug(f"Ignoring error: {error}")
                return None
            elif error.recovery_strategy == RecoveryStrategy.LOG_AND_CONTINUE:
                logger.warning(f"Continuing after error: {error}")
                return None

        # Re-raise if no recovery possible
        raise error

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            stats = {
                "total_errors": self.error_count,
                "recent_errors": len(self.error_history),
                "error_types": {},
                "error_rate_per_hour": 0,
            }

            if self.error_history:
                # Count error types
                for record in self.error_history:
                    error_type = record["error_type"]
                    stats["error_types"][error_type] = (
                        stats["error_types"].get(error_type, 0) + 1
                    )

                # Calculate error rate
                now = time.time()
                recent_errors = [
                    r for r in self.error_history if now - r["timestamp"] < 3600
                ]
                stats["error_rate_per_hour"] = len(recent_errors)

            return stats


# Global error recovery manager
_global_recovery_manager = ErrorRecoveryManager()


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager."""
    return _global_recovery_manager


def robust_operation(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_value: Any = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    operation_name: str = "",
):
    """Decorator for robust error handling with retries and fallbacks."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            recovery_manager = get_error_recovery_manager()
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {operation_name or func.__name__}: {e}"
                        )
                        if retry_delay > 0:
                            time.sleep(
                                retry_delay * (2**attempt)
                            )  # Exponential backoff
                        continue
                    else:
                        # Final attempt failed
                        break

            # All retries failed
            robust_error = RobustError(
                f"Operation failed after {max_retries + 1} attempts: {last_error}",
                severity=severity,
                recovery_strategy=(
                    RecoveryStrategy.FALLBACK
                    if fallback_value is not None
                    else RecoveryStrategy.FAIL_FAST
                ),
                original_error=last_error,
                context={"function": func.__name__, "attempts": max_retries + 1},
            )

            try:
                return recovery_manager.handle_error(
                    robust_error,
                    operation_name or func.__name__,
                    {"args": str(args)[:100], "kwargs": str(kwargs)[:100]},
                )
            except:
                # Last resort - return fallback value
                if fallback_value is not None:
                    logger.error(
                        f"Using fallback value for {operation_name or func.__name__}"
                    )
                    return fallback_value
                raise

        return wrapper

    return decorator


@contextmanager
def error_boundary(
    operation_name: str,
    fallback_value: Any = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
):
    """Context manager for error boundaries."""
    try:
        yield
    except Exception as e:
        robust_error = RobustError(
            f"Error in {operation_name}: {e}",
            severity=severity,
            original_error=e,
            context={"operation": operation_name},
        )

        recovery_manager = get_error_recovery_manager()
        try:
            result = recovery_manager.handle_error(robust_error, operation_name)
            if result is not None:
                return result
        except:
            pass

        if fallback_value is not None:
            logger.error(f"Using fallback value for {operation_name}")
            return fallback_value
        else:
            logger.error(f"Unhandled error in {operation_name}: {e}")
            raise


def log_and_suppress(error_types: tuple = (Exception,)):
    """Decorator to log errors and suppress them."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                logger.exception(f"Suppressed error in {func.__name__}: {e}")
                return None

        return wrapper

    return decorator


def validate_and_sanitize_input(
    input_validators: Optional[Dict[str, Callable]] = None,
    output_validator: Optional[Callable] = None,
):
    """Decorator for input/output validation and sanitization."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate inputs
            if input_validators:
                for param_name, validator in input_validators.items():
                    if param_name in kwargs:
                        try:
                            kwargs[param_name] = validator(kwargs[param_name])
                        except Exception as e:
                            raise RobustError(
                                f"Input validation failed for {param_name}: {e}",
                                severity=ErrorSeverity.HIGH,
                                original_error=e,
                            )

            # Execute function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                raise RobustError(
                    f"Function execution failed: {e}",
                    severity=ErrorSeverity.MEDIUM,
                    original_error=e,
                    context={"function": func.__name__},
                )

            # Validate output
            if output_validator:
                try:
                    result = output_validator(result)
                except Exception as e:
                    raise RobustError(
                        f"Output validation failed: {e}",
                        severity=ErrorSeverity.HIGH,
                        original_error=e,
                    )

            return result

        return wrapper

    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception,
):
    """Circuit breaker pattern implementation."""

    def decorator(func: Callable) -> Callable:
        failure_count = 0
        last_failure_time = 0
        state = "closed"  # closed, open, half_open

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal failure_count, last_failure_time, state

            current_time = time.time()

            # Check if circuit should be half-open
            if state == "open" and current_time - last_failure_time > timeout:
                state = "half_open"
                logger.info(f"Circuit breaker half-open for {func.__name__}")

            # Reject calls if circuit is open
            if state == "open":
                raise RobustError(
                    f"Circuit breaker open for {func.__name__}",
                    severity=ErrorSeverity.HIGH,
                    recovery_strategy=RecoveryStrategy.FALLBACK,
                )

            try:
                result = func(*args, **kwargs)

                # Success - reset failure count
                if state == "half_open":
                    state = "closed"
                    failure_count = 0
                    logger.info(f"Circuit breaker closed for {func.__name__}")

                return result

            except expected_exception as e:
                failure_count += 1
                last_failure_time = current_time

                if failure_count >= failure_threshold:
                    state = "open"
                    logger.warning(
                        f"Circuit breaker opened for {func.__name__} after {failure_count} failures"
                    )

                raise RobustError(
                    f"Circuit breaker failure: {e}",
                    severity=ErrorSeverity.MEDIUM,
                    original_error=e,
                    context={"failures": failure_count, "state": state},
                )

        return wrapper

    return decorator


def setup_robust_logging():
    """Setup robust logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("diffhe_errors.log")],
    )

    # Add error recovery manager handler
    recovery_manager = get_error_recovery_manager()

    def log_error_stats():
        """Periodically log error statistics."""
        stats = recovery_manager.get_error_statistics()
        if stats["total_errors"] > 0:
            logger.info(f"Error statistics: {stats}")

    # Schedule periodic stats logging (simplified)
    import threading

    def stats_thread():
        while True:
            time.sleep(300)  # Log every 5 minutes
            log_error_stats()

    thread = threading.Thread(target=stats_thread, daemon=True)
    thread.start()


# Circuit Breaker class for backward compatibility
class CircuitBreaker:
    """Circuit breaker implementation as a class."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Call a function through the circuit breaker."""
        with self._lock:
            current_time = time.time()
            
            # Check if circuit should be half-open
            if self.state == "open" and current_time - self.last_failure_time > self.timeout:
                self.state = "half_open"
                logger.info(f"Circuit breaker half-open for {func.__name__}")
            
            # Reject calls if circuit is open
            if self.state == "open":
                raise RobustError(
                    f"Circuit breaker open for {func.__name__}",
                    severity=ErrorSeverity.HIGH,
                    recovery_strategy=RecoveryStrategy.FALLBACK,
                )
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker closed for {func.__name__}")
                
                return result
                
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = current_time
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(
                        f"Circuit breaker opened for {func.__name__} after {self.failure_count} failures"
                    )
                
                raise RobustError(
                    f"Circuit breaker failure: {e}",
                    severity=ErrorSeverity.MEDIUM,
                    original_error=e,
                    context={"failures": self.failure_count, "state": self.state},
                )


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """Retry decorator with exponential backoff."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
                        raise RobustError(
                            f"Function failed after {max_retries + 1} attempts: {e}",
                            severity=ErrorSeverity.HIGH,
                            original_error=e,
                            context={
                                "function": func.__name__,
                                "attempts": max_retries + 1,
                                "max_delay": delay,
                            },
                        )
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    
    return decorator


# Initialize robust logging
setup_robust_logging()
