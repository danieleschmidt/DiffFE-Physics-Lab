"""Robust components for DiffFE-Physics-Lab."""

from .error_handling import (
    DiffFEError, ValidationError, ConvergenceError, BackendError,
    robust_execute, retry_with_backoff
)
from .monitoring import (
    PerformanceMonitor, MetricsCollector, HealthChecker,
    resource_monitor
)
from .security import (
    SecurityValidator, InputSanitizer, PermissionChecker,
    secure_mode
)
from .logging_system import (
    configure_logging, get_logger, AuditLogger,
    log_performance
)

__all__ = [
    # Error handling
    "DiffFEError", "ValidationError", "ConvergenceError", "BackendError",
    "robust_execute", "retry_with_backoff",
    
    # Monitoring
    "PerformanceMonitor", "MetricsCollector", "HealthChecker",
    "resource_monitor",
    
    # Security
    "SecurityValidator", "InputSanitizer", "PermissionChecker",
    "secure_mode",
    
    # Logging
    "configure_logging", "get_logger", "AuditLogger",
    "log_performance",
]