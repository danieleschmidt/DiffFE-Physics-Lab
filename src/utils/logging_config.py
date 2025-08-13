"""Comprehensive logging configuration with structured output and monitoring."""

import json
import logging
import logging.config
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def __init__(self, include_extra_fields: bool = True):
        super().__init__()
        self.include_extra_fields = include_extra_fields

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": record.thread,
            "process_id": record.process,
        }

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields if present
        if self.include_extra_fields:
            extra_fields = {
                key: value
                for key, value in record.__dict__.items()
                if key
                not in {
                    "name",
                    "msg",
                    "args",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "message",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "getMessage",
                }
            }
            if extra_fields:
                log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str, ensure_ascii=False)


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log messages."""

    def __init__(self, min_duration_ms: float = 100.0):
        super().__init__()
        self.min_duration_ms = min_duration_ms

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter based on performance metrics."""
        # Check if this is a performance-related message
        if hasattr(record, "duration_ms"):
            return record.duration_ms >= self.min_duration_ms

        # Allow all other messages
        return True


class SecurityFilter(logging.Filter):
    """Filter for security-related log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and sanitize security-sensitive information."""
        # List of sensitive patterns to redact
        sensitive_patterns = [
            "password",
            "token",
            "secret",
            "key",
            "auth",
            "credential",
            "session",
            "cookie",
        ]

        message = record.getMessage()

        # Redact sensitive information
        for pattern in sensitive_patterns:
            if pattern in message.lower():
                # Replace with asterisks but keep structure
                import re

                message = re.sub(
                    f"{pattern}[\\s]*[:=][\\s]*[^\\s]+",
                    f"{pattern}=***",
                    message,
                    flags=re.IGNORECASE,
                )

        # Update the record
        record.args = ()  # Clear args to prevent formatting issues
        record.msg = message

        return True


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler to prevent blocking."""

    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self._stop_event = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record asynchronously."""
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop oldest record if queue is full
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(record)
            except queue.Empty:
                pass

    def _worker(self):
        """Background worker to process log records."""
        while not self._stop_event.is_set():
            try:
                record = self.log_queue.get(timeout=1.0)
                self.target_handler.emit(record)
            except queue.Empty:
                continue
            except Exception as e:
                # Handle errors in logging gracefully
                print(f"Error in async log handler: {e}")

    def close(self):
        """Close the handler and stop worker thread."""
        self._stop_event.set()
        self.worker_thread.join(timeout=2.0)
        self.target_handler.close()
        super().close()


class MetricsCollector:
    """Collector for application metrics embedded in logs."""

    def __init__(self):
        self.metrics = {}
        self._lock = threading.Lock()

    def record_metric(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric value."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []

            metric_entry = {
                "timestamp": datetime.now().isoformat(),
                "value": value,
                "tags": tags or {},
            }

            self.metrics[name].append(metric_entry)

            # Keep only recent entries (last 1000)
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]

    def get_metrics(self, name: str = None) -> Dict[str, Any]:
        """Get recorded metrics."""
        with self._lock:
            if name:
                return self.metrics.get(name, [])
            return self.metrics.copy()

    def clear_metrics(self, name: str = None):
        """Clear recorded metrics."""
        with self._lock:
            if name:
                self.metrics.pop(name, None)
            else:
                self.metrics.clear()


# Global metrics collector
_global_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    return _global_metrics


class PerformanceLogger:
    """Context manager for performance logging."""

    def __init__(
        self, operation_name: str, logger: logging.Logger, min_log_duration: float = 0.0
    ):
        self.operation_name = operation_name
        self.logger = logger
        self.min_log_duration = min_log_duration
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = (self.end_time - self.start_time) * 1000  # Convert to ms

        if duration >= self.min_log_duration:
            level = logging.INFO if duration > 1000 else logging.DEBUG
            self.logger.log(
                level,
                f"Completed operation: {self.operation_name}",
                extra={
                    "operation": self.operation_name,
                    "duration_ms": duration,
                    "success": exc_type is None,
                },
            )

        # Record metric
        _global_metrics.record_metric(
            f"operation_duration_{self.operation_name}",
            duration,
            {"success": str(exc_type is None)},
        )

    def add_context(self, **kwargs):
        """Add context information to be logged."""
        for key, value in kwargs.items():
            setattr(self, key, value)


def setup_logging(
    config_path: Optional[str] = None,
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_structured_logging: bool = True,
    enable_async_logging: bool = True,
    enable_performance_logging: bool = True,
) -> None:
    """Setup comprehensive logging configuration.

    Parameters
    ----------
    config_path : str, optional
        Path to logging configuration file
    log_level : str, optional
        Default log level
    log_dir : str, optional
        Directory for log files
    enable_structured_logging : bool, optional
        Enable JSON structured logging
    enable_async_logging : bool, optional
        Enable asynchronous logging
    enable_performance_logging : bool, optional
        Enable performance logging filter
    """
    # Determine log directory
    if log_dir is None:
        log_dir = os.getenv("DIFFHE_LOG_DIR", "./logs")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Load configuration from file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            if config_path.endswith(".json"):
                config = json.load(f)
            else:
                # YAML configuration
                try:
                    import yaml

                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError(
                        "PyYAML required for YAML config: pip install pyyaml"
                    )

        logging.config.dictConfig(config)
        return

    # Create default configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "filters": {"security": {"()": SecurityFilter}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "filters": ["security"],
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filters": ["security"],
                "filename": str(log_path / "diffhe.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(log_path / "diffhe_errors.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "src": {
                "level": "DEBUG",
                "handlers": ["console", "file", "error_file"],
                "propagate": False,
            },
            "diffhe": {
                "level": "DEBUG",
                "handlers": ["console", "file", "error_file"],
                "propagate": False,
            },
        },
        "root": {"level": log_level, "handlers": ["console", "file"]},
    }

    # Add structured logging if enabled
    if enable_structured_logging:
        config["formatters"]["structured"] = {"()": StructuredFormatter}

        # Add structured log file
        config["handlers"]["structured_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "structured",
            "filters": ["security"],
            "filename": str(log_path / "diffhe_structured.log"),
            "maxBytes": 50 * 1024 * 1024,  # 50MB for JSON logs
            "backupCount": 3,
        }

        # Add to loggers
        for logger_config in config["loggers"].values():
            logger_config["handlers"].append("structured_file")

    # Add performance logging if enabled
    if enable_performance_logging:
        config["filters"]["performance"] = {
            "()": PerformanceFilter,
            "min_duration_ms": 10.0,
        }

        config["handlers"]["performance_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "structured" if enable_structured_logging else "detailed",
            "filters": ["security", "performance"],
            "filename": str(log_path / "diffhe_performance.log"),
            "maxBytes": 20 * 1024 * 1024,  # 20MB
            "backupCount": 3,
        }

    # Apply configuration
    logging.config.dictConfig(config)

    # Wrap handlers with async handlers if enabled
    if enable_async_logging:
        _wrap_handlers_async()

    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, Directory: {log_dir}")


def _wrap_handlers_async():
    """Wrap existing handlers with async handlers."""
    root_logger = logging.getLogger()

    for handler in root_logger.handlers.copy():
        if not isinstance(handler, AsyncLogHandler):
            async_handler = AsyncLogHandler(handler)
            async_handler.setLevel(handler.level)
            async_handler.setFormatter(handler.formatter)

            # Replace handler
            root_logger.removeHandler(handler)
            root_logger.addHandler(async_handler)


def get_logger(name: str) -> logging.Logger:
    """Get logger with standard configuration.

    Parameters
    ----------
    name : str
        Logger name

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Ensure logger has minimum required handlers if not configured
    if not logger.handlers and not logger.parent.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def log_performance(operation_name: str, min_duration: float = 0.0):
    """Decorator for performance logging.

    Parameters
    ----------
    operation_name : str
        Name of the operation
    min_duration : float, optional
        Minimum duration to log (seconds)

    Examples
    --------
    >>> @log_performance("matrix_multiply", min_duration=0.1)
    ... def matrix_mult(a, b):
    ...     return a @ b
    """

    def decorator(func):
        logger = get_logger(func.__module__)

        def wrapper(*args, **kwargs):
            with PerformanceLogger(operation_name, logger, min_duration * 1000):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def log_function_calls(logger_name: Optional[str] = None, include_args: bool = False):
    """Decorator for function call logging.

    Parameters
    ----------
    logger_name : str, optional
        Logger name to use
    include_args : bool, optional
        Whether to include function arguments
    """

    def decorator(func):
        logger = get_logger(logger_name or func.__module__)

        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"

            if include_args:
                args_str = ", ".join([repr(arg) for arg in args])
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                logger.debug(f"Calling {func_name}({all_args})")
            else:
                logger.debug(f"Calling {func_name}")

            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {func_name}")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


def create_audit_logger(name: str = "audit") -> logging.Logger:
    """Create audit logger for security and compliance.

    Parameters
    ----------
    name : str, optional
        Audit logger name

    Returns
    -------
    logging.Logger
        Audit logger instance
    """
    audit_logger = logging.getLogger(f"audit.{name}")

    # Create audit-specific handler if not exists
    if not any(isinstance(h, logging.FileHandler) for h in audit_logger.handlers):
        log_dir = os.getenv("DIFFHE_LOG_DIR", "./logs")
        audit_path = Path(log_dir) / "audit"
        audit_path.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            audit_path / f"{name}.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10,
        )
        handler.setFormatter(StructuredFormatter())
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False

    return audit_logger


# Context manager for adding context to all logs in a scope
class LogContext:
    """Context manager for adding context to all logs within a scope."""

    def __init__(self, **context):
        self.context = context
        self.old_factory = None

    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)
