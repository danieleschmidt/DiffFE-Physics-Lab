"""Advanced logging system for DiffFE-Physics-Lab."""

import logging
import logging.handlers
import json
import time
import threading
import re
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import wraps

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: float
    level: str
    logger: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    extra_data: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = None


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted string
        """
        # Extract basic information
        log_entry = LogEntry(
            timestamp=record.created,
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            module=getattr(record, 'module', None),
            function=getattr(record, 'funcName', None),
            line_number=getattr(record, 'lineno', None),
            extra_data=getattr(record, 'extra_data', None),
            performance_data=getattr(record, 'performance_data', None)
        )
        
        # Convert to dictionary and then JSON
        log_dict = asdict(log_entry)
        
        # Add exception information if present
        if record.exc_info:
            log_dict['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_dict, default=str)


class PerformanceLogFormatter(logging.Formatter):
    """Special formatter for performance logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format performance log record.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted string
        """
        perf_data = getattr(record, 'performance_data', {})
        
        if perf_data:
            duration = perf_data.get('duration', 0)
            memory_mb = perf_data.get('memory_delta_mb', 0)
            operation = perf_data.get('operation', 'unknown')
            
            return (f"{self.formatTime(record)} - PERF - {operation}: "
                   f"duration={duration:.3f}s, memory={memory_mb:.1f}MB - {record.getMessage()}")
        else:
            return super().format(record)


class AuditLogger:
    """Audit logging for security and compliance."""
    
    def __init__(self, log_file: str = "audit.log"):
        """Initialize audit logger.
        
        Args:
            log_file: Audit log file path
        """
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler for audit logs
        audit_file = LOGS_DIR / log_file
        handler = logging.handlers.RotatingFileHandler(
            audit_file, maxBytes=10*1024*1024, backupCount=5
        )
        
        # Use structured formatter for audit logs
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def log_access(self, user_id: str, resource: str, action: str, 
                   success: bool, **extra):
        """Log access attempt.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            success: Whether access was successful
            **extra: Additional context
        """
        self.logger.info(
            f"Access {action} on {resource}",
            extra={
                'extra_data': {
                    'event_type': 'access',
                    'user_id': user_id,
                    'resource': resource,
                    'action': action,
                    'success': success,
                    **extra
                }
            }
        )
    
    def log_security_event(self, event_type: str, severity: str, 
                          description: str, **extra):
        """Log security event.
        
        Args:
            event_type: Type of security event
            severity: Event severity (low, medium, high, critical)
            description: Event description
            **extra: Additional context
        """
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        self.logger.log(
            log_level,
            f"Security event: {description}",
            extra={
                'extra_data': {
                    'event_type': 'security',
                    'security_event_type': event_type,
                    'severity': severity,
                    'description': description,
                    **extra
                }
            }
        )
    
    def log_data_operation(self, operation: str, data_type: str, 
                          record_count: Optional[int] = None, **extra):
        """Log data operation.
        
        Args:
            operation: Type of operation (create, read, update, delete)
            data_type: Type of data being operated on
            record_count: Number of records affected
            **extra: Additional context
        """
        self.logger.info(
            f"Data operation: {operation} {data_type}",
            extra={
                'extra_data': {
                    'event_type': 'data_operation',
                    'operation': operation,
                    'data_type': data_type,
                    'record_count': record_count,
                    **extra
                }
            }
        )


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, log_file: str = "performance.log"):
        """Initialize performance logger.
        
        Args:
            log_file: Performance log file path
        """
        self.logger = logging.getLogger("performance")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler for performance logs
        perf_file = LOGS_DIR / log_file
        handler = logging.handlers.RotatingFileHandler(
            perf_file, maxBytes=50*1024*1024, backupCount=3
        )
        
        # Use performance formatter
        handler.setFormatter(PerformanceLogFormatter())
        self.logger.addHandler(handler)
        
        self.logger.propagate = False
    
    def log_operation(self, operation: str, duration: float, 
                     memory_delta_mb: float = 0.0, **metrics):
        """Log performance metrics for an operation.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            memory_delta_mb: Memory change in MB
            **metrics: Additional performance metrics
        """
        perf_data = {
            'operation': operation,
            'duration': duration,
            'memory_delta_mb': memory_delta_mb,
            **metrics
        }
        
        self.logger.info(
            f"Operation completed: {operation}",
            extra={'performance_data': perf_data}
        )


class LogAnalyzer:
    """Analyze log files for patterns and issues."""
    
    def __init__(self, log_file: Union[str, Path]):
        """Initialize log analyzer.
        
        Args:
            log_file: Log file to analyze
        """
        self.log_file = Path(log_file)
        self.patterns = {
            'errors': [r'ERROR', r'CRITICAL', r'Exception', r'Failed'],
            'warnings': [r'WARNING', r'WARN'],
            'security': [r'Security', r'Access denied', r'Authentication'],
            'performance': [r'slow', r'timeout', r'memory'],
        }
    
    def analyze_log_file(self, max_lines: Optional[int] = None) -> Dict[str, Any]:
        """Analyze log file for patterns.
        
        Args:
            max_lines: Maximum number of lines to analyze
            
        Returns:
            Analysis results
        """
        if not self.log_file.exists():
            return {"error": f"Log file not found: {self.log_file}"}
        
        results = {
            'file': str(self.log_file),
            'analysis_time': time.time(),
            'total_lines': 0,
            'pattern_counts': {pattern: 0 for pattern in self.patterns},
            'issues': [],
            'recommendations': []
        }
        
        try:
            with open(self.log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if max_lines and line_num > max_lines:
                        break
                    
                    results['total_lines'] = line_num
                    
                    # Check for patterns
                    for pattern_name, pattern_list in self.patterns.items():
                        for pattern in pattern_list:
                            if re.search(pattern, line, re.IGNORECASE):
                                results['pattern_counts'][pattern_name] += 1
                                
                                # Collect significant issues
                                if pattern_name == 'errors' and len(results['issues']) < 10:
                                    results['issues'].append({
                                        'line_number': line_num,
                                        'type': pattern_name,
                                        'content': line.strip()[:200]  # Truncate long lines
                                    })
        
        except Exception as e:
            results['error'] = f"Error analyzing file: {e}"
        
        # Generate recommendations
        self._generate_recommendations(results)
        
        return results
    
    def _generate_recommendations(self, results: Dict[str, Any]):
        """Generate recommendations based on analysis.
        
        Args:
            results: Analysis results to update
        """
        recommendations = []
        pattern_counts = results['pattern_counts']
        
        # Check error rates
        error_rate = pattern_counts['errors'] / max(results['total_lines'], 1) * 100
        if error_rate > 5:
            recommendations.append(f"High error rate: {error_rate:.1f}% - investigate error causes")
        
        # Check warning rates
        warning_rate = pattern_counts['warnings'] / max(results['total_lines'], 1) * 100
        if warning_rate > 10:
            recommendations.append(f"High warning rate: {warning_rate:.1f}% - review warning conditions")
        
        # Check security events
        if pattern_counts['security'] > 0:
            recommendations.append(f"Security events detected: {pattern_counts['security']} - review security logs")
        
        # Check performance issues
        if pattern_counts['performance'] > 0:
            recommendations.append(f"Performance issues detected: {pattern_counts['performance']} - review performance")
        
        results['recommendations'] = recommendations


def configure_logging(level: str = "INFO", 
                     enable_file_logging: bool = True,
                     enable_structured_logging: bool = True,
                     log_file: str = "diffhe.log"):
    """Configure logging system.
    
    Args:
        level: Logging level
        enable_file_logging: Whether to log to files
        enable_structured_logging: Whether to use structured JSON logging
        log_file: Main log file name
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    
    if enable_structured_logging:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    root_logger.addHandler(console_handler)
    
    # File handler
    if enable_file_logging:
        log_path = LOGS_DIR / log_file
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=20*1024*1024, backupCount=5
        )
        file_handler.setLevel(numeric_level)
        
        if enable_structured_logging:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        
        root_logger.addHandler(file_handler)
    
    logging.info(f"Logging configured: level={level}, file_logging={enable_file_logging}, "
                f"structured={enable_structured_logging}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_performance(operation_name: str):
    """Decorator to log performance of functions.
    
    Args:
        operation_name: Name of the operation to log
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful operation
                logger = get_logger(f"{func.__module__}.{func.__name__}")
                logger.info(
                    f"Operation completed: {operation_name}",
                    extra={
                        'performance_data': {
                            'operation': operation_name,
                            'function': func.__name__,
                            'duration': duration,
                            'success': True
                        }
                    }
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log failed operation
                logger = get_logger(f"{func.__module__}.{func.__name__}")
                logger.error(
                    f"Operation failed: {operation_name}",
                    extra={
                        'performance_data': {
                            'operation': operation_name,
                            'function': func.__name__,
                            'duration': duration,
                            'success': False,
                            'error': str(e)
                        }
                    }
                )
                
                raise
        
        return wrapper
    return decorator


# Global instances
global_audit_logger = AuditLogger()
global_performance_logger = PerformanceLogger()

# Configure default logging
configure_logging()