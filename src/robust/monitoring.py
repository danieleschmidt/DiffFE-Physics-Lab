"""Performance monitoring and health checking for DiffFE-Physics-Lab."""

import time
import logging
import threading
import os

# Mock psutil if not available
try:
    import psutil
except ImportError:
    # Mock psutil for demonstration purposes
    class MockProcess:
        def memory_info(self):
            return type('MemInfo', (), {'rss': 50 * 1024 * 1024})()  # 50MB
        def cpu_percent(self):
            return 5.0
    
    class MockVirtualMemory:
        percent = 45.0
        available = 8 * 1024 * 1024 * 1024  # 8GB
    
    class MockDiskUsage:
        percent = 60.0
    
    class MockPsutil:
        @staticmethod
        def Process():
            return MockProcess()
        
        @staticmethod
        def cpu_percent(interval=None):
            return 15.0
        
        @staticmethod
        def virtual_memory():
            return MockVirtualMemory()
        
        @staticmethod
        def disk_usage(path):
            return MockDiskUsage()
        
        class NoSuchProcess(Exception):
            pass
    
    psutil = MockPsutil()
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict, deque
import functools

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_before: int
    memory_after: int
    memory_peak: int
    cpu_percent: float
    success: bool
    error_type: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta(self) -> int:
        """Memory change during operation."""
        return self.memory_after - self.memory_before


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to store
        """
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.operation_stats = defaultdict(list)
        self.active_operations = {}
        self._lock = threading.Lock()
        
        logger.info(f"Performance monitor initialized with max_history={max_history}")
    
    @contextmanager
    def measure_operation(self, operation_name: str, **metadata):
        """Context manager for measuring operation performance.
        
        Args:
            operation_name: Name of the operation
            **metadata: Additional metadata to store
        """
        start_time = time.time()
        process = psutil.Process()
        
        # Get initial state
        memory_before = process.memory_info().rss
        memory_peak = memory_before
        cpu_percent_start = process.cpu_percent()
        
        # Store active operation info
        operation_id = f"{operation_name}_{start_time}"
        
        with self._lock:
            self.active_operations[operation_id] = {
                "name": operation_name,
                "start_time": start_time,
                "memory_before": memory_before,
                "metadata": metadata
            }
        
        success = True
        error_type = None
        
        try:
            # Monitor memory usage periodically in background
            monitor_thread = threading.Thread(
                target=self._monitor_memory_usage,
                args=(operation_id, process),
                daemon=True
            )
            monitor_thread.start()
            
            yield
            
        except Exception as e:
            success = False
            error_type = type(e).__name__
            logger.warning(f"Operation {operation_name} failed: {e}")
            raise
            
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Get final state
            memory_after = process.memory_info().rss
            cpu_percent_end = process.cpu_percent()
            
            # Get peak memory from monitoring
            with self._lock:
                operation_info = self.active_operations.pop(operation_id, {})
                memory_peak = operation_info.get("memory_peak", memory_after)
            
            # Create metrics record
            metrics = PerformanceMetrics(
                operation=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                cpu_percent=(cpu_percent_start + cpu_percent_end) / 2,
                success=success,
                error_type=error_type,
                additional_data=metadata
            )
            
            # Store metrics
            with self._lock:
                self.metrics_history.append(metrics)
                self.operation_stats[operation_name].append(metrics)
            
            # Log performance info
            self._log_performance_metrics(metrics)
    
    def _monitor_memory_usage(self, operation_id: str, process: psutil.Process):
        """Monitor memory usage during operation."""
        while operation_id in self.active_operations:
            try:
                current_memory = process.memory_info().rss
                
                with self._lock:
                    if operation_id in self.active_operations:
                        current_peak = self.active_operations[operation_id].get("memory_peak", 0)
                        self.active_operations[operation_id]["memory_peak"] = max(current_peak, current_memory)
                
                time.sleep(0.1)  # Check every 100ms
                
            except (psutil.NoSuchProcess, KeyError):
                break
    
    def _log_performance_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        memory_mb = metrics.memory_delta / (1024 * 1024)
        peak_mb = metrics.memory_peak / (1024 * 1024)
        
        if metrics.success:
            logger.info(f"Operation '{metrics.operation}' completed in {metrics.duration:.3f}s, "
                       f"memory delta: {memory_mb:.1f}MB, peak: {peak_mb:.1f}MB, "
                       f"CPU: {metrics.cpu_percent:.1f}%")
        else:
            logger.warning(f"Operation '{metrics.operation}' failed after {metrics.duration:.3f}s, "
                          f"error: {metrics.error_type}")
    
    def get_operation_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get summary statistics for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Summary statistics
        """
        with self._lock:
            metrics_list = self.operation_stats.get(operation_name, [])
        
        if not metrics_list:
            return {"operation": operation_name, "count": 0}
        
        durations = [m.duration for m in metrics_list]
        memory_deltas = [m.memory_delta for m in metrics_list]
        success_count = sum(1 for m in metrics_list if m.success)
        
        return {
            "operation": operation_name,
            "count": len(metrics_list),
            "success_rate": success_count / len(metrics_list),
            "duration": {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
                "total": sum(durations)
            },
            "memory": {
                "min_delta_mb": min(memory_deltas) / (1024 * 1024),
                "max_delta_mb": max(memory_deltas) / (1024 * 1024),
                "avg_delta_mb": sum(memory_deltas) / len(memory_deltas) / (1024 * 1024)
            }
        }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all operations."""
        with self._lock:
            operation_names = list(self.operation_stats.keys())
        
        return {name: self.get_operation_summary(name) for name in operation_names}
    
    def export_metrics(self, filename: str = "performance_metrics.json"):
        """Export metrics to file.
        
        Args:
            filename: Output filename
        """
        import json
        
        # Convert metrics to serializable format
        serializable_metrics = []
        
        with self._lock:
            for metrics in self.metrics_history:
                serializable_metrics.append({
                    "operation": metrics.operation,
                    "start_time": metrics.start_time,
                    "end_time": metrics.end_time,
                    "duration": metrics.duration,
                    "memory_before_mb": metrics.memory_before / (1024 * 1024),
                    "memory_after_mb": metrics.memory_after / (1024 * 1024),
                    "memory_peak_mb": metrics.memory_peak / (1024 * 1024),
                    "memory_delta_mb": metrics.memory_delta / (1024 * 1024),
                    "cpu_percent": metrics.cpu_percent,
                    "success": metrics.success,
                    "error_type": metrics.error_type,
                    "additional_data": metrics.additional_data
                })
        
        export_data = {
            "timestamp": time.time(),
            "total_operations": len(serializable_metrics),
            "metrics": serializable_metrics,
            "summaries": self.get_all_summaries()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(serializable_metrics)} metrics to {filename}")


class MetricsCollector:
    """Collect and aggregate various system metrics."""
    
    def __init__(self, collection_interval: float = 1.0):
        """Initialize metrics collector.
        
        Args:
            collection_interval: Time between metric collections in seconds
        """
        self.collection_interval = collection_interval
        self.metrics = defaultdict(list)
        self.is_collecting = False
        self.collection_thread = None
        self._lock = threading.Lock()
    
    def start_collection(self):
        """Start continuous metrics collection."""
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info(f"Started metrics collection with interval {self.collection_interval}s")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        logger.info("Stopped metrics collection")
    
    def _collect_loop(self):
        """Main collection loop."""
        while self.is_collecting:
            try:
                timestamp = time.time()
                
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                disk_info = psutil.disk_usage('/')
                
                # Collect process metrics
                process = psutil.Process()
                process_memory = process.memory_info().rss
                process_cpu = process.cpu_percent()
                
                # Store metrics
                with self._lock:
                    self.metrics["timestamp"].append(timestamp)
                    self.metrics["system_cpu_percent"].append(cpu_percent)
                    self.metrics["system_memory_percent"].append(memory_info.percent)
                    self.metrics["system_memory_available_mb"].append(memory_info.available / (1024 * 1024))
                    self.metrics["disk_usage_percent"].append(disk_info.percent)
                    self.metrics["process_memory_mb"].append(process_memory / (1024 * 1024))
                    self.metrics["process_cpu_percent"].append(process_cpu)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics snapshot."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            process = psutil.Process()
            process_memory = process.memory_info().rss
            process_cpu = process.cpu_percent()
            
            return {
                "timestamp": time.time(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "memory_available_mb": memory_info.available / (1024 * 1024),
                    "disk_usage_percent": disk_info.percent
                },
                "process": {
                    "memory_mb": process_memory / (1024 * 1024),
                    "cpu_percent": process_cpu
                }
            }
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {"error": str(e)}
    
    def get_metric_summary(self, metric_name: str, window_size: Optional[int] = None) -> Dict[str, float]:
        """Get summary statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            window_size: Number of recent samples to include (None for all)
            
        Returns:
            Summary statistics
        """
        with self._lock:
            values = self.metrics.get(metric_name, [])
        
        if not values:
            return {}
        
        if window_size:
            values = values[-window_size:]
        
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "count": len(values)
        }


class HealthChecker:
    """System health monitoring and alerting."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks = {}
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "response_time_ms": 5000.0
        }
        self.alerts = []
    
    def register_check(self, name: str, check_func: Callable[[], bool], 
                      description: str = ""):
        """Register a health check.
        
        Args:
            name: Check name
            check_func: Function that returns True if healthy
            description: Check description
        """
        self.checks[name] = {
            "function": check_func,
            "description": description,
            "last_result": None,
            "last_check_time": None
        }
        
        logger.info(f"Registered health check: {name}")
    
    def set_threshold(self, metric: str, threshold: float):
        """Set threshold for a metric.
        
        Args:
            metric: Metric name
            threshold: Threshold value
        """
        self.thresholds[metric] = threshold
        logger.info(f"Set threshold for {metric}: {threshold}")
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks.
        
        Returns:
            Health check results
        """
        results = {
            "overall_healthy": True,
            "timestamp": time.time(),
            "checks": {},
            "system_metrics": {},
            "alerts": []
        }
        
        # Run custom checks
        for name, check_info in self.checks.items():
            try:
                start_time = time.time()
                is_healthy = check_info["function"]()
                duration = time.time() - start_time
                
                check_info["last_result"] = is_healthy
                check_info["last_check_time"] = start_time
                
                results["checks"][name] = {
                    "healthy": is_healthy,
                    "duration_ms": duration * 1000,
                    "description": check_info["description"]
                }
                
                if not is_healthy:
                    results["overall_healthy"] = False
                    alert = f"Health check failed: {name}"
                    results["alerts"].append(alert)
                    self.alerts.append({"timestamp": time.time(), "message": alert})
                    logger.warning(alert)
                
            except Exception as e:
                results["checks"][name] = {
                    "healthy": False,
                    "error": str(e),
                    "description": check_info["description"]
                }
                results["overall_healthy"] = False
                
                alert = f"Health check error in {name}: {e}"
                results["alerts"].append(alert)
                self.alerts.append({"timestamp": time.time(), "message": alert})
                logger.error(alert)
        
        # Check system metrics against thresholds
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            results["system_metrics"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info.percent,
                "disk_usage_percent": disk_info.percent
            }
            
            # Check thresholds
            if cpu_percent > self.thresholds["cpu_percent"]:
                alert = f"High CPU usage: {cpu_percent:.1f}% > {self.thresholds['cpu_percent']}%"
                results["alerts"].append(alert)
                results["overall_healthy"] = False
                logger.warning(alert)
            
            if memory_info.percent > self.thresholds["memory_percent"]:
                alert = f"High memory usage: {memory_info.percent:.1f}% > {self.thresholds['memory_percent']}%"
                results["alerts"].append(alert)
                results["overall_healthy"] = False
                logger.warning(alert)
            
            if disk_info.percent > self.thresholds["disk_usage_percent"]:
                alert = f"High disk usage: {disk_info.percent:.1f}% > {self.thresholds['disk_usage_percent']}%"
                results["alerts"].append(alert)
                results["overall_healthy"] = False
                logger.warning(alert)
                
        except Exception as e:
            logger.error(f"Error checking system metrics: {e}")
            results["system_metrics"] = {"error": str(e)}
        
        if results["overall_healthy"]:
            logger.debug("All health checks passed")
        else:
            logger.warning(f"Health issues detected: {len(results['alerts'])} alerts")
        
        return results
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        alerts = self.alerts[-limit:] if limit else self.alerts
        return alerts.copy()


# Global instances
global_performance_monitor = PerformanceMonitor()
global_metrics_collector = MetricsCollector()
global_health_checker = HealthChecker()


@contextmanager
def resource_monitor(operation_name: str, **metadata):
    """Context manager for monitoring resource usage.
    
    Args:
        operation_name: Name of the operation
        **metadata: Additional metadata
    """
    with global_performance_monitor.measure_operation(operation_name, **metadata):
        yield


def log_performance(operation_name: str):
    """Decorator for logging performance of functions.
    
    Args:
        operation_name: Name of the operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with resource_monitor(operation_name, function=func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Built-in health checks
def check_memory_usage() -> bool:
    """Check if memory usage is within acceptable limits."""
    memory_info = psutil.virtual_memory()
    return memory_info.percent < 90.0


def check_disk_space() -> bool:
    """Check if disk space is sufficient."""
    disk_info = psutil.disk_usage('/')
    return disk_info.percent < 95.0


def check_cpu_usage() -> bool:
    """Check if CPU usage is reasonable."""
    cpu_percent = psutil.cpu_percent(interval=1.0)
    return cpu_percent < 95.0


# Register default health checks
global_health_checker.register_check("memory_usage", check_memory_usage, "System memory usage check")
global_health_checker.register_check("disk_space", check_disk_space, "Disk space availability check")
global_health_checker.register_check("cpu_usage", check_cpu_usage, "CPU usage check")