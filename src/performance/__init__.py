"""Performance optimization and monitoring package.

This package provides comprehensive performance optimization tools including:
- Caching with LRU eviction and TTL support
- Function-level profiling and performance analysis
- Performance optimization with parallel processing
- Real-time performance monitoring and alerting

Examples
--------
Basic Usage:
>>> from src.performance import optimize, profile, log_metric
>>>
>>> @optimize(strategy="parallel")
>>> @profile
>>> def expensive_computation(data):
...     return sum(x**2 for x in data)
>>>
>>> result = expensive_computation(range(100000))
>>> log_metric("computation_time", 1.23)

Advanced Usage:
>>> from src.performance import PerformanceOptimizer, PerformanceMonitor
>>>
>>> # Custom optimizer
>>> optimizer = PerformanceOptimizer()
>>>
>>> # Start monitoring
>>> monitor = PerformanceMonitor()
>>> monitor.start_monitoring()
>>> monitor.add_alert_rule("high_cpu", "cpu_percent", 80.0)
"""

# Cache management
from .cache import CacheManager, cached, clear_global_cache, get_global_cache

# Performance monitoring
from .monitor import (
    AlertLevel,
    ApplicationMetrics,
    PerformanceAlert,
    PerformanceMonitor,
    SystemMetrics,
    add_alert_rule,
    get_global_monitor,
    log_metric,
)

# Performance optimization
from .optimizer import (
    OptimizationConfig,
    PerformanceMetrics,
    PerformanceOptimizer,
    get_global_optimizer,
    optimize,
    parallel_map,
)

# Performance profiling
from .profiler import (
    CPUMonitor,
    MemoryMonitor,
    PerformanceProfiler,
    ProfileEntry,
    ProfileStats,
    get_global_profiler,
    profile,
)

__all__ = [
    # Cache
    "CacheManager",
    "cached",
    "get_global_cache",
    "clear_global_cache",
    # Profiler
    "PerformanceProfiler",
    "ProfileEntry",
    "ProfileStats",
    "CPUMonitor",
    "MemoryMonitor",
    "get_global_profiler",
    "profile",
    # Optimizer
    "PerformanceOptimizer",
    "OptimizationConfig",
    "PerformanceMetrics",
    "get_global_optimizer",
    "optimize",
    "parallel_map",
    # Monitor
    "PerformanceMonitor",
    "SystemMetrics",
    "ApplicationMetrics",
    "PerformanceAlert",
    "AlertLevel",
    "get_global_monitor",
    "log_metric",
    "add_alert_rule",
]

# Version information
__version__ = "1.0.0"
__author__ = "DiffFE-Physics-Lab Team"
__description__ = "Performance optimization and monitoring tools"
