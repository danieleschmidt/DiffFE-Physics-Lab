"""Performance profiling utilities for optimization."""

import functools
import logging
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ProfileEntry:
    """Single profiling measurement."""

    name: str
    start_time: float
    end_time: float
    duration: float
    memory_start: float
    memory_end: float
    memory_delta: float
    cpu_percent: float
    thread_id: int
    call_stack: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileStats:
    """Aggregated profiling statistics."""

    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    avg_time: float = 0.0
    total_memory: float = 0.0
    avg_memory: float = 0.0
    max_memory: float = 0.0
    percentiles: Dict[str, float] = field(default_factory=dict)
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))


class PerformanceProfiler:
    """Comprehensive performance profiler for code analysis.

    Provides detailed profiling capabilities including:
    - Function-level timing and memory profiling
    - Call stack analysis
    - Statistical aggregation and percentile calculations
    - Real-time monitoring and alerting
    - Thread-safe profiling for concurrent code

    Examples
    --------
    >>> profiler = PerformanceProfiler()
    >>>
    >>> @profiler.profile
    >>> def expensive_function(n):
    ...     return sum(range(n))
    >>>
    >>> result = expensive_function(1000000)
    >>> stats = profiler.get_stats('expensive_function')
    >>> print(f"Average time: {stats.avg_time:.4f}s")
    """

    def __init__(
        self, enable_memory_profiling: bool = True, enable_stack_trace: bool = False
    ):
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_stack_trace = enable_stack_trace
        self.entries = []
        self.stats = {}
        self.active_profiles = {}  # Thread-local active profiles
        self._lock = threading.RLock()

        # Performance monitoring
        self.cpu_monitor = CPUMonitor() if self._is_monitoring_available() else None
        self.memory_monitor = (
            MemoryMonitor() if self._is_monitoring_available() else None
        )

        # Alert thresholds
        self.alert_thresholds = {
            "execution_time": 5.0,  # seconds
            "memory_usage": 100.0,  # MB
            "cpu_usage": 80.0,  # percent
        }

        self.alert_callbacks = []

        logger.info(
            f"Performance profiler initialized (memory={enable_memory_profiling}, stack={enable_stack_trace})"
        )

    def _is_monitoring_available(self) -> bool:
        """Check if system monitoring is available."""
        try:
            import psutil

            return True
        except ImportError:
            return False

    @contextmanager
    def profile_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling code blocks.

        Parameters
        ----------
        name : str
            Name of the profiled section
        metadata : Dict[str, Any], optional
            Additional metadata to store

        Examples
        --------
        >>> with profiler.profile_context('database_query'):
        ...     result = expensive_db_operation()
        """
        start_time = time.time()
        thread_id = threading.get_ident()

        # Memory monitoring
        memory_start = self._get_memory_usage() if self.enable_memory_profiling else 0.0

        # CPU monitoring
        cpu_start = self._get_cpu_percent()

        # Stack trace
        call_stack = self._get_call_stack() if self.enable_stack_trace else None

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            memory_end = (
                self._get_memory_usage() if self.enable_memory_profiling else 0.0
            )
            memory_delta = memory_end - memory_start
            cpu_end = self._get_cpu_percent()

            entry = ProfileEntry(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_start=memory_start,
                memory_end=memory_end,
                memory_delta=memory_delta,
                cpu_percent=(cpu_start + cpu_end) / 2,
                thread_id=thread_id,
                call_stack=call_stack,
                metadata=metadata or {},
            )

            self._record_entry(entry)

    def profile(self, name: Optional[str] = None, include_args: bool = False):
        """Decorator for profiling functions.

        Parameters
        ----------
        name : str, optional
            Custom name for profiling, defaults to function name
        include_args : bool, optional
            Whether to include function arguments in metadata

        Returns
        -------
        Callable
            Decorated function with profiling
        """

        def decorator(func):
            profile_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                metadata = {}

                if include_args:
                    # Store function arguments (be careful with large objects)
                    try:
                        metadata["args"] = str(args)[:1000]  # Limit string length
                        metadata["kwargs"] = str(kwargs)[:1000]
                    except:
                        metadata["args_error"] = "Failed to serialize arguments"

                with self.profile_context(profile_name, metadata):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def _record_entry(self, entry: ProfileEntry):
        """Record profiling entry and update statistics."""
        with self._lock:
            self.entries.append(entry)

            # Update statistics
            if entry.name not in self.stats:
                self.stats[entry.name] = ProfileStats(name=entry.name)

            stats = self.stats[entry.name]
            stats.call_count += 1
            stats.total_time += entry.duration
            stats.min_time = min(stats.min_time, entry.duration)
            stats.max_time = max(stats.max_time, entry.duration)
            stats.avg_time = stats.total_time / stats.call_count

            if self.enable_memory_profiling:
                stats.total_memory += abs(entry.memory_delta)
                stats.avg_memory = stats.total_memory / stats.call_count
                stats.max_memory = max(stats.max_memory, abs(entry.memory_delta))

            # Track recent times for percentile calculation
            stats.recent_times.append(entry.duration)

            # Update percentiles periodically
            if stats.call_count % 10 == 0:
                self._update_percentiles(stats)

            # Check for performance alerts
            self._check_alerts(entry)

    def _update_percentiles(self, stats: ProfileStats):
        """Update percentile statistics."""
        if not stats.recent_times:
            return

        times = sorted(stats.recent_times)
        count = len(times)

        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            index = int((p / 100.0) * count)
            if index >= count:
                index = count - 1
            stats.percentiles[f"p{p}"] = times[index]

    def _check_alerts(self, entry: ProfileEntry):
        """Check if entry triggers performance alerts."""
        alerts = []

        # Execution time alert
        if entry.duration > self.alert_thresholds["execution_time"]:
            alerts.append(
                {
                    "type": "slow_execution",
                    "threshold": self.alert_thresholds["execution_time"],
                    "actual": entry.duration,
                    "entry": entry,
                }
            )

        # Memory usage alert
        if (
            self.enable_memory_profiling
            and abs(entry.memory_delta) > self.alert_thresholds["memory_usage"]
        ):
            alerts.append(
                {
                    "type": "high_memory_usage",
                    "threshold": self.alert_thresholds["memory_usage"],
                    "actual": abs(entry.memory_delta),
                    "entry": entry,
                }
            )

        # CPU usage alert
        if entry.cpu_percent > self.alert_thresholds["cpu_usage"]:
            alerts.append(
                {
                    "type": "high_cpu_usage",
                    "threshold": self.alert_thresholds["cpu_usage"],
                    "actual": entry.cpu_percent,
                    "entry": entry,
                }
            )

        # Trigger alert callbacks
        for alert in alerts:
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger performance alert."""
        logger.warning(
            f"Performance alert: {alert['type']} - "
            f"{alert['entry'].name} took {alert['actual']:.3f} "
            f"(threshold: {alert['threshold']})"
        )

        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for performance alerts.

        Parameters
        ----------
        callback : Callable[[Dict[str, Any]], None]
            Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
        logger.info("Added performance alert callback")

    def set_alert_threshold(self, alert_type: str, threshold: float):
        """Set alert threshold.

        Parameters
        ----------
        alert_type : str
            Type of alert ('execution_time', 'memory_usage', 'cpu_usage')
        threshold : float
            Threshold value
        """
        if alert_type in self.alert_thresholds:
            self.alert_thresholds[alert_type] = threshold
            logger.info(f"Set alert threshold {alert_type} = {threshold}")
        else:
            logger.warning(f"Unknown alert type: {alert_type}")

    def get_stats(
        self, name: Optional[str] = None
    ) -> Union[ProfileStats, Dict[str, ProfileStats]]:
        """Get profiling statistics.

        Parameters
        ----------
        name : str, optional
            Name of specific function, or all if None

        Returns
        -------
        Union[ProfileStats, Dict[str, ProfileStats]]
            Statistics for function or all functions
        """
        with self._lock:
            if name:
                return self.stats.get(name)
            else:
                return dict(self.stats)

    def get_recent_entries(
        self, name: Optional[str] = None, limit: int = 100
    ) -> List[ProfileEntry]:
        """Get recent profiling entries.

        Parameters
        ----------
        name : str, optional
            Filter by function name
        limit : int, optional
            Maximum number of entries to return

        Returns
        -------
        List[ProfileEntry]
            Recent profiling entries
        """
        with self._lock:
            entries = self.entries

            if name:
                entries = [e for e in entries if e.name == name]

            return entries[-limit:]

    def get_slowest_calls(self, limit: int = 10) -> List[ProfileEntry]:
        """Get slowest function calls.

        Parameters
        ----------
        limit : int, optional
            Number of slowest calls to return

        Returns
        -------
        List[ProfileEntry]
            Slowest function calls
        """
        with self._lock:
            sorted_entries = sorted(
                self.entries, key=lambda e: e.duration, reverse=True
            )
            return sorted_entries[:limit]

    def get_memory_intensive_calls(self, limit: int = 10) -> List[ProfileEntry]:
        """Get most memory-intensive function calls.

        Parameters
        ----------
        limit : int, optional
            Number of calls to return

        Returns
        -------
        List[ProfileEntry]
            Most memory-intensive function calls
        """
        if not self.enable_memory_profiling:
            return []

        with self._lock:
            sorted_entries = sorted(
                self.entries, key=lambda e: abs(e.memory_delta), reverse=True
            )
            return sorted_entries[:limit]

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive profiling report.

        Returns
        -------
        Dict[str, Any]
            Detailed profiling report
        """
        with self._lock:
            total_entries = len(self.entries)
            total_time = sum(e.duration for e in self.entries)

            # Top functions by total time
            function_times = defaultdict(float)
            function_calls = defaultdict(int)

            for entry in self.entries:
                function_times[entry.name] += entry.duration
                function_calls[entry.name] += 1

            top_functions = sorted(
                function_times.items(), key=lambda x: x[1], reverse=True
            )[:10]

            # Memory statistics
            memory_stats = {}
            if self.enable_memory_profiling:
                total_memory = sum(abs(e.memory_delta) for e in self.entries)
                memory_stats = {
                    "total_memory_delta": total_memory,
                    "avg_memory_per_call": (
                        total_memory / total_entries if total_entries > 0 else 0
                    ),
                }

            # Thread statistics
            thread_stats = defaultdict(lambda: {"calls": 0, "time": 0.0})
            for entry in self.entries:
                thread_stats[entry.thread_id]["calls"] += 1
                thread_stats[entry.thread_id]["time"] += entry.duration

            report = {
                "summary": {
                    "total_entries": total_entries,
                    "total_time": total_time,
                    "unique_functions": len(self.stats),
                    "avg_call_time": (
                        total_time / total_entries if total_entries > 0 else 0
                    ),
                },
                "top_functions_by_time": top_functions,
                "function_statistics": {
                    name: {
                        "call_count": stats.call_count,
                        "total_time": stats.total_time,
                        "avg_time": stats.avg_time,
                        "min_time": stats.min_time,
                        "max_time": stats.max_time,
                        "percentiles": stats.percentiles,
                    }
                    for name, stats in self.stats.items()
                },
                "memory_statistics": memory_stats,
                "thread_statistics": dict(thread_stats),
                "slowest_calls": [
                    {
                        "name": e.name,
                        "duration": e.duration,
                        "memory_delta": (
                            e.memory_delta if self.enable_memory_profiling else None
                        ),
                        "timestamp": e.start_time,
                    }
                    for e in self.get_slowest_calls(5)
                ],
            }

            return report

    def reset(self):
        """Reset all profiling data."""
        with self._lock:
            self.entries.clear()
            self.stats.clear()
            self.active_profiles.clear()

        logger.info("Profiler data reset")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self._is_monitoring_available():
            return 0.0

        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except:
            return 0.0

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        if not self._is_monitoring_available():
            return 0.0

        try:
            return psutil.cpu_percent(interval=None)
        except:
            return 0.0

    def _get_call_stack(self) -> List[str]:
        """Get current call stack."""
        try:
            stack = traceback.extract_stack()
            # Filter out profiler internals
            filtered_stack = []
            for frame in stack:
                if "profiler.py" not in frame.filename:
                    filtered_stack.append(
                        f"{frame.filename}:{frame.lineno} in {frame.name}"
                    )
            return filtered_stack[-10:]  # Last 10 frames
        except:
            return []


class CPUMonitor:
    """CPU usage monitoring."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.history = deque(maxlen=3600)  # 1 hour of data
        self._monitoring = False
        self._thread = None

    def start_monitoring(self):
        """Start CPU monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("CPU monitoring started")

    def stop_monitoring(self):
        """Stop CPU monitoring."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("CPU monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=self.interval)
                timestamp = time.time()

                self.history.append(
                    {"timestamp": timestamp, "cpu_percent": cpu_percent}
                )

            except Exception as e:
                logger.error(f"CPU monitoring error: {e}")

            time.sleep(self.interval)

    def get_current_usage(self) -> float:
        """Get current CPU usage."""
        try:
            return psutil.cpu_percent(interval=None)
        except:
            return 0.0

    def get_average_usage(self, window_seconds: int = 60) -> float:
        """Get average CPU usage over time window."""
        if not self.history:
            return 0.0

        cutoff_time = time.time() - window_seconds
        recent_data = [d for d in self.history if d["timestamp"] > cutoff_time]

        if not recent_data:
            return 0.0

        return sum(d["cpu_percent"] for d in recent_data) / len(recent_data)


class MemoryMonitor:
    """Memory usage monitoring."""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.get_current_usage()

    def get_current_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

    def get_memory_delta(self) -> float:
        """Get memory change since baseline."""
        return self.get_current_usage() - self.baseline_memory

    def reset_baseline(self):
        """Reset memory baseline."""
        self.baseline_memory = self.get_current_usage()


# Global profiler instance
_global_profiler = None


def get_global_profiler() -> PerformanceProfiler:
    """Get global profiler instance.

    Returns
    -------
    PerformanceProfiler
        Global profiler instance
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile(name: Optional[str] = None, include_args: bool = False):
    """Global profiling decorator.

    Parameters
    ----------
    name : str, optional
        Custom profile name
    include_args : bool, optional
        Include function arguments in profiling

    Returns
    -------
    Callable
        Decorated function with profiling
    """
    return get_global_profiler().profile(name=name, include_args=include_args)
