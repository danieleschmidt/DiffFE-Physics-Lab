"""Unit tests for performance module."""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from src.performance.cache import CacheManager, cached, LRUCache
from src.performance.profiler import PerformanceProfiler, ProfileEntry, ProfileStats
from src.performance.optimizer import PerformanceOptimizer, OptimizationConfig
from src.performance.monitor import PerformanceMonitor, SystemMetrics, ApplicationMetrics, AlertLevel


class TestCacheManager:
    """Test cases for CacheManager."""
    
    def test_cache_manager_init(self):
        """Test cache manager initialization."""
        cache = CacheManager(max_size=100, ttl=300.0)
        
        assert cache.max_size == 100
        assert cache.ttl == 300.0
        assert len(cache._cache) == 0
    
    def test_cache_set_get(self):
        """Test basic cache set and get operations."""
        cache = CacheManager(max_size=10)
        
        # Set and get value
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        # Get non-existent key
        assert cache.get('nonexistent') is None
        assert cache.get('nonexistent', 'default') == 'default'
    
    def test_cache_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        cache = CacheManager(max_size=10, ttl=0.1)  # 100ms TTL
        
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.get('key1') is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = CacheManager(max_size=2)
        
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        # Access key1 to make it more recently used
        cache.get('key1')
        
        # Add key3, should evict key2 (least recently used)
        cache.set('key3', 'value3')
        
        assert cache.get('key1') == 'value1'  # Still there
        assert cache.get('key2') is None      # Evicted
        assert cache.get('key3') == 'value3'  # New value
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = CacheManager(max_size=10)
        
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        assert len(cache) == 2
        
        cache.clear()
        assert len(cache) == 0
        assert cache.get('key1') is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = CacheManager(max_size=10)
        
        # Initial stats
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
        
        # Add some data and test access
        cache.set('key1', 'value1')
        cache.get('key1')  # Hit
        cache.get('key2')  # Miss
        
        stats = cache.get_stats()
        assert stats['size'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_cached_decorator(self):
        """Test the cached function decorator."""
        cache = CacheManager(max_size=10)
        
        call_count = 0
        
        @cached(cache)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # No additional call
        
        # Different args should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2


class TestPerformanceProfiler:
    """Test cases for PerformanceProfiler."""
    
    def test_profiler_init(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler(
            enable_memory_profiling=True,
            enable_stack_trace=False
        )
        
        assert profiler.enable_memory_profiling is True
        assert profiler.enable_stack_trace is False
        assert len(profiler.entries) == 0
        assert len(profiler.stats) == 0
    
    def test_profile_context_manager(self):
        """Test profiling context manager."""
        profiler = PerformanceProfiler()
        
        with profiler.profile_context('test_operation'):
            time.sleep(0.01)  # Small delay
        
        assert len(profiler.entries) == 1
        entry = profiler.entries[0]
        assert entry.name == 'test_operation'
        assert entry.duration > 0.005  # At least 5ms
        assert 'test_operation' in profiler.stats
    
    def test_profile_decorator(self):
        """Test profiling function decorator."""
        profiler = PerformanceProfiler()
        
        @profiler.profile()
        def test_function(n):
            return sum(range(n))
        
        result = test_function(1000)
        assert result == sum(range(1000))
        
        assert len(profiler.entries) == 1
        assert profiler.entries[0].name == 'test_function'
        assert 'test_function' in profiler.stats
    
    def test_profile_with_args(self):
        """Test profiling with function arguments."""
        profiler = PerformanceProfiler()
        
        @profiler.profile(include_args=True)
        def test_function(x, y=10):
            return x * y
        
        result = test_function(5, y=20)
        assert result == 100
        
        entry = profiler.entries[0]
        assert 'args' in entry.metadata
        assert 'kwargs' in entry.metadata
    
    def test_get_stats(self):
        """Test getting profiling statistics."""
        profiler = PerformanceProfiler()
        
        # Profile a function multiple times
        @profiler.profile()
        def test_function(delay):
            time.sleep(delay)
            return delay
        
        test_function(0.01)
        test_function(0.02)
        test_function(0.015)
        
        stats = profiler.get_stats('test_function')
        assert stats is not None
        assert stats.call_count == 3
        assert stats.total_time > 0.04  # At least 40ms total
        assert stats.avg_time > 0.01   # Average > 10ms
        assert stats.min_time > 0.005  # Min > 5ms
        assert stats.max_time > 0.015  # Max > 15ms
    
    def test_get_slowest_calls(self):
        """Test getting slowest function calls."""
        profiler = PerformanceProfiler()
        
        @profiler.profile()
        def fast_function():
            time.sleep(0.001)
        
        @profiler.profile()
        def slow_function():
            time.sleep(0.01)
        
        fast_function()
        slow_function()
        fast_function()
        
        slowest = profiler.get_slowest_calls(limit=2)
        assert len(slowest) == 2
        assert slowest[0].name == 'slow_function'  # Slowest first
        assert slowest[0].duration > slowest[1].duration
    
    def test_generate_report(self):
        """Test generating profiling report."""
        profiler = PerformanceProfiler()
        
        @profiler.profile()
        def test_function():
            time.sleep(0.001)
        
        test_function()
        test_function()
        
        report = profiler.generate_report()
        
        assert 'summary' in report
        assert 'top_functions_by_time' in report
        assert 'function_statistics' in report
        assert 'slowest_calls' in report
        
        assert report['summary']['total_entries'] == 2
        assert report['summary']['unique_functions'] == 1
    
    def test_reset_profiler(self):
        """Test resetting profiler data."""
        profiler = PerformanceProfiler()
        
        @profiler.profile()
        def test_function():
            pass
        
        test_function()
        assert len(profiler.entries) == 1
        assert len(profiler.stats) == 1
        
        profiler.reset()
        assert len(profiler.entries) == 0
        assert len(profiler.stats) == 0


class TestPerformanceOptimizer:
    """Test cases for PerformanceOptimizer."""
    
    def test_optimizer_init(self):
        """Test optimizer initialization."""
        config = OptimizationConfig(
            max_threads=4,
            max_processes=2,
            batch_size=50
        )
        
        optimizer = PerformanceOptimizer(config)
        assert optimizer.config.max_threads == 4
        assert optimizer.config.max_processes == 2
        assert optimizer.config.batch_size == 50
    
    def test_performance_context(self):
        """Test performance monitoring context."""
        optimizer = PerformanceOptimizer()
        
        with optimizer.performance_context('test_task'):
            time.sleep(0.001)
        
        assert len(optimizer._task_history) == 1
        task = optimizer._task_history[0]
        assert task['task'] == 'test_task'
        assert task['execution_time'] > 0
    
    def test_optimize_decorator_auto(self):
        """Test optimize decorator with auto strategy."""
        optimizer = PerformanceOptimizer()
        
        @optimizer.optimize(strategy="auto")
        def test_function(data):
            return sum(x**2 for x in data)
        
        result = test_function(range(100))
        expected = sum(x**2 for x in range(100))
        assert result == expected
    
    def test_optimize_decorator_cache(self):
        """Test optimize decorator with cache strategy."""
        optimizer = PerformanceOptimizer()
        
        call_count = 0
        
        @optimizer.optimize(strategy="cache")
        def expensive_function(n):
            nonlocal call_count
            call_count += 1
            return n**2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 25
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = expensive_function(5)
        assert result2 == 25
        assert call_count == 1  # No additional call
    
    def test_parallel_map(self):
        """Test parallel map function."""
        optimizer = PerformanceOptimizer()
        
        def square(x):
            return x**2
        
        data = list(range(10))
        result = optimizer.parallel_map(square, data, chunk_size=3)
        expected = [x**2 for x in data]
        
        assert sorted(result) == sorted(expected)
    
    def test_detect_bottlenecks(self):
        """Test bottleneck detection."""
        optimizer = PerformanceOptimizer()
        
        # Simulate some slow tasks
        with optimizer.performance_context('slow_task'):
            time.sleep(0.002)  # 2ms - should be flagged as slow
        
        # Add multiple slow executions
        for _ in range(10):
            optimizer._task_history.append({
                'task': 'slow_task',
                'execution_time': 2.0,  # 2 seconds - very slow
                'memory_delta': 50.0,
                'timestamp': time.time()
            })
        
        bottlenecks = optimizer.detect_bottlenecks()
        
        assert len(bottlenecks) > 0
        slow_bottlenecks = [b for b in bottlenecks if b['type'] == 'slow_execution']
        assert len(slow_bottlenecks) > 0
    
    def test_get_metrics(self):
        """Test getting performance metrics."""
        optimizer = PerformanceOptimizer()
        
        # Add some task history
        optimizer._task_history.append({
            'task': 'test_task',
            'execution_time': 0.1,
            'memory_delta': 10.0,
            'timestamp': time.time()
        })
        
        metrics = optimizer.get_metrics()
        assert isinstance(metrics.execution_time, float)
        assert isinstance(metrics.memory_usage_mb, float)
    
    def test_generate_optimization_report(self):
        """Test generating optimization report."""
        optimizer = PerformanceOptimizer()
        
        # Add some task history
        optimizer._task_history.extend([
            {
                'task': 'task1',
                'execution_time': 0.1,
                'memory_delta': 5.0,
                'timestamp': time.time()
            },
            {
                'task': 'task2', 
                'execution_time': 0.2,
                'memory_delta': 10.0,
                'timestamp': time.time()
            }
        ])
        
        report = optimizer.generate_optimization_report()
        
        assert 'summary' in report
        assert 'performance_metrics' in report
        assert 'top_tasks_by_time' in report
        assert 'recommendations' in report
        assert 'configuration' in report


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def test_monitor_init(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor(
            monitoring_interval=2.0,
            history_size=1000,
            enable_system_monitoring=True
        )
        
        assert monitor.monitoring_interval == 2.0
        assert monitor.history_size == 1000
        assert monitor.enable_system_monitoring is True
        assert len(monitor.system_metrics_history) == 0
    
    @patch('src.performance.monitor.psutil')
    def test_collect_system_metrics(self, mock_psutil):
        """Test system metrics collection."""
        # Mock psutil methods
        mock_psutil.cpu_percent.return_value = 45.5
        
        mock_memory = Mock()
        mock_memory.percent = 67.8
        mock_memory.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.percent = 85.2
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        mock_psutil.pids.return_value = list(range(150))
        
        mock_process = Mock()
        mock_process.num_threads.return_value = 8
        mock_psutil.Process.return_value = mock_process
        
        monitor = PerformanceMonitor()
        metrics = monitor._collect_system_metrics()
        
        assert metrics is not None
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 67.8
        assert metrics.disk_usage_percent == 85.2
        assert metrics.process_count == 150
        assert metrics.thread_count == 8
    
    def test_log_application_metric(self):
        """Test logging application metrics."""
        monitor = PerformanceMonitor()
        
        # Log some metrics
        monitor.log_application_metric('response_time', 150.0)
        monitor.log_application_metric('response_time', 200.0)
        monitor.log_application_metric('error_count', 5.0)
        
        # Check that metrics are stored
        assert 'response_time' in monitor._custom_metrics
        assert len(monitor._custom_metrics['response_time']) == 2
        assert monitor._custom_metrics['response_time'] == [150.0, 200.0]
        assert monitor._custom_metrics['error_count'] == [5.0]
    
    def test_add_alert_rule(self):
        """Test adding custom alert rules."""
        monitor = PerformanceMonitor()
        
        monitor.add_alert_rule(
            'high_response_time',
            'response_time_avg',
            threshold=500.0,
            level=AlertLevel.WARNING
        )
        
        assert 'high_response_time' in monitor.alert_rules
        rule = monitor.alert_rules['high_response_time']
        assert rule['metric'] == 'response_time_avg'
        assert rule['threshold'] == 500.0
        assert rule['level'] == AlertLevel.WARNING
    
    def test_alert_callbacks(self):
        """Test alert callback system."""
        monitor = PerformanceMonitor()
        
        callback_called = False
        alert_data = None
        
        def test_callback(alert):
            nonlocal callback_called, alert_data
            callback_called = True
            alert_data = alert
        
        monitor.add_alert_callback(test_callback)
        
        # Manually trigger an alert
        from src.performance.monitor import PerformanceAlert
        test_alert = PerformanceAlert(
            alert_type='test_alert',
            level=AlertLevel.WARNING,
            message='Test alert message'
        )
        
        monitor._trigger_alert(test_alert)
        
        assert callback_called is True
        assert alert_data == test_alert
    
    def test_get_current_metrics(self):
        """Test getting current metrics."""
        monitor = PerformanceMonitor()
        
        # Add some mock data
        mock_system_metrics = SystemMetrics(cpu_percent=50.0, memory_percent=60.0)
        mock_app_metrics = ApplicationMetrics(request_count=100, error_count=5)
        
        monitor.system_metrics_history.append(mock_system_metrics)
        monitor.application_metrics_history.append(mock_app_metrics)
        
        current = monitor.get_current_metrics()
        
        assert 'system' in current
        assert 'application' in current
        assert current['system']['cpu_percent'] == 50.0
        assert current['application']['request_count'] == 100
    
    def test_get_trend_analysis(self):
        """Test trend analysis."""
        monitor = PerformanceMonitor()
        
        # Add trend data
        monitor._trend_data['cpu_percent'] = [10.0, 20.0, 30.0, 40.0, 50.0]
        
        analysis = monitor.get_trend_analysis('cpu_percent', window_minutes=5)
        
        assert 'metric_name' in analysis
        assert 'statistics' in analysis
        assert 'trend_direction' in analysis
        assert analysis['metric_name'] == 'cpu_percent'
        assert analysis['statistics']['mean'] == 30.0  # Average of the values
    
    def test_generate_performance_report(self):
        """Test generating performance report."""
        monitor = PerformanceMonitor()
        
        # Add some mock data
        mock_system_metrics = SystemMetrics(cpu_percent=45.0)
        monitor.system_metrics_history.append(mock_system_metrics)
        
        report = monitor.generate_performance_report(hours=1)
        
        assert 'report_period' in report
        assert 'generated_at' in report
        assert 'current_metrics' in report
        assert 'alert_summary' in report
        assert 'system_health' in report
        assert report['report_period'] == '1 hours'


class TestPerformanceIntegration:
    """Integration tests for performance components."""
    
    def test_profiler_optimizer_integration(self):
        """Test profiler and optimizer working together."""
        profiler = PerformanceProfiler()
        optimizer = PerformanceOptimizer()
        
        @profiler.profile()
        @optimizer.optimize(strategy="cache")
        def expensive_computation(n):
            time.sleep(0.001)  # Small delay
            return sum(range(n))
        
        # Call function multiple times
        result1 = expensive_computation(100)
        result2 = expensive_computation(100)  # Should use cache
        
        assert result1 == result2 == sum(range(100))
        
        # Check profiler recorded both calls
        assert len(profiler.entries) == 2
        
        # Check optimizer has task history
        assert len(optimizer._task_history) > 0
    
    def test_cache_profiler_integration(self):
        """Test cache and profiler integration."""
        cache = CacheManager(max_size=10)
        profiler = PerformanceProfiler()
        
        @cached(cache)
        @profiler.profile()
        def cached_function(x):
            time.sleep(0.001)
            return x * 2
        
        # First call - cache miss, should be profiled
        result1 = cached_function(5)
        assert result1 == 10
        
        # Second call - cache hit, should still be profiled but faster
        result2 = cached_function(5)
        assert result2 == 10
        
        # Check profiler recorded both calls
        assert len(profiler.entries) == 2
        
        # Check cache stats
        stats = cache.get_stats()
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1
    
    def test_monitor_alert_integration(self):
        """Test monitor and alert system integration."""
        monitor = PerformanceMonitor()
        
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
        
        monitor.add_alert_callback(alert_handler)
        monitor.add_alert_rule('test_metric', 'test_value', threshold=50.0)
        
        # Log metrics that should trigger alert
        monitor.log_application_metric('test_value', 75.0)
        
        # Simulate metrics collection and alert checking
        app_metrics = monitor._collect_application_metrics()
        if app_metrics:
            monitor._check_application_alerts(app_metrics)
        
        # Should have triggered an alert
        assert len(alerts_received) > 0
        assert alerts_received[0].alert_type == 'custom_test_metric'


class TestPerformanceErrors:
    """Test error handling in performance components."""
    
    def test_cache_invalid_config(self):
        """Test cache with invalid configuration."""
        with pytest.raises(ValueError):
            CacheManager(max_size=0)  # Invalid size
        
        with pytest.raises(ValueError):
            CacheManager(ttl=-1.0)  # Invalid TTL
    
    def test_profiler_invalid_operations(self):
        """Test profiler error handling."""
        profiler = PerformanceProfiler()
        
        # Test with invalid parameters
        with pytest.raises(ValueError):
            profiler.get_trend_analysis('nonexistent_metric')
    
    def test_optimizer_resource_limits(self):
        """Test optimizer with resource constraints."""
        config = OptimizationConfig(
            max_threads=1,
            max_processes=1,
            memory_limit_mb=10.0  # Very low limit
        )
        
        optimizer = PerformanceOptimizer(config)
        
        # Should handle memory optimization gracefully
        optimizer.optimize_memory_usage(target_mb=5.0)
        
        # Should not crash even with low resources
        @optimizer.optimize(strategy="parallel")
        def test_function(data):
            return sum(data)
        
        result = test_function(range(100))
        assert result == sum(range(100))
    
    def test_monitor_missing_dependencies(self):
        """Test monitor without psutil."""
        with patch('src.performance.monitor.psutil', None):
            monitor = PerformanceMonitor()
            
            # Should handle missing psutil gracefully
            metrics = monitor._collect_system_metrics()
            # May return None or minimal metrics
            assert metrics is None or isinstance(metrics, SystemMetrics)