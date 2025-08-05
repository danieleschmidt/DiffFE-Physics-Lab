"""Performance optimization utilities."""

import time
import numpy as np
import threading
import logging
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import queue
import multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    max_threads: int = field(default_factory=lambda: min(8, mp.cpu_count()))
    max_processes: int = field(default_factory=lambda: mp.cpu_count())
    enable_thread_pool: bool = True
    enable_process_pool: bool = True
    thread_pool_timeout: float = 30.0
    process_pool_timeout: float = 60.0
    batch_size: int = 100
    enable_vectorization: bool = True
    memory_limit_mb: float = 1024.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization analysis."""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    thread_count: int = 0
    cache_hit_rate: float = 0.0
    throughput: float = 0.0
    bottleneck_analysis: Dict[str, Any] = field(default_factory=dict)


class PerformanceOptimizer:
    """Comprehensive performance optimization system.
    
    Provides optimization capabilities including:
    - Parallel processing with thread and process pools
    - Vectorized operations
    - Memory management and monitoring
    - Bottleneck detection and analysis
    - Adaptive optimization strategies
    
    Examples
    --------
    >>> optimizer = PerformanceOptimizer()
    >>> 
    >>> # Optimize function execution
    >>> @optimizer.optimize
    >>> def expensive_computation(data):
    ...     return sum(x**2 for x in data)
    >>> 
    >>> result = expensive_computation(range(1000000))
    >>> metrics = optimizer.get_metrics()
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.metrics = PerformanceMetrics()
        self.thread_pool = None
        self.process_pool = None
        self._optimization_cache = {}
        self._lock = threading.RLock()
        
        # Performance monitoring
        self._active_tasks = defaultdict(int)
        self._task_history = []
        self._bottlenecks = []
        
        # Initialize pools
        self._initialize_pools()
        
        logger.info(f"Performance optimizer initialized (threads={self.config.max_threads}, processes={self.config.max_processes})")
    
    def _initialize_pools(self):
        """Initialize thread and process pools."""
        try:
            if self.config.enable_thread_pool:
                self.thread_pool = ThreadPoolExecutor(
                    max_workers=self.config.max_threads,
                    thread_name_prefix="perf_thread"
                )
                logger.info(f"Thread pool initialized with {self.config.max_threads} workers")
            
            if self.config.enable_process_pool:
                self.process_pool = ProcessPoolExecutor(
                    max_workers=self.config.max_processes
                )
                logger.info(f"Process pool initialized with {self.config.max_processes} workers")
                
        except Exception as e:
            logger.error(f"Failed to initialize pools: {e}")
    
    @contextmanager
    def performance_context(self, task_name: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        with self._lock:
            self._active_tasks[task_name] += 1
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            with self._lock:
                self._active_tasks[task_name] -= 1
                self._task_history.append({
                    'task': task_name,
                    'execution_time': execution_time,
                    'memory_delta': memory_delta,
                    'timestamp': start_time
                })
                
                # Update global metrics
                self.metrics.execution_time = execution_time
                self.metrics.memory_usage_mb = memory_delta
    
    def optimize(self, strategy: str = "auto"):
        """Decorator for function optimization.
        
        Parameters
        ----------
        strategy : str, optional
            Optimization strategy ('auto', 'parallel', 'vectorize', 'cache')
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Determine optimization strategy
                if strategy == "auto":
                    opt_strategy = self._determine_strategy(func, args, kwargs)
                else:
                    opt_strategy = strategy
                
                with self.performance_context(func.__name__):
                    return self._apply_optimization(func, opt_strategy, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def _determine_strategy(self, func: Callable, args: Tuple, kwargs: Dict) -> str:
        """Automatically determine best optimization strategy."""
        # Simple heuristics for strategy selection
        total_data_size = sum(
            len(arg) if hasattr(arg, '__len__') else 1 
            for arg in args if not callable(arg)
        )
        
        if total_data_size > 10000:
            return "parallel"
        elif total_data_size > 1000:
            return "vectorize"
        else:
            return "cache"
    
    def _apply_optimization(self, func: Callable, strategy: str, *args, **kwargs):
        """Apply the specified optimization strategy."""
        if strategy == "parallel":
            return self._parallel_execution(func, *args, **kwargs)
        elif strategy == "vectorize":
            return self._vectorized_execution(func, *args, **kwargs)
        elif strategy == "cache":
            return self._cached_execution(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _parallel_execution(self, func: Callable, *args, **kwargs):
        """Execute function with parallel processing."""
        if not self.thread_pool:
            return func(*args, **kwargs)
        
        # Check if we can parallelize the data
        if args and hasattr(args[0], '__iter__') and len(args[0]) > self.config.batch_size:
            data = args[0]
            batch_size = max(1, len(data) // self.config.max_threads)
            batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
            
            try:
                # Submit batch jobs
                futures = []
                for batch in batches:
                    future = self.thread_pool.submit(func, batch, *args[1:], **kwargs)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in futures:
                    result = future.result(timeout=self.config.thread_pool_timeout)
                    results.extend(result if hasattr(result, '__iter__') else [result])
                
                return results
                
            except Exception as e:
                logger.warning(f"Parallel execution failed, falling back to serial: {e}")
                return func(*args, **kwargs)
        
        return func(*args, **kwargs)
    
    def _vectorized_execution(self, func: Callable, *args, **kwargs):
        """Execute function with vectorization if possible."""
        if not self.config.enable_vectorization:
            return func(*args, **kwargs)
        
        # Simple vectorization for numpy-compatible functions
        try:
            if args and hasattr(args[0], '__iter__'):
                import numpy as np
                data = np.asarray(args[0])
                
                # Try to apply function vectorized
                if hasattr(np, func.__name__):
                    np_func = getattr(np, func.__name__)
                    return np_func(data, *args[1:], **kwargs)
                
        except Exception as e:
            logger.debug(f"Vectorization failed: {e}")
        
        return func(*args, **kwargs)
    
    def _cached_execution(self, func: Callable, *args, **kwargs):
        """Execute function with result caching."""
        # Create cache key
        cache_key = self._create_cache_key(func, args, kwargs)
        
        with self._lock:
            if cache_key in self._optimization_cache:
                self.metrics.cache_hit_rate += 1
                return self._optimization_cache[cache_key]
        
        # Execute and cache result
        result = func(*args, **kwargs)
        
        with self._lock:
            self._optimization_cache[cache_key] = result
        
        return result
    
    def _create_cache_key(self, func: Callable, args: Tuple, kwargs: Dict) -> str:
        """Create a cache key for function call."""
        try:
            # Simple hash-based key
            key_parts = [func.__name__]
            
            for arg in args:
                if hasattr(arg, '__hash__'):
                    key_parts.append(str(hash(arg)))
                else:
                    key_parts.append(str(arg)[:100])  # Truncate long args
            
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={hash(v) if hasattr(v, '__hash__') else str(v)[:50]}")
            
            return "_".join(key_parts)
            
        except Exception:
            # Fallback to simple string representation
            return f"{func.__name__}_{len(args)}_{len(kwargs)}"
    
    def parallel_map(self, func: Callable, iterable, chunk_size: Optional[int] = None, use_processes: bool = False):
        """Parallel map function.
        
        Parameters
        ----------
        func : Callable
            Function to apply to each element
        iterable : Iterable
            Data to process
        chunk_size : int, optional
            Chunk size for processing
        use_processes : bool, optional
            Use process pool instead of thread pool
        """
        pool = self.process_pool if use_processes else self.thread_pool
        
        if not pool:
            # Fallback to sequential processing
            return list(map(func, iterable))
        
        try:
            data_list = list(iterable)
            
            if chunk_size is None:
                worker_count = self.config.max_processes if use_processes else self.config.max_threads  
                chunk_size = max(1, len(data_list) // worker_count)
            
            # Create chunks
            chunks = [data_list[i:i+chunk_size] for i in range(0, len(data_list), chunk_size)]
            
            # Process chunks in parallel
            def process_chunk(chunk):
                return [func(item) for item in chunk]
            
            timeout = self.config.process_pool_timeout if use_processes else self.config.thread_pool_timeout
            
            futures = [pool.submit(process_chunk, chunk) for chunk in chunks]
            results = []
            
            for future in futures:
                chunk_result = future.result(timeout=timeout)
                results.extend(chunk_result)
            
            return results
            
        except Exception as e:
            logger.warning(f"Parallel map failed, falling back to sequential: {e}")
            return list(map(func, iterable))
    
    def optimize_memory_usage(self, target_mb: Optional[float] = None):
        """Optimize memory usage by clearing caches and managing pools."""
        target = target_mb or self.config.memory_limit_mb
        current_memory = self._get_memory_usage()
        
        if current_memory > target:
            logger.info(f"Memory usage {current_memory:.1f}MB exceeds target {target:.1f}MB, optimizing...")
            
            # Clear optimization cache
            cache_size = len(self._optimization_cache)
            self._optimization_cache.clear()
            logger.info(f"Cleared {cache_size} cached results")
            
            # Trim task history
            if len(self._task_history) > 1000:
                self._task_history = self._task_history[-500:]
                logger.info("Trimmed task history")
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            new_memory = self._get_memory_usage()
            logger.info(f"Memory optimized: {current_memory:.1f}MB -> {new_memory:.1f}MB")
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks from task history."""
        if len(self._task_history) < 10:
            return []
        
        # Analyze task performance
        task_stats = defaultdict(lambda: {'times': [], 'memory': [], 'count': 0})
        
        for task in self._task_history[-1000:]:  # Last 1000 tasks
            task_name = task['task']
            task_stats[task_name]['times'].append(task['execution_time'])
            task_stats[task_name]['memory'].append(task['memory_delta'])
            task_stats[task_name]['count'] += 1
        
        bottlenecks = []
        
        for task_name, stats in task_stats.items():
            if stats['count'] < 5:  # Skip tasks with few samples
                continue
            
            times = np.array(stats['times'])
            memory = np.array(stats['memory'])
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_memory = np.mean(memory)
            
            # Detect slow tasks
            if avg_time > 1.0:  # Slower than 1 second
                bottlenecks.append({
                    'type': 'slow_execution',
                    'task': task_name,
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'severity': 'high' if avg_time > 5.0 else 'medium'
                })
            
            # Detect memory-intensive tasks
            if avg_memory > 100.0:  # More than 100MB
                bottlenecks.append({
                    'type': 'high_memory',
                    'task': task_name,
                    'avg_memory': avg_memory,
                    'severity': 'high' if avg_memory > 500.0 else 'medium'
                })
            
            # Detect high variance tasks
            if std_time > avg_time * 0.5:  # High variance
                bottlenecks.append({
                    'type': 'inconsistent_performance',
                    'task': task_name,
                    'avg_time': avg_time,
                    'variance': std_time,
                    'severity': 'medium'
                })
        
        # Sort by severity
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        bottlenecks.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)
        
        self._bottlenecks = bottlenecks
        
        if bottlenecks:
            logger.warning(f"Detected {len(bottlenecks)} performance bottlenecks")
            for bottleneck in bottlenecks[:5]:  # Log top 5
                logger.warning(f"  {bottleneck['type']}: {bottleneck['task']} ({bottleneck['severity']})")
        
        return bottlenecks
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            # Update metrics
            self.metrics.thread_count = len(self._active_tasks)
            self.metrics.cache_hit_rate = len(self._optimization_cache)
            
            if self._task_history:
                recent_tasks = self._task_history[-100:]  # Last 100 tasks
                avg_time = np.mean([t['execution_time'] for t in recent_tasks])
                self.metrics.throughput = len(recent_tasks) / max(avg_time * len(recent_tasks), 1.0)
            
            # Add bottleneck analysis
            self.metrics.bottleneck_analysis = {
                'bottleneck_count': len(self._bottlenecks),
                'high_severity': len([b for b in self._bottlenecks if b['severity'] == 'high']),
                'medium_severity': len([b for b in self._bottlenecks if b['severity'] == 'medium'])
            }
        
        return self.metrics
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        bottlenecks = self.detect_bottlenecks()
        metrics = self.get_metrics()
        
        # Task analysis
        task_summary = defaultdict(lambda: {'count': 0, 'total_time': 0.0, 'avg_time': 0.0})
        
        for task in self._task_history[-1000:]:
            task_name = task['task']
            task_summary[task_name]['count'] += 1
            task_summary[task_name]['total_time'] += task['execution_time']
        
        for task_name, data in task_summary.items():
            data['avg_time'] = data['total_time'] / data['count']
        
        # Top tasks by time
        top_tasks = sorted(
            task_summary.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:10]
        
        report = {
            'summary': {
                'total_tasks': len(self._task_history),
                'unique_tasks': len(task_summary),
                'active_tasks': len(self._active_tasks),
                'cache_entries': len(self._optimization_cache),
                'bottlenecks_detected': len(bottlenecks)
            },
            'performance_metrics': {
                'avg_execution_time': metrics.execution_time,
                'memory_usage_mb': metrics.memory_usage_mb,
                'throughput': metrics.throughput,
                'cache_hit_rate': metrics.cache_hit_rate
            },
            'top_tasks_by_time': [(name, data) for name, data in top_tasks],
            'bottlenecks': bottlenecks[:10],  # Top 10 bottlenecks
            'recommendations': self._generate_recommendations(bottlenecks, task_summary),
            'configuration': {
                'max_threads': self.config.max_threads,
                'max_processes': self.config.max_processes,
                'batch_size': self.config.batch_size,
                'memory_limit_mb': self.config.memory_limit_mb
            }
        }
        
        return report
    
    def _generate_recommendations(self, bottlenecks: List[Dict], task_summary: Dict) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze bottlenecks
        slow_tasks = [b for b in bottlenecks if b['type'] == 'slow_execution']
        memory_tasks = [b for b in bottlenecks if b['type'] == 'high_memory']
        
        if slow_tasks:
            recommendations.append(
                f"Consider parallelizing {len(slow_tasks)} slow tasks: "
                f"{', '.join([t['task'] for t in slow_tasks[:3]])}"
            )
        
        if memory_tasks:
            recommendations.append(
                f"Optimize memory usage for {len(memory_tasks)} memory-intensive tasks"
            )
        
        # Thread pool utilization
        if self.config.max_threads < mp.cpu_count():
            recommendations.append(
                f"Consider increasing thread pool size from {self.config.max_threads} to {mp.cpu_count()}"
            )
        
        # Cache utilization
        if len(self._optimization_cache) < 100:
            recommendations.append("Cache is underutilized - consider enabling caching for more functions")
        
        # Batch size optimization
        large_tasks = [name for name, data in task_summary.items() if data['count'] > 1000]
        if large_tasks:
            recommendations.append(f"Consider increasing batch size for high-frequency tasks: {', '.join(large_tasks[:3])}")
        
        return recommendations
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def shutdown(self):
        """Shutdown optimizer and clean up resources."""
        logger.info("Shutting down performance optimizer...")
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            logger.info("Thread pool shut down")
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            logger.info("Process pool shut down")
        
        with self._lock:
            self._optimization_cache.clear()
            self._task_history.clear()
            self._bottlenecks.clear()
        
        logger.info("Performance optimizer shut down successfully")
    
    def __del__(self):
        """Cleanup when optimizer is destroyed."""
        try:
            self.shutdown()
        except:
            pass


# Global optimizer instance
_global_optimizer = None


def get_global_optimizer() -> PerformanceOptimizer:
    """Get global optimizer instance.
    
    Returns
    -------
    PerformanceOptimizer
        Global optimizer instance
    """
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def optimize(strategy: str = "auto"):
    """Global optimization decorator.
    
    Parameters
    ----------
    strategy : str, optional
        Optimization strategy to use
        
    Returns
    -------
    Callable
        Decorated function with optimization
    """
    return get_global_optimizer().optimize(strategy=strategy)


def parallel_map(func: Callable, iterable, chunk_size: Optional[int] = None, use_processes: bool = False):
    """Global parallel map function.
    
    Parameters
    ----------
    func : Callable
        Function to apply
    iterable : Iterable
        Data to process
    chunk_size : int, optional
        Chunk size for processing
    use_processes : bool, optional
        Use process pool instead of thread pool
        
    Returns
    -------
    List
        Results from parallel processing
    """
    return get_global_optimizer().parallel_map(func, iterable, chunk_size, use_processes)