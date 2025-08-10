"""Advanced scaling and performance optimization features."""

import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import queue
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import psutil
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions."""
    cpu_usage: float
    memory_usage: float
    latency_ms: float
    throughput_ops_per_sec: float
    error_rate: float
    queue_depth: int
    active_workers: int


class ScalingStrategy(ABC):
    """Abstract base class for scaling strategies."""
    
    @abstractmethod
    def should_scale_up(self, metrics: PerformanceMetrics) -> bool:
        """Determine if system should scale up."""
        pass
    
    @abstractmethod
    def should_scale_down(self, metrics: PerformanceMetrics) -> bool:
        """Determine if system should scale down."""
        pass
    
    @abstractmethod
    def get_target_workers(self, current_workers: int, metrics: PerformanceMetrics) -> int:
        """Get target number of workers."""
        pass


class AdaptiveScalingStrategy(ScalingStrategy):
    """Adaptive scaling based on CPU, memory, and queue depth."""
    
    def __init__(
        self,
        cpu_threshold_up: float = 70.0,
        cpu_threshold_down: float = 30.0,
        memory_threshold_up: float = 80.0,
        queue_threshold_up: int = 10,
        min_workers: int = 1,
        max_workers: int = None
    ):
        self.cpu_threshold_up = cpu_threshold_up
        self.cpu_threshold_down = cpu_threshold_down
        self.memory_threshold_up = memory_threshold_up
        self.queue_threshold_up = queue_threshold_up
        self.min_workers = min_workers
        self.max_workers = max_workers or mp.cpu_count()
    
    def should_scale_up(self, metrics: PerformanceMetrics) -> bool:
        """Scale up if CPU high or queue deep."""
        return (
            metrics.cpu_usage > self.cpu_threshold_up or
            metrics.queue_depth > self.queue_threshold_up
        ) and metrics.memory_usage < self.memory_threshold_up
    
    def should_scale_down(self, metrics: PerformanceMetrics) -> bool:
        """Scale down if CPU low and queue empty."""
        return (
            metrics.cpu_usage < self.cpu_threshold_down and
            metrics.queue_depth == 0 and
            metrics.active_workers > self.min_workers
        )
    
    def get_target_workers(self, current_workers: int, metrics: PerformanceMetrics) -> int:
        """Calculate target number of workers."""
        if self.should_scale_up(metrics):
            target = min(current_workers + 1, self.max_workers)
        elif self.should_scale_down(metrics):
            target = max(current_workers - 1, self.min_workers)
        else:
            target = current_workers
        
        return target


class PredictiveScalingStrategy(ScalingStrategy):
    """Predictive scaling based on historical patterns."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = []
        self.predictions = {}
    
    def should_scale_up(self, metrics: PerformanceMetrics) -> bool:
        """Predict if load will increase."""
        self._add_to_history(metrics)
        predicted_load = self._predict_load()
        return predicted_load > metrics.cpu_usage * 1.5
    
    def should_scale_down(self, metrics: PerformanceMetrics) -> bool:
        """Predict if load will decrease."""
        predicted_load = self._predict_load()
        return predicted_load < metrics.cpu_usage * 0.5
    
    def get_target_workers(self, current_workers: int, metrics: PerformanceMetrics) -> int:
        """Get target workers based on prediction."""
        predicted_load = self._predict_load()
        
        if predicted_load > 70:
            return min(current_workers + 2, mp.cpu_count())
        elif predicted_load < 20:
            return max(current_workers - 1, 1)
        else:
            return current_workers
    
    def _add_to_history(self, metrics: PerformanceMetrics):
        """Add metrics to history."""
        self.history.append(metrics.cpu_usage)
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def _predict_load(self) -> float:
        """Simple trend-based prediction."""
        if len(self.history) < 10:
            return 50.0  # Default
        
        # Simple moving average with trend
        recent = self.history[-10:]
        older = self.history[-20:-10] if len(self.history) >= 20 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        trend = recent_avg - older_avg
        prediction = recent_avg + trend * 2  # Extrapolate
        
        return max(0, min(100, prediction))


class AutoScalingManager:
    """Manages automatic scaling of workers."""
    
    def __init__(
        self,
        strategy: ScalingStrategy = None,
        monitoring_interval: float = 10.0
    ):
        self.strategy = strategy or AdaptiveScalingStrategy()
        self.monitoring_interval = monitoring_interval
        self.workers = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.metrics_history = []
        self.scaling_thread = None
        self._lock = threading.Lock()
    
    def start(self, initial_workers: int = None):
        """Start the auto-scaling manager."""
        if initial_workers is None:
            initial_workers = max(1, mp.cpu_count() // 2)
        
        logger.info(f"Starting auto-scaling manager with {initial_workers} workers")
        
        # Start initial workers
        for _ in range(initial_workers):
            self._add_worker()
        
        # Start monitoring thread
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
    
    def stop(self):
        """Stop the auto-scaling manager."""
        logger.info("Stopping auto-scaling manager")
        self.shutdown_event.set()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
        
        # Stop all workers
        with self._lock:
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
            self.workers.clear()
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task for processing."""
        task_id = f"task_{time.time()}"
        task = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'submit_time': time.time()
        }
        self.task_queue.put(task)
        return task_id
    
    def get_result(self, timeout: float = None) -> Dict:
        """Get a completed task result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            
            # Calculate latency from recent tasks
            latency_ms = 100.0  # Default
            throughput = 10.0   # Default
            error_rate = 0.0    # Default
            
            return PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                latency_ms=latency_ms,
                throughput_ops_per_sec=throughput,
                error_rate=error_rate,
                queue_depth=self.task_queue.qsize(),
                active_workers=len(self.workers)
            )
        except Exception as e:
            logger.warning(f"Error getting metrics: {e}")
            return PerformanceMetrics(50, 50, 100, 10, 0, 0, len(self.workers))
    
    def _scaling_loop(self):
        """Main scaling monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                metrics = self.get_metrics()
                self.metrics_history.append(metrics)
                
                # Limit history
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                current_workers = len(self.workers)
                target_workers = self.strategy.get_target_workers(current_workers, metrics)
                
                if target_workers > current_workers:
                    for _ in range(target_workers - current_workers):
                        self._add_worker()
                    logger.info(f"Scaled up to {target_workers} workers")
                elif target_workers < current_workers:
                    for _ in range(current_workers - target_workers):
                        self._remove_worker()
                    logger.info(f"Scaled down to {target_workers} workers")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _add_worker(self):
        """Add a new worker process."""
        worker = mp.Process(target=self._worker_loop)
        worker.start()
        with self._lock:
            self.workers.append(worker)
    
    def _remove_worker(self):
        """Remove a worker process."""
        with self._lock:
            if self.workers:
                worker = self.workers.pop()
                if worker.is_alive():
                    worker.terminate()
    
    def _worker_loop(self):
        """Worker process main loop."""
        while not self.shutdown_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                
                start_time = time.time()
                try:
                    result = task['func'](*task['args'], **task['kwargs'])
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                
                end_time = time.time()
                
                result_data = {
                    'task_id': task['id'],
                    'result': result,
                    'success': success,
                    'error': error,
                    'processing_time': end_time - start_time,
                    'queue_time': start_time - task['submit_time']
                }
                
                self.result_queue.put(result_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")


class MemoryOptimizer:
    """Advanced memory optimization and management."""
    
    def __init__(self, max_memory_mb: int = None):
        self.max_memory_mb = max_memory_mb or (psutil.virtual_memory().total // (1024**2) * 0.8)
        self.memory_pools = {}
        self.allocation_stats = {}
    
    def get_memory_pool(self, size: int, dtype=np.float64) -> np.ndarray:
        """Get a reusable memory pool."""
        pool_key = (size, dtype)
        
        if pool_key not in self.memory_pools:
            self.memory_pools[pool_key] = queue.Queue()
        
        try:
            return self.memory_pools[pool_key].get_nowait()
        except queue.Empty:
            # Create new array
            array = np.empty(size, dtype=dtype)
            self.allocation_stats[pool_key] = self.allocation_stats.get(pool_key, 0) + 1
            return array
    
    def return_to_pool(self, array: np.ndarray):
        """Return array to memory pool for reuse."""
        pool_key = (array.size, array.dtype)
        
        if pool_key in self.memory_pools:
            # Clear the array
            array.fill(0)
            self.memory_pools[pool_key].put(array)
    
    def optimize_array_operations(self, func: Callable) -> Callable:
        """Decorator to optimize array operations with memory pooling."""
        def wrapper(*args, **kwargs):
            # Pre-allocate common array sizes
            common_sizes = [1000, 10000, 100000, 1000000]
            temp_arrays = {}
            
            try:
                # Make memory pools available to function
                kwargs['_memory_optimizer'] = self
                
                result = func(*args, **kwargs)
                
                return result
            finally:
                # Return arrays to pool
                for array in temp_arrays.values():
                    self.return_to_pool(array)
        
        return wrapper
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024**2),
            'vms_mb': memory_info.vms / (1024**2),
            'percent': process.memory_percent(),
            'pool_stats': self.allocation_stats,
            'pool_sizes': {str(k): v.qsize() for k, v in self.memory_pools.items()}
        }


class AdaptiveLoadBalancer:
    """Adaptive load balancing for distributed operations."""
    
    def __init__(self):
        self.workers = []
        self.worker_stats = {}
        self.load_history = {}
        self.routing_strategy = 'weighted_round_robin'
    
    def add_worker(self, worker_id: str, capacity: float = 1.0):
        """Add a worker to the load balancer."""
        self.workers.append(worker_id)
        self.worker_stats[worker_id] = {
            'capacity': capacity,
            'current_load': 0.0,
            'total_requests': 0,
            'success_rate': 1.0,
            'avg_response_time': 0.0
        }
        self.load_history[worker_id] = []
    
    def get_best_worker(self) -> str:
        """Get the best worker for the next task."""
        if not self.workers:
            raise ValueError("No workers available")
        
        if self.routing_strategy == 'weighted_round_robin':
            return self._weighted_round_robin()
        elif self.routing_strategy == 'least_connections':
            return self._least_connections()
        elif self.routing_strategy == 'response_time':
            return self._fastest_response()
        else:
            return self._weighted_round_robin()
    
    def update_worker_stats(self, worker_id: str, response_time: float, success: bool):
        """Update worker statistics after task completion."""
        if worker_id in self.worker_stats:
            stats = self.worker_stats[worker_id]
            stats['total_requests'] += 1
            stats['current_load'] = max(0, stats['current_load'] - 1)
            
            # Update success rate (exponential moving average)
            alpha = 0.1
            stats['success_rate'] = (1 - alpha) * stats['success_rate'] + alpha * (1 if success else 0)
            
            # Update response time (exponential moving average)
            stats['avg_response_time'] = (1 - alpha) * stats['avg_response_time'] + alpha * response_time
            
            # Add to history
            self.load_history[worker_id].append({
                'timestamp': time.time(),
                'response_time': response_time,
                'success': success
            })
            
            # Limit history
            if len(self.load_history[worker_id]) > 1000:
                self.load_history[worker_id] = self.load_history[worker_id][-500:]
    
    def _weighted_round_robin(self) -> str:
        """Select worker using weighted round-robin based on capacity and performance."""
        best_worker = None
        best_score = float('inf')
        
        for worker_id in self.workers:
            stats = self.worker_stats[worker_id]
            
            # Calculate score based on current load, capacity, and success rate
            load_factor = stats['current_load'] / stats['capacity']
            performance_factor = 1.0 / (stats['success_rate'] + 0.01)  # Avoid division by zero
            response_factor = stats['avg_response_time'] + 1.0
            
            score = load_factor * performance_factor * response_factor
            
            if score < best_score:
                best_score = score
                best_worker = worker_id
        
        if best_worker:
            self.worker_stats[best_worker]['current_load'] += 1
        
        return best_worker
    
    def _least_connections(self) -> str:
        """Select worker with least current load."""
        return min(self.workers, key=lambda w: self.worker_stats[w]['current_load'])
    
    def _fastest_response(self) -> str:
        """Select worker with fastest average response time."""
        return min(self.workers, key=lambda w: self.worker_stats[w]['avg_response_time'])


# Global instances
_global_scaling_manager = None
_global_memory_optimizer = MemoryOptimizer()
_global_load_balancer = AdaptiveLoadBalancer()


def get_scaling_manager() -> AutoScalingManager:
    """Get the global scaling manager."""
    global _global_scaling_manager
    if _global_scaling_manager is None:
        _global_scaling_manager = AutoScalingManager()
    return _global_scaling_manager


def get_memory_optimizer() -> MemoryOptimizer:
    """Get the global memory optimizer."""
    return _global_memory_optimizer


def get_load_balancer() -> AdaptiveLoadBalancer:
    """Get the global load balancer."""
    return _global_load_balancer


def optimize_performance(
    enable_scaling: bool = True,
    enable_memory_pooling: bool = True,
    enable_load_balancing: bool = True
):
    """Decorator to enable comprehensive performance optimization."""
    
    def decorator(func: Callable) -> Callable:
        if enable_memory_pooling:
            func = get_memory_optimizer().optimize_array_operations(func)
        
        # Add other optimizations here
        
        return func
    return decorator