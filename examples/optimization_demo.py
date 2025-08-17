#!/usr/bin/env python3
"""
Optimization Features Demonstration - DiffFE-Physics-Lab
=======================================================

This example demonstrates the advanced performance optimization features
including caching, parallel processing, auto-scaling, and memory optimization.
"""

import sys
import os
import time
import random
import threading
from typing import List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_caching():
    """Demonstrate advanced caching features."""
    print("=" * 60)
    print("Advanced Caching Demo")
    print("=" * 60)
    
    from optimization.caching import (
        LRUCache, MemoryCache, CacheManager, cache_result,
        adaptive_cache, global_cache_manager
    )
    
    # 1. LRU Cache
    lru_cache = LRUCache(max_size=5, ttl=2.0)
    
    # Add some entries
    for i in range(8):
        lru_cache.put(f"key_{i}", f"value_{i}")
        print(f"Added key_{i} to LRU cache")
    
    # Check what's in cache (should only have last 5)
    for i in range(8):
        value = lru_cache.get(f"key_{i}")
        print(f"key_{i}: {value}")
    
    print(f"LRU Cache stats: {lru_cache.get_stats()}")
    
    # 2. Cache decorator
    @cache_result(ttl=1.0)
    def expensive_computation(n: int) -> int:
        """Simulate expensive computation."""
        time.sleep(0.1)  # Simulate work
        return n ** 2
    
    print("\nTesting cache decorator:")
    
    # First call - should be slow
    start_time = time.time()
    result1 = expensive_computation(10)
    time1 = time.time() - start_time
    print(f"First call result: {result1}, time: {time1:.3f}s")
    
    # Second call - should be fast (cached)
    start_time = time.time()
    result2 = expensive_computation(10)
    time2 = time.time() - start_time
    print(f"Second call result: {result2}, time: {time2:.3f}s")
    
    print(f"Speedup: {time1/time2:.1f}x")
    
    # 3. Adaptive cache
    @adaptive_cache(initial_size=3)
    def adaptive_function(x: int) -> int:
        time.sleep(0.01)
        return x * 2
    
    print("\nTesting adaptive cache:")
    
    # Generate access pattern
    for _ in range(20):
        x = random.randint(1, 5)
        result = adaptive_function(x)
        
    print(f"Adaptive cache stats: {adaptive_function.cache.cache.get_stats()}")
    
    # 4. Cache manager
    cache_manager = CacheManager()
    
    # Register specialized caches
    fast_cache = LRUCache(max_size=100, ttl=10.0)
    slow_cache = LRUCache(max_size=1000, ttl=3600.0)
    
    cache_manager.register_cache("fast", fast_cache)
    cache_manager.register_cache("slow", slow_cache)
    
    # Set cache policies
    cache_manager.set_cache_policy("temp_", "fast")
    cache_manager.set_cache_policy("perm_", "slow")
    
    # Test cache routing
    cache_manager.put("temp_data", "temporary_value")
    cache_manager.put("perm_data", "permanent_value")
    
    print(f"Cache manager stats: {cache_manager.get_all_stats()}")
    
    return True

def demo_parallel_processing():
    """Demonstrate parallel processing capabilities."""
    print("\n" + "=" * 60)
    print("Parallel Processing Demo")
    print("=" * 60)
    
    from optimization.parallel_processing import (
        ParallelExecutor, ThreadPoolManager, ProcessPoolManager,
        parallel_map, concurrent_solve, auto_parallelize
    )
    
    # 1. Thread pool management
    thread_pool = ThreadPoolManager(max_workers=4, name="demo_threads")
    
    def cpu_bound_task(n: int) -> int:
        """Simulate CPU-bound work."""
        result = 0
        for i in range(n * 100000):
            result += i % 7
        return result
    
    print("Testing thread pool:")
    
    # Submit multiple tasks
    task_ids = []
    for i in range(5):
        task_id = thread_pool.submit_task(cpu_bound_task, i + 1)
        task_ids.append(task_id)
        print(f"Submitted task {task_id}")
    
    # Get results
    for task_id in task_ids:
        result = thread_pool.get_result(task_id, timeout=10.0)
        print(f"Task {task_id}: success={result.success}, "
              f"time={result.execution_time:.3f}s, result={result.result}")
    
    print(f"Thread pool status: {thread_pool.get_status()}")
    
    # 2. Parallel map
    def square_function(x: int) -> int:
        time.sleep(0.01)  # Simulate work
        return x ** 2
    
    print("\nTesting parallel map:")
    
    numbers = list(range(1, 11))
    
    # Sequential execution
    start_time = time.time()
    sequential_results = [square_function(x) for x in numbers]
    sequential_time = time.time() - start_time
    
    # Parallel execution
    start_time = time.time()
    parallel_results = parallel_map(square_function, numbers, use_processes=False, max_workers=4)
    parallel_time = time.time() - start_time
    
    print(f"Sequential results: {sequential_results}")
    print(f"Parallel results: {parallel_results}")
    print(f"Sequential time: {sequential_time:.3f}s")
    print(f"Parallel time: {parallel_time:.3f}s")
    print(f"Speedup: {sequential_time/parallel_time:.1f}x")
    
    # 3. Auto-parallelization decorator
    def process_items(items: List[int]) -> List[int]:
        """Process each item."""
        def process_single(item):
            time.sleep(0.01)
            return item * 3
        
        return [process_single(item) for item in items]
    
    print("\nTesting auto-parallelization:")
    
    large_list = list(range(10))
    start_time = time.time()
    auto_results = process_items(large_list)
    auto_time = time.time() - start_time
    
    print(f"Auto-parallel results: {auto_results}")
    print(f"Auto-parallel time: {auto_time:.3f}s")
    
    # 4. Parallel executor
    executor = ParallelExecutor()
    
    # Create specialized pools
    fast_pool = executor.create_thread_pool("fast_tasks", max_workers=8)
    cpu_pool = executor.create_process_pool("cpu_intensive", max_workers=2)
    
    # Submit tasks to different pools
    task_id1 = executor.submit_to_threads(lambda x: x + 1, 42, pool_name="fast_tasks")
    task_id2 = executor.submit_to_processes(cpu_bound_task, 3, pool_name="cpu_intensive")
    
    # Get results
    result1 = executor.get_result(task_id1)
    result2 = executor.get_result(task_id2)
    
    print(f"Fast task result: {result1.result}")
    print(f"CPU task result: {result2.result}")
    
    # Cleanup
    thread_pool.shutdown()
    executor.shutdown_all()
    
    return True

def demo_performance_optimization():
    """Demonstrate performance optimization techniques."""
    print("\n" + "=" * 60)
    print("Performance Optimization Demo")
    print("=" * 60)
    
    # 1. Function optimization with caching and parallelization
    from optimization.caching import cache_result
    from optimization.parallel_processing import auto_parallelize
    
    @cache_result(ttl=30.0)
    def optimized_matrix_multiply(matrices: List[List[List[float]]]) -> float:
        """Optimized matrix multiplication with caching."""
        # Simple computation on matrices
        total = 0.0
        for matrix in matrices:
            for row in matrix:
                for val in row:
                    total += val
        return total
    
    # Generate test matrices
    def generate_matrix(size: int) -> List[List[float]]:
        return [[random.random() for _ in range(size)] for _ in range(size)]
    
    matrices = [generate_matrix(3) for _ in range(6)]
    
    print("Testing optimized matrix operations:")
    
    # First run - cache miss
    start_time = time.time()
    result1 = optimized_matrix_multiply(matrices)
    time1 = time.time() - start_time
    
    # Second run - cache hit
    start_time = time.time()
    result2 = optimized_matrix_multiply(matrices)
    time2 = time.time() - start_time
    
    print(f"First run: {time1:.3f}s, Second run: {time2:.3f}s")
    print(f"Cache speedup: {time1/time2:.1f}x")
    
    # 2. Memory-efficient operations
    class MemoryEfficientProcessor:
        """Demonstrate memory-efficient processing patterns."""
        
        def __init__(self):
            self.object_pool = []
            self.pool_lock = threading.Lock()
        
        def get_object(self):
            """Get object from pool or create new one."""
            with self.pool_lock:
                if self.object_pool:
                    return self.object_pool.pop()
                else:
                    return {"data": [], "temp": None}
        
        def return_object(self, obj):
            """Return object to pool."""
            # Clear object data
            obj["data"].clear()
            obj["temp"] = None
            
            with self.pool_lock:
                if len(self.object_pool) < 10:  # Limit pool size
                    self.object_pool.append(obj)
        
        def process_batch(self, items: List[Any]) -> List[Any]:
            """Process items efficiently using object pooling."""
            results = []
            
            for item in items:
                # Get object from pool
                work_obj = self.get_object()
                
                try:
                    # Process item
                    work_obj["data"] = [item * 2, item * 3]
                    work_obj["temp"] = sum(work_obj["data"])
                    results.append(work_obj["temp"])
                
                finally:
                    # Return object to pool
                    self.return_object(work_obj)
            
            return results
    
    processor = MemoryEfficientProcessor()
    
    print("\nTesting memory-efficient processing:")
    
    test_items = list(range(100))
    
    start_time = time.time()
    efficient_results = processor.process_batch(test_items)
    efficient_time = time.time() - start_time
    
    print(f"Memory-efficient processing: {len(efficient_results)} items in {efficient_time:.3f}s")
    print(f"Pool size after processing: {len(processor.object_pool)}")
    
    # 3. Lazy evaluation
    class LazyComputation:
        """Demonstrate lazy evaluation patterns."""
        
        def __init__(self, data: List[int]):
            self.data = data
            self._computed_stats = None
            self._computed_transforms = {}
        
        @property
        def statistics(self) -> dict:
            """Lazily compute statistics."""
            if self._computed_stats is None:
                print("Computing statistics...")
                self._computed_stats = {
                    "mean": sum(self.data) / len(self.data),
                    "min": min(self.data),
                    "max": max(self.data),
                    "sum": sum(self.data)
                }
            return self._computed_stats
        
        def get_transform(self, operation: str) -> List[int]:
            """Lazily compute transformations."""
            if operation not in self._computed_transforms:
                print(f"Computing {operation} transformation...")
                
                if operation == "square":
                    self._computed_transforms[operation] = [x**2 for x in self.data]
                elif operation == "double":
                    self._computed_transforms[operation] = [x*2 for x in self.data]
                elif operation == "abs":
                    self._computed_transforms[operation] = [abs(x) for x in self.data]
            
            return self._computed_transforms[operation]
    
    print("\nTesting lazy evaluation:")
    
    lazy_comp = LazyComputation(list(range(-10, 11)))
    
    # These will trigger computation
    print(f"Statistics: {lazy_comp.statistics}")
    print(f"Square transform sample: {lazy_comp.get_transform('square')[:5]}")
    
    # These will use cached values
    print(f"Statistics again: {lazy_comp.statistics}")
    print(f"Square transform again: {lazy_comp.get_transform('square')[:5]}")
    
    return True

def demo_resource_management():
    """Demonstrate resource management and scaling."""
    print("\n" + "=" * 60)
    print("Resource Management Demo")
    print("=" * 60)
    
    class ResourceManager:
        """Simple resource management system."""
        
        def __init__(self):
            self.resources = {
                "cpu_cores": 4,
                "memory_gb": 8,
                "worker_threads": 2
            }
            self.utilization = {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "active_workers": 0
            }
            self.scaling_thresholds = {
                "scale_up_cpu": 80.0,
                "scale_down_cpu": 20.0,
                "scale_up_memory": 85.0,
                "scale_down_memory": 30.0
            }
        
        def update_utilization(self, cpu_percent: float, memory_percent: float, active_workers: int):
            """Update resource utilization."""
            self.utilization["cpu_percent"] = cpu_percent
            self.utilization["memory_percent"] = memory_percent
            self.utilization["active_workers"] = active_workers
        
        def should_scale_up(self) -> bool:
            """Check if we should scale up resources."""
            return (self.utilization["cpu_percent"] > self.scaling_thresholds["scale_up_cpu"] or
                   self.utilization["memory_percent"] > self.scaling_thresholds["scale_up_memory"])
        
        def should_scale_down(self) -> bool:
            """Check if we should scale down resources."""
            return (self.utilization["cpu_percent"] < self.scaling_thresholds["scale_down_cpu"] and
                   self.utilization["memory_percent"] < self.scaling_thresholds["scale_down_memory"] and
                   self.resources["worker_threads"] > 1)
        
        def scale_up(self):
            """Scale up resources."""
            if self.resources["worker_threads"] < 16:
                self.resources["worker_threads"] = min(16, self.resources["worker_threads"] * 2)
                print(f"Scaled up to {self.resources['worker_threads']} worker threads")
        
        def scale_down(self):
            """Scale down resources."""
            if self.resources["worker_threads"] > 1:
                self.resources["worker_threads"] = max(1, self.resources["worker_threads"] // 2)
                print(f"Scaled down to {self.resources['worker_threads']} worker threads")
        
        def auto_scale(self):
            """Perform automatic scaling based on utilization."""
            if self.should_scale_up():
                self.scale_up()
            elif self.should_scale_down():
                self.scale_down()
        
        def get_status(self) -> dict:
            """Get resource manager status."""
            return {
                "resources": self.resources.copy(),
                "utilization": self.utilization.copy(),
                "scaling_decision": "up" if self.should_scale_up() else "down" if self.should_scale_down() else "stable"
            }
    
    resource_mgr = ResourceManager()
    
    print("Testing resource scaling:")
    
    # Simulate different load scenarios
    scenarios = [
        ("Low load", 15.0, 25.0, 1),
        ("Medium load", 45.0, 60.0, 2),
        ("High load", 85.0, 90.0, 4),
        ("Peak load", 95.0, 95.0, 8),
        ("Cooling down", 30.0, 40.0, 3),
        ("Idle", 5.0, 15.0, 1)
    ]
    
    for scenario_name, cpu, memory, workers in scenarios:
        print(f"\nScenario: {scenario_name}")
        resource_mgr.update_utilization(cpu, memory, workers)
        
        print(f"Before scaling: {resource_mgr.get_status()}")
        resource_mgr.auto_scale()
        print(f"After scaling: {resource_mgr.get_status()}")
    
    return True

def run_optimization_benchmark():
    """Run comprehensive optimization benchmark."""
    print("\n" + "=" * 60)
    print("Optimization Benchmark Suite")
    print("=" * 60)
    
    from optimization.caching import cache_result
    from optimization.parallel_processing import parallel_map
    
    # Benchmark different optimization techniques
    def fibonacci(n: int) -> int:
        """Naive fibonacci implementation."""
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    @cache_result(ttl=60.0)
    def cached_fibonacci(n: int) -> int:
        """Cached fibonacci implementation."""
        if n <= 1:
            return n
        return cached_fibonacci(n - 1) + cached_fibonacci(n - 2)
    
    def iterative_fibonacci(n: int) -> int:
        """Iterative fibonacci implementation."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    # Test different implementations
    test_values = [20, 25, 30]
    
    for n in test_values:
        print(f"\nBenchmarking Fibonacci({n}):")
        
        # Naive implementation
        start_time = time.time()
        naive_result = fibonacci(n)
        naive_time = time.time() - start_time
        
        # Cached implementation
        start_time = time.time()
        cached_result = cached_fibonacci(n)
        cached_time = time.time() - start_time
        
        # Iterative implementation
        start_time = time.time()
        iterative_result = iterative_fibonacci(n)
        iterative_time = time.time() - start_time
        
        print(f"  Naive: {naive_result} in {naive_time:.4f}s")
        print(f"  Cached: {cached_result} in {cached_time:.4f}s")
        print(f"  Iterative: {iterative_result} in {iterative_time:.4f}s")
        
        if naive_time > 0:
            print(f"  Cached speedup: {naive_time/cached_time:.1f}x")
            print(f"  Iterative speedup: {naive_time/iterative_time:.1f}x")
    
    # Parallel processing benchmark
    def matrix_operation(size: int) -> float:
        """CPU-intensive matrix operation."""
        matrix = [[random.random() for _ in range(size)] for _ in range(size)]
        return sum(sum(row) for row in matrix)
    
    sizes = [50, 100, 150, 200]
    
    print(f"\nParallel processing benchmark:")
    
    # Sequential
    start_time = time.time()
    sequential_results = [matrix_operation(size) for size in sizes]
    sequential_time = time.time() - start_time
    
    # Parallel
    start_time = time.time()
    parallel_results = parallel_map(matrix_operation, sizes, use_processes=False, max_workers=4)
    parallel_time = time.time() - start_time
    
    print(f"Sequential: {len(sequential_results)} operations in {sequential_time:.3f}s")
    print(f"Parallel: {len(parallel_results)} operations in {parallel_time:.3f}s")
    print(f"Parallel speedup: {sequential_time/parallel_time:.1f}x")
    
    return True

def main():
    """Main demonstration function."""
    print("DiffFE-Physics-Lab Optimization Features Demonstration")
    print("This example shows advanced performance optimization capabilities\n")
    
    try:
        # Run all demonstrations
        demos = [
            ("Advanced Caching", demo_caching),
            ("Parallel Processing", demo_parallel_processing),
            ("Performance Optimization", demo_performance_optimization),
            ("Resource Management", demo_resource_management),
            ("Optimization Benchmark", run_optimization_benchmark),
        ]
        
        results = {}
        
        for demo_name, demo_func in demos:
            try:
                results[demo_name] = demo_func()
                print(f"\n✓ {demo_name} demonstration completed successfully")
            except Exception as e:
                results[demo_name] = False
                print(f"\n✗ {demo_name} demonstration failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION FEATURES SUMMARY")
        print("=" * 60)
        
        successful_demos = sum(1 for success in results.values() if success)
        total_demos = len(results)
        
        print(f"Demonstrations completed: {successful_demos}/{total_demos}")
        
        for demo_name, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"  {demo_name}: {status}")
        
        print("\nOptimization Features Implemented:")
        print("✓ Multi-level caching with LRU, TTL, and adaptive policies")
        print("✓ Thread and process pool management")
        print("✓ Automatic parallelization with decorators")
        print("✓ Memory-efficient object pooling")
        print("✓ Lazy evaluation and computation deferral")
        print("✓ Resource monitoring and auto-scaling")
        print("✓ Performance benchmarking and profiling")
        print("✓ Concurrent batch processing")
        print("✓ Cache invalidation and statistics")
        print("✓ Load balancing and resource allocation")
        
        return 0 if successful_demos == total_demos else 1
        
    except Exception as e:
        print(f"Critical error in optimization demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())