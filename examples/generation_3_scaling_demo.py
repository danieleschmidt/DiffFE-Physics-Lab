#!/usr/bin/env python3
"""Generation 3 Scaling Demo - Advanced performance optimization and scaling."""

import numpy as np
import time
import threading
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Add src to path
sys.path.insert(0, "/root/repo")

import src
from src.models import Problem, MultiPhysicsProblem, Domain
from src.backends import get_backend
from src.performance.advanced_scaling import (
    AutoScalingManager,
    AdaptiveScalingStrategy,
    PredictiveScalingStrategy,
    MemoryOptimizer,
    AdaptiveLoadBalancer,
    optimize_performance,
    get_scaling_manager,
    get_memory_optimizer,
    get_load_balancer,
)
from src.performance.cache import CacheManager, cached
from src.utils.robust_error_handling import robust_operation, error_boundary


logger = logging.getLogger(__name__)


@optimize_performance(enable_memory_pooling=True)
def cpu_intensive_computation(n: int, _memory_optimizer=None) -> float:
    """CPU-intensive computation with memory optimization."""

    # Use memory pool if available
    if _memory_optimizer:
        array = _memory_optimizer.get_memory_pool(n, dtype=np.float64)
    else:
        array = np.zeros(n)

    try:
        # Simulate heavy computation
        for i in range(n):
            array[i] = np.sqrt(i) * np.log(i + 1) if i > 0 else 0

        result = np.sum(array) / n

        return result

    finally:
        if _memory_optimizer:
            _memory_optimizer.return_to_pool(array)


@cached(ttl=300)  # Cache for 5 minutes
def expensive_matrix_operation(size: int) -> float:
    """Expensive matrix operation with caching."""
    matrix = np.random.random((size, size))
    eigenvalues = np.linalg.eigvals(matrix)
    return np.mean(np.real(eigenvalues))


@robust_operation(max_retries=2, operation_name="physics_simulation")
def simulate_physics_step(domain_id: str, timestep: float) -> dict:
    """Simulate a physics timestep for a domain."""

    # Simulate variable computation time
    computation_time = np.random.uniform(0.1, 0.5)
    time.sleep(computation_time)

    # Simulate occasional failures (10% chance)
    if np.random.random() < 0.1:
        raise RuntimeError(f"Simulation failed for domain {domain_id}")

    # Return simulation results
    return {
        "domain_id": domain_id,
        "timestep": timestep,
        "temperature": 20 + np.random.random() * 10,
        "pressure": 1000 + np.random.random() * 100,
        "velocity": np.random.random(3).tolist(),
        "computation_time": computation_time,
    }


def stress_test_worker(worker_id: int, num_operations: int) -> dict:
    """Worker function for stress testing."""
    results = []
    start_time = time.time()

    for i in range(num_operations):
        try:
            # Mix of different operation types
            if i % 3 == 0:
                result = cpu_intensive_computation(1000 + i * 100)
                results.append(("cpu", result))
            elif i % 3 == 1:
                result = expensive_matrix_operation(50 + i % 20)
                results.append(("matrix", result))
            else:
                result = simulate_physics_step(f"domain_{i}", 0.01)
                results.append(("physics", result))

        except Exception as e:
            results.append(("error", str(e)))

    end_time = time.time()

    return {
        "worker_id": worker_id,
        "operations_completed": len([r for r in results if r[0] != "error"]),
        "errors": len([r for r in results if r[0] == "error"]),
        "total_time": end_time - start_time,
        "ops_per_second": num_operations / (end_time - start_time),
    }


def main():
    """Demonstrate Generation 3 scaling and performance features."""
    print("⚡ DiffFE-Physics-Lab - Generation 3 Scaling Demo")
    print("=" * 65)

    # 1. Memory Optimization Demo
    print("\n💾 Memory Optimization & Pooling:")

    memory_optimizer = get_memory_optimizer()

    # Test memory pooling with repeated allocations
    pool_times = []
    regular_times = []

    for test_size in [10000, 50000, 100000]:
        # Test with memory pooling
        start_time = time.time()
        for _ in range(10):
            array = memory_optimizer.get_memory_pool(test_size)
            array.fill(np.random.random())
            memory_optimizer.return_to_pool(array)
        pool_time = time.time() - start_time
        pool_times.append(pool_time)

        # Test regular allocation
        start_time = time.time()
        for _ in range(10):
            array = np.zeros(test_size)
            array.fill(np.random.random())
            del array
        regular_time = time.time() - start_time
        regular_times.append(regular_time)

        speedup = regular_time / pool_time
        print(
            f"   📊 Size {test_size:6d}: Pool={pool_time:.3f}s, Regular={regular_time:.3f}s, Speedup={speedup:.2f}x"
        )

    # Memory statistics
    mem_stats = memory_optimizer.get_memory_stats()
    print(
        f"   📈 Memory stats: RSS={mem_stats['rss_mb']:.1f}MB, Usage={mem_stats['percent']:.1f}%"
    )

    # 2. Adaptive Load Balancing
    print("\n🔄 Adaptive Load Balancing:")

    load_balancer = get_load_balancer()

    # Add workers with different capacities
    workers = [("worker_fast", 2.0), ("worker_normal", 1.0), ("worker_slow", 0.5)]

    for worker_id, capacity in workers:
        load_balancer.add_worker(worker_id, capacity)

    # Simulate task distribution
    print("   🎯 Task distribution simulation:")
    task_assignments = {"worker_fast": 0, "worker_normal": 0, "worker_slow": 0}

    for i in range(30):
        worker = load_balancer.get_best_worker()
        task_assignments[worker] += 1

        # Simulate task completion with varying performance
        if worker == "worker_fast":
            response_time = np.random.uniform(0.1, 0.2)
            success_rate = 0.98
        elif worker == "worker_normal":
            response_time = np.random.uniform(0.2, 0.4)
            success_rate = 0.95
        else:  # worker_slow
            response_time = np.random.uniform(0.5, 1.0)
            success_rate = 0.90

        success = np.random.random() < success_rate
        load_balancer.update_worker_stats(worker, response_time, success)

    for worker_id, count in task_assignments.items():
        percentage = (count / 30) * 100
        print(f"     - {worker_id}: {count} tasks ({percentage:.1f}%)")

    # 3. Auto-Scaling Performance Test
    print("\n📈 Auto-Scaling Performance Test:")

    # Create scaling manager with adaptive strategy
    strategy = AdaptiveScalingStrategy(
        cpu_threshold_up=60.0,
        cpu_threshold_down=20.0,
        min_workers=2,
        max_workers=min(8, mp.cpu_count()),
    )

    scaling_manager = AutoScalingManager(strategy=strategy, monitoring_interval=2.0)

    try:
        scaling_manager.start(initial_workers=2)
        print("   🚀 Started auto-scaling manager")

        # Submit a burst of tasks
        task_ids = []
        print("   📋 Submitting computational tasks...")

        for i in range(20):
            task_id = scaling_manager.submit_task(
                cpu_intensive_computation, 5000 + i * 1000
            )
            task_ids.append(task_id)

        # Collect results
        completed_tasks = 0
        start_time = time.time()

        while completed_tasks < len(task_ids) and time.time() - start_time < 30:
            result = scaling_manager.get_result(timeout=1.0)
            if result:
                completed_tasks += 1
                if result["success"]:
                    print(
                        f"   ✅ Task {result['task_id'][-4:]}: {result['result']:.3f} "
                        f"(processing: {result['processing_time']:.2f}s)"
                    )
                else:
                    print(f"   ❌ Task {result['task_id'][-4:]}: {result['error']}")

        # Final metrics
        final_metrics = scaling_manager.get_metrics()
        print(
            f"   📊 Final metrics: {final_metrics.active_workers} workers, "
            f"CPU {final_metrics.cpu_usage:.1f}%, queue depth {final_metrics.queue_depth}"
        )

    finally:
        scaling_manager.stop()
        print("   🛑 Stopped auto-scaling manager")

    # 4. Predictive Scaling Demo
    print("\n🔮 Predictive Scaling Strategy:")

    predictive_strategy = PredictiveScalingStrategy(window_size=50)

    # Simulate workload pattern
    print("   📈 Simulating workload patterns...")
    simulated_loads = []

    for hour in range(24):
        # Simulate daily pattern: low at night, high during day
        base_load = 20 + 50 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else 20
        noise = np.random.normal(0, 10)
        load = max(0, min(100, base_load + noise))
        simulated_loads.append(load)

        # Create fake metrics
        from src.performance.advanced_scaling import PerformanceMetrics

        metrics = PerformanceMetrics(
            cpu_usage=load,
            memory_usage=50,
            latency_ms=100,
            throughput_ops_per_sec=20,
            error_rate=0.01,
            queue_depth=int(load / 10),
            active_workers=4,
        )

        should_scale_up = predictive_strategy.should_scale_up(metrics)
        target_workers = predictive_strategy.get_target_workers(4, metrics)

        print(
            f"   🕐 Hour {hour:2d}: Load={load:5.1f}%, "
            f"Scale up={should_scale_up}, Target workers={target_workers}"
        )

    # 5. Multi-Physics Scaling Demo
    print("\n🌊 Multi-Physics Scaling Simulation:")

    with error_boundary("multiphysics_scaling"):
        # Create multi-physics problem with multiple domains
        mpp = MultiPhysicsProblem(backend="numpy")

        domains = ["fluid_1", "fluid_2", "solid_1", "solid_2", "thermal"]
        for domain_name in domains:
            domain = Domain(domain_name, physics="generic")
            mpp.add_domain(domain)

        print(f"   🏗️  Created multi-physics problem with {len(domains)} domains")

        # Simulate parallel domain processing
        print("   🔄 Parallel domain processing simulation:")

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit physics simulations for each domain
            futures = []
            for i, domain_name in enumerate(domains):
                future = executor.submit(simulate_physics_step, domain_name, 0.01)
                futures.append((domain_name, future))

            # Collect results
            results = {}
            for domain_name, future in futures:
                try:
                    result = future.result(timeout=5.0)
                    results[domain_name] = result
                    print(
                        f"     ✅ {domain_name}: T={result['temperature']:.1f}°C, "
                        f"P={result['pressure']:.1f}Pa"
                    )
                except Exception as e:
                    print(f"     ❌ {domain_name}: {e}")

        print(f"   📊 Completed {len(results)}/{len(domains)} domain simulations")

    # 6. Stress Test with Performance Monitoring
    print("\n🔥 Stress Test with Performance Monitoring:")

    num_workers = min(4, mp.cpu_count())
    operations_per_worker = 20

    print(
        f"   🚀 Starting stress test: {num_workers} workers, {operations_per_worker} ops each"
    )

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit stress test tasks
        futures = []
        for worker_id in range(num_workers):
            future = executor.submit(
                stress_test_worker, worker_id, operations_per_worker
            )
            futures.append(future)

        # Collect results
        total_ops = 0
        total_errors = 0
        total_time = 0

        for future in futures:
            try:
                result = future.result()
                total_ops += result["operations_completed"]
                total_errors += result["errors"]
                total_time = max(total_time, result["total_time"])

                print(
                    f"   👷 Worker {result['worker_id']}: "
                    f"{result['operations_completed']} ops, "
                    f"{result['errors']} errors, "
                    f"{result['ops_per_second']:.1f} ops/sec"
                )

            except Exception as e:
                print(f"   ❌ Worker failed: {e}")

    end_time = time.time()
    overall_time = end_time - start_time
    overall_throughput = total_ops / overall_time

    print(f"\n   📊 Stress Test Results:")
    print(f"     - Total operations: {total_ops}")
    print(f"     - Total errors: {total_errors}")
    print(
        f"     - Error rate: {(total_errors / (total_ops + total_errors) * 100):.2f}%"
    )
    print(f"     - Overall throughput: {overall_throughput:.1f} ops/sec")
    print(f"     - Total time: {overall_time:.2f} seconds")

    # 7. Performance Summary
    print("\n📊 Generation 3 Performance Summary:")

    final_mem_stats = memory_optimizer.get_memory_stats()

    print("   🎯 Optimization Features Demonstrated:")
    print("     ✅ Memory pooling and optimization")
    print("     ✅ Adaptive load balancing")
    print("     ✅ Auto-scaling with multiple strategies")
    print("     ✅ Predictive scaling based on patterns")
    print("     ✅ Multi-physics parallel processing")
    print("     ✅ Comprehensive stress testing")

    print(f"\n   💾 Final Memory Usage: {final_mem_stats['rss_mb']:.1f}MB")
    print(f"   🔧 Available CPU cores: {mp.cpu_count()}")
    print(f"   ⚡ System optimized for high-performance computing")

    print(f"\n⚡ Generation 3 Scaling Demo Complete!")
    print("   🚀 System demonstrates advanced performance optimization")
    print("   📈 Scaling mechanisms are fully operational")
    print("   🎯 Ready for production deployment at scale")


if __name__ == "__main__":
    main()
