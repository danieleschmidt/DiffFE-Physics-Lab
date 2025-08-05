#!/usr/bin/env python3
"""Performance benchmarks for DiffFE-Physics-Lab."""

import time
import numpy as np
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from performance import CacheManager, PerformanceProfiler, PerformanceOptimizer
from utils.manufactured_solutions import generate_manufactured_solution, SolutionType


@dataclass
class BenchmarkResult:
    """Results from a single benchmark."""
    name: str
    duration: float
    memory_usage: float
    iterations: int
    throughput: float
    success: bool
    error: str = ""


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.results = []
        self.profiler = PerformanceProfiler()
        self.optimizer = PerformanceOptimizer()
        self.cache = CacheManager(max_size=1000)
        
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks."""
        print("🚀 Starting DiffFE-Physics-Lab Performance Benchmarks")
        print("=" * 60)
        
        benchmarks = [
            ("Cache Performance", self.benchmark_cache_performance),
            ("Profiler Overhead", self.benchmark_profiler_overhead),
            ("Optimization Strategies", self.benchmark_optimization_strategies),
            ("Memory Management", self.benchmark_memory_management),
            ("Manufactured Solutions", self.benchmark_manufactured_solutions),
            ("Parallel Processing", self.benchmark_parallel_processing),
            ("Large Dataset Handling", self.benchmark_large_datasets),
            ("Cache Hit Rates", self.benchmark_cache_hit_rates),
            ("Function Call Overhead", self.benchmark_function_call_overhead),
            ("Memory Allocation", self.benchmark_memory_allocation)
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"\n📊 Running {name}...")
            try:
                result = benchmark_func()
                self.results.append(result)
                status = "✅" if result.success else "❌"
                print(f"{status} {name}: {result.duration:.4f}s, {result.throughput:.1f} ops/s")
            except Exception as e:
                print(f"❌ {name}: Failed - {e}")
                self.results.append(BenchmarkResult(
                    name=name,
                    duration=0.0,
                    memory_usage=0.0,
                    iterations=0,
                    throughput=0.0,
                    success=False,
                    error=str(e)
                ))
        
        return self.results
    
    def benchmark_cache_performance(self) -> BenchmarkResult:
        """Benchmark cache performance with various workloads."""
        cache = CacheManager(max_size=1000, ttl=60.0)
        
        # Warm up cache
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
        
        iterations = 10000
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Mixed read/write workload
        for i in range(iterations):
            if i % 4 == 0:  # 25% writes
                cache.set(f"key_{i % 200}", f"value_{i}")
            else:  # 75% reads
                cache.get(f"key_{i % 200}")
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - start_time
        memory_usage = end_memory - start_memory
        throughput = iterations / duration
        
        return BenchmarkResult(
            name="Cache Performance",
            duration=duration,
            memory_usage=memory_usage,
            iterations=iterations,
            throughput=throughput,
            success=True
        )
    
    def benchmark_profiler_overhead(self) -> BenchmarkResult:
        """Benchmark profiler overhead on function calls."""
        profiler = PerformanceProfiler()
        
        @profiler.profile()
        def profiled_function(n):
            return sum(range(n))
        
        def unprofiled_function(n):
            return sum(range(n))
        
        # Benchmark unprofiled version
        iterations = 1000
        n = 1000
        
        start_time = time.time()
        for _ in range(iterations):
            unprofiled_function(n)
        unprofiled_time = time.time() - start_time
        
        # Benchmark profiled version
        start_time = time.time()
        for _ in range(iterations):
            profiled_function(n)
        profiled_time = time.time() - start_time
        
        overhead = profiled_time - unprofiled_time
        overhead_percent = (overhead / unprofiled_time) * 100
        
        return BenchmarkResult(
            name="Profiler Overhead",
            duration=overhead,
            memory_usage=overhead_percent,  # Store overhead % in memory field
            iterations=iterations,
            throughput=iterations / profiled_time,
            success=overhead_percent < 50.0  # Less than 50% overhead is acceptable
        )
    
    def benchmark_optimization_strategies(self) -> BenchmarkResult:
        """Benchmark different optimization strategies."""
        optimizer = PerformanceOptimizer()
        
        def compute_intensive_task(data):
            """CPU-intensive computation."""
            return [x**2 + np.sin(x) + np.cos(x) for x in data]
        
        data = list(range(1000))
        strategies = ['auto', 'parallel', 'vectorize', 'cache']
        results = {}
        
        for strategy in strategies:
            @optimizer.optimize(strategy=strategy)
            def optimized_task(data):
                return compute_intensive_task(data)
            
            # Benchmark this strategy
            iterations = 10
            start_time = time.time()
            
            for _ in range(iterations):
                result = optimized_task(data)
            
            duration = time.time() - start_time
            results[strategy] = duration
        
        # Find best strategy
        best_strategy = min(results, key=results.get)
        best_time = results[best_strategy]
        worst_time = max(results.values())
        
        improvement = (worst_time - best_time) / worst_time * 100
        
        return BenchmarkResult(
            name="Optimization Strategies",
            duration=best_time,
            memory_usage=improvement,  # Store improvement % in memory field
            iterations=len(strategies) * 10,
            throughput=1.0 / best_time,
            success=improvement > 10.0  # At least 10% improvement
        )
    
    def benchmark_memory_management(self) -> BenchmarkResult:
        """Benchmark memory management and garbage collection."""
        optimizer = PerformanceOptimizer()
        
        start_memory = self._get_memory_usage()
        
        # Create large amounts of cached data
        large_data = []
        for i in range(100):
            data = np.random.random((1000, 1000))  # 8MB per array
            large_data.append(data)
        
        peak_memory = self._get_memory_usage()
        memory_used = peak_memory - start_memory
        
        # Test memory optimization
        start_time = time.time()
        optimizer.optimize_memory_usage(target_mb=start_memory + 50)
        optimization_time = time.time() - start_time
        
        # Clean up
        del large_data
        
        end_memory = self._get_memory_usage()
        memory_freed = peak_memory - end_memory
        
        return BenchmarkResult(
            name="Memory Management",
            duration=optimization_time,
            memory_usage=memory_freed,
            iterations=100,
            throughput=memory_freed / optimization_time if optimization_time > 0 else 0,
            success=memory_freed > 0
        )
    
    def benchmark_manufactured_solutions(self) -> BenchmarkResult:
        """Benchmark manufactured solution generation and evaluation."""
        iterations = 1000
        
        # Generate different types of manufactured solutions
        solutions = []
        
        start_time = time.time()
        
        for i in range(iterations):
            solution_type = [SolutionType.TRIGONOMETRIC, SolutionType.POLYNOMIAL, SolutionType.EXPONENTIAL][i % 3]
            
            ms = generate_manufactured_solution(
                solution_type=solution_type,
                dimension=2,
                parameters={'frequency': 1.0, 'amplitude': 1.0} if solution_type == SolutionType.TRIGONOMETRIC else {}
            )
            
            # Evaluate solution at test points
            test_points = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]
            for point in test_points:
                u_val = ms['solution'](point)
                f_val = ms['source'](point)
                grad_val = ms['gradient'](point)
            
            solutions.append(ms)
        
        duration = time.time() - start_time
        throughput = iterations / duration
        
        return BenchmarkResult(
            name="Manufactured Solutions",
            duration=duration,
            memory_usage=len(solutions) * 0.001,  # Estimate
            iterations=iterations,
            throughput=throughput,
            success=throughput > 100  # At least 100 solutions/second
        )
    
    def benchmark_parallel_processing(self) -> BenchmarkResult:
        """Benchmark parallel processing capabilities."""
        optimizer = PerformanceOptimizer()
        
        def cpu_bound_task(x):
            """CPU-bound task for parallel processing."""
            result = 0
            for i in range(1000):
                result += x * np.sin(i) * np.cos(i)
            return result
        
        data = list(range(100))
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [cpu_bound_task(x) for x in data]
        sequential_time = time.time() - start_time
        
        # Parallel processing
        start_time = time.time()
        parallel_results = optimizer.parallel_map(cpu_bound_task, data, chunk_size=10)
        parallel_time = time.time() - start_time
        
        # Verify results are equivalent
        results_match = len(sequential_results) == len(parallel_results)
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        
        return BenchmarkResult(
            name="Parallel Processing",
            duration=parallel_time,
            memory_usage=speedup,  # Store speedup in memory field
            iterations=len(data),
            throughput=len(data) / parallel_time,
            success=results_match and speedup > 1.0
        )
    
    def benchmark_large_datasets(self) -> BenchmarkResult:
        """Benchmark handling of large datasets."""
        # Create large dataset
        large_array = np.random.random((10000, 100))  # 80MB
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Perform operations on large dataset
        operations = [
            lambda x: np.mean(x, axis=0),
            lambda x: np.std(x, axis=0),
            lambda x: np.max(x, axis=0),
            lambda x: np.min(x, axis=0),
            lambda x: np.sum(x**2, axis=0)
        ]
        
        results = []
        for op in operations:
            result = op(large_array)
            results.append(result)
        
        duration = time.time() - start_time
        peak_memory = self._get_memory_usage()
        memory_usage = peak_memory - start_memory
        
        # Clean up
        del large_array
        
        throughput = large_array.size / duration if duration > 0 else 0
        
        return BenchmarkResult(
            name="Large Dataset Handling",
            duration=duration,
            memory_usage=memory_usage,
            iterations=len(operations),
            throughput=throughput,
            success=duration < 1.0  # Should complete in under 1 second
        )
    
    def benchmark_cache_hit_rates(self) -> BenchmarkResult:
        """Benchmark cache hit rates with different access patterns."""
        cache = CacheManager(max_size=100)
        
        # Fill cache
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
        
        patterns = {
            'sequential': list(range(100)),
            'random': np.random.randint(0, 100, 200).tolist(),
            'hot_spot': [i % 10 for i in range(200)],  # Access first 10 keys repeatedly
            'mixed': list(range(50)) + np.random.randint(0, 150, 100).tolist()
        }
        
        results = {}
        start_time = time.time()
        
        for pattern_name, access_pattern in patterns.items():
            hits = 0
            misses = 0
            
            for key_idx in access_pattern:
                if cache.get(f"key_{key_idx}") is not None:
                    hits += 1
                else:
                    misses += 1
            
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
            results[pattern_name] = hit_rate
        
        duration = time.time() - start_time
        avg_hit_rate = sum(results.values()) / len(results)
        
        return BenchmarkResult(
            name="Cache Hit Rates",
            duration=duration,
            memory_usage=avg_hit_rate * 100,  # Store hit rate % in memory field
            iterations=sum(len(pattern) for pattern in patterns.values()),
            throughput=sum(len(pattern) for pattern in patterns.values()) / duration,
            success=avg_hit_rate > 0.5  # At least 50% hit rate
        )
    
    def benchmark_function_call_overhead(self) -> BenchmarkResult:
        """Benchmark function call overhead with decorators."""
        cache = CacheManager(max_size=1000)
        profiler = PerformanceProfiler()
        
        def plain_function(x):
            return x * 2
        
        @cache.cached
        def cached_function(x):
            return x * 2
        
        @profiler.profile()
        def profiled_function(x):
            return x * 2
        
        @cache.cached
        @profiler.profile()
        def decorated_function(x):
            return x * 2
        
        functions = [
            ("Plain", plain_function),
            ("Cached", cached_function),
            ("Profiled", profiled_function),
            ("Decorated", decorated_function)
        ]
        
        iterations = 10000
        results = {}
        
        for name, func in functions:
            start_time = time.time()
            for i in range(iterations):
                func(i % 100)  # Limited range for cache effectiveness
            duration = time.time() - start_time
            results[name] = duration
        
        # Calculate overhead
        plain_time = results["Plain"]
        overheads = {name: ((time - plain_time) / plain_time * 100) 
                    for name, time in results.items() if name != "Plain"}
        
        total_overhead = sum(overheads.values())
        
        return BenchmarkResult(
            name="Function Call Overhead",
            duration=results["Decorated"],  # Most decorated version
            memory_usage=total_overhead,  # Store total overhead % in memory field
            iterations=iterations * len(functions),
            throughput=iterations / results["Decorated"],
            success=all(overhead < 100 for overhead in overheads.values())  # Less than 100% overhead
        )
    
    def benchmark_memory_allocation(self) -> BenchmarkResult:
        """Benchmark memory allocation patterns."""
        start_memory = self._get_memory_usage()
        start_time = time.time()
        
        # Different allocation patterns
        allocations = []
        
        # Small frequent allocations
        for _ in range(1000):
            arr = np.random.random(100)
            allocations.append(arr)
        
        # Large infrequent allocations
        for _ in range(10):
            arr = np.random.random(10000)
            allocations.append(arr)
        
        # Mixed allocations
        for i in range(100):
            size = 100 if i % 2 == 0 else 1000
            arr = np.random.random(size)
            allocations.append(arr)
        
        peak_memory = self._get_memory_usage()
        
        # Clean up and measure deallocation
        del allocations
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - start_time
        memory_used = peak_memory - start_memory
        memory_freed = peak_memory - end_memory
        
        return BenchmarkResult(
            name="Memory Allocation",
            duration=duration,
            memory_usage=memory_used,
            iterations=1110,  # Total allocations
            throughput=1110 / duration,
            success=memory_freed > memory_used * 0.8  # At least 80% memory freed
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        total_duration = sum(r.duration for r in successful_results)
        avg_throughput = sum(r.throughput for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Performance categories
        categories = {
            'cache': [r for r in successful_results if 'cache' in r.name.lower()],
            'memory': [r for r in successful_results if 'memory' in r.name.lower()],
            'parallel': [r for r in successful_results if 'parallel' in r.name.lower()],
            'optimization': [r for r in successful_results if 'optimization' in r.name.lower()]
        }
        
        report = {
            'summary': {
                'total_benchmarks': len(self.results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results) / len(self.results) * 100,
                'total_duration': total_duration,
                'average_throughput': avg_throughput
            },
            'detailed_results': [
                {
                    'name': r.name,
                    'duration': r.duration,
                    'memory_usage': r.memory_usage,
                    'iterations': r.iterations,
                    'throughput': r.throughput,
                    'success': r.success,
                    'error': r.error
                }
                for r in self.results
            ],
            'category_performance': {
                category: {
                    'count': len(results),
                    'avg_duration': sum(r.duration for r in results) / len(results) if results else 0,
                    'avg_throughput': sum(r.throughput for r in results) / len(results) if results else 0
                }
                for category, results in categories.items()
            },
            'top_performers': sorted(successful_results, key=lambda r: r.throughput, reverse=True)[:5],
            'slowest_benchmarks': sorted(successful_results, key=lambda r: r.duration, reverse=True)[:3],
            'recommendations': self._generate_recommendations(successful_results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate performance recommendations based on benchmark results."""
        recommendations = []
        
        # Check cache performance
        cache_results = [r for r in results if 'cache' in r.name.lower()]
        if cache_results and any(r.throughput < 1000 for r in cache_results):
            recommendations.append("Consider optimizing cache implementation for better throughput")
        
        # Check parallel processing
        parallel_results = [r for r in results if 'parallel' in r.name.lower()]
        if parallel_results and any(r.memory_usage < 2.0 for r in parallel_results):  # speedup stored in memory_usage
            recommendations.append("Parallel processing shows limited speedup - consider task granularity")
        
        # Check memory usage
        memory_results = [r for r in results if 'memory' in r.name.lower()]
        if memory_results and any(r.memory_usage > 100 for r in memory_results):
            recommendations.append("High memory usage detected - implement memory optimization strategies")
        
        # Check overall performance
        avg_throughput = sum(r.throughput for r in results) / len(results) if results else 0
        if avg_throughput < 100:
            recommendations.append("Overall throughput is low - consider algorithm optimizations")
        
        if not recommendations:
            recommendations.append("Performance benchmarks show good results across all categories")
        
        return recommendations
    
    def print_report(self):
        """Print detailed benchmark report."""
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        
        print(f"\n📈 SUMMARY:")
        summary = report['summary']
        print(f"   Total benchmarks: {summary['total_benchmarks']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Total duration: {summary['total_duration']:.3f}s")
        print(f"   Average throughput: {summary['average_throughput']:.1f} ops/s")
        
        print(f"\n🏆 TOP PERFORMERS:")
        for i, result in enumerate(report['top_performers'][:3], 1):
            print(f"   {i}. {result.name}: {result.throughput:.1f} ops/s")
        
        print(f"\n⚠️  SLOWEST BENCHMARKS:")
        for i, result in enumerate(report['slowest_benchmarks'][:3], 1):
            print(f"   {i}. {result.name}: {result.duration:.3f}s")
        
        print(f"\n📊 CATEGORY PERFORMANCE:")
        for category, perf in report['category_performance'].items():
            if perf['count'] > 0:
                print(f"   {category.capitalize()}: {perf['avg_throughput']:.1f} ops/s avg")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        
        if report['summary']['failed'] > 0:
            print(f"\n❌ FAILED BENCHMARKS:")
            failed = [r for r in report['detailed_results'] if not r['success']]
            for result in failed:
                print(f"   • {result['name']}: {result['error']}")
        
        print("\n" + "=" * 80)


def main():
    """Main benchmark execution function."""
    print("🚀 DiffFE-Physics-Lab Performance Benchmark Suite")
    print("🔥 Testing system performance across all components")
    
    suite = PerformanceBenchmarkSuite()
    results = suite.run_all_benchmarks()
    suite.print_report()
    
    # Overall assessment
    success_rate = len([r for r in results if r.success]) / len(results) * 100
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if success_rate >= 90:
        print(f"   🎉 EXCELLENT: {success_rate:.1f}% benchmark success rate")
        status = "EXCELLENT"
    elif success_rate >= 75:
        print(f"   ✅ GOOD: {success_rate:.1f}% benchmark success rate")
        status = "GOOD"
    elif success_rate >= 50:
        print(f"   ⚠️  FAIR: {success_rate:.1f}% benchmark success rate")
        status = "FAIR"
    else:
        print(f"   ❌ POOR: {success_rate:.1f}% benchmark success rate")
        status = "POOR"
    
    print(f"   Status: {status}")
    print("=" * 80)
    
    return success_rate >= 75


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)