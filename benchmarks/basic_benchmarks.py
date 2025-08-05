#!/usr/bin/env python3
"""Basic performance benchmarks for DiffFE-Physics-Lab (no external dependencies)."""

import time
import sys
import os
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Callable
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from performance.cache import CacheManager
from performance.profiler import PerformanceProfiler
from utils.manufactured_solutions import generate_manufactured_solution, SolutionType


@dataclass
class BenchmarkResult:
    """Results from a single benchmark."""
    name: str
    duration: float
    operations: int
    throughput: float
    success: bool
    notes: str = ""


class BasicBenchmarkSuite:
    """Basic performance benchmark suite without external dependencies."""
    
    def __init__(self):
        self.results = []
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all basic performance benchmarks."""
        print("🚀 Starting DiffFE-Physics-Lab Basic Performance Benchmarks")
        print("=" * 60)
        
        benchmarks = [
            ("Cache Operations", self.benchmark_cache_operations),
            ("Profiler Basic", self.benchmark_profiler_basic),
            ("Mathematical Operations", self.benchmark_math_operations),
            ("String Operations", self.benchmark_string_operations),
            ("List Operations", self.benchmark_list_operations),
            ("Dictionary Operations", self.benchmark_dict_operations),
            ("Function Calls", self.benchmark_function_calls),
            ("Manufactured Solutions Basic", self.benchmark_manufactured_solutions_basic),
            ("Memory Pattern", self.benchmark_memory_pattern),
            ("CPU Intensive", self.benchmark_cpu_intensive)
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"\n📊 Running {name}...")
            try:
                result = benchmark_func()
                self.results.append(result)
                status = "✅" if result.success else "❌"
                print(f"{status} {name}: {result.duration:.4f}s, {result.throughput:.1f} ops/s")
                if result.notes:
                    print(f"   ℹ️  {result.notes}")
            except Exception as e:
                print(f"❌ {name}: Failed - {e}")
                self.results.append(BenchmarkResult(
                    name=name,
                    duration=0.0,
                    operations=0,
                    throughput=0.0,
                    success=False,
                    notes=f"Error: {str(e)}"
                ))
        
        return self.results
    
    def benchmark_cache_operations(self) -> BenchmarkResult:
        """Benchmark basic cache operations."""
        cache = CacheManager(max_size=1000)
        
        operations = 10000
        start_time = time.time()
        
        # Mixed cache operations
        for i in range(operations):
            if i % 4 == 0:  # 25% writes
                cache.set(f"key_{i}", f"value_{i}")
            else:  # 75% reads
                cache.get(f"key_{i % 100}")
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        # Check cache stats
        stats = cache.get_stats()
        hit_rate = stats.get('hit_rate', 0)
        
        return BenchmarkResult(
            name="Cache Operations",
            duration=duration,
            operations=operations,
            throughput=throughput,
            success=throughput > 1000,  # Should handle > 1000 ops/sec
            notes=f"Hit rate: {hit_rate:.1%}, Cache size: {stats.get('size', 0)}"
        )
    
    def benchmark_profiler_basic(self) -> BenchmarkResult:
        """Benchmark basic profiler functionality."""
        profiler = PerformanceProfiler()
        
        @profiler.profile()
        def test_function(n):
            total = 0
            for i in range(n):
                total += i * i
            return total
        
        operations = 100
        start_time = time.time()
        
        for i in range(operations):
            result = test_function(100)
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        # Check profiler stats
        stats = profiler.get_stats('test_function')
        
        return BenchmarkResult(
            name="Profiler Basic",
            duration=duration,
            operations=operations,
            throughput=throughput,
            success=stats is not None and stats.call_count == operations,
            notes=f"Profiled calls: {stats.call_count if stats else 0}, Avg time: {stats.avg_time:.6f}s" if stats else "No stats"
        )
    
    def benchmark_math_operations(self) -> BenchmarkResult:
        """Benchmark mathematical operations."""
        operations = 100000
        start_time = time.time()
        
        result = 0
        for i in range(operations):
            # Mix of operations
            x = i * 0.01
            result += math.sin(x) + math.cos(x) + math.sqrt(abs(x)) + x**2
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        return BenchmarkResult(
            name="Mathematical Operations",
            duration=duration,
            operations=operations,
            throughput=throughput,
            success=throughput > 10000,  # Should handle > 10k ops/sec
            notes=f"Final result: {result:.2f}"
        )
    
    def benchmark_string_operations(self) -> BenchmarkResult:
        """Benchmark string operations."""
        operations = 10000
        start_time = time.time()
        
        strings = []
        for i in range(operations):
            # Various string operations
            s = f"test_string_{i}"
            s = s.upper().lower()
            s = s.replace("_", "-")
            s = s.split("-")
            strings.append("".join(s))
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        return BenchmarkResult(
            name="String Operations",
            duration=duration,
            operations=operations,
            throughput=throughput,
            success=len(strings) == operations,
            notes=f"Generated {len(strings)} strings"
        )
    
    def benchmark_list_operations(self) -> BenchmarkResult:
        """Benchmark list operations."""
        operations = 10000
        start_time = time.time()
        
        # Create and manipulate lists
        data = []
        for i in range(operations):
            data.append(i)
            if len(data) > 100:
                data.pop(0)  # Remove from front
            if i % 10 == 0:
                data.sort()
        
        # Additional operations
        filtered = [x for x in data if x % 2 == 0]
        mapped = [x * 2 for x in filtered]
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        return BenchmarkResult(
            name="List Operations",
            duration=duration,
            operations=operations,
            throughput=throughput,
            success=len(mapped) > 0,
            notes=f"Final list size: {len(data)}, Mapped: {len(mapped)}"
        )
    
    def benchmark_dict_operations(self) -> BenchmarkResult:
        """Benchmark dictionary operations."""
        operations = 10000
        start_time = time.time()
        
        data = {}
        for i in range(operations):
            key = f"key_{i % 1000}"  # Reuse keys to test updates
            data[key] = i
            
            if i % 100 == 0:
                # Occasional lookups and deletions
                if key in data:
                    value = data[key]
                if len(data) > 500:
                    keys_to_delete = list(data.keys())[:10]
                    for k in keys_to_delete:
                        del data[k]
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        return BenchmarkResult(
            name="Dictionary Operations",
            duration=duration,
            operations=operations,
            throughput=throughput,
            success=len(data) > 0,
            notes=f"Final dict size: {len(data)}"
        )
    
    def benchmark_function_calls(self) -> BenchmarkResult:
        """Benchmark function call overhead."""
        
        def simple_function(x, y):
            return x + y
        
        def complex_function(x, y, z=0):
            result = x * y + z
            if result > 100:
                result = result % 100
            return result
        
        lambda_func = lambda x, y: x * y
        
        operations = 10000
        start_time = time.time()
        
        results = []
        for i in range(operations):
            # Mix of function types
            if i % 3 == 0:
                results.append(simple_function(i, i+1))
            elif i % 3 == 1:
                results.append(complex_function(i, i+1, i+2))
            else:
                results.append(lambda_func(i, i+1))
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        return BenchmarkResult(
            name="Function Calls",
            duration=duration,
            operations=operations,
            throughput=throughput,
            success=len(results) == operations,
            notes=f"Results generated: {len(results)}"
        )
    
    def benchmark_manufactured_solutions_basic(self) -> BenchmarkResult:
        """Benchmark basic manufactured solution operations."""
        operations = 100
        start_time = time.time()
        
        solutions = []
        for i in range(operations):
            # Alternate between solution types
            solution_type = [SolutionType.TRIGONOMETRIC, SolutionType.POLYNOMIAL, SolutionType.EXPONENTIAL][i % 3]
            
            try:
                ms = generate_manufactured_solution(
                    solution_type=solution_type,
                    dimension=2,
                    parameters={'frequency': 1.0, 'amplitude': 1.0} if solution_type == SolutionType.TRIGONOMETRIC else {}
                )
                
                # Test evaluation
                test_point = [0.5, 0.5]
                u_val = ms['solution'](test_point)
                f_val = ms['source'](test_point)
                
                solutions.append((ms, u_val, f_val))
            except Exception as e:
                # Continue with other solutions if one fails
                pass
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        return BenchmarkResult(
            name="Manufactured Solutions Basic",
            duration=duration,
            operations=operations,
            throughput=throughput,
            success=len(solutions) > operations * 0.8,  # At least 80% success
            notes=f"Solutions created: {len(solutions)}/{operations}"
        )
    
    def benchmark_memory_pattern(self) -> BenchmarkResult:
        """Benchmark memory allocation patterns."""
        operations = 1000
        start_time = time.time()
        
        # Different allocation patterns
        small_lists = []
        large_lists = []
        
        for i in range(operations):
            # Small frequent allocations
            small_list = list(range(100))
            small_lists.append(small_list)
            
            # Occasional large allocations
            if i % 10 == 0:
                large_list = list(range(1000))
                large_lists.append(large_list)
            
            # Clean up occasionally to simulate real usage
            if len(small_lists) > 100:
                small_lists = small_lists[-50:]  # Keep last 50
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        total_elements = sum(len(lst) for lst in small_lists) + sum(len(lst) for lst in large_lists)
        
        return BenchmarkResult(
            name="Memory Pattern",
            duration=duration,
            operations=operations,
            throughput=throughput,
            success=total_elements > 0,
            notes=f"Total elements: {total_elements}, Small lists: {len(small_lists)}, Large lists: {len(large_lists)}"
        )
    
    def benchmark_cpu_intensive(self) -> BenchmarkResult:
        """Benchmark CPU-intensive operations."""
        operations = 1000
        start_time = time.time()
        
        results = []
        for i in range(operations):
            # CPU-intensive computation
            n = 100 + i % 100
            
            # Calculate some mathematical series
            result = 0
            for j in range(n):
                result += math.sin(j * 0.1) * math.cos(j * 0.1)
                result += j**0.5 if j > 0 else 0
                
            # Some recursion-like pattern
            fib_like = 1
            a, b = 0, 1
            for _ in range(min(20, n)):
                a, b = b, a + b
                fib_like = b
            
            results.append(result + fib_like)
        
        duration = time.time() - start_time
        throughput = operations / duration
        
        return BenchmarkResult(
            name="CPU Intensive",
            duration=duration,
            operations=operations,
            throughput=throughput,
            success=len(results) == operations,
            notes=f"Computations completed: {len(results)}"
        )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        total_duration = sum(r.duration for r in successful_results)
        total_operations = sum(r.operations for r in successful_results)
        avg_throughput = sum(r.throughput for r in successful_results) / len(successful_results) if successful_results else 0
        
        return {
            'summary': {
                'total_benchmarks': len(self.results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results) / len(self.results) * 100,
                'total_duration': total_duration,
                'total_operations': total_operations,
                'average_throughput': avg_throughput
            },
            'results': [
                {
                    'name': r.name,
                    'duration': r.duration,
                    'operations': r.operations,
                    'throughput': r.throughput,
                    'success': r.success,
                    'notes': r.notes
                }
                for r in self.results
            ],
            'top_performers': sorted(successful_results, key=lambda r: r.throughput, reverse=True)[:3],
            'slowest_benchmarks': sorted(successful_results, key=lambda r: r.duration, reverse=True)[:3]
        }
    
    def print_report(self):
        """Print detailed benchmark report."""
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print("📊 BASIC PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        
        summary = report['summary']
        print(f"\n📈 SUMMARY:")
        print(f"   Total benchmarks: {summary['total_benchmarks']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        print(f"   Total duration: {summary['total_duration']:.3f}s")
        print(f"   Total operations: {summary['total_operations']:,}")
        print(f"   Average throughput: {summary['average_throughput']:.1f} ops/s")
        
        print(f"\n🏆 TOP PERFORMERS:")
        for i, result in enumerate(report['top_performers'], 1):
            print(f"   {i}. {result.name}: {result.throughput:.1f} ops/s")
        
        print(f"\n⏱️  DETAILED RESULTS:")
        for result in report['results']:
            status = "✅" if result['success'] else "❌"
            print(f"   {status} {result['name']}")
            print(f"      Duration: {result['duration']:.4f}s")
            print(f"      Operations: {result['operations']:,}")
            print(f"      Throughput: {result['throughput']:.1f} ops/s")
            if result['notes']:
                print(f"      Notes: {result['notes']}")
        
        if summary['failed'] > 0:
            print(f"\n❌ FAILED BENCHMARKS:")
            failed_results = [r for r in report['results'] if not r['success']]
            for result in failed_results:
                print(f"   • {result['name']}: {result['notes']}")
        
        print("\n" + "=" * 80)


def main():
    """Main benchmark execution function."""
    print("🚀 DiffFE-Physics-Lab Basic Performance Benchmark Suite")
    print("🔥 Testing core system performance (no external dependencies)")
    
    suite = BasicBenchmarkSuite()
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
    
    # Performance assessment
    report = suite.generate_report()
    avg_throughput = report['summary']['average_throughput']
    
    if avg_throughput > 10000:
        perf_status = "HIGH PERFORMANCE"
        perf_emoji = "🚀"
    elif avg_throughput > 1000:
        perf_status = "GOOD PERFORMANCE"
        perf_emoji = "✅"
    elif avg_throughput > 100:
        perf_status = "FAIR PERFORMANCE"
        perf_emoji = "⚠️"
    else:
        perf_status = "NEEDS OPTIMIZATION"
        perf_emoji = "❌"
    
    print(f"   {perf_emoji} Performance: {perf_status} ({avg_throughput:.1f} ops/s avg)")
    print(f"   Status: {status}")
    print("=" * 80)
    
    return success_rate >= 75 and avg_throughput > 100


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)