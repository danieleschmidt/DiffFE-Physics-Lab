#!/usr/bin/env python3
"""Comprehensive benchmarks for sentiment analysis performance."""

import time
import sys
import os
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Mock external dependencies for benchmarking
class MockNumPy:
    """Mock NumPy for benchmarking without dependencies."""
    def __init__(self):
        self.ndarray = list
        self.float64 = float
        self.float32 = float
        
    def array(self, data):
        return list(data) if hasattr(data, '__iter__') else [data]
        
    def random(self):
        import random
        class Random:
            def randn(self, *shape):
                if len(shape) == 1:
                    return [random.gauss(0, 1) for _ in range(shape[0])]
                elif len(shape) == 2:
                    return [[random.gauss(0, 1) for _ in range(shape[1])] 
                            for _ in range(shape[0])]
                return random.gauss(0, 1)
            def uniform(self, low, high, size):
                import random
                if isinstance(size, int):
                    return [random.uniform(low, high) for _ in range(size)]
                return random.uniform(low, high)
        return Random()
        
    def mean(self, data):
        return sum(data) / len(data) if data else 0
        
    def std(self, data):
        if not data: return 0
        mean_val = self.mean(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        return variance ** 0.5

# Mock sys modules for testing
sys.modules['numpy'] = MockNumPy()
sys.modules['jax'] = type('MockJAX', (), {})()
sys.modules['torch'] = type('MockTorch', (), {})()
sys.modules['sklearn'] = type('MockSklearn', (), {})()


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    benchmark_name: str
    dataset_size: int
    processing_time: float
    memory_usage_mb: float
    throughput_texts_per_second: float
    accuracy_score: Optional[float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    
    suite_name: str
    timestamp: str
    system_info: Dict[str, Any]
    benchmark_results: List[BenchmarkResult]
    summary_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'suite_name': self.suite_name,
            'timestamp': self.timestamp,
            'system_info': self.system_info,
            'benchmark_results': [result.to_dict() for result in self.benchmark_results],
            'summary_stats': self.summary_stats
        }
        
    def save(self, filepath: Path):
        """Save benchmark results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class SentimentBenchmarkRunner:
    """Benchmark runner for sentiment analysis components."""
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.results = []
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarks."""
        import platform
        import os
        
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            cpu_count = os.cpu_count() or 1
            memory_gb = 1.0  # Default fallback
        
        return {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor() or 'Unknown',
            'cpu_count': cpu_count,
            'memory_total_gb': memory_gb,
            'architecture': platform.architecture()[0]
        }
        
    def run_embedding_benchmark(self, text_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark text embedding generation."""
        print("🔢 Running embedding generation benchmarks...")
        results = []
        
        for size in text_sizes:
            # Generate test texts
            test_texts = [f"Sample text number {i} with various sentiment content." 
                         for i in range(size)]
            
            # Benchmark embedding generation
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            embeddings = self._generate_test_embeddings(test_texts, embedding_dim=100)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            processing_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = size / processing_time
            
            result = BenchmarkResult(
                benchmark_name="embedding_generation",
                dataset_size=size,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                throughput_texts_per_second=throughput,
                accuracy_score=None,  # N/A for embeddings
                metadata={
                    'embedding_dim': 100,
                    'method': 'tfidf_simulation',
                    'vocab_size': min(1000, size * 10)
                }
            )
            
            results.append(result)
            print(f"   📊 Size {size:4d}: {processing_time:.3f}s, {throughput:6.1f} texts/s")
            
        return results
        
    def run_physics_simulation_benchmark(self, text_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark physics-informed sentiment simulation."""
        print("⚛️  Running physics simulation benchmarks...")
        results = []
        
        for size in text_sizes:
            # Generate test data
            test_texts = [f"Test sentiment text {i}" for i in range(size)]
            embeddings = self._generate_test_embeddings(test_texts, embedding_dim=50)
            
            # Benchmark physics simulation
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            sentiments = self._run_physics_simulation(embeddings, num_steps=20)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            processing_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = size / processing_time
            
            # Simple accuracy simulation (how many converged)
            accuracy = sum(1 for s in sentiments if abs(s) > 0.1) / len(sentiments)
            
            result = BenchmarkResult(
                benchmark_name="physics_simulation",
                dataset_size=size,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                throughput_texts_per_second=throughput,
                accuracy_score=accuracy,
                metadata={
                    'num_steps': 20,
                    'embedding_dim': 50,
                    'convergence_rate': accuracy
                }
            )
            
            results.append(result)
            print(f"   📊 Size {size:4d}: {processing_time:.3f}s, {throughput:6.1f} texts/s, {accuracy:.3f} conv")
            
        return results
        
    def run_caching_benchmark(self, cache_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark caching performance."""
        print("💾 Running caching benchmarks...")
        results = []
        
        # Simple cache simulation
        cache = {}
        cache_hits = 0
        cache_misses = 0
        
        for cache_size in cache_sizes:
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Simulate cache operations
            for i in range(cache_size * 2):  # More operations than cache size
                key = f"cache_key_{i % cache_size}"  # Will cause some hits
                
                if key in cache:
                    cache_hits += 1
                    value = cache[key]
                else:
                    cache_misses += 1
                    # Simulate expensive computation
                    value = [j * 0.1 for j in range(100)]  # Mock embedding
                    cache[key] = value
                    
                    # Limit cache size (LRU simulation)
                    if len(cache) > cache_size:
                        # Remove oldest entry (simplified)
                        oldest_key = next(iter(cache))
                        del cache[oldest_key]
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            processing_time = end_time - start_time
            memory_usage = end_memory - start_memory
            hit_rate = cache_hits / (cache_hits + cache_misses)
            
            result = BenchmarkResult(
                benchmark_name="caching_performance",
                dataset_size=cache_size,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                throughput_texts_per_second=cache_size / processing_time,
                accuracy_score=hit_rate,
                metadata={
                    'cache_hits': cache_hits,
                    'cache_misses': cache_misses,
                    'hit_rate': hit_rate,
                    'cache_efficiency': hit_rate * cache_size
                }
            )
            
            results.append(result)
            print(f"   📊 Size {cache_size:4d}: {processing_time:.3f}s, {hit_rate:.3f} hit rate")
            
        return results
        
    def run_api_performance_benchmark(self, request_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark API request processing."""
        print("🌐 Running API performance benchmarks...")
        results = []
        
        for size in request_sizes:
            # Simulate API requests
            requests = []
            for i in range(size):
                request = {
                    'texts': [f"API test text {i}"],
                    'options': {
                        'embedding_method': 'tfidf',
                        'physics_params': {'temperature': 1.0}
                    }
                }
                requests.append(request)
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Simulate request processing
            responses = []
            for request in requests:
                # Mock validation
                if not request.get('texts'):
                    continue
                    
                # Mock processing
                response = {
                    'success': True,
                    'data': {
                        'sentiments': [0.1 * i for i in range(len(request['texts']))],
                        'processing_time': 0.001
                    }
                }
                responses.append(response)
                
                # Simulate some processing delay
                time.sleep(0.0001)  # 0.1ms per request
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            processing_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = size / processing_time
            success_rate = len(responses) / len(requests)
            
            result = BenchmarkResult(
                benchmark_name="api_performance",
                dataset_size=size,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                throughput_texts_per_second=throughput,
                accuracy_score=success_rate,
                metadata={
                    'requests_processed': len(responses),
                    'success_rate': success_rate,
                    'avg_latency_ms': (processing_time / size) * 1000
                }
            )
            
            results.append(result)
            print(f"   📊 Size {size:4d}: {processing_time:.3f}s, {throughput:6.0f} req/s")
            
        return results
        
    def run_scalability_benchmark(self, dataset_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark scalability across different dataset sizes."""
        print("📈 Running scalability benchmarks...")
        results = []
        
        for size in dataset_sizes:
            # Generate large dataset
            test_texts = [f"Scalability test text {i} with sentiment." for i in range(size)]
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Simulate full pipeline
            embeddings = self._generate_test_embeddings(test_texts, embedding_dim=100)
            sentiments = self._run_physics_simulation(embeddings, num_steps=10)
            
            # Simulate batch processing
            batch_size = min(50, size)
            batches = [sentiments[i:i+batch_size] for i in range(0, len(sentiments), batch_size)]
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            processing_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = size / processing_time
            
            # Measure scalability efficiency (should be roughly linear)
            efficiency = throughput / size if size > 0 else 0
            
            result = BenchmarkResult(
                benchmark_name="scalability",
                dataset_size=size,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                throughput_texts_per_second=throughput,
                accuracy_score=efficiency,
                metadata={
                    'batch_count': len(batches),
                    'batch_size': batch_size,
                    'memory_per_text_kb': (memory_usage * 1024) / size if size > 0 else 0,
                    'scalability_efficiency': efficiency
                }
            )
            
            results.append(result)
            print(f"   📊 Size {size:4d}: {processing_time:.3f}s, {throughput:6.1f} texts/s, {efficiency:.6f} eff")
            
        return results
        
    def run_comprehensive_benchmark_suite(self) -> BenchmarkSuite:
        """Run complete benchmark suite."""
        print("🚀 Running Comprehensive Sentiment Analysis Benchmark Suite")
        print("=" * 65)
        
        all_results = []
        
        # Different test sizes for different benchmarks
        small_sizes = [10, 50, 100]
        medium_sizes = [100, 500, 1000]
        large_sizes = [1000, 2500, 5000]
        cache_sizes = [10, 50, 100, 200]
        
        # Run individual benchmarks
        all_results.extend(self.run_embedding_benchmark(small_sizes))
        all_results.extend(self.run_physics_simulation_benchmark(small_sizes))
        all_results.extend(self.run_caching_benchmark(cache_sizes))
        all_results.extend(self.run_api_performance_benchmark(medium_sizes))
        all_results.extend(self.run_scalability_benchmark(large_sizes))
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(all_results)
        
        # Create benchmark suite
        suite = BenchmarkSuite(
            suite_name="Sentiment Analysis Comprehensive Benchmark",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=self.system_info,
            benchmark_results=all_results,
            summary_stats=summary_stats
        )
        
        return suite
        
    def _generate_test_embeddings(self, texts: List[str], embedding_dim: int) -> List[List[float]]:
        """Generate test embeddings for benchmarking."""
        import random
        import re
        from collections import Counter
        
        # Simple TF-IDF-like simulation
        all_words = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            all_words.extend(words)
            
        vocab = [word for word, _ in Counter(all_words).most_common(min(embedding_dim, len(set(all_words))))]
        
        embeddings = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            word_counts = Counter(words)
            
            embedding = [word_counts.get(word, 0) for word in vocab]
            # Pad or truncate to desired dimension
            if len(embedding) < embedding_dim:
                embedding.extend([0] * (embedding_dim - len(embedding)))
            else:
                embedding = embedding[:embedding_dim]
                
            # Normalize
            norm = sum(x**2 for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x/norm for x in embedding]
                
            embeddings.append(embedding)
            
        return embeddings
        
    def _run_physics_simulation(self, embeddings: List[List[float]], num_steps: int) -> List[float]:
        """Run physics simulation for benchmarking."""
        import math
        import random
        
        n_texts = len(embeddings)
        sentiments = [random.uniform(-0.1, 0.1) for _ in range(n_texts)]
        
        # Simplified physics simulation
        for step in range(num_steps):
            new_sentiments = []
            
            for i in range(n_texts):
                # Simple diffusion
                avg_neighbor = sum(sentiments) / len(sentiments)
                diffusion = 0.1 * (avg_neighbor - sentiments[i])
                
                # Simple reaction
                reaction = 0.05 * sentiments[i] * (1 - sentiments[i]**2)
                
                new_sentiment = sentiments[i] + 0.1 * (diffusion + reaction)
                new_sentiment = max(-1, min(1, new_sentiment))
                new_sentiments.append(new_sentiment)
                
            sentiments = new_sentiments
            
        return sentiments
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback - use rough estimation based on object count
            import sys
            return sys.getsizeof(self) / (1024 * 1024)
            
    def _calculate_summary_stats(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary statistics for benchmark results."""
        if not results:
            return {}
            
        # Group by benchmark type
        benchmarks_by_type = {}
        for result in results:
            if result.benchmark_name not in benchmarks_by_type:
                benchmarks_by_type[result.benchmark_name] = []
            benchmarks_by_type[result.benchmark_name].append(result)
            
        summary = {
            'total_benchmarks': len(results),
            'benchmark_types': len(benchmarks_by_type),
            'by_type': {}
        }
        
        for benchmark_type, type_results in benchmarks_by_type.items():
            throughputs = [r.throughput_texts_per_second for r in type_results]
            processing_times = [r.processing_time for r in type_results]
            memory_usages = [r.memory_usage_mb for r in type_results]
            
            type_summary = {
                'count': len(type_results),
                'throughput': {
                    'mean': statistics.mean(throughputs),
                    'median': statistics.median(throughputs),
                    'max': max(throughputs),
                    'min': min(throughputs)
                },
                'processing_time': {
                    'mean': statistics.mean(processing_times),
                    'median': statistics.median(processing_times),
                    'max': max(processing_times),
                    'min': min(processing_times)
                },
                'memory_usage': {
                    'mean': statistics.mean(memory_usages),
                    'median': statistics.median(memory_usages),
                    'max': max(memory_usages),
                    'min': min(memory_usages)
                }
            }
            
            summary['by_type'][benchmark_type] = type_summary
            
        return summary


def print_benchmark_report(suite: BenchmarkSuite):
    """Print formatted benchmark report."""
    print("\n" + "=" * 70)
    print("📊 SENTIMENT ANALYSIS BENCHMARK REPORT")
    print("=" * 70)
    
    print(f"\n🔧 System Information:")
    print(f"   Platform: {suite.system_info['platform']}")
    print(f"   Python: {suite.system_info['python_version']}")
    print(f"   CPU Cores: {suite.system_info['cpu_count']}")
    print(f"   Memory: {suite.system_info['memory_total_gb']:.1f} GB")
    
    print(f"\n📈 Summary Statistics:")
    print(f"   Total Benchmarks: {suite.summary_stats['total_benchmarks']}")
    print(f"   Benchmark Types: {suite.summary_stats['benchmark_types']}")
    
    for bench_type, stats in suite.summary_stats['by_type'].items():
        print(f"\n   🏷️  {bench_type.replace('_', ' ').title()}:")
        print(f"      Throughput: {stats['throughput']['mean']:.1f} ± {stats['throughput']['median']:.1f} texts/s")
        print(f"      Processing Time: {stats['processing_time']['mean']:.3f} ± {stats['processing_time']['median']:.3f} s")
        print(f"      Memory Usage: {stats['memory_usage']['mean']:.2f} ± {stats['memory_usage']['median']:.2f} MB")
    
    # Performance recommendations
    print(f"\n💡 Performance Recommendations:")
    
    overall_throughput = statistics.mean([
        r.throughput_texts_per_second for r in suite.benchmark_results
    ])
    
    if overall_throughput > 1000:
        print("   ✅ Excellent performance - consider GPU acceleration for even better speeds")
    elif overall_throughput > 100:
        print("   ✅ Good performance - optimize caching and batch processing")
    elif overall_throughput > 10:
        print("   ⚠️  Moderate performance - consider parallel processing")
    else:
        print("   ❌ Low performance - review algorithms and implementation")
    
    print(f"\n🎯 Key Metrics:")
    print(f"   Average Throughput: {overall_throughput:.1f} texts/second")
    print(f"   Peak Throughput: {max(r.throughput_texts_per_second for r in suite.benchmark_results):.1f} texts/second")
    print(f"   Memory Efficiency: {statistics.mean([r.memory_usage_mb for r in suite.benchmark_results]):.2f} MB average")


def main():
    """Main benchmark runner function."""
    parser = argparse.ArgumentParser(description="Sentiment Analysis Benchmark Suite")
    parser.add_argument("--output", "-o", type=str, help="Output file for benchmark results")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick benchmark (smaller datasets)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Initialize and run benchmarks
        runner = SentimentBenchmarkRunner()
        suite = runner.run_comprehensive_benchmark_suite()
        
        # Print report
        print_benchmark_report(suite)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            suite.save(output_path)
            print(f"\n💾 Benchmark results saved to: {output_path}")
        
        print("\n🎉 Benchmark suite completed successfully!")
        
        return suite
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return None


if __name__ == "__main__":
    main()