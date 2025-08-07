"""Performance optimization utilities for sentiment analysis."""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue
import psutil


@dataclass
class OptimizationConfig:
    """Configuration for sentiment analysis optimization."""
    
    # Parallelization settings
    enable_multiprocessing: bool = True
    max_workers: int = field(default_factory=lambda: min(8, mp.cpu_count()))
    batch_size_threshold: int = 50  # When to use parallel processing
    
    # Memory optimization
    enable_memory_optimization: bool = True
    memory_limit_mb: int = 1024  # Memory limit for caching
    gc_frequency: int = 100  # Garbage collection frequency
    
    # Computation optimization
    enable_jit_compilation: bool = True
    use_gpu_if_available: bool = True
    embedding_precision: str = 'float32'  # float32 or float64
    
    # Batching optimization
    adaptive_batch_sizing: bool = True
    min_batch_size: int = 5
    max_batch_size: int = 200
    target_processing_time: float = 0.1  # Target time per batch in seconds
    
    # Caching optimization
    precompute_common_embeddings: bool = True
    embedding_cache_warmup: bool = True
    analysis_result_caching: bool = True


class ParallelSentimentProcessor:
    """Parallel processor for sentiment analysis tasks."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize parallel processor.
        
        Parameters
        ----------
        config : OptimizationConfig
            Optimization configuration
        """
        self.config = config
        self._executor = None
        self._memory_monitor = MemoryMonitor() if config.enable_memory_optimization else None
        
    def __enter__(self):
        """Enter context manager."""
        if self.config.enable_multiprocessing:
            self._executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self._executor:
            self._executor.shutdown(wait=True)
            
    def process_batches_parallel(
        self,
        text_batches: List[List[str]],
        analyzer_func: Callable,
        **kwargs
    ) -> List[Any]:
        """Process text batches in parallel.
        
        Parameters
        ----------
        text_batches : List[List[str]]
            Batches of texts to process
        analyzer_func : Callable
            Function to analyze each batch
        **kwargs
            Additional arguments for analyzer function
            
        Returns
        -------
        List[Any]
            Results for each batch
        """
        if len(text_batches) < 2 or not self.config.enable_multiprocessing:
            # Process sequentially for small workloads
            return [analyzer_func(batch, **kwargs) for batch in text_batches]
            
        # Submit tasks to executor
        futures = []
        for i, batch in enumerate(text_batches):
            future = self._executor.submit(analyzer_func, batch, **kwargs)
            futures.append((i, future))
            
        # Collect results in order
        results = [None] * len(text_batches)
        for i, future in futures:
            try:
                results[i] = future.result(timeout=300)  # 5 minute timeout
            except Exception as e:
                # Handle failed batch
                print(f"Batch {i} failed: {e}")
                results[i] = None
                
        return results
        
    def process_large_dataset_streaming(
        self,
        texts: List[str],
        analyzer_func: Callable,
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """Process large dataset with streaming approach.
        
        Parameters
        ----------
        texts : List[str]
            Large list of texts
        analyzer_func : Callable
            Analysis function
        chunk_size : int, optional
            Size of chunks to process, auto-determined if None
        **kwargs
            Additional arguments for analyzer function
            
        Returns
        -------
        List[Any]
            Analysis results
        """
        if chunk_size is None:
            chunk_size = self._determine_optimal_chunk_size(len(texts))
            
        results = []
        
        # Process in chunks to manage memory
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            
            # Memory check
            if self._memory_monitor and self._memory_monitor.should_pause():
                time.sleep(1)  # Brief pause to allow memory cleanup
                
            chunk_result = analyzer_func(chunk, **kwargs)
            results.extend(chunk_result if isinstance(chunk_result, list) else [chunk_result])
            
            # Trigger garbage collection periodically
            if i % (chunk_size * self.config.gc_frequency) == 0:
                import gc
                gc.collect()
                
        return results
        
    def _determine_optimal_chunk_size(self, total_size: int) -> int:
        """Determine optimal chunk size based on system resources."""
        # Base chunk size on available memory and CPU cores
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        suggested_chunk = min(
            total_size // (self.config.max_workers * 2),  # Distribute across workers
            int(available_memory_mb / 10),  # Conservative memory usage
            self.config.max_batch_size
        )
        
        return max(suggested_chunk, self.config.min_batch_size)


class MemoryMonitor:
    """Monitor memory usage during sentiment analysis."""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        """Initialize memory monitor.
        
        Parameters
        ----------
        warning_threshold : float, optional
            Memory usage threshold for warnings, by default 0.8
        critical_threshold : float, optional
            Memory usage threshold for critical alerts, by default 0.95
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._last_check = 0
        self._check_interval = 5  # Check every 5 seconds
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'system_usage_percent': memory.percent / 100,
            'system_available_mb': memory.available / (1024 * 1024),
            'process_memory_mb': process.memory_info().rss / (1024 * 1024),
            'process_memory_percent': process.memory_percent() / 100
        }
        
    def should_pause(self) -> bool:
        """Check if processing should pause due to memory pressure."""
        current_time = time.time()
        if current_time - self._last_check < self._check_interval:
            return False
            
        self._last_check = current_time
        usage = self.get_memory_usage()
        
        return (
            usage['system_usage_percent'] > self.critical_threshold or
            usage['process_memory_percent'] > self.critical_threshold
        )
        
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        usage = self.get_memory_usage()
        recommendations = []
        
        if usage['system_usage_percent'] > self.warning_threshold:
            recommendations.append("Reduce batch size to lower memory usage")
            recommendations.append("Enable result streaming instead of batching")
            
        if usage['process_memory_mb'] > 500:
            recommendations.append("Clear embedding cache periodically")
            recommendations.append("Use float32 precision instead of float64")
            
        return recommendations


class AdaptiveBatchSizer:
    """Adaptive batch sizing for optimal performance."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize adaptive batch sizer.
        
        Parameters
        ----------
        config : OptimizationConfig
            Optimization configuration
        """
        self.config = config
        self._performance_history = []
        self._current_batch_size = config.min_batch_size
        
    def get_optimal_batch_size(self, num_texts: int) -> int:
        """Get optimal batch size for given number of texts.
        
        Parameters
        ----------
        num_texts : int
            Number of texts to process
            
        Returns
        -------
        int
            Optimal batch size
        """
        if not self.config.adaptive_batch_sizing:
            return min(self.config.max_batch_size, max(self.config.min_batch_size, num_texts // 4))
            
        # Adapt based on performance history
        if len(self._performance_history) > 3:
            self._adapt_batch_size()
            
        return min(self._current_batch_size, num_texts)
        
    def record_performance(self, batch_size: int, processing_time: float, memory_usage: float):
        """Record performance metrics for batch size adaptation.
        
        Parameters
        ----------
        batch_size : int
            Batch size used
        processing_time : float
            Time taken to process batch
        memory_usage : float
            Memory usage during processing
        """
        self._performance_history.append({
            'batch_size': batch_size,
            'processing_time': processing_time,
            'memory_usage': memory_usage,
            'efficiency': batch_size / processing_time,  # Texts per second
            'timestamp': time.time()
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 3600  # Keep last hour
        self._performance_history = [
            entry for entry in self._performance_history 
            if entry['timestamp'] > cutoff_time
        ]
        
    def _adapt_batch_size(self):
        """Adapt batch size based on performance history."""
        if len(self._performance_history) < 3:
            return
            
        recent_entries = self._performance_history[-3:]
        
        # Calculate average efficiency and memory usage
        avg_efficiency = np.mean([entry['efficiency'] for entry in recent_entries])
        avg_memory = np.mean([entry['memory_usage'] for entry in recent_entries])
        
        # Adapt batch size
        if avg_efficiency > 50 and avg_memory < 0.7:  # Good performance, low memory
            self._current_batch_size = min(
                self.config.max_batch_size,
                int(self._current_batch_size * 1.2)
            )
        elif avg_efficiency < 20 or avg_memory > 0.9:  # Poor performance or high memory
            self._current_batch_size = max(
                self.config.min_batch_size,
                int(self._current_batch_size * 0.8)
            )


class SentimentAnalysisOptimizer:
    """Main optimizer for sentiment analysis performance."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize sentiment analysis optimizer.
        
        Parameters
        ----------
        config : OptimizationConfig, optional
            Optimization configuration, creates default if None
        """
        self.config = config or OptimizationConfig()
        self.batch_sizer = AdaptiveBatchSizer(self.config)
        self.memory_monitor = MemoryMonitor() if self.config.enable_memory_optimization else None
        self._optimization_stats = {
            'total_texts_processed': 0,
            'total_processing_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_warnings': 0
        }
        
    def optimize_analysis_pipeline(
        self,
        analyzer_func: Callable,
        texts: List[str],
        **kwargs
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Optimize the sentiment analysis pipeline.
        
        Parameters
        ----------
        analyzer_func : Callable
            Sentiment analysis function to optimize
        texts : List[str]
            Texts to analyze
        **kwargs
            Additional arguments for analyzer function
            
        Returns
        -------
        Tuple[List[Any], Dict[str, Any]]
            Analysis results and optimization statistics
        """
        start_time = time.time()
        
        # Determine processing strategy
        if len(texts) < self.config.batch_size_threshold:
            # Process directly for small datasets
            results = [analyzer_func(texts, **kwargs)]
            if isinstance(results[0], list):
                results = results[0]
        else:
            # Use optimized processing for large datasets
            results = self._process_large_dataset_optimized(analyzer_func, texts, **kwargs)
            
        total_time = time.time() - start_time
        
        # Update statistics
        self._optimization_stats['total_texts_processed'] += len(texts)
        self._optimization_stats['total_processing_time'] += total_time
        
        # Generate optimization report
        optimization_stats = {
            'processing_time': total_time,
            'texts_processed': len(texts),
            'throughput_texts_per_second': len(texts) / total_time,
            'memory_usage': self.memory_monitor.get_memory_usage() if self.memory_monitor else {},
            'batch_configuration': {
                'adaptive_sizing': self.config.adaptive_batch_sizing,
                'parallel_processing': self.config.enable_multiprocessing,
                'max_workers': self.config.max_workers
            },
            'recommendations': self._get_optimization_recommendations()
        }
        
        return results, optimization_stats
        
    def _process_large_dataset_optimized(
        self,
        analyzer_func: Callable,
        texts: List[str],
        **kwargs
    ) -> List[Any]:
        """Process large dataset with all optimizations enabled."""
        # Determine optimal batch size
        optimal_batch_size = self.batch_sizer.get_optimal_batch_size(len(texts))
        
        # Create batches
        text_batches = [
            texts[i:i + optimal_batch_size]
            for i in range(0, len(texts), optimal_batch_size)
        ]
        
        # Process with parallelization if enabled
        with ParallelSentimentProcessor(self.config) as processor:
            if len(text_batches) > 1 and self.config.enable_multiprocessing:
                batch_results = processor.process_batches_parallel(
                    text_batches, analyzer_func, **kwargs
                )
            else:
                batch_results = []
                for batch in text_batches:
                    batch_start = time.time()
                    
                    # Memory check
                    if self.memory_monitor and self.memory_monitor.should_pause():
                        time.sleep(0.5)
                        self._optimization_stats['memory_warnings'] += 1
                        
                    result = analyzer_func(batch, **kwargs)
                    batch_time = time.time() - batch_start
                    
                    # Record performance for adaptation
                    memory_usage = 0.5  # Placeholder, would get actual usage
                    self.batch_sizer.record_performance(
                        len(batch), batch_time, memory_usage
                    )
                    
                    batch_results.append(result)
        
        # Flatten results
        final_results = []
        for batch_result in batch_results:
            if batch_result is not None:
                if isinstance(batch_result, list):
                    final_results.extend(batch_result)
                else:
                    final_results.append(batch_result)
                    
        return final_results
        
    def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current performance."""
        recommendations = []
        
        stats = self._optimization_stats
        if stats['total_texts_processed'] > 0:
            avg_throughput = stats['total_texts_processed'] / stats['total_processing_time']
            
            if avg_throughput < 10:
                recommendations.append("Enable multiprocessing for better throughput")
                recommendations.append("Consider increasing batch size")
                
            if avg_throughput > 1000:
                recommendations.append("Consider enabling GPU acceleration if available")
                
        if self.memory_monitor:
            memory_recs = self.memory_monitor.get_memory_recommendations()
            recommendations.extend(memory_recs)
            
        if stats['memory_warnings'] > 5:
            recommendations.append("Reduce batch size to prevent memory issues")
            
        return recommendations
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        stats = self._optimization_stats.copy()
        
        if stats['total_processing_time'] > 0:
            stats['average_throughput'] = stats['total_texts_processed'] / stats['total_processing_time']
        else:
            stats['average_throughput'] = 0
            
        stats['current_batch_size'] = self.batch_sizer._current_batch_size
        stats['optimization_config'] = {
            'multiprocessing_enabled': self.config.enable_multiprocessing,
            'max_workers': self.config.max_workers,
            'adaptive_batching': self.config.adaptive_batch_sizing,
            'memory_optimization': self.config.enable_memory_optimization
        }
        
        return stats


def optimize_sentiment_analyzer_factory(config: Optional[OptimizationConfig] = None):
    """Factory function to create optimized sentiment analyzer.
    
    Parameters
    ----------
    config : OptimizationConfig, optional
        Optimization configuration
        
    Returns
    -------
    Callable
        Optimized analyzer function
    """
    optimizer = SentimentAnalysisOptimizer(config)
    
    def optimized_analyzer(analyzer_instance, texts: List[str], **kwargs):
        """Optimized wrapper around sentiment analyzer."""
        
        def analyzer_func(text_batch, **analysis_kwargs):
            """Internal analyzer function for optimization."""
            return analyzer_instance.analyze(text_batch, **analysis_kwargs)
            
        results, stats = optimizer.optimize_analysis_pipeline(
            analyzer_func, texts, **kwargs
        )
        
        # Return results with optimization statistics
        if kwargs.get('return_optimization_stats', False):
            return results, stats
        else:
            return results
            
    return optimized_analyzer


# Context manager for performance optimization
class PerformanceOptimizationContext:
    """Context manager for sentiment analysis performance optimization."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.optimizer = SentimentAnalysisOptimizer(self.config)
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self.optimizer
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"Performance optimization context completed in {total_time:.2f} seconds")
            
            # Print performance report
            report = self.optimizer.get_performance_report()
            if report['total_texts_processed'] > 0:
                print(f"Processed {report['total_texts_processed']} texts")
                print(f"Average throughput: {report['average_throughput']:.1f} texts/second")


# GPU acceleration utilities (placeholder for JAX/PyTorch integration)
def enable_gpu_acceleration():
    """Enable GPU acceleration if available."""
    try:
        import jax
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.device_kind == 'gpu']
        if gpu_devices:
            print(f"GPU acceleration enabled with {len(gpu_devices)} GPU(s)")
            return True
    except ImportError:
        pass
        
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU acceleration enabled with {torch.cuda.device_count()} GPU(s)")
            return True
    except ImportError:
        pass
        
    print("GPU acceleration not available")
    return False