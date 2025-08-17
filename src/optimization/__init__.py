"""Performance optimization modules for DiffFE-Physics-Lab."""

from .caching import (
    CacheManager, LRUCache, MemoryCache, RedisCache,
    cache_result, adaptive_cache
)
from .parallel_processing import (
    ParallelExecutor, ThreadPoolManager, ProcessPoolManager,
    parallel_map, concurrent_solve, auto_parallelize
)
from .performance_tuning import (
    PerformanceTuner, ProfileAnalyzer, BenchmarkSuite,
    optimize_function, profile_performance
)
from .auto_scaling import (
    AutoScaler, ResourceManager, LoadBalancer,
    scale_workers, adaptive_resources
)
from .memory_optimization import (
    MemoryManager, ObjectPool, LazyLoader,
    optimize_memory, memory_efficient
)

__all__ = [
    # Caching
    "CacheManager", "LRUCache", "MemoryCache", "RedisCache",
    "cache_result", "adaptive_cache",
    
    # Parallel processing
    "ParallelExecutor", "ThreadPoolManager", "ProcessPoolManager",
    "parallel_map", "concurrent_solve", "auto_parallelize",
    
    # Performance tuning
    "PerformanceTuner", "ProfileAnalyzer", "BenchmarkSuite",
    "optimize_function", "profile_performance",
    
    # Auto-scaling
    "AutoScaler", "ResourceManager", "LoadBalancer",
    "scale_workers", "adaptive_resources",
    
    # Memory optimization
    "MemoryManager", "ObjectPool", "LazyLoader",
    "optimize_memory", "memory_efficient",
]