"""Advanced caching system with enterprise-scale features."""

import asyncio
import hashlib
import logging
import pickle
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

try:
    import pymemcache

    MEMCACHED_AVAILABLE = True
except ImportError:
    pymemcache = None
    MEMCACHED_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AdvancedCacheEntry:
    """Enhanced cache entry with enterprise features."""

    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    ttl: Optional[float] = None
    size_bytes: int = 0
    priority: float = 1.0
    dependencies: Set[str] = field(default_factory=set)
    compute_cost: float = 1.0
    access_pattern: str = "random"  # random, sequential, temporal
    compression_ratio: float = 1.0
    serialization_time: float = 0.0

    def __post_init__(self):
        if self.last_access == 0.0:
            self.last_access = self.timestamp


@dataclass
class CacheInvalidationRule:
    """Cache invalidation rule configuration."""

    pattern: str
    dependencies: Set[str] = field(default_factory=set)
    condition: Optional[Callable] = None
    cascade: bool = True


class AssemblyMatrixCache:
    """Specialized cache for finite element assembly matrices.

    Features:
    - Sparse matrix optimized storage
    - Mesh-aware caching keys
    - Operator dependency tracking
    - Automatic invalidation on mesh changes
    """

    def __init__(self, max_memory_gb: float = 2.0):
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self.cache = OrderedDict()
        self.memory_usage = 0
        self.mesh_versions = {}  # mesh_id -> version
        self.operator_cache = {}  # operator_id -> cached matrices
        self._lock = threading.RLock()

        logger.info(
            f"Assembly matrix cache initialized (max memory: {max_memory_gb:.1f}GB)"
        )

    def _generate_matrix_key(
        self, mesh_id: str, operator_type: str, params: Dict
    ) -> str:
        """Generate cache key for assembly matrix."""
        key_data = {
            "mesh_id": mesh_id,
            "operator": operator_type,
            "params": sorted(params.items()),
            "mesh_version": self.mesh_versions.get(mesh_id, 0),
        }
        key_str = pickle.dumps(key_data)
        return f"matrix_{hashlib.md5(key_str).hexdigest()}"

    def get_matrix(
        self, mesh_id: str, operator_type: str, params: Dict
    ) -> Optional[Any]:
        """Get cached assembly matrix."""
        with self._lock:
            key = self._generate_matrix_key(mesh_id, operator_type, params)

            if key not in self.cache:
                return None

            entry = self.cache[key]

            # Check if mesh has been updated
            if self.mesh_versions.get(mesh_id, 0) != entry.get("mesh_version", 0):
                del self.cache[key]
                self.memory_usage -= entry["size_bytes"]
                return None

            # Update access statistics
            entry["access_count"] += 1
            entry["last_access"] = time.time()

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            return entry["matrix"]

    def set_matrix(self, mesh_id: str, operator_type: str, params: Dict, matrix: Any):
        """Cache assembly matrix."""
        with self._lock:
            key = self._generate_matrix_key(mesh_id, operator_type, params)

            # Estimate matrix size
            try:
                if hasattr(matrix, "data"):  # Sparse matrix
                    size_bytes = (
                        matrix.data.nbytes
                        + matrix.indices.nbytes
                        + matrix.indptr.nbytes
                    )
                elif hasattr(matrix, "nbytes"):  # Dense array
                    size_bytes = matrix.nbytes
                else:
                    size_bytes = len(pickle.dumps(matrix))
            except:
                size_bytes = 1024 * 1024  # 1MB fallback

            # Ensure memory limit
            while self.memory_usage + size_bytes > self.max_memory_bytes and self.cache:
                # Remove least recently used entry
                old_key, old_entry = self.cache.popitem(last=False)
                self.memory_usage -= old_entry["size_bytes"]
                logger.debug(f"Evicted matrix cache entry: {old_key}")

            # Store matrix with metadata
            entry = {
                "matrix": matrix,
                "mesh_version": self.mesh_versions.get(mesh_id, 0),
                "timestamp": time.time(),
                "access_count": 0,
                "last_access": time.time(),
                "size_bytes": size_bytes,
                "operator_type": operator_type,
                "mesh_id": mesh_id,
            }

            self.cache[key] = entry
            self.memory_usage += size_bytes

            # Track operator usage
            if operator_type not in self.operator_cache:
                self.operator_cache[operator_type] = []
            self.operator_cache[operator_type].append(key)

    def invalidate_mesh(self, mesh_id: str):
        """Invalidate all cached matrices for a specific mesh."""
        with self._lock:
            # Update mesh version
            self.mesh_versions[mesh_id] = self.mesh_versions.get(mesh_id, 0) + 1

            # Remove cached entries for this mesh
            keys_to_remove = []
            for key, entry in self.cache.items():
                if entry.get("mesh_id") == mesh_id:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                entry = self.cache.pop(key)
                self.memory_usage -= entry["size_bytes"]

            logger.info(
                f"Invalidated {len(keys_to_remove)} matrix cache entries for mesh {mesh_id}"
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "total_entries": len(self.cache),
                "memory_usage_mb": self.memory_usage / (1024 * 1024),
                "memory_limit_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_utilization": self.memory_usage / self.max_memory_bytes,
                "operator_counts": {
                    op: len(keys) for op, keys in self.operator_cache.items()
                },
                "mesh_versions": dict(self.mesh_versions),
            }


class DistributedCacheBackend:
    """Distributed cache backend using Redis."""

    def __init__(self, redis_config: Dict[str, Any], key_prefix: str = "pde_solver"):
        if not REDIS_AVAILABLE:
            raise RuntimeError(
                "Redis not available. Install redis-py: pip install redis"
            )

        self.redis_client = redis.Redis(**redis_config)
        self.key_prefix = key_prefix
        self.serializer = pickle
        self.compression_threshold = 1024  # bytes

        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis cache backend")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _make_key(self, key: str) -> str:
        """Create prefixed key for Redis."""
        return f"{self.key_prefix}:{key}"

    def get(self, key: str) -> Optional[AdvancedCacheEntry]:
        """Get value from Redis cache."""
        try:
            redis_key = self._make_key(key)
            data = self.redis_client.get(redis_key)

            if data is None:
                return None

            # Deserialize entry
            entry_data = self.serializer.loads(data)

            # Check TTL
            if (
                entry_data.get("ttl")
                and time.time() - entry_data["timestamp"] > entry_data["ttl"]
            ):
                self.redis_client.delete(redis_key)
                return None

            # Create entry object
            entry = AdvancedCacheEntry(**entry_data)
            entry.access_count += 1
            entry.last_access = time.time()

            # Update access statistics in Redis
            entry_data.update(
                {"access_count": entry.access_count, "last_access": entry.last_access}
            )
            self.redis_client.set(redis_key, self.serializer.dumps(entry_data))

            return entry

        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None

    def set(self, key: str, entry: AdvancedCacheEntry) -> None:
        """Set value in Redis cache."""
        try:
            redis_key = self._make_key(key)

            # Serialize entry
            entry_data = {
                "value": entry.value,
                "timestamp": entry.timestamp,
                "access_count": entry.access_count,
                "last_access": entry.last_access,
                "ttl": entry.ttl,
                "size_bytes": entry.size_bytes,
                "priority": entry.priority,
                "dependencies": list(entry.dependencies),
                "compute_cost": entry.compute_cost,
                "access_pattern": entry.access_pattern,
            }

            serialized_data = self.serializer.dumps(entry_data)

            # Set in Redis with TTL if specified
            if entry.ttl:
                self.redis_client.setex(redis_key, int(entry.ttl), serialized_data)
            else:
                self.redis_client.set(redis_key, serialized_data)

        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")

    def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            redis_key = self._make_key(key)
            result = self.redis_client.delete(redis_key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            pattern = f"{self.key_prefix}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} Redis cache entries")
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    def keys(self) -> List[str]:
        """Get all cache keys."""
        try:
            pattern = f"{self.key_prefix}:*"
            redis_keys = self.redis_client.keys(pattern)
            # Remove prefix from keys
            return [
                key.decode("utf-8").replace(f"{self.key_prefix}:", "")
                for key in redis_keys
            ]
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []


class JITCompiledOperatorCache:
    """Cache for JIT-compiled operators with hot/cold classification."""

    def __init__(self, hot_threshold: int = 5, compilation_cache_size: int = 100):
        self.hot_threshold = hot_threshold
        self.compilation_cache = OrderedDict()
        self.compilation_cache_size = compilation_cache_size
        self.operator_stats = defaultdict(
            lambda: {"calls": 0, "compile_time": 0.0, "avg_runtime": 0.0}
        )
        self.hot_operators = set()
        self.cold_operators = set()
        self._lock = threading.RLock()

        logger.info(f"JIT operator cache initialized (hot threshold: {hot_threshold})")

    def get_compiled_operator(
        self, operator_key: str, compile_func: Callable
    ) -> Callable:
        """Get or compile operator with JIT caching."""
        with self._lock:
            # Check if already compiled
            if operator_key in self.compilation_cache:
                compiled_op = self.compilation_cache[operator_key]
                # Move to end (most recently used)
                self.compilation_cache.move_to_end(operator_key)

                # Update stats
                self.operator_stats[operator_key]["calls"] += 1

                # Promote to hot if threshold reached
                if (
                    self.operator_stats[operator_key]["calls"] >= self.hot_threshold
                    and operator_key not in self.hot_operators
                ):
                    self.hot_operators.add(operator_key)
                    self.cold_operators.discard(operator_key)
                    logger.debug(f"Promoted operator to hot: {operator_key}")

                return compiled_op

            # Compile operator
            start_time = time.time()

            try:
                # Use JAX JIT compilation
                if hasattr(compile_func, "__call__"):
                    compiled_op = jax.jit(compile_func, static_argnums=(0,))
                else:
                    compiled_op = compile_func

                compile_time = time.time() - start_time

                # Cache compiled operator
                if len(self.compilation_cache) >= self.compilation_cache_size:
                    # Remove least recently used
                    old_key, _ = self.compilation_cache.popitem(last=False)
                    self.cold_operators.discard(old_key)
                    logger.debug(f"Evicted compiled operator: {old_key}")

                self.compilation_cache[operator_key] = compiled_op

                # Update statistics
                stats = self.operator_stats[operator_key]
                stats["calls"] += 1
                stats["compile_time"] = compile_time

                # Initially mark as cold
                self.cold_operators.add(operator_key)

                logger.debug(
                    f"Compiled and cached operator: {operator_key} (compile time: {compile_time:.3f}s)"
                )

                return compiled_op

            except Exception as e:
                logger.error(f"Failed to compile operator {operator_key}: {e}")
                raise

    def precompile_hot_operators(self, operator_definitions: Dict[str, Callable]):
        """Precompile frequently used operators."""
        logger.info(f"Precompiling {len(operator_definitions)} operators...")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for op_key, compile_func in operator_definitions.items():
                future = executor.submit(
                    self.get_compiled_operator, op_key, compile_func
                )
                futures.append((op_key, future))

            for op_key, future in futures:
                try:
                    future.result(timeout=30.0)  # 30 second timeout per compilation
                    logger.debug(f"Precompiled operator: {op_key}")
                except Exception as e:
                    logger.warning(f"Failed to precompile operator {op_key}: {e}")

        logger.info("Operator precompilation completed")

    def get_operator_stats(self) -> Dict[str, Any]:
        """Get operator compilation and usage statistics."""
        with self._lock:
            return {
                "total_operators": len(self.compilation_cache),
                "hot_operators": len(self.hot_operators),
                "cold_operators": len(self.cold_operators),
                "cache_utilization": len(self.compilation_cache)
                / self.compilation_cache_size,
                "operator_stats": dict(self.operator_stats),
                "hot_operator_list": list(self.hot_operators),
                "compilation_cache_size": self.compilation_cache_size,
            }


class AdaptiveCacheManager:
    """Adaptive cache manager that adjusts strategies based on workload patterns."""

    def __init__(
        self,
        initial_max_size: int = 1000,
        initial_max_memory_mb: int = 100,
        adaptation_interval: float = 60.0,  # seconds
        enable_distributed: bool = False,
        redis_config: Optional[Dict] = None,
    ):
        self.max_size = initial_max_size
        self.max_memory_mb = initial_max_memory_mb
        self.adaptation_interval = adaptation_interval

        # Initialize backends
        self.memory_cache = self._create_memory_cache()
        self.assembly_cache = AssemblyMatrixCache(max_memory_gb=2.0)
        self.jit_cache = JITCompiledOperatorCache()

        # Distributed caching
        self.distributed_cache = None
        if enable_distributed and redis_config and REDIS_AVAILABLE:
            try:
                self.distributed_cache = DistributedCacheBackend(redis_config)
                logger.info("Distributed caching enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize distributed cache: {e}")

        # Performance monitoring
        self.hit_rates = deque(maxlen=100)
        self.miss_rates = deque(maxlen=100)
        self.eviction_rates = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)

        # Adaptation state
        self.last_adaptation = time.time()
        self.adaptation_lock = threading.RLock()
        self.performance_history = defaultdict(list)

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "adaptations": 0,
            "errors": 0,
        }

        # Start adaptation thread
        self.adaptation_thread = threading.Thread(
            target=self._adaptation_loop, daemon=True
        )
        self.adaptation_thread.start()

        logger.info("Adaptive cache manager initialized")

    def _create_memory_cache(self):
        """Create memory cache backend."""
        from .cache import MemoryCache

        return MemoryCache(self.max_size, self.max_memory_mb)

    def get(self, key: str, cache_type: str = "general") -> Any:
        """Get value from appropriate cache."""
        start_time = time.time()

        try:
            result = None

            if cache_type == "assembly":
                # Assembly matrix cache
                parts = key.split(":")
                if len(parts) >= 3:
                    mesh_id, operator_type = parts[1], parts[2]
                    params = {} if len(parts) < 4 else eval(parts[3])
                    result = self.assembly_cache.get_matrix(
                        mesh_id, operator_type, params
                    )

            elif cache_type == "distributed" and self.distributed_cache:
                # Distributed cache
                entry = self.distributed_cache.get(key)
                result = entry.value if entry else None

            else:
                # Memory cache
                entry = self.memory_cache.get(key)
                result = entry.value if entry else None

            # Update statistics
            if result is not None:
                self.stats["hits"] += 1
                access_time = time.time() - start_time
                self.performance_history["hit_times"].append(access_time)
            else:
                self.stats["misses"] += 1

            return result

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats["errors"] += 1
            return None

    def set(self, key: str, value: Any, cache_type: str = "general", **kwargs):
        """Set value in appropriate cache."""
        try:
            if cache_type == "assembly":
                # Assembly matrix cache
                mesh_id = kwargs.get("mesh_id", "default")
                operator_type = kwargs.get("operator_type", "unknown")
                params = kwargs.get("params", {})
                self.assembly_cache.set_matrix(mesh_id, operator_type, params, value)

            elif cache_type == "distributed" and self.distributed_cache:
                # Distributed cache
                entry = AdvancedCacheEntry(
                    value=value,
                    timestamp=time.time(),
                    ttl=kwargs.get("ttl"),
                    priority=kwargs.get("priority", 1.0),
                    compute_cost=kwargs.get("compute_cost", 1.0),
                )
                self.distributed_cache.set(key, entry)

            else:
                # Memory cache
                from .cache import CacheEntry

                entry = CacheEntry(
                    value=value, timestamp=time.time(), ttl=kwargs.get("ttl")
                )
                self.memory_cache.set(key, entry)

            self.stats["sets"] += 1

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.stats["errors"] += 1

    def get_compiled_operator(
        self, operator_key: str, compile_func: Callable
    ) -> Callable:
        """Get JIT compiled operator."""
        return self.jit_cache.get_compiled_operator(operator_key, compile_func)

    def _adaptation_loop(self):
        """Background adaptation loop."""
        while True:
            try:
                time.sleep(self.adaptation_interval)
                self._adapt_cache_settings()
            except Exception as e:
                logger.error(f"Cache adaptation error: {e}")

    def _adapt_cache_settings(self):
        """Adapt cache settings based on performance metrics."""
        with self.adaptation_lock:
            current_time = time.time()

            # Calculate performance metrics
            total_requests = self.stats["hits"] + self.stats["misses"]
            if total_requests == 0:
                return

            hit_rate = self.stats["hits"] / total_requests
            self.hit_rates.append(hit_rate)

            # Get memory usage
            if hasattr(self.memory_cache, "total_size_bytes"):
                memory_usage_ratio = self.memory_cache.total_size_bytes / (
                    self.max_memory_mb * 1024 * 1024
                )
                self.memory_usage_history.append(memory_usage_ratio)

            # Adaptive adjustments
            avg_hit_rate = sum(self.hit_rates) / len(self.hit_rates)
            avg_memory_usage = (
                sum(self.memory_usage_history) / len(self.memory_usage_history)
                if self.memory_usage_history
                else 0.5
            )

            # Adjust cache size based on performance
            if avg_hit_rate < 0.7 and avg_memory_usage < 0.8:
                # Low hit rate, increase cache size
                new_max_size = min(self.max_size * 1.2, 10000)
                new_max_memory = min(self.max_memory_mb * 1.1, 1000)

                if (
                    new_max_size != self.max_size
                    or new_max_memory != self.max_memory_mb
                ):
                    logger.info(
                        f"Increasing cache limits: size {self.max_size} -> {new_max_size}, "
                        f"memory {self.max_memory_mb}MB -> {new_max_memory}MB"
                    )

                    self.max_size = int(new_max_size)
                    self.max_memory_mb = int(new_max_memory)
                    self._recreate_memory_cache()

            elif avg_hit_rate > 0.9 and avg_memory_usage > 0.9:
                # High hit rate but high memory usage, could reduce cache size
                new_max_size = max(self.max_size * 0.9, 100)
                new_max_memory = max(self.max_memory_mb * 0.9, 10)

                if (
                    new_max_size != self.max_size
                    or new_max_memory != self.max_memory_mb
                ):
                    logger.info(
                        f"Reducing cache limits: size {self.max_size} -> {new_max_size}, "
                        f"memory {self.max_memory_mb}MB -> {new_max_memory}MB"
                    )

                    self.max_size = int(new_max_size)
                    self.max_memory_mb = int(new_max_memory)
                    self._recreate_memory_cache()

            self.stats["adaptations"] += 1
            self.last_adaptation = current_time

    def _recreate_memory_cache(self):
        """Recreate memory cache with new settings."""
        old_cache = self.memory_cache
        self.memory_cache = self._create_memory_cache()

        # Copy hot entries to new cache
        if hasattr(old_cache, "cache"):
            for key, entry in list(old_cache.cache.items())[-self.max_size // 2 :]:
                try:
                    self.memory_cache.set(key, entry)
                except:
                    pass

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.adaptation_lock:
            stats = {
                "general_cache": self.stats.copy(),
                "assembly_cache": self.assembly_cache.get_cache_stats(),
                "jit_cache": self.jit_cache.get_operator_stats(),
                "performance": {
                    "avg_hit_rate": (
                        sum(self.hit_rates) / len(self.hit_rates)
                        if self.hit_rates
                        else 0
                    ),
                    "avg_memory_usage": (
                        sum(self.memory_usage_history) / len(self.memory_usage_history)
                        if self.memory_usage_history
                        else 0
                    ),
                    "adaptations_count": self.stats["adaptations"],
                    "last_adaptation": self.last_adaptation,
                },
                "configuration": {
                    "max_size": self.max_size,
                    "max_memory_mb": self.max_memory_mb,
                    "adaptation_interval": self.adaptation_interval,
                    "distributed_enabled": self.distributed_cache is not None,
                },
            }

            if self.distributed_cache:
                try:
                    stats["distributed_cache"] = {
                        "connected": True,
                        "keys_count": len(self.distributed_cache.keys()),
                    }
                except:
                    stats["distributed_cache"] = {"connected": False}

            return stats

    def precompile_operators(self, operator_definitions: Dict[str, Callable]):
        """Precompile operators for better performance."""
        self.jit_cache.precompile_hot_operators(operator_definitions)

    def invalidate_pattern(self, pattern: str, cache_type: str = "general") -> int:
        """Invalidate cache entries matching pattern."""
        if cache_type == "assembly":
            # Assembly cache pattern invalidation would need custom implementation
            return 0
        elif cache_type == "distributed" and self.distributed_cache:
            # Distributed cache pattern invalidation
            keys = self.distributed_cache.keys()
            import fnmatch

            matching_keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]

            for key in matching_keys:
                self.distributed_cache.delete(key)

            return len(matching_keys)
        else:
            # Memory cache pattern invalidation
            if hasattr(self.memory_cache, "keys"):
                keys = self.memory_cache.keys()
                import fnmatch

                matching_keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]

                for key in matching_keys:
                    self.memory_cache.delete(key)

                return len(matching_keys)

            return 0

    def shutdown(self):
        """Shutdown cache manager and cleanup resources."""
        logger.info("Shutting down adaptive cache manager...")

        # Stop adaptation thread
        if hasattr(self, "adaptation_thread") and self.adaptation_thread.is_alive():
            # Note: daemon thread will terminate with main process
            pass

        # Clear caches
        if hasattr(self.memory_cache, "clear"):
            self.memory_cache.clear()

        self.assembly_cache.cache.clear()
        self.jit_cache.compilation_cache.clear()

        if self.distributed_cache:
            # Don't clear distributed cache as it may be shared
            pass

        logger.info("Cache manager shutdown completed")


# Global adaptive cache manager
_global_adaptive_cache = None


def get_adaptive_cache() -> AdaptiveCacheManager:
    """Get global adaptive cache manager."""
    global _global_adaptive_cache
    if _global_adaptive_cache is None:
        _global_adaptive_cache = AdaptiveCacheManager()
    return _global_adaptive_cache


def cached_assembly_matrix(mesh_id: str, operator_type: str, params: Dict):
    """Decorator for caching assembly matrices."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_adaptive_cache()

            # Try to get from cache
            matrix = cache.get(
                f"assembly:{mesh_id}:{operator_type}:{str(params)}",
                cache_type="assembly",
            )

            if matrix is not None:
                return matrix

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(
                f"assembly:{mesh_id}:{operator_type}:{str(params)}",
                result,
                cache_type="assembly",
                mesh_id=mesh_id,
                operator_type=operator_type,
                params=params,
            )

            return result

        return wrapper

    return decorator


def jit_compiled_operator(operator_key: str):
    """Decorator for JIT-compiled operators."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_adaptive_cache()
            compiled_func = cache.get_compiled_operator(operator_key, func)
            return compiled_func(*args, **kwargs)

        return wrapper

    return decorator
