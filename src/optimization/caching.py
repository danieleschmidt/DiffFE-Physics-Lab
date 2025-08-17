"""Advanced caching system for DiffFE-Physics-Lab."""

import time
import hashlib
import pickle
import threading
import weakref
from typing import Any, Dict, Optional, Callable, Union, Tuple
from collections import OrderedDict
from functools import wraps
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update access information."""
        self.access_count += 1


class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
        
        logger.info(f"LRU cache initialized: max_size={max_size}, ttl={ttl}")
    
    def _compute_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            return 100  # Default estimate
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.stats["misses"] += 1
                self.stats["evictions"] += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.touch()
            self.stats["hits"] += 1
            
            logger.debug(f"Cache hit for key: {key}")
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (overrides default)
        """
        with self.lock:
            # Create cache entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=self._compute_size(value),
                ttl=ttl or self.ttl
            )
            
            # Remove existing entry if present
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = entry
            self.stats["size"] += entry.size_bytes
            
            # Evict oldest entries if necessary
            while len(self.cache) > self.max_size:
                oldest_key, oldest_entry = self.cache.popitem(last=False)
                self.stats["size"] -= oldest_entry.size_bytes
                self.stats["evictions"] += 1
                logger.debug(f"Evicted cache entry: {oldest_key}")
            
            logger.debug(f"Cached value for key: {key}")
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if key was present
        """
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.stats["size"] -= entry.size_bytes
                logger.debug(f"Invalidated cache entry: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.stats["size"] = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / max(total_requests, 1)
            
            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self.stats["evictions"],
                "entries": len(self.cache),
                "size_bytes": self.stats["size"],
                "max_size": self.max_size
            }


class MemoryCache:
    """Simple in-memory cache with automatic cleanup."""
    
    def __init__(self, cleanup_interval: float = 300.0):
        """Initialize memory cache.
        
        Args:
            cleanup_interval: Cleanup interval in seconds
        """
        self.cache = {}
        self.lock = threading.RLock()
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
    
    def _cleanup(self):
        """Remove expired entries."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            self.last_cleanup = current_time
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self._cleanup()
        
        with self.lock:
            entry = self.cache.get(key)
            if entry and not entry.is_expired():
                entry.touch()
                return entry.value
            elif entry:
                del self.cache[key]
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache."""
        with self.lock:
            self.cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        with self.lock:
            return self.cache.pop(key, None) is not None
    
    def clear(self):
        """Clear all entries."""
        with self.lock:
            self.cache.clear()


class RedisCache:
    """Redis-backed cache (mock implementation for demonstration)."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, prefix: str = "diffhe:"):
        """Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            prefix: Key prefix
        """
        self.host = host
        self.port = port
        self.db = db
        self.prefix = prefix
        self.connected = False
        
        # Mock Redis connection
        self.memory_fallback = MemoryCache()
        logger.info(f"Redis cache initialized (using memory fallback)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        # Use memory fallback for demonstration
        return self.memory_fallback.get(f"{self.prefix}{key}")
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in Redis cache."""
        # Use memory fallback for demonstration
        self.memory_fallback.put(f"{self.prefix}{key}", value, ttl)
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        return self.memory_fallback.invalidate(f"{self.prefix}{key}")
    
    def clear(self):
        """Clear all entries."""
        self.memory_fallback.clear()


class CacheManager:
    """Unified cache management system."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.caches = {}
        self.default_cache = LRUCache(max_size=1000, ttl=3600.0)
        self.cache_policies = {}
        
        logger.info("Cache manager initialized")
    
    def register_cache(self, name: str, cache_instance: Union[LRUCache, MemoryCache, RedisCache]):
        """Register a cache instance.
        
        Args:
            name: Cache name
            cache_instance: Cache instance
        """
        self.caches[name] = cache_instance
        logger.info(f"Registered cache: {name}")
    
    def set_cache_policy(self, pattern: str, cache_name: str):
        """Set cache policy for key pattern.
        
        Args:
            pattern: Key pattern
            cache_name: Cache to use for this pattern
        """
        self.cache_policies[pattern] = cache_name
        logger.info(f"Set cache policy: {pattern} -> {cache_name}")
    
    def _select_cache(self, key: str):
        """Select appropriate cache for key."""
        for pattern, cache_name in self.cache_policies.items():
            if pattern in key:
                return self.caches.get(cache_name, self.default_cache)
        return self.default_cache
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from appropriate cache."""
        cache = self._select_cache(key)
        return cache.get(key)
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in appropriate cache."""
        cache = self._select_cache(key)
        cache.put(key, value, ttl)
    
    def invalidate(self, key: str) -> bool:
        """Invalidate key from all caches."""
        invalidated = False
        for cache in [self.default_cache] + list(self.caches.values()):
            if cache.invalidate(key):
                invalidated = True
        return invalidated
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches."""
        stats = {"default": self.default_cache.get_stats()}
        for name, cache in self.caches.items():
            if hasattr(cache, 'get_stats'):
                stats[name] = cache.get_stats()
        return stats


# Global cache manager
global_cache_manager = CacheManager()


def cache_key_generator(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Create a deterministic key from arguments
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    
    # Serialize and hash
    key_str = str(key_data)
    return hashlib.md5(key_str.encode()).hexdigest()


def cache_result(ttl: Optional[float] = None, cache_name: Optional[str] = None,
                key_func: Optional[Callable] = None):
    """Decorator to cache function results.
    
    Args:
        ttl: Time to live for cached result
        cache_name: Specific cache to use
        key_func: Custom key generation function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{cache_key_generator(*args, **kwargs)}"
            
            # Try to get from cache
            if cache_name:
                cache = global_cache_manager.caches.get(cache_name)
                if cache:
                    result = cache.get(cache_key)
                else:
                    result = global_cache_manager.get(cache_key)
            else:
                result = global_cache_manager.get(cache_key)
            
            if result is not None:
                logger.debug(f"Cache hit for function: {func.__name__}")
                return result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for function: {func.__name__}, executing")
            result = func(*args, **kwargs)
            
            # Cache the result
            if cache_name:
                cache = global_cache_manager.caches.get(cache_name)
                if cache:
                    cache.put(cache_key, result, ttl)
                else:
                    global_cache_manager.put(cache_key, result, ttl)
            else:
                global_cache_manager.put(cache_key, result, ttl)
            
            return result
        
        # Add cache control methods
        wrapper.invalidate_cache = lambda *args, **kwargs: global_cache_manager.invalidate(
            f"{func.__module__}.{func.__name__}:{cache_key_generator(*args, **kwargs)}"
        )
        
        return wrapper
    return decorator


class AdaptiveCache:
    """Cache that adapts based on access patterns."""
    
    def __init__(self, initial_size: int = 100):
        """Initialize adaptive cache.
        
        Args:
            initial_size: Initial cache size
        """
        self.cache = LRUCache(max_size=initial_size)
        self.access_pattern = {}
        self.resize_threshold = 0.9  # Resize when hit rate falls below this
        self.min_size = 50
        self.max_size = 10000
        
    def get(self, key: str) -> Optional[Any]:
        """Get value and track access pattern."""
        result = self.cache.get(key)
        
        # Track access pattern
        if key not in self.access_pattern:
            self.access_pattern[key] = {"hits": 0, "misses": 0}
        
        if result is not None:
            self.access_pattern[key]["hits"] += 1
        else:
            self.access_pattern[key]["misses"] += 1
        
        # Adapt cache size if needed
        self._adapt_cache_size()
        
        return result
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache."""
        self.cache.put(key, value, ttl)
    
    def _adapt_cache_size(self):
        """Adapt cache size based on hit rate."""
        stats = self.cache.get_stats()
        
        if stats["hit_rate"] < self.resize_threshold and self.cache.max_size < self.max_size:
            # Increase cache size
            new_size = min(int(self.cache.max_size * 1.5), self.max_size)
            logger.info(f"Increasing cache size from {self.cache.max_size} to {new_size}")
            self.cache.max_size = new_size
        
        elif stats["hit_rate"] > 0.95 and self.cache.max_size > self.min_size:
            # Decrease cache size if hit rate is very high
            new_size = max(int(self.cache.max_size * 0.8), self.min_size)
            logger.info(f"Decreasing cache size from {self.cache.max_size} to {new_size}")
            self.cache.max_size = new_size


def adaptive_cache(initial_size: int = 100):
    """Decorator for adaptive caching.
    
    Args:
        initial_size: Initial cache size
    """
    cache_instance = AdaptiveCache(initial_size)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{cache_key_generator(*args, **kwargs)}"
            
            result = cache_instance.get(cache_key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result)
            
            return result
        
        wrapper.cache = cache_instance
        return wrapper
    
    return decorator