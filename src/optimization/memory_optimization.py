"""Memory optimization utilities for DiffFE-Physics-Lab."""

import gc
import threading
import weakref
from typing import Any, Dict, List, Optional, TypeVar, Generic
from collections import deque
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ObjectPool(Generic[T]):
    """Object pool for memory-efficient reuse."""
    
    def __init__(self, factory: callable, max_size: int = 100):
        """Initialize object pool.
        
        Args:
            factory: Function to create new objects
            max_size: Maximum pool size
        """
        self.factory = factory
        self.max_size = max_size
        self.pool = deque()
        self.lock = threading.Lock()
        self.created_count = 0
        self.reused_count = 0
    
    def get(self) -> T:
        """Get object from pool or create new one."""
        with self.lock:
            if self.pool:
                obj = self.pool.popleft()
                self.reused_count += 1
                return obj
            else:
                obj = self.factory()
                self.created_count += 1
                return obj
    
    def put(self, obj: T):
        """Return object to pool."""
        with self.lock:
            if len(self.pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self.lock:
            return {
                "pool_size": len(self.pool),
                "max_size": self.max_size,
                "created_count": self.created_count,
                "reused_count": self.reused_count,
                "reuse_rate": self.reused_count / max(self.created_count + self.reused_count, 1)
            }


class MemoryManager:
    """Memory management utilities."""
    
    def __init__(self):
        """Initialize memory manager."""
        self.tracked_objects = weakref.WeakSet()
        self.object_pools = {}
    
    def track_object(self, obj: Any):
        """Track object for memory management."""
        self.tracked_objects.add(obj)
    
    def force_gc(self):
        """Force garbage collection."""
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        return collected
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss // (1024 * 1024),
            "vms_mb": memory_info.vms // (1024 * 1024),
            "tracked_objects": len(self.tracked_objects)
        }
    
    def create_object_pool(self, name: str, factory: callable, max_size: int = 100) -> ObjectPool:
        """Create named object pool."""
        pool = ObjectPool(factory, max_size)
        self.object_pools[name] = pool
        return pool
    
    def get_object_pool(self, name: str) -> Optional[ObjectPool]:
        """Get named object pool."""
        return self.object_pools.get(name)


class LazyLoader:
    """Lazy loading utilities."""
    
    def __init__(self, loader_func: callable):
        """Initialize lazy loader.
        
        Args:
            loader_func: Function to load data when needed
        """
        self.loader_func = loader_func
        self._loaded = False
        self._data = None
        self._lock = threading.Lock()
    
    @property
    def data(self):
        """Get data, loading if necessary."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._data = self.loader_func()
                    self._loaded = True
        return self._data
    
    def reload(self):
        """Force reload of data."""
        with self._lock:
            self._loaded = False
            self._data = None
    
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._loaded


def optimize_memory(func: callable) -> callable:
    """Decorator to optimize memory usage."""
    def wrapper(*args, **kwargs):
        # Force garbage collection before execution
        gc.collect()
        
        result = func(*args, **kwargs)
        
        # Force garbage collection after execution
        gc.collect()
        
        return result
    
    return wrapper


def memory_efficient(pool_name: str = None):
    """Decorator for memory-efficient function execution."""
    def decorator(func: callable) -> callable:
        def wrapper(*args, **kwargs):
            # Use object pool if specified
            if pool_name:
                # This would use a global memory manager
                pass
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global memory manager
global_memory_manager = MemoryManager()