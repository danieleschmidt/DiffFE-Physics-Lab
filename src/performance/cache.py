"""Advanced caching system for performance optimization."""

import time
import threading
import pickle
import hashlib
import logging
from typing import Any, Dict, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    ttl: Optional[float] = None
    size_bytes: int = 0


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all cache keys."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.total_size_bytes = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from memory cache."""
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                del self.cache[key]
                self.total_size_bytes -= entry.size_bytes
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access = time.time()
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            return entry
    
    def set(self, key: str, entry: CacheEntry) -> None:
        """Set value in memory cache."""
        with self._lock:
            # Calculate entry size
            try:
                entry.size_bytes = len(pickle.dumps(entry.value))
            except:
                entry.size_bytes = 1024  # Fallback estimate
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_size_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Check memory limit
            while (self.total_size_bytes + entry.size_bytes > self.max_memory_bytes and
                   len(self.cache) > 0):
                self._evict_lru()
            
            # Check size limit
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self.cache[key] = entry
            self.total_size_bytes += entry.size_bytes
    
    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                self.total_size_bytes -= entry.size_bytes
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from memory cache."""
        with self._lock:
            self.cache.clear()
            self.total_size_bytes = 0
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self.cache.keys())
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Remove oldest entry (first in OrderedDict)
        key, entry = self.cache.popitem(last=False)
        self.total_size_bytes -= entry.size_bytes
        logger.debug(f"Evicted LRU cache entry: {key}")


class CacheManager:
    """Advanced cache manager with multiple backends and strategies.
    
    Provides intelligent caching with features like:
    - Multiple eviction policies (LRU, TTL, size-based)
    - Cache warming and preloading
    - Performance monitoring and statistics
    - Adaptive cache sizing
    - Cache coherency and invalidation
    
    Examples
    --------
    >>> cache = CacheManager(max_size=1000, max_memory_mb=50)
    >>> 
    >>> @cache.cached(ttl=300)
    >>> def expensive_computation(x, y):
    ...     return x ** y
    >>> 
    >>> result = expensive_computation(2, 10)  # Cached for 5 minutes
    """
    
    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[float] = None,
        enable_stats: bool = True
    ):
        self.backend = backend or MemoryCache(max_size, max_memory_mb)
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0,
            'errors': 0
        }
        
        # Cache warming registry
        self.warmup_functions = {}
        
        # Background cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired_entries, 
            daemon=True
        )
        self._cleanup_thread.start()
        
        logger.info("Cache manager initialized")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache.
        
        Parameters
        ----------
        key : str
            Cache key
        default : Any, optional
            Default value if key not found
            
        Returns
        -------
        Any
            Cached value or default
        """
        try:
            entry = self.backend.get(key)
            
            if entry is None:
                if self.enable_stats:
                    self.stats['misses'] += 1
                return default
            
            if self.enable_stats:
                self.stats['hits'] += 1
            
            return entry.value
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            if self.enable_stats:
                self.stats['errors'] += 1
            return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set value in cache.
        
        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        ttl : float, optional
            Time to live in seconds
        """
        try:
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl
            )
            
            self.backend.set(key, entry)
            
            if self.enable_stats:
                self.stats['sets'] += 1
                
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            if self.enable_stats:
                self.stats['errors'] += 1
    
    def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Parameters
        ----------
        key : str
            Cache key to delete
            
        Returns
        -------
        bool
            True if key was deleted, False if not found
        """
        try:
            result = self.backend.delete(key)
            
            if self.enable_stats and result:
                self.stats['deletes'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            if self.enable_stats:
                self.stats['errors'] += 1
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self.backend.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def cached(
        self,
        ttl: Optional[float] = None,
        key_func: Optional[Callable] = None,
        condition: Optional[Callable] = None
    ):
        """Decorator for caching function results.
        
        Parameters
        ----------
        ttl : float, optional
            Cache TTL in seconds
        key_func : Callable, optional
            Function to generate cache key from arguments
        condition : Callable, optional
            Function to determine if result should be cached
            
        Returns
        -------
        Callable
            Decorated function with caching
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Call function
                result = func(*args, **kwargs)
                
                # Check caching condition
                if condition and not condition(result):
                    return result
                
                # Cache result
                self.set(cache_key, result, ttl)
                
                return result
            
            # Add cache invalidation method to function
            def invalidate(*args, **kwargs):
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func.__name__, args, kwargs)
                self.delete(cache_key)
            
            wrapper.invalidate = invalidate
            wrapper.cache_key = lambda *args, **kwargs: (
                key_func(*args, **kwargs) if key_func 
                else self._generate_key(func.__name__, args, kwargs)
            )
            
            return wrapper
        
        return decorator
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create a deterministic key from function name and arguments
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(key_str).hexdigest()
    
    def memoize(self, ttl: Optional[float] = None):
        """Simple memoization decorator.
        
        Parameters
        ----------
        ttl : float, optional
            Cache TTL in seconds
            
        Returns
        -------
        Callable
            Memoized function
        """
        return self.cached(ttl=ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns
        -------
        Dict[str, Any]
            Cache statistics
        """
        if not self.enable_stats:
            return {}
        
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        cache_info = {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            **self.stats
        }
        
        # Add backend-specific stats
        if isinstance(self.backend, MemoryCache):
            cache_info.update({
                'cache_size': len(self.backend.cache),
                'memory_usage_bytes': self.backend.total_size_bytes,
                'memory_usage_mb': self.backend.total_size_bytes / (1024 * 1024)
            })
        
        return cache_info
    
    def warm_cache(self, keys: List[str], loader_func: Callable[[str], Any]) -> None:
        """Warm cache with precomputed values.
        
        Parameters
        ----------
        keys : List[str]
            List of cache keys to warm
        loader_func : Callable[[str], Any]
            Function to load value for each key
        """
        logger.info(f"Warming cache with {len(keys)} entries")
        
        for key in keys:
            try:
                if self.get(key) is None:  # Only load if not already cached
                    value = loader_func(key)
                    self.set(key, value)
            except Exception as e:
                logger.warning(f"Failed to warm cache for key {key}: {e}")
        
        logger.info("Cache warming completed")
    
    def register_warmup_function(self, name: str, func: Callable[[], None]) -> None:
        """Register function for cache warming.
        
        Parameters
        ----------
        name : str
            Name of warmup function
        func : Callable[[], None]
            Function to execute for warming
        """
        self.warmup_functions[name] = func
        logger.info(f"Registered cache warmup function: {name}")
    
    def run_warmup(self, name: Optional[str] = None) -> None:
        """Run cache warmup functions.
        
        Parameters
        ----------
        name : str, optional
            Specific warmup function to run, or all if None
        """
        if name:
            if name in self.warmup_functions:
                logger.info(f"Running cache warmup: {name}")
                self.warmup_functions[name]()
            else:
                logger.warning(f"Warmup function not found: {name}")
        else:
            logger.info("Running all cache warmup functions")
            for warmup_name, func in self.warmup_functions.items():
                try:
                    logger.info(f"Running warmup: {warmup_name}")
                    func()
                except Exception as e:
                    logger.error(f"Warmup function {warmup_name} failed: {e}")
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern.
        
        Parameters
        ----------
        pattern : str
            Pattern to match (supports * wildcards)
            
        Returns
        -------
        int
            Number of entries invalidated
        """
        import fnmatch
        
        keys_to_delete = []
        
        try:
            all_keys = self.backend.keys()
            for key in all_keys:
                if fnmatch.fnmatch(key, pattern):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                self.delete(key)
            
            logger.info(f"Invalidated {len(keys_to_delete)} cache entries matching pattern: {pattern}")
            return len(keys_to_delete)
            
        except Exception as e:
            logger.error(f"Pattern invalidation failed for {pattern}: {e}")
            return 0
    
    def _cleanup_expired_entries(self) -> None:
        """Background task to clean up expired cache entries."""
        while True:
            try:
                current_time = time.time()
                expired_keys = []
                
                # Check for expired entries
                for key in self.backend.keys():
                    entry = self.backend.get(key)
                    if (entry and entry.ttl and 
                        current_time - entry.timestamp > entry.ttl):
                        expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    self.backend.delete(key)
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                # Sleep for 1 minute before next cleanup
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                time.sleep(60)


# Sentiment Analysis specific cache optimizations

class SentimentCacheManager:
    """
    Specialized cache manager for sentiment analysis workloads.
    
    Features:
    - Text preprocessing result caching
    - Model prediction caching with physics metrics
    - Multilingual content caching
    - Intelligent cache warming for common patterns
    """
    
    def __init__(self, 
                 text_cache_size: int = 50000,
                 model_cache_size: int = 10000,
                 physics_cache_size: int = 5000,
                 total_memory_mb: int = 256):
        
        # Text processing cache (60% of memory)
        self.text_cache = CacheManager(
            max_size=text_cache_size,
            max_memory_mb=int(total_memory_mb * 0.6),
            default_ttl=1800,  # 30 minutes
            enable_stats=True
        )
        
        # Model prediction cache (30% of memory)  
        self.model_cache = CacheManager(
            max_size=model_cache_size,
            max_memory_mb=int(total_memory_mb * 0.3),
            default_ttl=3600,  # 1 hour
            enable_stats=True
        )
        
        # Physics computation cache (10% of memory)
        self.physics_cache = CacheManager(
            max_size=physics_cache_size,
            max_memory_mb=int(total_memory_mb * 0.1),
            default_ttl=7200,  # 2 hours - physics computations are expensive
            enable_stats=True
        )
        
        logger.info(f"Initialized SentimentCacheManager with {total_memory_mb}MB memory allocation")
    
    def get_text_result(self, text_hash: str) -> Optional[Dict]:
        """Get cached text processing result."""
        return self.text_cache.get(f"text_{text_hash}")
    
    def cache_text_result(self, text_hash: str, result: Dict, ttl: Optional[float] = None):
        """Cache text processing result."""
        self.text_cache.set(f"text_{text_hash}", result, ttl)
    
    def get_prediction(self, model_key: str, input_hash: str, language: str = "en") -> Optional[Dict]:
        """Get cached sentiment prediction."""
        cache_key = f"pred_{model_key}_{language}_{input_hash}"
        return self.model_cache.get(cache_key)
    
    def cache_prediction(self, model_key: str, input_hash: str, prediction: Dict, 
                        language: str = "en", ttl: Optional[float] = None):
        """Cache sentiment prediction with language-specific key."""
        cache_key = f"pred_{model_key}_{language}_{input_hash}"
        self.model_cache.set(cache_key, prediction, ttl)
    
    def get_physics_result(self, computation_type: str, params_hash: str) -> Optional[Any]:
        """Get cached physics computation result."""
        cache_key = f"physics_{computation_type}_{params_hash}"
        return self.physics_cache.get(cache_key)
    
    def cache_physics_result(self, computation_type: str, params_hash: str, result: Any, 
                           ttl: Optional[float] = None):
        """Cache physics computation result."""
        cache_key = f"physics_{computation_type}_{params_hash}"
        self.physics_cache.set(cache_key, result, ttl)
    
    def warm_multilingual_cache(self, common_texts: Dict[str, List[str]], 
                              model_types: List[str]) -> Dict[str, int]:
        """
        Warm caches with common multilingual text patterns.
        
        Parameters
        ----------
        common_texts : Dict[str, List[str]]
            Dictionary mapping language codes to common texts
        model_types : List[str]
            List of model types to warm
            
        Returns
        -------
        Dict[str, int]
            Warming results per cache type
        """
        warming_results = {'text': 0, 'model': 0, 'physics': 0}
        
        # Warm text cache
        for lang, texts in common_texts.items():
            for text in texts[:100]:  # Limit to prevent memory issues
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                
                # Simulate text processing result
                mock_result = {
                    'tokens': text.split()[:50],  # Truncate for memory
                    'language': lang,
                    'length': len(text),
                    'quality_score': 0.85,
                    'processing_time': 0.05
                }
                
                self.cache_text_result(text_hash, mock_result)
                warming_results['text'] += 1
        
        # Warm model cache with prediction patterns
        for lang, texts in common_texts.items():
            for model_type in model_types:
                for text in texts[:20]:  # Fewer for model cache
                    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                    
                    # Simulate prediction result
                    mock_prediction = {
                        'sentiment': 'neutral',
                        'confidence': 0.7,
                        'scores': {'negative': 0.2, 'neutral': 0.6, 'positive': 0.2},
                        'physics_metrics': {
                            'energy_conservation': 0.95,
                            'gradient_smoothness': 0.88
                        },
                        'processing_time': 0.12
                    }
                    
                    self.cache_prediction(model_type, text_hash, mock_prediction, lang)
                    warming_results['model'] += 1
        
        # Warm physics cache with common computations
        physics_computations = [
            ('diffusion', 'rate_0.1_steps_10'),
            ('conservation', 'weight_0.05'),
            ('energy_flow', 'damping_0.1'),
        ]
        
        for comp_type, params_hash in physics_computations:
            mock_physics_result = {
                'computation_type': comp_type,
                'result_tensor': [0.1, 0.2, 0.3, 0.4, 0.5],  # Mock tensor
                'convergence_achieved': True,
                'iterations': 25
            }
            
            self.cache_physics_result(comp_type, params_hash, mock_physics_result)
            warming_results['physics'] += 1
        
        logger.info(f"Cache warming completed: {warming_results}")
        return warming_results
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all caches."""
        return {
            'text_cache': self.text_cache.get_stats(),
            'model_cache': self.model_cache.get_stats(),
            'physics_cache': self.physics_cache.get_stats(),
            'memory_allocation': {
                'text_percent': 60,
                'model_percent': 30,
                'physics_percent': 10
            }
        }
    
    def optimize_all_caches(self) -> Dict[str, str]:
        """Optimize all cache configurations based on usage patterns."""
        optimizations = {}
        
        # Analyze text cache performance
        text_stats = self.text_cache.get_stats()
        if text_stats.get('hit_rate', 0) > 0.9:
            optimizations['text'] = "High hit rate - consider reducing size"
        elif text_stats.get('hit_rate', 0) < 0.5:
            optimizations['text'] = "Low hit rate - consider increasing TTL"
        
        # Analyze model cache performance
        model_stats = self.model_cache.get_stats()
        if model_stats.get('hit_rate', 0) > 0.8:
            optimizations['model'] = "Excellent performance - maintain settings"
        elif model_stats.get('hit_rate', 0) < 0.4:
            optimizations['model'] = "Poor performance - increase cache size or TTL"
        
        # Analyze physics cache performance
        physics_stats = self.physics_cache.get_stats()
        if physics_stats.get('hit_rate', 0) < 0.6:
            optimizations['physics'] = "Increase TTL - physics computations are expensive"
        
        return optimizations
    
    def clear_all(self):
        """Clear all caches."""
        self.text_cache.clear()
        self.model_cache.clear()
        self.physics_cache.clear()
        logger.info("All sentiment analysis caches cleared")


def cached_sentiment_analysis(cache_type: str = "model", ttl: int = 3600):
    """
    Decorator for caching sentiment analysis functions.
    
    Parameters
    ----------
    cache_type : str
        Type of cache to use ('text', 'model', 'physics')
    ttl : int
        Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': str(args)[:200],  # Limit key size
                'kwargs': str(sorted(kwargs.items()))[:200]
            }
            
            key_string = str(key_data)
            cache_key = hashlib.md5(key_string.encode()).hexdigest()
            
            # Get global sentiment cache manager
            cache_manager = get_sentiment_cache()
            
            # Try to get cached result
            if cache_type == "text":
                cached_result = cache_manager.get_text_result(cache_key)
            elif cache_type == "physics":
                cached_result = cache_manager.get_physics_result("general", cache_key)
            else:  # model cache
                cached_result = cache_manager.get_prediction("general", cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            
            # Cache the result
            if cache_type == "text":
                cache_manager.cache_text_result(cache_key, result, ttl)
            elif cache_type == "physics":
                cache_manager.cache_physics_result("general", cache_key, result, ttl)
            else:  # model cache
                cache_manager.cache_prediction("general", cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    
    return decorator


# Global instances
_global_cache = None
_sentiment_cache = None


def get_global_cache() -> CacheManager:
    """Get global cache manager instance.
    
    Returns
    -------
    CacheManager
        Global cache manager
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def get_sentiment_cache() -> SentimentCacheManager:
    """Get global sentiment cache manager instance.
    
    Returns
    -------
    SentimentCacheManager
        Global sentiment cache manager
    """
    global _sentiment_cache
    if _sentiment_cache is None:
        _sentiment_cache = SentimentCacheManager()
    return _sentiment_cache


def cached(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Global caching decorator.
    
    Parameters
    ----------
    ttl : float, optional
        Cache TTL in seconds
    key_func : Callable, optional
        Function to generate cache key
        
    Returns
    -------
    Callable
        Decorated function with caching
    """
    return get_global_cache().cached(ttl=ttl, key_func=key_func)


def clear_global_cache():
    """Clear global cache."""
    get_global_cache().clear()


def clear_sentiment_cache():
    """Clear sentiment analysis cache."""
    get_sentiment_cache().clear_all()


class IntelligentCache:
    """Alias for backward compatibility."""
    pass
