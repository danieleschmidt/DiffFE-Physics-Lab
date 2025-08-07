"""Advanced caching system for sentiment analysis components."""

import time
import hashlib
import pickle
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import json
from collections import OrderedDict
import weakref


@dataclass
class CacheEntry:
    """Cache entry for sentiment analysis results."""
    
    key: str
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float]  # Time to live in seconds
    metadata: Dict[str, Any]
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
            
        if current_time is None:
            current_time = time.time()
            
        return (current_time - self.timestamp) > self.ttl
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Handle numpy arrays in value
        if isinstance(self.value, np.ndarray):
            result['value'] = self.value.tolist()
            result['value_type'] = 'numpy_array'
            result['value_shape'] = self.value.shape
            result['value_dtype'] = str(self.value.dtype)
        else:
            result['value_type'] = 'other'
            
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        if data.get('value_type') == 'numpy_array':
            value = np.array(data['value'], dtype=data['value_dtype']).reshape(data['value_shape'])
        else:
            value = data['value']
            
        return cls(
            key=data['key'],
            value=value,
            timestamp=data['timestamp'],
            access_count=data['access_count'],
            size_bytes=data['size_bytes'],
            ttl=data['ttl'],
            metadata=data['metadata']
        )


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = 3600):
        """Initialize LRU cache.
        
        Parameters
        ----------
        max_size : int, optional
            Maximum number of entries, by default 1000
        default_ttl : float, optional
            Default TTL in seconds, by default 3600 (1 hour)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._total_size_bytes = 0
        self._hits = 0
        self._misses = 0
        
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            elif isinstance(obj, str):
                return len(obj.encode('utf-8'))
            else:
                # Fallback to pickle size
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Conservative estimate
            return 1024
            
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired(current_time):
                expired_keys.append(key)
                
        for key in expired_keys:
            self._remove_entry(key)
            
    def _remove_entry(self, key: str):
        """Remove entry and update stats."""
        if key in self._cache:
            entry = self._cache[key]
            self._total_size_bytes -= entry.size_bytes
            del self._cache[key]
            
    def _evict_lru(self):
        """Evict least recently used entries."""
        while len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            # Clean up expired entries periodically
            if len(self._cache) % 100 == 0:
                self._evict_expired()
                
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired():
                    self._remove_entry(key)
                    self._misses += 1
                    return None
                    
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.access_count += 1
                self._hits += 1
                
                return entry.value
            else:
                self._misses += 1
                return None
                
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Put value into cache."""
        with self._lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
                
            # Estimate size
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
                
            # Evict LRU entries if needed
            self._evict_lru()
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=0,
                size_bytes=size_bytes,
                ttl=ttl,
                metadata=metadata or {}
            )
            
            self._cache[key] = entry
            self._total_size_bytes += size_bytes
            
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
            
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._total_size_bytes = 0
            
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_size_bytes': self._total_size_bytes,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'expired_entries': sum(1 for entry in self._cache.values() if entry.is_expired())
            }


class SentimentAnalysisCache:
    """Specialized cache for sentiment analysis results."""
    
    def __init__(
        self,
        max_size: int = 10000,
        embedding_cache_ttl: float = 3600 * 24,  # 24 hours
        analysis_cache_ttl: float = 3600,  # 1 hour
        model_cache_ttl: float = 3600 * 12,  # 12 hours
        persist_to_disk: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """Initialize sentiment analysis cache.
        
        Parameters
        ----------
        max_size : int, optional
            Maximum number of entries per cache, by default 10000
        embedding_cache_ttl : float, optional
            TTL for text embeddings in seconds, by default 24 hours
        analysis_cache_ttl : float, optional
            TTL for analysis results in seconds, by default 1 hour
        model_cache_ttl : float, optional
            TTL for trained models in seconds, by default 12 hours
        persist_to_disk : bool, optional
            Whether to persist cache to disk, by default True
        cache_dir : Path, optional
            Directory for cache persistence, by default None
        """
        self.embedding_cache = LRUCache(max_size, embedding_cache_ttl)
        self.analysis_cache = LRUCache(max_size, analysis_cache_ttl)
        self.model_cache = LRUCache(max_size // 10, model_cache_ttl)  # Smaller for models
        
        self.persist_to_disk = persist_to_disk
        self.cache_dir = cache_dir or Path.home() / '.sentiment_cache'
        
        if self.persist_to_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
            
        # Cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
    def _generate_text_key(self, texts: List[str], method: str, dim: int) -> str:
        """Generate cache key for text embeddings."""
        # Sort texts for consistent hashing
        text_content = '|'.join(sorted(texts))
        content_hash = hashlib.sha256(text_content.encode()).hexdigest()[:16]
        return f"embed_{method}_{dim}_{content_hash}"
        
    def _generate_analysis_key(
        self, 
        texts: List[str], 
        physics_params: Dict[str, Any],
        method: str
    ) -> str:
        """Generate cache key for analysis results."""
        text_content = '|'.join(sorted(texts))
        text_hash = hashlib.sha256(text_content.encode()).hexdigest()[:16]
        
        # Sort physics params for consistent hashing
        params_str = json.dumps(physics_params, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
        
        return f"analysis_{method}_{text_hash}_{params_hash}"
        
    def _generate_model_key(self, model_config: Dict[str, Any]) -> str:
        """Generate cache key for trained models."""
        config_str = json.dumps(model_config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return f"model_{config_hash}"
        
    def get_embeddings(
        self, 
        texts: List[str], 
        method: str, 
        dim: int
    ) -> Optional[np.ndarray]:
        """Get cached text embeddings."""
        key = self._generate_text_key(texts, method, dim)
        return self.embedding_cache.get(key)
        
    def put_embeddings(
        self, 
        texts: List[str], 
        method: str, 
        dim: int, 
        embeddings: np.ndarray
    ):
        """Cache text embeddings."""
        key = self._generate_text_key(texts, method, dim)
        metadata = {
            'num_texts': len(texts),
            'embedding_method': method,
            'embedding_dim': dim,
            'avg_text_length': np.mean([len(text) for text in texts])
        }
        self.embedding_cache.put(key, embeddings, metadata=metadata)
        
    def get_analysis_result(
        self, 
        texts: List[str], 
        physics_params: Dict[str, Any],
        method: str
    ) -> Optional[Any]:
        """Get cached analysis result."""
        key = self._generate_analysis_key(texts, physics_params, method)
        return self.analysis_cache.get(key)
        
    def put_analysis_result(
        self, 
        texts: List[str], 
        physics_params: Dict[str, Any],
        method: str,
        result: Any
    ):
        """Cache analysis result."""
        key = self._generate_analysis_key(texts, physics_params, method)
        metadata = {
            'num_texts': len(texts),
            'method': method,
            'physics_params': physics_params,
            'result_type': type(result).__name__
        }
        self.analysis_cache.put(key, result, metadata=metadata)
        
    def get_model(self, model_config: Dict[str, Any]) -> Optional[Any]:
        """Get cached trained model."""
        key = self._generate_model_key(model_config)
        return self.model_cache.get(key)
        
    def put_model(self, model_config: Dict[str, Any], model: Any):
        """Cache trained model."""
        key = self._generate_model_key(model_config)
        metadata = {
            'model_type': type(model).__name__,
            'config': model_config
        }
        self.model_cache.put(key, model, metadata=metadata)
        
    def invalidate_embeddings(self, texts: Optional[List[str]] = None):
        """Invalidate embedding cache entries."""
        if texts is None:
            # Clear all embeddings
            keys_to_delete = [k for k in self.embedding_cache._cache.keys() if k.startswith('embed_')]
            for key in keys_to_delete:
                self.embedding_cache.delete(key)
        else:
            # Invalidate specific texts (all methods/dimensions)
            text_hash = hashlib.sha256('|'.join(sorted(texts)).encode()).hexdigest()[:16]
            keys_to_delete = [
                k for k in self.embedding_cache._cache.keys() 
                if k.startswith('embed_') and text_hash in k
            ]
            for key in keys_to_delete:
                self.embedding_cache.delete(key)
                
    def warm_up_cache(
        self, 
        common_texts: List[List[str]],
        embedding_methods: List[str] = None,
        embedding_dims: List[int] = None
    ):
        """Pre-warm cache with common text combinations."""
        if embedding_methods is None:
            embedding_methods = ['tfidf']
        if embedding_dims is None:
            embedding_dims = [300]
            
        # This would need to be integrated with actual embedding generation
        # For now, just mark cache locations as important
        for texts in common_texts:
            for method in embedding_methods:
                for dim in embedding_dims:
                    key = self._generate_text_key(texts, method, dim)
                    # Could add placeholder or priority metadata
                    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'embedding_cache': self.embedding_cache.stats(),
            'analysis_cache': self.analysis_cache.stats(), 
            'model_cache': self.model_cache.stats(),
            'total_memory_mb': (
                self.embedding_cache._total_size_bytes +
                self.analysis_cache._total_size_bytes +
                self.model_cache._total_size_bytes
            ) / (1024 * 1024),
            'cache_dir': str(self.cache_dir) if self.persist_to_disk else None
        }
        
    def _periodic_cleanup(self):
        """Periodic cleanup of expired entries."""
        while True:
            time.sleep(300)  # Every 5 minutes
            try:
                with self.embedding_cache._lock:
                    self.embedding_cache._evict_expired()
                with self.analysis_cache._lock:
                    self.analysis_cache._evict_expired()
                with self.model_cache._lock:
                    self.model_cache._evict_expired()
                    
                # Persist to disk if enabled
                if self.persist_to_disk:
                    self._save_to_disk()
                    
            except Exception:
                # Ignore cleanup errors to avoid breaking main thread
                pass
                
    def _save_to_disk(self):
        """Save cache state to disk."""
        try:
            # Save embedding cache metadata only (embeddings can be regenerated)
            embedding_metadata = {}
            with self.embedding_cache._lock:
                for key, entry in self.embedding_cache._cache.items():
                    if not entry.is_expired():
                        embedding_metadata[key] = {
                            'timestamp': entry.timestamp,
                            'access_count': entry.access_count,
                            'metadata': entry.metadata,
                            'ttl': entry.ttl
                        }
                        
            metadata_file = self.cache_dir / 'embedding_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(embedding_metadata, f, indent=2)
                
        except Exception:
            # Don't fail if persistence fails
            pass
            
    def _load_from_disk(self):
        """Load cache state from disk."""
        try:
            metadata_file = self.cache_dir / 'embedding_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    embedding_metadata = json.load(f)
                    
                # Restore metadata for cache warming decisions
                self._disk_metadata = embedding_metadata
                
        except Exception:
            # Don't fail if loading fails
            self._disk_metadata = {}


class AdaptiveCacheManager:
    """Adaptive cache manager that adjusts cache sizes based on usage patterns."""
    
    def __init__(self, cache: SentimentAnalysisCache, adaptation_interval: float = 3600):
        """Initialize adaptive cache manager.
        
        Parameters
        ----------
        cache : SentimentAnalysisCache
            Cache instance to manage
        adaptation_interval : float, optional
            Interval between adaptations in seconds, by default 3600
        """
        self.cache = cache
        self.adaptation_interval = adaptation_interval
        self._last_adaptation = 0
        self._usage_history = []
        
    def should_adapt(self) -> bool:
        """Check if cache should be adapted."""
        return time.time() - self._last_adaptation > self.adaptation_interval
        
    def adapt_cache_sizes(self):
        """Adapt cache sizes based on usage patterns."""
        if not self.should_adapt():
            return
            
        stats = self.cache.get_cache_stats()
        
        # Get hit rates
        embed_hit_rate = stats['embedding_cache']['hit_rate']
        analysis_hit_rate = stats['analysis_cache']['hit_rate']
        
        # Adapt sizes based on hit rates
        if embed_hit_rate > 0.8 and analysis_hit_rate < 0.5:
            # Embeddings are cached well, analysis not so much
            # Consider increasing analysis cache size
            pass
        elif analysis_hit_rate > 0.8 and embed_hit_rate < 0.5:
            # Analysis cached well, embeddings not so much
            # Consider increasing embedding cache size
            pass
            
        # Record adaptation
        self._last_adaptation = time.time()
        self._usage_history.append({
            'timestamp': self._last_adaptation,
            'stats': stats
        })
        
        # Keep only recent history
        cutoff_time = self._last_adaptation - (self.adaptation_interval * 24)  # 24 intervals
        self._usage_history = [
            entry for entry in self._usage_history 
            if entry['timestamp'] > cutoff_time
        ]


# Global cache instance
_global_sentiment_cache = None


def get_global_cache() -> SentimentAnalysisCache:
    """Get global sentiment analysis cache instance."""
    global _global_sentiment_cache
    
    if _global_sentiment_cache is None:
        _global_sentiment_cache = SentimentAnalysisCache()
        
    return _global_sentiment_cache


def clear_global_cache():
    """Clear global cache."""
    global _global_sentiment_cache
    
    if _global_sentiment_cache is not None:
        _global_sentiment_cache.embedding_cache.clear()
        _global_sentiment_cache.analysis_cache.clear()
        _global_sentiment_cache.model_cache.clear()


def cache_warmup_from_file(filepath: Union[str, Path], cache: Optional[SentimentAnalysisCache] = None):
    """Warm up cache from a file of common texts.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Path to file containing texts (one per line)
    cache : SentimentAnalysisCache, optional
        Cache instance, uses global cache if None
    """
    if cache is None:
        cache = get_global_cache()
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
            
        # Group texts into batches for warming
        batch_size = 10
        text_batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        cache.warm_up_cache(text_batches)
        
    except Exception as e:
        print(f"Cache warmup failed: {e}")


# Context manager for cache transactions
class CacheTransaction:
    """Context manager for cache operations with rollback capability."""
    
    def __init__(self, cache: SentimentAnalysisCache):
        self.cache = cache
        self.backup_keys = set()
        self.backup_values = {}
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Rollback on exception
            self.rollback()
            
    def backup_key(self, cache_type: str, key: str):
        """Backup a cache key before modification."""
        cache_obj = getattr(self.cache, f"{cache_type}_cache")
        value = cache_obj.get(key)
        if value is not None:
            self.backup_keys.add((cache_type, key))
            self.backup_values[(cache_type, key)] = value
            
    def rollback(self):
        """Rollback cache changes."""
        for cache_type, key in self.backup_keys:
            cache_obj = getattr(self.cache, f"{cache_type}_cache")
            if (cache_type, key) in self.backup_values:
                cache_obj.put(key, self.backup_values[(cache_type, key)])
            else:
                cache_obj.delete(key)