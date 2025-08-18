"""Utility functions and decorators for FEM solver scaling integration.

This module provides high-level utilities for integrating Generation 3 scaling
features with existing FEM solvers, including decorators, context managers,
and optimization helpers.
"""

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .advanced_optimization import get_jax_engine, get_mesh_refinement
from .advanced_cache import get_adaptive_cache  
from .parallel_processing import get_parallel_engine, get_resource_monitor
from .advanced_scaling import get_memory_optimizer, get_scaling_manager

logger = logging.getLogger(__name__)


def auto_scale_solver(problem_size_threshold: int = 10000, 
                     enable_features: Dict[str, bool] = None):
    """Decorator that automatically enables scaling features based on problem size.
    
    Parameters
    ----------
    problem_size_threshold : int, optional
        Threshold for enabling advanced features, by default 10000
    enable_features : Dict[str, bool], optional
        Override which features to enable, by default None
    """
    if enable_features is None:
        enable_features = {
            "jit_compilation": True,
            "caching": True,
            "parallel_processing": True,
            "adaptive_mesh": True,
            "memory_pooling": True
        }
    
    def decorator(solve_func: Callable) -> Callable:
        @functools.wraps(solve_func)
        async def wrapper(*args, **kwargs):
            # Estimate problem size from arguments
            problem_size = _estimate_problem_size(*args, **kwargs)
            
            # Enable scaling features if problem is large enough
            scaling_context = {}
            if problem_size >= problem_size_threshold:
                scaling_context.update(enable_features)
                logger.info(f"Auto-scaling enabled for problem size {problem_size}")
            else:
                # For small problems, only enable lightweight features
                scaling_context.update({
                    "jit_compilation": enable_features.get("jit_compilation", True),
                    "caching": enable_features.get("caching", True),
                    "parallel_processing": False,
                    "adaptive_mesh": False,
                    "memory_pooling": enable_features.get("memory_pooling", True)
                })
            
            # Execute with scaling context
            if hasattr(args[0], 'scaling_context'):  # EnhancedFEMSolver
                async with args[0].scaling_context(**scaling_context):
                    return await solve_func(*args, **kwargs)
            else:
                # Standard solver - just call normally
                return await solve_func(*args, **kwargs)
                
        return wrapper
    return decorator


def performance_monitor(log_level: str = "INFO", 
                       track_resources: bool = True,
                       alert_thresholds: Dict[str, float] = None):
    """Decorator for comprehensive performance monitoring.
    
    Parameters
    ----------
    log_level : str, optional
        Logging level for performance data, by default "INFO"
    track_resources : bool, optional
        Whether to track system resource usage, by default True  
    alert_thresholds : Dict[str, float], optional
        Resource usage thresholds for alerts, by default None
    """
    if alert_thresholds is None:
        alert_thresholds = {
            "cpu_usage": 90.0,
            "memory_usage": 85.0,
            "solve_time": 300.0  # 5 minutes
        }
    
    def decorator(solve_func: Callable) -> Callable:
        @functools.wraps(solve_func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Start resource monitoring if requested
            resource_monitor = None
            if track_resources:
                resource_monitor = get_resource_monitor()
                if not resource_monitor.monitoring_active:
                    resource_monitor.start_monitoring()
            
            try:
                # Execute solve function
                result = await solve_func(*args, **kwargs)
                
                # Collect performance data
                solve_time = time.time() - start_time
                performance_data = {
                    "function": solve_func.__name__,
                    "solve_time": solve_time,
                    "success": True
                }
                
                # Add resource data if available
                if resource_monitor:
                    current_metrics = resource_monitor.get_current_metrics()
                    if current_metrics:
                        performance_data.update({
                            "cpu_usage": current_metrics.cpu_usage,
                            "memory_usage": current_metrics.memory_usage,
                            "gpu_usage": current_metrics.gpu_usage
                        })
                        
                        # Check for alerts
                        for metric, threshold in alert_thresholds.items():
                            value = performance_data.get(metric, 0)
                            if value > threshold:
                                logger.warning(
                                    f"Performance alert: {metric}={value:.1f} exceeds threshold {threshold:.1f}"
                                )
                
                # Log performance data
                log_func = getattr(logger, log_level.lower())
                log_func(f"Performance: {performance_data}")
                
                return result
                
            except Exception as e:
                solve_time = time.time() - start_time
                logger.error(f"Solve failed after {solve_time:.3f}s: {e}")
                raise
                
        return wrapper
    return decorator


def adaptive_caching(cache_key_func: Callable = None,
                    ttl: Optional[float] = None,
                    invalidation_patterns: List[str] = None):
    """Decorator for adaptive caching of solver results.
    
    Parameters
    ----------
    cache_key_func : Callable, optional
        Function to generate cache keys, by default None
    ttl : Optional[float], optional
        Time-to-live for cache entries, by default None
    invalidation_patterns : List[str], optional
        Patterns for cache invalidation, by default None
    """
    def decorator(solve_func: Callable) -> Callable:
        @functools.wraps(solve_func)
        async def wrapper(*args, **kwargs):
            cache_manager = get_adaptive_cache()
            
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = _generate_default_cache_key(solve_func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {solve_func.__name__}: {cache_key}")
                return cached_result
            
            # Execute function
            result = await solve_func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_key, result, ttl=ttl)
            logger.debug(f"Cache stored for {solve_func.__name__}: {cache_key}")
            
            return result
            
        return wrapper
    return decorator


def memory_optimized(pool_arrays: bool = True,
                    preallocate_size: Optional[int] = None):
    """Decorator for memory optimization using memory pools.
    
    Parameters
    ----------
    pool_arrays : bool, optional
        Whether to use memory pooling for arrays, by default True
    preallocate_size : Optional[int], optional
        Size to preallocate in memory pool, by default None
    """
    def decorator(solve_func: Callable) -> Callable:
        @functools.wraps(solve_func)
        async def wrapper(*args, **kwargs):
            memory_optimizer = get_memory_optimizer()
            
            # Preallocate memory if requested
            preallocated_arrays = {}
            if preallocate_size:
                common_sizes = [preallocate_size // 4, preallocate_size // 2, preallocate_size]
                for size in common_sizes:
                    array = memory_optimizer.get_memory_pool(size)
                    preallocated_arrays[size] = array
            
            try:
                # Make memory optimizer available to function
                kwargs['_memory_optimizer'] = memory_optimizer
                
                # Execute function
                result = await solve_func(*args, **kwargs)
                
                return result
                
            finally:
                # Return arrays to pool
                for array in preallocated_arrays.values():
                    memory_optimizer.return_to_pool(array)
                    
        return wrapper
    return decorator


@asynccontextmanager
async def scaling_session(solver_type: str = "enhanced",
                         scaling_level: str = "standard",
                         session_options: Dict[str, Any] = None):
    """Context manager for a complete scaling session.
    
    Parameters
    ----------
    solver_type : str, optional
        Type of solver to use, by default "enhanced"
    scaling_level : str, optional
        Scaling level configuration, by default "standard"
    session_options : Dict[str, Any], optional
        Additional session options, by default None
    """
    session_start = time.time()
    session_id = f"scaling_session_{int(session_start)}"
    
    logger.info(f"Starting scaling session {session_id}")
    
    # Initialize components based on scaling level
    components = _initialize_scaling_components(scaling_level, session_options or {})
    
    try:
        yield components
        
    finally:
        # Cleanup and collect session metrics
        session_duration = time.time() - session_start
        
        # Gather performance metrics from all components
        session_metrics = {
            "session_id": session_id,
            "duration": session_duration,
            "scaling_level": scaling_level,
            "components_used": list(components.keys())
        }
        
        # Component-specific metrics
        if "cache_manager" in components:
            session_metrics["cache_stats"] = components["cache_manager"].get_comprehensive_stats()
        
        if "resource_monitor" in components:
            session_metrics["resource_stats"] = {
                "final_metrics": components["resource_monitor"].get_current_metrics()
            }
        
        # Cleanup components
        await _cleanup_scaling_components(components)
        
        logger.info(f"Scaling session {session_id} completed in {session_duration:.3f}s")
        logger.debug(f"Session metrics: {session_metrics}")


class ScalingBenchmark:
    """Benchmark utility for measuring scaling performance."""
    
    def __init__(self, problem_sizes: List[int] = None, 
                 scaling_levels: List[str] = None):
        """Initialize scaling benchmark.
        
        Parameters
        ----------
        problem_sizes : List[int], optional
            Problem sizes to benchmark, by default None
        scaling_levels : List[str], optional  
            Scaling levels to test, by default None
        """
        self.problem_sizes = problem_sizes or [100, 1000, 10000, 100000]
        self.scaling_levels = scaling_levels or ["minimal", "standard", "aggressive"]
        self.benchmark_results = []
        
    async def run_benchmark(self, solver_func: Callable, 
                          problem_generator: Callable,
                          iterations: int = 3) -> Dict[str, Any]:
        """Run comprehensive scaling benchmark.
        
        Parameters
        ----------
        solver_func : Callable
            Solver function to benchmark
        problem_generator : Callable
            Function that generates problems of given size
        iterations : int, optional
            Number of iterations per configuration, by default 3
            
        Returns
        -------
        Dict[str, Any]
            Benchmark results
        """
        logger.info(f"Starting scaling benchmark with {len(self.problem_sizes)} sizes and {len(self.scaling_levels)} levels")
        
        benchmark_start = time.time()
        
        for problem_size in self.problem_sizes:
            for scaling_level in self.scaling_levels:
                logger.info(f"Benchmarking problem_size={problem_size}, scaling_level={scaling_level}")
                
                level_times = []
                level_memory = []
                
                for iteration in range(iterations):
                    # Generate problem
                    problem_params = problem_generator(problem_size)
                    
                    # Run with scaling session
                    async with scaling_session(scaling_level=scaling_level) as components:
                        start_time = time.time()
                        
                        # Execute solver
                        try:
                            result = await solver_func(**problem_params)
                            solve_time = time.time() - start_time
                            level_times.append(solve_time)
                            
                            # Get memory usage if available
                            if "resource_monitor" in components:
                                metrics = components["resource_monitor"].get_current_metrics()
                                if metrics:
                                    level_memory.append(metrics.memory_usage)
                            
                            logger.debug(f"  Iteration {iteration+1}: {solve_time:.3f}s")
                            
                        except Exception as e:
                            logger.error(f"Benchmark failed: {e}")
                            level_times.append(float('inf'))
                
                # Record results
                result_record = {
                    "problem_size": problem_size,
                    "scaling_level": scaling_level,
                    "avg_time": np.mean(level_times),
                    "std_time": np.std(level_times),
                    "min_time": np.min(level_times),
                    "max_time": np.max(level_times),
                    "avg_memory": np.mean(level_memory) if level_memory else 0,
                    "iterations": iterations,
                    "success_rate": sum(1 for t in level_times if t != float('inf')) / iterations
                }
                
                self.benchmark_results.append(result_record)
                
                logger.info(f"  Results: avg_time={result_record['avg_time']:.3f}s, "
                           f"success_rate={result_record['success_rate']:.2f}")
        
        total_time = time.time() - benchmark_start
        
        # Generate summary
        summary = {
            "total_benchmark_time": total_time,
            "configurations_tested": len(self.benchmark_results),
            "problem_sizes": self.problem_sizes,
            "scaling_levels": self.scaling_levels,
            "detailed_results": self.benchmark_results,
            "performance_summary": self._generate_performance_summary()
        }
        
        logger.info(f"Scaling benchmark completed in {total_time:.3f}s")
        return summary
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary from benchmark results."""
        if not self.benchmark_results:
            return {}
        
        # Find best configurations
        successful_results = [r for r in self.benchmark_results if r['success_rate'] > 0.8]
        
        if not successful_results:
            return {"status": "No successful configurations"}
        
        # Best overall performance
        best_overall = min(successful_results, key=lambda x: x['avg_time'])
        
        # Best by problem size
        best_by_size = {}
        for size in self.problem_sizes:
            size_results = [r for r in successful_results if r['problem_size'] == size]
            if size_results:
                best_by_size[size] = min(size_results, key=lambda x: x['avg_time'])
        
        # Scaling efficiency
        scaling_efficiency = {}
        for level in self.scaling_levels:
            level_results = [r for r in successful_results if r['scaling_level'] == level]
            if level_results:
                avg_time = np.mean([r['avg_time'] for r in level_results])
                avg_success = np.mean([r['success_rate'] for r in level_results])
                scaling_efficiency[level] = {
                    "avg_time": avg_time,
                    "avg_success_rate": avg_success,
                    "efficiency_score": avg_success / max(avg_time, 0.001)  # Avoid division by zero
                }
        
        return {
            "best_overall_config": {
                "problem_size": best_overall['problem_size'],
                "scaling_level": best_overall['scaling_level'],
                "avg_time": best_overall['avg_time']
            },
            "best_by_problem_size": best_by_size,
            "scaling_level_efficiency": scaling_efficiency,
            "total_configurations": len(successful_results)
        }


# Utility functions

def _estimate_problem_size(*args, **kwargs) -> int:
    """Estimate problem size from function arguments."""
    # Look for common size indicators
    size_indicators = ['num_elements', 'nx', 'ny', 'nz', 'n_nodes', 'n_dofs']
    
    total_size = 0
    
    # Check keyword arguments
    for indicator in size_indicators:
        if indicator in kwargs:
            value = kwargs[indicator]
            if isinstance(value, int):
                total_size += value
            elif isinstance(value, (list, tuple)):
                total_size += np.prod(value)
    
    # Check positional arguments for numeric values
    for arg in args:
        if isinstance(arg, int) and arg > 1:
            total_size += arg
    
    return max(total_size, 10)  # Minimum size


def _generate_default_cache_key(func_name: str, *args, **kwargs) -> str:
    """Generate a default cache key from function arguments."""
    import hashlib
    
    # Create a string representation of arguments
    key_parts = [func_name]
    
    for arg in args:
        if isinstance(arg, (int, float, str)):
            key_parts.append(str(arg))
        elif hasattr(arg, '__hash__'):
            key_parts.append(str(hash(arg)))
    
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (int, float, str)):
            key_parts.append(f"{key}={value}")
    
    key_string = "_".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def _initialize_scaling_components(scaling_level: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize scaling components based on level."""
    components = {}
    
    # Always initialize cache manager
    components["cache_manager"] = get_adaptive_cache()
    
    if scaling_level in ["standard", "aggressive"]:
        components["parallel_engine"] = get_parallel_engine()
        components["resource_monitor"] = get_resource_monitor()
        components["memory_optimizer"] = get_memory_optimizer()
    
    if scaling_level == "aggressive":
        components["jax_engine"] = get_jax_engine()
        components["mesh_refinement"] = get_mesh_refinement()
        components["scaling_manager"] = get_scaling_manager()
    
    return components


async def _cleanup_scaling_components(components: Dict[str, Any]):
    """Cleanup scaling components."""
    # Most components are global singletons, so we don't shut them down
    # Just log the cleanup
    logger.debug(f"Cleaned up scaling components: {list(components.keys())}")


# High-level integration functions

async def solve_with_auto_scaling(solve_func: Callable, 
                                *args, 
                                auto_optimize: bool = True,
                                benchmark_mode: bool = False,
                                **kwargs) -> Tuple[Any, Optional[Dict]]:
    """High-level function to solve with automatic scaling optimization.
    
    Parameters
    ----------
    solve_func : Callable
        The solver function to execute
    *args
        Positional arguments for solver
    auto_optimize : bool, optional  
        Whether to automatically optimize based on problem size, by default True
    benchmark_mode : bool, optional
        Whether to collect detailed benchmarking data, by default False
    **kwargs
        Keyword arguments for solver
        
    Returns
    -------
    Tuple[Any, Optional[Dict]]
        Solver result and optional performance data
    """
    benchmark_data = None
    
    if benchmark_mode:
        # Run with comprehensive benchmarking
        start_time = time.time()
        
        # Estimate problem size for optimization
        if auto_optimize:
            problem_size = _estimate_problem_size(*args, **kwargs)
            scaling_level = "aggressive" if problem_size > 50000 else "standard" if problem_size > 1000 else "minimal"
        else:
            scaling_level = "standard"
        
        async with scaling_session(scaling_level=scaling_level) as components:
            result = await solve_func(*args, **kwargs)
            
            # Collect benchmark data
            total_time = time.time() - start_time
            benchmark_data = {
                "solve_time": total_time,
                "scaling_level_used": scaling_level,
                "estimated_problem_size": problem_size if auto_optimize else "unknown",
                "components_used": list(components.keys())
            }
            
            if "resource_monitor" in components:
                metrics = components["resource_monitor"].get_current_metrics()
                if metrics:
                    benchmark_data.update({
                        "final_cpu_usage": metrics.cpu_usage,
                        "final_memory_usage": metrics.memory_usage
                    })
    else:
        # Standard execution with optional auto-optimization
        if auto_optimize and hasattr(solve_func, '__self__') and hasattr(solve_func.__self__, 'optimize_for_problem_size'):
            problem_size = _estimate_problem_size(*args, **kwargs)
            solve_func.__self__.optimize_for_problem_size(problem_size)
        
        result = await solve_func(*args, **kwargs)
    
    return result, benchmark_data