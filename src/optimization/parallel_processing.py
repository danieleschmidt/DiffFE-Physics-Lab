"""Parallel processing and concurrency for DiffFE-Physics-Lab."""

import time
import threading
import multiprocessing
import concurrent.futures
import queue
from typing import Any, Callable, Dict, List, Optional, Union, Iterator
from dataclasses import dataclass
from functools import wraps, partial
import logging

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a parallel task."""
    task_id: str
    result: Any
    success: bool
    error: Optional[Exception] = None
    execution_time: float = 0.0


class ThreadPoolManager:
    """Advanced thread pool management."""
    
    def __init__(self, max_workers: Optional[int] = None, name: str = "default"):
        """Initialize thread pool manager.
        
        Args:
            max_workers: Maximum number of worker threads
            name: Pool name for identification
        """
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.name = name
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=f"DiffFE-{name}"
        )
        self.active_tasks = {}
        self.task_counter = 0
        self.lock = threading.Lock()
        
        logger.info(f"Thread pool '{name}' initialized with {self.max_workers} workers")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task to the thread pool.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        with self.lock:
            self.task_counter += 1
            task_id = f"{self.name}_task_{self.task_counter}"
        
        future = self.executor.submit(func, *args, **kwargs)
        
        with self.lock:
            self.active_tasks[task_id] = {
                "future": future,
                "submitted_at": time.time(),
                "function": func.__name__
            }
        
        logger.debug(f"Submitted task {task_id} to thread pool")
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get result of a submitted task.
        
        Args:
            task_id: Task identifier
            timeout: Timeout in seconds
            
        Returns:
            Task result
        """
        with self.lock:
            task_info = self.active_tasks.get(task_id)
        
        if not task_info:
            return TaskResult(task_id, None, False, ValueError(f"Task {task_id} not found"))
        
        future = task_info["future"]
        start_time = time.time()
        
        try:
            result = future.result(timeout=timeout)
            execution_time = time.time() - start_time
            
            with self.lock:
                self.active_tasks.pop(task_id, None)
            
            logger.debug(f"Task {task_id} completed successfully in {execution_time:.3f}s")
            return TaskResult(task_id, result, True, execution_time=execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            with self.lock:
                self.active_tasks.pop(task_id, None)
            
            logger.error(f"Task {task_id} failed after {execution_time:.3f}s: {e}")
            return TaskResult(task_id, None, False, e, execution_time)
    
    def wait_for_all(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for all active tasks to complete.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            List of all task results
        """
        with self.lock:
            task_ids = list(self.active_tasks.keys())
        
        results = []
        for task_id in task_ids:
            result = self.get_result(task_id, timeout)
            results.append(result)
        
        logger.info(f"Completed {len(results)} tasks from thread pool '{self.name}'")
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get thread pool status."""
        with self.lock:
            active_count = len(self.active_tasks)
            
            return {
                "name": self.name,
                "max_workers": self.max_workers,
                "active_tasks": active_count,
                "tasks_submitted": self.task_counter
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool."""
        logger.info(f"Shutting down thread pool '{self.name}'")
        self.executor.shutdown(wait=wait)


class ProcessPoolManager:
    """Advanced process pool management."""
    
    def __init__(self, max_workers: Optional[int] = None, name: str = "default"):
        """Initialize process pool manager.
        
        Args:
            max_workers: Maximum number of worker processes
            name: Pool name for identification
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.name = name
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )
        self.active_tasks = {}
        self.task_counter = 0
        self.lock = threading.Lock()
        
        logger.info(f"Process pool '{name}' initialized with {self.max_workers} workers")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task to the process pool."""
        with self.lock:
            self.task_counter += 1
            task_id = f"{self.name}_proc_{self.task_counter}"
        
        future = self.executor.submit(func, *args, **kwargs)
        
        with self.lock:
            self.active_tasks[task_id] = {
                "future": future,
                "submitted_at": time.time(),
                "function": func.__name__
            }
        
        logger.debug(f"Submitted task {task_id} to process pool")
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get result of a submitted task."""
        with self.lock:
            task_info = self.active_tasks.get(task_id)
        
        if not task_info:
            return TaskResult(task_id, None, False, ValueError(f"Task {task_id} not found"))
        
        future = task_info["future"]
        start_time = time.time()
        
        try:
            result = future.result(timeout=timeout)
            execution_time = time.time() - start_time
            
            with self.lock:
                self.active_tasks.pop(task_id, None)
            
            logger.debug(f"Process task {task_id} completed in {execution_time:.3f}s")
            return TaskResult(task_id, result, True, execution_time=execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            with self.lock:
                self.active_tasks.pop(task_id, None)
            
            logger.error(f"Process task {task_id} failed after {execution_time:.3f}s: {e}")
            return TaskResult(task_id, None, False, e, execution_time)
    
    def shutdown(self, wait: bool = True):
        """Shutdown the process pool."""
        logger.info(f"Shutting down process pool '{self.name}'")
        self.executor.shutdown(wait=wait)


class ParallelExecutor:
    """Unified parallel execution manager."""
    
    def __init__(self):
        """Initialize parallel executor."""
        self.thread_pools = {}
        self.process_pools = {}
        self.default_thread_pool = ThreadPoolManager(name="default_threads")
        self.default_process_pool = ProcessPoolManager(name="default_processes")
        
        logger.info("Parallel executor initialized")
    
    def create_thread_pool(self, name: str, max_workers: Optional[int] = None) -> ThreadPoolManager:
        """Create a named thread pool.
        
        Args:
            name: Pool name
            max_workers: Maximum workers
            
        Returns:
            Thread pool manager
        """
        pool = ThreadPoolManager(max_workers, name)
        self.thread_pools[name] = pool
        return pool
    
    def create_process_pool(self, name: str, max_workers: Optional[int] = None) -> ProcessPoolManager:
        """Create a named process pool.
        
        Args:
            name: Pool name
            max_workers: Maximum workers
            
        Returns:
            Process pool manager
        """
        pool = ProcessPoolManager(max_workers, name)
        self.process_pools[name] = pool
        return pool
    
    def submit_to_threads(self, func: Callable, *args, pool_name: Optional[str] = None, **kwargs) -> str:
        """Submit task to thread pool.
        
        Args:
            func: Function to execute
            *args: Function arguments
            pool_name: Specific pool to use
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if pool_name and pool_name in self.thread_pools:
            pool = self.thread_pools[pool_name]
        else:
            pool = self.default_thread_pool
        
        return pool.submit_task(func, *args, **kwargs)
    
    def submit_to_processes(self, func: Callable, *args, pool_name: Optional[str] = None, **kwargs) -> str:
        """Submit task to process pool.
        
        Args:
            func: Function to execute
            *args: Function arguments
            pool_name: Specific pool to use
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        if pool_name and pool_name in self.process_pools:
            pool = self.process_pools[pool_name]
        else:
            pool = self.default_process_pool
        
        return pool.submit_task(func, *args, **kwargs)
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get result from any pool."""
        # Try thread pools first
        for pool in [self.default_thread_pool] + list(self.thread_pools.values()):
            try:
                result = pool.get_result(task_id, 0.001)  # Very short timeout for check
                if result.task_id == task_id:
                    return result
            except:
                continue
        
        # Try process pools
        for pool in [self.default_process_pool] + list(self.process_pools.values()):
            try:
                result = pool.get_result(task_id, timeout)
                if result.task_id == task_id:
                    return result
            except:
                continue
        
        return TaskResult(task_id, None, False, ValueError(f"Task {task_id} not found"))
    
    def parallel_map(self, func: Callable, iterable: List[Any], 
                    use_processes: bool = False, chunk_size: Optional[int] = None) -> List[TaskResult]:
        """Execute function in parallel over iterable.
        
        Args:
            func: Function to execute
            iterable: Items to process
            use_processes: Use processes instead of threads
            chunk_size: Chunk size for processing
            
        Returns:
            List of results
        """
        if not iterable:
            return []
        
        chunk_size = chunk_size or max(1, len(iterable) // (self.default_thread_pool.max_workers * 2))
        
        # Submit tasks
        task_ids = []
        
        if use_processes:
            for item in iterable:
                task_id = self.submit_to_processes(func, item)
                task_ids.append(task_id)
        else:
            for item in iterable:
                task_id = self.submit_to_threads(func, item)
                task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            result = self.get_result(task_id)
            results.append(result)
        
        logger.info(f"Parallel map completed: {len(results)} items processed")
        return results
    
    def shutdown_all(self):
        """Shutdown all pools."""
        logger.info("Shutting down all parallel execution pools")
        
        self.default_thread_pool.shutdown()
        self.default_process_pool.shutdown()
        
        for pool in self.thread_pools.values():
            pool.shutdown()
        
        for pool in self.process_pools.values():
            pool.shutdown()


# Global parallel executor
global_parallel_executor = ParallelExecutor()


def parallel_map(func: Callable, iterable: List[Any], 
                use_processes: bool = False, max_workers: Optional[int] = None) -> List[Any]:
    """Simple parallel map function.
    
    Args:
        func: Function to apply
        iterable: Items to process
        use_processes: Use processes instead of threads
        max_workers: Maximum workers
        
    Returns:
        List of results
    """
    if not iterable:
        return []
    
    if use_processes:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(func, iterable))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(func, iterable))
    
    logger.info(f"Parallel map processed {len(results)} items")
    return results


def concurrent_solve(problems: List[Any], solver_func: Callable,
                    use_processes: bool = False) -> List[TaskResult]:
    """Solve multiple problems concurrently.
    
    Args:
        problems: List of problems to solve
        solver_func: Solver function
        use_processes: Use processes instead of threads
        
    Returns:
        List of solver results
    """
    logger.info(f"Starting concurrent solve of {len(problems)} problems")
    
    return global_parallel_executor.parallel_map(
        solver_func, problems, use_processes=use_processes
    )


def auto_parallelize(threshold: int = 10, use_processes: bool = False):
    """Decorator to automatically parallelize function over iterable arguments.
    
    Args:
        threshold: Minimum items to trigger parallelization
        use_processes: Use processes instead of threads
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Look for iterable arguments
            iterable_args = []
            
            for i, arg in enumerate(args):
                if hasattr(arg, '__iter__') and not isinstance(arg, (str, bytes)):
                    try:
                        length = len(arg)
                        if length >= threshold:
                            iterable_args.append((i, arg, length))
                    except TypeError:
                        pass
            
            # If we found suitable iterables, parallelize
            if iterable_args:
                # Use the largest iterable
                arg_index, iterable, length = max(iterable_args, key=lambda x: x[2])
                
                logger.info(f"Auto-parallelizing {func.__name__} over {length} items")
                
                # Create partial function with other arguments
                def partial_func(item):
                    new_args = list(args)
                    new_args[arg_index] = item
                    return func(*new_args, **kwargs)
                
                # Execute in parallel
                return parallel_map(partial_func, iterable, use_processes=use_processes)
            
            # Normal execution
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class AsyncBatch:
    """Asynchronous batch processing."""
    
    def __init__(self, batch_size: int = 10, max_wait: float = 1.0):
        """Initialize async batch processor.
        
        Args:
            batch_size: Maximum batch size
            max_wait: Maximum wait time before processing incomplete batch
        """
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.queue = queue.Queue()
        self.processing = False
        self.results = {}
        self.lock = threading.Lock()
        
    def add_item(self, item: Any, item_id: str) -> None:
        """Add item to batch queue.
        
        Args:
            item: Item to process
            item_id: Unique identifier for the item
        """
        self.queue.put((item, item_id))
        
        with self.lock:
            if not self.processing and self.queue.qsize() >= self.batch_size:
                self._start_batch_processing()
    
    def _start_batch_processing(self):
        """Start processing a batch."""
        if self.processing:
            return
        
        self.processing = True
        
        def process_batch():
            batch = []
            
            # Collect items for batch
            while len(batch) < self.batch_size and not self.queue.empty():
                try:
                    item, item_id = self.queue.get_nowait()
                    batch.append((item, item_id))
                except queue.Empty:
                    break
            
            if batch:
                logger.info(f"Processing batch of {len(batch)} items")
                
                # Process batch (placeholder)
                for item, item_id in batch:
                    self.results[item_id] = f"processed_{item}"
            
            with self.lock:
                self.processing = False
        
        # Start processing in background
        thread = threading.Thread(target=process_batch, daemon=True)
        thread.start()
    
    def get_result(self, item_id: str, timeout: float = 5.0) -> Optional[Any]:
        """Get result for item.
        
        Args:
            item_id: Item identifier
            timeout: Timeout in seconds
            
        Returns:
            Processed result or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if item_id in self.results:
                return self.results.pop(item_id)
            time.sleep(0.01)
        
        return None