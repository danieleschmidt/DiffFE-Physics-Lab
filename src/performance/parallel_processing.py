"""High-performance parallel processing and scaling system."""

import asyncio
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from jax import pmap, vmap

try:
    import mpi4py.MPI as MPI

    MPI_AVAILABLE = True
except ImportError:
    MPI = None
    MPI_AVAILABLE = False

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a parallel task execution."""

    task_id: str
    result: Any
    execution_time: float
    worker_id: str
    memory_usage: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class ResourceMetrics:
    """System resource usage metrics."""

    cpu_usage: float
    memory_usage: float
    gpu_usage: List[float] = field(default_factory=list)
    disk_io: float = 0.0
    network_io: float = 0.0
    timestamp: float = field(default_factory=time.time)


class WorkStealingQueue:
    """Work-stealing queue for dynamic load balancing."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.local_queue = queue.deque()
        self.global_queues = {}  # worker_id -> queue
        self.lock = threading.RLock()
        self.stats = {"tasks_processed": 0, "tasks_stolen": 0, "tasks_donated": 0}

    def put_task(self, task: Any):
        """Add task to local queue."""
        with self.lock:
            self.local_queue.append(task)

    def get_task(self) -> Optional[Any]:
        """Get task from local queue or steal from other workers."""
        with self.lock:
            # Try local queue first
            if self.local_queue:
                self.stats["tasks_processed"] += 1
                return self.local_queue.popleft()

            # Try to steal from other workers
            for worker_id, other_queue in self.global_queues.items():
                if worker_id != self.worker_id and other_queue.size() > 1:
                    stolen_task = other_queue.steal_task()
                    if stolen_task:
                        self.stats["tasks_stolen"] += 1
                        return stolen_task

            return None

    def steal_task(self) -> Optional[Any]:
        """Allow other workers to steal from this queue."""
        with self.lock:
            if len(self.local_queue) > 1:  # Keep at least one task
                self.stats["tasks_donated"] += 1
                return self.local_queue.pop()
            return None

    def size(self) -> int:
        """Get current queue size."""
        with self.lock:
            return len(self.local_queue)

    def register_global_queue(self, worker_id: str, work_queue: "WorkStealingQueue"):
        """Register another worker's queue for work stealing."""
        self.global_queues[worker_id] = work_queue


class ResourceMonitor:
    """Real-time resource usage monitor."""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = []
        self.max_history = 1000
        self.monitoring_active = False
        self.monitor_thread = None
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import GPUtil

            return True
        except ImportError:
            return False

    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)

                # Limit history size
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history :]

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU and memory
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # GPU usage
        gpu_usage = []
        if self.gpu_available:
            try:
                import GPUtil

                gpus = GPUtil.getGPUs()
                gpu_usage = [gpu.load * 100 for gpu in gpus]
            except:
                pass

        # Disk and network I/O
        disk_io = 0.0
        network_io = 0.0

        try:
            disk_stats = psutil.disk_io_counters()
            if disk_stats:
                disk_io = disk_stats.read_bytes + disk_stats.write_bytes

            network_stats = psutil.net_io_counters()
            if network_stats:
                network_io = network_stats.bytes_sent + network_stats.bytes_recv
        except:
            pass

        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            disk_io=disk_io,
            network_io=network_io,
        )

    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get current resource metrics."""
        if not self.metrics_history:
            return self._collect_metrics()
        return self.metrics_history[-1]

    def get_average_metrics(self, window_size: int = 10) -> ResourceMetrics:
        """Get average metrics over a window."""
        if not self.metrics_history:
            return self._collect_metrics()

        recent_metrics = self.metrics_history[-window_size:]

        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_gpu = []

        if recent_metrics and recent_metrics[0].gpu_usage:
            num_gpus = len(recent_metrics[0].gpu_usage)
            avg_gpu = [
                sum(m.gpu_usage[i] for m in recent_metrics if len(m.gpu_usage) > i)
                / len(recent_metrics)
                for i in range(num_gpus)
            ]

        return ResourceMetrics(
            cpu_usage=avg_cpu, memory_usage=avg_memory, gpu_usage=avg_gpu
        )


class GPUResourcePool:
    """GPU resource pool for multi-GPU systems."""

    def __init__(self):
        self.available_devices = []
        self.busy_devices = set()
        self.device_usage = {}
        self.lock = threading.RLock()
        self._initialize_devices()

    def _initialize_devices(self):
        """Initialize available GPU devices."""
        try:
            # JAX GPU detection
            devices = jax.devices("gpu")
            self.available_devices = [d.id for d in devices]
            logger.info(
                f"Found {len(self.available_devices)} GPU devices: {self.available_devices}"
            )

            # Initialize usage tracking
            for device_id in self.available_devices:
                self.device_usage[device_id] = {
                    "tasks": 0,
                    "memory": 0.0,
                    "utilization": 0.0,
                }

        except Exception as e:
            logger.warning(f"Failed to initialize GPU devices: {e}")
            self.available_devices = []

    def acquire_device(self, memory_requirement: float = 0.0) -> Optional[int]:
        """Acquire a GPU device for computation."""
        with self.lock:
            # Find device with lowest utilization
            best_device = None
            lowest_utilization = float("inf")

            for device_id in self.available_devices:
                if device_id not in self.busy_devices:
                    utilization = self.device_usage[device_id]["utilization"]
                    memory_used = self.device_usage[device_id]["memory"]

                    # Check if device has enough memory
                    if memory_requirement > 0:
                        try:
                            # Get device memory info (this would need actual implementation)
                            available_memory = self._get_device_memory(device_id)
                            if available_memory < memory_requirement:
                                continue
                        except:
                            pass

                    if utilization < lowest_utilization:
                        lowest_utilization = utilization
                        best_device = device_id

            if best_device is not None:
                self.busy_devices.add(best_device)
                self.device_usage[best_device]["tasks"] += 1
                logger.debug(f"Acquired GPU device {best_device}")
                return best_device

            return None

    def release_device(self, device_id: int):
        """Release a GPU device."""
        with self.lock:
            if device_id in self.busy_devices:
                self.busy_devices.remove(device_id)
                self.device_usage[device_id]["tasks"] = max(
                    0, self.device_usage[device_id]["tasks"] - 1
                )
                logger.debug(f"Released GPU device {device_id}")

    def _get_device_memory(self, device_id: int) -> float:
        """Get available memory for a device."""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if device_id < len(gpus):
                gpu = gpus[device_id]
                return gpu.memoryFree
        except:
            pass
        return float("inf")  # Assume unlimited if can't check

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get GPU pool statistics."""
        with self.lock:
            return {
                "total_devices": len(self.available_devices),
                "busy_devices": len(self.busy_devices),
                "available_devices": len(self.available_devices)
                - len(self.busy_devices),
                "device_usage": dict(self.device_usage),
                "device_ids": self.available_devices,
            }


class ParallelAssemblyEngine:
    """Parallel finite element assembly engine."""

    def __init__(
        self,
        num_threads: Optional[int] = None,
        enable_gpu: bool = True,
        chunk_size: str = "auto",
    ):
        self.num_threads = num_threads or min(8, os.cpu_count())
        self.enable_gpu = enable_gpu and len(jax.devices("gpu")) > 0
        self.chunk_size = chunk_size

        # Thread pool for CPU assembly
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)

        # GPU resource pool
        self.gpu_pool = GPUResourcePool() if self.enable_gpu else None

        # Work stealing queues for load balancing
        self.work_queues = {}
        for i in range(self.num_threads):
            worker_id = f"worker_{i}"
            self.work_queues[worker_id] = WorkStealingQueue(worker_id)

        # Register queues for work stealing
        for worker_id, queue in self.work_queues.items():
            for other_id, other_queue in self.work_queues.items():
                if worker_id != other_id:
                    queue.register_global_queue(other_id, other_queue)

        logger.info(
            f"Parallel assembly engine initialized: {self.num_threads} threads, GPU: {self.enable_gpu}"
        )

    def assemble_parallel(
        self,
        elements: List[Any],
        assembly_func: Callable,
        reduction_func: Optional[Callable] = None,
    ) -> Any:
        """Assemble system in parallel."""
        if not elements:
            return None

        # Determine chunk size
        if self.chunk_size == "auto":
            chunk_size = max(1, len(elements) // (self.num_threads * 4))
        else:
            chunk_size = self.chunk_size

        # Split elements into chunks
        chunks = [
            elements[i : i + chunk_size] for i in range(0, len(elements), chunk_size)
        ]

        # Decide on execution strategy
        if self.enable_gpu and len(elements) > 1000:  # Use GPU for large problems
            return self._assemble_gpu(chunks, assembly_func, reduction_func)
        else:
            return self._assemble_cpu(chunks, assembly_func, reduction_func)

    def _assemble_cpu(
        self,
        chunks: List[List],
        assembly_func: Callable,
        reduction_func: Optional[Callable],
    ) -> Any:
        """CPU-based parallel assembly."""
        futures = []

        # Submit chunks to thread pool
        for i, chunk in enumerate(chunks):
            future = self.thread_pool.submit(
                self._process_chunk, chunk, assembly_func, f"cpu_worker_{i}"
            )
            futures.append(future)

        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                if result.success:
                    results.append(result.result)
                else:
                    logger.error(f"Assembly chunk failed: {result.error}")
            except Exception as e:
                logger.error(f"Assembly future failed: {e}")

        # Reduce results
        if reduction_func and results:
            return reduction_func(results)
        elif results:
            # Default reduction: sum for matrices/arrays
            if hasattr(results[0], "shape") or hasattr(results[0], "__add__"):
                final_result = results[0]
                for result in results[1:]:
                    final_result = final_result + result
                return final_result
            else:
                return results

        return None

    def _assemble_gpu(
        self,
        chunks: List[List],
        assembly_func: Callable,
        reduction_func: Optional[Callable],
    ) -> Any:
        """GPU-accelerated parallel assembly."""
        if not self.gpu_pool:
            return self._assemble_cpu(chunks, assembly_func, reduction_func)

        # Acquire GPU device
        device_id = self.gpu_pool.acquire_device()
        if device_id is None:
            logger.warning("No GPU device available, falling back to CPU")
            return self._assemble_cpu(chunks, assembly_func, reduction_func)

        try:
            # Convert assembly function to JAX
            jax_assembly_func = jax.jit(
                assembly_func, device=jax.devices("gpu")[device_id]
            )

            # Process chunks on GPU
            results = []
            for chunk in chunks:
                try:
                    # Convert chunk data to JAX arrays if needed
                    jax_chunk = self._prepare_gpu_data(chunk)
                    result = jax_assembly_func(jax_chunk)
                    results.append(result)
                except Exception as e:
                    logger.error(f"GPU assembly chunk failed: {e}")
                    # Fallback to CPU for this chunk
                    cpu_result = self._process_chunk(
                        chunk, assembly_func, f"gpu_fallback_{device_id}"
                    )
                    if cpu_result.success:
                        results.append(cpu_result.result)

            # Reduce results on GPU
            if results and reduction_func:
                return reduction_func(results)
            elif results:
                # Default GPU reduction
                return jnp.sum(jnp.stack(results), axis=0)

            return None

        finally:
            self.gpu_pool.release_device(device_id)

    def _process_chunk(
        self, chunk: List, assembly_func: Callable, worker_id: str
    ) -> TaskResult:
        """Process a single chunk of elements."""
        start_time = time.time()

        try:
            # Get memory usage before processing
            process = psutil.Process()
            memory_before = process.memory_info().rss

            # Process chunk
            result = assembly_func(chunk)

            # Calculate execution metrics
            execution_time = time.time() - start_time
            memory_after = process.memory_info().rss
            memory_usage = (memory_after - memory_before) / (1024 * 1024)  # MB

            return TaskResult(
                task_id=f"chunk_{id(chunk)}",
                result=result,
                execution_time=execution_time,
                worker_id=worker_id,
                memory_usage=memory_usage,
                success=True,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Chunk processing failed in {worker_id}: {e}")

            return TaskResult(
                task_id=f"chunk_{id(chunk)}",
                result=None,
                execution_time=execution_time,
                worker_id=worker_id,
                success=False,
                error=str(e),
            )

    def _prepare_gpu_data(self, chunk: List) -> Any:
        """Prepare data for GPU processing."""
        # Convert chunk to JAX arrays if possible
        try:
            if isinstance(chunk[0], (list, tuple, np.ndarray)):
                return jnp.array(chunk)
            else:
                # Keep as is for custom data types
                return chunk
        except:
            return chunk

    def get_assembly_stats(self) -> Dict[str, Any]:
        """Get assembly performance statistics."""
        stats = {
            "num_threads": self.num_threads,
            "gpu_enabled": self.enable_gpu,
            "work_stealing_stats": {},
        }

        # Work stealing statistics
        for worker_id, queue in self.work_queues.items():
            stats["work_stealing_stats"][worker_id] = queue.stats.copy()

        # GPU pool statistics
        if self.gpu_pool:
            stats["gpu_pool"] = self.gpu_pool.get_pool_stats()

        return stats

    def shutdown(self):
        """Shutdown parallel assembly engine."""
        logger.info("Shutting down parallel assembly engine...")

        self.thread_pool.shutdown(wait=True)

        logger.info("Parallel assembly engine shutdown completed")


class DistributedComputeManager:
    """Distributed computing manager using MPI or Ray."""

    def __init__(self, backend: str = "auto", **kwargs):
        self.backend = backend
        self.initialized = False
        self.rank = 0
        self.size = 1
        self.node_resources = {}

        if backend == "auto":
            if RAY_AVAILABLE:
                self.backend = "ray"
            elif MPI_AVAILABLE:
                self.backend = "mpi"
            else:
                self.backend = "local"
                logger.warning(
                    "No distributed backend available, using local processing"
                )

        self._initialize_backend(**kwargs)

    def _initialize_backend(self, **kwargs):
        """Initialize the distributed computing backend."""
        try:
            if self.backend == "ray":
                self._initialize_ray(**kwargs)
            elif self.backend == "mpi":
                self._initialize_mpi(**kwargs)
            else:
                self._initialize_local(**kwargs)

            self.initialized = True
            logger.info(
                f"Distributed backend '{self.backend}' initialized (rank: {self.rank}, size: {self.size})"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize distributed backend '{self.backend}': {e}"
            )
            self.backend = "local"
            self._initialize_local(**kwargs)

    def _initialize_ray(self, **kwargs):
        """Initialize Ray backend."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray not available")

        if not ray.is_initialized():
            ray.init(**kwargs)

        # Get cluster information
        cluster_resources = ray.cluster_resources()
        self.node_resources = cluster_resources
        self.size = cluster_resources.get("CPU", 1)

    def _initialize_mpi(self, **kwargs):
        """Initialize MPI backend."""
        if not MPI_AVAILABLE:
            raise RuntimeError("MPI not available")

        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # Gather node information
        hostname = MPI.Get_processor_name()
        all_hostnames = comm.allgather(hostname)
        self.node_resources = {"hostnames": all_hostnames}

    def _initialize_local(self, **kwargs):
        """Initialize local processing backend."""
        self.rank = 0
        self.size = 1
        self.node_resources = {
            "CPU": os.cpu_count(),
            "memory": psutil.virtual_memory().total,
        }

    @ray.remote
    def _ray_compute_task(
        self, task_func: Callable, task_data: Any, task_id: str
    ) -> TaskResult:
        """Ray remote task execution."""
        start_time = time.time()

        try:
            result = task_func(task_data)
            execution_time = time.time() - start_time

            return TaskResult(
                task_id=task_id,
                result=result,
                execution_time=execution_time,
                worker_id=ray.get_runtime_context().worker_id,
                success=True,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task_id,
                result=None,
                execution_time=execution_time,
                worker_id=ray.get_runtime_context().worker_id,
                success=False,
                error=str(e),
            )

    def distribute_computation(
        self,
        tasks: List[Tuple[str, Any]],
        compute_func: Callable,
        timeout: float = 300.0,
    ) -> List[TaskResult]:
        """Distribute computation across available resources."""
        if not self.initialized:
            logger.error("Distributed backend not initialized")
            return []

        if self.backend == "ray":
            return self._distribute_ray(tasks, compute_func, timeout)
        elif self.backend == "mpi":
            return self._distribute_mpi(tasks, compute_func, timeout)
        else:
            return self._distribute_local(tasks, compute_func, timeout)

    def _distribute_ray(
        self, tasks: List[Tuple[str, Any]], compute_func: Callable, timeout: float
    ) -> List[TaskResult]:
        """Distribute tasks using Ray."""
        # Submit all tasks
        futures = []
        for task_id, task_data in tasks:
            future = self._ray_compute_task.remote(compute_func, task_data, task_id)
            futures.append(future)

        # Wait for results with timeout
        try:
            results = ray.get(futures, timeout=timeout)
            return results
        except ray.exceptions.RayTimeoutError:
            logger.error(f"Ray computation timed out after {timeout}s")
            return []

    def _distribute_mpi(
        self, tasks: List[Tuple[str, Any]], compute_func: Callable, timeout: float
    ) -> List[TaskResult]:
        """Distribute tasks using MPI."""
        comm = MPI.COMM_WORLD

        # Scatter tasks to workers
        if self.rank == 0:  # Master process
            # Distribute tasks among workers
            tasks_per_worker = len(tasks) // self.size
            scattered_tasks = []

            for i in range(self.size):
                start_idx = i * tasks_per_worker
                end_idx = (
                    start_idx + tasks_per_worker if i < self.size - 1 else len(tasks)
                )
                worker_tasks = tasks[start_idx:end_idx]
                scattered_tasks.append(worker_tasks)
        else:
            scattered_tasks = None

        # Scatter tasks
        local_tasks = comm.scatter(scattered_tasks, root=0)

        # Process local tasks
        local_results = []
        for task_id, task_data in local_tasks:
            start_time = time.time()

            try:
                result = compute_func(task_data)
                execution_time = time.time() - start_time

                local_results.append(
                    TaskResult(
                        task_id=task_id,
                        result=result,
                        execution_time=execution_time,
                        worker_id=f"mpi_rank_{self.rank}",
                        success=True,
                    )
                )

            except Exception as e:
                execution_time = time.time() - start_time
                local_results.append(
                    TaskResult(
                        task_id=task_id,
                        result=None,
                        execution_time=execution_time,
                        worker_id=f"mpi_rank_{self.rank}",
                        success=False,
                        error=str(e),
                    )
                )

        # Gather results
        all_results = comm.gather(local_results, root=0)

        if self.rank == 0:
            # Flatten results
            flattened_results = []
            for worker_results in all_results:
                flattened_results.extend(worker_results)
            return flattened_results
        else:
            return []

    def _distribute_local(
        self, tasks: List[Tuple[str, Any]], compute_func: Callable, timeout: float
    ) -> List[TaskResult]:
        """Process tasks locally."""
        results = []

        for task_id, task_data in tasks:
            start_time = time.time()

            try:
                result = compute_func(task_data)
                execution_time = time.time() - start_time

                results.append(
                    TaskResult(
                        task_id=task_id,
                        result=result,
                        execution_time=execution_time,
                        worker_id="local_worker",
                        success=True,
                    )
                )

            except Exception as e:
                execution_time = time.time() - start_time
                results.append(
                    TaskResult(
                        task_id=task_id,
                        result=None,
                        execution_time=execution_time,
                        worker_id="local_worker",
                        success=False,
                        error=str(e),
                    )
                )

        return results

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get distributed cluster information."""
        return {
            "backend": self.backend,
            "rank": self.rank,
            "size": self.size,
            "node_resources": self.node_resources,
            "initialized": self.initialized,
        }

    def shutdown(self):
        """Shutdown distributed computing manager."""
        logger.info(f"Shutting down distributed backend '{self.backend}'...")

        if self.backend == "ray" and RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()

        logger.info("Distributed backend shutdown completed")


class AutoScalingManager:
    """Auto-scaling manager based on resource usage and performance metrics."""

    def __init__(
        self,
        resource_monitor: ResourceMonitor,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 30.0,
        evaluation_window: int = 60,  # seconds
        min_workers: int = 1,
        max_workers: int = 16,
    ):
        self.resource_monitor = resource_monitor
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.evaluation_window = evaluation_window
        self.min_workers = min_workers
        self.max_workers = max_workers

        self.current_workers = min_workers
        self.scaling_history = []
        self.last_scaling_action = time.time()
        self.scaling_cooldown = 30.0  # seconds

        self.scaling_active = False
        self.scaling_thread = None

        logger.info(
            f"Auto-scaling manager initialized: {min_workers}-{max_workers} workers"
        )

    def start_auto_scaling(self):
        """Start auto-scaling monitoring."""
        if self.scaling_active:
            return

        self.scaling_active = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        logger.info("Auto-scaling started")

    def stop_auto_scaling(self):
        """Stop auto-scaling monitoring."""
        self.scaling_active = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
        logger.info("Auto-scaling stopped")

    def _scaling_loop(self):
        """Main auto-scaling evaluation loop."""
        while self.scaling_active:
            try:
                time.sleep(10)  # Check every 10 seconds

                # Check if enough time has passed since last scaling action
                if time.time() - self.last_scaling_action < self.scaling_cooldown:
                    continue

                # Get average resource usage over evaluation window
                avg_metrics = self.resource_monitor.get_average_metrics(
                    self.evaluation_window // 10
                )

                # Make scaling decision
                self._evaluate_scaling(avg_metrics)

            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")

    def _evaluate_scaling(self, metrics: ResourceMetrics):
        """Evaluate whether scaling action is needed."""
        cpu_usage = metrics.cpu_usage
        memory_usage = metrics.memory_usage

        # Calculate combined resource pressure
        resource_pressure = max(cpu_usage, memory_usage)

        scaling_action = None

        if (
            resource_pressure > self.scale_up_threshold
            and self.current_workers < self.max_workers
        ):
            # Scale up
            new_workers = min(self.current_workers + 1, self.max_workers)
            scaling_action = ("scale_up", new_workers, resource_pressure)

        elif (
            resource_pressure < self.scale_down_threshold
            and self.current_workers > self.min_workers
        ):
            # Scale down
            new_workers = max(self.current_workers - 1, self.min_workers)
            scaling_action = ("scale_down", new_workers, resource_pressure)

        if scaling_action:
            action, new_workers, pressure = scaling_action
            logger.info(
                f"Auto-scaling {action}: {self.current_workers} -> {new_workers} workers "
                f"(resource pressure: {pressure:.1f}%)"
            )

            self.current_workers = new_workers
            self.last_scaling_action = time.time()

            # Record scaling history
            self.scaling_history.append(
                {
                    "timestamp": time.time(),
                    "action": action,
                    "old_workers": (
                        self.current_workers
                        if action == "scale_down"
                        else self.current_workers - 1
                    ),
                    "new_workers": new_workers,
                    "resource_pressure": pressure,
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                }
            )

            # Limit history size
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-100:]

    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        return {
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "scaling_active": self.scaling_active,
            "last_scaling_action": self.last_scaling_action,
            "scaling_history_count": len(self.scaling_history),
            "recent_scaling_history": (
                self.scaling_history[-10:] if self.scaling_history else []
            ),
        }

    def manual_scale(self, target_workers: int):
        """Manually set number of workers."""
        if target_workers < self.min_workers or target_workers > self.max_workers:
            logger.error(
                f"Target workers {target_workers} outside valid range [{self.min_workers}, {self.max_workers}]"
            )
            return False

        old_workers = self.current_workers
        self.current_workers = target_workers
        self.last_scaling_action = time.time()

        logger.info(f"Manual scaling: {old_workers} -> {target_workers} workers")

        # Record in history
        self.scaling_history.append(
            {
                "timestamp": time.time(),
                "action": "manual_scale",
                "old_workers": old_workers,
                "new_workers": target_workers,
                "resource_pressure": 0.0,
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
            }
        )

        return True


# Global instances
_global_resource_monitor = None
_global_parallel_engine = None
_global_distributed_manager = None
_global_autoscaling_manager = None


def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor."""
    global _global_resource_monitor
    if _global_resource_monitor is None:
        _global_resource_monitor = ResourceMonitor()
    return _global_resource_monitor


def get_parallel_engine() -> ParallelAssemblyEngine:
    """Get global parallel assembly engine."""
    global _global_parallel_engine
    if _global_parallel_engine is None:
        _global_parallel_engine = ParallelAssemblyEngine()
    return _global_parallel_engine


def get_distributed_manager(
    backend: str = "auto", **kwargs
) -> DistributedComputeManager:
    """Get global distributed compute manager."""
    global _global_distributed_manager
    if _global_distributed_manager is None:
        _global_distributed_manager = DistributedComputeManager(backend, **kwargs)
    return _global_distributed_manager


def get_autoscaling_manager() -> AutoScalingManager:
    """Get global auto-scaling manager."""
    global _global_autoscaling_manager
    if _global_autoscaling_manager is None:
        monitor = get_resource_monitor()
        _global_autoscaling_manager = AutoScalingManager(monitor)
    return _global_autoscaling_manager


def parallel_assembly(elements: List[Any], assembly_func: Callable):
    """Decorator for parallel assembly operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            engine = get_parallel_engine()
            return engine.assemble_parallel(elements, assembly_func)

        return wrapper

    return decorator


def distributed_compute(tasks: List[Tuple[str, Any]], backend: str = "auto"):
    """Decorator for distributed computation."""

    def decorator(compute_func):
        @wraps(compute_func)
        def wrapper(*args, **kwargs):
            manager = get_distributed_manager(backend)
            return manager.distribute_computation(tasks, compute_func)

        return wrapper

    return decorator
