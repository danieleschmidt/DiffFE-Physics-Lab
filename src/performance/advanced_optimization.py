"""Advanced optimization features for enterprise-scale performance."""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

import aiofiles
import aiohttp
import asyncpg
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad, jit, pmap, vmap
from jax.experimental import checkify

logger = logging.getLogger(__name__)


@dataclass
class MeshElement:
    """Mesh element for adaptive refinement."""

    id: str
    vertices: np.ndarray
    error_indicator: float = 0.0
    refinement_level: int = 0
    is_refined: bool = False
    children: List["MeshElement"] = field(default_factory=list)
    parent: Optional["MeshElement"] = None


@dataclass
class OptimizationMetrics:
    """Optimization performance metrics."""

    iteration: int
    objective_value: float
    gradient_norm: float
    step_size: float
    convergence_rate: float
    memory_usage_mb: float
    computation_time: float


class AdaptiveMeshRefinement:
    """Adaptive mesh refinement with load balancing."""

    def __init__(
        self,
        refinement_threshold: float = 0.1,
        coarsening_threshold: float = 0.01,
        max_refinement_level: int = 10,
        load_balancing_enabled: bool = True,
    ):
        self.refinement_threshold = refinement_threshold
        self.coarsening_threshold = coarsening_threshold
        self.max_refinement_level = max_refinement_level
        self.load_balancing_enabled = load_balancing_enabled

        self.mesh_elements = {}  # id -> MeshElement
        self.refinement_history = []
        self.load_distribution = {}  # processor_id -> element_count

        logger.info(
            f"Adaptive mesh refinement initialized (threshold: {refinement_threshold})"
        )

    def compute_error_indicators(
        self, solution: np.ndarray, elements: List[MeshElement]
    ) -> Dict[str, float]:
        """Compute error indicators for mesh elements."""
        error_indicators = {}

        for element in elements:
            try:
                # Compute element-wise error using gradient-based indicator
                element_solution = self._extract_element_solution(solution, element)

                # Compute gradient
                gradient = np.gradient(element_solution)
                gradient_magnitude = np.linalg.norm(gradient)

                # Normalize by element size
                element_size = self._compute_element_size(element)
                error_indicator = gradient_magnitude * element_size**0.5

                error_indicators[element.id] = error_indicator
                element.error_indicator = error_indicator

            except Exception as e:
                logger.warning(
                    f"Failed to compute error indicator for element {element.id}: {e}"
                )
                error_indicators[element.id] = 0.0
                element.error_indicator = 0.0

        return error_indicators

    def refine_mesh(
        self, elements: List[MeshElement], error_indicators: Dict[str, float]
    ) -> List[MeshElement]:
        """Perform adaptive mesh refinement."""
        refined_elements = []
        refinement_count = 0
        coarsening_count = 0

        for element in elements:
            error = error_indicators.get(element.id, 0.0)

            # Refinement decision
            if (
                error > self.refinement_threshold
                and element.refinement_level < self.max_refinement_level
                and not element.is_refined
            ):

                # Refine element
                children = self._refine_element(element)
                refined_elements.extend(children)
                element.is_refined = True
                element.children = children
                refinement_count += 1

                logger.debug(f"Refined element {element.id} (error: {error:.6f})")

            elif (
                error < self.coarsening_threshold
                and element.parent is not None
                and element.refinement_level > 0
            ):

                # Consider coarsening
                if self._can_coarsen(element):
                    coarsened_element = self._coarsen_element(element)
                    refined_elements.append(coarsened_element)
                    coarsening_count += 1

                    logger.debug(f"Coarsened element {element.id} (error: {error:.6f})")
                else:
                    refined_elements.append(element)
            else:
                # Keep element unchanged
                if not element.is_refined:
                    refined_elements.append(element)

        # Update mesh elements registry
        for element in refined_elements:
            self.mesh_elements[element.id] = element

        # Record refinement history
        self.refinement_history.append(
            {
                "timestamp": time.time(),
                "refinement_count": refinement_count,
                "coarsening_count": coarsening_count,
                "total_elements": len(refined_elements),
            }
        )

        logger.info(
            f"Mesh adaptation completed: +{refinement_count} refined, -{coarsening_count} coarsened, "
            f"total: {len(refined_elements)} elements"
        )

        # Load balancing
        if self.load_balancing_enabled:
            refined_elements = self._balance_load(refined_elements)

        return refined_elements

    def _extract_element_solution(
        self, solution: np.ndarray, element: MeshElement
    ) -> np.ndarray:
        """Extract solution values for a specific element."""
        # Simplified: assume solution is indexed by element vertices
        # In practice, this would use proper finite element interpolation
        vertex_indices = getattr(element, "vertex_indices", [0])
        return (
            solution[vertex_indices]
            if len(vertex_indices) <= len(solution)
            else solution[: len(vertex_indices)]
        )

    def _compute_element_size(self, element: MeshElement) -> float:
        """Compute characteristic size of an element."""
        vertices = element.vertices
        if vertices.shape[0] < 2:
            return 1.0

        # Compute diameter (max distance between vertices)
        distances = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                distances.append(dist)

        return max(distances) if distances else 1.0

    def _refine_element(self, element: MeshElement) -> List[MeshElement]:
        """Refine a single element into children."""
        # Simplified 2D refinement: split element into 4 children
        children = []

        if element.vertices.shape[0] >= 3:  # At least a triangle
            # Compute midpoints and create child elements
            vertices = element.vertices
            center = np.mean(vertices, axis=0)

            for i in range(len(vertices)):
                # Create child element
                child_vertices = np.array(
                    [vertices[i], vertices[(i + 1) % len(vertices)], center]
                )

                child = MeshElement(
                    id=f"{element.id}_child_{i}",
                    vertices=child_vertices,
                    refinement_level=element.refinement_level + 1,
                    parent=element,
                )
                children.append(child)
        else:
            # Fallback: just create a copy with higher refinement level
            child = MeshElement(
                id=f"{element.id}_refined",
                vertices=element.vertices.copy(),
                refinement_level=element.refinement_level + 1,
                parent=element,
            )
            children.append(child)

        return children

    def _can_coarsen(self, element: MeshElement) -> bool:
        """Check if element can be coarsened."""
        if element.parent is None:
            return False

        # Check if all siblings have low error indicators
        parent = element.parent
        if parent.children:
            for sibling in parent.children:
                if sibling.error_indicator > self.coarsening_threshold:
                    return False

        return True

    def _coarsen_element(self, element: MeshElement) -> MeshElement:
        """Coarsen element by returning to parent."""
        parent = element.parent
        parent.is_refined = False
        parent.children = []
        parent.refinement_level = max(0, parent.refinement_level - 1)
        return parent

    def _balance_load(self, elements: List[MeshElement]) -> List[MeshElement]:
        """Balance computational load across processors."""
        if not elements:
            return elements

        # Simulate load balancing by sorting elements by computational weight
        def compute_weight(element):
            # Higher refinement level = more computational cost
            return 2**element.refinement_level

        # Sort elements by weight (descending)
        weighted_elements = [(element, compute_weight(element)) for element in elements]
        weighted_elements.sort(key=lambda x: x[1], reverse=True)

        # Distribute to processors (simulated)
        num_processors = 4  # Could be dynamic based on available resources
        processor_loads = [0] * num_processors
        processor_elements = [[] for _ in range(num_processors)]

        for element, weight in weighted_elements:
            # Assign to processor with lowest current load
            min_load_processor = min(
                range(num_processors), key=lambda i: processor_loads[i]
            )
            processor_elements[min_load_processor].append(element)
            processor_loads[min_load_processor] += weight

        # Update load distribution tracking
        self.load_distribution = {
            f"processor_{i}": len(processor_elements[i]) for i in range(num_processors)
        }

        logger.debug(f"Load balancing: {self.load_distribution}")

        # Return flattened list (in practice, elements would be distributed to actual processors)
        return elements

    def get_mesh_stats(self) -> Dict[str, Any]:
        """Get mesh refinement statistics."""
        if not self.mesh_elements:
            return {"total_elements": 0}

        elements = list(self.mesh_elements.values())

        refinement_levels = [e.refinement_level for e in elements]
        error_indicators = [e.error_indicator for e in elements]

        return {
            "total_elements": len(elements),
            "refinement_levels": {
                "min": min(refinement_levels) if refinement_levels else 0,
                "max": max(refinement_levels) if refinement_levels else 0,
                "avg": (
                    sum(refinement_levels) / len(refinement_levels)
                    if refinement_levels
                    else 0
                ),
            },
            "error_indicators": {
                "min": min(error_indicators) if error_indicators else 0,
                "max": max(error_indicators) if error_indicators else 0,
                "avg": (
                    sum(error_indicators) / len(error_indicators)
                    if error_indicators
                    else 0
                ),
            },
            "load_distribution": self.load_distribution,
            "refinement_history_count": len(self.refinement_history),
        }


class JAXOptimizationEngine:
    """JAX-based optimization with advanced features."""

    def __init__(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        enable_checkpointing: bool = True,
        enable_mixed_precision: bool = False,
    ):
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.enable_checkpointing = enable_checkpointing
        self.enable_mixed_precision = enable_mixed_precision

        # Create optimizer
        self.optimizer = self._create_optimizer(optimizer, learning_rate)
        self.opt_state = None

        # Compilation cache
        self.compiled_functions = {}

        # Checkpointing
        self.checkpointed_functions = {}

        # Mixed precision policy
        if enable_mixed_precision:
            self.precision_policy = jnp.bfloat16
        else:
            self.precision_policy = jnp.float32

        logger.info(
            f"JAX optimization engine initialized: {optimizer}, lr={learning_rate}"
        )

    def _create_optimizer(self, optimizer_name: str, learning_rate: float):
        """Create Optax optimizer."""
        optimizers = {
            "adam": optax.adam(learning_rate),
            "adamw": optax.adamw(learning_rate),
            "sgd": optax.sgd(learning_rate),
            "rmsprop": optax.rmsprop(learning_rate),
            "adagrad": optax.adagrad(learning_rate),
        }

        if optimizer_name not in optimizers:
            logger.warning(f"Unknown optimizer {optimizer_name}, using Adam")
            optimizer_name = "adam"

        return optimizers[optimizer_name]

    def compile_objective(
        self, objective_func: Callable, static_argnums: Tuple[int, ...] = ()
    ) -> Callable:
        """Compile objective function with JAX JIT."""
        func_key = f"objective_{id(objective_func)}"

        if func_key not in self.compiled_functions:
            # Apply mixed precision if enabled
            if self.enable_mixed_precision:

                def mixed_precision_wrapper(params, *args):
                    # Cast inputs to mixed precision
                    params = jax.tree_map(
                        lambda x: x.astype(self.precision_policy), params
                    )
                    result = objective_func(params, *args)
                    # Cast result back to float32 for stability
                    return result.astype(jnp.float32)

                compiled_func = jit(
                    mixed_precision_wrapper, static_argnums=static_argnums
                )
            else:
                compiled_func = jit(objective_func, static_argnums=static_argnums)

            self.compiled_functions[func_key] = compiled_func
            logger.debug(f"Compiled objective function: {func_key}")

        return self.compiled_functions[func_key]

    def compile_gradient(
        self, objective_func: Callable, static_argnums: Tuple[int, ...] = ()
    ) -> Callable:
        """Compile gradient function with JAX."""
        func_key = f"gradient_{id(objective_func)}"

        if func_key not in self.compiled_functions:
            # Create gradient function
            grad_func = grad(objective_func, argnums=0)

            # Apply checkpointing if enabled
            if self.enable_checkpointing:
                grad_func = self._apply_gradient_checkpointing(grad_func)

            # Apply mixed precision if enabled
            if self.enable_mixed_precision:

                def mixed_precision_grad_wrapper(params, *args):
                    params = jax.tree_map(
                        lambda x: x.astype(self.precision_policy), params
                    )
                    grads = grad_func(params, *args)
                    return jax.tree_map(lambda x: x.astype(jnp.float32), grads)

                compiled_func = jit(
                    mixed_precision_grad_wrapper, static_argnums=static_argnums
                )
            else:
                compiled_func = jit(grad_func, static_argnums=static_argnums)

            self.compiled_functions[func_key] = compiled_func
            logger.debug(f"Compiled gradient function: {func_key}")

        return self.compiled_functions[func_key]

    def _apply_gradient_checkpointing(self, grad_func: Callable) -> Callable:
        """Apply gradient checkpointing for memory efficiency."""

        def checkpointed_grad_func(params, *args):
            # Use JAX's remat (rematerialization) for gradient checkpointing
            checkpointed_func = jax.remat(grad_func)
            return checkpointed_func(params, *args)

        return checkpointed_grad_func

    def optimize_step(
        self, params: Dict[str, Any], objective_func: Callable, *args
    ) -> Tuple[Dict[str, Any], OptimizationMetrics]:
        """Perform a single optimization step."""
        start_time = time.time()

        # Initialize optimizer state if needed
        if self.opt_state is None:
            self.opt_state = self.optimizer.init(params)

        # Compile functions
        compiled_objective = self.compile_objective(objective_func)
        compiled_gradient = self.compile_gradient(objective_func)

        # Compute objective and gradients
        objective_value = compiled_objective(params, *args)
        gradients = compiled_gradient(params, *args)

        # Update parameters
        updates, self.opt_state = self.optimizer.update(gradients, self.opt_state)
        new_params = optax.apply_updates(params, updates)

        # Compute metrics
        gradient_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_leaves(gradients)))
        step_size = jnp.sqrt(sum(jnp.sum(u**2) for u in jax.tree_leaves(updates)))

        computation_time = time.time() - start_time

        # Estimate memory usage (simplified)
        memory_usage_mb = sum(p.size * p.itemsize for p in jax.tree_leaves(params)) / (
            1024 * 1024
        )

        metrics = OptimizationMetrics(
            iteration=getattr(self, "_iteration", 0),
            objective_value=float(objective_value),
            gradient_norm=float(gradient_norm),
            step_size=float(step_size),
            convergence_rate=0.0,  # Would need history to compute
            memory_usage_mb=memory_usage_mb,
            computation_time=computation_time,
        )

        self._iteration = getattr(self, "_iteration", 0) + 1

        return new_params, metrics

    def optimize(
        self,
        initial_params: Dict[str, Any],
        objective_func: Callable,
        num_iterations: int = 100,
        convergence_tolerance: float = 1e-6,
        *args,
    ) -> Tuple[Dict[str, Any], List[OptimizationMetrics]]:
        """Run full optimization loop."""
        params = initial_params
        metrics_history = []

        logger.info(f"Starting optimization: {num_iterations} iterations")

        for i in range(num_iterations):
            params, metrics = self.optimize_step(params, objective_func, *args)
            metrics_history.append(metrics)

            # Check convergence
            if i > 0:
                convergence_rate = abs(
                    metrics_history[-1].objective_value
                    - metrics_history[-2].objective_value
                )
                metrics_history[-1].convergence_rate = convergence_rate

                if convergence_rate < convergence_tolerance:
                    logger.info(f"Converged after {i+1} iterations")
                    break

            if i % 10 == 0:
                logger.debug(
                    f"Iteration {i}: objective={metrics.objective_value:.6f}, "
                    f"grad_norm={metrics.gradient_norm:.6f}"
                )

        logger.info(
            f"Optimization completed: final objective={metrics_history[-1].objective_value:.6f}"
        )

        return params, metrics_history

    def get_compiled_function_stats(self) -> Dict[str, Any]:
        """Get statistics about compiled functions."""
        return {
            "compiled_functions_count": len(self.compiled_functions),
            "checkpointing_enabled": self.enable_checkpointing,
            "mixed_precision_enabled": self.enable_mixed_precision,
            "precision_policy": str(self.precision_policy),
            "optimizer": self.optimizer_name,
            "learning_rate": self.learning_rate,
        }


class AsyncIOManager:
    """Asynchronous I/O operations for high-performance data processing."""

    def __init__(self, max_concurrent_operations: int = 100):
        self.max_concurrent_operations = max_concurrent_operations
        self.semaphore = asyncio.Semaphore(max_concurrent_operations)
        self.session = None
        self.db_pool = None

        logger.info(
            f"Async I/O manager initialized (max concurrent: {max_concurrent_operations})"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        if self.db_pool:
            await self.db_pool.close()

    async def read_file_async(self, file_path: str) -> str:
        """Asynchronously read file."""
        async with self.semaphore:
            try:
                async with aiofiles.open(file_path, "r") as f:
                    content = await f.read()
                return content
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return ""

    async def write_file_async(self, file_path: str, content: str) -> bool:
        """Asynchronously write file."""
        async with self.semaphore:
            try:
                # Ensure parent directory exists
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)

                async with aiofiles.open(file_path, "w") as f:
                    await f.write(content)
                return True
            except Exception as e:
                logger.error(f"Failed to write file {file_path}: {e}")
                return False

    async def read_multiple_files(self, file_paths: List[str]) -> List[Tuple[str, str]]:
        """Read multiple files concurrently."""
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(self._read_file_with_path(file_path))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        successful_results = [
            result for result in results if not isinstance(result, Exception)
        ]

        return successful_results

    async def _read_file_with_path(self, file_path: str) -> Tuple[str, str]:
        """Helper to read file and return with path."""
        content = await self.read_file_async(file_path)
        return (file_path, content)

    async def batch_write_files(self, file_data: List[Tuple[str, str]]) -> List[bool]:
        """Write multiple files concurrently."""
        tasks = []
        for file_path, content in file_data:
            task = asyncio.create_task(self.write_file_async(file_path, content))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to False
        return [
            result if not isinstance(result, Exception) else False for result in results
        ]

    async def initialize_database_pool(
        self, database_url: str, min_connections: int = 5, max_connections: int = 20
    ):
        """Initialize database connection pool."""
        try:
            self.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=min_connections,
                max_size=max_connections,
                command_timeout=60,
            )
            logger.info(
                f"Database pool initialized: {min_connections}-{max_connections} connections"
            )
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")

    async def execute_query_batch(self, queries: List[str]) -> List[Any]:
        """Execute multiple database queries concurrently."""
        if not self.db_pool:
            logger.error("Database pool not initialized")
            return []

        async def execute_single_query(query: str):
            async with self.db_pool.acquire() as conn:
                try:
                    return await conn.fetch(query)
                except Exception as e:
                    logger.error(f"Query failed: {e}")
                    return None

        tasks = [execute_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        return [result for result in results if not isinstance(result, Exception)]

    async def stream_large_file(
        self, file_path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Stream large file in chunks."""
        async with self.semaphore:
            try:
                async with aiofiles.open(file_path, "rb") as f:
                    while True:
                        chunk = await f.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
            except Exception as e:
                logger.error(f"Failed to stream file {file_path}: {e}")

    async def download_file_async(self, url: str, destination: str) -> bool:
        """Download file asynchronously."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        # Ensure parent directory exists
                        Path(destination).parent.mkdir(parents=True, exist_ok=True)

                        async with aiofiles.open(destination, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        return True
                    else:
                        logger.error(f"HTTP {response.status} for {url}")
                        return False
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                return False

    def get_io_stats(self) -> Dict[str, Any]:
        """Get I/O operation statistics."""
        return {
            "max_concurrent_operations": self.max_concurrent_operations,
            "current_semaphore_value": (
                self.semaphore._value if hasattr(self.semaphore, "_value") else 0
            ),
            "session_active": self.session is not None and not self.session.closed,
            "database_pool_active": self.db_pool is not None,
        }


class BatchProcessor:
    """Batch processing system for multiple problems."""

    def __init__(
        self, batch_size: int = 10, max_workers: int = 4, enable_async: bool = True
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_async = enable_async

        self.processing_queue = asyncio.Queue() if enable_async else []
        self.results_queue = asyncio.Queue() if enable_async else []

        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_stats = {
            "batches_processed": 0,
            "total_items": 0,
            "failed_items": 0,
            "avg_batch_time": 0.0,
        }

        logger.info(
            f"Batch processor initialized: batch_size={batch_size}, workers={max_workers}"
        )

    async def add_to_batch(self, item: Any, priority: int = 0):
        """Add item to processing batch."""
        if self.enable_async:
            await self.processing_queue.put((priority, time.time(), item))
        else:
            self.processing_queue.append((priority, time.time(), item))

    async def process_batch(
        self, processor_func: Callable, *args, **kwargs
    ) -> List[Any]:
        """Process a batch of items."""
        batch = []

        # Collect batch items
        if self.enable_async:
            for _ in range(self.batch_size):
                try:
                    priority, timestamp, item = await asyncio.wait_for(
                        self.processing_queue.get(), timeout=1.0
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
        else:
            batch = self.processing_queue[: self.batch_size]
            self.processing_queue = self.processing_queue[self.batch_size :]

        if not batch:
            return []

        start_time = time.time()

        try:
            # Process batch
            if self.enable_async and asyncio.iscoroutinefunction(processor_func):
                results = await processor_func(batch, *args, **kwargs)
            else:
                # Use thread pool for CPU-bound work
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    self.thread_pool, processor_func, batch, *args, **kwargs
                )

            # Update statistics
            batch_time = time.time() - start_time
            self.processing_stats["batches_processed"] += 1
            self.processing_stats["total_items"] += len(batch)

            # Update average batch time
            prev_avg = self.processing_stats["avg_batch_time"]
            batch_count = self.processing_stats["batches_processed"]
            self.processing_stats["avg_batch_time"] = (
                prev_avg * (batch_count - 1) + batch_time
            ) / batch_count

            logger.debug(f"Processed batch of {len(batch)} items in {batch_time:.3f}s")

            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.processing_stats["failed_items"] += len(batch)
            return []

    async def process_continuous(
        self,
        processor_func: Callable,
        stop_condition: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        """Process items continuously in batches."""
        logger.info("Starting continuous batch processing")

        while True:
            # Check stop condition
            if stop_condition and stop_condition():
                logger.info("Stop condition met, ending batch processing")
                break

            try:
                results = await self.process_batch(processor_func, *args, **kwargs)

                if results and self.enable_async:
                    # Store results
                    for result in results:
                        await self.results_queue.put(result)

                # Short pause to prevent busy waiting
                if not results:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in continuous processing: {e}")
                await asyncio.sleep(1.0)  # Longer pause on error

    async def get_results(self, max_results: int = None) -> List[Any]:
        """Get processed results."""
        results = []

        if not self.enable_async:
            return []

        count = 0
        while not self.results_queue.empty() and (
            max_results is None or count < max_results
        ):
            try:
                result = await asyncio.wait_for(self.results_queue.get(), timeout=0.1)
                results.append(result)
                count += 1
            except asyncio.TimeoutError:
                break

        return results

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        queue_size = 0
        if self.enable_async:
            queue_size = self.processing_queue.qsize()
        else:
            queue_size = len(self.processing_queue)

        return {
            **self.processing_stats,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "queue_size": queue_size,
            "enable_async": self.enable_async,
        }

    def shutdown(self):
        """Shutdown batch processor."""
        logger.info("Shutting down batch processor...")
        self.thread_pool.shutdown(wait=True)
        logger.info("Batch processor shutdown completed")


# Global instances
_global_mesh_refinement = None
_global_jax_engine = None
_global_async_io = None
_global_batch_processor = None


def get_mesh_refinement() -> AdaptiveMeshRefinement:
    """Get global adaptive mesh refinement instance."""
    global _global_mesh_refinement
    if _global_mesh_refinement is None:
        _global_mesh_refinement = AdaptiveMeshRefinement()
    return _global_mesh_refinement


def get_jax_engine() -> JAXOptimizationEngine:
    """Get global JAX optimization engine."""
    global _global_jax_engine
    if _global_jax_engine is None:
        _global_jax_engine = JAXOptimizationEngine()
    return _global_jax_engine


def get_async_io() -> AsyncIOManager:
    """Get global async I/O manager."""
    global _global_async_io
    if _global_async_io is None:
        _global_async_io = AsyncIOManager()
    return _global_async_io


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor."""
    global _global_batch_processor
    if _global_batch_processor is None:
        _global_batch_processor = BatchProcessor()
    return _global_batch_processor


def adaptive_refinement(refinement_threshold: float = 0.1):
    """Decorator for adaptive mesh refinement."""

    def decorator(solve_func):
        @wraps(solve_func)
        async def wrapper(mesh_elements, *args, **kwargs):
            refinement = get_mesh_refinement()
            refinement.refinement_threshold = refinement_threshold

            # Initial solve
            solution = await solve_func(mesh_elements, *args, **kwargs)

            # Compute error indicators and refine
            error_indicators = refinement.compute_error_indicators(
                solution, mesh_elements
            )
            refined_elements = refinement.refine_mesh(mesh_elements, error_indicators)

            # Re-solve on refined mesh
            if len(refined_elements) != len(mesh_elements):
                logger.info(
                    f"Mesh refined: {len(mesh_elements)} -> {len(refined_elements)} elements"
                )
                solution = await solve_func(refined_elements, *args, **kwargs)

            return solution, refined_elements

        return wrapper

    return decorator


def jax_optimized(
    static_argnums: Tuple[int, ...] = (), enable_checkpointing: bool = True
):
    """Decorator for JAX optimization."""

    def decorator(objective_func):
        @wraps(objective_func)
        def wrapper(*args, **kwargs):
            engine = get_jax_engine()
            engine.enable_checkpointing = enable_checkpointing

            # Compile and return optimized function
            return engine.compile_objective(objective_func, static_argnums)(
                *args, **kwargs
            )

        return wrapper

    return decorator


def async_batch_process(batch_size: int = 10):
    """Decorator for async batch processing."""

    def decorator(process_func):
        @wraps(process_func)
        async def wrapper(items, *args, **kwargs):
            processor = get_batch_processor()
            processor.batch_size = batch_size

            # Add items to batch
            for item in items:
                await processor.add_to_batch(item)

            # Process batch
            return await processor.process_batch(process_func, *args, **kwargs)

        return wrapper

    return decorator
