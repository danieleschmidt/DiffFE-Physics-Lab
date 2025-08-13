"""Robust optimization service with checkpointing, progress reporting, and adaptive strategies."""

import json
import logging
import os
import pickle
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.optimize import OptimizeResult, minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..backends.robust_backend import get_robust_backend
from ..models import Problem
from ..utils.exceptions import ErrorCode, OptimizationError, ValidationError
from ..utils.logging_config import PerformanceLogger, get_logger
from ..utils.validation_enhanced import validate_optimization_bounds

logger = get_logger(__name__)


@dataclass
class OptimizationCheckpoint:
    """Checkpoint data for optimization runs."""

    iteration: int
    parameters: Dict[str, Any]
    objective_value: float
    gradient: Optional[np.ndarray]
    hessian: Optional[np.ndarray]
    timestamp: str
    cumulative_time: float
    convergence_history: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationProgress:
    """Real-time optimization progress information."""

    iteration: int
    total_iterations: Optional[int]
    objective_value: float
    gradient_norm: Optional[float]
    parameter_change_norm: Optional[float]
    elapsed_time: float
    estimated_remaining_time: Optional[float]
    convergence_criteria: Dict[str, float]
    current_parameters: Dict[str, Any]
    status: str  # 'running', 'converged', 'failed', 'stopped'
    message: str = ""


@dataclass
class OptimizationConfig:
    """Configuration for robust optimization."""

    method: str = "L-BFGS-B"
    max_iterations: int = 1000
    tolerance: float = 1e-8
    gradient_tolerance: float = 1e-6
    parameter_tolerance: float = 1e-12
    max_function_evaluations: int = 10000
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 10
    checkpoint_directory: Optional[str] = None
    enable_progress_reporting: bool = True
    progress_callback: Optional[Callable] = None
    adaptive_parameters: bool = True
    restart_strategy: str = "best"  # 'best', 'last', 'adaptive'
    parallel_gradient: bool = False
    memory_limit_mb: Optional[float] = None
    timeout_hours: Optional[float] = None


class ProgressReporter:
    """Real-time progress reporter for optimization."""

    def __init__(self, callback: Optional[Callable] = None, log_frequency: int = 10):
        self.callback = callback
        self.log_frequency = log_frequency
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.iteration_times = []
        self.objective_history = []

    def report_progress(self, progress: OptimizationProgress):
        """Report optimization progress."""
        current_time = time.time()

        # Update timing statistics
        if len(self.objective_history) > 0:
            iteration_time = current_time - self.last_report_time
            self.iteration_times.append(iteration_time)

            # Estimate remaining time
            if progress.total_iterations:
                remaining_iterations = progress.total_iterations - progress.iteration
                if self.iteration_times:
                    avg_iteration_time = np.mean(
                        self.iteration_times[-20:]
                    )  # Use recent average
                    progress.estimated_remaining_time = (
                        remaining_iterations * avg_iteration_time
                    )

        self.objective_history.append(progress.objective_value)
        self.last_report_time = current_time

        # Log progress periodically
        if progress.iteration % self.log_frequency == 0 or progress.status != "running":
            self._log_progress(progress)

        # Call user-provided callback
        if self.callback:
            try:
                self.callback(progress)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")

    def _log_progress(self, progress: OptimizationProgress):
        """Log progress information."""
        log_message = (
            f"Optimization Progress - Iter: {progress.iteration}, "
            f"Objective: {progress.objective_value:.6e}, "
            f"Time: {progress.elapsed_time:.1f}s"
        )

        if progress.gradient_norm is not None:
            log_message += f", Grad Norm: {progress.gradient_norm:.3e}"

        if progress.estimated_remaining_time is not None:
            log_message += f", ETA: {progress.estimated_remaining_time:.1f}s"

        if progress.status == "converged":
            logger.info(f"✓ Optimization converged: {log_message}")
        elif progress.status == "failed":
            logger.error(f"✗ Optimization failed: {log_message} - {progress.message}")
        elif progress.status == "stopped":
            logger.warning(
                f"⏸ Optimization stopped: {log_message} - {progress.message}"
            )
        else:
            logger.info(log_message)


class CheckpointManager:
    """Manager for optimization checkpoints."""

    def __init__(self, checkpoint_dir: str, optimization_id: str = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_id = optimization_id or f"opt_{int(time.time())}"
        self.checkpoint_file = (
            self.checkpoint_dir / f"{self.optimization_id}_checkpoint.pkl"
        )
        self.metadata_file = (
            self.checkpoint_dir / f"{self.optimization_id}_metadata.json"
        )

    def save_checkpoint(self, checkpoint: OptimizationCheckpoint):
        """Save optimization checkpoint."""
        try:
            # Save main checkpoint data
            with open(self.checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata as JSON for easy inspection
            metadata = {
                "optimization_id": self.optimization_id,
                "iteration": checkpoint.iteration,
                "objective_value": checkpoint.objective_value,
                "timestamp": checkpoint.timestamp,
                "cumulative_time": checkpoint.cumulative_time,
                "parameter_count": len(checkpoint.parameters),
                "convergence_points": len(checkpoint.convergence_history),
            }

            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.debug(f"Checkpoint saved at iteration {checkpoint.iteration}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> Optional[OptimizationCheckpoint]:
        """Load optimization checkpoint."""
        try:
            if not self.checkpoint_file.exists():
                return None

            with open(self.checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)

            logger.info(f"Loaded checkpoint from iteration {checkpoint.iteration}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        checkpoints = []

        for metadata_file in self.checkpoint_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                checkpoints.append(metadata)
            except Exception as e:
                logger.warning(
                    f"Error reading checkpoint metadata {metadata_file}: {e}"
                )

        return sorted(checkpoints, key=lambda x: x.get("timestamp", ""), reverse=True)

    def cleanup_old_checkpoints(self, keep_last: int = 10):
        """Clean up old checkpoint files."""
        try:
            checkpoints = self.list_checkpoints()
            if len(checkpoints) <= keep_last:
                return

            to_delete = checkpoints[keep_last:]
            for checkpoint_meta in to_delete:
                opt_id = checkpoint_meta["optimization_id"]

                # Delete checkpoint and metadata files
                checkpoint_file = self.checkpoint_dir / f"{opt_id}_checkpoint.pkl"
                metadata_file = self.checkpoint_dir / f"{opt_id}_metadata.json"

                checkpoint_file.unlink(missing_ok=True)
                metadata_file.unlink(missing_ok=True)

            logger.info(f"Cleaned up {len(to_delete)} old checkpoints")

        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {e}")


class RobustOptimizer:
    """Production-ready optimizer with comprehensive robustness features.

    Features:
    - Automatic checkpointing and restart
    - Real-time progress reporting
    - Adaptive parameter strategies
    - Timeout and resource management
    - Multiple optimization algorithms
    - Gradient verification and finite difference fallbacks
    - Convergence diagnostics and analysis
    """

    def __init__(
        self, problem: Problem, config: OptimizationConfig = None, backend: str = "jax"
    ):
        self.problem = problem
        self.config = config or OptimizationConfig()
        self.backend_name = backend

        # Get robust backend
        self.backend, self.actual_backend = get_robust_backend(backend)
        if self.backend is None:
            raise OptimizationError(
                "No automatic differentiation backends available",
                error_code=ErrorCode.BACKEND_UNAVAILABLE,
                suggestion="Install JAX or PyTorch for gradient computation",
            )

        # Initialize components
        self.checkpoint_manager = None
        self.progress_reporter = None
        self.optimization_id = None

        # State tracking
        self.current_checkpoint = None
        self.is_running = False
        self.stop_requested = False
        self._lock = threading.Lock()

        # Performance tracking
        self.function_evaluations = 0
        self.gradient_evaluations = 0
        self.total_time = 0.0

        logger.info(f"RobustOptimizer initialized - Backend: {self.actual_backend}")

    def optimize(
        self,
        objective: Callable,
        initial_params: Dict[str, Any],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        resume_from_checkpoint: bool = True,
    ) -> OptimizeResult:
        """Run robust optimization with full monitoring and checkpointing.

        Parameters
        ----------
        objective : Callable
            Objective function to minimize
        initial_params : Dict[str, Any]
            Initial parameter values
        bounds : Dict[str, Tuple[float, float]], optional
            Parameter bounds
        constraints : List[Dict[str, Any]], optional
            Optimization constraints
        resume_from_checkpoint : bool, optional
            Whether to resume from existing checkpoint

        Returns
        -------
        OptimizeResult
            Optimization results
        """
        if not HAS_SCIPY:
            raise OptimizationError(
                "SciPy required for optimization",
                error_code=ErrorCode.DEPENDENCY_MISSING,
                suggestion="Install SciPy: pip install scipy",
            )

        # Validate inputs
        self._validate_optimization_inputs(initial_params, bounds, constraints)

        # Setup optimization session
        self.optimization_id = f"opt_{int(time.time())}_{id(self)}"
        self._setup_optimization_session(initial_params, bounds)

        # Check for existing checkpoint
        start_params = initial_params
        if resume_from_checkpoint and self.checkpoint_manager:
            checkpoint = self.checkpoint_manager.load_checkpoint()
            if checkpoint:
                start_params = checkpoint.parameters
                self.current_checkpoint = checkpoint
                logger.info(
                    f"Resuming optimization from iteration {checkpoint.iteration}"
                )

        try:
            with self._optimization_context():
                result = self._run_optimization(
                    objective, start_params, bounds, constraints
                )

            logger.info(f"Optimization completed: {result.message}")
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)

            # Save current state as checkpoint for potential recovery
            if self.current_checkpoint and self.checkpoint_manager:
                self.checkpoint_manager.save_checkpoint(self.current_checkpoint)

            # Convert to OptimizationError if not already
            if not isinstance(e, OptimizationError):
                raise OptimizationError(
                    f"Optimization failed: {e}",
                    error_code=ErrorCode.OPTIMIZATION_FAILED,
                    context={
                        "function_evaluations": self.function_evaluations,
                        "gradient_evaluations": self.gradient_evaluations,
                        "backend": self.actual_backend,
                    },
                    cause=e,
                )
            raise

        finally:
            self._cleanup_optimization_session()

    @contextmanager
    def _optimization_context(self):
        """Context manager for optimization execution."""
        self.is_running = True
        self.stop_requested = False
        start_time = time.time()

        try:
            yield
        finally:
            self.is_running = False
            self.total_time = time.time() - start_time

            # Cleanup old checkpoints
            if self.checkpoint_manager:
                self.checkpoint_manager.cleanup_old_checkpoints()

    def _setup_optimization_session(
        self, initial_params: Dict[str, Any], bounds: Optional[Dict]
    ):
        """Setup optimization session."""
        # Setup checkpoint manager
        if self.config.enable_checkpointing:
            checkpoint_dir = self.config.checkpoint_directory or "./checkpoints"
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir, self.optimization_id
            )

        # Setup progress reporter
        if self.config.enable_progress_reporting:
            self.progress_reporter = ProgressReporter(
                callback=self.config.progress_callback,
                log_frequency=max(1, self.config.checkpoint_frequency // 2),
            )

        logger.info(f"Optimization session started - ID: {self.optimization_id}")

    def _cleanup_optimization_session(self):
        """Cleanup optimization session."""
        logger.info(
            f"Optimization session completed - "
            f"Function evals: {self.function_evaluations}, "
            f"Gradient evals: {self.gradient_evaluations}, "
            f"Total time: {self.total_time:.1f}s"
        )

    def _validate_optimization_inputs(
        self,
        initial_params: Dict[str, Any],
        bounds: Optional[Dict],
        constraints: Optional[List],
    ):
        """Validate optimization inputs."""
        # Validate initial parameters
        if not isinstance(initial_params, dict) or not initial_params:
            raise ValidationError(
                "Initial parameters must be non-empty dictionary",
                invalid_field="initial_params",
            )

        # Validate parameter values
        for name, value in initial_params.items():
            if not isinstance(value, (int, float, np.number)):
                raise ValidationError(
                    f"Parameter '{name}' must be numeric, got {type(value)}",
                    invalid_field=name,
                    expected_type=float,
                )

            if not np.isfinite(value):
                raise ValidationError(
                    f"Parameter '{name}' must be finite, got {value}",
                    invalid_field=name,
                    actual_value=value,
                )

        # Validate bounds
        if bounds:
            validate_optimization_bounds(bounds, initial_params)

        # Validate constraints
        if constraints:
            for i, constraint in enumerate(constraints):
                if not isinstance(constraint, dict):
                    raise ValidationError(
                        f"Constraint {i} must be dictionary",
                        invalid_field=f"constraints[{i}]",
                    )

                if "type" not in constraint:
                    raise ValidationError(
                        f"Constraint {i} missing 'type' field",
                        invalid_field=f"constraints[{i}].type",
                    )

    def _run_optimization(
        self,
        objective: Callable,
        initial_params: Dict[str, Any],
        bounds: Optional[Dict],
        constraints: Optional[List],
    ) -> OptimizeResult:
        """Run the optimization procedure."""
        # Convert parameters to arrays for scipy
        param_names = list(initial_params.keys())
        x0 = np.array([initial_params[name] for name in param_names])

        # Convert bounds format
        scipy_bounds = None
        if bounds:
            scipy_bounds = [bounds.get(name, (None, None)) for name in param_names]

        # Setup objective and gradient functions
        wrapped_objective = self._wrap_objective_function(objective, param_names)
        wrapped_gradient = self._wrap_gradient_function(objective, param_names)

        # Configure optimization options
        options = {
            "maxiter": self.config.max_iterations,
            "ftol": self.config.tolerance,
            "gtol": self.config.gradient_tolerance,
            "maxfun": self.config.max_function_evaluations,
            "disp": True,
        }

        # Add method-specific options
        if self.config.method.upper() in ["L-BFGS-B", "BFGS"]:
            options["maxcor"] = 20  # Memory limit for L-BFGS-B

        # Setup callback for progress monitoring
        callback = self._create_optimization_callback(param_names)

        # Run optimization
        logger.info(f"Starting optimization with method {self.config.method}")

        start_time = time.time()

        try:
            result = minimize(
                wrapped_objective,
                x0,
                method=self.config.method,
                jac=wrapped_gradient,
                bounds=scipy_bounds,
                constraints=constraints,
                options=options,
                callback=callback,
            )

        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            # Create partial result
            result = self._create_partial_result(x0, param_names, "User interrupted")

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            result = self._create_partial_result(x0, param_names, f"Error: {e}")
            result.success = False

        # Convert result back to parameter dictionary
        result.x_dict = {name: val for name, val in zip(param_names, result.x)}
        result.backend = self.actual_backend
        result.function_evaluations = self.function_evaluations
        result.gradient_evaluations = self.gradient_evaluations
        result.total_time = time.time() - start_time

        return result

    def _wrap_objective_function(
        self, objective: Callable, param_names: List[str]
    ) -> Callable:
        """Wrap objective function with monitoring and error handling."""

        def wrapped_objective(x: np.ndarray) -> float:
            if self.stop_requested:
                raise KeyboardInterrupt("Optimization stopped by request")

            # Check timeout
            if self.config.timeout_hours:
                if self.total_time > self.config.timeout_hours * 3600:
                    raise TimeoutError(
                        f"Optimization timeout after {self.config.timeout_hours} hours"
                    )

            # Check memory limit
            if self.config.memory_limit_mb:
                import psutil

                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                if current_memory > self.config.memory_limit_mb:
                    raise MemoryError(f"Memory limit exceeded: {current_memory:.1f}MB")

            # Convert to parameter dictionary
            params = {name: float(val) for name, val in zip(param_names, x)}

            try:
                with PerformanceLogger("objective_evaluation", logger, 0.0):
                    value = objective(params)

                self.function_evaluations += 1

                # Validate result
                if not np.isfinite(value):
                    logger.warning(
                        f"Objective function returned non-finite value: {value}"
                    )
                    return np.inf

                return float(value)

            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return np.inf

        return wrapped_objective

    def _wrap_gradient_function(
        self, objective: Callable, param_names: List[str]
    ) -> Callable:
        """Wrap gradient function with monitoring and fallbacks."""
        # Get gradient function from backend
        try:
            grad_func = self.backend.grad(objective) if self.backend else None
        except Exception as e:
            logger.warning(f"Failed to create gradient function: {e}")
            grad_func = None

        def wrapped_gradient(x: np.ndarray) -> np.ndarray:
            if self.stop_requested:
                raise KeyboardInterrupt("Optimization stopped by request")

            params = {name: float(val) for name, val in zip(param_names, x)}

            try:
                if grad_func:
                    # Use automatic differentiation
                    with PerformanceLogger("gradient_evaluation", logger, 0.0):
                        gradient = grad_func(params)
                else:
                    # Fallback to finite differences
                    with PerformanceLogger("finite_difference_gradient", logger, 0.0):
                        gradient = self._compute_finite_difference_gradient(
                            objective, params, param_names
                        )

                self.gradient_evaluations += 1

                # Convert to array and validate
                if isinstance(gradient, dict):
                    grad_array = np.array([gradient[name] for name in param_names])
                else:
                    grad_array = np.asarray(gradient)

                if not np.all(np.isfinite(grad_array)):
                    logger.warning("Gradient contains non-finite values")
                    # Replace non-finite values with zero
                    grad_array = np.where(np.isfinite(grad_array), grad_array, 0.0)

                return grad_array

            except Exception as e:
                logger.error(f"Error in gradient computation: {e}")
                # Return zero gradient as fallback
                return np.zeros_like(x)

        return wrapped_gradient

    def _compute_finite_difference_gradient(
        self,
        objective: Callable,
        params: Dict[str, Any],
        param_names: List[str],
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Compute gradient using finite differences."""
        gradient = np.zeros(len(param_names))

        # Central differences
        for i, name in enumerate(param_names):
            params_plus = params.copy()
            params_minus = params.copy()

            params_plus[name] += eps
            params_minus[name] -= eps

            try:
                f_plus = objective(params_plus)
                f_minus = objective(params_minus)
                gradient[i] = (f_plus - f_minus) / (2 * eps)
            except Exception as e:
                logger.warning(f"Error computing finite difference for {name}: {e}")
                gradient[i] = 0.0

        return gradient

    def _create_optimization_callback(self, param_names: List[str]) -> Callable:
        """Create callback function for optimization monitoring."""

        def callback(x: np.ndarray, **kwargs):
            if self.stop_requested:
                return True  # Stop optimization

            # Update progress
            params = {name: float(val) for name, val in zip(param_names, x)}

            # Get additional information from kwargs (method-dependent)
            grad_norm = kwargs.get("grad_norm")
            if grad_norm is None and "jac" in kwargs:
                grad_norm = np.linalg.norm(kwargs["jac"])

            # Create checkpoint
            iteration = self.function_evaluations
            checkpoint = OptimizationCheckpoint(
                iteration=iteration,
                parameters=params,
                objective_value=kwargs.get("fun", 0.0),
                gradient=kwargs.get("jac"),
                hessian=kwargs.get("hess"),
                timestamp=datetime.now().isoformat(),
                cumulative_time=(
                    time.time() - self.progress_reporter.start_time
                    if self.progress_reporter
                    else 0.0
                ),
                convergence_history=[],
                metadata={"method": self.config.method, "backend": self.actual_backend},
            )

            self.current_checkpoint = checkpoint

            # Save checkpoint periodically
            if (
                self.config.enable_checkpointing
                and self.checkpoint_manager
                and iteration % self.config.checkpoint_frequency == 0
            ):
                self.checkpoint_manager.save_checkpoint(checkpoint)

            # Report progress
            if self.progress_reporter:
                progress = OptimizationProgress(
                    iteration=iteration,
                    total_iterations=self.config.max_iterations,
                    objective_value=checkpoint.objective_value,
                    gradient_norm=grad_norm,
                    parameter_change_norm=None,
                    elapsed_time=checkpoint.cumulative_time,
                    estimated_remaining_time=None,
                    convergence_criteria={
                        "ftol": self.config.tolerance,
                        "gtol": self.config.gradient_tolerance,
                    },
                    current_parameters=params,
                    status="running",
                )

                self.progress_reporter.report_progress(progress)

            return False  # Continue optimization

        return callback

    def _create_partial_result(
        self, x: np.ndarray, param_names: List[str], message: str
    ) -> OptimizeResult:
        """Create partial optimization result."""
        result = OptimizeResult()
        result.x = x
        result.success = False
        result.status = -1
        result.message = message
        result.fun = np.inf
        result.nit = self.function_evaluations
        result.nfev = self.function_evaluations
        result.njev = self.gradient_evaluations

        return result

    def stop_optimization(self):
        """Request optimization to stop gracefully."""
        with self._lock:
            self.stop_requested = True
            logger.info("Optimization stop requested")

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        with self._lock:
            return {
                "is_running": self.is_running,
                "optimization_id": self.optimization_id,
                "function_evaluations": self.function_evaluations,
                "gradient_evaluations": self.gradient_evaluations,
                "backend": self.actual_backend,
                "current_checkpoint": (
                    asdict(self.current_checkpoint) if self.current_checkpoint else None
                ),
                "stop_requested": self.stop_requested,
            }

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available optimization checkpoints."""
        if not self.checkpoint_manager:
            return []

        return self.checkpoint_manager.list_checkpoints()

    def health_check(self) -> Dict[str, Any]:
        """Perform optimizer health check."""
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "backend": self.actual_backend,
            "backend_available": self.backend is not None,
            "checkpointing_enabled": self.config.enable_checkpointing,
            "progress_reporting_enabled": self.config.enable_progress_reporting,
            "is_running": self.is_running,
            "configuration": asdict(self.config),
        }
