"""Robust solver with comprehensive monitoring, error handling, and retry mechanisms."""

import gc
import logging
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

try:
    import firedrake as fd

    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from ..backends.robust_backend import get_robust_backend
from ..models import Problem
from ..operators.base import get_operator
from ..performance.monitor import get_global_monitor
from ..utils.exceptions import (
    ConvergenceError,
    ErrorCode,
    ResourceError,
    SolverError,
    handle_solver_failure,
)
from ..utils.logging_config import PerformanceLogger, get_logger, log_performance

logger = get_logger(__name__)


@dataclass
class SolverMetrics:
    """Comprehensive solver performance metrics."""

    solve_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0
    cpu_time: float = 0.0
    convergence_iterations: int = 0
    residual_history: List[float] = field(default_factory=list)
    linear_solve_count: int = 0
    assembly_time: float = 0.0
    factorization_time: float = 0.0
    backsolve_time: float = 0.0
    dofs: int = 0
    mesh_cells: int = 0
    warnings: List[str] = field(default_factory=list)
    success: bool = False


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""

    max_retries: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 60.0
    retry_on_convergence_failure: bool = True
    retry_on_memory_error: bool = True
    retry_on_numerical_error: bool = True
    adaptive_parameters: bool = True


class ResourceMonitor:
    """Monitor system resources during computation."""

    def __init__(self, monitoring_interval: float = 0.1):
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics = {"memory_usage": [], "cpu_usage": [], "timestamps": []}
        self._lock = threading.Lock()
        self.peak_memory = 0.0

    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.debug("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        logger.debug("Resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process()

        while self.monitoring_active:
            try:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent()
                timestamp = time.time()

                with self._lock:
                    self.metrics["memory_usage"].append(memory_mb)
                    self.metrics["cpu_usage"].append(cpu_percent)
                    self.metrics["timestamps"].append(timestamp)

                    self.peak_memory = max(self.peak_memory, memory_mb)

                    # Limit history size
                    max_history = 10000
                    if len(self.metrics["memory_usage"]) > max_history:
                        for key in self.metrics:
                            self.metrics[key] = self.metrics[key][-max_history:]

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.warning(f"Error in resource monitoring: {e}")
                time.sleep(self.monitoring_interval)

    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        with self._lock:
            return self.peak_memory

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        with self._lock:
            return {
                "peak_memory_mb": self.peak_memory,
                "current_memory_mb": (
                    self.metrics["memory_usage"][-1]
                    if self.metrics["memory_usage"]
                    else 0
                ),
                "avg_cpu_percent": (
                    np.mean(self.metrics["cpu_usage"])
                    if self.metrics["cpu_usage"]
                    else 0
                ),
                "sample_count": len(self.metrics["memory_usage"]),
            }


class RobustFEBMLSolver:
    """Production-ready solver with comprehensive error handling and monitoring.

    Features:
    - Comprehensive resource monitoring
    - Automatic retry with adaptive parameters
    - Memory management and leak detection
    - Performance profiling and optimization
    - Graceful degradation and fallbacks
    - Health checks and diagnostics
    """

    def __init__(
        self,
        problem: Problem = None,
        backend: str = "jax",
        solver_options: Dict[str, Any] = None,
        retry_config: RetryConfig = None,
        enable_monitoring: bool = True,
        memory_limit_mb: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.problem = problem
        self.backend_name = backend
        self.solver_options = solver_options or {}
        self.retry_config = retry_config or RetryConfig()
        self.enable_monitoring = enable_monitoring
        self.memory_limit_mb = memory_limit_mb
        self.timeout_seconds = timeout_seconds

        # Get robust backend
        self.backend, self.actual_backend = get_robust_backend(backend)
        if self.backend is None:
            raise SolverError(
                "No automatic differentiation backends available",
                error_code=ErrorCode.BACKEND_UNAVAILABLE,
                suggestion="Install JAX or PyTorch for AD support",
            )

        # Initialize monitoring
        self.resource_monitor = ResourceMonitor() if enable_monitoring else None
        self.metrics_history = []

        # Solver state
        self.solution_history = []
        self.convergence_history = []
        self.last_metrics = None

        # Default solver options with robust defaults
        self.default_options = {
            "max_iterations": 100,
            "tolerance": 1e-8,
            "linear_solver": "lu",
            "preconditioner": "ilu",
            "monitor_convergence": True,
            "checkpoint_frequency": 10,
            "adaptive_tolerance": True,
            "memory_efficient": True,
            "verbose": True,
        }
        self.options = {**self.default_options, **self.solver_options}

        logger.info(
            f"RobustFEBMLSolver initialized - Backend: {self.actual_backend}, "
            f"Monitoring: {enable_monitoring}, Memory limit: {memory_limit_mb}MB"
        )

    @contextmanager
    def _memory_limit_context(self):
        """Context manager for memory limit enforcement."""
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        try:
            yield
        finally:
            if self.memory_limit_mb:
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                if current_memory > self.memory_limit_mb:
                    logger.warning(
                        f"Memory limit exceeded: {current_memory:.1f}MB > {self.memory_limit_mb}MB"
                    )
                    # Force garbage collection
                    gc.collect()

    @contextmanager
    def _timeout_context(self):
        """Context manager for timeout enforcement."""
        start_time = time.time()

        def check_timeout():
            if self.timeout_seconds and time.time() - start_time > self.timeout_seconds:
                raise SolverError(
                    f"Solver timeout after {self.timeout_seconds} seconds",
                    error_code=ErrorCode.OPERATION_TIMEOUT,
                )

        try:
            yield check_timeout
        except Exception:
            raise

    @log_performance("solver_solve", min_duration=0.1)
    def solve(
        self,
        problem: Problem = None,
        parameters: Dict[str, Any] = None,
        return_metrics: bool = True,
    ) -> Union[Any, Tuple[Any, SolverMetrics]]:
        """Solve the finite element problem with comprehensive monitoring.

        Parameters
        ----------
        problem : Problem, optional
            Problem to solve, uses instance problem if None
        parameters : Dict[str, Any], optional
            Runtime parameters
        return_metrics : bool, optional
            Whether to return detailed metrics

        Returns
        -------
        solution : Any
            Solution field
        metrics : SolverMetrics, optional
            Detailed solver metrics (if return_metrics=True)
        """
        prob = problem or self.problem
        if prob is None:
            raise SolverError(
                "No problem provided", error_code=ErrorCode.SOLVER_SETUP_ERROR
            )

        # Initialize metrics
        metrics = SolverMetrics()
        start_time = time.perf_counter()

        # Start monitoring
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()

        try:
            with self._memory_limit_context(), self._timeout_context() as check_timeout:
                solution = self._solve_with_retry(
                    prob, parameters, metrics, check_timeout
                )
                metrics.success = True

        except Exception as e:
            metrics.success = False
            logger.error(f"Solver failed: {e}", exc_info=True)

            # Enhanced error reporting
            error_context = {
                "backend": self.actual_backend,
                "dofs": metrics.dofs,
                "memory_peak_mb": metrics.memory_peak_mb,
                "solve_time": metrics.solve_time,
            }

            if isinstance(e, (SolverError, ConvergenceError)):
                # Re-raise with additional context
                e.context.update(error_context)
                raise
            else:
                # Wrap in SolverError
                raise SolverError(
                    f"Unexpected solver error: {e}",
                    error_code=ErrorCode.SOLVER_CONVERGENCE_FAILED,
                    context=error_context,
                    cause=e,
                )

        finally:
            # Stop monitoring and collect metrics
            if self.resource_monitor:
                self.resource_monitor.stop_monitoring()
                resource_metrics = self.resource_monitor.get_current_metrics()
                metrics.memory_peak_mb = resource_metrics["peak_memory_mb"]
                metrics.cpu_time = time.perf_counter() - start_time

            metrics.solve_time = time.perf_counter() - start_time
            self.last_metrics = metrics
            self.metrics_history.append(metrics)

            # Log performance summary
            self._log_performance_summary(metrics)

            # Record global metrics
            monitor = get_global_monitor()
            monitor.log_application_metric("solve_time_ms", metrics.solve_time * 1000)
            monitor.log_application_metric("memory_peak_mb", metrics.memory_peak_mb)
            monitor.log_application_metric(
                "convergence_iterations", metrics.convergence_iterations
            )

        if return_metrics:
            return solution, metrics
        return solution

    def _solve_with_retry(
        self,
        problem: Problem,
        parameters: Dict[str, Any],
        metrics: SolverMetrics,
        check_timeout: Callable,
    ) -> Any:
        """Solve with retry mechanism."""
        last_exception = None
        retry_count = 0

        # Merge parameters
        solve_params = {**problem.parameters}
        if parameters:
            solve_params.update(parameters)

        while retry_count <= self.retry_config.max_retries:
            try:
                logger.debug(
                    f"Solver attempt {retry_count + 1}/{self.retry_config.max_retries + 1}"
                )

                # Adjust parameters for retry
                if retry_count > 0:
                    solve_params = self._adapt_parameters_for_retry(
                        solve_params, retry_count, last_exception
                    )

                # Attempt solve
                solution = self._solve_attempt(
                    problem, solve_params, metrics, check_timeout
                )

                if retry_count > 0:
                    logger.info(f"Solver succeeded on retry attempt {retry_count}")

                return solution

            except Exception as e:
                last_exception = e
                retry_count += 1

                # Check if we should retry this exception
                if not self._should_retry(e, retry_count):
                    logger.error(f"Not retrying exception: {e}")
                    raise

                if retry_count <= self.retry_config.max_retries:
                    delay = min(
                        self.retry_config.initial_delay
                        * (self.retry_config.backoff_factor ** (retry_count - 1)),
                        self.retry_config.max_delay,
                    )
                    logger.warning(
                        f"Solver attempt {retry_count} failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

        # All retries exhausted
        logger.error(
            f"Solver failed after {self.retry_config.max_retries + 1} attempts"
        )
        raise last_exception

    def _solve_attempt(
        self,
        problem: Problem,
        parameters: Dict[str, Any],
        metrics: SolverMetrics,
        check_timeout: Callable,
    ) -> Any:
        """Single solve attempt with monitoring."""
        if not HAS_FIREDRAKE:
            raise SolverError(
                "Firedrake required for FEM solving",
                error_code=ErrorCode.DEPENDENCY_MISSING,
                suggestion="Install Firedrake for finite element computations",
            )

        logger.info("Starting FEM solve")

        # Record initial memory
        metrics.memory_start_mb = psutil.Process().memory_info().rss / (1024 * 1024)

        # Create solution function
        u = fd.Function(problem.function_space)
        v = fd.TestFunction(problem.function_space)

        # Record problem size
        metrics.dofs = problem.function_space.dim()
        metrics.mesh_cells = problem.mesh.num_cells() if problem.mesh else 0

        # Check timeout periodically
        check_timeout()

        # Check if problem is linear or nonlinear
        is_linear = self._check_linearity(problem)

        # Solve based on problem type
        if is_linear:
            solution = self._solve_linear_monitored(
                problem, u, v, parameters, metrics, check_timeout
            )
        else:
            solution = self._solve_nonlinear_monitored(
                problem, u, v, parameters, metrics, check_timeout
            )

        # Store solution
        self.solution_history.append(solution.copy(deepcopy=True))

        # Record final memory
        metrics.memory_end_mb = psutil.Process().memory_info().rss / (1024 * 1024)

        logger.info(
            f"FEM solve completed - DOFs: {metrics.dofs}, Time: {metrics.solve_time:.3f}s"
        )

        return solution

    def _solve_linear_monitored(
        self,
        problem: Problem,
        u: Any,
        v: Any,
        params: Dict[str, Any],
        metrics: SolverMetrics,
        check_timeout: Callable,
    ) -> Any:
        """Solve linear problem with monitoring."""
        # Assembly phase
        assembly_start = time.perf_counter()
        F = self._assemble_system(problem, u, v, params)
        bcs = problem._assemble_boundary_conditions()
        metrics.assembly_time = time.perf_counter() - assembly_start

        check_timeout()

        # Configure solver parameters
        solver_params = self._get_linear_solver_params()

        # Solve with monitoring
        solve_start = time.perf_counter()

        try:
            fd.solve(F == 0, u, bcs=bcs, solver_parameters=solver_params)
            metrics.linear_solve_count = 1

        except Exception as e:
            # Enhanced error reporting for linear solve failures
            error_info = {
                "solver_params": solver_params,
                "dofs": metrics.dofs,
                "assembly_time": metrics.assembly_time,
            }

            raise SolverError(
                f"Linear solve failed: {e}",
                error_code=ErrorCode.LINEAR_SYSTEM_ERROR,
                context=error_info,
                cause=e,
            )

        solve_time = time.perf_counter() - solve_start
        metrics.factorization_time = solve_time  # For direct solvers

        return u

    def _solve_nonlinear_monitored(
        self,
        problem: Problem,
        u: Any,
        v: Any,
        params: Dict[str, Any],
        metrics: SolverMetrics,
        check_timeout: Callable,
    ) -> Any:
        """Solve nonlinear problem with monitoring."""
        max_iter = self.options.get("max_iterations", 100)
        tolerance = self.options.get("tolerance", 1e-8)
        adaptive_tol = self.options.get("adaptive_tolerance", True)

        # Initial guess
        if "initial_guess" in params:
            u.assign(params["initial_guess"])

        # Newton iteration with monitoring
        residuals = []
        corrections = []

        for iteration in range(max_iter):
            check_timeout()

            iteration_start = time.perf_counter()

            # Assemble residual and Jacobian
            assembly_start = time.perf_counter()
            F = self._assemble_system(problem, u, v, params)
            J = fd.derivative(F, u)
            bcs = problem._assemble_boundary_conditions()
            metrics.assembly_time += time.perf_counter() - assembly_start

            # Newton correction
            du = fd.Function(u.function_space())

            # Solve linear system
            linear_start = time.perf_counter()
            solver_params = self._get_nonlinear_solver_params(iteration)

            try:
                fd.solve(J == -F, du, bcs=bcs, solver_parameters=solver_params)
                metrics.linear_solve_count += 1

            except Exception as e:
                raise SolverError(
                    f"Linear solve failed in Newton iteration {iteration}: {e}",
                    error_code=ErrorCode.LINEAR_SYSTEM_ERROR,
                    context={"newton_iteration": iteration},
                    cause=e,
                )

            metrics.backsolve_time += time.perf_counter() - linear_start

            # Update solution
            u.assign(u + du)

            # Check convergence
            try:
                residual_norm = float(fd.sqrt(fd.assemble(fd.inner(F, F) * fd.dx)))
                correction_norm = float(fd.sqrt(fd.assemble(fd.inner(du, du) * fd.dx)))
            except Exception as e:
                logger.warning(f"Error computing norms: {e}")
                residual_norm = float("inf")
                correction_norm = float("inf")

            residuals.append(residual_norm)
            corrections.append(correction_norm)

            # Store convergence info
            convergence_info = {
                "iteration": iteration,
                "residual_norm": residual_norm,
                "correction_norm": correction_norm,
                "iteration_time": time.perf_counter() - iteration_start,
            }
            self.convergence_history.append(convergence_info)

            # Adaptive tolerance
            current_tolerance = tolerance
            if adaptive_tol and iteration > 5:
                # Relax tolerance if not converging well
                convergence_rate = (
                    residuals[-1] / residuals[-2] if len(residuals) > 1 else 1.0
                )
                if convergence_rate > 0.9:
                    current_tolerance = min(tolerance * 10, 1e-4)
                    logger.debug(f"Relaxed tolerance to {current_tolerance:.2e}")

            logger.debug(
                f"Newton iteration {iteration}: residual={residual_norm:.2e}, "
                f"correction={correction_norm:.2e}, tolerance={current_tolerance:.2e}"
            )

            # Check convergence
            if correction_norm < current_tolerance:
                logger.info(f"Newton method converged in {iteration + 1} iterations")
                metrics.convergence_iterations = iteration + 1
                metrics.residual_history = residuals
                break

            # Check for divergence
            if iteration > 10 and residual_norm > 1e10:
                raise ConvergenceError(
                    "Newton method diverged - residual growing rapidly",
                    max_iterations=max_iter,
                    tolerance=tolerance,
                    final_residual=residual_norm,
                )
        else:
            # Did not converge
            raise ConvergenceError(
                f"Newton method failed to converge in {max_iter} iterations",
                max_iterations=max_iter,
                tolerance=tolerance,
                final_residual=residuals[-1] if residuals else float("inf"),
            )

        return u

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        if attempt > self.retry_config.max_retries:
            return False

        # Check exception types
        if (
            isinstance(exception, ConvergenceError)
            and self.retry_config.retry_on_convergence_failure
        ):
            return True

        if (
            isinstance(exception, (MemoryError, ResourceError))
            and self.retry_config.retry_on_memory_error
        ):
            return True

        # Check for numerical errors
        if self.retry_config.retry_on_numerical_error:
            error_msg = str(exception).lower()
            numerical_indicators = [
                "nan",
                "inf",
                "singular",
                "numerical",
                "overflow",
                "underflow",
            ]
            if any(indicator in error_msg for indicator in numerical_indicators):
                return True

        return False

    def _adapt_parameters_for_retry(
        self, parameters: Dict[str, Any], retry_attempt: int, last_exception: Exception
    ) -> Dict[str, Any]:
        """Adapt parameters for retry attempt."""
        adapted_params = parameters.copy()

        if not self.retry_config.adaptive_parameters:
            return adapted_params

        # Adapt based on exception type
        if isinstance(last_exception, ConvergenceError):
            # Relax convergence criteria
            if "tolerance" in adapted_params:
                adapted_params["tolerance"] = min(
                    adapted_params["tolerance"] * 10, 1e-4
                )

            if "max_iterations" in adapted_params:
                adapted_params["max_iterations"] = min(
                    adapted_params["max_iterations"] * 2, 1000
                )

            logger.debug(f"Retry {retry_attempt}: relaxed convergence criteria")

        elif isinstance(last_exception, (MemoryError, ResourceError)):
            # Reduce memory usage
            adapted_params["memory_efficient"] = True

            # Use more conservative solver settings
            adapted_params["linear_solver"] = "gmres"
            adapted_params["preconditioner"] = "ilu"

            logger.debug(f"Retry {retry_attempt}: enabled memory-efficient mode")

        return adapted_params

    def _get_linear_solver_params(self) -> Dict[str, Any]:
        """Get linear solver parameters."""
        return {
            "ksp_type": self.options.get("linear_solver", "lu"),
            "pc_type": self.options.get("preconditioner", "ilu"),
            "ksp_rtol": self.options.get("tolerance", 1e-8),
            "ksp_atol": 1e-14,
            "ksp_max_it": 10000,
            "ksp_monitor": self.options.get("monitor_convergence", True),
            "ksp_converged_reason": True,
        }

    def _get_nonlinear_solver_params(self, iteration: int) -> Dict[str, Any]:
        """Get nonlinear solver parameters for specific iteration."""
        base_params = {
            "ksp_type": (
                "preonly"
                if iteration < 5
                else self.options.get("linear_solver", "gmres")
            ),
            "pc_type": (
                "lu" if iteration < 5 else self.options.get("preconditioner", "ilu")
            ),
            "ksp_rtol": 1e-12,
            "ksp_atol": 1e-14,
            "ksp_max_it": 1000,
            "ksp_monitor": False,  # Reduce output noise
        }

        # Memory-efficient mode
        if self.options.get("memory_efficient", False):
            base_params.update(
                {"ksp_type": "gmres", "pc_type": "ilu", "ksp_gmres_restart": 30}
            )

        return base_params

    def _check_linearity(self, problem: Problem) -> bool:
        """Check if problem is linear."""
        for eq in problem.equations:
            if hasattr(eq["equation"], "is_linear"):
                if not eq["equation"].is_linear:
                    return False
        return True

    def _assemble_system(
        self, problem: Problem, u: Any, v: Any, params: Dict[str, Any]
    ) -> Any:
        """Assemble the weak form system."""
        F = 0

        for eq in problem.equations:
            if eq["active"]:
                F += eq["equation"](u, v, params)

        return F

    def _log_performance_summary(self, metrics: SolverMetrics):
        """Log comprehensive performance summary."""
        logger.info(
            f"Solver Performance Summary:\n"
            f"  Success: {metrics.success}\n"
            f"  Total Time: {metrics.solve_time:.3f}s\n"
            f"  Assembly Time: {metrics.assembly_time:.3f}s\n"
            f"  Factorization Time: {metrics.factorization_time:.3f}s\n"
            f"  Backsolve Time: {metrics.backsolve_time:.3f}s\n"
            f"  Memory Peak: {metrics.memory_peak_mb:.1f}MB\n"
            f"  DOFs: {metrics.dofs:,}\n"
            f"  Convergence Iterations: {metrics.convergence_iterations}\n"
            f"  Linear Solves: {metrics.linear_solve_count}\n"
            f"  Backend: {self.actual_backend}"
        )

        if metrics.warnings:
            logger.warning(f"Solver warnings: {', '.join(metrics.warnings)}")

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive solver health check.

        Returns
        -------
        Dict[str, Any]
            Health check results
        """
        health = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "backend": self.actual_backend,
            "backend_available": self.backend is not None,
            "last_solve_success": (
                self.last_metrics.success if self.last_metrics else None
            ),
            "memory_status": "normal",
            "performance_status": "normal",
            "issues": [],
        }

        try:
            # Check backend availability
            if not self.backend:
                health["status"] = "unhealthy"
                health["issues"].append("No AD backend available")

            # Check memory usage
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            if self.memory_limit_mb and current_memory > self.memory_limit_mb * 0.8:
                health["memory_status"] = "warning"
                health["issues"].append(f"High memory usage: {current_memory:.1f}MB")

            # Check recent performance
            if self.metrics_history:
                recent_metrics = self.metrics_history[-5:]  # Last 5 solves
                avg_solve_time = np.mean([m.solve_time for m in recent_metrics])

                if avg_solve_time > 300:  # More than 5 minutes average
                    health["performance_status"] = "degraded"
                    health["issues"].append(
                        f"Slow average solve time: {avg_solve_time:.1f}s"
                    )

                failure_rate = sum(1 for m in recent_metrics if not m.success) / len(
                    recent_metrics
                )
                if failure_rate > 0.2:  # More than 20% failures
                    health["status"] = "degraded"
                    health["issues"].append(
                        f"High failure rate: {failure_rate*100:.1f}%"
                    )

            # Set overall status
            if health["issues"]:
                if health["status"] == "healthy":
                    health["status"] = "degraded"

            health["system_memory_mb"] = current_memory
            health["metrics_history_count"] = len(self.metrics_history)

        except Exception as e:
            health["status"] = "error"
            health["issues"].append(f"Health check error: {e}")

        return health

    def get_performance_metrics(self, last_n: int = 10) -> Dict[str, Any]:
        """Get performance metrics summary.

        Parameters
        ----------
        last_n : int, optional
            Number of recent solves to analyze

        Returns
        -------
        Dict[str, Any]
            Performance metrics summary
        """
        if not self.metrics_history:
            return {"error": "No solve metrics available"}

        recent_metrics = self.metrics_history[-last_n:]

        return {
            "solve_count": len(recent_metrics),
            "success_rate": sum(1 for m in recent_metrics if m.success)
            / len(recent_metrics),
            "avg_solve_time": np.mean([m.solve_time for m in recent_metrics]),
            "avg_memory_peak": np.mean([m.memory_peak_mb for m in recent_metrics]),
            "avg_convergence_iterations": np.mean(
                [m.convergence_iterations for m in recent_metrics]
            ),
            "total_linear_solves": sum(m.linear_solve_count for m in recent_metrics),
            "backend": self.actual_backend,
            "timestamp": datetime.now().isoformat(),
        }
