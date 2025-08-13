"""Optimization service for parameter estimation and design optimization."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from scipy.optimize import OptimizeResult, differential_evolution, minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..backends import get_backend
from ..models import Problem

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Supported optimization methods."""

    LBFGS = "lbfgs"
    BFGS = "bfgs"
    CG = "cg"
    NELDER_MEAD = "nelder-mead"
    DIFFERENTIAL_EVOLUTION = "differential-evolution"
    ADAM = "adam"
    SGD = "sgd"


@dataclass
class OptimizationResult:
    """Result of optimization process."""

    success: bool
    optimal_parameters: Dict[str, Any]
    optimal_value: float
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    message: str
    convergence_history: List[Dict[str, Any]]
    timing_info: Dict[str, float]


class OptimizationService:
    """Service for parameter optimization and inverse problems.

    Provides comprehensive optimization capabilities including gradient-based
    methods, derivative-free optimization, and multi-objective optimization.

    Parameters
    ----------
    problem : Problem, optional
        Associated FEM problem
    backend : str, optional
        AD backend for gradient computation, by default 'jax'

    Examples
    --------
    >>> service = OptimizationService(problem)
    >>> result = service.minimize_scalar(objective_func, bounds=(0, 10))
    >>> result = service.minimize_vector(objective_func, initial_guess)
    """

    def __init__(self, problem: Problem = None, backend: str = "jax"):
        self.problem = problem
        self.backend = get_backend(backend)

        # Optimization state
        self.history = []
        self.current_iteration = 0
        self.best_result = None

        # Default options
        self.default_options = {
            "maxiter": 1000,
            "ftol": 1e-9,
            "gtol": 1e-6,
            "disp": True,
            "return_all": False,
        }

        logger.info(f"OptimizationService initialized with backend: {backend}")

    def minimize_scalar(
        self,
        objective: Callable[[float], float],
        bounds: Tuple[float, float],
        method: str = "brent",
        options: Dict[str, Any] = None,
    ) -> OptimizationResult:
        """Minimize scalar objective function.

        Parameters
        ----------
        objective : Callable[[float], float]
            Objective function to minimize
        bounds : Tuple[float, float]
            Lower and upper bounds
        method : str, optional
            Optimization method, by default 'brent'
        options : Dict[str, Any], optional
            Optimization options

        Returns
        -------
        OptimizationResult
            Optimization result
        """
        if not HAS_SCIPY:
            raise ImportError("SciPy required for scalar optimization")

        from scipy.optimize import minimize_scalar

        opts = {**self.default_options, **(options or {})}

        logger.info(f"Starting scalar optimization with method: {method}")

        import time

        start_time = time.time()

        # Wrap objective to collect history
        eval_count = [0]

        def wrapped_objective(x):
            eval_count[0] += 1
            value = objective(x)
            self.history.append(
                {
                    "iteration": eval_count[0],
                    "parameter": x,
                    "objective": value,
                    "timestamp": time.time() - start_time,
                }
            )
            return value

        # Optimize
        result = minimize_scalar(
            wrapped_objective, bounds=bounds, method=method, options=opts
        )

        end_time = time.time()

        # Convert to our result format
        opt_result = OptimizationResult(
            success=result.success,
            optimal_parameters={"x": result.x},
            optimal_value=result.fun,
            iterations=result.nit if hasattr(result, "nit") else eval_count[0],
            function_evaluations=(
                result.nfev if hasattr(result, "nfev") else eval_count[0]
            ),
            gradient_evaluations=0,
            message=(
                result.message
                if hasattr(result, "message")
                else "Optimization completed"
            ),
            convergence_history=self.history.copy(),
            timing_info={
                "total_time": end_time - start_time,
                "avg_eval_time": (end_time - start_time) / max(eval_count[0], 1),
            },
        )

        self.best_result = opt_result
        logger.info(f"Scalar optimization completed: f*={opt_result.optimal_value:.6e}")

        return opt_result

    def minimize_vector(
        self,
        objective: Callable[[np.ndarray], float],
        initial_guess: Union[np.ndarray, Dict[str, Any]],
        bounds: Optional[List[Tuple[float, float]]] = None,
        method: OptimizationMethod = OptimizationMethod.LBFGS,
        constraints: Optional[List[Dict]] = None,
        options: Dict[str, Any] = None,
    ) -> OptimizationResult:
        """Minimize vector objective function.

        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Objective function to minimize
        initial_guess : np.ndarray or Dict[str, Any]
            Initial parameter values
        bounds : List[Tuple[float, float]], optional
            Parameter bounds
        method : OptimizationMethod, optional
            Optimization method, by default LBFGS
        constraints : List[Dict], optional
            Optimization constraints
        options : Dict[str, Any], optional
            Method-specific options

        Returns
        -------
        OptimizationResult
            Optimization result
        """
        logger.info(f"Starting vector optimization with method: {method.value}")

        # Convert initial guess to array if needed
        if isinstance(initial_guess, dict):
            param_names = list(initial_guess.keys())
            x0 = np.array([initial_guess[name] for name in param_names])
        else:
            param_names = [f"param_{i}" for i in range(len(initial_guess))]
            x0 = np.asarray(initial_guess)

        # Use backend-specific optimization if available
        if method in [OptimizationMethod.ADAM, OptimizationMethod.SGD]:
            return self._optimize_with_backend(
                objective, x0, param_names, method, options
            )
        else:
            return self._optimize_with_scipy(
                objective, x0, param_names, bounds, method, constraints, options
            )

    def _optimize_with_scipy(
        self,
        objective: Callable,
        x0: np.ndarray,
        param_names: List[str],
        bounds: Optional[List[Tuple[float, float]]],
        method: OptimizationMethod,
        constraints: Optional[List[Dict]],
        options: Dict[str, Any],
    ) -> OptimizationResult:
        """Optimize using SciPy methods."""
        if not HAS_SCIPY:
            raise ImportError("SciPy required for optimization")

        import time

        start_time = time.time()

        # Setup history tracking
        self.history = []
        eval_counts = {"func": 0, "grad": 0}

        def wrapped_objective(x):
            eval_counts["func"] += 1
            value = objective(x)
            self.history.append(
                {
                    "iteration": eval_counts["func"],
                    "parameters": x.copy(),
                    "objective": value,
                    "timestamp": time.time() - start_time,
                }
            )
            if eval_counts["func"] % 10 == 0:
                logger.debug(f"Iteration {eval_counts['func']}: f={value:.6e}")
            return value

        # Gradient function
        jac = None
        try:
            grad_func = self.backend.grad(objective)

            def wrapped_gradient(x):
                eval_counts["grad"] += 1
                return grad_func(x)

            if method not in [
                OptimizationMethod.NELDER_MEAD,
                OptimizationMethod.DIFFERENTIAL_EVOLUTION,
            ]:
                jac = wrapped_gradient
        except Exception as e:
            logger.warning(f"Gradient computation failed: {e}")
            jac = None

        # Prepare options
        opts = {**self.default_options, **(options or {})}

        # Choose SciPy method
        if method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            # Global optimization
            result = differential_evolution(
                wrapped_objective,
                bounds=bounds or [(-np.inf, np.inf)] * len(x0),
                **opts,
            )
        else:
            # Local optimization
            scipy_method = method.value.replace("-", "_")
            result = minimize(
                wrapped_objective,
                x0,
                method=scipy_method,
                jac=jac,
                bounds=bounds,
                constraints=constraints,
                options=opts,
            )

        end_time = time.time()

        # Convert result
        optimal_params = {name: val for name, val in zip(param_names, result.x)}

        opt_result = OptimizationResult(
            success=result.success,
            optimal_parameters=optimal_params,
            optimal_value=result.fun,
            iterations=result.nit if hasattr(result, "nit") else eval_counts["func"],
            function_evaluations=(
                result.nfev if hasattr(result, "nfev") else eval_counts["func"]
            ),
            gradient_evaluations=(
                result.njev if hasattr(result, "njev") else eval_counts["grad"]
            ),
            message=result.message,
            convergence_history=self.history.copy(),
            timing_info={
                "total_time": end_time - start_time,
                "avg_eval_time": (end_time - start_time) / max(eval_counts["func"], 1),
            },
        )

        self.best_result = opt_result
        logger.info(
            f"Vector optimization completed: f*={opt_result.optimal_value:.6e}, "
            f"iterations={opt_result.iterations}"
        )

        return opt_result

    def _optimize_with_backend(
        self,
        objective: Callable,
        x0: np.ndarray,
        param_names: List[str],
        method: OptimizationMethod,
        options: Dict[str, Any],
    ) -> OptimizationResult:
        """Optimize using backend-specific optimizers."""
        import time

        start_time = time.time()

        # Convert to backend format
        params = self.backend.to_array(x0)

        # Setup options
        opts = {**self.default_options, **(options or {})}
        num_steps = opts.get("maxiter", 1000)
        learning_rate = opts.get("learning_rate", 0.01)

        # History tracking
        self.history = []

        def tracked_objective(p):
            value = objective(self.backend.from_array(p))
            self.history.append(
                {
                    "iteration": len(self.history) + 1,
                    "parameters": self.backend.from_array(p).copy(),
                    "objective": value,
                    "timestamp": time.time() - start_time,
                }
            )
            return value

        # Optimize
        optimal_params_array = self.backend.optimize(
            tracked_objective,
            params,
            num_steps=num_steps,
            learning_rate=learning_rate,
            optimizer=method.value,
        )

        end_time = time.time()

        # Convert result
        optimal_params_np = self.backend.from_array(optimal_params_array)
        optimal_params = {
            name: val for name, val in zip(param_names, optimal_params_np)
        }
        optimal_value = objective(optimal_params_np)

        opt_result = OptimizationResult(
            success=True,  # Backend optimizers don't return success flag
            optimal_parameters=optimal_params,
            optimal_value=optimal_value,
            iterations=num_steps,
            function_evaluations=len(self.history),
            gradient_evaluations=num_steps,
            message=f"Optimization completed with {method.value}",
            convergence_history=self.history.copy(),
            timing_info={
                "total_time": end_time - start_time,
                "avg_eval_time": (end_time - start_time) / num_steps,
            },
        )

        self.best_result = opt_result
        logger.info(
            f"Backend optimization completed: f*={opt_result.optimal_value:.6e}"
        )

        return opt_result

    def parameter_sweep(
        self,
        objective: Callable,
        parameter_ranges: Dict[str, np.ndarray],
        n_samples: int = None,
    ) -> Dict[str, Any]:
        """Perform parameter sweep analysis.

        Parameters
        ----------
        objective : Callable
            Objective function
        parameter_ranges : Dict[str, np.ndarray]
            Parameter ranges to sweep
        n_samples : int, optional
            Number of samples per parameter (for grid)

        Returns
        -------
        Dict[str, Any]
            Sweep results including parameter values and objectives
        """
        logger.info(f"Starting parameter sweep for {len(parameter_ranges)} parameters")

        param_names = list(parameter_ranges.keys())

        if len(param_names) == 1:
            # 1D sweep
            param_name = param_names[0]
            param_values = parameter_ranges[param_name]
            objectives = []

            for value in param_values:
                obj_val = objective({param_name: value})
                objectives.append(obj_val)

            return {
                "parameter_names": param_names,
                "parameter_values": {param_name: param_values},
                "objectives": np.array(objectives),
                "best_index": np.argmin(objectives),
                "best_parameters": {param_name: param_values[np.argmin(objectives)]},
                "best_objective": np.min(objectives),
            }

        elif len(param_names) == 2:
            # 2D sweep
            param1, param2 = param_names
            values1 = parameter_ranges[param1]
            values2 = parameter_ranges[param2]

            objectives = np.zeros((len(values1), len(values2)))

            for i, val1 in enumerate(values1):
                for j, val2 in enumerate(values2):
                    obj_val = objective({param1: val1, param2: val2})
                    objectives[i, j] = obj_val

            best_idx = np.unravel_index(np.argmin(objectives), objectives.shape)

            return {
                "parameter_names": param_names,
                "parameter_values": {param1: values1, param2: values2},
                "objectives": objectives,
                "best_index": best_idx,
                "best_parameters": {
                    param1: values1[best_idx[0]],
                    param2: values2[best_idx[1]],
                },
                "best_objective": objectives[best_idx],
            }

        else:
            # High-dimensional sweep using Latin hypercube sampling
            from numpy.random import default_rng

            rng = default_rng(42)

            n_samples = n_samples or 100
            samples = []
            objectives = []

            for _ in range(n_samples):
                sample = {}
                for param_name in param_names:
                    param_range = parameter_ranges[param_name]
                    sample[param_name] = rng.choice(param_range)

                obj_val = objective(sample)
                samples.append(sample)
                objectives.append(obj_val)

            best_idx = np.argmin(objectives)

            return {
                "parameter_names": param_names,
                "samples": samples,
                "objectives": np.array(objectives),
                "best_index": best_idx,
                "best_parameters": samples[best_idx],
                "best_objective": objectives[best_idx],
            }

    def sensitivity_analysis(
        self,
        objective: Callable,
        nominal_parameters: Dict[str, float],
        perturbation_size: float = 0.01,
    ) -> Dict[str, float]:
        """Perform local sensitivity analysis.

        Parameters
        ----------
        objective : Callable
            Objective function
        nominal_parameters : Dict[str, float]
            Nominal parameter values
        perturbation_size : float, optional
            Relative perturbation size, by default 0.01

        Returns
        -------
        Dict[str, float]
            Sensitivity coefficients for each parameter
        """
        logger.info("Computing parameter sensitivities")

        nominal_value = objective(nominal_parameters)
        sensitivities = {}

        for param_name, nominal_val in nominal_parameters.items():
            # Forward difference
            perturbed_params = nominal_parameters.copy()
            perturbation = abs(nominal_val) * perturbation_size
            if perturbation == 0:
                perturbation = perturbation_size

            perturbed_params[param_name] = nominal_val + perturbation
            perturbed_value = objective(perturbed_params)

            # Compute sensitivity
            sensitivity = (perturbed_value - nominal_value) / perturbation
            sensitivities[param_name] = sensitivity

            logger.debug(f"Sensitivity of {param_name}: {sensitivity:.6e}")

        return sensitivities

    def multi_start_optimization(
        self,
        objective: Callable,
        bounds: List[Tuple[float, float]],
        n_starts: int = 10,
        method: OptimizationMethod = OptimizationMethod.LBFGS,
        options: Dict[str, Any] = None,
    ) -> List[OptimizationResult]:
        """Perform multi-start optimization to find global optimum.

        Parameters
        ----------
        objective : Callable
            Objective function
        bounds : List[Tuple[float, float]]
            Parameter bounds
        n_starts : int, optional
            Number of random starts, by default 10
        method : OptimizationMethod, optional
            Optimization method, by default LBFGS
        options : Dict[str, Any], optional
            Optimization options

        Returns
        -------
        List[OptimizationResult]
            Results from all starts, sorted by objective value
        """
        logger.info(f"Starting multi-start optimization with {n_starts} starts")

        from numpy.random import default_rng

        rng = default_rng(42)

        results = []

        for start in range(n_starts):
            # Generate random initial guess within bounds
            initial_guess = np.array([rng.uniform(low, high) for low, high in bounds])

            logger.debug(f"Start {start + 1}/{n_starts}")

            try:
                result = self.minimize_vector(
                    objective=objective,
                    initial_guess=initial_guess,
                    bounds=bounds,
                    method=method,
                    options=options,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Start {start + 1} failed: {e}")

        # Sort by objective value
        results.sort(key=lambda r: r.optimal_value)

        if results:
            best_result = results[0]
            logger.info(
                f"Multi-start optimization completed. "
                f"Best result: f*={best_result.optimal_value:.6e}"
            )

        return results

    def get_convergence_plot_data(self) -> Dict[str, np.ndarray]:
        """Get data for convergence plotting.

        Returns
        -------
        Dict[str, np.ndarray]
            Convergence data arrays
        """
        if not self.history:
            return {}

        iterations = [h["iteration"] for h in self.history]
        objectives = [h["objective"] for h in self.history]
        timestamps = [h["timestamp"] for h in self.history]

        return {
            "iterations": np.array(iterations),
            "objectives": np.array(objectives),
            "timestamps": np.array(timestamps),
        }

    def reset_history(self) -> None:
        """Reset optimization history."""
        self.history = []
        self.current_iteration = 0
        logger.debug("Optimization history reset")

    def __repr__(self) -> str:
        return (
            f"OptimizationService("
            f"backend={self.backend.name}, "
            f"evaluations={len(self.history)}, "
            f"best_value={self.best_result.optimal_value if self.best_result else 'None'}"
            f")"
        )
