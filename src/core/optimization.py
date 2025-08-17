"""Optimization algorithms for DiffFE-Physics-Lab."""

import logging
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ParameterOptimizer(ABC):
    """Abstract base class for parameter optimization."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        """Initialize optimizer.
        
        Args:
            learning_rate: Learning rate for optimization
            max_iterations: Maximum number of iterations
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.history = []
        
    @abstractmethod
    def optimize(self, objective: Callable, initial_params: Dict[str, Any], 
                **kwargs) -> Dict[str, Any]:
        """Optimize parameters. Must be implemented by subclasses."""
        pass
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.history


class GradientDescentOptimizer(ParameterOptimizer):
    """Simple gradient descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-8):
        """Initialize gradient descent optimizer.
        
        Args:
            learning_rate: Learning rate
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
        """
        super().__init__(learning_rate, max_iterations)
        self.tolerance = tolerance
    
    def optimize(self, objective: Callable, initial_params: Dict[str, Any],
                **kwargs) -> Dict[str, Any]:
        """Optimize using gradient descent.
        
        Args:
            objective: Objective function to minimize
            initial_params: Initial parameter values
            **kwargs: Additional optimization options
            
        Returns:
            Optimized parameters
        """
        logger.info("Starting gradient descent optimization")
        
        current_params = initial_params.copy()
        best_params = current_params.copy()
        best_objective = float('inf')
        
        self.history = []
        
        for iteration in range(self.max_iterations):
            # Evaluate objective
            objective_value = objective(current_params)
            
            # Track best solution
            if objective_value < best_objective:
                best_objective = objective_value
                best_params = current_params.copy()
            
            # Record history
            iteration_data = {
                "iteration": iteration,
                "objective": objective_value,
                "params": current_params.copy()
            }
            self.history.append(iteration_data)
            
            # Check convergence
            if abs(objective_value) < self.tolerance:
                logger.info(f"Converged at iteration {iteration}")
                break
            
            # Simple finite difference gradient approximation
            gradients = self._compute_gradients(objective, current_params)
            
            # Update parameters
            for param_name, gradient in gradients.items():
                current_params[param_name] -= self.learning_rate * gradient
            
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}: objective = {objective_value}")
        
        logger.info(f"Optimization complete. Best objective: {best_objective}")
        return best_params
    
    def _compute_gradients(self, objective: Callable, params: Dict[str, Any],
                          epsilon: float = 1e-6) -> Dict[str, float]:
        """Compute gradients using finite differences.
        
        Args:
            objective: Objective function
            params: Current parameters
            epsilon: Finite difference step size
            
        Returns:
            Gradient dictionary
        """
        gradients = {}
        base_value = objective(params)
        
        for param_name, param_value in params.items():
            # Only compute gradients for numeric parameters
            if isinstance(param_value, (int, float)):
                # Forward difference
                perturbed_params = params.copy()
                perturbed_params[param_name] = param_value + epsilon
                perturbed_value = objective(perturbed_params)
                
                gradient = (perturbed_value - base_value) / epsilon
                gradients[param_name] = gradient
        
        return gradients


class GridSearchOptimizer(ParameterOptimizer):
    """Grid search optimizer for discrete parameter spaces."""
    
    def __init__(self, parameter_ranges: Dict[str, List], **kwargs):
        """Initialize grid search optimizer.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to value lists
            **kwargs: Additional optimizer arguments
        """
        super().__init__(**kwargs)
        self.parameter_ranges = parameter_ranges
    
    def optimize(self, objective: Callable, initial_params: Dict[str, Any],
                **kwargs) -> Dict[str, Any]:
        """Optimize using grid search.
        
        Args:
            objective: Objective function to minimize
            initial_params: Initial parameter values (not used in grid search)
            **kwargs: Additional optimization options
            
        Returns:
            Best parameters found
        """
        logger.info("Starting grid search optimization")
        
        best_params = initial_params.copy()
        best_objective = float('inf')
        
        self.history = []
        iteration = 0
        
        # Generate all parameter combinations
        from itertools import product
        
        param_names = list(self.parameter_ranges.keys())
        param_values = [self.parameter_ranges[name] for name in param_names]
        
        for value_combination in product(*param_values):
            # Create parameter dictionary
            current_params = initial_params.copy()
            for name, value in zip(param_names, value_combination):
                current_params[name] = value
            
            # Evaluate objective
            objective_value = objective(current_params)
            
            # Track best solution
            if objective_value < best_objective:
                best_objective = objective_value
                best_params = current_params.copy()
            
            # Record history
            iteration_data = {
                "iteration": iteration,
                "objective": objective_value,
                "params": current_params.copy()
            }
            self.history.append(iteration_data)
            
            iteration += 1
            
            if iteration >= self.max_iterations:
                logger.warning(f"Reached maximum iterations ({self.max_iterations})")
                break
        
        logger.info(f"Grid search complete. Best objective: {best_objective}")
        logger.info(f"Evaluated {iteration} parameter combinations")
        
        return best_params


class BayesianOptimizer(ParameterOptimizer):
    """Bayesian optimization using Gaussian processes (simplified implementation)."""
    
    def __init__(self, acquisition_function: str = "expected_improvement", **kwargs):
        """Initialize Bayesian optimizer.
        
        Args:
            acquisition_function: Acquisition function to use
            **kwargs: Additional optimizer arguments
        """
        super().__init__(**kwargs)
        self.acquisition_function = acquisition_function
        self.observations = []
    
    def optimize(self, objective: Callable, initial_params: Dict[str, Any],
                **kwargs) -> Dict[str, Any]:
        """Optimize using Bayesian optimization.
        
        Args:
            objective: Objective function to minimize
            initial_params: Initial parameter values
            **kwargs: Additional optimization options
            
        Returns:
            Best parameters found
        """
        logger.info("Starting Bayesian optimization")
        
        # Simplified Bayesian optimization - random search with Gaussian process surrogate
        # In a full implementation, this would use libraries like scikit-optimize
        
        best_params = initial_params.copy()
        best_objective = float('inf')
        
        self.history = []
        self.observations = []
        
        # Start with initial evaluation
        initial_objective = objective(initial_params)
        best_objective = initial_objective
        
        self.observations.append((initial_params.copy(), initial_objective))
        
        for iteration in range(self.max_iterations):
            # Simple acquisition: random search around best point
            candidate_params = self._generate_candidate(best_params)
            
            # Evaluate candidate
            objective_value = objective(candidate_params)
            self.observations.append((candidate_params.copy(), objective_value))
            
            # Update best solution
            if objective_value < best_objective:
                best_objective = objective_value
                best_params = candidate_params.copy()
                logger.debug(f"New best at iteration {iteration}: {best_objective}")
            
            # Record history
            iteration_data = {
                "iteration": iteration,
                "objective": objective_value,
                "params": candidate_params.copy(),
                "is_best": objective_value < best_objective
            }
            self.history.append(iteration_data)
        
        logger.info(f"Bayesian optimization complete. Best objective: {best_objective}")
        return best_params
    
    def _generate_candidate(self, best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate candidate parameters using acquisition function."""
        import random
        
        candidate_params = best_params.copy()
        
        # Simple random perturbation around best point
        for param_name, param_value in candidate_params.items():
            if isinstance(param_value, (int, float)):
                noise_scale = abs(param_value) * 0.1 + 0.01  # 10% + small constant
                noise = random.gauss(0, noise_scale)
                candidate_params[param_name] = param_value + noise
        
        return candidate_params


def create_optimizer(optimizer_type: str = "gradient_descent", **kwargs) -> ParameterOptimizer:
    """Factory function to create optimizers.
    
    Args:
        optimizer_type: Type of optimizer
        **kwargs: Optimizer-specific arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type == "gradient_descent":
        return GradientDescentOptimizer(**kwargs)
    elif optimizer_type == "grid_search":
        return GridSearchOptimizer(**kwargs)
    elif optimizer_type == "bayesian":
        return BayesianOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")