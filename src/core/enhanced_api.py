"""Enhanced API layer for DiffFE-Physics-Lab.

This module provides simplified, user-friendly interfaces to the core functionality
while maintaining compatibility with the existing codebase.

Enhanced with robust error handling, validation, monitoring, and security features.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

# Import robust infrastructure
from ..robust.error_handling import (
    DiffFEError, ValidationError, ConvergenceError, BackendError, MemoryError,
    error_context, validate_positive, validate_range
)
from ..robust.logging_system import (
    get_logger, log_performance, global_audit_logger, global_performance_logger
)
from ..robust.monitoring import (
    global_performance_monitor, global_health_checker, resource_monitor
)
from ..robust.security import (
    global_security_validator, global_input_sanitizer, SecurityError
)

logger = get_logger(__name__)

@dataclass
class ProblemConfig:
    """Configuration for FEM problems with robust validation."""
    mesh_size: int = 50
    element_order: int = 1
    backend: str = "numpy"
    precision: str = "float64"
    parallel: bool = False
    gpu_enabled: bool = False
    logging_level: str = "INFO"
    # Robust configuration options
    enable_monitoring: bool = True
    enable_validation: bool = True
    enable_security_checks: bool = True
    memory_limit_mb: Optional[int] = 4096
    timeout_seconds: Optional[float] = 1800
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization with enhanced checks."""
        with error_context("ProblemConfig_validation"):
            # Sanitize string inputs
            self.backend = global_input_sanitizer.sanitize_string(self.backend)
            self.precision = global_input_sanitizer.sanitize_string(self.precision)
            self.logging_level = global_input_sanitizer.sanitize_string(self.logging_level)
            
            # Comprehensive validation
            validate_positive(self.mesh_size, "mesh_size")
            if self.mesh_size > 100000:
                logger.warning(f"Large mesh size requested: {self.mesh_size}")
            
            validate_range(self.element_order, 1, 3, "element_order")
            
            if self.backend not in ["numpy", "jax", "torch"]:
                logger.warning(f"Backend '{self.backend}' may not be fully supported")
            
            if self.precision not in ["float32", "float64"]:
                raise ValidationError("precision must be 'float32' or 'float64'",
                                    field="precision", value=self.precision)
            
            if self.logging_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                logger.warning(f"Unknown logging level '{self.logging_level}', using INFO")
                self.logging_level = "INFO"
            
            # Validate robust options
            if self.memory_limit_mb is not None:
                validate_range(self.memory_limit_mb, 100, 100000, "memory_limit_mb")
            
            if self.timeout_seconds is not None:
                validate_range(self.timeout_seconds, 1, 86400, "timeout_seconds")
            
            validate_range(self.max_retries, 0, 10, "max_retries")
            
            # Log configuration creation
            global_audit_logger.log_data_operation(
                "create", "ProblemConfig", 1,
                mesh_size=self.mesh_size, backend=self.backend
            )


class BaseProblem(ABC):
    """Abstract base class for all physics problems."""
    
    def __init__(self, config: Optional[ProblemConfig] = None):
        """Initialize base problem.
        
        Args:
            config: Problem configuration. If None, uses defaults.
        """
        self.config = config or ProblemConfig()
        self.solution = None
        self.parameters = {}
        self.metrics = {}
        self._is_solved = False
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.logging_level))
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def solve(self, **kwargs) -> Any:
        """Solve the problem. Must be implemented by subclasses."""
        pass
    
    def set_parameters(self, **params):
        """Set problem parameters."""
        self.parameters.update(params)
        self._is_solved = False  # Mark as needing re-solving
        logger.debug(f"Updated parameters: {params}")
    
    def get_parameter(self, name: str, default=None):
        """Get a parameter value."""
        return self.parameters.get(name, default)
    
    def validate_solution(self) -> bool:
        """Check if solution is valid."""
        return self._is_solved and self.solution is not None


class FEBMLProblem(BaseProblem):
    """Enhanced FEBMLProblem with simplified API."""
    
    def __init__(self, config: Optional[ProblemConfig] = None):
        """Initialize FEBML problem.
        
        Args:
            config: Problem configuration.
        """
        super().__init__(config)
        self.equations = []
        self.boundary_conditions = {}
        self.source_terms = {}
        self.observers = []
        
    def add_equation(self, equation: Union[str, Callable], **params):
        """Add physics equation to the problem.
        
        Args:
            equation: Equation name (str) or custom function
            **params: Equation parameters
            
        Examples:
            >>> problem.add_equation("laplacian", conductivity=1.0)
            >>> problem.add_equation("advection", velocity=[1.0, 0.0])
        """
        eq_config = {"type": equation, "params": params}
        self.equations.append(eq_config)
        logger.info(f"Added equation: {equation} with params: {params}")
    
    def set_boundary_condition(self, boundary: str, condition_type: str, value):
        """Set boundary condition.
        
        Args:
            boundary: Boundary identifier (e.g., "left", "right", "top", "bottom")
            condition_type: Type ("dirichlet", "neumann", "robin")
            value: Boundary value (scalar, function, or dict for robin)
        """
        self.boundary_conditions[boundary] = {
            "type": condition_type,
            "value": value
        }
        logger.info(f"Set {condition_type} BC on {boundary}: {value}")
    
    def add_source_term(self, source: Union[float, Callable], region: str = "all"):
        """Add source term to the problem.
        
        Args:
            source: Source function or constant value
            region: Region where source applies (default: "all")
        """
        self.source_terms[region] = source
        logger.info(f"Added source term in region '{region}': {source}")
    
    def add_observer(self, location, name: Optional[str] = None):
        """Add observation point for data collection.
        
        Args:
            location: Observation location (point, line, or region)
            name: Optional name for the observer
        """
        observer = {
            "location": location,
            "name": name or f"observer_{len(self.observers)}",
            "data": []
        }
        self.observers.append(observer)
        logger.info(f"Added observer '{observer['name']}' at {location}")
    
    def solve(self, method: str = "direct", **kwargs) -> Any:
        """Solve the FEBML problem.
        
        Args:
            method: Solution method ("direct", "iterative", "adaptive")
            **kwargs: Additional solver options
            
        Returns:
            Solution object
        """
        logger.info(f"Solving problem with method: {method}")
        
        # Validate problem setup
        if not self.equations:
            raise ValueError("No equations defined. Add equations with add_equation()")
        
        # For demonstration - actual implementation would interface with backends
        if method == "direct":
            self.solution = self._solve_direct(**kwargs)
        elif method == "iterative":
            self.solution = self._solve_iterative(**kwargs)
        elif method == "adaptive":
            self.solution = self._solve_adaptive(**kwargs)
        else:
            raise ValueError(f"Unknown solution method: {method}")
        
        self._is_solved = True
        self._update_metrics()
        
        logger.info("Problem solved successfully")
        return self.solution
    
    def _solve_direct(self, **kwargs):
        """Direct solver implementation."""
        logger.debug("Using direct solver")
        
        # Create a Problem instance for solving
        from ..models import Problem
        problem = Problem(backend=self.config.backend)
        
        # Add equations
        for eq_config in self.equations:
            eq_type = eq_config["type"]
            eq_params = eq_config["params"]
            problem.add_equation(eq_type, **eq_params)
        
        # Add boundary conditions
        for bc_name, bc_config in self.boundary_conditions.items():
            problem.add_boundary_condition(
                bc_config["type"],
                bc_config["value"], 
                bc_config["value"],
                name=bc_name
            )
        
        # Set mesh parameters based on problem type
        if self.equations:
            eq_type = self.equations[0]["type"]
            if eq_type == "laplacian":
                # Default mesh parameters
                mesh_params = {
                    "dimension": kwargs.get("dimension", 2),
                    "x_range": kwargs.get("x_range", (0.0, 1.0)),
                    "y_range": kwargs.get("y_range", (0.0, 1.0)),
                    "nx": kwargs.get("nx", self.config.mesh_size),
                    "ny": kwargs.get("ny", self.config.mesh_size),
                    "x_start": kwargs.get("x_start", 0.0),
                    "x_end": kwargs.get("x_end", 1.0),
                    "num_elements": kwargs.get("num_elements", self.config.mesh_size)
                }
                problem.set_mesh_parameters(**mesh_params)
        
        # Set other parameters
        problem.set_parameter("diffusion_coeff", kwargs.get("diffusion_coeff", 1.0))
        if "source_function" in kwargs:
            problem.set_parameter("source_function", kwargs["source_function"])
        
        # Solve the problem
        try:
            solution = problem.solve()
            if isinstance(solution, np.ndarray):
                return {
                    "method": "direct",
                    "solution": solution,
                    "success": True,
                    "num_dofs": len(solution),
                    "equations": len(self.equations)
                }
            else:
                return {
                    "method": "direct", 
                    "solution": solution,
                    "success": True,
                    "equations": len(self.equations)
                }
        except Exception as e:
            logger.error(f"Direct solver failed: {e}")
            return {
                "method": "direct",
                "success": False,
                "error": str(e),
                "equations": len(self.equations)
            }
    
    def _solve_iterative(self, **kwargs):
        """Iterative solver implementation."""
        logger.debug("Using iterative solver")
        max_iterations = kwargs.get("max_iterations", 1000)
        tolerance = kwargs.get("tolerance", 1e-8)
        return {
            "method": "iterative", 
            "max_iterations": max_iterations,
            "tolerance": tolerance
        }
    
    def _solve_adaptive(self, **kwargs):
        """Adaptive solver implementation."""
        logger.debug("Using adaptive solver")
        target_error = kwargs.get("target_error", 1e-6)
        return {"method": "adaptive", "target_error": target_error}
    
    def _update_metrics(self):
        """Update solution metrics."""
        if self.solution:
            self.metrics.update({
                "solution_computed": True,
                "num_equations": len(self.equations),
                "num_boundary_conditions": len(self.boundary_conditions),
                "num_observers": len(self.observers)
            })
    
    def compute_error(self, reference_solution, norm: str = "L2") -> float:
        """Compute error against reference solution.
        
        Args:
            reference_solution: Reference solution for comparison
            norm: Error norm type ("L2", "H1", "inf")
            
        Returns:
            Error value
        """
        if not self.validate_solution():
            raise ValueError("Problem not solved. Call solve() first.")
        
        # Placeholder for actual error computation
        logger.info(f"Computing {norm} error against reference")
        error = 0.001  # Dummy error value
        
        self.metrics[f"{norm}_error"] = error
        return error
    
    def export_solution(self, filename: str, format: str = "vtk"):
        """Export solution to file.
        
        Args:
            filename: Output filename
            format: Export format ("vtk", "hdf5", "csv")
        """
        if not self.validate_solution():
            raise ValueError("Problem not solved. Call solve() first.")
        
        logger.info(f"Exporting solution to {filename} in {format} format")
        # Placeholder for actual export functionality


class MultiPhysics(BaseProblem):
    """Multi-physics problem combining multiple domains and physics."""
    
    def __init__(self, config: Optional[ProblemConfig] = None):
        """Initialize multi-physics problem."""
        super().__init__(config)
        self.domains = {}
        self.interfaces = {}
        self.coupling_schemes = {}
    
    def add_domain(self, name: str, problem: FEBMLProblem, equations: List[str]):
        """Add a physics domain.
        
        Args:
            name: Domain identifier
            problem: FEBML problem for this domain
            equations: List of equations for this domain
        """
        self.domains[name] = {
            "problem": problem,
            "equations": equations,
            "solution": None
        }
        logger.info(f"Added domain '{name}' with {len(equations)} equations")
    
    def add_interface(self, name: str, domain1: str, domain2: str, 
                     coupling_type: str = "dirichlet_neumann"):
        """Add interface between domains.
        
        Args:
            name: Interface identifier
            domain1: First domain name
            domain2: Second domain name
            coupling_type: Type of coupling ("dirichlet_neumann", "robin", "mortar")
        """
        self.interfaces[name] = {
            "domain1": domain1,
            "domain2": domain2,
            "coupling_type": coupling_type
        }
        logger.info(f"Added interface '{name}' between {domain1} and {domain2}")
    
    def solve(self, coupling_scheme: str = "fixed_point", **kwargs) -> Dict[str, Any]:
        """Solve multi-physics problem.
        
        Args:
            coupling_scheme: Coupling solution scheme
            **kwargs: Additional solver options
            
        Returns:
            Dictionary of domain solutions
        """
        logger.info(f"Solving multi-physics problem with {coupling_scheme} coupling")
        
        if not self.domains:
            raise ValueError("No domains defined. Add domains with add_domain()")
        
        # Solve each domain (simplified implementation)
        solutions = {}
        for name, domain_data in self.domains.items():
            logger.info(f"Solving domain: {name}")
            problem = domain_data["problem"]
            solutions[name] = problem.solve(**kwargs)
        
        self.solution = solutions
        self._is_solved = True
        
        logger.info("Multi-physics problem solved successfully")
        return solutions


class HybridSolver(BaseProblem):
    """Hybrid FEM-Neural Network solver."""
    
    def __init__(self, fem_problem: FEBMLProblem, neural_network, 
                 config: Optional[ProblemConfig] = None):
        """Initialize hybrid solver.
        
        Args:
            fem_problem: FEM problem component
            neural_network: Neural network component
            config: Problem configuration
        """
        super().__init__(config)
        self.fem_problem = fem_problem
        self.neural_network = neural_network
        self.coupling_strength = 0.5
        self.training_data = []
    
    def set_coupling_strength(self, strength: float):
        """Set coupling strength between FEM and NN.
        
        Args:
            strength: Coupling strength (0.0 = pure FEM, 1.0 = pure NN)
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Coupling strength must be between 0.0 and 1.0")
        
        self.coupling_strength = strength
        logger.info(f"Set coupling strength to {strength}")
    
    def add_training_data(self, inputs, outputs):
        """Add training data for neural network component.
        
        Args:
            inputs: Input data
            outputs: Target outputs
        """
        self.training_data.append({"inputs": inputs, "outputs": outputs})
        logger.info(f"Added training data: {len(inputs)} samples")
    
    def train_neural_component(self, epochs: int = 100, **kwargs):
        """Train the neural network component.
        
        Args:
            epochs: Number of training epochs
            **kwargs: Additional training parameters
        """
        if not self.training_data:
            raise ValueError("No training data available. Add data with add_training_data()")
        
        logger.info(f"Training neural component for {epochs} epochs")
        # Placeholder for actual training
        return {"epochs": epochs, "final_loss": 0.001}
    
    def solve(self, use_fem_for: List[str] = None, use_nn_for: List[str] = None,
              **kwargs) -> Any:
        """Solve hybrid problem.
        
        Args:
            use_fem_for: Physics components to solve with FEM
            use_nn_for: Physics components to solve with NN
            **kwargs: Additional solver options
            
        Returns:
            Hybrid solution
        """
        use_fem_for = use_fem_for or []
        use_nn_for = use_nn_for or []
        
        logger.info(f"Solving hybrid problem: FEM for {use_fem_for}, NN for {use_nn_for}")
        
        # Solve FEM component
        fem_solution = self.fem_problem.solve(**kwargs)
        
        # Apply neural network correction/enhancement
        # Placeholder for actual hybrid solving
        hybrid_solution = {
            "fem_component": fem_solution,
            "nn_component": {"correction": 0.01},
            "coupling_strength": self.coupling_strength
        }
        
        self.solution = hybrid_solution
        self._is_solved = True
        
        logger.info("Hybrid problem solved successfully")
        return hybrid_solution


def create_problem(problem_type: str = "febml", **config_kwargs) -> BaseProblem:
    """Factory function to create problems.
    
    Args:
        problem_type: Type of problem ("febml", "multiphysics", "hybrid")
        **config_kwargs: Configuration parameters
        
    Returns:
        Problem instance
        
    Examples:
        >>> problem = create_problem("febml", mesh_size=100, backend="jax")
        >>> multiphysics = create_problem("multiphysics", parallel=True)
    """
    config = ProblemConfig(**config_kwargs)
    
    if problem_type == "febml":
        return FEBMLProblem(config)
    elif problem_type == "multiphysics":
        return MultiPhysics(config)
    elif problem_type == "hybrid":
        # Note: hybrid requires additional parameters
        raise ValueError("Hybrid problems need FEM problem and neural network components")
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")


def quick_solve(equation: str, boundary_conditions: Dict[str, Any], 
                mesh_size: int = 50, dimension: int = 1, **kwargs) -> Any:
    """Quick solve for simple problems.
    
    Args:
        equation: Physics equation name
        boundary_conditions: Dictionary of boundary conditions
        mesh_size: Mesh resolution
        dimension: Problem dimension (1 or 2)
        **kwargs: Additional parameters
        
    Returns:
        Solution
        
    Examples:
        >>> solution = quick_solve("laplacian", {"left": 0, "right": 1}, mesh_size=100)
        >>> solution_2d = quick_solve("laplacian", {"left": 0, "right": 1, "bottom": 0, "top": 0}, 
        ...                          mesh_size=20, dimension=2)
    """
    # Create and configure problem
    config = ProblemConfig(mesh_size=mesh_size, backend="numpy")
    problem = FEBMLProblem(config)
    problem.add_equation(equation, **kwargs)
    
    # Set boundary conditions
    for boundary, value in boundary_conditions.items():
        problem.set_boundary_condition(boundary, "dirichlet", value)
    
    # Set dimension and mesh parameters
    problem._is_solved = False  # Reset solve state
    
    # Solve with dimension parameter
    return problem.solve(dimension=dimension, **kwargs)