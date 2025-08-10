"""Core problem definition classes for differentiable FEM."""

from typing import Dict, Any, Optional, Callable, Union, List
import numpy as np
try:
    import jax.numpy as jnp
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from ..backends import get_backend
from ..utils.validation import validate_function_space, validate_boundary_conditions


class Problem:
    """Base class for differentiable finite element problems.
    
    This class provides the foundation for defining and solving
    differentiable finite element problems with automatic differentiation.
    
    Parameters
    ----------
    mesh : firedrake.Mesh
        Computational mesh
    function_space : firedrake.FunctionSpace
        Function space for the problem
    backend : str, optional
        AD backend ('jax' or 'torch'), by default 'jax'
    """
    
    def __init__(
        self,
        mesh=None,
        function_space=None, 
        backend: str = 'numpy'
    ):
        if not HAS_FIREDRAKE:
            # Create minimal problem for demonstration without Firedrake
            print("⚠️  Firedrake not available - running in demo mode")
            
        self.mesh = mesh
        self.function_space = function_space
        self.backend_name = backend
        self.backend = get_backend(backend)
        
        # Problem state
        self.equations = []
        self.boundary_conditions = {}
        self.parameters = {}
        self.solution = None
        
        # Differentiation state
        self._gradient_func = None
        self._jacobian_func = None
        
        if function_space is not None:
            validate_function_space(function_space)
    
    def add_equation(self, equation: Callable, name: str = None) -> 'Problem':
        """Add a physics equation to the problem.
        
        Parameters
        ----------
        equation : Callable
            Function defining the weak form
        name : str, optional
            Name for the equation
            
        Returns
        -------
        Problem
            Self for method chaining
        """
        eq_name = name or f"equation_{len(self.equations)}"
        self.equations.append({
            'name': eq_name,
            'equation': equation,
            'active': True
        })
        return self
    
    def add_boundary_condition(
        self, 
        bc_type: str,
        boundary_id: Union[int, str],
        value: Union[float, Callable],
        name: str = None
    ) -> 'Problem':
        """Add boundary condition to the problem.
        
        Parameters
        ----------
        bc_type : str
            Type of boundary condition ('dirichlet', 'neumann', 'robin')
        boundary_id : int or str
            Boundary marker or name
        value : float or Callable
            Boundary value or function
        name : str, optional
            Name for the boundary condition
            
        Returns
        -------
        Problem
            Self for method chaining
        """
        bc_name = name or f"{bc_type}_{boundary_id}"
        self.boundary_conditions[bc_name] = {
            'type': bc_type,
            'boundary': boundary_id,
            'value': value
        }
        validate_boundary_conditions({bc_name: self.boundary_conditions[bc_name]})
        return self
    
    def set_parameter(self, name: str, value: Any) -> 'Problem':
        """Set a problem parameter.
        
        Parameters
        ----------
        name : str
            Parameter name
        value : Any
            Parameter value
            
        Returns
        -------
        Problem
            Self for method chaining
        """
        self.parameters[name] = value
        return self
    
    def solve(self, parameters: Dict[str, Any] = None) -> Any:
        """Solve the finite element problem.
        
        Parameters
        ----------
        parameters : Dict[str, Any], optional
            Runtime parameters to override defaults
            
        Returns
        -------
        Any
            Solution field
        """
        if not HAS_FIREDRAKE:
            raise RuntimeError("Firedrake required for FEM solve")
            
        # Merge runtime parameters
        solve_params = {**self.parameters}
        if parameters:
            solve_params.update(parameters)
        
        # Create function for solution
        if self.function_space is None:
            raise ValueError("Function space must be defined before solving")
            
        u = fd.Function(self.function_space)
        v = fd.TestFunction(self.function_space)
        
        # Assemble weak form
        F = 0
        for eq in self.equations:
            if eq['active']:
                F += eq['equation'](u, v, solve_params)
        
        # Apply boundary conditions
        bcs = self._assemble_boundary_conditions()
        
        # Solve
        fd.solve(F == 0, u, bcs=bcs)
        
        self.solution = u
        return u
    
    def _assemble_boundary_conditions(self) -> List:
        """Convert boundary condition definitions to Firedrake BCs."""
        bcs = []
        for bc_name, bc_def in self.boundary_conditions.items():
            if bc_def['type'] == 'dirichlet':
                if callable(bc_def['value']):
                    # Function-based BC
                    value = fd.Function(self.function_space)
                    value.interpolate(bc_def['value'])
                else:
                    # Constant BC  
                    value = fd.Constant(bc_def['value'])
                
                bc = fd.DirichletBC(
                    self.function_space,
                    value,
                    bc_def['boundary']
                )
                bcs.append(bc)
        return bcs
    
    def differentiable(self, func: Callable) -> Callable:
        """Decorator to make a function differentiable.
        
        Parameters
        ----------
        func : Callable
            Function to make differentiable
            
        Returns
        -------
        Callable
            Differentiable version of the function
        """
        if self.backend_name == 'jax' and HAS_JAX:
            return jax.jit(func)
        elif self.backend_name == 'torch' and HAS_TORCH:
            # For PyTorch, we'd need to handle this differently
            return func
        else:
            return func
    
    def compute_gradient(
        self, 
        objective: Callable,
        parameters: List[str]
    ) -> Callable:
        """Compute gradient of objective with respect to parameters.
        
        Parameters
        ----------
        objective : Callable
            Objective function to differentiate
        parameters : List[str]
            Parameter names to differentiate with respect to
            
        Returns
        -------
        Callable
            Gradient function
        """
        if self.backend_name == 'jax' and HAS_JAX:
            return jax.grad(objective)
        elif self.backend_name == 'torch' and HAS_TORCH:
            def torch_grad(*args):
                # PyTorch gradient computation
                inputs = [torch.tensor(arg, requires_grad=True) for arg in args]
                output = objective(*inputs)
                gradients = torch.autograd.grad(output, inputs, create_graph=True)
                return [g.detach().numpy() for g in gradients]
            return torch_grad
        else:
            raise NotImplementedError(f"Gradient computation not supported for backend {self.backend_name}")
    
    def optimize(
        self,
        objective: Callable,
        initial_guess: Dict[str, Any],
        method: str = 'lbfgs',
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Optimize parameters to minimize objective function.
        
        Parameters
        ----------
        objective : Callable
            Objective function to minimize
        initial_guess : Dict[str, Any]
            Initial parameter values
        method : str, optional
            Optimization method, by default 'lbfgs'
        options : Dict[str, Any], optional
            Optimizer options
            
        Returns
        -------
        Dict[str, Any]
            Optimization result
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError("scipy required for optimization")
        
        # Extract parameter values and names
        param_names = list(initial_guess.keys())
        x0 = np.array([initial_guess[name] for name in param_names])
        
        # Wrapper function for scipy
        def scipy_objective(x):
            params = {name: val for name, val in zip(param_names, x)}
            return float(objective(params))
        
        # Compute gradient if backend supports it
        jac = None
        if self.backend_name in ['jax', 'torch']:
            grad_func = self.compute_gradient(objective, param_names)
            def scipy_gradient(x):
                params = {name: val for name, val in zip(param_names, x)}
                return np.array(grad_func(params))
            jac = scipy_gradient
        
        # Run optimization
        result = minimize(
            scipy_objective,
            x0,
            method=method,
            jac=jac,
            options=options or {}
        )
        
        # Convert result back to parameter dict
        optimal_params = {name: val for name, val in zip(param_names, result.x)}
        
        return {
            'success': result.success,
            'parameters': optimal_params,
            'objective_value': result.fun,
            'iterations': result.nit,
            'message': result.message
        }
    
    def generate_observations(
        self,
        num_points: int,
        noise_level: float = 0.0,
        seed: int = None
    ) -> np.ndarray:
        """Generate synthetic observations for inverse problems.
        
        Parameters
        ----------
        num_points : int
            Number of observation points
        noise_level : float, optional
            Gaussian noise standard deviation, by default 0.0
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Synthetic observations
        """
        if seed is not None:
            np.random.seed(seed)
            
        if self.solution is None:
            raise ValueError("Must solve problem before generating observations")
        
        # Sample solution at random points
        coords = np.random.uniform(0, 1, (num_points, self.mesh.geometric_dimension()))
        
        # Evaluate solution (simplified - real implementation would use Firedrake)
        observations = np.random.normal(0, 1, num_points)  # Placeholder
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, num_points)
            observations += noise
            
        return observations
    
    def __repr__(self) -> str:
        return (
            f"Problem("
            f"backend={self.backend_name}, "
            f"equations={len(self.equations)}, "
            f"bcs={len(self.boundary_conditions)}"
            f")"
        )


class FEBMLProblem(Problem):
    """Specialized problem class for FEBML (Finite Element-Based Machine Learning).
    
    Extends the base Problem class with machine learning specific functionality
    like automatic experiment tracking, reproducibility features, and enhanced
    optimization capabilities.
    """
    
    def __init__(self, *args, experiment_name: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_name = experiment_name
        self.metrics = {}
        self.checkpoints = {}
        
    def log_metric(self, name: str, value: float) -> None:
        """Log a metric value for experiment tracking."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def checkpoint(self, solution, name: str) -> None:
        """Save a checkpoint of the solution."""
        self.checkpoints[name] = {
            'solution': solution,
            'parameters': self.parameters.copy(),
            'timestamp': np.datetime64('now')
        }
    
    def parameterize_field(self, params: np.ndarray) -> Any:
        """Convert parameter array to field representation."""
        # Placeholder implementation
        return params
    
    def observe(self, solution) -> np.ndarray:
        """Extract observations from solution."""
        # Placeholder implementation  
        return np.array([solution]) if np.isscalar(solution) else solution
