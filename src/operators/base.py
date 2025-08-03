"""Base operator class and registration system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional, Union
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

# Global operator registry
_OPERATOR_REGISTRY: Dict[str, type] = {}


def register_operator(name: str):
    """Decorator to register an operator class.
    
    Parameters
    ----------
    name : str
        Name to register the operator under
        
    Returns
    -------
    Callable
        Decorator function
    """
    def decorator(cls):
        _OPERATOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_operator(name: str) -> type:
    """Get registered operator class by name.
    
    Parameters
    ----------
    name : str
        Operator name
        
    Returns
    -------
    type
        Operator class
        
    Raises
    ------
    KeyError
        If operator not found
    """
    if name not in _OPERATOR_REGISTRY:
        raise KeyError(f"Operator '{name}' not found. Available: {list(_OPERATOR_REGISTRY.keys())}")
    return _OPERATOR_REGISTRY[name]


class BaseOperator(ABC):
    """Base class for all differentiable operators.
    
    This class defines the interface that all physics operators must implement
    to support automatic differentiation through finite element operations.
    
    Parameters
    ----------
    backend : str, optional
        Backend for automatic differentiation ('jax' or 'torch')
    **kwargs
        Additional operator-specific parameters
    """
    
    def __init__(self, backend: str = 'jax', **kwargs):
        self.backend = backend
        self.params = kwargs
        self._compiled_forward = None
        self._compiled_adjoint = None
        
        # Validate backend availability
        if backend == 'jax' and not HAS_JAX:
            raise ImportError("JAX backend requested but not available")
        elif backend == 'torch' and not HAS_TORCH:
            raise ImportError("PyTorch backend requested but not available")
    
    @abstractmethod
    def forward_assembly(
        self, 
        trial: Any, 
        test: Any, 
        params: Dict[str, Any]
    ) -> Any:
        """Assemble the forward operator.
        
        Parameters
        ----------
        trial : Any
            Trial function
        test : Any
            Test function  
        params : Dict[str, Any]
            Problem parameters
            
        Returns
        -------
        Any
            Assembled weak form
        """
        pass
    
    @abstractmethod
    def adjoint_assembly(
        self,
        grad_output: Any,
        trial: Any,
        test: Any,
        params: Dict[str, Any]
    ) -> Any:
        """Assemble the adjoint operator.
        
        Parameters
        ----------
        grad_output : Any
            Gradient with respect to output
        trial : Any
            Trial function
        test : Any  
            Test function
        params : Dict[str, Any]
            Problem parameters
            
        Returns
        -------
        Any
            Adjoint weak form
        """
        pass
    
    def __call__(
        self, 
        trial: Any, 
        test: Any, 
        params: Dict[str, Any] = None
    ) -> Any:
        """Apply the operator.
        
        Parameters
        ----------
        trial : Any
            Trial function
        test : Any
            Test function
        params : Dict[str, Any], optional
            Problem parameters
            
        Returns
        -------
        Any
            Result of operator application
        """
        if params is None:
            params = {}
            
        # Merge operator params with runtime params
        all_params = {**self.params, **params}
        
        return self.forward_assembly(trial, test, all_params)
    
    @property
    def is_linear(self) -> bool:
        """Whether the operator is linear.
        
        Returns
        -------
        bool
            True if operator is linear
        """
        return getattr(self, '_is_linear', False)
    
    @property
    def is_symmetric(self) -> bool:
        """Whether the operator is symmetric.
        
        Returns
        -------
        bool
            True if operator is symmetric
        """
        return getattr(self, '_is_symmetric', False)
    
    def compile(self) -> 'BaseOperator':
        """Compile the operator for improved performance.
        
        Returns
        -------
        BaseOperator
            Self for method chaining
        """
        if self.backend == 'jax' and HAS_JAX:
            self._compiled_forward = jax.jit(self.forward_assembly)
            self._compiled_adjoint = jax.jit(self.adjoint_assembly)
        # PyTorch compilation would be handled differently
        return self
    
    def validate_inputs(
        self, 
        trial: Any, 
        test: Any, 
        params: Dict[str, Any]
    ) -> None:
        """Validate operator inputs.
        
        Parameters
        ----------
        trial : Any
            Trial function
        test : Any
            Test function
        params : Dict[str, Any]
            Parameters
            
        Raises
        ------
        ValueError
            If inputs are invalid
        """
        # Basic validation - subclasses can override
        if trial is None:
            raise ValueError("Trial function cannot be None")
        if test is None:
            raise ValueError("Test function cannot be None")
    
    def compute_error(
        self,
        computed_solution: Any,
        exact_solution: Callable,
        norm_type: str = 'L2'
    ) -> float:
        """Compute error between computed and exact solutions.
        
        Parameters
        ----------
        computed_solution : Any
            Computed numerical solution
        exact_solution : Callable
            Exact solution function
        norm_type : str, optional
            Type of norm ('L2', 'H1', 'Linf'), by default 'L2'
            
        Returns
        -------
        float
            Error value
        """
        # Placeholder implementation
        return 0.0
    
    def manufactured_solution(self, **kwargs) -> Dict[str, Callable]:
        """Generate manufactured solution for verification.
        
        Parameters
        ----------
        **kwargs
            Parameters for manufactured solution
            
        Returns
        -------
        Dict[str, Callable]
            Dictionary with 'solution' and 'source' functions
        """
        # Default implementation - subclasses should override
        def solution(x):
            return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        
        def source(x):
            return 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        
        return {'solution': solution, 'source': source}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backend={self.backend}, params={self.params})"


class LinearOperator(BaseOperator):
    """Base class for linear operators."""
    
    _is_linear = True
    
    def apply_matrix(
        self, 
        function_space: Any,
        params: Dict[str, Any] = None
    ) -> Any:
        """Assemble operator as matrix.
        
        Parameters
        ----------
        function_space : Any
            Function space
        params : Dict[str, Any], optional
            Parameters
            
        Returns
        -------
        Any
            Assembled matrix
        """
        # Default implementation for linear operators
        raise NotImplementedError("Matrix assembly not implemented")


class NonlinearOperator(BaseOperator):
    """Base class for nonlinear operators."""
    
    _is_linear = False
    
    def linearize(
        self,
        solution: Any,
        params: Dict[str, Any] = None
    ) -> LinearOperator:
        """Linearize the operator around a solution.
        
        Parameters
        ----------
        solution : Any
            Solution to linearize around
        params : Dict[str, Any], optional
            Parameters
            
        Returns
        -------
        LinearOperator
            Linearized operator
        """
        raise NotImplementedError("Linearization not implemented")
    
    def newton_step(
        self,
        current_solution: Any,
        params: Dict[str, Any] = None
    ) -> Any:
        """Compute Newton step for nonlinear solve.
        
        Parameters
        ----------
        current_solution : Any
            Current solution estimate
        params : Dict[str, Any], optional
            Parameters
            
        Returns
        -------
        Any
            Newton correction
        """
        # Default implementation using linearization
        linear_op = self.linearize(current_solution, params)
        # Would need to solve linear system here
        raise NotImplementedError("Newton step not implemented")
