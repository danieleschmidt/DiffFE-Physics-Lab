"""Base automatic differentiation backend interface."""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

# Global backend registry and default
_BACKEND_REGISTRY: Dict[str, type] = {}
_DEFAULT_BACKEND: str = "jax"


def register_backend(name: str):
    """Decorator to register a backend class.

    Parameters
    ----------
    name : str
        Backend name

    Returns
    -------
    Callable
        Decorator function
    """

    def decorator(cls):
        _BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(backend_name: str = None) -> "ADBackend":
    """Get backend instance by name.

    Parameters
    ----------
    backend_name : str, optional
        Backend name, uses default if None

    Returns
    -------
    ADBackend
        Backend instance

    Raises
    ------
    ValueError
        If backend not found or not available
    """
    if backend_name is None:
        backend_name = _DEFAULT_BACKEND

    if backend_name not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(f"Backend '{backend_name}' not found. Available: {available}")

    backend_cls = _BACKEND_REGISTRY[backend_name]

    try:
        return backend_cls()
    except ImportError as e:
        raise ImportError(f"Backend '{backend_name}' dependencies not available: {e}")


def set_default_backend(backend_name: str) -> None:
    """Set the default backend.

    Parameters
    ----------
    backend_name : str
        Backend name

    Raises
    ------
    ValueError
        If backend not found
    """
    global _DEFAULT_BACKEND

    if backend_name not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(f"Backend '{backend_name}' not found. Available: {available}")

    # Test that backend is actually available
    try:
        get_backend(backend_name)
        _DEFAULT_BACKEND = backend_name
    except ImportError as e:
        raise ImportError(f"Cannot set default backend '{backend_name}': {e}")


def list_backends() -> Dict[str, bool]:
    """List all registered backends and their availability.

    Returns
    -------
    Dict[str, bool]
        Dictionary mapping backend names to availability status
    """
    status = {}
    for name in _BACKEND_REGISTRY:
        try:
            get_backend(name)
            status[name] = True
        except ImportError:
            status[name] = False
    return status


class ADBackend(ABC):
    """Abstract base class for automatic differentiation backends.

    This class defines the interface that all AD backends must implement
    to provide consistent gradient computation across different frameworks.
    """

    def __init__(self):
        self._check_dependencies()
        self.name = self.__class__.__name__.replace("Backend", "").lower()

    @abstractmethod
    def _check_dependencies(self) -> None:
        """Check that required dependencies are available.

        Raises
        ------
        ImportError
            If dependencies are missing
        """
        pass

    @abstractmethod
    def grad(self, func: Callable, argnums: int = 0, has_aux: bool = False) -> Callable:
        """Compute gradient of a function.

        Parameters
        ----------
        func : Callable
            Function to differentiate
        argnums : int, optional
            Argument number to differentiate with respect to, by default 0
        has_aux : bool, optional
            Whether function returns auxiliary data, by default False

        Returns
        -------
        Callable
            Gradient function
        """
        pass

    @abstractmethod
    def jacobian(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute Jacobian of a function.

        Parameters
        ----------
        func : Callable
            Function to differentiate
        argnums : int, optional
            Argument number to differentiate with respect to, by default 0

        Returns
        -------
        Callable
            Jacobian function
        """
        pass

    @abstractmethod
    def hessian(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute Hessian of a function.

        Parameters
        ----------
        func : Callable
            Function to differentiate
        argnums : int, optional
            Argument number to differentiate with respect to, by default 0

        Returns
        -------
        Callable
            Hessian function
        """
        pass

    @abstractmethod
    def jit(self, func: Callable, **kwargs) -> Callable:
        """Just-in-time compile a function.

        Parameters
        ----------
        func : Callable
            Function to compile
        **kwargs
            Backend-specific compilation options

        Returns
        -------
        Callable
            Compiled function
        """
        pass

    @abstractmethod
    def vmap(self, func: Callable, in_axes: Any = 0, out_axes: Any = 0) -> Callable:
        """Vectorize a function over leading array axes.

        Parameters
        ----------
        func : Callable
            Function to vectorize
        in_axes : Any, optional
            Input axes to vectorize over, by default 0
        out_axes : Any, optional
            Output axes to vectorize over, by default 0

        Returns
        -------
        Callable
            Vectorized function
        """
        pass

    def optimize(
        self,
        loss_func: Callable,
        initial_params: Any,
        num_steps: int = 1000,
        learning_rate: float = 0.01,
        optimizer: str = "adam",
    ) -> Any:
        """Optimize parameters using gradient descent.

        Parameters
        ----------
        loss_func : Callable
            Loss function to minimize
        initial_params : Any
            Initial parameter values
        num_steps : int, optional
            Number of optimization steps, by default 1000
        learning_rate : float, optional
            Learning rate, by default 0.01
        optimizer : str, optional
            Optimizer type ('sgd', 'adam'), by default 'adam'

        Returns
        -------
        Any
            Optimized parameters
        """
        # Default implementation using basic gradient descent
        grad_func = self.grad(loss_func)
        params = initial_params

        # Simple SGD implementation
        for step in range(num_steps):
            grads = grad_func(params)

            if optimizer == "sgd":
                # Simple SGD update
                params = self._sgd_update(params, grads, learning_rate)
            elif optimizer == "adam":
                # Would need to implement Adam with momentum
                warnings.warn("Adam optimizer not implemented, falling back to SGD")
                params = self._sgd_update(params, grads, learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")

        return params

    def _sgd_update(self, params: Any, grads: Any, lr: float) -> Any:
        """Simple SGD parameter update.

        Should be overridden by backends for better performance.
        """
        # This is a placeholder - real implementations would be backend-specific
        return params  # Simplified

    @abstractmethod
    def to_array(self, data: Any) -> Any:
        """Convert data to backend array format.

        Parameters
        ----------
        data : Any
            Input data

        Returns
        -------
        Any
            Backend array
        """
        pass

    @abstractmethod
    def from_array(self, array: Any) -> Any:
        """Convert backend array to standard format.

        Parameters
        ----------
        array : Any
            Backend array

        Returns
        -------
        Any
            Standard format (usually numpy)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
