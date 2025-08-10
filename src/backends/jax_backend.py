"""JAX automatic differentiation backend."""

from typing import Callable, Any
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jacobian, hessian, jit, vmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    # Create dummy jax.numpy for type hints when JAX is not available
    class DummyJNP:
        ndarray = np.ndarray
        float32 = np.float32
        float64 = np.float64
        int32 = np.int32
        int64 = np.int64
    jnp = DummyJNP()
    # Create dummy functions
    def dummy_func(*args, **kwargs):
        raise ImportError("JAX not installed. Install with: pip install jax jaxlib")
    grad = jacobian = hessian = jit = vmap = dummy_func

from .base import ADBackend, register_backend


@register_backend('jax')
class JAXBackend(ADBackend):
    """JAX-based automatic differentiation backend.
    
    Provides high-performance automatic differentiation using JAX
    with XLA compilation and GPU support.
    
    Examples
    --------
    >>> backend = JAXBackend()
    >>> grad_func = backend.grad(lambda x: x**2)
    >>> gradient = grad_func(3.0)  # Returns 6.0
    """
    
    def _check_dependencies(self) -> None:
        """Check JAX availability."""
        if not HAS_JAX:
            raise ImportError(
                "JAX backend requires JAX. Install with: pip install jax jaxlib"
            )
    
    def grad(
        self, 
        func: Callable, 
        argnums: int = 0,
        has_aux: bool = False
    ) -> Callable:
        """Compute gradient using JAX.
        
        Parameters
        ----------
        func : Callable
            Function to differentiate
        argnums : int, optional
            Argument to differentiate with respect to, by default 0
        has_aux : bool, optional
            Whether function returns auxiliary data, by default False
            
        Returns
        -------
        Callable
            Gradient function
        """
        return grad(func, argnums=argnums, has_aux=has_aux)
    
    def jacobian(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute Jacobian using JAX.
        
        Parameters
        ----------
        func : Callable
            Function to differentiate
        argnums : int, optional
            Argument to differentiate with respect to, by default 0
            
        Returns
        -------
        Callable
            Jacobian function
        """
        return jacobian(func, argnums=argnums)
    
    def hessian(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute Hessian using JAX.
        
        Parameters
        ----------
        func : Callable
            Function to differentiate
        argnums : int, optional
            Argument to differentiate with respect to, by default 0
            
        Returns
        -------
        Callable
            Hessian function
        """
        return hessian(func, argnums=argnums)
    
    def jit(self, func: Callable, **kwargs) -> Callable:
        """JIT compile function using JAX.
        
        Parameters
        ----------
        func : Callable
            Function to compile
        **kwargs
            JAX-specific compilation options
            
        Returns
        -------
        Callable
            Compiled function
        """
        return jit(func, **kwargs)
    
    def vmap(
        self, 
        func: Callable, 
        in_axes: Any = 0,
        out_axes: Any = 0
    ) -> Callable:
        """Vectorize function using JAX vmap.
        
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
        return vmap(func, in_axes=in_axes, out_axes=out_axes)
    
    def optimize(
        self,
        loss_func: Callable,
        initial_params: Any,
        num_steps: int = 1000,
        learning_rate: float = 0.01,
        optimizer: str = 'adam'
    ) -> Any:
        """Optimize using JAX-based optimizers.
        
        Parameters
        ----------
        loss_func : Callable
            Loss function to minimize
        initial_params : Any
            Initial parameters
        num_steps : int, optional
            Number of steps, by default 1000
        learning_rate : float, optional
            Learning rate, by default 0.01
        optimizer : str, optional
            Optimizer type, by default 'adam'
            
        Returns
        -------
        Any
            Optimized parameters
        """
        # Try to use optax if available, otherwise fall back to simple SGD
        try:
            import optax
            return self._optax_optimize(
                loss_func, initial_params, num_steps, learning_rate, optimizer
            )
        except ImportError:
            # Fall back to simple implementation
            return self._simple_optimize(
                loss_func, initial_params, num_steps, learning_rate, optimizer
            )
    
    def _optax_optimize(
        self,
        loss_func: Callable,
        initial_params: Any,
        num_steps: int,
        learning_rate: float,
        optimizer: str
    ) -> Any:
        """Optimize using optax optimizers."""
        import optax
        
        # Select optimizer
        if optimizer == 'sgd':
            opt = optax.sgd(learning_rate)
        elif optimizer == 'adam':
            opt = optax.adam(learning_rate)
        elif optimizer == 'adamw':
            opt = optax.adamw(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Initialize optimizer state
        opt_state = opt.init(initial_params)
        params = initial_params
        
        # Gradient function
        grad_func = self.grad(loss_func)
        
        # Optimization loop
        for step in range(num_steps):
            grads = grad_func(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        
        return params
    
    def _simple_optimize(
        self,
        loss_func: Callable,
        initial_params: Any,
        num_steps: int,
        learning_rate: float,
        optimizer: str
    ) -> Any:
        """Simple optimization without optax."""
        grad_func = self.grad(loss_func)
        params = initial_params
        
        # Simple momentum for Adam approximation
        if optimizer == 'adam':
            m = jnp.zeros_like(params)  # First moment
            v = jnp.zeros_like(params)  # Second moment
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            
            for step in range(num_steps):
                grads = grad_func(params)
                
                # Update biased first and second moments
                m = beta1 * m + (1 - beta1) * grads
                v = beta2 * v + (1 - beta2) * grads**2
                
                # Bias correction
                m_hat = m / (1 - beta1**(step + 1))
                v_hat = v / (1 - beta2**(step + 1))
                
                # Update parameters
                params = params - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        else:
            # Simple SGD
            for step in range(num_steps):
                grads = grad_func(params)
                params = params - learning_rate * grads
        
        return params
    
    def to_array(self, data: Any) -> jnp.ndarray:
        """Convert data to JAX array.
        
        Parameters
        ----------
        data : Any
            Input data
            
        Returns
        -------
        jnp.ndarray
            JAX array
        """
        return jnp.asarray(data)
    
    def from_array(self, array: jnp.ndarray) -> np.ndarray:
        """Convert JAX array to numpy.
        
        Parameters
        ----------
        array : jnp.ndarray
            JAX array
            
        Returns
        -------
        np.ndarray
            NumPy array
        """
        return np.asarray(array)
    
    def random_normal(
        self, 
        key: Any, 
        shape: tuple, 
        dtype=jnp.float32
    ) -> jnp.ndarray:
        """Generate random normal samples.
        
        Parameters
        ----------
        key : Any
            JAX random key
        shape : tuple
            Output shape
        dtype : dtype, optional
            Output dtype, by default jnp.float32
            
        Returns
        -------
        jnp.ndarray
            Random samples
        """
        return jax.random.normal(key, shape, dtype=dtype)
    
    def make_random_key(self, seed: int = 0) -> Any:
        """Create JAX random key.
        
        Parameters
        ----------
        seed : int, optional
            Random seed, by default 0
            
        Returns
        -------
        Any
            JAX random key
        """
        return jax.random.PRNGKey(seed)


# Make JAX the default backend if available
if HAS_JAX:
    from .base import _DEFAULT_BACKEND
    if _DEFAULT_BACKEND == 'jax':
        pass  # Already default
