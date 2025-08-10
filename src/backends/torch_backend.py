"""PyTorch automatic differentiation backend."""

from typing import Callable, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.autograd import grad
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create dummy torch for type hints when PyTorch is not available
    class DummyTorch:
        class Tensor:
            pass
        float32 = float
        float64 = float  
        int32 = int
        int64 = int
        bool = bool
    torch = DummyTorch()
    # Create dummy functions
    def dummy_func(*args, **kwargs):
        raise ImportError("PyTorch not installed. Install with: pip install torch")
    grad = dummy_func
    # Create dummy nn module
    class DummyNN:
        class Module:
            pass
    nn = DummyNN()

from .base import ADBackend, register_backend


@register_backend('torch')
class TorchBackend(ADBackend):
    """PyTorch-based automatic differentiation backend.
    
    Provides automatic differentiation using PyTorch's autograd system
    with support for GPU acceleration and neural network integration.
    
    Examples
    --------
    >>> backend = TorchBackend()
    >>> grad_func = backend.grad(lambda x: x**2)
    >>> gradient = grad_func(torch.tensor(3.0, requires_grad=True))
    """
    
    def _check_dependencies(self) -> None:
        """Check PyTorch availability."""
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch backend requires PyTorch. Install with: pip install torch"
            )
    
    def grad(
        self, 
        func: Callable, 
        argnums: int = 0,
        has_aux: bool = False
    ) -> Callable:
        """Compute gradient using PyTorch autograd.
        
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
        def grad_func(*args):
            # Convert inputs to tensors with gradient tracking
            torch_args = []
            for i, arg in enumerate(args):
                if i == argnums:
                    if isinstance(arg, torch.Tensor):
                        tensor_arg = arg.clone().detach().requires_grad_(True)
                    else:
                        tensor_arg = torch.tensor(
                            arg, dtype=torch.float32, requires_grad=True
                        )
                    torch_args.append(tensor_arg)
                else:
                    if isinstance(arg, torch.Tensor):
                        torch_args.append(arg)
                    else:
                        torch_args.append(torch.tensor(arg, dtype=torch.float32))
            
            # Compute function value
            if has_aux:
                output, aux = func(*torch_args)
                # Compute gradient
                gradients = grad(
                    output, 
                    torch_args[argnums], 
                    create_graph=True,
                    retain_graph=True
                )[0]
                return gradients, aux
            else:
                output = func(*torch_args)
                # Compute gradient
                gradients = grad(
                    output, 
                    torch_args[argnums], 
                    create_graph=True,
                    retain_graph=True
                )[0]
                return gradients
        
        return grad_func
    
    def jacobian(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute Jacobian using PyTorch.
        
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
        def jacobian_func(*args):
            # Convert to tensors
            torch_args = []
            for i, arg in enumerate(args):
                if i == argnums:
                    if isinstance(arg, torch.Tensor):
                        tensor_arg = arg.clone().detach().requires_grad_(True)
                    else:
                        tensor_arg = torch.tensor(
                            arg, dtype=torch.float32, requires_grad=True
                        )
                    torch_args.append(tensor_arg)
                else:
                    if isinstance(arg, torch.Tensor):
                        torch_args.append(arg)
                    else:
                        torch_args.append(torch.tensor(arg, dtype=torch.float32))
            
            # Compute function
            output = func(*torch_args)
            input_tensor = torch_args[argnums]
            
            if output.dim() == 0:  # Scalar output
                jacobian_matrix = grad(
                    output, input_tensor, create_graph=True
                )[0]
            else:  # Vector output
                jacobian_rows = []
                for i in range(output.shape[0]):
                    grad_i = grad(
                        output[i], 
                        input_tensor, 
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    jacobian_rows.append(grad_i)
                jacobian_matrix = torch.stack(jacobian_rows)
            
            return jacobian_matrix
        
        return jacobian_func
    
    def hessian(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute Hessian using PyTorch.
        
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
        def hessian_func(*args):
            # First compute gradient
            grad_func = self.grad(func, argnums)
            
            # Then compute gradient of gradient
            def scalar_grad_func(*grad_args):
                grad_result = grad_func(*grad_args)
                if grad_result.dim() > 0:
                    # For vector gradient, sum to get scalar
                    return grad_result.sum()
                return grad_result
            
            hessian_func_inner = self.jacobian(scalar_grad_func, argnums)
            return hessian_func_inner(*args)
        
        return hessian_func
    
    def jit(self, func: Callable, **kwargs) -> Callable:
        """JIT compile function using TorchScript.
        
        Parameters
        ----------
        func : Callable
            Function to compile
        **kwargs
            TorchScript compilation options
            
        Returns
        -------
        Callable
            Compiled function
        """
        try:
            # Attempt TorchScript compilation
            traced_func = torch.jit.trace(func, **kwargs)
            return traced_func
        except Exception:
            # Fall back to original function if tracing fails
            return func
    
    def vmap(
        self, 
        func: Callable, 
        in_axes: Any = 0,
        out_axes: Any = 0
    ) -> Callable:
        """Vectorize function using PyTorch vmap.
        
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
        def vmap_func(*args):
            # Simple vectorization using batch dimension
            # More sophisticated implementation would use torch.vmap if available
            if isinstance(in_axes, int):
                batch_size = args[0].shape[in_axes]
                results = []
                
                for i in range(batch_size):
                    # Extract slice for each batch element
                    batch_args = []
                    for arg in args:
                        if isinstance(arg, torch.Tensor):
                            batch_args.append(arg[i])
                        else:
                            batch_args.append(arg)
                    
                    result = func(*batch_args)
                    results.append(result)
                
                # Stack results
                return torch.stack(results, dim=out_axes)
            else:
                # More complex in_axes handling would go here
                raise NotImplementedError("Complex in_axes not implemented")
        
        return vmap_func
    
    def optimize(
        self,
        loss_func: Callable,
        initial_params: Any,
        num_steps: int = 1000,
        learning_rate: float = 0.01,
        optimizer: str = 'adam'
    ) -> Any:
        """Optimize using PyTorch optimizers.
        
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
        # Convert to tensor
        if isinstance(initial_params, torch.Tensor):
            params = initial_params.clone().detach().requires_grad_(True)
        else:
            params = torch.tensor(
                initial_params, dtype=torch.float32, requires_grad=True
            )
        
        # Select optimizer
        if optimizer.lower() == 'sgd':
            opt = torch.optim.SGD([params], lr=learning_rate)
        elif optimizer.lower() == 'adam':
            opt = torch.optim.Adam([params], lr=learning_rate)
        elif optimizer.lower() == 'adamw':
            opt = torch.optim.AdamW([params], lr=learning_rate)
        elif optimizer.lower() == 'lbfgs':
            opt = torch.optim.LBFGS([params], lr=learning_rate, max_iter=20)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Optimization loop
        for step in range(num_steps):
            def closure():
                opt.zero_grad()
                loss = loss_func(params)
                loss.backward()
                return loss
            
            if optimizer.lower() == 'lbfgs':
                opt.step(closure)
            else:
                opt.zero_grad()
                loss = loss_func(params)
                loss.backward()
                opt.step()
        
        return params
    
    def to_array(self, data: Any) -> torch.Tensor:
        """Convert data to PyTorch tensor.
        
        Parameters
        ----------
        data : Any
            Input data
            
        Returns
        -------
        torch.Tensor
            PyTorch tensor
        """
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        else:
            return torch.tensor(data, dtype=torch.float32)
    
    def from_array(self, array: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy.
        
        Parameters
        ----------
        array : torch.Tensor
            PyTorch tensor
            
        Returns
        -------
        np.ndarray
            NumPy array
        """
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        else:
            return np.asarray(array)
    
    def random_normal(
        self, 
        shape: tuple, 
        dtype=torch.float32,
        device=None
    ) -> torch.Tensor:
        """Generate random normal samples.
        
        Parameters
        ----------
        shape : tuple
            Output shape
        dtype : torch.dtype, optional
            Output dtype, by default torch.float32
        device : torch.device, optional
            Device to place tensor on
            
        Returns
        -------
        torch.Tensor
            Random samples
        """
        return torch.randn(shape, dtype=dtype, device=device)
    
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducibility.
        
        Parameters
        ----------
        seed : int
            Random seed
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def get_device(self) -> str:
        """Get available compute device.
        
        Returns
        -------
        str
            Device name ('cuda' or 'cpu')
        """
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def neural_network_wrapper(
        self,
        architecture: list,
        activation: str = 'relu'
    ) -> nn.Module:
        """Create neural network wrapper for physics-informed methods.
        
        Parameters
        ----------
        architecture : list
            List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        activation : str, optional
            Activation function, by default 'relu'
            
        Returns
        -------
        nn.Module
            Neural network model
        """
        layers = []
        
        # Activation function mapping
        act_funcs = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'gelu': nn.GELU,
            'swish': nn.SiLU
        }
        
        if activation not in act_funcs:
            raise ValueError(f"Unknown activation: {activation}")
        
        act_func = act_funcs[activation]
        
        # Build layers
        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i + 1]))
            if i < len(architecture) - 2:  # No activation on output layer
                layers.append(act_func())
        
        return nn.Sequential(*layers)
    
    def physics_informed_loss(
        self,
        network: nn.Module,
        pde_residual_func: Callable,
        boundary_func: Callable,
        collocation_points: torch.Tensor,
        boundary_points: torch.Tensor,
        pde_weight: float = 1.0,
        bc_weight: float = 1.0
    ) -> Callable:
        """Create physics-informed loss function.
        
        Parameters
        ----------
        network : nn.Module
            Neural network
        pde_residual_func : Callable
            PDE residual function
        boundary_func : Callable
            Boundary condition function
        collocation_points : torch.Tensor
            Interior collocation points
        boundary_points : torch.Tensor
            Boundary points
        pde_weight : float, optional
            PDE loss weight, by default 1.0
        bc_weight : float, optional
            Boundary condition weight, by default 1.0
            
        Returns
        -------
        Callable
            Physics-informed loss function
        """
        def loss_func():
            # PDE residual loss
            u_pred = network(collocation_points)
            pde_residual = pde_residual_func(collocation_points, u_pred, network)
            pde_loss = torch.mean(pde_residual**2)
            
            # Boundary condition loss
            u_boundary = network(boundary_points)
            bc_residual = boundary_func(boundary_points, u_boundary)
            bc_loss = torch.mean(bc_residual**2)
            
            # Combined loss
            total_loss = pde_weight * pde_loss + bc_weight * bc_loss
            
            return total_loss
        
        return loss_func
    
    def __repr__(self) -> str:
        device = self.get_device()
        return f"TorchBackend(device={device})"


# Make PyTorch backend available if PyTorch is installed
if HAS_TORCH:
    # Register some common activation functions
    torch.nn.functional.swish = lambda x: x * torch.sigmoid(x)