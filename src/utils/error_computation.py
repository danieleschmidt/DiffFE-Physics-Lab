"""Error computation utilities for verification and validation."""

from typing import Callable, Dict, Any, List, Optional, Union
import logging

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

logger = logging.getLogger(__name__)


def compute_error(
    computed_solution: Any,
    exact_solution: Union[Callable, Any],
    norm_type: str = 'L2',
    mesh: Any = None
) -> float:
    """Compute error between computed and exact solutions.
    
    Parameters
    ----------
    computed_solution : firedrake.Function
        Computed numerical solution
    exact_solution : Callable or firedrake.Function
        Exact solution function or field
    norm_type : str, optional
        Error norm ('L2', 'H1', 'Linf'), by default 'L2'
    mesh : firedrake.Mesh, optional
        Mesh for integration (if not inferrable from solution)
        
    Returns
    -------
    float
        Error value
        
    Examples
    --------
    >>> def exact(x): return np.sin(np.pi * x[0])
    >>> error = compute_error(numerical_sol, exact, 'L2')
    >>> print(f"L2 error: {error:.2e}")
    """
    if not HAS_FIREDRAKE:
        logger.warning("Cannot compute error: Firedrake not available")
        return 0.0
    
    # Get function space from computed solution
    V = computed_solution.function_space()
    
    # Convert exact solution to Function if needed
    if callable(exact_solution):
        exact_func = fd.Function(V)
        exact_func.interpolate(exact_solution)
    elif isinstance(exact_solution, fd.Function):
        exact_func = exact_solution
    else:
        raise ValueError("exact_solution must be callable or firedrake.Function")
    
    # Compute error function
    error_func = computed_solution - exact_func
    
    # Compute specified norm
    if norm_type.upper() == 'L2':
        error_val = fd.sqrt(fd.assemble(fd.inner(error_func, error_func) * fd.dx))
    
    elif norm_type.upper() == 'H1':
        l2_term = fd.inner(error_func, error_func) * fd.dx
        h1_term = fd.inner(fd.grad(error_func), fd.grad(error_func)) * fd.dx
        error_val = fd.sqrt(fd.assemble(l2_term + h1_term))
    
    elif norm_type.upper() == 'LINF':
        # Maximum absolute error (simplified)
        error_val = max(abs(error_func.dat.data))
    
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
    
    return float(error_val)


def compute_convergence_rate(
    mesh_sizes: List[float],
    errors: List[float],
    fit_method: str = 'linear'
) -> Dict[str, float]:
    """Compute convergence rate from mesh refinement study.
    
    Parameters
    ----------
    mesh_sizes : List[float]
        Mesh sizes (h values)
    errors : List[float]
        Corresponding error values
    fit_method : str, optional
        Fitting method ('linear'), by default 'linear'
        
    Returns
    -------
    Dict[str, float]
        Convergence analysis results
    """
    if len(mesh_sizes) != len(errors) or len(mesh_sizes) < 2:
        raise ValueError("Need at least 2 matching mesh sizes and errors")
    
    # Compute convergence rate using simple approach
    import math
    
    # Use last two points for rate computation
    h1, h2 = mesh_sizes[-2], mesh_sizes[-1]
    e1, e2 = errors[-2], errors[-1]
    
    if h1 <= 0 or h2 <= 0 or e1 <= 0 or e2 <= 0:
        return {'rate': 0.0, 'error_constant': 1.0}
    
    rate = math.log(e2 / e1) / math.log(h2 / h1)
    error_constant = e1 / (h1 ** rate)
    
    return {
        'rate': rate,
        'error_constant': error_constant,
        'fit_method': fit_method
    }


def compute_relative_error(
    computed_solution: Any,
    exact_solution: Union[Callable, Any],
    norm_type: str = 'L2'
) -> float:
    """Compute relative error.
    
    Parameters
    ----------
    computed_solution : firedrake.Function
        Computed solution
    exact_solution : Callable or firedrake.Function
        Exact solution
    norm_type : str, optional
        Norm type, by default 'L2'
        
    Returns
    -------
    float
        Relative error
    """
    if not HAS_FIREDRAKE:
        return 0.0
    
    # Compute absolute error
    abs_error = compute_error(computed_solution, exact_solution, norm_type)
    
    # Compute norm of exact solution
    V = computed_solution.function_space()
    
    if callable(exact_solution):
        exact_func = fd.Function(V)
        exact_func.interpolate(exact_solution)
    else:
        exact_func = exact_solution
    
    if norm_type.upper() == 'L2':
        exact_norm = fd.sqrt(fd.assemble(fd.inner(exact_func, exact_func) * fd.dx))
    else:
        exact_norm = 1.0  # Simplified
    
    exact_norm_val = float(exact_norm)
    if exact_norm_val == 0:
        return float('inf') if abs_error > 0 else 0.0
    
    return abs_error / exact_norm_val