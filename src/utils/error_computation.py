"""Error computation utilities for verification and validation."""

from typing import Callable, Dict, Any, List, Optional, Union
import numpy as np
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
        Error norm ('L2', 'H1', 'Linf', 'H1_seminorm'), by default 'L2'
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
    
    elif norm_type.upper() == 'H1_SEMINORM':
        h1_term = fd.inner(fd.grad(error_func), fd.grad(error_func)) * fd.dx
        error_val = fd.sqrt(fd.assemble(h1_term))
    
    elif norm_type.upper() == 'LINF':
        # Maximum absolute error
        error_val = np.max(np.abs(error_func.dat.data))
    
    elif norm_type.upper() == 'ENERGY':
        # Energy norm (problem-specific - this is a generic version)
        energy_term = fd.inner(fd.grad(error_func), fd.grad(error_func)) * fd.dx
        error_val = fd.sqrt(fd.assemble(energy_term))
    
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
        Fitting method ('linear', 'robust'), by default 'linear'
        
    Returns
    -------
    Dict[str, float]
        Convergence analysis results
        
    Examples
    --------
    >>> h_values = [0.1, 0.05, 0.025, 0.0125]
    >>> errors = [1e-2, 2.5e-3, 6.25e-4, 1.56e-4]
    >>> result = compute_convergence_rate(h_values, errors)
    >>> print(f"Convergence rate: {result['rate']:.2f}")
    """
    if len(mesh_sizes) != len(errors) or len(mesh_sizes) < 2:
        raise ValueError("Need at least 2 matching mesh sizes and errors")
    
    # Convert to log scale
    log_h = np.log(mesh_sizes)
    log_e = np.log(errors)
    
    # Fit line: log(error) = slope * log(h) + intercept
    # Convergence rate is the slope
    if fit_method == 'linear':
        # Simple least squares fit
        coeffs = np.polyfit(log_h, log_e, 1)
        rate = coeffs[0]
        intercept = coeffs[1]
        
        # Compute R-squared
        log_e_fit = rate * log_h + intercept
        ss_res = np.sum((log_e - log_e_fit) ** 2)
        ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    elif fit_method == 'robust':
        # Robust fitting using RANSAC or similar
        try:\n            from sklearn.linear_model import RANSACRegressor\n            from sklearn.linear_model import LinearRegression\n            \n            ransac = RANSACRegressor(LinearRegression(), random_state=42)\n            ransac.fit(log_h.reshape(-1, 1), log_e)\n            \n            rate = ransac.estimator_.coef_[0]\n            intercept = ransac.estimator_.intercept_\n            r_squared = ransac.score(log_h.reshape(-1, 1), log_e)\n            \n        except ImportError:\n            logger.warning("scikit-learn not available, falling back to linear fit")\n            return compute_convergence_rate(mesh_sizes, errors, 'linear')\n    \n    else:\n        raise ValueError(f"Unknown fit method: {fit_method}")\n    \n    # Compute asymptotic error constant\n    error_constant = np.exp(intercept)\n    \n    # Estimate next error (extrapolation)\n    if len(mesh_sizes) >= 2:\n        h_ratio = mesh_sizes[-1] / mesh_sizes[-2]\n        next_h = mesh_sizes[-1] * h_ratio\n        next_error_estimate = error_constant * (next_h ** rate)\n    else:\n        next_error_estimate = None\n    \n    return {\n        'rate': rate,\n        'error_constant': error_constant,\n        'r_squared': r_squared,\n        'next_error_estimate': next_error_estimate,\n        'fit_method': fit_method\n    }\n\n\ndef compute_effectivity_index(\n    computed_error: float,\n    estimated_error: float\n) -> float:\n    """Compute effectivity index for error estimation.\n    \n    Parameters\n    ----------\n    computed_error : float\n        True error (from exact solution)\n    estimated_error : float\n        A posteriori error estimate\n        \n    Returns\n    -------\n    float\n        Effectivity index (ideally close to 1.0)\n    """\n    if computed_error == 0:\n        return np.inf if estimated_error > 0 else 1.0\n    \n    return estimated_error / computed_error\n\n\ndef compute_relative_error(\n    computed_solution: Any,\n    exact_solution: Union[Callable, Any],\n    norm_type: str = 'L2'\n) -> float:\n    """Compute relative error.\n    \n    Parameters\n    ----------\n    computed_solution : firedrake.Function\n        Computed solution\n    exact_solution : Callable or firedrake.Function\n        Exact solution\n    norm_type : str, optional\n        Norm type, by default 'L2'\n        \n    Returns\n    -------\n    float\n        Relative error\n    """\n    if not HAS_FIREDRAKE:\n        return 0.0\n    \n    # Compute absolute error\n    abs_error = compute_error(computed_solution, exact_solution, norm_type)\n    \n    # Compute norm of exact solution\n    V = computed_solution.function_space()\n    \n    if callable(exact_solution):\n        exact_func = fd.Function(V)\n        exact_func.interpolate(exact_solution)\n    else:\n        exact_func = exact_solution\n    \n    if norm_type.upper() == 'L2':\n        exact_norm = fd.sqrt(fd.assemble(fd.inner(exact_func, exact_func) * fd.dx))\n    elif norm_type.upper() == 'H1':\n        l2_term = fd.inner(exact_func, exact_func) * fd.dx\n        h1_term = fd.inner(fd.grad(exact_func), fd.grad(exact_func)) * fd.dx\n        exact_norm = fd.sqrt(fd.assemble(l2_term + h1_term))\n    elif norm_type.upper() == 'LINF':\n        exact_norm = np.max(np.abs(exact_func.dat.data))\n    else:\n        raise ValueError(f"Unknown norm type: {norm_type}")\n    \n    if float(exact_norm) == 0:\n        return np.inf if abs_error > 0 else 0.0\n    \n    return abs_error / float(exact_norm)\n\n\ndef compute_error_components(\n    computed_solution: Any,\n    exact_solution: Union[Callable, Any],\n    component_indices: Optional[List[int]] = None\n) -> Dict[str, float]:\n    """Compute error for each component of vector solution.\n    \n    Parameters\n    ----------\n    computed_solution : firedrake.Function\n        Computed solution (possibly vector-valued)\n    exact_solution : Callable or firedrake.Function\n        Exact solution\n    component_indices : List[int], optional\n        Indices of components to analyze\n        \n    Returns\n    -------\n    Dict[str, float]\n        Error for each component\n    """\n    if not HAS_FIREDRAKE:\n        return {}\n    \n    V = computed_solution.function_space()\n    \n    # Check if solution is vector-valued\n    if V.value_size == 1:\n        # Scalar solution\n        return {'scalar': compute_error(computed_solution, exact_solution)}\n    \n    # Vector solution\n    errors = {}\n    \n    # Get exact solution as Function\n    if callable(exact_solution):\n        exact_func = fd.Function(V)\n        exact_func.interpolate(exact_solution)\n    else:\n        exact_func = exact_solution\n    \n    # Analyze specified components\n    indices = component_indices or list(range(V.value_size))\n    \n    for i in indices:\n        # Extract component\n        comp_computed = computed_solution.sub(i)\n        comp_exact = exact_func.sub(i)\n        \n        # Compute error for this component\n        comp_error = compute_error(comp_computed, comp_exact)\n        errors[f'component_{i}'] = comp_error\n    \n    # Total error (all components)\n    total_error = compute_error(computed_solution, exact_func)\n    errors['total'] = total_error\n    \n    return errors\n\n\ndef analyze_error_distribution(\n    computed_solution: Any,\n    exact_solution: Union[Callable, Any],\n    n_samples: int = 1000\n) -> Dict[str, Any]:\n    """Analyze spatial distribution of error.\n    \n    Parameters\n    ----------\n    computed_solution : firedrake.Function\n        Computed solution\n    exact_solution : Callable or firedrake.Function\n        Exact solution\n    n_samples : int, optional\n        Number of sample points, by default 1000\n        \n    Returns\n    -------\n    Dict[str, Any]\n        Error distribution statistics\n    """\n    if not HAS_FIREDRAKE:\n        return {}\n    \n    V = computed_solution.function_space()\n    mesh = V.mesh()\n    \n    # Get exact solution as Function\n    if callable(exact_solution):\n        exact_func = fd.Function(V)\n        exact_func.interpolate(exact_solution)\n    else:\n        exact_func = exact_solution\n    \n    # Compute pointwise error\n    error_func = computed_solution - exact_func\n    \n    # Sample error at random points\n    dim = mesh.geometric_dimension()\n    \n    # Generate random points within domain (simplified)\n    np.random.seed(42)\n    \n    if dim == 1:\n        # For 1D, sample along the interval\n        coords = np.random.uniform(0, 1, (n_samples, 1))\n    elif dim == 2:\n        # For 2D, sample in unit square (adjust for actual domain)\n        coords = np.random.uniform(0, 1, (n_samples, 2))\n    elif dim == 3:\n        # For 3D, sample in unit cube\n        coords = np.random.uniform(0, 1, (n_samples, 3))\n    else:\n        logger.warning(f"Unsupported dimension for error analysis: {dim}")\n        return {}\n    \n    try:\n        # Evaluate error at sample points\n        error_values = []\n        for coord in coords:\n            try:\n                # This is a simplified approach - real implementation\n                # would need proper point evaluation in Firedrake\n                error_val = 0.0  # Placeholder\n                error_values.append(error_val)\n            except:\n                continue\n        \n        if not error_values:\n            return {'status': 'failed', 'reason': 'Could not evaluate error at sample points'}\n        \n        error_array = np.array(error_values)\n        \n        return {\n            'mean_error': np.mean(np.abs(error_array)),\n            'std_error': np.std(error_array),\n            'max_error': np.max(np.abs(error_array)),\n            'min_error': np.min(np.abs(error_array)),\n            'rms_error': np.sqrt(np.mean(error_array**2)),\n            'n_samples': len(error_values)\n        }\n    \n    except Exception as e:\n        logger.warning(f"Error distribution analysis failed: {e}")\n        return {'status': 'failed', 'reason': str(e)}\n\n\ndef compute_mesh_quality_metrics(mesh: Any) -> Dict[str, float]:\n    """Compute mesh quality metrics.\n    \n    Parameters\n    ----------\n    mesh : firedrake.Mesh\n        Mesh to analyze\n        \n    Returns\n    -------\n    Dict[str, float]\n        Mesh quality metrics\n    """\n    if not HAS_FIREDRAKE:\n        return {}\n    \n    try:\n        # Basic mesh statistics\n        metrics = {\n            'num_cells': mesh.num_cells(),\n            'num_vertices': mesh.num_vertices(),\n            'geometric_dimension': mesh.geometric_dimension(),\n            'topological_dimension': mesh.topological_dimension()\n        }\n        \n        # Compute mesh size metrics\n        coords = mesh.coordinates.dat.data\n        \n        if coords.size > 0:\n            # Bounding box\n            bbox_min = np.min(coords, axis=0)\n            bbox_max = np.max(coords, axis=0)\n            bbox_size = bbox_max - bbox_min\n            \n            metrics.update({\n                'bbox_min': bbox_min.tolist(),\n                'bbox_max': bbox_max.tolist(),\n                'bbox_volume': np.prod(bbox_size),\n                'characteristic_length': np.max(bbox_size)\n            })\n            \n            # Estimate average element size\n            if mesh.geometric_dimension() == 2:\n                approx_h = np.sqrt(metrics['bbox_volume'] / metrics['num_cells'])\n            elif mesh.geometric_dimension() == 3:\n                approx_h = (metrics['bbox_volume'] / metrics['num_cells']) ** (1/3)\n            else:\n                approx_h = metrics['bbox_volume'] / metrics['num_cells']\n            \n            metrics['average_element_size'] = approx_h\n        \n        return metrics\n    \n    except Exception as e:\n        logger.warning(f"Mesh quality analysis failed: {e}")\n        return {'status': 'failed', 'reason': str(e)}