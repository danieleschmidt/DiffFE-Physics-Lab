"""Manufactured solution generators for verification."""

from typing import Dict, Callable, Any, Optional, Tuple
import math

# Handle optional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Provide fallback implementations
    class NumpyFallback:
        @staticmethod
        def array(data):
            return list(data) if hasattr(data, '__iter__') else [data]
        
        @staticmethod
        def sin(x):
            return math.sin(x)
        
        @staticmethod
        def cos(x):
            return math.cos(x)
            
        @staticmethod
        def exp(x):
            return math.exp(x)
        
        @property
        def pi(self):
            return math.pi
        
        @staticmethod
        def isnan(x):
            return x != x
        
        @staticmethod
        def isinf(x):
            return x == float('inf') or x == float('-inf')
        
        @staticmethod
        def any(arr):
            return any(arr)
    
    np = NumpyFallback()
    np.pi = math.pi

try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback logger
    class Logger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    logger = Logger()

from enum import Enum


class SolutionType(Enum):
    """Types of manufactured solutions."""
    POLYNOMIAL = "polynomial"
    TRIGONOMETRIC = "trigonometric"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    MIXED = "mixed"


def generate_manufactured_solution(
    solution_type: SolutionType = SolutionType.TRIGONOMETRIC,
    dimension: int = 2,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Callable]:
    """Generate manufactured solution for verification studies.
    
    Creates analytical solution and corresponding source term for
    testing finite element implementations.
    
    Parameters
    ----------
    solution_type : SolutionType, optional
        Type of manufactured solution, by default TRIGONOMETRIC
    dimension : int, optional
        Spatial dimension (1, 2, or 3), by default 2
    parameters : Dict[str, Any], optional
        Solution parameters
        
    Returns
    -------
    Dict[str, Callable]
        Dictionary containing 'solution', 'source', 'gradient', 'laplacian' functions
        
    Examples
    --------
    >>> mms = generate_manufactured_solution(SolutionType.TRIGONOMETRIC, dimension=2)
    >>> u_exact = mms['solution']
    >>> f_source = mms['source']
    >>> # Use in FEM problem: -∇²u = f with u = u_exact on boundary
    """
    params = parameters or {}
    
    if solution_type == SolutionType.POLYNOMIAL:
        return _polynomial_solution(dimension, params)
    elif solution_type == SolutionType.TRIGONOMETRIC:
        return _trigonometric_solution(dimension, params)
    elif solution_type == SolutionType.EXPONENTIAL:
        return _exponential_solution(dimension, params)
    elif solution_type == SolutionType.GAUSSIAN:
        return _gaussian_solution(dimension, params)
    elif solution_type == SolutionType.MIXED:
        return _mixed_solution(dimension, params)
    else:
        raise ValueError(f"Unknown solution type: {solution_type}")


def _polynomial_solution(dimension: int, params: Dict[str, Any]) -> Dict[str, Callable]:
    """Generate polynomial manufactured solution."""
    degree = params.get('degree', 2)
    coeffs = params.get('coefficients', None)
    
    if dimension == 1:
        if coeffs is None:
            coeffs = [1.0, 2.0, 1.0]  # u = x² + 2x + 1
        
        def solution(x):
            result = 0.0
            for i, c in enumerate(coeffs):
                result += c * (x[0] ** i)
            return result
        
        def gradient(x):
            grad = 0.0
            for i, c in enumerate(coeffs[1:], 1):
                grad += i * c * (x[0] ** (i-1))
            return np.array([grad])
        
        def source(x):
            # Second derivative
            lapl = 0.0
            for i, c in enumerate(coeffs[2:], 2):
                lapl += i * (i-1) * c * (x[0] ** (i-2))
            return -lapl  # For -∇²u = f
    
    elif dimension == 2:
        if coeffs is None:
            # u = x² + y² + xy
            coeffs = {'x0': 0, 'x1': 0, 'y1': 0, 'x2': 1, 'y2': 1, 'xy': 1}
        
        def solution(x):
            return (coeffs.get('x0', 0) + 
                   coeffs.get('x1', 0) * x[0] + 
                   coeffs.get('y1', 0) * x[1] +
                   coeffs.get('x2', 0) * x[0]**2 + 
                   coeffs.get('y2', 0) * x[1]**2 + 
                   coeffs.get('xy', 0) * x[0] * x[1])
        
        def gradient(x):
            dudx = (coeffs.get('x1', 0) + 
                   2 * coeffs.get('x2', 0) * x[0] + 
                   coeffs.get('xy', 0) * x[1])
            dudy = (coeffs.get('y1', 0) + 
                   2 * coeffs.get('y2', 0) * x[1] + 
                   coeffs.get('xy', 0) * x[0])
            return np.array([dudx, dudy])
        
        def source(x):
            # Laplacian
            lapl = (2 * coeffs.get('x2', 0) + 2 * coeffs.get('y2', 0))
            return -lapl
    
    elif dimension == 3:
        if coeffs is None:
            # u = x² + y² + z² + xyz
            coeffs = {'x2': 1, 'y2': 1, 'z2': 1, 'xyz': 1}
        
        def solution(x):
            return (coeffs.get('x2', 0) * x[0]**2 + 
                   coeffs.get('y2', 0) * x[1]**2 +
                   coeffs.get('z2', 0) * x[2]**2 + 
                   coeffs.get('xyz', 0) * x[0] * x[1] * x[2])
        
        def gradient(x):
            dudx = 2 * coeffs.get('x2', 0) * x[0] + coeffs.get('xyz', 0) * x[1] * x[2]
            dudy = 2 * coeffs.get('y2', 0) * x[1] + coeffs.get('xyz', 0) * x[0] * x[2]
            dudz = 2 * coeffs.get('z2', 0) * x[2] + coeffs.get('xyz', 0) * x[0] * x[1]
            return np.array([dudx, dudy, dudz])
        
        def source(x):
            lapl = (2 * coeffs.get('x2', 0) + 2 * coeffs.get('y2', 0) + 2 * coeffs.get('z2', 0))
            return -lapl
    
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")
    
    def laplacian(x):
        return -source(x)
    
    return {
        'solution': solution,
        'source': source,
        'gradient': gradient,
        'laplacian': laplacian,
        'type': 'polynomial',
        'degree': degree
    }


def _trigonometric_solution(dimension: int, params: Dict[str, Any]) -> Dict[str, Callable]:
    """Generate trigonometric manufactured solution."""
    frequency = params.get('frequency', 1.0)
    amplitude = params.get('amplitude', 1.0)
    phase = params.get('phase', 0.0)
    
    if dimension == 1:
        def solution(x):
            return amplitude * np.sin(frequency * np.pi * x[0] + phase)
        
        def gradient(x):
            dudx = amplitude * frequency * np.pi * np.cos(frequency * np.pi * x[0] + phase)
            return np.array([dudx])
        
        def source(x):
            # -d²u/dx²
            return amplitude * (frequency * np.pi)**2 * np.sin(frequency * np.pi * x[0] + phase)
    
    elif dimension == 2:
        def solution(x):
            return amplitude * np.sin(frequency * np.pi * x[0] + phase) * np.sin(frequency * np.pi * x[1] + phase)
        
        def gradient(x):
            sin_x = np.sin(frequency * np.pi * x[0] + phase)
            sin_y = np.sin(frequency * np.pi * x[1] + phase)
            cos_x = np.cos(frequency * np.pi * x[0] + phase)
            cos_y = np.cos(frequency * np.pi * x[1] + phase)
            
            dudx = amplitude * frequency * np.pi * cos_x * sin_y
            dudy = amplitude * frequency * np.pi * sin_x * cos_y
            return np.array([dudx, dudy])
        
        def source(x):
            # -∇²u = -(-∂²u/∂x² - ∂²u/∂y²)
            return 2 * amplitude * (frequency * np.pi)**2 * np.sin(frequency * np.pi * x[0] + phase) * np.sin(frequency * np.pi * x[1] + phase)
    
    elif dimension == 3:
        def solution(x):
            return (amplitude * 
                   np.sin(frequency * np.pi * x[0] + phase) * 
                   np.sin(frequency * np.pi * x[1] + phase) * 
                   np.sin(frequency * np.pi * x[2] + phase))
        
        def gradient(x):
            sin_x = np.sin(frequency * np.pi * x[0] + phase)
            sin_y = np.sin(frequency * np.pi * x[1] + phase)
            sin_z = np.sin(frequency * np.pi * x[2] + phase)
            cos_x = np.cos(frequency * np.pi * x[0] + phase)
            cos_y = np.cos(frequency * np.pi * x[1] + phase)
            cos_z = np.cos(frequency * np.pi * x[2] + phase)
            
            dudx = amplitude * frequency * np.pi * cos_x * sin_y * sin_z
            dudy = amplitude * frequency * np.pi * sin_x * cos_y * sin_z
            dudz = amplitude * frequency * np.pi * sin_x * sin_y * cos_z
            return np.array([dudx, dudy, dudz])
        
        def source(x):
            return 3 * amplitude * (frequency * np.pi)**2 * solution(x)
    
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")
    
    def laplacian(x):
        return -source(x)
    
    return {
        'solution': solution,
        'source': source,
        'gradient': gradient,
        'laplacian': laplacian,
        'type': 'trigonometric',
        'frequency': frequency,
        'amplitude': amplitude
    }


def _exponential_solution(dimension: int, params: Dict[str, Any]) -> Dict[str, Callable]:
    """Generate exponential manufactured solution."""
    decay_rate = params.get('decay_rate', 1.0)
    amplitude = params.get('amplitude', 1.0)
    center = params.get('center', [0.5] * dimension)
    
    if len(center) != dimension:
        center = [0.5] * dimension
    
    def solution(x):
        r_squared = sum((x[i] - center[i])**2 for i in range(dimension))
        return amplitude * np.exp(-decay_rate * r_squared)
    
    def gradient(x):
        r_squared = sum((x[i] - center[i])**2 for i in range(dimension))
        exp_term = amplitude * np.exp(-decay_rate * r_squared)
        
        grad = []
        for i in range(dimension):
            grad_i = -2 * decay_rate * (x[i] - center[i]) * exp_term
            grad.append(grad_i)
        
        return np.array(grad)
    
    def source(x):
        r_squared = sum((x[i] - center[i])**2 for i in range(dimension))
        exp_term = amplitude * np.exp(-decay_rate * r_squared)
        
        # Laplacian of exp(-α|x-c|²) = exp(-α|x-c|²) * (-2αd + 4α²|x-c|²)
        lapl = exp_term * (-2 * decay_rate * dimension + 4 * decay_rate**2 * r_squared)
        return -lapl
    
    def laplacian(x):
        return -source(x)
    
    return {
        'solution': solution,
        'source': source,
        'gradient': gradient,
        'laplacian': laplacian,
        'type': 'exponential',
        'decay_rate': decay_rate,
        'center': center
    }


def _gaussian_solution(dimension: int, params: Dict[str, Any]) -> Dict[str, Callable]:
    """Generate Gaussian manufactured solution."""
    sigma = params.get('sigma', 0.1)
    amplitude = params.get('amplitude', 1.0)
    center = params.get('center', [0.5] * dimension)
    
    if len(center) != dimension:
        center = [0.5] * dimension
    
    def solution(x):
        r_squared = sum((x[i] - center[i])**2 for i in range(dimension))
        return amplitude * np.exp(-r_squared / (2 * sigma**2))
    
    def gradient(x):
        r_squared = sum((x[i] - center[i])**2 for i in range(dimension))
        exp_term = amplitude * np.exp(-r_squared / (2 * sigma**2))
        
        grad = []
        for i in range(dimension):
            grad_i = -(x[i] - center[i]) / sigma**2 * exp_term
            grad.append(grad_i)
        
        return np.array(grad)
    
    def source(x):
        r_squared = sum((x[i] - center[i])**2 for i in range(dimension))
        exp_term = amplitude * np.exp(-r_squared / (2 * sigma**2))
        
        # Laplacian of Gaussian
        lapl = exp_term * (r_squared / sigma**4 - dimension / sigma**2)
        return -lapl
    
    def laplacian(x):
        return -source(x)
    
    return {
        'solution': solution,
        'source': source,
        'gradient': gradient,
        'laplacian': laplacian,
        'type': 'gaussian',
        'sigma': sigma,
        'center': center
    }


def _mixed_solution(dimension: int, params: Dict[str, Any]) -> Dict[str, Callable]:
    """Generate mixed (polynomial + trigonometric) solution."""
    poly_coeffs = params.get('polynomial_coeffs', [1.0, 1.0])
    trig_freq = params.get('trigonometric_frequency', 1.0)
    trig_amp = params.get('trigonometric_amplitude', 0.5)
    
    # Get individual solutions
    poly_sol = _polynomial_solution(dimension, {'coefficients': poly_coeffs})
    trig_sol = _trigonometric_solution(dimension, {
        'frequency': trig_freq, 
        'amplitude': trig_amp
    })
    
    def solution(x):
        return poly_sol['solution'](x) + trig_sol['solution'](x)
    
    def gradient(x):
        return poly_sol['gradient'](x) + trig_sol['gradient'](x)
    
    def source(x):
        return poly_sol['source'](x) + trig_sol['source'](x)
    
    def laplacian(x):
        return poly_sol['laplacian'](x) + trig_sol['laplacian'](x)
    
    return {
        'solution': solution,
        'source': source,
        'gradient': gradient,
        'laplacian': laplacian,
        'type': 'mixed',
        'components': ['polynomial', 'trigonometric']
    }


def create_boundary_conditions_from_mms(
    manufactured_solution: Dict[str, Callable],
    boundary_markers: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Create boundary conditions from manufactured solution.
    
    Parameters
    ----------
    manufactured_solution : Dict[str, Callable]
        MMS solution dictionary
    boundary_markers : Dict[str, Any]
        Boundary marker definitions
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Boundary conditions
    """
    boundary_conditions = {}
    solution_func = manufactured_solution['solution']
    gradient_func = manufactured_solution.get('gradient', None)
    
    for marker_name, marker_info in boundary_markers.items():
        if marker_info.get('type') == 'dirichlet':
            # Dirichlet BC: u = u_exact
            boundary_conditions[f"dirichlet_{marker_name}"] = {
                'type': 'dirichlet',
                'boundary': marker_info['id'],
                'value': solution_func
            }
        
        elif marker_info.get('type') == 'neumann' and gradient_func is not None:
            # Neumann BC: ∂u/∂n = (∇u_exact) · n
            def neumann_value(x):
                grad = gradient_func(x)
                normal = marker_info.get('normal', np.array([1.0]))  # Default normal
                return np.dot(grad, normal)
            
            boundary_conditions[f"neumann_{marker_name}"] = {
                'type': 'neumann',
                'boundary': marker_info['id'],
                'value': neumann_value
            }
    
    return boundary_conditions


def verify_mms_consistency(
    manufactured_solution: Dict[str, Callable],
    test_points: Optional[np.ndarray] = None,
    tolerance: float = 1e-10
) -> Dict[str, Any]:
    """Verify consistency of manufactured solution.
    
    Parameters
    ----------
    manufactured_solution : Dict[str, Callable]
        MMS solution to verify
    test_points : np.ndarray, optional
        Points to test at
    tolerance : float, optional
        Numerical tolerance, by default 1e-10
        
    Returns
    -------
    Dict[str, Any]
        Verification results
    """
    if test_points is None:
        # Default test points
        test_points = np.array([
            [0.1, 0.1],
            [0.5, 0.5],
            [0.9, 0.9],
            [0.2, 0.8],
            [0.8, 0.2]
        ])
    
    solution_func = manufactured_solution['solution']
    source_func = manufactured_solution['source']
    gradient_func = manufactured_solution.get('gradient', None)
    laplacian_func = manufactured_solution.get('laplacian', None)
    
    results = {
        'consistent': True,
        'errors': [],
        'max_error': 0.0,
        'test_points': len(test_points)
    }
    
    for i, point in enumerate(test_points):
        try:
            # Evaluate functions
            u_val = solution_func(point)
            f_val = source_func(point)
            
            # Basic consistency checks
            if np.isnan(u_val) or np.isinf(u_val):
                results['errors'].append(f"Point {i}: solution is NaN/Inf")
                results['consistent'] = False
            
            if np.isnan(f_val) or np.isinf(f_val):
                results['errors'].append(f"Point {i}: source is NaN/Inf")
                results['consistent'] = False
            
            # Check gradient consistency (if available)
            if gradient_func is not None:
                grad_val = gradient_func(point)
                if np.any(np.isnan(grad_val)) or np.any(np.isinf(grad_val)):
                    results['errors'].append(f"Point {i}: gradient is NaN/Inf")
                    results['consistent'] = False
            
            # Check Laplacian consistency (if available)
            if laplacian_func is not None:
                lapl_val = laplacian_func(point)
                if np.isnan(lapl_val) or np.isinf(lapl_val):
                    results['errors'].append(f"Point {i}: Laplacian is NaN/Inf")
                    results['consistent'] = False
                
                # For Poisson equation: -∇²u = f
                expected_source = -lapl_val
                error = abs(f_val - expected_source)
                results['max_error'] = max(results['max_error'], error)
                
                if error > tolerance:
                    results['errors'].append(
                        f"Point {i}: source/Laplacian inconsistency, error={error:.2e}"
                    )
                    results['consistent'] = False
        
        except Exception as e:
            results['errors'].append(f"Point {i}: evaluation failed: {e}")
            results['consistent'] = False
    
    logger.info(f"MMS verification: {'PASSED' if results['consistent'] else 'FAILED'}")
    if not results['consistent']:
        for error in results['errors']:
            logger.warning(f"MMS verification error: {error}")
    
    return results


def generate_vector_mms(
    dimension: int = 2,
    num_components: int = 2,
    solution_type: SolutionType = SolutionType.TRIGONOMETRIC,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Callable]:
    """Generate vector-valued manufactured solution.
    
    Parameters
    ----------
    dimension : int, optional
        Spatial dimension, by default 2
    num_components : int, optional
        Number of vector components, by default 2
    solution_type : SolutionType, optional
        Solution type, by default TRIGONOMETRIC
    parameters : Dict[str, Any], optional
        Solution parameters
        
    Returns
    -------
    Dict[str, Callable]
        Vector MMS solution
    """
    params = parameters or {}
    
    # Generate individual component solutions
    component_solutions = []
    for i in range(num_components):
        # Vary parameters for each component
        comp_params = params.copy()
        if 'frequency' in comp_params:
            comp_params['frequency'] *= (i + 1)
        if 'phase' in comp_params:
            comp_params['phase'] += i * np.pi / 4
        
        comp_sol = generate_manufactured_solution(solution_type, dimension, comp_params)
        component_solutions.append(comp_sol)
    
    def solution(x):
        return np.array([comp_sol['solution'](x) for comp_sol in component_solutions])
    
    def source(x):
        return np.array([comp_sol['source'](x) for comp_sol in component_solutions])
    
    def gradient(x):
        # Returns list of gradients for each component
        return [comp_sol['gradient'](x) for comp_sol in component_solutions]
    
    def divergence(x):
        # Compute divergence: ∇ · u = ∂u₁/∂x₁ + ∂u₂/∂x₂ + ...
        div = 0.0
        grads = gradient(x)
        for i in range(min(num_components, dimension)):
            div += grads[i][i]  # ∂uᵢ/∂xᵢ
        return div
    
    return {
        'solution': solution,
        'source': source,
        'gradient': gradient,
        'divergence': divergence,
        'type': f'vector_{solution_type.value}',
        'num_components': num_components,
        'dimension': dimension
    }


# Convenience functions for direct import (backward compatibility)
def polynomial_2d(degree: int = 2):
    """Generate 2D polynomial manufactured solution."""
    sol = generate_manufactured_solution(SolutionType.POLYNOMIAL, 2, {'degree': degree})
    return sol['solution'], sol['source']


def trigonometric_2d(frequency: float = 1.0):
    """Generate 2D trigonometric manufactured solution."""
    sol = generate_manufactured_solution(SolutionType.TRIGONOMETRIC, 2, {'frequency': frequency})
    return sol['solution'], sol['source']


def exponential_2d(decay_rate: float = 1.0):
    """Generate 2D exponential manufactured solution."""
    sol = generate_manufactured_solution(SolutionType.EXPONENTIAL, 2, {'decay_rate': decay_rate})
    return sol['solution'], sol['source']


def gaussian_2d(sigma: float = 0.1):
    """Generate 2D Gaussian manufactured solution."""
    sol = generate_manufactured_solution(SolutionType.GAUSSIAN, 2, {'sigma': sigma})
    return sol['solution'], sol['source']


# Additional compatibility functions
def laplace_manufactured_solution(solution_type: str = "polynomial"):
    """Generate Laplace equation manufactured solution."""
    if solution_type == "polynomial":
        return polynomial_2d(degree=2)
    elif solution_type == "trigonometric":
        return trigonometric_2d(frequency=1.0)
    else:
        return polynomial_2d(degree=2)
