"""Simple manufactured solutions without external dependencies."""

import math
from typing import Tuple


def polynomial_2d(x: float, y: float) -> float:
    """Simple polynomial manufactured solution in 2D.
    
    Returns u(x,y) = x^2 + y^2
    """
    return x**2 + y**2


def trigonometric_2d(x: float, y: float) -> float:
    """Simple trigonometric manufactured solution in 2D.
    
    Returns u(x,y) = sin(πx)sin(πy)
    """
    return math.sin(math.pi * x) * math.sin(math.pi * y)


def exponential_2d(x: float, y: float) -> float:
    """Simple exponential manufactured solution in 2D.
    
    Returns u(x,y) = exp(-(x^2 + y^2))
    """
    return math.exp(-(x**2 + y**2))


def laplace_manufactured_solution(x: float, y: float) -> Tuple[float, float]:
    """Manufactured solution for Laplace equation.
    
    Returns (u, source) where u = x^2 + y^2 and source = -∇²u = -4
    """
    u = polynomial_2d(x, y)
    source = 4.0  # -∇²(x² + y²) = -4
    return u, source


def compute_source_term(x: float, y: float, solution_type: str = "polynomial") -> float:
    """Compute source term for manufactured solutions.
    
    Args:
        x, y: Coordinates
        solution_type: Type of solution ("polynomial", "trigonometric", "exponential")
    
    Returns:
        Source term value
    """
    if solution_type == "polynomial":
        return 4.0  # -∇²(x² + y²) = -4
    elif solution_type == "trigonometric":
        return 2 * math.pi**2 * math.sin(math.pi * x) * math.sin(math.pi * y)
    elif solution_type == "exponential":
        r2 = x**2 + y**2
        exp_term = math.exp(-r2)
        return -exp_term * (-4 + 4*r2)  # -∇²(exp(-r²))
    else:
        raise ValueError(f"Unknown solution type: {solution_type}")


# Simple functions that can be imported without numpy
__all__ = [
    "polynomial_2d",
    "trigonometric_2d", 
    "exponential_2d",
    "laplace_manufactured_solution",
    "compute_source_term"
]