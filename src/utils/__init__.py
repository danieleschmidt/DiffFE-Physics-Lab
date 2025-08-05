"""Utility functions for DiffFE-Physics-Lab."""

from .validation import (
    validate_mesh,
    validate_function_space,
    validate_boundary_conditions
)
from .error_computation import compute_error
from .manufactured_solutions import generate_manufactured_solution

__all__ = [
    'validate_mesh',
    'validate_function_space', 
    'validate_boundary_conditions',
    'compute_error',
    'generate_manufactured_solution'
]