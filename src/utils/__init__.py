"""Utility functions for DiffFE-Physics-Lab."""

from .error_computation import compute_error
from .manufactured_solutions import generate_manufactured_solution
from .validation import (
    validate_boundary_conditions,
    validate_function_space,
    validate_mesh,
)

__all__ = [
    "validate_mesh",
    "validate_function_space",
    "validate_boundary_conditions",
    "compute_error",
    "generate_manufactured_solution",
]
