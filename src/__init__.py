"""DiffFE-Physics-Lab: Differentiable Finite Element Framework.

A comprehensive framework for combining finite element methods with automatic
differentiation, enabling gradient-based optimization of physical systems.
"""

__version__ = "1.0.0-dev"
__author__ = "DiffFE-Physics-Lab Team"
__email__ = "team@diffhe-physics.org"
__license__ = "BSD-3-Clause"

from .models import Problem, MultiPhysicsProblem
from .operators import (
    laplacian,
    elasticity, 
    navier_stokes,
    hyperelastic,
    maxwell,
    advection
)
from .services import (
    FEBMLSolver,
    OptimizationService,
    AssemblyEngine
)
from .backends import get_backend, set_default_backend
from .utils import (
    validate_mesh,
    compute_error,
    generate_manufactured_solution
)

__all__ = [
    # Core classes
    "Problem",
    "MultiPhysicsProblem", 
    "FEBMLSolver",
    
    # Operators
    "laplacian",
    "elasticity",
    "navier_stokes", 
    "hyperelastic",
    "maxwell",
    "advection",
    
    # Services
    "OptimizationService",
    "AssemblyEngine",
    
    # Backend management
    "get_backend",
    "set_default_backend",
    
    # Utilities
    "validate_mesh",
    "compute_error",
    "generate_manufactured_solution"
]
