"""Core services for DiffFE-Physics-Lab."""

from .solver import FEBMLSolver
from .optimization import OptimizationService
from .assembly import AssemblyEngine

__all__ = [
    'FEBMLSolver',
    'OptimizationService',
    'AssemblyEngine'
]