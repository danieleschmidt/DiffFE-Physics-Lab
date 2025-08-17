"""Core DiffFE-Physics-Lab components."""

from .enhanced_api import FEBMLProblem, MultiPhysics, HybridSolver
from .optimization import ParameterOptimizer, GradientDescentOptimizer
from .validation import ValidationSuite, ConvergenceAnalyzer

__all__ = [
    "FEBMLProblem",
    "MultiPhysics", 
    "HybridSolver",
    "ParameterOptimizer",
    "GradientDescentOptimizer",
    "ValidationSuite",
    "ConvergenceAnalyzer",
]