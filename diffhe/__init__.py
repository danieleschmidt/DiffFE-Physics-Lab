"""DiffFE-Physics-Lab: Differentiable Finite Elements with ML Integration.

Gradient-based optimization of physical systems through differentiable PDE solvers.
"""

from .mesh import FEMesh
from .solver import DifferentiableFESolver
from .loss import PhysicsLoss
from .neural import NeuralPDE

__version__ = "0.1.0"
__all__ = ["FEMesh", "DifferentiableFESolver", "PhysicsLoss", "NeuralPDE"]
