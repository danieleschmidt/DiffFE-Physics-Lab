"""Core problem models for DiffFE-Physics-Lab."""

from .fields import BoundaryCondition, Field, ParametricField
from .multiphysics import Domain, MultiPhysicsProblem
from .problem import FEBMLProblem, Problem

__all__ = [
    "Problem",
    "FEBMLProblem",
    "MultiPhysicsProblem",
    "Domain",
    "Field",
    "ParametricField",
    "BoundaryCondition",
]
