"""Core problem models for DiffFE-Physics-Lab."""

from .problem import Problem, FEBMLProblem
from .multiphysics import MultiPhysicsProblem, Domain
from .fields import Field, ParametricField, BoundaryCondition

__all__ = [
    "Problem",
    "FEBMLProblem", 
    "MultiPhysicsProblem",
    "Domain",
    "Field",
    "ParametricField",
    "BoundaryCondition"
]
