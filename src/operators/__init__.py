"""Differentiable physics operators for DiffFE-Physics-Lab."""

from .base import BaseOperator, register_operator, get_operator
from .laplacian import laplacian, LaplacianOperator
from .elasticity import elasticity, ElasticityOperator
from .fluid import navier_stokes, NavierStokesOperator, incompressibility
from .nonlinear import hyperelastic, HyperelasticOperator
from .electromagnetic import maxwell, MaxwellOperator
from .transport import advection, AdvectionOperator

__all__ = [
    # Base functionality
    "BaseOperator",
    "register_operator", 
    "get_operator",
    
    # Operator functions
    "laplacian",
    "elasticity",
    "navier_stokes",
    "incompressibility", 
    "hyperelastic",
    "maxwell",
    "advection",
    
    # Operator classes
    "LaplacianOperator",
    "ElasticityOperator", 
    "NavierStokesOperator",
    "HyperelasticOperator",
    "MaxwellOperator",
    "AdvectionOperator"
]
