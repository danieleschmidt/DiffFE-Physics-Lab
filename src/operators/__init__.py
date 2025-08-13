"""Differentiable physics operators for DiffFE-Physics-Lab."""

from .base import BaseOperator, get_operator, register_operator
from .elasticity import ElasticityOperator, elasticity
from .electromagnetic import MaxwellOperator, maxwell
from .fluid import NavierStokesOperator, incompressibility, navier_stokes
from .laplacian import LaplacianOperator, laplacian
from .nonlinear import HyperelasticOperator, hyperelastic
from .transport import AdvectionOperator, advection

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
    "AdvectionOperator",
]
