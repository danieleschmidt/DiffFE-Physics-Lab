"""Differentiable physics operators for DiffFE-Physics-Lab."""

from .base import BaseOperator, register_operator, get_operator
from .laplacian import laplacian, LaplacianOperator
from .elasticity import elasticity, ElasticityOperator
try:
    from .fluid import navier_stokes, NavierStokesOperator, incompressibility
    from .nonlinear import hyperelastic, HyperelasticOperator
    from .electromagnetic import maxwell, MaxwellOperator
    from .transport import advection, AdvectionOperator
    HAS_ADVANCED_OPERATORS = True
except ImportError:
    HAS_ADVANCED_OPERATORS = False

# Sentiment analysis operators
from .sentiment import (
    SentimentDiffusionOperator,
    SentimentReactionOperator,
    SentimentGradientOperator,
    SentimentLaplacianOperator,
    SentimentAdvectionOperator,
    CompositeSentimentOperator
)

__all__ = [
    # Base functionality
    "BaseOperator",
    "register_operator", 
    "get_operator",
    
    # Core operator functions
    "laplacian",
    "elasticity",
    
    # Core operator classes
    "LaplacianOperator",
    "ElasticityOperator",
    
    # Sentiment analysis operators
    "SentimentDiffusionOperator",
    "SentimentReactionOperator",
    "SentimentGradientOperator",
    "SentimentLaplacianOperator", 
    "SentimentAdvectionOperator",
    "CompositeSentimentOperator"
]

if HAS_ADVANCED_OPERATORS:
    __all__.extend([
        # Advanced operator functions
        "navier_stokes",
        "incompressibility", 
        "hyperelastic",
        "maxwell",
        "advection",
        
        # Advanced operator classes
        "NavierStokesOperator",
        "HyperelasticOperator",
        "MaxwellOperator",
        "AdvectionOperator"
    ])
