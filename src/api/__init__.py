"""REST API for DiffFE-Physics-Lab."""

from .app import create_app, app
from .routes import register_routes
from .middleware import setup_middleware
from .serializers import (
    ProblemSerializer,
    SolutionSerializer,
    MeshSerializer,
    OptimizationResultSerializer
)

__all__ = [
    "create_app",
    "app",
    "register_routes",
    "setup_middleware",
    "ProblemSerializer",
    "SolutionSerializer", 
    "MeshSerializer",
    "OptimizationResultSerializer"
]
