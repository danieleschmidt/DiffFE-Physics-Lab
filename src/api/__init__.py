"""REST API for DiffFE-Physics-Lab."""

from .app import app, create_app
from .middleware import setup_middleware
from .routes import register_routes
from .serializers import (
    MeshSerializer,
    OptimizationResultSerializer,
    ProblemSerializer,
    SolutionSerializer,
)

__all__ = [
    "create_app",
    "app",
    "register_routes",
    "setup_middleware",
    "ProblemSerializer",
    "SolutionSerializer",
    "MeshSerializer",
    "OptimizationResultSerializer",
]
