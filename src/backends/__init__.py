"""Automatic differentiation backend management."""

from .base import ADBackend, get_backend, list_backends, set_default_backend
from .jax_backend import JAXBackend
from .numpy_backend import NumpyBackend
from .torch_backend import TorchBackend

__all__ = [
    "ADBackend",
    "get_backend",
    "set_default_backend",
    "list_backends",
    "NumpyBackend",
    "JAXBackend",
    "TorchBackend",
]
