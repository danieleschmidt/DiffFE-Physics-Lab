"""Automatic differentiation backend management."""

from .base import ADBackend, get_backend, set_default_backend, list_backends
from .jax_backend import JAXBackend
from .torch_backend import TorchBackend

__all__ = [
    "ADBackend",
    "get_backend",
    "set_default_backend", 
    "list_backends",
    "JAXBackend",
    "TorchBackend"
]
