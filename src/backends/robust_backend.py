"""Robust backend management with dependency validation and graceful fallbacks."""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils.exceptions import BackendError, ConfigurationError, handle_import_error
from .base import ADBackend, register_backend

logger = logging.getLogger(__name__)


@dataclass
class BackendStatus:
    """Status information for a backend."""

    name: str
    available: bool
    version: Optional[str] = None
    gpu_support: bool = False
    device_count: int = 0
    memory_info: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    fallback_backends: List[str] = None


class BackendManager:
    """Robust backend manager with dependency validation and fallbacks.

    Provides comprehensive backend detection, validation, and fallback
    mechanisms for production-ready deployment.

    Examples
    --------
    >>> manager = BackendManager()
    >>> backend = manager.get_best_available_backend()
    >>> if backend is None:
    ...     print("No backends available - using fallback")
    """

    def __init__(self, enable_fallbacks: bool = True):
        self.enable_fallbacks = enable_fallbacks
        self._backend_cache = {}
        self._status_cache = {}
        self._fallback_hierarchy = ["jax", "torch", "numpy"]
        self.refresh_backend_status()

    def refresh_backend_status(self) -> None:
        """Refresh backend availability status."""
        logger.info("Checking backend availability...")

        # Check JAX
        self._status_cache["jax"] = self._check_jax_status()

        # Check PyTorch
        self._status_cache["torch"] = self._check_torch_status()

        # Check numpy (always available fallback)
        self._status_cache["numpy"] = self._check_numpy_status()

        # Log status
        available_backends = [
            name for name, status in self._status_cache.items() if status.available
        ]
        logger.info(f"Available backends: {available_backends}")

        if not available_backends:
            logger.critical("No automatic differentiation backends available!")

    def _check_jax_status(self) -> BackendStatus:
        """Check JAX backend status."""
        try:
            import jax
            import jax.numpy as jnp

            # Basic functionality test
            test_array = jnp.array([1.0, 2.0, 3.0])
            _ = jax.grad(lambda x: jnp.sum(x**2))(test_array)

            # GPU information
            gpu_support = False
            device_count = 0
            try:
                gpu_devices = jax.devices("gpu")
                gpu_support = len(gpu_devices) > 0
                device_count = len(jax.devices())
            except:
                pass

            # Memory information
            memory_info = None
            try:
                if gpu_support:
                    # Get GPU memory info if available
                    memory_info = {"total_devices": device_count}
            except:
                pass

            return BackendStatus(
                name="jax",
                available=True,
                version=jax.__version__,
                gpu_support=gpu_support,
                device_count=device_count,
                memory_info=memory_info,
            )

        except ImportError as e:
            return BackendStatus(
                name="jax",
                available=False,
                error_message=f"JAX not available: {e}",
                fallback_backends=["torch", "numpy"],
            )
        except Exception as e:
            return BackendStatus(
                name="jax",
                available=False,
                error_message=f"JAX test failed: {e}",
                fallback_backends=["torch", "numpy"],
            )

    def _check_torch_status(self) -> BackendStatus:
        """Check PyTorch backend status."""
        try:
            import torch

            # Basic functionality test
            test_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            loss = torch.sum(test_tensor**2)
            loss.backward()

            # GPU information
            gpu_support = (
                torch.cuda.is_available()
                if hasattr(torch.cuda, "is_available")
                else False
            )
            device_count = torch.cuda.device_count() if gpu_support else 1

            # Memory information
            memory_info = None
            try:
                if gpu_support:
                    memory_info = {
                        "gpu_memory_total": torch.cuda.get_device_properties(
                            0
                        ).total_memory,
                        "gpu_memory_cached": torch.cuda.memory_cached(0),
                        "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                    }
            except:
                pass

            return BackendStatus(
                name="torch",
                available=True,
                version=torch.__version__,
                gpu_support=gpu_support,
                device_count=device_count,
                memory_info=memory_info,
            )

        except ImportError as e:
            return BackendStatus(
                name="torch",
                available=False,
                error_message=f"PyTorch not available: {e}",
                fallback_backends=["jax", "numpy"],
            )
        except Exception as e:
            return BackendStatus(
                name="torch",
                available=False,
                error_message=f"PyTorch test failed: {e}",
                fallback_backends=["jax", "numpy"],
            )

    def _check_numpy_status(self) -> BackendStatus:
        """Check NumPy backend status (finite difference fallback)."""
        try:
            import numpy as np

            # Basic test
            test_array = np.array([1.0, 2.0, 3.0])
            _ = np.sum(test_array**2)

            return BackendStatus(
                name="numpy",
                available=True,
                version=np.__version__,
                gpu_support=False,
                device_count=1,
            )

        except ImportError as e:
            return BackendStatus(
                name="numpy", available=False, error_message=f"NumPy not available: {e}"
            )
        except Exception as e:
            return BackendStatus(
                name="numpy", available=False, error_message=f"NumPy test failed: {e}"
            )

    def get_backend_status(self, backend_name: str) -> Optional[BackendStatus]:
        """Get status for a specific backend.

        Parameters
        ----------
        backend_name : str
            Name of the backend

        Returns
        -------
        Optional[BackendStatus]
            Backend status or None if unknown
        """
        return self._status_cache.get(backend_name)

    def list_available_backends(self) -> List[str]:
        """List all available backends.

        Returns
        -------
        List[str]
            Names of available backends
        """
        return [name for name, status in self._status_cache.items() if status.available]

    def get_best_available_backend(self, prefer_gpu: bool = True) -> Optional[str]:
        """Get the best available backend.

        Parameters
        ----------
        prefer_gpu : bool, optional
            Whether to prefer GPU-enabled backends

        Returns
        -------
        Optional[str]
            Name of best backend or None if none available
        """
        available = self.list_available_backends()

        if not available:
            logger.warning("No backends available")
            return None

        # Score backends
        scores = {}
        for name in available:
            status = self._status_cache[name]
            score = 0

            # Base scores
            if name == "jax":
                score += 100  # JAX is preferred for AD
            elif name == "torch":
                score += 80  # PyTorch is good
            elif name == "numpy":
                score += 10  # NumPy is fallback only

            # GPU bonus
            if prefer_gpu and status.gpu_support:
                score += 50

            # Device count bonus
            score += status.device_count * 5

            scores[name] = score

        best_backend = max(scores, key=scores.get)
        logger.info(
            f"Selected best backend: {best_backend} (score: {scores[best_backend]})"
        )

        return best_backend

    def get_backend_with_fallback(
        self, preferred_backend: str
    ) -> Tuple[Optional[Any], str]:
        """Get backend instance with fallback support.

        Parameters
        ----------
        preferred_backend : str
            Preferred backend name

        Returns
        -------
        Tuple[Optional[Any], str]
            (Backend instance, actual backend name used)
        """
        # Try preferred backend first
        if preferred_backend in self._status_cache:
            status = self._status_cache[preferred_backend]
            if status.available:
                try:
                    backend = self._create_backend_instance(preferred_backend)
                    return backend, preferred_backend
                except Exception as e:
                    logger.warning(f"Failed to create {preferred_backend} backend: {e}")

        # Try fallbacks if enabled
        if self.enable_fallbacks:
            fallbacks = self._get_fallback_chain(preferred_backend)
            for fallback_name in fallbacks:
                if fallback_name in self._status_cache:
                    status = self._status_cache[fallback_name]
                    if status.available:
                        try:
                            backend = self._create_backend_instance(fallback_name)
                            logger.info(f"Using fallback backend: {fallback_name}")
                            return backend, fallback_name
                        except Exception as e:
                            logger.warning(f"Fallback {fallback_name} failed: {e}")

        # No backend available
        logger.error("No working backends available")
        return None, "none"

    def _get_fallback_chain(self, preferred_backend: str) -> List[str]:
        """Get fallback chain for a backend."""
        if preferred_backend in self._status_cache:
            status = self._status_cache[preferred_backend]
            if status.fallback_backends:
                return status.fallback_backends

        # Default fallback hierarchy
        fallbacks = [b for b in self._fallback_hierarchy if b != preferred_backend]
        return fallbacks

    def _create_backend_instance(self, backend_name: str) -> Any:
        """Create backend instance."""
        # Use existing backend creation logic
        from .base import get_backend

        return get_backend(backend_name)

    def validate_backend_compatibility(
        self, backend_name: str, requirements: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate backend meets requirements.

        Parameters
        ----------
        backend_name : str
            Backend to validate
        requirements : Dict[str, Any]
            Requirements to check (gpu_required, min_memory, etc.)

        Returns
        -------
        Tuple[bool, List[str]]
            (Compatible, List of issues)
        """
        if backend_name not in self._status_cache:
            return False, [f"Backend '{backend_name}' not found"]

        status = self._status_cache[backend_name]
        if not status.available:
            return False, [
                f"Backend '{backend_name}' not available: {status.error_message}"
            ]

        issues = []

        # Check GPU requirement
        if requirements.get("gpu_required", False) and not status.gpu_support:
            issues.append(f"Backend '{backend_name}' does not support GPU")

        # Check minimum memory
        min_memory = requirements.get("min_memory_gb", 0)
        if min_memory > 0 and status.memory_info:
            if "gpu_memory_total" in status.memory_info:
                available_gb = status.memory_info["gpu_memory_total"] / (1024**3)
                if available_gb < min_memory:
                    issues.append(
                        f"Insufficient GPU memory: {available_gb:.1f}GB < {min_memory}GB"
                    )

        # Check device count
        min_devices = requirements.get("min_devices", 1)
        if status.device_count < min_devices:
            issues.append(
                f"Insufficient devices: {status.device_count} < {min_devices}"
            )

        return len(issues) == 0, issues

    def generate_backend_report(self) -> Dict[str, Any]:
        """Generate comprehensive backend status report.

        Returns
        -------
        Dict[str, Any]
            Detailed backend report
        """
        report = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "total_backends_checked": len(self._status_cache),
            "available_backends": self.list_available_backends(),
            "backend_details": {},
            "recommendations": [],
        }

        for name, status in self._status_cache.items():
            report["backend_details"][name] = {
                "available": status.available,
                "version": status.version,
                "gpu_support": status.gpu_support,
                "device_count": status.device_count,
                "memory_info": status.memory_info,
                "error_message": status.error_message,
            }

        # Generate recommendations
        available = self.list_available_backends()
        if not available:
            report["recommendations"].append(
                "No AD backends available. Install JAX or PyTorch for better performance."
            )
        elif "jax" not in available and "torch" not in available:
            report["recommendations"].append(
                "Consider installing JAX or PyTorch for automatic differentiation support."
            )

        # Check for GPU support
        gpu_backends = [
            name
            for name, status in self._status_cache.items()
            if status.available and status.gpu_support
        ]
        if not gpu_backends:
            report["recommendations"].append(
                "No GPU support detected. Install CUDA-enabled versions for better performance."
            )

        return report


# Fallback implementations


class NumPyBackend(ADBackend):
    """Fallback backend using NumPy with finite differences.

    Provides basic gradient computation using numerical differentiation
    when proper AD backends are not available.
    """

    def _check_dependencies(self) -> None:
        """Check NumPy availability."""
        try:
            import numpy as np

            self.np = np
        except ImportError:
            handle_import_error("numpy", "pip install numpy")

    def grad(self, func: Callable, argnums: int = 0, has_aux: bool = False) -> Callable:
        """Compute gradient using finite differences."""

        def grad_func(*args, **kwargs):
            eps = 1e-8
            x = args[argnums]

            if hasattr(x, "__iter__"):
                # Array input
                grad_result = self.np.zeros_like(x)
                for i, xi in enumerate(x):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps

                    args_plus = list(args)
                    args_minus = list(args)
                    args_plus[argnums] = x_plus
                    args_minus[argnums] = x_minus

                    f_plus = func(*args_plus, **kwargs)
                    f_minus = func(*args_minus, **kwargs)

                    if has_aux:
                        f_plus, _ = f_plus
                        f_minus, _ = f_minus

                    grad_result[i] = (f_plus - f_minus) / (2 * eps)

                return grad_result
            else:
                # Scalar input
                f_plus = func(
                    *(args[:argnums] + (x + eps,) + args[argnums + 1 :]), **kwargs
                )
                f_minus = func(
                    *(args[:argnums] + (x - eps,) + args[argnums + 1 :]), **kwargs
                )

                if has_aux:
                    f_plus, _ = f_plus
                    f_minus, _ = f_minus

                return (f_plus - f_minus) / (2 * eps)

        return grad_func

    def jacobian(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute Jacobian using finite differences."""

        def jacobian_func(*args, **kwargs):
            eps = 1e-8
            x = args[argnums]

            # Compute output dimension
            f_center = func(*args, **kwargs)
            if hasattr(f_center, "__iter__"):
                output_dim = len(f_center)
            else:
                output_dim = 1
                f_center = [f_center]

            if hasattr(x, "__iter__"):
                input_dim = len(x)
                jac = self.np.zeros((output_dim, input_dim))

                for i in range(input_dim):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps

                    args_plus = list(args)
                    args_minus = list(args)
                    args_plus[argnums] = x_plus
                    args_minus[argnums] = x_minus

                    f_plus = func(*args_plus, **kwargs)
                    f_minus = func(*args_minus, **kwargs)

                    if not hasattr(f_plus, "__iter__"):
                        f_plus = [f_plus]
                        f_minus = [f_minus]

                    jac[:, i] = [
                        (fp - fm) / (2 * eps) for fp, fm in zip(f_plus, f_minus)
                    ]

                return jac if output_dim > 1 else jac[0]
            else:
                # Scalar input
                f_plus = func(
                    *(args[:argnums] + (x + eps,) + args[argnums + 1 :]), **kwargs
                )
                f_minus = func(
                    *(args[:argnums] + (x - eps,) + args[argnums + 1 :]), **kwargs
                )

                if not hasattr(f_plus, "__iter__"):
                    f_plus = [f_plus]
                    f_minus = [f_minus]

                return [(fp - fm) / (2 * eps) for fp, fm in zip(f_plus, f_minus)]

        return jacobian_func

    def hessian(self, func: Callable, argnums: int = 0) -> Callable:
        """Compute Hessian using finite differences."""
        grad_func = self.grad(func, argnums)
        return self.jacobian(grad_func, argnums)

    def jit(self, func: Callable, **kwargs) -> Callable:
        """No-op JIT compilation for NumPy."""
        logger.warning("JIT compilation not available with NumPy backend")
        return func

    def vmap(self, func: Callable, in_axes: Any = 0, out_axes: Any = 0) -> Callable:
        """Vectorization using NumPy."""

        def vmap_func(*args):
            # Simplified vectorization
            if in_axes == 0:
                # Vectorize over first axis of first argument
                results = []
                for i in range(len(args[0])):
                    result = func(
                        *(arg[i] if j == 0 else arg for j, arg in enumerate(args))
                    )
                    results.append(result)
                return self.np.array(results)
            else:
                logger.warning("Complex vmap not supported with NumPy backend")
                return func(*args)

        return vmap_func

    def to_array(self, data: Any) -> Any:
        """Convert to NumPy array."""
        return self.np.asarray(data)

    def from_array(self, array: Any) -> Any:
        """Convert from NumPy array."""
        return array


# Register NumPy fallback backend
@register_backend("numpy")
class NumPyBackendRegistered(NumPyBackend):
    """Registered NumPy fallback backend."""

    pass


# Global backend manager instance
_global_manager = None


def get_global_backend_manager() -> BackendManager:
    """Get global backend manager instance.

    Returns
    -------
    BackendManager
        Global backend manager
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = BackendManager()
    return _global_manager


def get_robust_backend(preferred_backend: str = None) -> Tuple[Optional[Any], str]:
    """Get backend with robust fallback support.

    Parameters
    ----------
    preferred_backend : str, optional
        Preferred backend name

    Returns
    -------
    Tuple[Optional[Any], str]
        (Backend instance, actual backend used)
    """
    manager = get_global_backend_manager()

    if preferred_backend is None:
        preferred_backend = manager.get_best_available_backend()

    if preferred_backend is None:
        logger.error("No backends available")
        return None, "none"

    return manager.get_backend_with_fallback(preferred_backend)


def validate_backend_requirements(
    backend_name: str, requirements: Dict[str, Any]
) -> bool:
    """Validate backend meets requirements.

    Parameters
    ----------
    backend_name : str
        Backend to validate
    requirements : Dict[str, Any]
        Requirements dictionary

    Returns
    -------
    bool
        True if requirements are met
    """
    manager = get_global_backend_manager()
    compatible, issues = manager.validate_backend_compatibility(
        backend_name, requirements
    )

    if not compatible:
        logger.warning(f"Backend {backend_name} compatibility issues: {issues}")

    return compatible
