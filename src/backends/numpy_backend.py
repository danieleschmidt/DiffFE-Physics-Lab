"""NumPy backend for basic automatic differentiation."""

from typing import Any, Callable

import numpy as np

from .base import ADBackend, register_backend


@register_backend("numpy")
class NumpyBackend(ADBackend):
    """NumPy-based backend with basic finite difference differentiation.

    Provides basic differentiation capabilities using NumPy and finite
    differences. This is a fallback backend when JAX/PyTorch are not available.

    Examples
    --------
    >>> backend = NumpyBackend()
    >>> grad_func = backend.grad(lambda x: x**2)
    >>> gradient = grad_func(3.0)  # Approximately 6.0
    """

    def __init__(self):
        self.name = "numpy"
        self.h = 1e-8  # Finite difference step size

    def _check_dependencies(self) -> bool:
        """Check if NumPy is available."""
        try:
            import numpy

            return True
        except ImportError:
            return False

    def from_array(self, array: np.ndarray) -> np.ndarray:
        """Convert array to backend format (no-op for NumPy)."""
        return np.asarray(array)

    @property
    def is_available(self) -> bool:
        """Check if NumPy backend is available (always True)."""
        return True

    def grad(self, func: Callable) -> Callable:
        """Compute gradient using finite differences."""

        def gradient_func(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 0:
                # Scalar input
                return (func(x + self.h) - func(x - self.h)) / (2 * self.h)
            else:
                # Vector input
                grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += self.h
                    x_minus[i] -= self.h
                    grad[i] = (func(x_plus) - func(x_minus)) / (2 * self.h)
                return grad

        return gradient_func

    def jacobian(self, func: Callable) -> Callable:
        """Compute Jacobian using finite differences."""

        def jacobian_func(x):
            x = np.asarray(x, dtype=float)
            f0 = func(x)
            f0 = np.asarray(f0)

            if x.ndim == 0:
                x = np.array([x])

            if f0.ndim == 0:
                f0 = np.array([f0])

            jac = np.zeros((len(f0), len(x)))

            for j in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[j] += self.h
                x_minus[j] -= self.h
                f_plus = np.asarray(func(x_plus))
                f_minus = np.asarray(func(x_minus))
                jac[:, j] = (f_plus - f_minus) / (2 * self.h)

            return jac

        return jacobian_func

    def hessian(self, func: Callable) -> Callable:
        """Compute Hessian using finite differences."""

        def hessian_func(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 0:
                x = np.array([x])

            n = len(x)
            hess = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i == j:
                        # Diagonal elements
                        x_plus = x.copy()
                        x_minus = x.copy()
                        x_plus[i] += self.h
                        x_minus[i] -= self.h
                        f_center = func(x)
                        f_plus = func(x_plus)
                        f_minus = func(x_minus)
                        hess[i, j] = (f_plus - 2 * f_center + f_minus) / (self.h**2)
                    else:
                        # Off-diagonal elements
                        x_pp = x.copy()
                        x_pm = x.copy()
                        x_mp = x.copy()
                        x_mm = x.copy()

                        x_pp[i] += self.h
                        x_pp[j] += self.h
                        x_pm[i] += self.h
                        x_pm[j] -= self.h
                        x_mp[i] -= self.h
                        x_mp[j] += self.h
                        x_mm[i] -= self.h
                        x_mm[j] -= self.h

                        hess[i, j] = (
                            func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)
                        ) / (4 * self.h**2)

            return hess

        return hessian_func

    def jit(self, func: Callable) -> Callable:
        """JIT compilation (no-op for NumPy)."""
        return func  # NumPy doesn't have JIT, so return original function

    def vmap(self, func: Callable) -> Callable:
        """Vectorized mapping."""

        def vmapped_func(inputs):
            inputs = np.asarray(inputs)
            if inputs.ndim == 1:
                # Single input vector
                return func(inputs)
            else:
                # Multiple inputs
                results = []
                for input_vec in inputs:
                    results.append(func(input_vec))
                return np.array(results)

        return vmapped_func

    def to_array(self, data: Any) -> np.ndarray:
        """Convert data to NumPy array."""
        return np.asarray(data)

    def solve_linear(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b."""
        return np.linalg.solve(A, b)

    def optimize(
        self,
        objective: Callable,
        initial_guess: np.ndarray,
        bounds: Any = None,
        method: str = "bfgs",
        **kwargs
    ) -> np.ndarray:
        """Optimize objective function using scipy.optimize."""
        try:
            from scipy.optimize import minimize

            result = minimize(
                objective, initial_guess, method=method.upper(), bounds=bounds, **kwargs
            )
            return result.x
        except ImportError:
            # Fallback: simple gradient descent
            x = initial_guess.copy()
            lr = kwargs.get("lr", 0.01)
            max_iter = kwargs.get("max_iter", 1000)

            grad_func = self.grad(objective)

            for _ in range(max_iter):
                grad = grad_func(x)
                x = x - lr * grad

            return x
