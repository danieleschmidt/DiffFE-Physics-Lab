"""Laplacian operator implementation."""

from typing import Any, Callable, Dict

import numpy as np

try:
    import firedrake as fd

    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from .base import BaseOperator, LinearOperator, register_operator


@register_operator("laplacian")
class LaplacianOperator(LinearOperator):
    """Differentiable Laplacian operator for diffusion problems.

    Implements the weak form: ∫ κ ∇u · ∇v dx = ∫ f v dx
    where κ is the diffusion coefficient and f is the source term.

    Parameters
    ----------
    diffusion_coeff : float or Callable, optional
        Diffusion coefficient κ, by default 1.0
    **kwargs
        Additional parameters passed to BaseOperator

    Examples
    --------
    >>> op = LaplacianOperator(diffusion_coeff=2.0)
    >>> weak_form = op(trial, test, {'source': lambda x: 1.0})
    """

    _is_linear = True
    _is_symmetric = True

    def __init__(self, diffusion_coeff=1.0, **kwargs):
        super().__init__(**kwargs)
        self.diffusion_coeff = diffusion_coeff

    def forward_assembly(self, trial: Any, test: Any, params: Dict[str, Any]) -> Any:
        """Assemble Laplacian weak form.

        Parameters
        ----------
        trial : firedrake.TrialFunction or Function
            Trial function u
        test : firedrake.TestFunction
            Test function v
        params : Dict[str, Any]
            Parameters including 'source' term

        Returns
        -------
        firedrake.Form
            Weak form: ∫ κ ∇u · ∇v dx - ∫ f v dx
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")

        self.validate_inputs(trial, test, params)

        # Get diffusion coefficient
        kappa = params.get("diffusion_coeff", self.diffusion_coeff)
        if callable(kappa):
            # Function-based coefficient
            kappa_func = fd.Function(trial.function_space())
            kappa_func.interpolate(kappa)
            kappa = kappa_func
        elif not isinstance(kappa, (fd.Function, fd.Constant)):
            # Scalar coefficient
            kappa = fd.Constant(float(kappa))

        # Bilinear form: ∫ κ ∇u · ∇v dx
        a = kappa * fd.inner(fd.grad(trial), fd.grad(test)) * fd.dx

        # Linear form: ∫ f v dx
        if "source" in params:
            source = params["source"]
            if callable(source):
                # Function-based source
                f_func = fd.Function(test.function_space())
                f_func.interpolate(source)
                source = f_func
            elif not isinstance(source, (fd.Function, fd.Constant)):
                # Scalar source
                source = fd.Constant(float(source))

            L = source * test * fd.dx
            return a - L
        else:
            return a

    def adjoint_assembly(
        self, grad_output: Any, trial: Any, test: Any, params: Dict[str, Any]
    ) -> Any:
        """Assemble adjoint Laplacian operator.

        For the symmetric Laplacian, the adjoint is the same as the forward operator.

        Parameters
        ----------
        grad_output : Any
            Gradient with respect to output
        trial : Any
            Trial function
        test : Any
            Test function
        params : Dict[str, Any]
            Parameters

        Returns
        -------
        Any
            Adjoint weak form
        """
        # For symmetric operators, adjoint equals forward
        return self.forward_assembly(trial, test, params)

    def apply_matrix(self, function_space: Any, params: Dict[str, Any] = None) -> Any:
        """Assemble Laplacian as matrix.

        Parameters
        ----------
        function_space : firedrake.FunctionSpace
            Function space
        params : Dict[str, Any], optional
            Parameters

        Returns
        -------
        firedrake.Matrix
            Assembled stiffness matrix
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for matrix assembly")

        if params is None:
            params = {}

        trial = fd.TrialFunction(function_space)
        test = fd.TestFunction(function_space)

        # Get bilinear form (exclude source term)
        source_backup = params.pop("source", None)
        a = self.forward_assembly(trial, test, params)
        if source_backup is not None:
            params["source"] = source_backup

        return fd.assemble(a)

    def manufactured_solution(self, **kwargs) -> Dict[str, Callable]:
        """Generate manufactured solution for Laplacian.

        Parameters
        ----------
        **kwargs
            Additional parameters (e.g., 'frequency', 'dimension')

        Returns
        -------
        Dict[str, Callable]
            Dictionary with 'solution' and 'source' functions
        """
        frequency = kwargs.get("frequency", 1.0)
        dimension = kwargs.get("dimension", 2)
        kappa = kwargs.get("diffusion_coeff", self.diffusion_coeff)

        if dimension == 1:

            def solution(x):
                return np.sin(frequency * np.pi * x[0])

            def source(x):
                return (
                    kappa * (frequency * np.pi) ** 2 * np.sin(frequency * np.pi * x[0])
                )

        elif dimension == 2:

            def solution(x):
                return np.sin(frequency * np.pi * x[0]) * np.sin(
                    frequency * np.pi * x[1]
                )

            def source(x):
                return (
                    kappa
                    * 2
                    * (frequency * np.pi) ** 2
                    * np.sin(frequency * np.pi * x[0])
                    * np.sin(frequency * np.pi * x[1])
                )

        elif dimension == 3:

            def solution(x):
                return (
                    np.sin(frequency * np.pi * x[0])
                    * np.sin(frequency * np.pi * x[1])
                    * np.sin(frequency * np.pi * x[2])
                )

            def source(x):
                return kappa * 3 * (frequency * np.pi) ** 2 * solution(x)

        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

        return {"solution": solution, "source": source}

    def compute_error(
        self, computed_solution: Any, exact_solution: Callable, norm_type: str = "L2"
    ) -> float:
        """Compute error for Laplacian problems.

        Parameters
        ----------
        computed_solution : firedrake.Function
            Computed solution
        exact_solution : Callable
            Exact solution function
        norm_type : str, optional
            Error norm type ('L2', 'H1'), by default 'L2'

        Returns
        -------
        float
            Error value
        """
        if not HAS_FIREDRAKE:
            return 0.0

        # Interpolate exact solution
        exact_func = fd.Function(computed_solution.function_space())
        exact_func.interpolate(exact_solution)

        # Compute error
        error_func = computed_solution - exact_func

        if norm_type.upper() == "L2":
            return fd.sqrt(fd.assemble(fd.inner(error_func, error_func) * fd.dx))
        elif norm_type.upper() == "H1":
            l2_error = fd.inner(error_func, error_func) * fd.dx
            h1_error = fd.inner(fd.grad(error_func), fd.grad(error_func)) * fd.dx
            return fd.sqrt(fd.assemble(l2_error + h1_error))
        elif norm_type.upper() == "LINF":
            return np.max(np.abs(error_func.dat.data))
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")


def laplacian(trial: Any, test: Any, diffusion_coeff=1.0, source=None, **kwargs) -> Any:
    """Convenience function for Laplacian operator.

    Parameters
    ----------
    trial : Any
        Trial function
    test : Any
        Test function
    diffusion_coeff : float or Callable, optional
        Diffusion coefficient, by default 1.0
    source : float or Callable, optional
        Source term, by default None
    **kwargs
        Additional parameters

    Returns
    -------
    Any
        Assembled weak form

    Examples
    --------
    >>> # Basic diffusion
    >>> form = laplacian(u, v, diffusion_coeff=2.0)
    >>>
    >>> # With source term
    >>> form = laplacian(u, v, source=lambda x: x[0]**2)
    """
    params = {"diffusion_coeff": diffusion_coeff, **kwargs}
    if source is not None:
        params["source"] = source

    op = LaplacianOperator(**kwargs)
    return op(trial, test, params)
