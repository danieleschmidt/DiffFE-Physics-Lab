"""Linear elasticity operator implementation."""

from typing import Any, Callable, Dict, Tuple

import numpy as np

try:
    import firedrake as fd

    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from .base import BaseOperator, LinearOperator, register_operator


@register_operator("elasticity")
class ElasticityOperator(LinearOperator):
    """Linear elasticity operator for solid mechanics.

    Implements the weak form for linear elasticity:
    ∫ σ(ε(u)) : ε(v) dx = ∫ f · v dx + ∫ t · v ds

    where σ is the stress tensor, ε is the strain tensor, f is body force,
    and t is traction on the boundary.

    Parameters
    ----------
    E : float, optional
        Young's modulus, by default 1.0
    nu : float, optional
        Poisson's ratio, by default 0.3
    plane_stress : bool, optional
        Use plane stress assumption, by default False
    **kwargs
        Additional parameters

    Examples
    --------
    >>> op = ElasticityOperator(E=200e9, nu=0.3)
    >>> weak_form = op(u, v, {'body_force': [0, -9.81]})
    """

    _is_linear = True
    _is_symmetric = True

    def __init__(self, E=1.0, nu=0.3, plane_stress=False, **kwargs):
        super().__init__(**kwargs)
        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        self.plane_stress = plane_stress

        # Validate material parameters
        if not 0 <= nu < 0.5:
            raise ValueError(f"Poisson's ratio must be in [0, 0.5), got {nu}")
        if E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {E}")

    def lame_parameters(self) -> Tuple[float, float]:
        """Compute Lamé parameters from E and nu.

        Returns
        -------
        Tuple[float, float]
            (lambda, mu) - Lamé parameters
        """
        if self.plane_stress:
            # Plane stress
            lmbda = self.E * self.nu / (1 - self.nu**2)
            mu = self.E / (2 * (1 + self.nu))
        else:
            # Plane strain or 3D
            lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            mu = self.E / (2 * (1 + self.nu))

        return lmbda, mu

    def strain_tensor(self, u: Any) -> Any:
        """Compute strain tensor from displacement.

        Parameters
        ----------
        u : firedrake.Function
            Displacement field

        Returns
        -------
        firedrake.Expression
            Strain tensor ε = 1/2(∇u + ∇u^T)
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required")

        return fd.sym(fd.grad(u))

    def stress_tensor(self, u: Any, params: Dict[str, Any] = None) -> Any:
        """Compute stress tensor from displacement.

        Parameters
        ----------
        u : firedrake.Function
            Displacement field
        params : Dict[str, Any], optional
            Parameters (can override material properties)

        Returns
        -------
        firedrake.Expression
            Stress tensor σ = λ tr(ε) I + 2μ ε
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required")

        if params is None:
            params = {}

        # Get material parameters
        E = params.get("E", self.E)
        nu = params.get("nu", self.nu)

        # Update Lamé parameters if material props changed
        if E != self.E or nu != self.nu:
            old_E, old_nu = self.E, self.nu
            self.E, self.nu = E, nu
            lmbda, mu = self.lame_parameters()
            self.E, self.nu = old_E, old_nu  # Restore
        else:
            lmbda, mu = self.lame_parameters()

        # Convert to Firedrake constants
        lmbda = fd.Constant(lmbda)
        mu = fd.Constant(mu)

        # Strain tensor
        eps = self.strain_tensor(u)

        # Stress tensor: σ = λ tr(ε) I + 2μ ε
        return lmbda * fd.tr(eps) * fd.Identity(u.geometric_dimension()) + 2 * mu * eps

    def forward_assembly(self, trial: Any, test: Any, params: Dict[str, Any]) -> Any:
        """Assemble elasticity weak form.

        Parameters
        ----------
        trial : firedrake.TrialFunction or Function
            Trial displacement field
        test : firedrake.TestFunction
            Test displacement field
        params : Dict[str, Any]
            Parameters including body forces, tractions

        Returns
        -------
        firedrake.Form
            Weak form for elasticity
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")

        self.validate_inputs(trial, test, params)

        # Material parameters
        lmbda, mu = self.lame_parameters()
        lmbda = fd.Constant(lmbda)
        mu = fd.Constant(mu)

        # Strain tensors
        eps_u = self.strain_tensor(trial)
        eps_v = self.strain_tensor(test)

        # Bilinear form: ∫ σ(ε(u)) : ε(v) dx
        # = ∫ [λ tr(ε(u)) tr(ε(v)) + 2μ ε(u) : ε(v)] dx
        a = (
            lmbda * fd.tr(eps_u) * fd.tr(eps_v) + 2 * mu * fd.inner(eps_u, eps_v)
        ) * fd.dx

        # Linear forms
        L = 0

        # Body force: ∫ f · v dx
        if "body_force" in params:
            f = params["body_force"]
            if isinstance(f, (list, tuple, np.ndarray)):
                # Vector body force
                f_vec = fd.Constant(f)
                L += fd.inner(f_vec, test) * fd.dx
            elif callable(f):
                # Function-based body force
                f_func = fd.Function(test.function_space())
                f_func.interpolate(f)
                L += fd.inner(f_func, test) * fd.dx
            else:
                raise ValueError("Body force must be vector, array, or function")

        # Neumann boundary conditions: ∫ t · v ds
        if "traction" in params:
            traction_bcs = params["traction"]
            if not isinstance(traction_bcs, dict):
                traction_bcs = {"boundary": traction_bcs}

            for boundary, t in traction_bcs.items():
                if isinstance(t, (list, tuple, np.ndarray)):
                    t_vec = fd.Constant(t)
                    L += fd.inner(t_vec, test) * fd.ds(boundary)
                elif callable(t):
                    t_func = fd.Function(test.function_space())
                    t_func.interpolate(t)
                    L += fd.inner(t_func, test) * fd.ds(boundary)

        return a - L if L != 0 else a

    def adjoint_assembly(
        self, grad_output: Any, trial: Any, test: Any, params: Dict[str, Any]
    ) -> Any:
        """Assemble adjoint elasticity operator.

        For symmetric elasticity, adjoint equals forward operator.
        """
        return self.forward_assembly(trial, test, params)

    def manufactured_solution(self, **kwargs) -> Dict[str, Callable]:
        """Generate manufactured solution for elasticity.

        Parameters
        ----------
        **kwargs
            Additional parameters

        Returns
        -------
        Dict[str, Callable]
            Dictionary with displacement and body force functions
        """
        dimension = kwargs.get("dimension", 2)
        frequency = kwargs.get("frequency", 1.0)

        lmbda, mu = self.lame_parameters()

        if dimension == 2:

            def displacement(x):
                u1 = np.sin(frequency * np.pi * x[0]) * np.cos(frequency * np.pi * x[1])
                u2 = np.cos(frequency * np.pi * x[0]) * np.sin(frequency * np.pi * x[1])
                return np.array([u1, u2])

            def body_force(x):
                # Compute div(σ) for manufactured solution
                f1 = (frequency * np.pi) ** 2 * (
                    (lmbda + 2 * mu)
                    * np.sin(frequency * np.pi * x[0])
                    * np.cos(frequency * np.pi * x[1])
                    + mu
                    * np.sin(frequency * np.pi * x[0])
                    * np.cos(frequency * np.pi * x[1])
                )
                f2 = (frequency * np.pi) ** 2 * (
                    mu
                    * np.cos(frequency * np.pi * x[0])
                    * np.sin(frequency * np.pi * x[1])
                    + (lmbda + 2 * mu)
                    * np.cos(frequency * np.pi * x[0])
                    * np.sin(frequency * np.pi * x[1])
                )
                return np.array([f1, f2])

        elif dimension == 3:

            def displacement(x):
                u1 = (
                    np.sin(frequency * np.pi * x[0])
                    * np.cos(frequency * np.pi * x[1])
                    * np.cos(frequency * np.pi * x[2])
                )
                u2 = (
                    np.cos(frequency * np.pi * x[0])
                    * np.sin(frequency * np.pi * x[1])
                    * np.cos(frequency * np.pi * x[2])
                )
                u3 = (
                    np.cos(frequency * np.pi * x[0])
                    * np.cos(frequency * np.pi * x[1])
                    * np.sin(frequency * np.pi * x[2])
                )
                return np.array([u1, u2, u3])

            def body_force(x):
                # Simplified 3D body force
                scale = (frequency * np.pi) ** 2 * (lmbda + 2 * mu)
                f1 = (
                    scale
                    * np.sin(frequency * np.pi * x[0])
                    * np.cos(frequency * np.pi * x[1])
                    * np.cos(frequency * np.pi * x[2])
                )
                f2 = (
                    scale
                    * np.cos(frequency * np.pi * x[0])
                    * np.sin(frequency * np.pi * x[1])
                    * np.cos(frequency * np.pi * x[2])
                )
                f3 = (
                    scale
                    * np.cos(frequency * np.pi * x[0])
                    * np.cos(frequency * np.pi * x[1])
                    * np.sin(frequency * np.pi * x[2])
                )
                return np.array([f1, f2, f3])

        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

        return {"displacement": displacement, "body_force": body_force}

    def compute_von_mises_stress(
        self, displacement: Any, params: Dict[str, Any] = None
    ) -> Any:
        """Compute von Mises stress from displacement.

        Parameters
        ----------
        displacement : firedrake.Function
            Displacement field
        params : Dict[str, Any], optional
            Parameters

        Returns
        -------
        firedrake.Function
            von Mises stress field
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required")

        # Compute stress tensor
        sigma = self.stress_tensor(displacement, params)

        # von Mises stress: sqrt(3/2 * dev(σ) : dev(σ))
        # For 2D: sqrt(σ_xx^2 - σ_xx*σ_yy + σ_yy^2 + 3*σ_xy^2)
        dim = displacement.geometric_dimension()

        if dim == 2:
            s11, s12 = sigma[0, 0], sigma[0, 1]
            s21, s22 = sigma[1, 0], sigma[1, 1]

            von_mises = fd.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)
        elif dim == 3:
            # Full 3D von Mises calculation
            trace_sigma = fd.tr(sigma)
            dev_sigma = sigma - trace_sigma / 3 * fd.Identity(3)
            von_mises = fd.sqrt(3 / 2 * fd.inner(dev_sigma, dev_sigma))
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        return von_mises


def elasticity(
    trial: Any,
    test: Any,
    E=1.0,
    nu=0.3,
    body_force=None,
    traction=None,
    plane_stress=False,
    **kwargs,
) -> Any:
    """Convenience function for elasticity operator.

    Parameters
    ----------
    trial : Any
        Trial displacement field
    test : Any
        Test displacement field
    E : float, optional
        Young's modulus, by default 1.0
    nu : float, optional
        Poisson's ratio, by default 0.3
    body_force : array-like or Callable, optional
        Body force vector, by default None
    traction : array-like or Callable, optional
        Boundary traction, by default None
    plane_stress : bool, optional
        Use plane stress, by default False
    **kwargs
        Additional parameters

    Returns
    -------
    Any
        Assembled weak form

    Examples
    --------
    >>> # Basic elasticity with body force
    >>> form = elasticity(u, v, E=200e9, nu=0.3, body_force=[0, -9810])
    >>>
    >>> # With traction boundary condition
    >>> form = elasticity(u, v, traction={1: [1000, 0]})
    """
    params = {"E": E, "nu": nu, **kwargs}
    if body_force is not None:
        params["body_force"] = body_force
    if traction is not None:
        params["traction"] = traction

    op = ElasticityOperator(E=E, nu=nu, plane_stress=plane_stress, **kwargs)
    return op(trial, test, params)
