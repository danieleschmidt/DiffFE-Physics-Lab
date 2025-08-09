"""Fluid dynamics operators for Navier-Stokes equations."""

from typing import Dict, Any, Callable, Tuple
import numpy as np

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from .base import BaseOperator, NonlinearOperator, register_operator


@register_operator("navier_stokes")
class NavierStokesOperator(NonlinearOperator):
    """Navier-Stokes operator for fluid flow.
    
    Implements the weak form for incompressible Navier-Stokes:
    ∫ (∂u/∂t + u·∇u) · v dx + ν ∫ ∇u : ∇v dx - ∫ p ∇·v dx = ∫ f · v dx
    ∫ q ∇·u dx = 0
    
    where u is velocity, p is pressure, ν is kinematic viscosity.
    
    Parameters
    ----------
    nu : float, optional
        Kinematic viscosity, by default 1.0
    steady : bool, optional
        Whether to solve steady-state problem, by default True
    **kwargs
        Additional parameters
    """
    
    _is_linear = False
    _is_symmetric = False
    
    def __init__(self, nu=1.0, steady=True, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu  # Kinematic viscosity
        self.steady = steady
        
        if nu <= 0:
            raise ValueError(f"Viscosity must be positive, got {nu}")
    
    def forward_assembly(
        self, 
        trial: Any, 
        test: Any, 
        params: Dict[str, Any]
    ) -> Any:
        """Assemble Navier-Stokes weak form.
        
        Parameters
        ----------
        trial : firedrake.Function or tuple
            Trial functions (velocity, pressure) or mixed function
        test : firedrake.TestFunction or tuple
            Test functions (velocity, pressure) or mixed test function
        params : Dict[str, Any]
            Parameters including body forces, velocity
            
        Returns
        -------
        firedrake.Form
            Weak form for Navier-Stokes
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        
        self.validate_inputs(trial, test, params)
        
        # Handle mixed function spaces
        if hasattr(trial, 'split'):
            # Mixed function space
            u, p = fd.split(trial)
            v, q = fd.split(test)
        else:
            # Separate functions (velocity only for simplified case)
            u, p = trial, None
            v, q = test, None
        
        nu = params.get('nu', self.nu)
        nu = fd.Constant(nu)
        
        # Viscous term: ν ∫ ∇u : ∇v dx
        F = nu * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
        
        # Convection term: ∫ (u·∇u) · v dx (nonlinear)
        if not self.steady or 'velocity' in params:
            velocity = params.get('velocity', u)
            F += fd.inner(fd.dot(velocity, fd.nabla_grad(u)), v) * fd.dx
        
        # Pressure term: -∫ p ∇·v dx
        if p is not None and q is not None:
            F -= p * fd.div(v) * fd.dx
            # Incompressibility: ∫ q ∇·u dx = 0
            F += q * fd.div(u) * fd.dx
        
        # Body force: ∫ f · v dx
        if 'body_force' in params:
            f = params['body_force']
            if isinstance(f, (list, tuple, np.ndarray)):
                f_vec = fd.Constant(f)
                F -= fd.inner(f_vec, v) * fd.dx
            elif callable(f):
                f_func = fd.Function(v.function_space())
                f_func.interpolate(f)
                F -= fd.inner(f_func, v) * fd.dx
        
        # Time derivative (if unsteady)
        if not self.steady and 'u_old' in params and 'dt' in params:
            u_old = params['u_old']
            dt = fd.Constant(params['dt'])
            F += fd.inner((u - u_old) / dt, v) * fd.dx
        
        return F
    
    def adjoint_assembly(
        self,
        grad_output: Any,
        trial: Any,
        test: Any,
        params: Dict[str, Any]
    ) -> Any:
        """Assemble adjoint Navier-Stokes operator.
        
        For nonlinear NS, the adjoint involves the transpose of linearization.
        """
        # Simplified adjoint - full implementation would require careful treatment
        return self.forward_assembly(trial, test, params)
    
    def linearize(
        self,
        solution: Any,
        params: Dict[str, Any] = None
    ) -> 'LinearizedNavierStokes':
        """Linearize Navier-Stokes around a solution.
        
        Parameters
        ----------
        solution : firedrake.Function
            Solution to linearize around
        params : Dict[str, Any], optional
            Parameters
            
        Returns
        -------
        LinearizedNavierStokes
            Linearized operator
        """
        return LinearizedNavierStokes(
            background_velocity=solution,
            nu=self.nu,
            steady=self.steady,
            **params or {}
        )


class LinearizedNavierStokes(BaseOperator):
    """Linearized Navier-Stokes operator."""
    
    _is_linear = True
    _is_symmetric = False
    
    def __init__(self, background_velocity, nu=1.0, steady=True, **kwargs):
        super().__init__(**kwargs)
        self.background_velocity = background_velocity
        self.nu = nu
        self.steady = steady
    
    def forward_assembly(self, trial, test, params):
        """Assemble linearized Navier-Stokes."""
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required")
        
        if hasattr(trial, 'split'):
            u, p = fd.split(trial)
            v, q = fd.split(test)
        else:
            u, p = trial, None
            v, q = test, None
        
        nu = fd.Constant(self.nu)
        u_bg = self.background_velocity
        
        # Linearized form
        F = nu * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
        
        # Linearized convection: ∫ (u_bg·∇u) · v dx + ∫ (u·∇u_bg) · v dx
        F += fd.inner(fd.dot(u_bg, fd.nabla_grad(u)), v) * fd.dx
        F += fd.inner(fd.dot(u, fd.nabla_grad(u_bg)), v) * fd.dx
        
        # Pressure terms
        if p is not None and q is not None:
            F -= p * fd.div(v) * fd.dx
            F += q * fd.div(u) * fd.dx
        
        return F
    
    def adjoint_assembly(self, grad_output, trial, test, params):
        """Assemble adjoint of linearized operator."""
        return self.forward_assembly(trial, test, params)


def navier_stokes(
    trial: Any,
    test: Any,
    nu=1.0,
    body_force=None,
    steady=True,
    **kwargs
) -> Any:
    """Convenience function for Navier-Stokes operator.
    
    Parameters
    ----------
    trial : Any
        Trial function(s)
    test : Any
        Test function(s)
    nu : float, optional
        Kinematic viscosity, by default 1.0
    body_force : array-like or Callable, optional
        Body force, by default None
    steady : bool, optional
        Steady-state problem, by default True
    **kwargs
        Additional parameters
        
    Returns
    -------
    Any
        Assembled weak form
    """
    params = {'nu': nu, 'steady': steady, **kwargs}
    if body_force is not None:
        params['body_force'] = body_force
    
    op = NavierStokesOperator(nu=nu, steady=steady, **kwargs)
    return op(trial, test, params)


def incompressibility(
    velocity_trial: Any,
    pressure_test: Any,
    **kwargs
) -> Any:
    """Incompressibility constraint: ∇·u = 0.
    
    Parameters
    ----------
    velocity_trial : Any
        Velocity trial function
    pressure_test : Any
        Pressure test function
    **kwargs
        Additional parameters
        
    Returns
    -------
    Any
        Incompressibility weak form
    """
    if not HAS_FIREDRAKE:
        raise ImportError("Firedrake required")
    
    return pressure_test * fd.div(velocity_trial) * fd.dx