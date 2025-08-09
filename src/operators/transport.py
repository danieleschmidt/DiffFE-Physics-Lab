"""Transport operators for advection-diffusion and related equations."""

from typing import Dict, Any, Callable
import numpy as np

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from .base import BaseOperator, LinearOperator, NonlinearOperator, register_operator


@register_operator("advection")
class AdvectionOperator(LinearOperator):
    """Advection operator for transport equations.
    
    Implements the advection term: ∇·(u φ) or u·∇φ
    where u is velocity field and φ is transported quantity.
    
    Parameters
    ----------
    velocity : array-like or Callable, optional
        Velocity field, by default None
    upwind : bool, optional
        Use upwind stabilization, by default True
    stabilization : str, optional
        Stabilization method ('none', 'upwind', 'supg'), by default 'upwind'
    **kwargs
        Additional parameters
    """
    
    _is_linear = True
    _is_symmetric = False  # Advection is not symmetric
    
    def __init__(self, velocity=None, upwind=True, stabilization='upwind', **kwargs):
        super().__init__(**kwargs)
        self.velocity = velocity
        self.upwind = upwind
        self.stabilization = stabilization
        
        supported_stabilizations = ['none', 'upwind', 'supg']
        if stabilization not in supported_stabilizations:
            raise ValueError(f"Unsupported stabilization: {stabilization}")
    
    def forward_assembly(
        self, 
        trial: Any, 
        test: Any, 
        params: Dict[str, Any]
    ) -> Any:
        """Assemble advection weak form.
        
        Parameters
        ----------
        trial : firedrake.TrialFunction or Function
            Trial scalar field
        test : firedrake.TestFunction
            Test function
        params : Dict[str, Any]
            Parameters including velocity field
            
        Returns
        -------
        firedrake.Form
            Advection weak form
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        
        self.validate_inputs(trial, test, params)
        
        # Get velocity field
        velocity = params.get('velocity', self.velocity)
        if velocity is None:
            raise ValueError("Velocity field must be provided")
        
        # Convert velocity to appropriate form
        if isinstance(velocity, (list, tuple, np.ndarray)):
            u_vec = fd.Constant(velocity)
        elif callable(velocity):
            # Create vector function space for velocity
            mesh = test.function_space().mesh()
            dim = mesh.geometric_dimension()
            V_vec = fd.VectorFunctionSpace(mesh, "CG", 1)
            u_vec = fd.Function(V_vec)
            u_vec.interpolate(velocity)
        elif hasattr(velocity, 'function_space'):
            u_vec = velocity
        else:
            raise ValueError("Invalid velocity field type")
        
        # Standard advection weak form: -∫ φ (u·∇v) dx + ∫ φ_upwind (u·n) v ds
        # Using integration by parts: ∫ (u·∇φ) v dx - ∫ φ_n (u·n) v ds
        
        if self.stabilization == 'none':
            # Standard Galerkin: ∫ (u·∇φ) v dx
            weak_form = fd.inner(fd.dot(u_vec, fd.grad(trial)), test) * fd.dx
        
        elif self.stabilization == 'upwind':
            # Upwind stabilization using DG methods or artificial diffusion
            weak_form = fd.inner(fd.dot(u_vec, fd.grad(trial)), test) * fd.dx
            
            # Add upwind flux on element boundaries (simplified)
            # This would require proper DG implementation for full upwinding
            
        elif self.stabilization == 'supg':
            # Streamline Upwind Petrov-Galerkin (SUPG)
            h = fd.CellDiameter(test.function_space().mesh())
            u_norm = fd.sqrt(fd.dot(u_vec, u_vec))
            
            # SUPG parameter
            tau = h / (2 * u_norm + 1e-12)  # Avoid division by zero
            
            # Standard term
            weak_form = fd.inner(fd.dot(u_vec, fd.grad(trial)), test) * fd.dx
            
            # SUPG stabilization term
            supg_test = test + tau * fd.dot(u_vec, fd.grad(test))
            weak_form = fd.inner(fd.dot(u_vec, fd.grad(trial)), supg_test) * fd.dx
        
        else:
            raise ValueError(f"Unknown stabilization: {self.stabilization}")
        
        return weak_form
    
    def adjoint_assembly(
        self,
        grad_output: Any,
        trial: Any,
        test: Any,
        params: Dict[str, Any]
    ) -> Any:
        """Assemble adjoint advection operator.
        
        The adjoint of advection operator involves -u·∇φ instead of u·∇φ.
        """
        # Get velocity field
        velocity = params.get('velocity', self.velocity)
        if isinstance(velocity, (list, tuple, np.ndarray)):
            # Negate velocity for adjoint
            adj_velocity = [-v for v in velocity]
            params_adj = params.copy()
            params_adj['velocity'] = adj_velocity
        else:
            # For function-based velocity, would need to negate
            params_adj = params.copy()
            # Simplified - full implementation would properly handle velocity negation
        
        return self.forward_assembly(trial, test, params_adj)
    
    def manufactured_solution(self, **kwargs) -> Dict[str, Callable]:
        """Generate manufactured solution for advection problems.
        
        Parameters
        ----------
        **kwargs
            Additional parameters
            
        Returns
        -------
        Dict[str, Callable]
            Dictionary with solution and source functions
        """
        dimension = kwargs.get('dimension', 2)
        frequency = kwargs.get('frequency', 1.0)
        velocity = kwargs.get('velocity', [1.0, 0.0] if dimension >= 2 else [1.0])
        
        if dimension == 1:
            def solution(x):
                return np.sin(frequency * np.pi * (x[0] - velocity[0] * 0))  # Traveling wave
            
            def source(x):
                # For pure advection, source is zero for traveling wave
                return 0.0
        
        elif dimension == 2:
            def solution(x):
                return np.sin(frequency * np.pi * x[0]) * np.exp(-x[1])
            
            def source(x):
                # Source for manufactured solution with given velocity
                u, v = velocity[0], velocity[1]
                dphidx = frequency * np.pi * np.cos(frequency * np.pi * x[0]) * np.exp(-x[1])
                dphidy = -np.sin(frequency * np.pi * x[0]) * np.exp(-x[1])
                return u * dphidx + v * dphidy
        
        else:
            raise ValueError(f"Manufactured solution not implemented for dimension {dimension}")
        
        return {'solution': solution, 'source': source, 'velocity': velocity}
    
    def compute_peclet_number(self, velocity: Any, diffusivity: float, 
                             mesh_size: float) -> float:
        """Compute cell Péclet number for stability analysis.
        
        Parameters
        ----------
        velocity : Any
            Velocity field
        diffusivity : float
            Diffusion coefficient
        mesh_size : float
            Characteristic mesh size
            
        Returns
        -------
        float
            Péclet number
        """
        if isinstance(velocity, (list, tuple, np.ndarray)):
            u_magnitude = np.sqrt(sum(v**2 for v in velocity))
        else:
            # For function-based velocity, would need proper norm computation
            u_magnitude = 1.0  # Simplified
        
        if diffusivity == 0:
            return np.inf
        
        return u_magnitude * mesh_size / (2 * diffusivity)


def advection(
    trial: Any,
    test: Any,
    velocity,
    stabilization='upwind',
    **kwargs
) -> Any:
    """Convenience function for advection operator.
    
    Parameters
    ----------
    trial : Any
        Trial function
    test : Any
        Test function
    velocity : array-like or Callable or firedrake.Function
        Velocity field
    stabilization : str, optional
        Stabilization method, by default 'upwind'
    **kwargs
        Additional parameters
        
    Returns
    -------
    Any
        Assembled weak form
    """
    params = {'velocity': velocity, 'stabilization': stabilization, **kwargs}
    
    op = AdvectionOperator(
        velocity=velocity,
        stabilization=stabilization,
        **kwargs
    )
    return op(trial, test, params)


@register_operator("advection_diffusion")
class AdvectionDiffusionOperator(LinearOperator):
    """Combined advection-diffusion operator.
    
    Implements: ∂φ/∂t + u·∇φ - ∇·(D∇φ) = f
    where D is diffusivity tensor.
    """
    
    _is_linear = True
    _is_symmetric = False
    
    def __init__(self, velocity=None, diffusivity=1.0, 
                 stabilization='supg', **kwargs):
        super().__init__(**kwargs)
        self.velocity = velocity
        self.diffusivity = diffusivity
        self.stabilization = stabilization
    
    def forward_assembly(self, trial, test, params):
        """Assemble advection-diffusion weak form."""
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required")
        
        # Diffusion term
        D = fd.Constant(params.get('diffusivity', self.diffusivity))
        diffusion_term = D * fd.inner(fd.grad(trial), fd.grad(test)) * fd.dx
        
        # Advection term
        advection_op = AdvectionOperator(
            velocity=params.get('velocity', self.velocity),
            stabilization=self.stabilization
        )
        advection_term = advection_op.forward_assembly(trial, test, params)
        
        return advection_term + diffusion_term
    
    def adjoint_assembly(self, grad_output, trial, test, params):
        """Assemble adjoint advection-diffusion operator."""
        # Diffusion is symmetric, advection needs sign change
        D = fd.Constant(params.get('diffusivity', self.diffusivity))
        diffusion_term = D * fd.inner(fd.grad(trial), fd.grad(test)) * fd.dx
        
        # Adjoint advection
        advection_op = AdvectionOperator(
            velocity=params.get('velocity', self.velocity),
            stabilization=self.stabilization
        )
        advection_term = advection_op.adjoint_assembly(grad_output, trial, test, params)
        
        return advection_term + diffusion_term


def advection_diffusion(
    trial: Any,
    test: Any,
    velocity,
    diffusivity=1.0,
    stabilization='supg',
    source=None,
    **kwargs
) -> Any:
    """Convenience function for advection-diffusion operator.
    
    Parameters
    ----------
    trial : Any
        Trial function
    test : Any
        Test function
    velocity : Any
        Velocity field
    diffusivity : float, optional
        Diffusion coefficient, by default 1.0
    stabilization : str, optional
        Stabilization method, by default 'supg'
    source : Any, optional
        Source term, by default None
    **kwargs
        Additional parameters
        
    Returns
    -------
    Any
        Assembled weak form
    """
    params = {
        'velocity': velocity,
        'diffusivity': diffusivity,
        'stabilization': stabilization,
        **kwargs
    }
    if source is not None:
        params['source'] = source
    
    op = AdvectionDiffusionOperator(
        velocity=velocity,
        diffusivity=diffusivity,
        stabilization=stabilization,
        **kwargs
    )
    
    weak_form = op(trial, test, params)
    
    # Add source term
    if source is not None:
        if not HAS_FIREDRAKE:
            return weak_form
        
        if isinstance(source, (float, int)):
            weak_form -= fd.Constant(source) * test * fd.dx
        elif callable(source):
            source_func = fd.Function(test.function_space())
            source_func.interpolate(source)
            weak_form -= source_func * test * fd.dx
        elif hasattr(source, 'function_space'):
            weak_form -= source * test * fd.dx
    
    return weak_form