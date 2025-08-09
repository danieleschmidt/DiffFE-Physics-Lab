"""Nonlinear operators for hyperelastic and other nonlinear problems."""

from typing import Dict, Any, Callable
import numpy as np

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from .base import BaseOperator, NonlinearOperator, register_operator


@register_operator("hyperelastic")
class HyperelasticOperator(NonlinearOperator):
    """Hyperelastic operator for nonlinear solid mechanics.
    
    Implements hyperelastic materials using strain energy density functions.
    Supports Neo-Hookean and other hyperelastic models.
    
    Parameters
    ----------
    material_model : str, optional
        Material model ('neo_hookean', 'mooney_rivlin'), by default 'neo_hookean'
    mu : float, optional
        Shear modulus, by default 1.0
    lmbda : float, optional
        Lamé parameter, by default 1.0
    **kwargs
        Additional parameters
    """
    
    _is_linear = False
    _is_symmetric = True  # Conservative problems
    
    def __init__(self, material_model='neo_hookean', mu=1.0, lmbda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.material_model = material_model
        self.mu = mu  # Shear modulus
        self.lmbda = lmbda  # Lamé parameter
        
        if mu <= 0 or lmbda < 0:
            raise ValueError("Material parameters must be non-negative")
        
        supported_models = ['neo_hookean', 'mooney_rivlin', 'saint_venant']
        if material_model not in supported_models:
            raise ValueError(f"Unsupported material model: {material_model}")
    
    def deformation_gradient(self, displacement: Any) -> Any:
        """Compute deformation gradient F = I + ∇u.
        
        Parameters
        ----------
        displacement : firedrake.Function
            Displacement field
            
        Returns
        -------
        firedrake.Expression
            Deformation gradient tensor
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required")
        
        dim = displacement.geometric_dimension()
        I = fd.Identity(dim)
        F = I + fd.grad(displacement)
        return F
    
    def right_cauchy_green(self, F: Any) -> Any:
        """Compute right Cauchy-Green tensor C = F^T F.
        
        Parameters
        ----------
        F : firedrake.Expression
            Deformation gradient
            
        Returns
        -------
        firedrake.Expression
            Right Cauchy-Green tensor
        """
        return F.T * F
    
    def invariants(self, C: Any) -> tuple:
        """Compute strain invariants.
        
        Parameters
        ----------
        C : firedrake.Expression
            Right Cauchy-Green tensor
            
        Returns
        -------
        tuple
            (I1, I2, I3) strain invariants
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required")
        
        I1 = fd.tr(C)
        I2 = 0.5 * (fd.tr(C)**2 - fd.tr(C * C))
        I3 = fd.det(C)
        
        return I1, I2, I3
    
    def strain_energy_density(self, displacement: Any, params: Dict[str, Any]) -> Any:
        """Compute strain energy density function.
        
        Parameters
        ----------
        displacement : firedrake.Function
            Displacement field
        params : Dict[str, Any]
            Material parameters
            
        Returns
        -------
        firedrake.Expression
            Strain energy density
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required")
        
        # Material parameters
        mu = params.get('mu', self.mu)
        lmbda = params.get('lambda', self.lmbda)
        
        # Deformation gradient and invariants
        F = self.deformation_gradient(displacement)
        C = self.right_cauchy_green(F)
        I1, I2, I3 = self.invariants(C)
        J = fd.sqrt(I3)  # Jacobian determinant
        
        # Strain energy based on material model
        if self.material_model == 'neo_hookean':
            # Neo-Hookean: W = μ/2(I₁ - 3) - μ ln(J) + λ/2(ln(J))²
            W = (mu / 2) * (I1 - 3) - mu * fd.ln(J) + (lmbda / 2) * (fd.ln(J))**2
        
        elif self.material_model == 'mooney_rivlin':
            # Mooney-Rivlin: W = C₁(I₁ - 3) + C₂(I₂ - 3) + bulk terms
            c1 = params.get('c1', mu / 4)  # C₁ = μ/4 for neo-Hookean limit
            c2 = params.get('c2', 0.0)     # C₂ = 0 for neo-Hookean
            W = c1 * (I1 - 3) + c2 * (I2 - 3) + (lmbda / 2) * (J - 1)**2
        
        elif self.material_model == 'saint_venant':
            # Saint Venant-Kirchhoff (for small strains)
            E = 0.5 * (C - fd.Identity(C.ufl_shape[0]))  # Green-Lagrange strain
            W = (lmbda / 2) * (fd.tr(E))**2 + mu * fd.tr(E * E)
        
        else:
            raise ValueError(f"Unknown material model: {self.material_model}")
        
        return W
    
    def first_piola_kirchhoff_stress(self, displacement: Any, params: Dict[str, Any]) -> Any:
        """Compute first Piola-Kirchhoff stress tensor.
        
        Parameters
        ----------
        displacement : firedrake.Function
            Displacement field
        params : Dict[str, Any]
            Parameters
            
        Returns
        -------
        firedrake.Expression
            First PK stress tensor P = ∂W/∂F
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required")
        
        # This would require automatic differentiation of strain energy
        # Simplified implementation using derivative
        W = self.strain_energy_density(displacement, params)
        F = self.deformation_gradient(displacement)
        
        # P = ∂W/∂F (this is a simplified approach)
        # In practice, this would be computed analytically for each material model
        return fd.derivative(W, F)
    
    def forward_assembly(
        self, 
        trial: Any, 
        test: Any, 
        params: Dict[str, Any]
    ) -> Any:
        """Assemble hyperelastic weak form.
        
        Parameters
        ----------
        trial : firedrake.Function
            Trial displacement field
        test : firedrake.TestFunction
            Test displacement field
        params : Dict[str, Any]
            Parameters including material properties
            
        Returns
        -------
        firedrake.Form
            Hyperelastic weak form
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        
        self.validate_inputs(trial, test, params)
        
        # Material parameters
        mu = fd.Constant(params.get('mu', self.mu))
        lmbda = fd.Constant(params.get('lambda', self.lmbda))
        
        # Deformation gradient
        dim = trial.geometric_dimension()
        I = fd.Identity(dim)
        F = I + fd.grad(trial)
        C = F.T * F
        
        # Strain invariants
        Ic = fd.tr(C)
        J = fd.det(F)
        
        # First Piola-Kirchhoff stress (simplified neo-Hookean)
        if self.material_model == 'neo_hookean':
            P = mu * (F - fd.inv(F.T)) + lmbda * fd.ln(J) * fd.inv(F.T)
        elif self.material_model == 'saint_venant':
            E = 0.5 * (C - I)  # Green-Lagrange strain
            S = lmbda * fd.tr(E) * I + 2 * mu * E  # Second PK stress
            P = F * S  # First PK stress
        else:
            # Simplified for other models
            P = mu * F + lmbda * (J - 1) * J * fd.inv(F.T)
        
        # Weak form: ∫ P : ∇v dx
        weak_form = fd.inner(P, fd.grad(test)) * fd.dx
        
        # Body forces
        if 'body_force' in params:
            f = params['body_force']
            if isinstance(f, (list, tuple, np.ndarray)):
                f_vec = fd.Constant(f)
                weak_form -= fd.inner(f_vec, test) * fd.dx
            elif callable(f):
                f_func = fd.Function(test.function_space())
                f_func.interpolate(f)
                weak_form -= fd.inner(f_func, test) * fd.dx
        
        return weak_form
    
    def adjoint_assembly(
        self,
        grad_output: Any,
        trial: Any,
        test: Any,
        params: Dict[str, Any]
    ) -> Any:
        """Assemble adjoint hyperelastic operator."""
        # For conservative hyperelastic problems, adjoint equals forward
        return self.forward_assembly(trial, test, params)
    
    def linearize(
        self,
        solution: Any,
        params: Dict[str, Any] = None
    ) -> 'LinearizedHyperelastic':
        """Linearize hyperelastic operator around solution.
        
        Parameters
        ----------
        solution : firedrake.Function
            Solution to linearize around
        params : Dict[str, Any], optional
            Parameters
            
        Returns
        -------
        LinearizedHyperelastic
            Linearized operator
        """
        return LinearizedHyperelastic(
            reference_displacement=solution,
            material_model=self.material_model,
            mu=self.mu,
            lmbda=self.lmbda,
            **params or {}
        )
    
    def manufactured_solution(self, **kwargs) -> Dict[str, Callable]:
        """Generate manufactured solution for hyperelastic problems.
        
        Parameters
        ----------
        **kwargs
            Additional parameters
            
        Returns
        -------
        Dict[str, Callable]
            Dictionary with displacement and body force functions
        """
        dimension = kwargs.get('dimension', 2)
        amplitude = kwargs.get('amplitude', 0.1)  # Small deformation
        
        if dimension == 2:
            def displacement(x):
                u1 = amplitude * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])
                u2 = amplitude * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
                return np.array([u1, u2])
            
            def body_force(x):
                # Simplified body force for small deformations
                f1 = -amplitude * (self.mu + self.lmbda) * np.pi**2 * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])
                f2 = -amplitude * (self.mu + self.lmbda) * np.pi**2 * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
                return np.array([f1, f2])
        
        else:
            raise ValueError(f"Manufactured solution not implemented for dimension {dimension}")
        
        return {'displacement': displacement, 'body_force': body_force}


class LinearizedHyperelastic(BaseOperator):
    """Linearized hyperelastic operator for Newton iterations."""
    
    _is_linear = True
    _is_symmetric = True
    
    def __init__(self, reference_displacement, material_model='neo_hookean', 
                 mu=1.0, lmbda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.reference_displacement = reference_displacement
        self.material_model = material_model
        self.mu = mu
        self.lmbda = lmbda
    
    def forward_assembly(self, trial, test, params):
        """Assemble linearized hyperelastic operator."""
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required")
        
        # This would implement the tangent stiffness matrix
        # Simplified implementation - would need full hyperelastic linearization
        mu = fd.Constant(self.mu)
        lmbda = fd.Constant(self.lmbda)
        
        # Linear elasticity approximation for tangent
        def epsilon(u):
            return fd.sym(fd.grad(u))
        
        def sigma(u):
            return lmbda * fd.tr(epsilon(u)) * fd.Identity(len(u)) + 2 * mu * epsilon(u)
        
        return fd.inner(sigma(trial), epsilon(test)) * fd.dx
    
    def adjoint_assembly(self, grad_output, trial, test, params):
        """Assemble adjoint of linearized operator."""
        return self.forward_assembly(trial, test, params)


def hyperelastic(
    trial: Any,
    test: Any,
    material_model='neo_hookean',
    mu=1.0,
    lmbda=1.0,
    body_force=None,
    **kwargs
) -> Any:
    """Convenience function for hyperelastic operator.
    
    Parameters
    ----------
    trial : Any
        Trial displacement field
    test : Any
        Test displacement field
    material_model : str, optional
        Material model, by default 'neo_hookean'
    mu : float, optional
        Shear modulus, by default 1.0
    lmbda : float, optional
        Lamé parameter, by default 1.0
    body_force : array-like or Callable, optional
        Body force, by default None
    **kwargs
        Additional parameters
        
    Returns
    -------
    Any
        Assembled weak form
    """
    params = {'material_model': material_model, 'mu': mu, 'lambda': lmbda, **kwargs}
    if body_force is not None:
        params['body_force'] = body_force
    
    op = HyperelasticOperator(
        material_model=material_model,
        mu=mu,
        lmbda=lmbda,
        **kwargs
    )
    return op(trial, test, params)