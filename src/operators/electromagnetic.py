"""Electromagnetic operators for Maxwell equations."""

from typing import Dict, Any, Callable
import numpy as np

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from .base import BaseOperator, LinearOperator, register_operator


@register_operator("maxwell")
class MaxwellOperator(LinearOperator):
    """Maxwell equations operator for electromagnetic problems.
    
    Implements various formulations of Maxwell equations including:
    - Time-harmonic Maxwell equations
    - Vector wave equation
    - Electrostatic problems
    
    Parameters
    ----------
    formulation : str, optional
        Maxwell formulation ('time_harmonic', 'wave', 'electrostatic'), by default 'time_harmonic'
    epsilon : float, optional
        Permittivity, by default 1.0
    mu : float, optional
        Permeability, by default 1.0
    sigma : float, optional
        Conductivity, by default 0.0
    omega : float, optional
        Angular frequency (for time-harmonic), by default 1.0
    **kwargs
        Additional parameters
    """
    
    _is_linear = True
    _is_symmetric = False  # Generally not symmetric due to boundary conditions
    
    def __init__(self, formulation='time_harmonic', epsilon=1.0, mu=1.0, 
                 sigma=0.0, omega=1.0, **kwargs):
        super().__init__(**kwargs)
        self.formulation = formulation
        self.epsilon = epsilon  # Permittivity
        self.mu = mu  # Permeability
        self.sigma = sigma  # Conductivity
        self.omega = omega  # Angular frequency
        
        supported_formulations = ['time_harmonic', 'wave', 'electrostatic', 'magnetostatic']
        if formulation not in supported_formulations:
            raise ValueError(f"Unsupported formulation: {formulation}")
        
        if epsilon <= 0 or mu <= 0:
            raise ValueError("Permittivity and permeability must be positive")
        if sigma < 0:
            raise ValueError("Conductivity must be non-negative")
    
    def forward_assembly(
        self, 
        trial: Any, 
        test: Any, 
        params: Dict[str, Any]
    ) -> Any:
        """Assemble Maxwell weak form.
        
        Parameters
        ----------
        trial : firedrake.TrialFunction or Function
            Trial field (E, H, or mixed)
        test : firedrake.TestFunction
            Test field
        params : Dict[str, Any]
            Parameters including material properties, sources
            
        Returns
        -------
        firedrake.Form
            Maxwell weak form
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        
        self.validate_inputs(trial, test, params)
        
        # Material parameters
        eps = fd.Constant(params.get('epsilon', self.epsilon))
        mu_param = fd.Constant(params.get('mu', self.mu))
        sig = fd.Constant(params.get('sigma', self.sigma))
        omega = fd.Constant(params.get('omega', self.omega))
        
        if self.formulation == 'time_harmonic':
            # Time-harmonic Maxwell: ∇×(1/μ ∇×E) - ω²εE - iωσE = -iωJ
            # Weak form: ∫ (1/μ)(∇×E)·(∇×v) dx - ω²∫ ε E·v dx - iω∫ σ E·v dx = -iω∫ J·v dx
            
            curl_term = (1 / mu_param) * fd.inner(fd.curl(trial), fd.curl(test)) * fd.dx
            mass_term = omega**2 * eps * fd.inner(trial, test) * fd.dx
            damping_term = omega * sig * fd.inner(trial, test) * fd.dx
            
            # For real formulation, separate real and imaginary parts
            weak_form = curl_term - mass_term - damping_term
        
        elif self.formulation == 'wave':
            # Vector wave equation: ∇×∇×E - k²E = 0, where k² = ω²με
            k_squared = omega**2 * mu_param * eps
            
            weak_form = (fd.inner(fd.curl(trial), fd.curl(test)) * fd.dx -
                        k_squared * fd.inner(trial, test) * fd.dx)
        
        elif self.formulation == 'electrostatic':
            # Electrostatic: ∇·(ε∇φ) = -ρ
            # Weak form: ∫ ε ∇φ·∇v dx = ∫ ρ v dx
            weak_form = eps * fd.inner(fd.grad(trial), fd.grad(test)) * fd.dx
        
        elif self.formulation == 'magnetostatic':
            # Magnetostatic: ∇×(1/μ ∇×A) = J
            # Weak form: ∫ (1/μ)(∇×A)·(∇×v) dx = ∫ J·v dx
            weak_form = (1 / mu_param) * fd.inner(fd.curl(trial), fd.curl(test)) * fd.dx
        
        else:
            raise ValueError(f"Unknown formulation: {self.formulation}")
        
        # Source terms
        if 'current_density' in params:
            J = params['current_density']
            if isinstance(J, (list, tuple, np.ndarray)):
                J_vec = fd.Constant(J)
                if self.formulation in ['time_harmonic', 'magnetostatic']:
                    weak_form -= fd.inner(J_vec, test) * fd.dx
            elif callable(J):
                J_func = fd.Function(test.function_space())
                J_func.interpolate(J)
                weak_form -= fd.inner(J_func, test) * fd.dx
        
        if 'charge_density' in params and self.formulation == 'electrostatic':
            rho = params['charge_density']
            if isinstance(rho, (float, int)):
                rho_const = fd.Constant(rho)
                weak_form -= rho_const * test * fd.dx
            elif callable(rho):
                rho_func = fd.Function(test.function_space())
                rho_func.interpolate(rho)
                weak_form -= rho_func * test * fd.dx
        
        return weak_form
    
    def adjoint_assembly(
        self,
        grad_output: Any,
        trial: Any,
        test: Any,
        params: Dict[str, Any]
    ) -> Any:
        """Assemble adjoint Maxwell operator.
        
        For Maxwell equations, the adjoint involves complex conjugation
        and potentially different boundary conditions.
        """
        # Simplified adjoint - full implementation would handle complex fields
        return self.forward_assembly(trial, test, params)
    
    def manufactured_solution(self, **kwargs) -> Dict[str, Callable]:
        """Generate manufactured solution for Maxwell problems.
        
        Parameters
        ----------
        **kwargs
            Additional parameters
            
        Returns
        -------
        Dict[str, Callable]
            Dictionary with field and source functions
        """
        dimension = kwargs.get('dimension', 3)
        frequency = kwargs.get('frequency', 1.0)
        
        if self.formulation == 'electrostatic' and dimension == 2:
            def potential(x):
                return np.sin(frequency * np.pi * x[0]) * np.sin(frequency * np.pi * x[1])
            
            def charge_density(x):
                return (2 * (frequency * np.pi)**2 * self.epsilon * 
                       np.sin(frequency * np.pi * x[0]) * np.sin(frequency * np.pi * x[1]))
            
            return {'potential': potential, 'charge_density': charge_density}
        
        elif self.formulation == 'time_harmonic' and dimension == 3:
            def electric_field(x):
                # Simple plane wave solution
                Ex = np.sin(frequency * np.pi * x[2])
                Ey = np.cos(frequency * np.pi * x[2])
                Ez = 0.0
                return np.array([Ex, Ey, Ez])
            
            def current_density(x):
                # Corresponding current density
                k = frequency * np.pi
                Jx = -k * self.omega * self.sigma * np.sin(k * x[2])
                Jy = -k * self.omega * self.sigma * np.cos(k * x[2])
                Jz = 0.0
                return np.array([Jx, Jy, Jz])
            
            return {'electric_field': electric_field, 'current_density': current_density}
        
        else:
            raise ValueError(f"Manufactured solution not implemented for {self.formulation} in {dimension}D")
    
    def compute_energy(self, field: Any, params: Dict[str, Any] = None) -> float:
        """Compute electromagnetic energy.
        
        Parameters
        ----------
        field : firedrake.Function
            Electromagnetic field
        params : Dict[str, Any], optional
            Parameters
            
        Returns
        -------
        float
            Energy value
        """
        if not HAS_FIREDRAKE:
            return 0.0
        
        if params is None:
            params = {}
        
        eps = params.get('epsilon', self.epsilon)
        mu_param = params.get('mu', self.mu)
        
        if self.formulation == 'electrostatic':
            # Electric energy: (1/2) ∫ ε |∇φ|² dx
            energy = 0.5 * eps * fd.assemble(fd.inner(fd.grad(field), fd.grad(field)) * fd.dx)
        
        elif self.formulation in ['time_harmonic', 'wave']:
            # Electromagnetic energy (simplified)
            electric_energy = 0.5 * eps * fd.assemble(fd.inner(field, field) * fd.dx)
            magnetic_energy = 0.5 / mu_param * fd.assemble(fd.inner(fd.curl(field), fd.curl(field)) * fd.dx)
            energy = electric_energy + magnetic_energy
        
        else:
            energy = 0.0
        
        return float(energy)


def maxwell(
    trial: Any,
    test: Any,
    formulation='time_harmonic',
    epsilon=1.0,
    mu=1.0,
    sigma=0.0,
    omega=1.0,
    current_density=None,
    charge_density=None,
    **kwargs
) -> Any:
    """Convenience function for Maxwell operator.
    
    Parameters
    ----------
    trial : Any
        Trial electromagnetic field
    test : Any
        Test field
    formulation : str, optional
        Maxwell formulation, by default 'time_harmonic'
    epsilon : float, optional
        Permittivity, by default 1.0
    mu : float, optional
        Permeability, by default 1.0
    sigma : float, optional
        Conductivity, by default 0.0
    omega : float, optional
        Angular frequency, by default 1.0
    current_density : array-like or Callable, optional
        Current density source, by default None
    charge_density : float or Callable, optional
        Charge density source, by default None
    **kwargs
        Additional parameters
        
    Returns
    -------
    Any
        Assembled weak form
    """
    params = {
        'formulation': formulation,
        'epsilon': epsilon,
        'mu': mu,
        'sigma': sigma,
        'omega': omega,
        **kwargs
    }
    
    if current_density is not None:
        params['current_density'] = current_density
    if charge_density is not None:
        params['charge_density'] = charge_density
    
    op = MaxwellOperator(
        formulation=formulation,
        epsilon=epsilon,
        mu=mu,
        sigma=sigma,
        omega=omega,
        **kwargs
    )
    return op(trial, test, params)