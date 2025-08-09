"""Enhanced validation utilities with physics constraints and robustness features."""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import logging
import numpy as np
import math
import re

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from .exceptions import (
    ValidationError, MeshValidationError, PhysicsConstraintError,
    handle_mesh_error, validate_and_raise
)

logger = logging.getLogger(__name__)


# Physical constants and constraints
PHYSICAL_CONSTANTS = {
    'speed_of_light': 299792458.0,  # m/s
    'boltzmann_constant': 1.380649e-23,  # J/K
    'gas_constant': 8.314462618,  # J/(mol·K)
    'stefan_boltzmann': 5.670374419e-8,  # W/(m²·K⁴)
    'planck_constant': 6.62607015e-34,  # J·s
    'permittivity_vacuum': 8.8541878128e-12,  # F/m
    'permeability_vacuum': 1.25663706212e-6,  # H/m
}

# Common parameter bounds for physical problems
PARAMETER_BOUNDS = {
    'temperature': (0.0, 1e6),  # Kelvin
    'pressure': (0.0, 1e12),  # Pa
    'density': (0.0, 1e6),  # kg/m³
    'viscosity': (0.0, 1e6),  # Pa·s
    'thermal_conductivity': (0.0, 1e6),  # W/(m·K)
    'specific_heat': (0.0, 1e6),  # J/(kg·K)
    'young_modulus': (0.0, 1e12),  # Pa
    'poisson_ratio': (-1.0, 0.5),  # dimensionless
    'diffusion_coefficient': (0.0, 1e6),  # m²/s
    'reaction_rate': (0.0, 1e6),  # 1/s
    'magnetic_permeability': (0.0, 1e6),  # H/m
    'electric_permittivity': (0.0, 1e6),  # F/m
    'velocity': (0.0, PHYSICAL_CONSTANTS['speed_of_light']),  # m/s
    'frequency': (0.0, 1e15),  # Hz
    'wavelength': (1e-15, 1e15),  # m
    'dt': (1e-12, 1e6),  # time step bounds
    'dx': (1e-12, 1e6),  # spatial step bounds
}


def validate_mesh(mesh: Any, check_quality: bool = True) -> None:
    """Validate mesh object with comprehensive quality checks.
    
    Parameters
    ----------
    mesh : firedrake.Mesh
        Mesh to validate
    check_quality : bool, optional
        Whether to perform quality checks, by default True
        
    Raises
    ------
    MeshValidationError
        If mesh is invalid
    """
    if not HAS_FIREDRAKE:
        logger.warning("Cannot validate mesh: Firedrake not available")
        return
    
    if mesh is None:
        handle_mesh_error(mesh, "Mesh cannot be None")
    
    if not isinstance(mesh, fd.MeshGeometry):
        handle_mesh_error(mesh, f"Expected firedrake.Mesh, got {type(mesh)}")
    
    # Check mesh dimension
    dim = mesh.geometric_dimension()
    if dim < 1 or dim > 3:
        handle_mesh_error(mesh, f"Unsupported mesh dimension: {dim}")
    
    # Check for degenerate elements
    num_cells = mesh.num_cells()
    num_vertices = mesh.num_vertices()
    
    if num_cells == 0:
        handle_mesh_error(mesh, "Mesh contains no cells")
    
    if num_vertices == 0:
        handle_mesh_error(mesh, "Mesh contains no vertices")
    
    # Basic topology checks
    if num_vertices < dim + 1:
        handle_mesh_error(mesh, f"Insufficient vertices ({num_vertices}) for {dim}D mesh")
    
    # Check mesh quality if requested
    if check_quality:
        _check_mesh_quality(mesh)
    
    logger.debug(f"Mesh validation passed: {num_cells} cells, "
                f"{num_vertices} vertices, dim={dim}")


def _check_mesh_quality(mesh: Any) -> None:
    """Check mesh quality metrics.
    
    Parameters
    ----------
    mesh : firedrake.Mesh
        Mesh to check
        
    Raises
    ------
    MeshValidationError
        If mesh quality is poor
    """
    try:
        # Get mesh coordinates
        coords = mesh.coordinates.dat.data_ro
        dim = mesh.geometric_dimension()
        
        # Check for duplicate vertices
        unique_coords = np.unique(coords, axis=0)
        if len(unique_coords) < len(coords) * 0.95:  # Allow some tolerance
            logger.warning("Mesh may contain duplicate vertices")
        
        # Check coordinate ranges
        coord_ranges = np.ptp(coords, axis=0)
        if np.any(coord_ranges < 1e-12):
            handle_mesh_error(mesh, "Mesh has degenerate coordinate range")
        
        # Check for extreme aspect ratios (simplified)
        max_range = np.max(coord_ranges)
        min_range = np.min(coord_ranges)
        aspect_ratio = max_range / min_range if min_range > 0 else float('inf')
        
        if aspect_ratio > 1e6:
            logger.warning(f"Mesh has extreme aspect ratio: {aspect_ratio:.2e}")
        
        # Check for coordinates outside reasonable bounds
        max_coord = np.max(np.abs(coords))
        if max_coord > 1e12:
            logger.warning(f"Mesh has very large coordinates: {max_coord:.2e}")
        
        logger.debug(f"Mesh quality check passed: aspect_ratio={aspect_ratio:.2e}")
        
    except Exception as e:
        logger.warning(f"Could not perform detailed mesh quality check: {e}")


def validate_function_space(function_space: Any) -> None:
    """Validate function space object with enhanced checks.
    
    Parameters
    ----------
    function_space : firedrake.FunctionSpace
        Function space to validate
        
    Raises
    ------
    ValidationError
        If function space is invalid
    """
    if not HAS_FIREDRAKE:
        logger.warning("Cannot validate function space: Firedrake not available")
        return
    
    if function_space is None:
        raise ValidationError("Function space cannot be None")
    
    if not isinstance(function_space, (fd.FunctionSpace, fd.MixedFunctionSpace)):
        raise ValidationError(f"Expected firedrake.FunctionSpace, got {type(function_space)}")
    
    # Validate underlying mesh
    try:
        validate_mesh(function_space.mesh())
    except ValidationError as e:
        raise ValidationError(f"Function space has invalid mesh: {e}")
    
    # Check degrees of freedom
    if function_space.dim() == 0:
        raise ValidationError("Function space has zero degrees of freedom")
    
    # Check for very large DOF counts that might cause memory issues
    dof_count = function_space.dim()
    if dof_count > 1e8:
        logger.warning(f"Very large DOF count: {dof_count} - may cause memory issues")
    
    logger.debug(f"Function space validation passed: {dof_count} DOFs")


def validate_boundary_conditions(
    boundary_conditions: Dict[str, Dict[str, Any]], 
    mesh_info: Optional[Dict[str, Any]] = None
) -> None:
    """Validate boundary conditions with enhanced checks.
    
    Parameters
    ----------
    boundary_conditions : Dict[str, Dict[str, Any]]
        Boundary conditions to validate
    mesh_info : Dict[str, Any], optional
        Mesh information for compatibility checks
        
    Raises
    ------
    ValidationError
        If boundary conditions are invalid
    """
    if not boundary_conditions:
        logger.debug("No boundary conditions to validate")
        return
    
    for bc_name, bc_def in boundary_conditions.items():
        validate_and_raise(
            isinstance(bc_def, dict),
            f"Boundary condition '{bc_name}' must be a dictionary",
            ValidationError,
            invalid_field=bc_name
        )
        
        # Check required fields
        required_fields = {'type', 'boundary', 'value'}
        missing_fields = required_fields - set(bc_def.keys())
        validate_and_raise(
            not missing_fields,
            f"Boundary condition '{bc_name}' missing fields: {missing_fields}",
            ValidationError,
            invalid_field=bc_name
        )
        
        # Validate BC type
        bc_type = bc_def['type']
        supported_types = {'dirichlet', 'neumann', 'robin', 'periodic'}
        validate_and_raise(
            bc_type in supported_types,
            f"Unsupported BC type '{bc_type}' for '{bc_name}'",
            ValidationError,
            invalid_field='type',
            constraint=f"Must be one of {supported_types}"
        )
        
        # Validate boundary identifier
        boundary_id = bc_def['boundary']
        if isinstance(boundary_id, str):
            # Named boundary - validate string format
            validate_and_raise(
                re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', boundary_id),
                f"Invalid boundary name '{boundary_id}' - must be valid identifier",
                ValidationError,
                invalid_field='boundary'
            )
        elif isinstance(boundary_id, (int, list, tuple)):
            # Numeric or multiple boundaries
            if isinstance(boundary_id, (list, tuple)):
                for bid in boundary_id:
                    validate_and_raise(
                        isinstance(bid, int) and bid >= 0,
                        f"Invalid boundary ID {bid} - must be non-negative integer",
                        ValidationError,
                        invalid_field='boundary'
                    )
            else:
                validate_and_raise(
                    boundary_id >= 0,
                    f"Invalid boundary ID {boundary_id} - must be non-negative",
                    ValidationError,
                    invalid_field='boundary'
                )
        
        # Validate boundary value
        _validate_boundary_value(bc_def['value'], bc_type, bc_name)
    
    logger.debug(f"Boundary conditions validation passed: {len(boundary_conditions)} BCs")


def _validate_boundary_value(value: Any, bc_type: str, bc_name: str) -> None:
    """Validate boundary condition value."""
    if callable(value):
        # Function-based boundary condition
        logger.debug(f"Boundary condition '{bc_name}' uses function value")
        return
    
    if bc_type in ['dirichlet', 'neumann']:
        # Should be numeric or array
        if isinstance(value, (int, float)):
            validate_and_raise(
                math.isfinite(value),
                f"Boundary value for '{bc_name}' must be finite",
                ValidationError,
                invalid_field='value'
            )
        elif hasattr(value, '__iter__'):
            # Array-like value
            try:
                array_value = np.asarray(value)
                validate_and_raise(
                    np.all(np.isfinite(array_value)),
                    f"All boundary values for '{bc_name}' must be finite",
                    ValidationError,
                    invalid_field='value'
                )
            except Exception as e:
                raise ValidationError(
                    f"Invalid array boundary value for '{bc_name}': {e}",
                    invalid_field='value'
                )
        else:
            raise ValidationError(
                f"Unsupported boundary value type for '{bc_name}': {type(value)}",
                invalid_field='value',
                expected_type=Union[float, list, callable]
            )


def validate_parameters(
    parameters: Dict[str, Any], 
    required_params: Optional[List[str]] = None,
    parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    physics_constraints: Optional[Dict[str, Callable]] = None
) -> None:
    """Validate problem parameters with physics constraints.
    
    Parameters
    ----------
    parameters : Dict[str, Any]
        Parameters to validate
    required_params : List[str], optional
        List of required parameter names
    parameter_bounds : Dict[str, Tuple[float, float]], optional
        Custom parameter bounds (min, max)
    physics_constraints : Dict[str, Callable], optional
        Custom physics constraint functions
        
    Raises
    ------
    ValidationError
        If parameters are invalid
    PhysicsConstraintError
        If physics constraints are violated
    """
    # Check required parameters
    if required_params:
        missing_params = set(required_params) - set(parameters.keys())
        validate_and_raise(
            not missing_params,
            f"Missing required parameters: {missing_params}",
            ValidationError,
            context={'missing_parameters': list(missing_params)}
        )
    
    # Validate each parameter
    bounds = {**PARAMETER_BOUNDS, **(parameter_bounds or {})}
    
    for param_name, param_value in parameters.items():
        # Basic type validation
        if param_value is None:
            continue  # Allow None values
        
        # Validate numeric parameters
        if isinstance(param_value, (int, float)):
            validate_numeric_parameter(param_name, param_value, bounds)
        
        # Validate array parameters
        elif hasattr(param_value, '__iter__') and not isinstance(param_value, str):
            validate_array_parameter(param_name, param_value, bounds)
        
        # Validate string parameters (identifiers, file paths, etc.)
        elif isinstance(param_value, str):
            validate_string_parameter(param_name, param_value)
        
        # Check physics constraints
        if physics_constraints and param_name in physics_constraints:
            constraint_func = physics_constraints[param_name]
            try:
                constraint_satisfied = constraint_func(param_value)
                if not constraint_satisfied:
                    raise PhysicsConstraintError(
                        f"Physics constraint violated for parameter '{param_name}'",
                        parameter_name=param_name,
                        parameter_value=param_value
                    )
            except Exception as e:
                raise PhysicsConstraintError(
                    f"Error checking physics constraint for '{param_name}': {e}",
                    parameter_name=param_name,
                    parameter_value=param_value
                )
    
    # Cross-parameter validation
    _validate_parameter_relationships(parameters)
    
    logger.debug(f"Parameters validation passed: {len(parameters)} parameters")


def validate_numeric_parameter(
    param_name: str, 
    param_value: Union[int, float],
    bounds: Dict[str, Tuple[float, float]]
) -> None:
    """Validate numeric parameter value and bounds."""
    # Check if value is finite
    validate_and_raise(
        math.isfinite(param_value),
        f"Parameter '{param_name}' must be finite (got {param_value})",
        ValidationError,
        invalid_field=param_name,
        actual_value=param_value
    )
    
    # Check bounds
    if param_name in bounds:
        min_val, max_val = bounds[param_name]
        if param_value < min_val or param_value > max_val:
            raise PhysicsConstraintError(
                f"Parameter '{param_name}' = {param_value} outside valid range [{min_val}, {max_val}]",
                parameter_name=param_name,
                parameter_value=param_value,
                valid_range=(min_val, max_val)
            )
    
    # Special checks for common physical parameters
    if param_name == 'poisson_ratio':
        # Poisson's ratio must be > -1 and < 0.5 for physically meaningful materials
        if not (-1.0 < param_value < 0.5):
            raise PhysicsConstraintError(
                f"Poisson's ratio {param_value} outside physically meaningful range (-1, 0.5)",
                parameter_name=param_name,
                parameter_value=param_value,
                physics_law="Material stability requires -1 < ν < 0.5"
            )
    
    elif param_name in ['density', 'viscosity', 'thermal_conductivity']:
        # These must be strictly positive
        if param_value <= 0:
            raise PhysicsConstraintError(
                f"Physical parameter '{param_name}' = {param_value} must be positive",
                parameter_name=param_name,
                parameter_value=param_value,
                physics_law="Physical properties must be positive"
            )


def validate_array_parameter(
    param_name: str,
    param_value: Any,
    bounds: Dict[str, Tuple[float, float]]
) -> None:
    """Validate array-like parameter."""
    try:
        array_value = np.asarray(param_value, dtype=float)
    except Exception as e:
        raise ValidationError(
            f"Cannot convert parameter '{param_name}' to numeric array: {e}",
            invalid_field=param_name,
            actual_value=str(param_value)
        )
    
    # Check for finite values
    validate_and_raise(
        np.all(np.isfinite(array_value)),
        f"All values in parameter '{param_name}' must be finite",
        ValidationError,
        invalid_field=param_name
    )
    
    # Check bounds if specified
    if param_name in bounds:
        min_val, max_val = bounds[param_name]
        if np.any(array_value < min_val) or np.any(array_value > max_val):
            raise PhysicsConstraintError(
                f"Some values in parameter '{param_name}' outside valid range [{min_val}, {max_val}]",
                parameter_name=param_name,
                parameter_value=f"array of shape {array_value.shape}",
                valid_range=(min_val, max_val)
            )


def validate_string_parameter(param_name: str, param_value: str) -> None:
    """Validate string parameter."""
    # Basic string validation
    validate_and_raise(
        isinstance(param_value, str),
        f"Parameter '{param_name}' must be a string",
        ValidationError,
        invalid_field=param_name,
        expected_type=str,
        actual_value=type(param_value).__name__
    )
    
    # Length check
    validate_and_raise(
        len(param_value) <= 1000,
        f"Parameter '{param_name}' string too long (max 1000 characters)",
        ValidationError,
        invalid_field=param_name
    )
    
    # Check for dangerous content
    dangerous_patterns = ['../', '\\', '<script', 'javascript:', 'eval(', 'exec(']
    for pattern in dangerous_patterns:
        if pattern in param_value.lower():
            logger.warning(f"Potentially dangerous content in parameter '{param_name}'")


def _validate_parameter_relationships(parameters: Dict[str, Any]) -> None:
    """Validate relationships between parameters."""
    # Check Reynolds number consistency
    if all(p in parameters for p in ['velocity', 'density', 'viscosity', 'length_scale']):
        try:
            velocity = float(parameters['velocity'])
            density = float(parameters['density'])
            viscosity = float(parameters['viscosity'])
            length = float(parameters['length_scale'])
            
            if viscosity > 0:
                reynolds = (velocity * density * length) / viscosity
                if reynolds > 1e8:
                    logger.warning(f"Very high Reynolds number: {reynolds:.2e}")
        except:
            pass  # Skip if conversion fails
    
    # Check CFL condition parameters
    if all(p in parameters for p in ['velocity', 'dx', 'dt']):
        try:
            velocity = float(parameters['velocity'])
            dx = float(parameters['dx'])
            dt = float(parameters['dt'])
            
            if dx > 0 and velocity > 0:
                cfl = (velocity * dt) / dx
                if cfl > 1.0:
                    logger.warning(f"CFL condition may be violated: CFL = {cfl:.3f} > 1")
        except:
            pass
    
    # Check Péclet number for advection-diffusion
    if all(p in parameters for p in ['velocity', 'diffusion_coefficient', 'length_scale']):
        try:
            velocity = float(parameters['velocity'])
            diffusion = float(parameters['diffusion_coefficient'])
            length = float(parameters['length_scale'])
            
            if diffusion > 0:
                peclet = (velocity * length) / diffusion
                if peclet > 100:
                    logger.warning(f"High Péclet number: {peclet:.2e} - consider stabilization")
        except:
            pass


def validate_physics_consistency(
    problem_type: str,
    parameters: Dict[str, Any]
) -> None:
    """Validate physics consistency for specific problem types.
    
    Parameters
    ----------
    problem_type : str
        Type of physics problem (e.g., 'elasticity', 'fluid', 'thermal')
    parameters : Dict[str, Any]
        Problem parameters
        
    Raises
    ------
    PhysicsConstraintError
        If physics consistency is violated
    """
    if problem_type == 'elasticity':
        _validate_elasticity_parameters(parameters)
    elif problem_type == 'fluid':
        _validate_fluid_parameters(parameters)
    elif problem_type == 'thermal':
        _validate_thermal_parameters(parameters)
    elif problem_type == 'electromagnetic':
        _validate_electromagnetic_parameters(parameters)


def _validate_elasticity_parameters(parameters: Dict[str, Any]) -> None:
    """Validate elasticity problem parameters."""
    # Check material stability
    if 'young_modulus' in parameters and 'poisson_ratio' in parameters:
        E = parameters['young_modulus']
        nu = parameters['poisson_ratio']
        
        # Bulk modulus must be positive
        if E > 0 and nu >= 0.5:
            raise PhysicsConstraintError(
                f"Material unstable: E={E}, ν={nu} gives negative bulk modulus",
                physics_law="Bulk modulus K = E/(3(1-2ν)) must be positive"
            )
        
        # Shear modulus must be positive
        if E <= 0:
            raise PhysicsConstraintError(
                f"Young's modulus {E} must be positive",
                parameter_name='young_modulus',
                parameter_value=E,
                physics_law="Material stiffness must be positive"
            )


def _validate_fluid_parameters(parameters: Dict[str, Any]) -> None:
    """Validate fluid mechanics parameters."""
    # Check viscosity positivity
    if 'viscosity' in parameters:
        mu = parameters['viscosity']
        if mu < 0:
            raise PhysicsConstraintError(
                f"Viscosity {mu} cannot be negative",
                parameter_name='viscosity',
                parameter_value=mu,
                physics_law="Viscosity must be non-negative for physical fluids"
            )


def _validate_thermal_parameters(parameters: Dict[str, Any]) -> None:
    """Validate heat transfer parameters."""
    # Check thermal diffusivity consistency
    if all(p in parameters for p in ['thermal_conductivity', 'density', 'specific_heat']):
        k = parameters['thermal_conductivity']
        rho = parameters['density']
        cp = parameters['specific_heat']
        
        if k <= 0 or rho <= 0 or cp <= 0:
            raise PhysicsConstraintError(
                "Thermal properties must be positive",
                context={'k': k, 'rho': rho, 'cp': cp},
                physics_law="Thermal diffusivity α = k/(ρcp) requires all positive values"
            )


def _validate_electromagnetic_parameters(parameters: Dict[str, Any]) -> None:
    """Validate electromagnetic parameters."""
    # Check Maxwell's equations consistency
    if 'frequency' in parameters:
        freq = parameters['frequency']
        if freq < 0:
            raise PhysicsConstraintError(
                f"Frequency {freq} cannot be negative",
                parameter_name='frequency',
                parameter_value=freq
            )
        
        # Check wavelength consistency
        if 'wavelength' in parameters:
            wavelength = parameters['wavelength']
            c = PHYSICAL_CONSTANTS['speed_of_light']
            expected_freq = c / wavelength if wavelength > 0 else float('inf')
            
            if abs(freq - expected_freq) / expected_freq > 0.1:  # 10% tolerance
                logger.warning(
                    f"Frequency-wavelength inconsistency: f={freq:.2e} Hz, "
                    f"λ={wavelength:.2e} m, expected f={expected_freq:.2e} Hz"
                )


# Utility functions for creating specialized validators

def create_parameter_validator(
    problem_type: str,
    custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    custom_constraints: Optional[Dict[str, Callable]] = None
) -> Callable:
    """Create a parameter validator for a specific problem type.
    
    Parameters
    ----------
    problem_type : str
        Type of physics problem
    custom_bounds : Dict[str, Tuple[float, float]], optional
        Custom parameter bounds
    custom_constraints : Dict[str, Callable], optional
        Custom constraint functions
        
    Returns
    -------
    Callable
        Parameter validation function
    """
    def validator(parameters: Dict[str, Any], required_params: List[str] = None) -> None:
        # Basic validation
        validate_parameters(
            parameters, 
            required_params,
            custom_bounds,
            custom_constraints
        )
        
        # Physics-specific validation
        validate_physics_consistency(problem_type, parameters)
    
    return validator


def validate_optimization_bounds(
    bounds: Dict[str, Tuple[float, float]],
    parameters: Dict[str, Any]
) -> None:
    """Validate optimization parameter bounds.
    
    Parameters
    ----------
    bounds : Dict[str, Tuple[float, float]]
        Parameter bounds for optimization
    parameters : Dict[str, Any]
        Initial parameter values
        
    Raises
    ------
    ValidationError
        If bounds are invalid
    """
    for param_name, (lower, upper) in bounds.items():
        # Check bounds validity
        validate_and_raise(
            lower < upper,
            f"Invalid bounds for '{param_name}': lower={lower} >= upper={upper}",
            ValidationError,
            invalid_field=param_name
        )
        
        validate_and_raise(
            math.isfinite(lower) and math.isfinite(upper),
            f"Bounds for '{param_name}' must be finite",
            ValidationError,
            invalid_field=param_name
        )
        
        # Check if initial value is within bounds
        if param_name in parameters:
            value = parameters[param_name]
            if isinstance(value, (int, float)):
                validate_and_raise(
                    lower <= value <= upper,
                    f"Initial value for '{param_name}' = {value} outside bounds [{lower}, {upper}]",
                    ValidationError,
                    invalid_field=param_name
                )


def validate_time_stepping_parameters(parameters: Dict[str, Any]) -> None:
    """Validate time-stepping parameters for transient problems.
    
    Parameters
    ----------
    parameters : Dict[str, Any]
        Time-stepping parameters
        
    Raises
    ------
    ValidationError
        If time-stepping parameters are invalid
    """
    required_params = ['dt', 'T_final']
    for param in required_params:
        validate_and_raise(
            param in parameters,
            f"Missing required time parameter: {param}",
            ValidationError,
            invalid_field=param
        )
    
    dt = parameters['dt']
    T_final = parameters['T_final']
    
    validate_and_raise(
        dt > 0,
        f"Time step dt = {dt} must be positive",
        ValidationError,
        invalid_field='dt'
    )
    
    validate_and_raise(
        T_final > 0,
        f"Final time T_final = {T_final} must be positive",
        ValidationError,
        invalid_field='T_final'
    )
    
    validate_and_raise(
        dt < T_final,
        f"Time step dt = {dt} must be smaller than final time T_final = {T_final}",
        ValidationError,
        context={'dt': dt, 'T_final': T_final}
    )
    
    # Check for reasonable number of time steps
    num_steps = int(T_final / dt)
    if num_steps > 1e6:
        logger.warning(f"Very large number of time steps: {num_steps}")
    elif num_steps < 10:
        logger.warning(f"Very small number of time steps: {num_steps}")


def validate_convergence_criteria(criteria: Dict[str, Any]) -> None:
    """Validate solver convergence criteria.
    
    Parameters
    ----------
    criteria : Dict[str, Any]
        Convergence criteria parameters
        
    Raises
    ------
    ValidationError
        If convergence criteria are invalid
    """
    if 'tolerance' in criteria:
        tol = criteria['tolerance']
        validate_and_raise(
            tol > 0,
            f"Tolerance {tol} must be positive",
            ValidationError,
            invalid_field='tolerance'
        )
        
        if tol < 1e-16:
            logger.warning(f"Very tight tolerance: {tol} - may cause convergence issues")
        elif tol > 1e-3:
            logger.warning(f"Very loose tolerance: {tol} - may affect solution accuracy")
    
    if 'max_iterations' in criteria:
        max_iter = criteria['max_iterations']
        validate_and_raise(
            max_iter > 0,
            f"Max iterations {max_iter} must be positive",
            ValidationError,
            invalid_field='max_iterations'
        )
        
        if max_iter > 10000:
            logger.warning(f"Very large max iterations: {max_iter}")
        elif max_iter < 10:
            logger.warning(f"Very small max iterations: {max_iter}")


def get_default_bounds(parameter_name: str) -> Optional[Tuple[float, float]]:
    """Get default bounds for a parameter.
    
    Parameters
    ----------
    parameter_name : str
        Name of the parameter
        
    Returns
    -------
    Optional[Tuple[float, float]]
        Default bounds (min, max) or None if not defined
    """
    return PARAMETER_BOUNDS.get(parameter_name)


def list_physical_constants() -> Dict[str, float]:
    """Get dictionary of physical constants.
    
    Returns
    -------
    Dict[str, float]
        Physical constants with their values
    """
    return PHYSICAL_CONSTANTS.copy()