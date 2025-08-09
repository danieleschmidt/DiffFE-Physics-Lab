"""Simple validation utilities without external dependencies."""

from typing import Any, Union


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def validate_positive_parameter(value: float, name: str) -> None:
    """Validate that a parameter is positive.
    
    Args:
        value: Parameter value to validate
        name: Parameter name for error messages
        
    Raises:
        ValidationError: If parameter is not positive
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"Parameter '{name}' must be numeric, got {type(value)}")
    
    if value <= 0:
        raise ValidationError(f"Parameter '{name}' must be positive, got {value}")


def validate_parameter_range(value: float, name: str, min_val: float = None, max_val: float = None) -> None:
    """Validate that a parameter is within a specified range.
    
    Args:
        value: Parameter value
        name: Parameter name
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Raises:
        ValidationError: If parameter is outside range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"Parameter '{name}' must be numeric, got {type(value)}")
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"Parameter '{name}' must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"Parameter '{name}' must be <= {max_val}, got {value}")


def validate_string_parameter(value: str, name: str, allowed_values: list = None) -> None:
    """Validate a string parameter.
    
    Args:
        value: String value to validate
        name: Parameter name
        allowed_values: List of allowed values (optional)
        
    Raises:
        ValidationError: If string is invalid
    """
    if not isinstance(value, str):
        raise ValidationError(f"Parameter '{name}' must be a string, got {type(value)}")
    
    if not value.strip():
        raise ValidationError(f"Parameter '{name}' cannot be empty")
    
    # Check for potentially malicious content
    suspicious_patterns = [';', '<', '>', '..', 'eval', 'exec', '__import__']
    for pattern in suspicious_patterns:
        if pattern in value:
            raise ValidationError(f"Parameter '{name}' contains suspicious pattern: {pattern}")
    
    if allowed_values is not None and value not in allowed_values:
        raise ValidationError(f"Parameter '{name}' must be one of {allowed_values}, got '{value}'")


def validate_file_path(path: str) -> None:
    """Validate a file path for security.
    
    Args:
        path: File path to validate
        
    Raises:
        ValidationError: If path is invalid or potentially dangerous
    """
    if not isinstance(path, str):
        raise ValidationError(f"File path must be a string, got {type(path)}")
    
    # Check for path traversal attempts
    if '..' in path:
        raise ValidationError("File path contains path traversal sequence '..'")
    
    if path.startswith('/'):
        raise ValidationError("Absolute paths not allowed")
    
    # Check for Windows reserved names
    windows_reserved = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                       'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                       'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
    
    path_name = path.upper().split('.')[0]  # Get name without extension
    if path_name in windows_reserved:
        raise ValidationError(f"File path uses reserved name: {path_name}")


def validate_coordinates(x: float, y: float, domain_bounds: tuple = None) -> None:
    """Validate coordinate values.
    
    Args:
        x, y: Coordinate values
        domain_bounds: Optional domain bounds as ((xmin, xmax), (ymin, ymax))
        
    Raises:
        ValidationError: If coordinates are invalid
    """
    for coord, name in [(x, 'x'), (y, 'y')]:
        if not isinstance(coord, (int, float)):
            raise ValidationError(f"Coordinate '{name}' must be numeric, got {type(coord)}")
        
        if coord != coord:  # Check for NaN
            raise ValidationError(f"Coordinate '{name}' is NaN")
        
        if coord == float('inf') or coord == float('-inf'):
            raise ValidationError(f"Coordinate '{name}' is infinite")
    
    if domain_bounds is not None:
        (xmin, xmax), (ymin, ymax) = domain_bounds
        if not (xmin <= x <= xmax):
            raise ValidationError(f"x coordinate {x} outside domain [{xmin}, {xmax}]")
        if not (ymin <= y <= ymax):
            raise ValidationError(f"y coordinate {y} outside domain [{ymin}, {ymax}]")


def validate_material_parameters(young_modulus: float = None, poisson_ratio: float = None,
                                density: float = None, conductivity: float = None) -> None:
    """Validate material parameters for physical reasonableness.
    
    Args:
        young_modulus: Young's modulus (Pa)
        poisson_ratio: Poisson's ratio (dimensionless)
        density: Density (kg/m³)
        conductivity: Thermal conductivity (W/m·K)
        
    Raises:
        ValidationError: If parameters are physically unrealistic
    """
    if young_modulus is not None:
        validate_positive_parameter(young_modulus, "young_modulus")
        if young_modulus > 1e15:  # Unrealistically high
            raise ValidationError(f"Young's modulus {young_modulus} Pa is unrealistically high")
    
    if poisson_ratio is not None:
        validate_parameter_range(poisson_ratio, "poisson_ratio", -1.0, 0.5)
        # Most materials have ν between 0 and 0.5
        if not (-0.1 <= poisson_ratio <= 0.5):
            print(f"Warning: Poisson's ratio {poisson_ratio} is outside typical range [0, 0.5]")
    
    if density is not None:
        validate_positive_parameter(density, "density")
        if density > 50000:  # Heavier than lead/gold
            raise ValidationError(f"Density {density} kg/m³ is unrealistically high")
    
    if conductivity is not None:
        validate_positive_parameter(conductivity, "conductivity")
        if conductivity > 1000:  # Higher than copper
            print(f"Warning: Thermal conductivity {conductivity} W/m·K is very high")


def validate_mesh_parameters(num_elements: int, element_order: int = 1) -> None:
    """Validate mesh generation parameters.
    
    Args:
        num_elements: Number of elements
        element_order: Polynomial order of elements
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if not isinstance(num_elements, int):
        raise ValidationError(f"num_elements must be an integer, got {type(num_elements)}")
    
    if num_elements <= 0:
        raise ValidationError(f"num_elements must be positive, got {num_elements}")
    
    if num_elements > 1000000:  # Arbitrary large limit
        raise ValidationError(f"num_elements {num_elements} is too large (max 1,000,000)")
    
    if not isinstance(element_order, int):
        raise ValidationError(f"element_order must be an integer, got {type(element_order)}")
    
    if element_order < 1 or element_order > 10:
        raise ValidationError(f"element_order must be between 1 and 10, got {element_order}")


def validate_solver_parameters(tolerance: float = None, max_iterations: int = None,
                              linear_solver: str = None) -> None:
    """Validate solver parameters.
    
    Args:
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        linear_solver: Linear solver type
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if tolerance is not None:
        validate_positive_parameter(tolerance, "tolerance")
        if tolerance > 1.0:
            raise ValidationError(f"Tolerance {tolerance} is too large (should be < 1.0)")
        if tolerance < 1e-16:
            raise ValidationError(f"Tolerance {tolerance} is too small (minimum 1e-16)")
    
    if max_iterations is not None:
        if not isinstance(max_iterations, int):
            raise ValidationError(f"max_iterations must be an integer, got {type(max_iterations)}")
        
        if max_iterations <= 0:
            raise ValidationError(f"max_iterations must be positive, got {max_iterations}")
        
        if max_iterations > 100000:
            print(f"Warning: max_iterations {max_iterations} is very large")
    
    if linear_solver is not None:
        allowed_solvers = ['direct', 'cg', 'gmres', 'bicgstab', 'mumps', 'superlu']
        validate_string_parameter(linear_solver, "linear_solver", allowed_solvers)


# Basic function space validation without Firedrake
def validate_function_space(function_space: Any) -> None:
    """Mock function space validation."""
    if function_space is None:
        raise ValidationError("Function space cannot be None")
    # Additional validation would require Firedrake


# Basic boundary condition validation
def validate_boundary_conditions(boundary_conditions: dict) -> None:
    """Validate boundary condition definitions.
    
    Args:
        boundary_conditions: Dictionary of boundary conditions
        
    Raises:
        ValidationError: If boundary conditions are invalid
    """
    if not isinstance(boundary_conditions, dict):
        raise ValidationError("Boundary conditions must be a dictionary")
    
    for name, bc in boundary_conditions.items():
        if not isinstance(bc, dict):
            raise ValidationError(f"Boundary condition '{name}' must be a dictionary")
        
        if 'type' not in bc:
            raise ValidationError(f"Boundary condition '{name}' missing 'type'")
        
        bc_type = bc['type']
        if bc_type not in ['dirichlet', 'neumann', 'robin']:
            raise ValidationError(f"Invalid boundary condition type '{bc_type}' for '{name}'")
        
        if 'boundary' not in bc:
            raise ValidationError(f"Boundary condition '{name}' missing 'boundary' marker")
        
        if 'value' not in bc:
            raise ValidationError(f"Boundary condition '{name}' missing 'value'")