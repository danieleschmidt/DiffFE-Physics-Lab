"""Validation utilities for FEM problems."""

from typing import Any, Dict, List, Optional
import logging

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def validate_mesh(mesh: Any) -> None:
    """Validate mesh object.
    
    Parameters
    ----------
    mesh : firedrake.Mesh
        Mesh to validate
        
    Raises
    ------
    ValidationError
        If mesh is invalid
    """
    if not HAS_FIREDRAKE:
        logger.warning("Cannot validate mesh: Firedrake not available")
        return
    
    if mesh is None:
        raise ValidationError("Mesh cannot be None")
    
    if not isinstance(mesh, fd.MeshGeometry):
        raise ValidationError(f"Expected firedrake.Mesh, got {type(mesh)}")
    
    # Check mesh dimension
    if mesh.geometric_dimension() < 1 or mesh.geometric_dimension() > 3:
        raise ValidationError(f"Unsupported mesh dimension: {mesh.geometric_dimension()}")
    
    # Check for degenerate elements
    if mesh.num_cells() == 0:
        raise ValidationError("Mesh contains no cells")
    
    if mesh.num_vertices() == 0:
        raise ValidationError("Mesh contains no vertices")
    
    # Check mesh quality (simplified)
    try:
        coords = mesh.coordinates.dat.data
        # Would need numpy for these checks
        logger.debug("Mesh coordinates check skipped (numpy not available)")
    except Exception as e:
        logger.warning(f"Could not validate mesh coordinates: {e}")
    
    logger.debug(f"Mesh validation passed: {mesh.num_cells()} cells, "
                f"{mesh.num_vertices()} vertices, dim={mesh.geometric_dimension()}")


def validate_function_space(function_space: Any) -> None:
    """Validate function space object.
    
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
    
    logger.debug(f"Function space validation passed: {function_space.dim()} DOFs")


def validate_boundary_conditions(boundary_conditions: Dict[str, Dict[str, Any]]) -> None:
    """Validate boundary conditions.
    
    Parameters
    ----------
    boundary_conditions : Dict[str, Dict[str, Any]]
        Boundary conditions to validate
        
    Raises
    ------
    ValidationError
        If boundary conditions are invalid
    """
    if not boundary_conditions:
        logger.debug("No boundary conditions to validate")
        return
    
    for bc_name, bc_def in boundary_conditions.items():
        if not isinstance(bc_def, dict):
            raise ValidationError(f"Boundary condition '{bc_name}' must be a dictionary")
        
        # Check required fields
        required_fields = {'type', 'boundary', 'value'}
        missing_fields = required_fields - set(bc_def.keys())
        if missing_fields:
            raise ValidationError(
                f"Boundary condition '{bc_name}' missing fields: {missing_fields}"
            )
        
        # Validate BC type
        bc_type = bc_def['type']
        supported_types = {'dirichlet', 'neumann', 'robin', 'periodic'}
        if bc_type not in supported_types:
            raise ValidationError(
                f"Unsupported BC type '{bc_type}' for '{bc_name}'. "
                f"Supported: {supported_types}"
            )
    
    logger.debug(f"Boundary conditions validation passed: {len(boundary_conditions)} BCs")


def validate_parameters(parameters: Dict[str, Any], 
                       required_params: Optional[List[str]] = None) -> None:
    """Validate problem parameters.
    
    Parameters
    ----------
    parameters : Dict[str, Any]
        Parameters to validate
    required_params : List[str], optional
        List of required parameter names
        
    Raises
    ------
    ValidationError
        If parameters are invalid
    """
    if required_params:
        missing_params = set(required_params) - set(parameters.keys())
        if missing_params:
            raise ValidationError(f"Missing required parameters: {missing_params}")
    
    logger.debug(f"Parameters validation passed: {len(parameters)} parameters")