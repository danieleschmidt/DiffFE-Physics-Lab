"""Validation utilities for FEM problems."""

from typing import Any, Dict, List, Optional
import numpy as np
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
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            raise ValidationError("Mesh contains invalid coordinates (NaN or Inf)")
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
        validate_mesh(function_space.mesh())\n    except ValidationError as e:\n        raise ValidationError(f"Function space has invalid mesh: {e}")\n    \n    # Check degrees of freedom\n    if function_space.dim() == 0:\n        raise ValidationError("Function space has zero degrees of freedom")\n    \n    # Check element family and degree\n    try:\n        element = function_space.ufl_element()\n        family = element.family()\n        degree = element.degree()\n        \n        if degree < 1:\n            logger.warning(f"Low polynomial degree: {degree}")\n            \n        # Check for supported element families\n        supported_families = {\n            'Lagrange', 'CG', 'DG', 'Discontinuous Lagrange',\n            'Raviart-Thomas', 'RT', 'Brezzi-Douglas-Marini', 'BDM',\n            'Nedelec 1st kind', 'N1curl', 'Nedelec 2nd kind', 'N2curl'\n        }\n        \n        if family not in supported_families:\n            logger.warning(f"Element family '{family}' may not be fully supported")\n    \n    except Exception as e:\n        logger.warning(f"Could not validate element properties: {e}")\n    \n    logger.debug(f"Function space validation passed: {function_space.dim()} DOFs")\n\n\ndef validate_boundary_conditions(\n    boundary_conditions: Dict[str, Dict[str, Any]]\n) -> None:\n    """Validate boundary conditions.\n    \n    Parameters\n    ----------\n    boundary_conditions : Dict[str, Dict[str, Any]]\n        Boundary conditions to validate\n        \n    Raises\n    ------\n    ValidationError\n        If boundary conditions are invalid\n    """\n    if not boundary_conditions:\n        logger.debug("No boundary conditions to validate")\n        return\n    \n    for bc_name, bc_def in boundary_conditions.items():\n        if not isinstance(bc_def, dict):\n            raise ValidationError(f"Boundary condition '{bc_name}' must be a dictionary")\n        \n        # Check required fields\n        required_fields = {'type', 'boundary', 'value'}\n        missing_fields = required_fields - set(bc_def.keys())\n        if missing_fields:\n            raise ValidationError(\n                f"Boundary condition '{bc_name}' missing fields: {missing_fields}"\n            )\n        \n        # Validate BC type\n        bc_type = bc_def['type']\n        supported_types = {'dirichlet', 'neumann', 'robin', 'periodic'}\n        if bc_type not in supported_types:\n            raise ValidationError(\n                f"Unsupported BC type '{bc_type}' for '{bc_name}'. "\n                f"Supported: {supported_types}"\n            )\n        \n        # Validate boundary identifier\n        boundary_id = bc_def['boundary']\n        if not isinstance(boundary_id, (int, str, list, tuple)):\n            raise ValidationError(\n                f"Boundary identifier for '{bc_name}' must be int, str, or list/tuple"\n            )\n        \n        # Validate value\n        value = bc_def['value']\n        if not (callable(value) or isinstance(value, (int, float, complex)) or\n                (HAS_FIREDRAKE and isinstance(value, (fd.Function, fd.Constant)))):\n            raise ValidationError(\n                f"Boundary value for '{bc_name}' must be callable, scalar, "\n                f"or Firedrake Function/Constant"\n            )\n        \n        # Type-specific validation\n        if bc_type == 'robin':\n            if 'robin_coeff' not in bc_def:\n                raise ValidationError(\n                    f"Robin BC '{bc_name}' missing 'robin_coeff' parameter"\n                )\n    \n    logger.debug(f"Boundary conditions validation passed: {len(boundary_conditions)} BCs")\n\n\ndef validate_parameters(\n    parameters: Dict[str, Any],\n    required_params: Optional[List[str]] = None,\n    param_bounds: Optional[Dict[str, tuple]] = None\n) -> None:\n    """Validate problem parameters.\n    \n    Parameters\n    ----------\n    parameters : Dict[str, Any]\n        Parameters to validate\n    required_params : List[str], optional\n        List of required parameter names\n    param_bounds : Dict[str, tuple], optional\n        Parameter bounds as {name: (min, max)}\n        \n    Raises\n    ------\n    ValidationError\n        If parameters are invalid\n    """\n    if required_params:\n        missing_params = set(required_params) - set(parameters.keys())\n        if missing_params:\n            raise ValidationError(f"Missing required parameters: {missing_params}")\n    \n    # Check parameter bounds\n    if param_bounds:\n        for param_name, bounds in param_bounds.items():\n            if param_name in parameters:\n                value = parameters[param_name]\n                if isinstance(value, (int, float)):\n                    min_val, max_val = bounds\n                    if value < min_val or value > max_val:\n                        raise ValidationError(\n                            f"Parameter '{param_name}' = {value} outside bounds [{min_val}, {max_val}]"\n                        )\n    \n    # Check for invalid values\n    for param_name, value in parameters.items():\n        if isinstance(value, (int, float)):\n            if np.isnan(value) or np.isinf(value):\n                raise ValidationError(\n                    f"Parameter '{param_name}' has invalid value: {value}"\n                )\n    \n    logger.debug(f"Parameters validation passed: {len(parameters)} parameters")\n\n\ndef validate_convergence_history(\n    convergence_history: List[Dict[str, Any]]\n) -> None:\n    """Validate convergence history data.\n    \n    Parameters\n    ----------\n    convergence_history : List[Dict[str, Any]]\n        Convergence history to validate\n        \n    Raises\n    ------\n    ValidationError\n        If convergence history is invalid\n    """\n    if not convergence_history:\n        return\n    \n    required_fields = {'iteration', 'residual_norm'}\n    \n    for i, entry in enumerate(convergence_history):\n        if not isinstance(entry, dict):\n            raise ValidationError(f"Convergence entry {i} must be a dictionary")\n        \n        missing_fields = required_fields - set(entry.keys())\n        if missing_fields:\n            raise ValidationError(\n                f"Convergence entry {i} missing fields: {missing_fields}"\n            )\n        \n        # Check for valid values\n        iteration = entry['iteration']\n        if not isinstance(iteration, int) or iteration < 0:\n            raise ValidationError(\n                f"Convergence entry {i}: iteration must be non-negative integer"\n            )\n        \n        residual = entry['residual_norm']\n        if not isinstance(residual, (int, float)) or residual < 0:\n            raise ValidationError(\n                f"Convergence entry {i}: residual_norm must be non-negative number"\n            )\n        \n        if np.isnan(residual) or np.isinf(residual):\n            raise ValidationError(\n                f"Convergence entry {i}: residual_norm is NaN or Inf"\n            )\n    \n    # Check monotonicity (warning only)\n    residuals = [entry['residual_norm'] for entry in convergence_history]\n    if len(residuals) > 1:\n        non_decreasing_count = sum(\n            1 for i in range(1, len(residuals)) \n            if residuals[i] > residuals[i-1]\n        )\n        if non_decreasing_count > len(residuals) * 0.5:\n            logger.warning("Convergence history shows poor convergence behavior")\n    \n    logger.debug(f"Convergence history validation passed: {len(convergence_history)} entries")\n\n\ndef check_numerical_stability(\n    matrix: Any,\n    condition_threshold: float = 1e12\n) -> Dict[str, Any]:\n    """Check numerical stability of system matrix.\n    \n    Parameters\n    ----------\n    matrix : Any\n        System matrix to check\n    condition_threshold : float, optional\n        Condition number threshold for warning, by default 1e12\n        \n    Returns\n    -------\n    Dict[str, Any]\n        Stability analysis results\n    """\n    if not HAS_FIREDRAKE:\n        return {'status': 'skipped', 'reason': 'Firedrake not available'}\n    \n    try:\n        # Convert to numpy if needed\n        if hasattr(matrix, 'M'):\n            # PETSc matrix\n            A = matrix.M.handle.getDenseArray()\n        elif hasattr(matrix, 'toarray'):\n            # Sparse matrix\n            A = matrix.toarray()\n        else:\n            A = np.array(matrix)\n        \n        # Basic checks\n        if A.size == 0:\n            return {'status': 'error', 'message': 'Empty matrix'}\n        \n        if np.any(np.isnan(A)) or np.any(np.isinf(A)):\n            return {'status': 'error', 'message': 'Matrix contains NaN or Inf values'}\n        \n        # Compute condition number (for small matrices only)\n        if A.shape[0] <= 1000:  # Avoid expensive computation for large matrices\n            try:\n                cond_num = np.linalg.cond(A)\n                \n                if cond_num > condition_threshold:\n                    status = 'warning'\n                    message = f'High condition number: {cond_num:.2e}'\n                else:\n                    status = 'ok'\n                    message = f'Condition number: {cond_num:.2e}'\n                \n                return {\n                    'status': status,\n                    'condition_number': cond_num,\n                    'message': message,\n                    'matrix_shape': A.shape,\n                    'matrix_norm': np.linalg.norm(A)\n                }\n            except np.linalg.LinAlgError:\n                return {\n                    'status': 'warning',\n                    'message': 'Could not compute condition number (singular matrix?)'\n                }\n        else:\n            return {\n                'status': 'ok',\n                'message': f'Matrix too large for condition number computation ({A.shape})'\n            }\n    \n    except Exception as e:\n        return {\n            'status': 'error',\n            'message': f'Stability check failed: {e}'\n        }\n\n\ndef validate_solution(\n    solution: Any,\n    function_space: Any = None,\n    bounds: Optional[tuple] = None\n) -> None:\n    """Validate computed solution.\n    \n    Parameters\n    ----------\n    solution : firedrake.Function\n        Solution to validate\n    function_space : firedrake.FunctionSpace, optional\n        Expected function space\n    bounds : tuple, optional\n        Expected solution bounds (min, max)\n        \n    Raises\n    ------\n    ValidationError\n        If solution is invalid\n    """\n    if not HAS_FIREDRAKE:\n        logger.warning("Cannot validate solution: Firedrake not available")\n        return\n    \n    if solution is None:\n        raise ValidationError("Solution cannot be None")\n    \n    if not isinstance(solution, fd.Function):\n        raise ValidationError(f"Expected firedrake.Function, got {type(solution)}")\n    \n    # Check function space compatibility\n    if function_space is not None:\n        if solution.function_space() != function_space:\n            raise ValidationError("Solution function space does not match expected")\n    \n    # Check for invalid values\n    try:\n        data = solution.dat.data\n        if np.any(np.isnan(data)):\n            raise ValidationError("Solution contains NaN values")\n        if np.any(np.isinf(data)):\n            raise ValidationError("Solution contains infinite values")\n        \n        # Check bounds if provided\n        if bounds is not None:\n            min_val, max_val = bounds\n            sol_min, sol_max = np.min(data), np.max(data)\n            \n            if sol_min < min_val or sol_max > max_val:\n                logger.warning(\n                    f"Solution values [{sol_min:.3e}, {sol_max:.3e}] "\n                    f"outside expected bounds [{min_val}, {max_val}]"\n                )\n    \n    except Exception as e:\n        logger.warning(f"Could not validate solution data: {e}")\n    \n    logger.debug("Solution validation passed")