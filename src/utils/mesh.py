"""Basic mesh generation utilities for FEM without Firedrake.

Enhanced with robust error handling, validation, monitoring, and security features.
"""

import numpy as np
import time
import logging
from typing import Tuple, Optional, List
from dataclasses import dataclass

# Import robust infrastructure
from ..robust.error_handling import (
    ValidationError, validate_positive, validate_range, error_context
)
from ..robust.logging_system import (
    get_logger, log_performance, global_audit_logger
)
from ..robust.monitoring import resource_monitor
from ..robust.security import (
    global_security_validator, global_input_sanitizer, SecurityError
)

logger = get_logger(__name__)


@dataclass
class SimpleMesh:
    """Simple mesh representation for basic FEM operations."""
    nodes: np.ndarray  # Node coordinates
    elements: np.ndarray  # Element connectivity
    boundary_nodes: dict  # Boundary node indices by name
    dimension: int  # Spatial dimension
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the mesh."""
        return self.nodes.shape[0]
    
    @property
    def num_elements(self) -> int:
        """Number of elements in the mesh."""
        return self.elements.shape[0]
    
    @property
    def num_nodes_per_element(self) -> int:
        """Number of nodes per element."""
        return self.elements.shape[1]


@log_performance("create_1d_mesh")
def create_1d_mesh(x_start: float = 0.0, x_end: float = 1.0, 
                   num_elements: int = 10, validate_inputs: bool = True) -> SimpleMesh:
    """Create a simple 1D mesh with robust validation and monitoring.
    
    Parameters
    ----------
    x_start : float, optional
        Starting x coordinate, by default 0.0
    x_end : float, optional
        Ending x coordinate, by default 1.0
    num_elements : int, optional
        Number of elements, by default 10
    validate_inputs : bool, optional
        Enable input validation, by default True
        
    Returns
    -------
    SimpleMesh
        1D mesh with linear elements
        
    Raises
    ------
    ValidationError
        If input parameters are invalid
    SecurityError
        If security validation fails
    """
    with error_context("create_1d_mesh", x_start=x_start, x_end=x_end, num_elements=num_elements):
        # Input validation
        if validate_inputs:
            validate_range(x_start, -1e6, 1e6, "x_start")
            validate_range(x_end, -1e6, 1e6, "x_end")
            if x_end <= x_start:
                raise ValidationError("x_end must be greater than x_start",
                                    field="domain", x_start=x_start, x_end=x_end)
            validate_range(num_elements, 1, 100000, "num_elements")
        
        # Security validation
        global_security_validator.validate_input([x_start, x_end, num_elements], "1d_mesh_params")
        
        with resource_monitor("1d_mesh_creation", num_elements=num_elements):
            try:
                # Create nodes
                nodes = np.linspace(x_start, x_end, num_elements + 1)
                nodes = nodes.reshape(-1, 1)  # Make it 2D array
                
                # Validate generated nodes
                if not np.isfinite(nodes).all():
                    raise ValidationError("Generated nodes contain NaN/inf values",
                                        field="nodes")
                
                # Create elements (line elements)
                elements = np.zeros((num_elements, 2), dtype=int)
                for i in range(num_elements):
                    elements[i] = [i, i + 1]
                
                # Define boundary nodes
                boundary_nodes = {
                    "left": np.array([0]),
                    "right": np.array([num_elements])
                }
                
                mesh = SimpleMesh(
                    nodes=nodes, 
                    elements=elements, 
                    boundary_nodes=boundary_nodes,
                    dimension=1
                )
                
                # Log mesh creation
                global_audit_logger.log_data_operation(
                    "create", "1D_mesh", num_elements,
                    num_nodes=len(nodes), domain=[x_start, x_end]
                )
                
                logger.debug(f"Created 1D mesh: {len(nodes)} nodes, {num_elements} elements, "
                           f"domain=[{x_start}, {x_end}]")
                
                return mesh
                
            except Exception as e:
                logger.error(f"Error creating 1D mesh: {e}", exc_info=True)
                raise


@log_performance("create_2d_rectangle_mesh")
def create_2d_rectangle_mesh(x_range: Tuple[float, float] = (0.0, 1.0),
                             y_range: Tuple[float, float] = (0.0, 1.0),
                             nx: int = 10, ny: int = 10, validate_inputs: bool = True) -> SimpleMesh:
    """Create a simple 2D rectangular mesh with triangular elements and robust validation.
    
    Parameters
    ----------
    x_range : Tuple[float, float], optional
        X coordinate range, by default (0.0, 1.0)
    y_range : Tuple[float, float], optional
        Y coordinate range, by default (0.0, 1.0)
    nx : int, optional
        Number of divisions in x direction, by default 10
    ny : int, optional
        Number of divisions in y direction, by default 10
    validate_inputs : bool, optional
        Enable input validation, by default True
        
    Returns
    -------
    SimpleMesh
        2D mesh with triangular elements
        
    Raises
    ------
    ValidationError
        If input parameters are invalid
    SecurityError
        If security validation fails
    """
    with error_context("create_2d_rectangle_mesh", x_range=x_range, y_range=y_range, nx=nx, ny=ny):
        # Input validation
        if validate_inputs:
            x_min, x_max = x_range
            y_min, y_max = y_range
            
            validate_range(x_min, -1e6, 1e6, "x_min")
            validate_range(x_max, -1e6, 1e6, "x_max")
            if x_max <= x_min:
                raise ValidationError("x_max must be greater than x_min",
                                    field="x_range", x_min=x_min, x_max=x_max)
            
            validate_range(y_min, -1e6, 1e6, "y_min")
            validate_range(y_max, -1e6, 1e6, "y_max")
            if y_max <= y_min:
                raise ValidationError("y_max must be greater than y_min",
                                    field="y_range", y_min=y_min, y_max=y_max)
            
            validate_range(nx, 1, 10000, "nx")
            validate_range(ny, 1, 10000, "ny")
            
            # Check for reasonable mesh size
            total_elements = 2 * nx * ny  # Two triangles per rectangle
            if total_elements > 1000000:  # 1M elements
                logger.warning(f"Large mesh requested: {total_elements} elements")
        
        # Security validation
        global_security_validator.validate_input(list(x_range) + list(y_range) + [nx, ny], "2d_mesh_params")
        
        with resource_monitor("2d_mesh_creation", nx=nx, ny=ny, total_elements=2*nx*ny):
            try:
                x_min, x_max = x_range
                y_min, y_max = y_range
                
                # Create structured grid nodes
                x_coords = np.linspace(x_min, x_max, nx + 1)
                y_coords = np.linspace(y_min, y_max, ny + 1)
                
                # Validate generated coordinates
                if not (np.isfinite(x_coords).all() and np.isfinite(y_coords).all()):
                    raise ValidationError("Generated coordinates contain NaN/inf values",
                                        field="coordinates")
                
                nodes = []
                for j in range(ny + 1):
                    for i in range(nx + 1):
                        nodes.append([x_coords[i], y_coords[j]])
                
                nodes = np.array(nodes)
                
                # Create triangular elements
                elements = []
                for j in range(ny):
                    for i in range(nx):
                        # Each rectangular cell creates two triangles
                        n1 = j * (nx + 1) + i       # bottom-left
                        n2 = j * (nx + 1) + i + 1   # bottom-right
                        n3 = (j + 1) * (nx + 1) + i # top-left
                        n4 = (j + 1) * (nx + 1) + i + 1  # top-right
                        
                        # Validate node indices
                        max_node_idx = len(nodes) - 1
                        for node_idx in [n1, n2, n3, n4]:
                            if node_idx > max_node_idx:
                                raise ValidationError(
                                    f"Invalid node index {node_idx} > {max_node_idx}",
                                    field="element_connectivity"
                                )
                        
                        # First triangle (bottom-left)
                        elements.append([n1, n2, n3])
                        # Second triangle (top-right)
                        elements.append([n2, n4, n3])
                
                elements = np.array(elements, dtype=int)
                
                # Define boundary nodes with validation
                try:
                    boundary_nodes = {
                        "left": np.array([j * (nx + 1) for j in range(ny + 1)]),
                        "right": np.array([j * (nx + 1) + nx for j in range(ny + 1)]),
                        "bottom": np.array(list(range(nx + 1))),
                        "top": np.array([ny * (nx + 1) + i for i in range(nx + 1)])
                    }
                    
                    # Validate boundary node indices
                    max_node_idx = len(nodes) - 1
                    for boundary_name, boundary_indices in boundary_nodes.items():
                        if np.any(boundary_indices > max_node_idx):
                            raise ValidationError(
                                f"Invalid boundary node indices for {boundary_name}",
                                field="boundary_nodes", boundary=boundary_name
                            )
                        
                except Exception as e:
                    raise ValidationError(f"Error creating boundary nodes: {e}",
                                        field="boundary_nodes") from e
                
                mesh = SimpleMesh(
                    nodes=nodes,
                    elements=elements,
                    boundary_nodes=boundary_nodes,
                    dimension=2
                )
                
                # Log mesh creation
                global_audit_logger.log_data_operation(
                    "create", "2D_mesh", len(elements),
                    num_nodes=len(nodes), nx=nx, ny=ny,
                    x_range=x_range, y_range=y_range
                )
                
                logger.debug(f"Created 2D mesh: {len(nodes)} nodes, {len(elements)} elements, "
                           f"domain={x_range}×{y_range}, resolution={nx}×{ny}")
                
                return mesh
                
            except Exception as e:
                logger.error(f"Error creating 2D mesh: {e}", exc_info=True)
                raise


class SimpleFunctionSpace:
    """Simple function space for basic FEM operations with robust validation."""
    
    def __init__(self, mesh: SimpleMesh, element_type: str = "P1", validate_inputs: bool = True):
        """Initialize function space with robust validation.
        
        Parameters
        ----------
        mesh : SimpleMesh
            Underlying mesh
        element_type : str, optional
            Element type ("P1" for linear), by default "P1"
        validate_inputs : bool, optional
            Enable input validation, by default True
        """
        with error_context("SimpleFunctionSpace_initialization"):
            # Input validation
            if validate_inputs:
                if mesh is None:
                    raise ValidationError("Mesh cannot be None", field="mesh")
                
                if not isinstance(mesh, SimpleMesh):
                    raise ValidationError("Mesh must be SimpleMesh instance",
                                        field="mesh", type=type(mesh))
                
                if mesh.num_nodes <= 0:
                    raise ValidationError("Mesh must have positive number of nodes",
                                        field="num_nodes", value=mesh.num_nodes)
                
                # Sanitize element type
                element_type = global_input_sanitizer.sanitize_string(element_type)
            
            # Security validation
            global_security_validator.validate_input(element_type, "element_type")
            
            self.mesh = mesh
            self.element_type = element_type
            self.validate_inputs = validate_inputs
            
            if element_type == "P1":
                # Linear elements
                self.num_dofs_per_element = mesh.num_nodes_per_element
                self.num_dofs = mesh.num_nodes
            else:
                raise ValidationError(f"Unsupported element type: {element_type}",
                                    field="element_type", value=element_type)
            
            # Log function space creation
            global_audit_logger.log_data_operation(
                "create", "SimpleFunctionSpace", 1,
                element_type=element_type, num_dofs=self.num_dofs,
                dimension=mesh.dimension
            )
            
            logger.debug(f"SimpleFunctionSpace created: {element_type} elements, "
                       f"{self.num_dofs} DOFs, {mesh.dimension}D")
    
    def get_dof_coordinates(self) -> np.ndarray:
        """Get coordinates of degrees of freedom."""
        return self.mesh.nodes
    
    def shape_functions_1d(self, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 1D linear shape functions and their derivatives with validation.
        
        Parameters
        ----------
        xi : np.ndarray
            Local coordinates in [-1, 1]
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Shape functions and their derivatives
            
        Raises
        ------
        ValidationError
            If local coordinates are invalid
        """
        with error_context("shape_functions_1d", num_points=len(xi)):
            # Input validation
            if self.validate_inputs:
                if not isinstance(xi, np.ndarray):
                    raise ValidationError("Local coordinates must be numpy array",
                                        field="xi", type=type(xi))
                
                if not np.isfinite(xi).all():
                    raise ValidationError("Local coordinates contain NaN/inf values",
                                        field="xi")
                
                # Check coordinate range (allow some tolerance)
                if np.any(xi < -1.1) or np.any(xi > 1.1):
                    logger.warning(f"Local coordinates outside [-1,1] range: min={xi.min():.3f}, max={xi.max():.3f}")
            
            if self.element_type != "P1":
                raise ValidationError("Only P1 elements supported", 
                                    field="element_type", value=self.element_type)
            
            try:
                # Linear shape functions
                N = np.zeros((len(xi), 2))
                N[:, 0] = 0.5 * (1 - xi)  # N1
                N[:, 1] = 0.5 * (1 + xi)  # N2
                
                # Derivatives in local coordinates
                dN_dxi = np.zeros((len(xi), 2))
                dN_dxi[:, 0] = -0.5  # dN1/dxi
                dN_dxi[:, 1] = 0.5   # dN2/dxi
                
                # Validate outputs
                if self.validate_inputs:
                    if not (np.isfinite(N).all() and np.isfinite(dN_dxi).all()):
                        raise ValidationError("Shape functions contain NaN/inf values",
                                            field="shape_functions")
                
                return N, dN_dxi
                
            except Exception as e:
                logger.error(f"Error computing 1D shape functions: {e}")
                raise
    
    def shape_functions_2d(self, xi: np.ndarray, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D triangular shape functions and their derivatives with validation.
        
        Parameters
        ----------
        xi : np.ndarray
            First local coordinate
        eta : np.ndarray  
            Second local coordinate
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Shape functions and their derivatives
            
        Raises
        ------
        ValidationError
            If local coordinates are invalid
        """
        with error_context("shape_functions_2d", num_points=len(xi)):
            # Input validation
            if self.validate_inputs:
                if not (isinstance(xi, np.ndarray) and isinstance(eta, np.ndarray)):
                    raise ValidationError("Local coordinates must be numpy arrays",
                                        field="coordinates")
                
                if len(xi) != len(eta):
                    raise ValidationError("Local coordinate arrays must have same length",
                                        field="coordinate_length", xi_len=len(xi), eta_len=len(eta))
                
                if not (np.isfinite(xi).all() and np.isfinite(eta).all()):
                    raise ValidationError("Local coordinates contain NaN/inf values",
                                        field="coordinates")
                
                # Check if coordinates are in reference triangle (xi >= 0, eta >= 0, xi + eta <= 1)
                if np.any(xi < -0.1) or np.any(eta < -0.1) or np.any(xi + eta > 1.1):
                    logger.warning("Local coordinates may be outside reference triangle")
            
            if self.element_type != "P1":
                raise ValidationError("Only P1 elements supported",
                                    field="element_type", value=self.element_type)
            
            try:
                # Linear triangle shape functions
                N = np.zeros((len(xi), 3))
                N[:, 0] = 1 - xi - eta  # N1
                N[:, 1] = xi            # N2
                N[:, 2] = eta           # N3
                
                # Derivatives in local coordinates
                dN_dxi = np.zeros((len(xi), 3, 2))  # [quad_point, shape_func, derivative]
                dN_dxi[:, 0, 0] = -1.0  # dN1/dxi
                dN_dxi[:, 0, 1] = -1.0  # dN1/deta
                dN_dxi[:, 1, 0] = 1.0   # dN2/dxi
                dN_dxi[:, 1, 1] = 0.0   # dN2/deta
                dN_dxi[:, 2, 0] = 0.0   # dN3/dxi
                dN_dxi[:, 2, 1] = 1.0   # dN3/deta
                
                # Validate outputs
                if self.validate_inputs:
                    if not (np.isfinite(N).all() and np.isfinite(dN_dxi).all()):
                        raise ValidationError("Shape functions contain NaN/inf values",
                                            field="shape_functions")
                    
                    # Check partition of unity
                    sum_N = np.sum(N, axis=1)
                    if not np.allclose(sum_N, 1.0, rtol=1e-12):
                        logger.warning("Shape functions do not satisfy partition of unity")
                
                return N, dN_dxi
                
            except Exception as e:
                logger.error(f"Error computing 2D shape functions: {e}")
                raise


@log_performance("gauss_quadrature_1d")
def gauss_quadrature_1d(n_points: int = 2, validate_inputs: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Get Gauss quadrature points and weights for 1D integration with validation.
    
    Parameters
    ----------
    n_points : int, optional
        Number of quadrature points, by default 2
    validate_inputs : bool, optional
        Enable input validation, by default True
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Quadrature points and weights
        
    Raises
    ------
    ValidationError
        If number of points is unsupported
    """
    with error_context("gauss_quadrature_1d", n_points=n_points):
        # Input validation
        if validate_inputs:
            validate_range(n_points, 1, 10, "n_points")
        
        # Security validation
        global_security_validator.validate_input(n_points, "n_points")
        
        try:
            if n_points == 1:
                points = np.array([0.0])
                weights = np.array([2.0])
            elif n_points == 2:
                points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
                weights = np.array([1.0, 1.0])
            elif n_points == 3:
                points = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
                weights = np.array([5/9, 8/9, 5/9])
            else:
                raise ValidationError(f"Unsupported number of quadrature points: {n_points}",
                                    field="n_points", value=n_points)
            
            # Validate quadrature rule
            if validate_inputs:
                if not (np.isfinite(points).all() and np.isfinite(weights).all()):
                    raise ValidationError("Quadrature points/weights contain NaN/inf values",
                                        field="quadrature")
                
                # Check that weights are positive
                if np.any(weights <= 0):
                    raise ValidationError("Quadrature weights must be positive",
                                        field="weights")
                
                # Check weight sum (should be 2 for interval [-1, 1])
                weight_sum = np.sum(weights)
                if not np.isclose(weight_sum, 2.0, rtol=1e-12):
                    logger.warning(f"Quadrature weights sum to {weight_sum}, expected 2.0")
            
            logger.debug(f"Generated 1D Gauss quadrature: {n_points} points")
            return points, weights
            
        except Exception as e:
            logger.error(f"Error generating 1D quadrature: {e}")
            raise


@log_performance("gauss_quadrature_triangle")
def gauss_quadrature_triangle(n_points: int = 3, validate_inputs: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Get quadrature points and weights for triangle integration with validation.
    
    Parameters
    ----------
    n_points : int, optional
        Number of quadrature points, by default 3
    validate_inputs : bool, optional
        Enable input validation, by default True
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Quadrature points (xi, eta) and weights
        
    Raises
    ------
    ValidationError
        If number of points is unsupported
    """
    with error_context("gauss_quadrature_triangle", n_points=n_points):
        # Input validation
        if validate_inputs:
            validate_range(n_points, 1, 10, "n_points")
        
        # Security validation
        global_security_validator.validate_input(n_points, "n_points")
        
        try:
            if n_points == 1:
                # Centroid
                points = np.array([[1/3, 1/3]])
                weights = np.array([0.5])
            elif n_points == 3:
                # 3-point rule
                points = np.array([
                    [1/6, 1/6],
                    [2/3, 1/6], 
                    [1/6, 2/3]
                ])
                weights = np.array([1/6, 1/6, 1/6])
            elif n_points == 4:
                # 4-point rule
                points = np.array([
                    [1/3, 1/3],
                    [0.6, 0.2],
                    [0.2, 0.6],
                    [0.2, 0.2]
                ])
                weights = np.array([-0.5625, 0.520833333, 0.520833333, 0.520833333]) / 2
            else:
                raise ValidationError(f"Unsupported number of quadrature points: {n_points}",
                                    field="n_points", value=n_points)
            
            # Validate quadrature rule
            if validate_inputs:
                if not (np.isfinite(points).all() and np.isfinite(weights).all()):
                    raise ValidationError("Quadrature points/weights contain NaN/inf values",
                                        field="quadrature")
                
                # Check that points are in reference triangle
                xi, eta = points[:, 0], points[:, 1]
                if np.any(xi < 0) or np.any(eta < 0) or np.any(xi + eta > 1):
                    logger.warning("Some quadrature points outside reference triangle")
                
                # Check weight sum (should be 0.5 for reference triangle area)
                weight_sum = np.sum(weights)
                if not np.isclose(weight_sum, 0.5, rtol=1e-12):
                    logger.warning(f"Triangle quadrature weights sum to {weight_sum}, expected 0.5")
            
            logger.debug(f"Generated triangle Gauss quadrature: {n_points} points")
            return points, weights
            
        except Exception as e:
            logger.error(f"Error generating triangle quadrature: {e}")
            raise