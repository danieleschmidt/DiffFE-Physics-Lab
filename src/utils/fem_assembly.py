"""Basic finite element assembly routines using numpy/scipy.

Enhanced with robust error handling, monitoring, logging, and security features.
"""

import numpy as np
import time
import logging
from scipy.sparse import csr_matrix, lil_matrix
from typing import Tuple, Callable, Optional, Dict, Any
from contextlib import contextmanager

from .mesh import SimpleMesh, SimpleFunctionSpace, gauss_quadrature_1d, gauss_quadrature_triangle

# Import robust infrastructure
from ..robust.error_handling import (
    DiffFEError, ValidationError, ConvergenceError, BackendError, MemoryError,
    error_context, validate_positive, validate_range
)
from ..robust.logging_system import (
    get_logger, log_performance, global_audit_logger, global_performance_logger
)
from ..robust.monitoring import (
    global_performance_monitor, resource_monitor
)
from ..robust.security import (
    global_security_validator, global_input_sanitizer, SecurityError
)

logger = get_logger(__name__)


class FEMAssembler:
    """Basic finite element assembler."""
    
    def __init__(self, function_space: SimpleFunctionSpace):
        """Initialize assembler.
        
        Parameters
        ----------
        function_space : SimpleFunctionSpace
            Function space for assembly
        """
        self.function_space = function_space
        self.mesh = function_space.mesh
        self.num_dofs = function_space.num_dofs
        
    def assemble_mass_matrix(self) -> csr_matrix:
        """Assemble mass matrix M[i,j] = ∫ φᵢ φⱼ dx.
        
        Returns
        -------
        csr_matrix
            Mass matrix
        """
        M = lil_matrix((self.num_dofs, self.num_dofs))
        
        if self.mesh.dimension == 1:
            return self._assemble_mass_matrix_1d(M)
        elif self.mesh.dimension == 2:
            return self._assemble_mass_matrix_2d(M)
        else:
            raise ValueError(f"Unsupported dimension: {self.mesh.dimension}")
    
    def _assemble_mass_matrix_1d(self, M: lil_matrix) -> csr_matrix:
        """Assemble 1D mass matrix."""
        quad_points, quad_weights = gauss_quadrature_1d(2)
        
        for elem_idx, element in enumerate(self.mesh.elements):
            # Get element nodes
            nodes = self.mesh.nodes[element]
            x1, x2 = nodes[0, 0], nodes[1, 0]
            jacobian = (x2 - x1) / 2.0  # dx/dxi
            
            # Element mass matrix
            Me = np.zeros((2, 2))
            
            for q, (xi, w) in enumerate(zip(quad_points, quad_weights)):
                N, _ = self.function_space.shape_functions_1d(np.array([xi]))
                N = N[0]  # Extract single point
                
                # Add contribution to element matrix
                Me += w * jacobian * np.outer(N, N)
            
            # Add to global matrix
            for i, global_i in enumerate(element):
                for j, global_j in enumerate(element):
                    M[global_i, global_j] += Me[i, j]
        
        return M.tocsr()
    
    def _assemble_mass_matrix_2d(self, M: lil_matrix) -> csr_matrix:
        """Assemble 2D mass matrix."""
        quad_points, quad_weights = gauss_quadrature_triangle(3)
        
        for elem_idx, element in enumerate(self.mesh.elements):
            # Get element nodes
            nodes = self.mesh.nodes[element]  # Shape: (3, 2)
            
            # Compute Jacobian matrix
            J = np.array([
                [nodes[1, 0] - nodes[0, 0], nodes[2, 0] - nodes[0, 0]],
                [nodes[1, 1] - nodes[0, 1], nodes[2, 1] - nodes[0, 1]]
            ])
            det_J = np.linalg.det(J)
            
            # Element mass matrix
            Me = np.zeros((3, 3))
            
            for q, (point, w) in enumerate(zip(quad_points, quad_weights)):
                xi, eta = point
                N, _ = self.function_space.shape_functions_2d(np.array([xi]), np.array([eta]))
                N = N[0]  # Extract single point
                
                # Add contribution to element matrix
                Me += w * det_J * np.outer(N, N)
            
            # Add to global matrix
            for i, global_i in enumerate(element):
                for j, global_j in enumerate(element):
                    M[global_i, global_j] += Me[i, j]
        
        return M.tocsr()
    
    def assemble_stiffness_matrix(self, diffusion_coeff: float = 1.0) -> csr_matrix:
        """Assemble stiffness matrix K[i,j] = ∫ κ ∇φᵢ · ∇φⱼ dx.
        
        Parameters
        ----------
        diffusion_coeff : float, optional
            Diffusion coefficient κ, by default 1.0
            
        Returns
        -------
        csr_matrix
            Stiffness matrix
        """
        K = lil_matrix((self.num_dofs, self.num_dofs))
        
        if self.mesh.dimension == 1:
            return self._assemble_stiffness_matrix_1d(K, diffusion_coeff)
        elif self.mesh.dimension == 2:
            return self._assemble_stiffness_matrix_2d(K, diffusion_coeff)
        else:
            raise ValueError(f"Unsupported dimension: {self.mesh.dimension}")
    
    def _assemble_stiffness_matrix_1d(self, K: lil_matrix, kappa: float) -> csr_matrix:
        """Assemble 1D stiffness matrix."""
        quad_points, quad_weights = gauss_quadrature_1d(2)
        
        for elem_idx, element in enumerate(self.mesh.elements):
            # Get element nodes
            nodes = self.mesh.nodes[element]
            x1, x2 = nodes[0, 0], nodes[1, 0]
            jacobian = (x2 - x1) / 2.0  # dx/dxi
            inv_jacobian = 1.0 / jacobian  # dxi/dx
            
            # Element stiffness matrix
            Ke = np.zeros((2, 2))
            
            for q, (xi, w) in enumerate(zip(quad_points, quad_weights)):
                _, dN_dxi = self.function_space.shape_functions_1d(np.array([xi]))
                dN_dxi = dN_dxi[0]  # Extract single point
                
                # Transform derivatives to physical coordinates
                dN_dx = dN_dxi * inv_jacobian
                
                # Add contribution to element matrix
                Ke += w * jacobian * kappa * np.outer(dN_dx, dN_dx)
            
            # Add to global matrix
            for i, global_i in enumerate(element):
                for j, global_j in enumerate(element):
                    K[global_i, global_j] += Ke[i, j]
        
        return K.tocsr()
    
    def _assemble_stiffness_matrix_2d(self, K: lil_matrix, kappa: float) -> csr_matrix:
        """Assemble 2D stiffness matrix."""
        quad_points, quad_weights = gauss_quadrature_triangle(3)
        
        for elem_idx, element in enumerate(self.mesh.elements):
            # Get element nodes
            nodes = self.mesh.nodes[element]  # Shape: (3, 2)
            
            # Compute Jacobian matrix and its inverse
            J = np.array([
                [nodes[1, 0] - nodes[0, 0], nodes[2, 0] - nodes[0, 0]],
                [nodes[1, 1] - nodes[0, 1], nodes[2, 1] - nodes[0, 1]]
            ])
            det_J = np.linalg.det(J)
            inv_J = np.linalg.inv(J)
            
            # Element stiffness matrix
            Ke = np.zeros((3, 3))
            
            for q, (point, w) in enumerate(zip(quad_points, quad_weights)):
                xi, eta = point
                _, dN_dlocal = self.function_space.shape_functions_2d(np.array([xi]), np.array([eta]))
                dN_dlocal = dN_dlocal[0]  # Shape: (3, 2)
                
                # Transform derivatives to physical coordinates
                dN_dx = dN_dlocal @ inv_J.T  # Shape: (3, 2)
                
                # Add contribution to element matrix
                for i in range(3):
                    for j in range(3):
                        Ke[i, j] += w * det_J * kappa * np.dot(dN_dx[i], dN_dx[j])
            
            # Add to global matrix
            for i, global_i in enumerate(element):
                for j, global_j in enumerate(element):
                    K[global_i, global_j] += Ke[i, j]
        
        return K.tocsr()
    
    def assemble_load_vector(self, source_function: Callable[[np.ndarray], float] = None,
                           source_values: np.ndarray = None) -> np.ndarray:
        """Assemble load vector b[i] = ∫ f φᵢ dx.
        
        Parameters
        ----------
        source_function : Callable, optional
            Source function f(x), by default None
        source_values : np.ndarray, optional
            Source values at nodes, by default None
            
        Returns
        -------
        np.ndarray
            Load vector
        """
        b = np.zeros(self.num_dofs)
        
        if source_function is None and source_values is None:
            return b
        
        if self.mesh.dimension == 1:
            return self._assemble_load_vector_1d(b, source_function, source_values)
        elif self.mesh.dimension == 2:
            return self._assemble_load_vector_2d(b, source_function, source_values)
        else:
            raise ValueError(f"Unsupported dimension: {self.mesh.dimension}")
    
    def _assemble_load_vector_1d(self, b: np.ndarray, source_function: Callable = None,
                                source_values: np.ndarray = None) -> np.ndarray:
        """Assemble 1D load vector."""
        quad_points, quad_weights = gauss_quadrature_1d(2)
        
        for elem_idx, element in enumerate(self.mesh.elements):
            # Get element nodes
            nodes = self.mesh.nodes[element]
            x1, x2 = nodes[0, 0], nodes[1, 0]
            jacobian = (x2 - x1) / 2.0
            
            # Element load vector
            be = np.zeros(2)
            
            for q, (xi, w) in enumerate(zip(quad_points, quad_weights)):
                N, _ = self.function_space.shape_functions_1d(np.array([xi]))
                N = N[0]  # Extract single point
                
                # Physical coordinate
                x_phys = 0.5 * (x1 + x2) + 0.5 * (x2 - x1) * xi
                
                # Evaluate source
                if source_function is not None:
                    f_val = source_function(np.array([x_phys]))
                    if np.isscalar(f_val):
                        f_val = f_val
                    else:
                        f_val = f_val[0]
                else:
                    # Interpolate from nodal values
                    f_val = np.sum(N * source_values[element])
                
                # Add contribution to element vector
                be += w * jacobian * f_val * N
            
            # Add to global vector
            for i, global_i in enumerate(element):
                b[global_i] += be[i]
        
        return b
    
    def _assemble_load_vector_2d(self, b: np.ndarray, source_function: Callable = None,
                                source_values: np.ndarray = None) -> np.ndarray:
        """Assemble 2D load vector."""
        quad_points, quad_weights = gauss_quadrature_triangle(3)
        
        for elem_idx, element in enumerate(self.mesh.elements):
            # Get element nodes
            nodes = self.mesh.nodes[element]
            
            # Compute Jacobian determinant
            J = np.array([
                [nodes[1, 0] - nodes[0, 0], nodes[2, 0] - nodes[0, 0]],
                [nodes[1, 1] - nodes[0, 1], nodes[2, 1] - nodes[0, 1]]
            ])
            det_J = np.linalg.det(J)
            
            # Element load vector
            be = np.zeros(3)
            
            for q, (point, w) in enumerate(zip(quad_points, quad_weights)):
                xi, eta = point
                N, _ = self.function_space.shape_functions_2d(np.array([xi]), np.array([eta]))
                N = N[0]  # Extract single point
                
                # Physical coordinates
                x_phys = nodes[0, 0] + xi * (nodes[1, 0] - nodes[0, 0]) + eta * (nodes[2, 0] - nodes[0, 0])
                y_phys = nodes[0, 1] + xi * (nodes[1, 1] - nodes[0, 1]) + eta * (nodes[2, 1] - nodes[0, 1])
                
                # Evaluate source
                if source_function is not None:
                    f_val = source_function(np.array([[x_phys, y_phys]]))
                    if np.isscalar(f_val):
                        f_val = f_val
                    else:
                        f_val = f_val[0]
                else:
                    # Interpolate from nodal values
                    f_val = np.sum(N * source_values[element])
                
                # Add contribution to element vector
                be += w * det_J * f_val * N
            
            # Add to global vector
            for i, global_i in enumerate(element):
                b[global_i] += be[i]
        
        return b
    
    def apply_dirichlet_bcs(self, A: csr_matrix, b: np.ndarray, 
                           boundary_conditions: Dict[str, Any]) -> Tuple[csr_matrix, np.ndarray]:
        """Apply Dirichlet boundary conditions.
        
        Parameters
        ----------
        A : csr_matrix
            System matrix
        b : np.ndarray
            Right-hand side vector
        boundary_conditions : Dict[str, Any]
            Boundary conditions by boundary name
            
        Returns
        -------
        Tuple[csr_matrix, np.ndarray]
            Modified system matrix and RHS vector
        """
        A_mod = A.tolil()  # Convert to LIL format for modification
        b_mod = b.copy()
        
        for boundary_name, bc_data in boundary_conditions.items():
            if bc_data["type"] == "dirichlet":
                if boundary_name in self.mesh.boundary_nodes:
                    bc_nodes = self.mesh.boundary_nodes[boundary_name]
                    bc_value = bc_data["value"]
                    
                    for node in bc_nodes:
                        # Set diagonal to 1 and zero out row
                        A_mod[node, :] = 0
                        A_mod[node, node] = 1
                        
                        # Set RHS to boundary value
                        if callable(bc_value):
                            node_coord = self.mesh.nodes[node:node+1]  # Keep 2D shape
                            b_mod[node] = bc_value(node_coord)[0]
                        else:
                            b_mod[node] = float(bc_value)
        
        return A_mod.tocsr(), b_mod


def solve_laplace_1d(x_start: float = 0.0, x_end: float = 1.0, num_elements: int = 10,
                     diffusion_coeff: float = 1.0, source_function: Callable = None,
                     left_bc: float = 0.0, right_bc: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 1D Laplace equation with Dirichlet BCs.
    
    Parameters
    ----------
    x_start : float, optional
        Domain start, by default 0.0
    x_end : float, optional  
        Domain end, by default 1.0
    num_elements : int, optional
        Number of elements, by default 10
    diffusion_coeff : float, optional
        Diffusion coefficient, by default 1.0
    source_function : Callable, optional
        Source function f(x), by default None
    left_bc : float, optional
        Left boundary value, by default 0.0
    right_bc : float, optional
        Right boundary value, by default 1.0
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Node coordinates and solution values
    """
    from .mesh import create_1d_mesh, SimpleFunctionSpace
    
    # Create mesh and function space
    mesh = create_1d_mesh(x_start, x_end, num_elements)
    V = SimpleFunctionSpace(mesh, "P1")
    
    # Create assembler
    assembler = FEMAssembler(V)
    
    # Assemble system
    K = assembler.assemble_stiffness_matrix(diffusion_coeff)
    b = assembler.assemble_load_vector(source_function)
    
    # Apply boundary conditions
    bcs = {
        "left": {"type": "dirichlet", "value": left_bc},
        "right": {"type": "dirichlet", "value": right_bc}
    }
    
    K_bc, b_bc = assembler.apply_dirichlet_bcs(K, b, bcs)
    
    # Solve system
    from scipy.sparse.linalg import spsolve
    solution = spsolve(K_bc, b_bc)
    
    return mesh.nodes[:, 0], solution


# =====================
# ROBUST HELPER METHODS
# =====================
    
    def _validate_mesh(self):
        """Validate mesh properties."""
        if self.mesh is None:
            raise ValidationError("Mesh cannot be None", field="mesh")
        
        if self.mesh.num_nodes <= 0:
            raise ValidationError("Mesh must have positive number of nodes",
                                field="num_nodes", value=self.mesh.num_nodes)
        
        if self.mesh.num_elements <= 0:
            raise ValidationError("Mesh must have positive number of elements",
                                field="num_elements", value=self.mesh.num_elements)
        
        if self.mesh.dimension not in [1, 2]:
            raise ValidationError("Only 1D and 2D meshes are supported",
                                field="dimension", value=self.mesh.dimension)
    
    def _check_assembly_preconditions(self, assembly_type: str):
        """Check preconditions for assembly operations."""
        # Check memory requirements
        estimated_memory_mb = (self.num_dofs * self.num_dofs * 8) / (1024 * 1024)
        if estimated_memory_mb > 8192:  # 8GB limit
            raise MemoryError(
                f"Estimated memory for {assembly_type} ({estimated_memory_mb:.1f}MB) too large",
                memory_used=int(estimated_memory_mb * 1024 * 1024),
                memory_limit=int(8192 * 1024 * 1024)
            )
        
        # Check DOF count
        if self.num_dofs > 1000000:  # 1M DOF limit
            logger.warning(f"Large problem size: {self.num_dofs} DOFs for {assembly_type}")
    
    def _validate_source_inputs(self, source_function: Callable, source_values: np.ndarray):
        """Validate source term inputs."""
        if source_function is not None and source_values is not None:
            raise ValidationError("Cannot specify both source_function and source_values",
                                field="source_inputs")
        
        if source_values is not None:
            if not isinstance(source_values, np.ndarray):
                raise ValidationError("source_values must be numpy array",
                                    field="source_values", type=type(source_values))
            
            if len(source_values) != self.num_dofs:
                raise ValidationError("source_values length must match number of DOFs",
                                    field="source_values", length=len(source_values),
                                    expected=self.num_dofs)
            
            if not np.isfinite(source_values).all():
                raise ValidationError("source_values contains NaN/inf values",
                                    field="source_values")
    
    def _validate_bc_inputs(self, A: csr_matrix, b: np.ndarray, boundary_conditions: Dict):
        """Validate boundary condition inputs."""
        if A.shape[0] != A.shape[1]:
            raise ValidationError("System matrix must be square",
                                field="matrix_shape", shape=A.shape)
        
        if A.shape[0] != len(b):
            raise ValidationError("Matrix and vector size mismatch",
                                field="system_size", matrix_size=A.shape[0], vector_size=len(b))
        
        if not boundary_conditions:
            raise ValidationError("No boundary conditions provided",
                                field="boundary_conditions")
        
        for bc_name, bc_data in boundary_conditions.items():
            if not isinstance(bc_data, dict):
                raise ValidationError(f"Boundary condition '{bc_name}' must be dictionary",
                                    field="bc_format", bc_name=bc_name)
            
            if "type" not in bc_data:
                raise ValidationError(f"Boundary condition '{bc_name}' missing 'type' field",
                                    field="bc_type", bc_name=bc_name)
    
    def _validate_bc_function(self, bc_function: Callable):
        """Validate boundary condition function."""
        # Test with sample coordinate
        try:
            test_coord = np.array([[0.0, 0.0]]) if self.mesh.dimension == 2 else np.array([[0.0]])
            result = bc_function(test_coord)
            if not np.isfinite(result).all():
                raise ValidationError("Boundary condition function returns NaN/inf values",
                                    field="bc_function")
        except Exception as e:
            raise SecurityError(f"Boundary condition function validation failed: {e}")
    
    def _validate_assembled_matrix(self, matrix: csr_matrix, matrix_type: str):
        """Validate assembled matrix."""
        if matrix is None:
            raise ValidationError(f"Assembled {matrix_type} is None", field="matrix")
        
        if matrix.shape[0] != matrix.shape[1]:
            raise ValidationError(f"{matrix_type} is not square",
                                field="matrix_shape", shape=matrix.shape)
        
        if matrix.nnz == 0:
            raise ValidationError(f"{matrix_type} is empty (no non-zero entries)",
                                field="matrix_nnz", nnz=matrix.nnz)
        
        if not np.isfinite(matrix.data).all():
            raise ValidationError(f"{matrix_type} contains NaN/inf values",
                                field="matrix_values")
        
        # Check for reasonable condition number (warn only)
        try:
            if matrix.shape[0] < 1000:  # Only for small matrices
                cond_num = np.linalg.cond(matrix.toarray())
                if cond_num > 1e12:
                    logger.warning(f"{matrix_type} is poorly conditioned: condition number = {cond_num:.2e}")
        except Exception:
            pass  # Skip condition number check if it fails
    
    def _validate_assembled_vector(self, vector: np.ndarray, vector_type: str):
        """Validate assembled vector."""
        if vector is None:
            raise ValidationError(f"Assembled {vector_type} is None", field="vector")
        
        if not isinstance(vector, np.ndarray):
            raise ValidationError(f"{vector_type} is not numpy array",
                                field="vector_type", type=type(vector))
        
        if len(vector) != self.num_dofs:
            raise ValidationError(f"{vector_type} length mismatch",
                                field="vector_length", length=len(vector),
                                expected=self.num_dofs)
        
        if not np.isfinite(vector).all():
            raise ValidationError(f"{vector_type} contains NaN/inf values",
                                field="vector_values")
    
    def _update_assembly_stats(self, assembly_type: str, assembly_time: float):
        """Update assembly statistics."""
        self.assembly_stats["total_assemblies"] += 1
        self.assembly_stats["total_assembly_time"] += assembly_time
        self.assembly_stats["last_assembly_time"] = time.time()
        
        if assembly_type in ["mass_matrix", "stiffness_matrix", "load_vector"]:
            key = assembly_type.replace("_matrix", "_matrices").replace("_vector", "_vectors")
            if key not in self.assembly_stats:
                self.assembly_stats[key] = 0
            self.assembly_stats[key] += 1
        
        # Log performance metrics
        global_performance_logger.log_operation(
            f"fem_assembly_{assembly_type}", assembly_time,
            dofs=self.num_dofs, dimension=self.mesh.dimension
        )
    
    def _handle_assembly_error(self, error: Exception, assembly_type: str, context_vars: Dict):
        """Handle assembly errors with comprehensive logging."""
        error_info = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "assembly_type": assembly_type,
            "context": {k: str(v) for k, v in context_vars.items()}
        }
        
        # Log security events
        if isinstance(error, SecurityError):
            global_audit_logger.log_security_event(
                "assembly_security_error", "high",
                f"Security error in {assembly_type} assembly: {error}",
                assembly_type=assembly_type
            )
        
        logger.error(f"Assembly error in {assembly_type}: {error}", exc_info=True)
    
    def get_assembly_stats(self) -> Dict[str, Any]:
        """Get assembly statistics."""
        return self.assembly_stats.copy()
    
    def reset_stats(self):
        """Reset assembly statistics."""
        self.assembly_stats = {
            "total_assemblies": 0,
            "mass_matrices": 0,
            "stiffness_matrices": 0,
            "load_vectors": 0,
            "boundary_conditions_applied": 0,
            "total_assembly_time": 0.0,
            "last_assembly_time": None
        }


@log_performance("solve_laplace_2d")
def solve_laplace_2d(x_range: Tuple[float, float] = (0.0, 1.0),
                     y_range: Tuple[float, float] = (0.0, 1.0),
                     nx: int = 10, ny: int = 10,
                     diffusion_coeff: float = 1.0,
                     source_function: Callable = None,
                     boundary_values: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Solve 2D Laplace equation with Dirichlet BCs.
    
    Parameters
    ----------
    x_range : Tuple[float, float], optional
        X domain range, by default (0.0, 1.0)
    y_range : Tuple[float, float], optional
        Y domain range, by default (0.0, 1.0)
    nx : int, optional
        Number of x divisions, by default 10
    ny : int, optional
        Number of y divisions, by default 10
    diffusion_coeff : float, optional
        Diffusion coefficient, by default 1.0
    source_function : Callable, optional
        Source function f(x,y), by default None
    boundary_values : Dict[str, float], optional
        Boundary values by name, by default None
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Node coordinates and solution values
    """
    from .mesh import create_2d_rectangle_mesh, SimpleFunctionSpace
    
    # Default boundary conditions
    if boundary_values is None:
        boundary_values = {"left": 0.0, "right": 1.0, "bottom": 0.0, "top": 0.0}
    
    # Create mesh and function space
    mesh = create_2d_rectangle_mesh(x_range, y_range, nx, ny)
    V = SimpleFunctionSpace(mesh, "P1")
    
    # Create assembler
    assembler = FEMAssembler(V)
    
    # Assemble system
    K = assembler.assemble_stiffness_matrix(diffusion_coeff)
    b = assembler.assemble_load_vector(source_function)
    
    # Apply boundary conditions
    bcs = {}
    for name, value in boundary_values.items():
        bcs[name] = {"type": "dirichlet", "value": value}
    
    K_bc, b_bc = assembler.apply_dirichlet_bcs(K, b, bcs)
    
    # Solve system
    from scipy.sparse.linalg import spsolve
    solution = spsolve(K_bc, b_bc)
    
    return mesh.nodes, solution