"""Basic FEM solver implementation without Firedrake dependency.

Enhanced with robust error handling, monitoring, logging, and security features.
"""

import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, bicgstab, gmres
from contextlib import contextmanager

from ..backends import get_backend
from ..models import Problem
from ..utils.mesh import SimpleMesh, SimpleFunctionSpace, create_1d_mesh, create_2d_rectangle_mesh
from ..utils.fem_assembly import FEMAssembler

# Import robust infrastructure
from ..robust.error_handling import (
    DiffFEError, ValidationError, ConvergenceError, BackendError, MemoryError,
    error_context, retry_with_backoff, validate_positive, validate_range
)
from ..robust.logging_system import (
    get_logger, log_performance, global_audit_logger, global_performance_logger
)
from ..robust.monitoring import (
    global_performance_monitor, global_health_checker, resource_monitor
)
from ..robust.security import (
    global_security_validator, global_input_sanitizer, SecurityError
)

logger = get_logger(__name__)


class BasicFEMSolver:
    """Basic finite element solver using numpy/scipy (no Firedrake).
    
    Enhanced with comprehensive error handling, validation, monitoring, and security features.
    """
    
    def __init__(self, backend: str = "numpy", solver_options: Dict[str, Any] = None,
                 enable_monitoring: bool = True, security_context: Optional[Any] = None):
        """Initialize basic FEM solver.
        
        Parameters
        ----------
        backend : str, optional
            Backend for computations, by default "numpy"
        solver_options : Dict[str, Any], optional
            Solver options, by default None
        enable_monitoring : bool, optional
            Enable performance monitoring, by default True
        security_context : Optional[Any], optional
            Security context for operations, by default None
        """
        with error_context("BasicFEMSolver_initialization"):
            # Validate and sanitize inputs
            backend = global_input_sanitizer.sanitize_string(backend)
            global_security_validator.validate_input(backend, "backend")
            
            if solver_options is not None:
                global_security_validator.validate_parameters(solver_options)
            
            self.backend_name = backend
            self.backend = get_backend(backend)
            self.solver_options = solver_options or {}
            self.enable_monitoring = enable_monitoring
            self.security_context = security_context
            
            # Solver state with enhanced tracking
            self.solution_history = []
            self.convergence_history = []
            self.error_history = []
            self.performance_metrics = []
            self.health_status = {"status": "initialized", "last_check": time.time()}
            
            # Robust default solver options
            self.default_options = {
                "max_iterations": 1000,
                "tolerance": 1e-8,
                "linear_solver": "direct",  # direct, bicgstab, gmres
                "monitor_convergence": True,
                "memory_limit_mb": 4096,  # 4GB memory limit
                "timeout_seconds": 1800,  # 30 minute timeout
                "validate_inputs": True,
                "security_checks": True,
                "enable_fallback": True,
            }
            
            # Validate and merge options
            validated_options = self._validate_solver_options(self.solver_options)
            self.options = {**self.default_options, **validated_options}
            
            # Register health checks
            global_health_checker.register_check(
                f"BasicFEMSolver_{id(self)}_health",
                self._health_check,
                "BasicFEMSolver health status"
            )
            
            # Log initialization
            global_audit_logger.log_data_operation(
                "create", "BasicFEMSolver", 1,
                backend=backend, enable_monitoring=enable_monitoring
            )
            
            logger.info(f"BasicFEMSolver initialized with backend: {backend}, "
                       f"monitoring: {enable_monitoring}, options: {len(self.options)} settings")
    
    @log_performance("solve_1d_laplace")
    @retry_with_backoff(max_retries=3, expected_exceptions=(ConvergenceError, MemoryError))
    def solve_1d_laplace(self, x_start: float = 0.0, x_end: float = 1.0, 
                         num_elements: int = 10, diffusion_coeff: float = 1.0,
                         source_function: Callable = None, left_bc: float = 0.0, 
                         right_bc: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Solve 1D Laplace equation with robust error handling and monitoring.
        
        Solves -κ d²u/dx² = f with Dirichlet boundary conditions.
        
        Parameters
        ----------
        x_start : float, optional
            Domain start, by default 0.0
        x_end : float, optional  
            Domain end, by default 1.0
        num_elements : int, optional
            Number of elements, by default 10
        diffusion_coeff : float, optional
            Diffusion coefficient κ, by default 1.0
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
            
        Raises
        ------
        ValidationError
            If input parameters are invalid
        ConvergenceError
            If solver fails to converge
        MemoryError
            If memory limits are exceeded
        SecurityError
            If security validation fails
        """
        with error_context("solve_1d_laplace", dimension=1, num_elements=num_elements):
            # Comprehensive input validation
            if self.options["validate_inputs"]:
                self._validate_1d_inputs(x_start, x_end, num_elements, diffusion_coeff, left_bc, right_bc)
            
            # Security validation
            if self.options["security_checks"]:
                self._security_validate_1d_inputs(x_start, x_end, num_elements, diffusion_coeff, 
                                                 source_function, left_bc, right_bc)
            
            with resource_monitor("solve_1d_laplace", num_elements=num_elements, backend=self.backend_name) as monitor:
                logger.info(f"Solving 1D Laplace equation on [{x_start}, {x_end}] with {num_elements} elements")
                
                try:
                    # Create mesh and function space with monitoring
                    mesh = create_1d_mesh(x_start, x_end, num_elements)
                    V = SimpleFunctionSpace(mesh, "P1")
                    assembler = FEMAssembler(V)
                    
                    # Memory check
                    self._check_memory_usage("after_mesh_creation")
                    
                    # Assemble system with monitoring
                    K = assembler.assemble_stiffness_matrix(diffusion_coeff)
                    b = assembler.assemble_load_vector(source_function)
                    
                    # Validate assembled system
                    self._validate_linear_system(K, b, "1D_Laplace")
                    
                    # Apply boundary conditions
                    bcs = {
                        "left": {"type": "dirichlet", "value": left_bc},
                        "right": {"type": "dirichlet", "value": right_bc}
                    }
                    
                    K_bc, b_bc = assembler.apply_dirichlet_bcs(K, b, bcs)
                    
                    # Final system validation
                    self._validate_linear_system(K_bc, b_bc, "1D_Laplace_with_BCs")
                    
                    # Solve system with monitoring
                    solution = self._solve_linear_system_robust(K_bc, b_bc)
                    
                    # Validate solution
                    self._validate_solution(solution, "1D_Laplace")
                    
                    # Store solution with metadata
                    solution_record = {
                        "solution": solution.copy(),
                        "timestamp": time.time(),
                        "problem_type": "1D_Laplace",
                        "num_elements": num_elements,
                        "diffusion_coeff": diffusion_coeff,
                        "domain": [x_start, x_end],
                        "boundary_conditions": bcs
                    }
                    self.solution_history.append(solution_record)
                    
                    # Log successful completion
                    global_audit_logger.log_data_operation(
                        "solve", "1D_Laplace", len(solution),
                        num_elements=num_elements, backend=self.backend_name
                    )
                    
                    logger.info(f"1D Laplace equation solved successfully: {len(solution)} DOFs")
                    return mesh.nodes[:, 0], solution
                    
                except Exception as e:
                    self._handle_solve_error(e, "1D_Laplace", locals())
                    raise
    
    @log_performance("solve_2d_laplace")
    @retry_with_backoff(max_retries=3, expected_exceptions=(ConvergenceError, MemoryError))
    def solve_2d_laplace(self, x_range: Tuple[float, float] = (0.0, 1.0),
                         y_range: Tuple[float, float] = (0.0, 1.0),
                         nx: int = 10, ny: int = 10, diffusion_coeff: float = 1.0,
                         source_function: Callable = None,
                         boundary_values: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Solve 2D Laplace equation with robust error handling and monitoring.
        
        Solves -κ ∇²u = f with Dirichlet boundary conditions.
        
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
            Diffusion coefficient κ, by default 1.0
        source_function : Callable, optional
            Source function f(x,y), by default None
        boundary_values : Dict[str, float], optional
            Boundary values by name, by default None
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Node coordinates and solution values
            
        Raises
        ------
        ValidationError
            If input parameters are invalid
        ConvergenceError
            If solver fails to converge
        MemoryError
            If memory limits are exceeded
        SecurityError
            If security validation fails
        """
        with error_context("solve_2d_laplace", dimension=2, nx=nx, ny=ny):
            # Comprehensive input validation
            if self.options["validate_inputs"]:
                self._validate_2d_inputs(x_range, y_range, nx, ny, diffusion_coeff, boundary_values)
            
            # Security validation
            if self.options["security_checks"]:
                self._security_validate_2d_inputs(x_range, y_range, nx, ny, diffusion_coeff,
                                                 source_function, boundary_values)
            
            with resource_monitor("solve_2d_laplace", nx=nx, ny=ny, backend=self.backend_name) as monitor:
                logger.info(f"Solving 2D Laplace equation on {x_range}×{y_range} with {nx}×{ny} elements")
                
                try:
                    # Default boundary conditions
                    if boundary_values is None:
                        boundary_values = {"left": 0.0, "right": 1.0, "bottom": 0.0, "top": 0.0}
                    
                    # Estimate memory requirements
                    estimated_dofs = (nx + 1) * (ny + 1)
                    self._check_memory_requirements(estimated_dofs, "2D_Laplace")
                    
                    # Create mesh and function space
                    mesh = create_2d_rectangle_mesh(x_range, y_range, nx, ny)
                    V = SimpleFunctionSpace(mesh, "P1")
                    assembler = FEMAssembler(V)
                    
                    # Memory check
                    self._check_memory_usage("after_mesh_creation")
                    
                    # Assemble system with monitoring
                    K = assembler.assemble_stiffness_matrix(diffusion_coeff)
                    b = assembler.assemble_load_vector(source_function)
                    
                    # Validate assembled system
                    self._validate_linear_system(K, b, "2D_Laplace")
                    
                    # Apply boundary conditions
                    bcs = {}
                    for name, value in boundary_values.items():
                        bcs[name] = {"type": "dirichlet", "value": value}
                    
                    K_bc, b_bc = assembler.apply_dirichlet_bcs(K, b, bcs)
                    
                    # Final system validation
                    self._validate_linear_system(K_bc, b_bc, "2D_Laplace_with_BCs")
                    
                    # Solve system with monitoring
                    solution = self._solve_linear_system_robust(K_bc, b_bc)
                    
                    # Validate solution
                    self._validate_solution(solution, "2D_Laplace")
                    
                    # Store solution with metadata
                    solution_record = {
                        "solution": solution.copy(),
                        "timestamp": time.time(),
                        "problem_type": "2D_Laplace",
                        "mesh_size": [nx, ny],
                        "diffusion_coeff": diffusion_coeff,
                        "domain": {"x_range": x_range, "y_range": y_range},
                        "boundary_conditions": bcs
                    }
                    self.solution_history.append(solution_record)
                    
                    # Log successful completion
                    global_audit_logger.log_data_operation(
                        "solve", "2D_Laplace", len(solution),
                        nx=nx, ny=ny, backend=self.backend_name
                    )
                    
                    logger.info(f"2D Laplace equation solved successfully: {len(solution)} DOFs")
                    return mesh.nodes, solution
                    
                except Exception as e:
                    self._handle_solve_error(e, "2D_Laplace", locals())
                    raise
    
    def solve_problem(self, problem: Problem, parameters: Dict[str, Any] = None) -> np.ndarray:
        """Solve a problem using basic FEM.
        
        Parameters
        ----------
        problem : Problem
            Problem to solve
        parameters : Dict[str, Any], optional
            Runtime parameters, by default None
            
        Returns
        -------
        np.ndarray
            Solution values
        """
        if parameters is None:
            parameters = {}
        
        # Merge problem parameters with runtime parameters
        solve_params = {**problem.parameters, **parameters}
        
        # Determine problem dimension and type
        mesh_params = solve_params.get('mesh', {})
        dimension = mesh_params.get('dimension', 1)
        
        if dimension == 1:
            return self._solve_problem_1d(problem, solve_params)
        elif dimension == 2:
            return self._solve_problem_2d(problem, solve_params)
        else:
            raise ValueError(f"Unsupported dimension: {dimension}")
    
    def _solve_problem_1d(self, problem: Problem, params: Dict[str, Any]) -> np.ndarray:
        """Solve 1D problem with basic FEM."""
        # Get mesh parameters
        mesh_params = params.get('mesh', {})
        x_start = mesh_params.get('x_start', 0.0)
        x_end = mesh_params.get('x_end', 1.0)
        num_elements = mesh_params.get('num_elements', 10)
        
        # Create mesh and function space
        mesh = create_1d_mesh(x_start, x_end, num_elements)
        V = SimpleFunctionSpace(mesh, "P1")
        assembler = FEMAssembler(V)
        
        # Get equation parameters
        diffusion_coeff = params.get('diffusion_coeff', 1.0)
        source_function = params.get('source_function', None)
        
        # Assemble system
        K = assembler.assemble_stiffness_matrix(diffusion_coeff)
        b = assembler.assemble_load_vector(source_function)
        
        # Apply boundary conditions
        boundary_conditions = problem.boundary_conditions
        K_bc, b_bc = assembler.apply_dirichlet_bcs(K, b, boundary_conditions)
        
        # Solve linear system
        solution = self._solve_linear_system(K_bc, b_bc)
        
        return solution
    
    def _solve_problem_2d(self, problem: Problem, params: Dict[str, Any]) -> np.ndarray:
        """Solve 2D problem with basic FEM."""
        # Get mesh parameters
        mesh_params = params.get('mesh', {})
        x_range = mesh_params.get('x_range', (0.0, 1.0))
        y_range = mesh_params.get('y_range', (0.0, 1.0))
        nx = mesh_params.get('nx', 10)
        ny = mesh_params.get('ny', 10)
        
        # Create mesh and function space
        mesh = create_2d_rectangle_mesh(x_range, y_range, nx, ny)
        V = SimpleFunctionSpace(mesh, "P1")
        assembler = FEMAssembler(V)
        
        # Get equation parameters
        diffusion_coeff = params.get('diffusion_coeff', 1.0)
        source_function = params.get('source_function', None)
        
        # Assemble system
        K = assembler.assemble_stiffness_matrix(diffusion_coeff)
        b = assembler.assemble_load_vector(source_function)
        
        # Apply boundary conditions
        boundary_conditions = problem.boundary_conditions
        K_bc, b_bc = assembler.apply_dirichlet_bcs(K, b, boundary_conditions)
        
        # Solve linear system
        solution = self._solve_linear_system(K_bc, b_bc)
        
        return solution
    
    def _solve_linear_system_robust(self, A: csr_matrix, b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b with robust error handling and monitoring.
        
        Parameters
        ----------
        A : csr_matrix
            System matrix
        b : np.ndarray
            Right-hand side vector
            
        Returns
        -------
        np.ndarray
            Solution vector
            
        Raises
        ------
        ConvergenceError
            If iterative solver fails to converge
        BackendError
            If solver backend fails
        MemoryError
            If insufficient memory for solution
        """
        with error_context("linear_system_solve", matrix_size=A.shape, nnz=A.nnz):
            solver_type = self.options.get("linear_solver", "direct")
            tolerance = self.options.get("tolerance", 1e-8)
            max_iterations = self.options.get("max_iterations", 1000)
            
            # Memory check before solving
            self._check_memory_usage("before_linear_solve")
            
            try:
                if solver_type == "direct":
                    logger.debug("Using direct solver (LU factorization)")
                    with resource_monitor("direct_solve", matrix_size=A.shape[0]):
                        solution = spsolve(A, b)
                        
                elif solver_type == "bicgstab":
                    logger.debug("Using BiCGSTAB iterative solver")
                    with resource_monitor("bicgstab_solve", matrix_size=A.shape[0]):
                        solution, info = bicgstab(A, b, tol=tolerance, maxiter=max_iterations)
                        if info != 0:
                            if info > 0:
                                raise ConvergenceError(
                                    f"BiCGSTAB failed to converge in {info} iterations",
                                    iterations=info, tolerance=tolerance
                                )
                            else:
                                raise BackendError(
                                    f"BiCGSTAB solver error: info={info}",
                                    backend="scipy", operation="bicgstab"
                                )
                
                elif solver_type == "gmres":
                    logger.debug("Using GMRES iterative solver")
                    with resource_monitor("gmres_solve", matrix_size=A.shape[0]):
                        solution, info = gmres(A, b, tol=tolerance, maxiter=max_iterations)
                        if info != 0:
                            if info > 0:
                                raise ConvergenceError(
                                    f"GMRES failed to converge in {info} iterations",
                                    iterations=info, tolerance=tolerance
                                )
                            else:
                                raise BackendError(
                                    f"GMRES solver error: info={info}",
                                    backend="scipy", operation="gmres"
                                )
                
                else:
                    raise ValidationError(f"Unknown solver type: {solver_type}", field="linear_solver")
                
                # Validate solution
                if solution is None:
                    raise BackendError("Solver returned None", backend="scipy", operation=solver_type)
                    
                return solution
                
            except np.linalg.LinAlgError as e:
                raise BackendError(
                    f"Linear algebra error in {solver_type} solver: {e}",
                    backend="scipy", operation=solver_type
                ) from e
            except MemoryError as e:
                self._handle_memory_error(e, "linear_solve")
                raise
            except Exception as e:
                raise BackendError(
                    f"Unexpected error in {solver_type} solver: {e}",
                    backend="scipy", operation=solver_type
                ) from e
    
    def _solve_linear_system(self, A: csr_matrix, b: np.ndarray) -> np.ndarray:
        """Legacy method - use _solve_linear_system_robust instead."""
        logger.warning("Using legacy _solve_linear_system method. Consider upgrading to _solve_linear_system_robust.")
        return self._solve_linear_system_robust(A, b)
    
    def compute_error_l2(self, computed_solution: np.ndarray, 
                        exact_solution: Callable, mesh: SimpleMesh) -> float:
        """Compute L2 error between computed and exact solutions.
        
        Parameters
        ----------
        computed_solution : np.ndarray
            Computed solution values at nodes
        exact_solution : Callable
            Exact solution function
        mesh : SimpleMesh
            Mesh for integration
            
        Returns
        -------
        float
            L2 error
        """
        # Evaluate exact solution at nodes
        exact_values = np.array([exact_solution(node.reshape(1, -1))[0] 
                                for node in mesh.nodes])
        
        # Compute error at nodes
        error_values = computed_solution - exact_values
        
        # Simple L2 norm (could be improved with proper integration)
        l2_error = np.sqrt(np.mean(error_values**2))
        
        return l2_error
    
    def manufactured_solution_1d(self, frequency: float = 1.0, 
                                diffusion_coeff: float = 1.0) -> Dict[str, Callable]:
        """Generate manufactured solution for 1D Laplace equation.
        
        Parameters
        ----------
        frequency : float, optional
            Frequency parameter, by default 1.0
        diffusion_coeff : float, optional
            Diffusion coefficient, by default 1.0
            
        Returns
        -------
        Dict[str, Callable]
            Dictionary with 'solution' and 'source' functions
        """
        def solution(x):
            if isinstance(x, np.ndarray) and x.ndim > 1:
                return np.sin(frequency * np.pi * x[:, 0])
            else:
                return np.sin(frequency * np.pi * x)
        
        def source(x):
            if isinstance(x, np.ndarray) and x.ndim > 1:
                return diffusion_coeff * (frequency * np.pi)**2 * np.sin(frequency * np.pi * x[:, 0])
            else:
                return diffusion_coeff * (frequency * np.pi)**2 * np.sin(frequency * np.pi * x)
        
        return {"solution": solution, "source": source}
    
    def manufactured_solution_2d(self, frequency: float = 1.0,
                                diffusion_coeff: float = 1.0) -> Dict[str, Callable]:
        """Generate manufactured solution for 2D Laplace equation.
        
        Parameters
        ----------
        frequency : float, optional
            Frequency parameter, by default 1.0
        diffusion_coeff : float, optional
            Diffusion coefficient, by default 1.0
            
        Returns
        -------
        Dict[str, Callable]
            Dictionary with 'solution' and 'source' functions
        """
        def solution(x):
            if isinstance(x, np.ndarray) and x.ndim > 1:
                return (np.sin(frequency * np.pi * x[:, 0]) * 
                       np.sin(frequency * np.pi * x[:, 1]))
            else:
                return (np.sin(frequency * np.pi * x[0]) * 
                       np.sin(frequency * np.pi * x[1]))
        
        def source(x):
            factor = 2 * diffusion_coeff * (frequency * np.pi)**2
            if isinstance(x, np.ndarray) and x.ndim > 1:
                return (factor * np.sin(frequency * np.pi * x[:, 0]) * 
                       np.sin(frequency * np.pi * x[:, 1]))
            else:
                return (factor * np.sin(frequency * np.pi * x[0]) * 
                       np.sin(frequency * np.pi * x[1]))
        
        return {"solution": solution, "source": source}
    
    def convergence_study(self, problem_type: str = "1d", 
                         exact_solution: Callable = None,
                         refinement_levels: int = 4) -> Dict[str, List]:
        """Perform mesh convergence study.
        
        Parameters
        ----------
        problem_type : str, optional
            Problem type ("1d" or "2d"), by default "1d"
        exact_solution : Callable, optional
            Exact solution for error computation, by default None
        refinement_levels : int, optional
            Number of refinement levels, by default 4
            
        Returns
        -------
        Dict[str, List]
            Convergence study results
        """
        results = {"mesh_sizes": [], "dofs": [], "errors": [], "convergence_rates": []}
        
        if exact_solution is None:
            if problem_type == "1d":
                manufactured = self.manufactured_solution_1d()
            else:
                manufactured = self.manufactured_solution_2d()
            exact_solution = manufactured["solution"]
        
        for level in range(refinement_levels):
            if problem_type == "1d":
                num_elements = 10 * (2**level)
                nodes, solution = self.solve_1d_laplace(
                    num_elements=num_elements,
                    source_function=manufactured["source"] if 'manufactured' in locals() else None
                )
                mesh = create_1d_mesh(0.0, 1.0, num_elements)
                h = 1.0 / num_elements
            else:
                nx = ny = 10 * (2**level)
                nodes, solution = self.solve_2d_laplace(
                    nx=nx, ny=ny,
                    source_function=manufactured["source"] if 'manufactured' in locals() else None
                )
                mesh = create_2d_rectangle_mesh((0.0, 1.0), (0.0, 1.0), nx, ny)
                h = 1.0 / nx
            
            # Compute error
            error = self.compute_error_l2(solution, exact_solution, mesh)
            
            results["mesh_sizes"].append(h)
            results["dofs"].append(len(solution))
            results["errors"].append(error)
            
            logger.info(f"Level {level}: h={h:.3e}, DOFs={len(solution)}, error={error:.3e}")
        
        # Compute convergence rates
        for i in range(1, len(results["errors"])):
            if results["errors"][i] > 0 and results["errors"][i-1] > 0:
                rate = np.log(results["errors"][i] / results["errors"][i-1]) / np.log(
                    results["mesh_sizes"][i] / results["mesh_sizes"][i-1]
                )
                results["convergence_rates"].append(rate)
        
        return results
    
    # =====================
    # ROBUST HELPER METHODS
    # =====================
    
    def _validate_solver_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize solver options.
        
        Parameters
        ----------
        options : Dict[str, Any]
            Solver options to validate
            
        Returns
        -------
        Dict[str, Any]
            Validated and sanitized options
        """
        validated = {}
        
        for key, value in options.items():
            if key == "max_iterations":
                validated[key] = max(1, min(int(value), 10000))
            elif key == "tolerance":
                validated[key] = max(1e-16, min(float(value), 1e-1))
            elif key == "linear_solver":
                if value not in ["direct", "bicgstab", "gmres"]:
                    logger.warning(f"Unknown linear solver '{value}', using 'direct'")
                    validated[key] = "direct"
                else:
                    validated[key] = value
            elif key == "memory_limit_mb":
                validated[key] = max(100, min(int(value), 100000))  # 100MB to 100GB
            elif key == "timeout_seconds":
                validated[key] = max(1, min(int(value), 86400))  # 1 sec to 1 day
            else:
                validated[key] = value
                
        return validated
    
    def _validate_1d_inputs(self, x_start: float, x_end: float, num_elements: int,
                           diffusion_coeff: float, left_bc: float, right_bc: float):
        """Validate 1D problem inputs."""
        validate_range(x_start, -1e6, 1e6, "x_start")
        validate_range(x_end, -1e6, 1e6, "x_end")
        if x_end <= x_start:
            raise ValidationError("x_end must be greater than x_start", 
                                field="domain", x_start=x_start, x_end=x_end)
        
        validate_range(num_elements, 1, 100000, "num_elements")
        validate_positive(diffusion_coeff, "diffusion_coeff")
        validate_range(left_bc, -1e6, 1e6, "left_bc")
        validate_range(right_bc, -1e6, 1e6, "right_bc")
    
    def _validate_2d_inputs(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
                           nx: int, ny: int, diffusion_coeff: float, boundary_values: Dict):
        """Validate 2D problem inputs."""
        x_start, x_end = x_range
        y_start, y_end = y_range
        
        validate_range(x_start, -1e6, 1e6, "x_start")
        validate_range(x_end, -1e6, 1e6, "x_end")
        if x_end <= x_start:
            raise ValidationError("x_end must be greater than x_start",
                                field="x_range", x_start=x_start, x_end=x_end)
        
        validate_range(y_start, -1e6, 1e6, "y_start")
        validate_range(y_end, -1e6, 1e6, "y_end")
        if y_end <= y_start:
            raise ValidationError("y_end must be greater than y_start",
                                field="y_range", y_start=y_start, y_end=y_end)
        
        validate_range(nx, 1, 10000, "nx")
        validate_range(ny, 1, 10000, "ny")
        validate_positive(diffusion_coeff, "diffusion_coeff")
        
        if boundary_values:
            for name, value in boundary_values.items():
                validate_range(value, -1e6, 1e6, f"boundary_value_{name}")
    
    def _security_validate_1d_inputs(self, x_start: float, x_end: float, num_elements: int,
                                   diffusion_coeff: float, source_function: Callable,
                                   left_bc: float, right_bc: float):
        """Security validation for 1D inputs."""
        global_security_validator.validate_input([x_start, x_end, num_elements, 
                                                diffusion_coeff, left_bc, right_bc], 
                                                "1d_parameters")
        
        if source_function is not None and not callable(source_function):
            raise SecurityError("Source function must be callable or None")
    
    def _security_validate_2d_inputs(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
                                   nx: int, ny: int, diffusion_coeff: float,
                                   source_function: Callable, boundary_values: Dict):
        """Security validation for 2D inputs."""
        global_security_validator.validate_input(list(x_range) + list(y_range) + 
                                                [nx, ny, diffusion_coeff], 
                                                "2d_parameters")
        
        if source_function is not None and not callable(source_function):
            raise SecurityError("Source function must be callable or None")
            
        if boundary_values is not None:
            global_security_validator.validate_parameters(boundary_values)
    
    def _validate_linear_system(self, A: csr_matrix, b: np.ndarray, context: str):
        """Validate assembled linear system."""
        if A.shape[0] != A.shape[1]:
            raise ValidationError(f"System matrix not square in {context}",
                                field="matrix_shape", shape=A.shape)
        
        if A.shape[0] != len(b):
            raise ValidationError(f"Matrix and vector size mismatch in {context}",
                                field="system_size", matrix_size=A.shape[0], vector_size=len(b))
        
        if A.nnz == 0:
            raise ValidationError(f"System matrix is empty in {context}",
                                field="matrix_nnz", nnz=A.nnz)
        
        # Check for NaN/inf values
        if not np.isfinite(A.data).all():
            raise ValidationError(f"System matrix contains NaN/inf values in {context}",
                                field="matrix_values")
        
        if not np.isfinite(b).all():
            raise ValidationError(f"RHS vector contains NaN/inf values in {context}",
                                field="rhs_values")
    
    def _validate_solution(self, solution: np.ndarray, context: str):
        """Validate computed solution."""
        if solution is None:
            raise ValidationError(f"Solution is None in {context}", field="solution")
        
        if not isinstance(solution, np.ndarray):
            raise ValidationError(f"Solution is not numpy array in {context}",
                                field="solution_type", type=type(solution))
        
        if not np.isfinite(solution).all():
            raise ValidationError(f"Solution contains NaN/inf values in {context}",
                                field="solution_values")
        
        # Check solution magnitude
        max_val = np.max(np.abs(solution))
        if max_val > 1e10:
            logger.warning(f"Large solution values detected in {context}: max={max_val}")
    
    def _check_memory_requirements(self, estimated_dofs: int, context: str):
        """Check if estimated memory requirements are feasible."""
        # Rough estimate: 8 bytes per DOF for solution + matrix storage
        estimated_memory_mb = (estimated_dofs * estimated_dofs * 8) / (1024 * 1024)
        
        memory_limit = self.options.get("memory_limit_mb", 4096)
        if estimated_memory_mb > memory_limit:
            raise MemoryError(
                f"Estimated memory requirement ({estimated_memory_mb:.1f}MB) exceeds limit ({memory_limit}MB) in {context}",
                memory_used=int(estimated_memory_mb * 1024 * 1024),
                memory_limit=int(memory_limit * 1024 * 1024)
            )
    
    def _check_memory_usage(self, context: str):
        """Check current memory usage."""
        try:
            import psutil
            current_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_limit = self.options.get("memory_limit_mb", 4096)
            
            if current_memory_mb > memory_limit:
                raise MemoryError(
                    f"Memory usage ({current_memory_mb:.1f}MB) exceeds limit ({memory_limit}MB) at {context}",
                    memory_used=int(current_memory_mb * 1024 * 1024),
                    memory_limit=int(memory_limit * 1024 * 1024)
                )
        except ImportError:
            logger.debug("psutil not available, skipping memory check")
    
    def _handle_memory_error(self, error: Exception, context: str):
        """Handle memory-related errors with recovery attempts."""
        logger.error(f"Memory error in {context}: {error}")
        
        # Attempt garbage collection
        import gc
        gc.collect()
        
        # Log memory error for monitoring
        global_audit_logger.log_security_event(
            "memory_exhaustion", "high",
            f"Memory error in BasicFEMSolver at {context}",
            context=context, error=str(error)
        )
    
    def _handle_solve_error(self, error: Exception, problem_type: str, context_vars: Dict):
        """Handle solve errors with comprehensive logging."""
        error_record = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "problem_type": problem_type,
            "traceback": traceback.format_exc(),
            "context": {k: str(v) for k, v in context_vars.items()}
        }
        
        self.error_history.append(error_record)
        
        # Log security event if it's a security error
        if isinstance(error, SecurityError):
            global_audit_logger.log_security_event(
                "solver_security_error", "high",
                f"Security error in {problem_type}: {error}",
                problem_type=problem_type
            )
        
        # Update health status
        self.health_status = {
            "status": "error",
            "last_error": str(error),
            "error_time": time.time()
        }
        
        logger.error(f"Solve error in {problem_type}: {error}", exc_info=True)
    
    def _health_check(self) -> bool:
        """Health check for the solver instance."""
        try:
            # Check recent error rate
            recent_errors = [e for e in self.error_history 
                           if time.time() - e["timestamp"] < 3600]  # Last hour
            
            if len(recent_errors) > 10:
                return False
            
            # Check if backend is still available
            if self.backend is None:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "solver_status": self.health_status,
            "backend": self.backend_name,
            "backend_available": self.backend is not None,
            "total_solutions": len(self.solution_history),
            "total_errors": len(self.error_history),
            "recent_error_rate": len([e for e in self.error_history 
                                     if time.time() - e["timestamp"] < 3600]),
            "options": self.options,
            "monitoring_enabled": self.enable_monitoring
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for recent operations."""
        return {
            "solution_count": len(self.solution_history),
            "error_count": len(self.error_history),
            "convergence_history_size": len(self.convergence_history),
            "backend": self.backend_name,
            "monitoring_enabled": self.enable_monitoring,
            "last_solve_time": (self.solution_history[-1]["timestamp"] 
                               if self.solution_history else None)
        }