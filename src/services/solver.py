"""Core FEBML solver implementation."""

from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
import numpy as np
import logging

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from ..backends import get_backend
from ..models import Problem
from ..operators.base import get_operator

logger = logging.getLogger(__name__)


class FEBMLSolver:
    """Main solver for differentiable finite element problems.
    
    Provides unified interface for solving FEM problems with automatic
    differentiation capabilities across different backends.
    
    Parameters
    ----------
    problem : Problem
        Problem definition
    backend : str, optional
        AD backend name, by default 'jax'
    solver_options : dict, optional
        Solver configuration options
        
    Examples
    --------
    >>> problem = Problem(mesh, function_space)
    >>> solver = FEBMLSolver(problem)
    >>> solution = solver.solve()
    """
    
    def __init__(
        self,
        problem: Problem = None,
        backend: str = 'jax',
        solver_options: Dict[str, Any] = None
    ):
        self.problem = problem
        self.backend = get_backend(backend)
        self.solver_options = solver_options or {}
        
        # Solver state
        self.solution_history = []
        self.convergence_history = []
        self.timing_info = {}
        
        # Default solver options
        self.default_options = {
            'max_iterations': 100,
            'tolerance': 1e-8,
            'linear_solver': 'lu',
            'preconditioner': 'ilu',
            'monitor_convergence': True,
            'checkpoint_frequency': 10
        }
        self.options = {**self.default_options, **self.solver_options}
        
        logger.info(f"FEBMLSolver initialized with backend: {backend}")
    
    def solve(
        self,
        problem: Problem = None,
        parameters: Dict[str, Any] = None,
        return_convergence: bool = False
    ) -> Union[Any, tuple]:
        """Solve the finite element problem.
        
        Parameters
        ----------
        problem : Problem, optional
            Problem to solve, uses instance problem if None
        parameters : Dict[str, Any], optional
            Runtime parameters
        return_convergence : bool, optional
            Whether to return convergence information
            
        Returns
        -------
        solution : Any
            Solution field
        convergence_info : dict, optional
            Convergence information (if return_convergence=True)
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for FEM solving")
        
        prob = problem or self.problem
        if prob is None:
            raise ValueError("No problem provided")
        
        logger.info("Starting FEM solve")
        
        # Merge parameters
        solve_params = {**prob.parameters}
        if parameters:
            solve_params.update(parameters)
        
        # Create solution function
        u = fd.Function(prob.function_space)
        v = fd.TestFunction(prob.function_space)
        
        # Check if problem is linear or nonlinear
        is_linear = self._check_linearity(prob)
        
        if is_linear:
            solution = self._solve_linear(prob, u, v, solve_params)
        else:
            solution = self._solve_nonlinear(prob, u, v, solve_params)
        
        # Store solution
        self.solution_history.append(solution.copy(deepcopy=True))
        
        logger.info("FEM solve completed")
        
        if return_convergence:
            convergence_info = {
                'iterations': len(self.convergence_history),
                'residual_history': self.convergence_history.copy(),
                'linear_problem': is_linear,
                'solver_options': self.options.copy()
            }
            return solution, convergence_info
        
        return solution
    
    def _check_linearity(self, problem: Problem) -> bool:
        """Check if problem is linear."""
        for eq in problem.equations:
            # Simple check - could be more sophisticated
            if hasattr(eq['equation'], 'is_linear'):
                if not eq['equation'].is_linear:
                    return False
        return True
    
    def _solve_linear(
        self,
        problem: Problem,
        u: Any,
        v: Any,
        params: Dict[str, Any]
    ) -> Any:
        """Solve linear problem."""
        # Assemble system
        F = self._assemble_system(problem, u, v, params)
        
        # Apply boundary conditions
        bcs = problem._assemble_boundary_conditions()
        
        # Solve linear system
        solver_params = {
            'ksp_type': self.options.get('linear_solver', 'lu'),
            'pc_type': self.options.get('preconditioner', 'ilu'),
            'ksp_rtol': self.options.get('tolerance', 1e-8),
            'ksp_monitor': self.options.get('monitor_convergence', True)
        }
        
        fd.solve(F == 0, u, bcs=bcs, solver_parameters=solver_params)
        
        return u
    
    def _solve_nonlinear(
        self,
        problem: Problem,
        u: Any,
        v: Any,
        params: Dict[str, Any]
    ) -> Any:
        """Solve nonlinear problem using Newton's method."""
        max_iter = self.options.get('max_iterations', 100)
        tolerance = self.options.get('tolerance', 1e-8)
        
        # Initial guess (zero or user-provided)
        if 'initial_guess' in params:
            u.assign(params['initial_guess'])
        
        # Newton iteration
        for iteration in range(max_iter):
            # Assemble residual and Jacobian
            F = self._assemble_system(problem, u, v, params)
            J = fd.derivative(F, u)
            
            # Apply boundary conditions
            bcs = problem._assemble_boundary_conditions()
            
            # Solve for Newton correction
            du = fd.Function(u.function_space())
            
            solver_params = {
                'ksp_type': self.options.get('linear_solver', 'preonly'),
                'pc_type': self.options.get('preconditioner', 'lu'),
                'ksp_rtol': 1e-12,
                'ksp_monitor': False
            }
            
            fd.solve(J == -F, du, bcs=bcs, solver_parameters=solver_params)
            
            # Update solution
            u.assign(u + du)
            
            # Check convergence
            residual_norm = fd.sqrt(fd.assemble(fd.inner(F, F) * fd.dx))
            correction_norm = fd.sqrt(fd.assemble(fd.inner(du, du) * fd.dx))
            
            self.convergence_history.append({
                'iteration': iteration,
                'residual_norm': float(residual_norm),
                'correction_norm': float(correction_norm)
            })
            
            logger.debug(f"Newton iteration {iteration}: "
                        f"residual={residual_norm:.2e}, "
                        f"correction={correction_norm:.2e}")
            
            if correction_norm < tolerance:
                logger.info(f"Newton method converged in {iteration + 1} iterations")
                break
        else:
            logger.warning(f"Newton method did not converge in {max_iter} iterations")
        
        return u
    
    def _assemble_system(
        self,
        problem: Problem,
        u: Any,
        v: Any,
        params: Dict[str, Any]
    ) -> Any:
        """Assemble the weak form system."""
        F = 0
        
        for eq in problem.equations:
            if eq['active']:
                F += eq['equation'](u, v, params)
        
        return F
    
    def optimize_parameters(
        self,
        objective: Callable,
        initial_params: Dict[str, Any],
        bounds: Dict[str, tuple] = None,
        method: str = 'lbfgs',
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Optimize problem parameters using gradient-based methods.
        
        Parameters
        ----------
        objective : Callable
            Objective function to minimize
        initial_params : Dict[str, Any]
            Initial parameter values
        bounds : Dict[str, tuple], optional
            Parameter bounds
        method : str, optional
            Optimization method, by default 'lbfgs'
        options : Dict[str, Any], optional
            Optimization options
            
        Returns
        -------
        Dict[str, Any]
            Optimization result
        """
        if self.problem is None:
            raise ValueError("Problem must be set for optimization")
        
        return self.problem.optimize(
            objective=objective,
            initial_guess=initial_params,
            method=method,
            options=options
        )
    
    def compute_sensitivities(
        self,
        output_functional: Callable,
        parameters: List[str]
    ) -> Dict[str, np.ndarray]:
        """Compute sensitivities using adjoint method.
        
        Parameters
        ----------
        output_functional : Callable
            Output functional to differentiate
        parameters : List[str]
            Parameter names for sensitivity analysis
            
        Returns
        -------
        Dict[str, np.ndarray]
            Sensitivities for each parameter
        """
        # Placeholder for adjoint-based sensitivity computation
        logger.info(f"Computing sensitivities for parameters: {parameters}")
        
        sensitivities = {}
        for param in parameters:
            # This would implement adjoint method
            sensitivities[param] = np.zeros(1)  # Placeholder
        
        return sensitivities
    
    def verify_convergence(
        self,
        exact_solution: Callable = None,
        refinement_levels: int = 3
    ) -> Dict[str, Any]:
        """Verify solver convergence using mesh refinement.
        
        Parameters
        ----------
        exact_solution : Callable, optional
            Exact solution for error computation
        refinement_levels : int, optional
            Number of refinement levels, by default 3
            
        Returns
        -------
        Dict[str, Any]
            Convergence study results
        """
        if not HAS_FIREDRAKE or self.problem is None:
            return {}
        
        logger.info(f"Running convergence study with {refinement_levels} levels")
        
        results = {
            'mesh_sizes': [],
            'dofs': [],
            'errors': [],
            'convergence_rates': []
        }
        
        original_mesh = self.problem.mesh
        
        for level in range(refinement_levels):
            # Refine mesh
            if level > 0:
                self.problem.mesh = fd.refine(self.problem.mesh)
                self.problem.function_space = fd.FunctionSpace(
                    self.problem.mesh, 
                    self.problem.function_space.ufl_element()
                )
            
            # Solve on current mesh
            solution = self.solve()
            
            # Compute mesh size and DOFs
            h = fd.sqrt(2.0) / fd.sqrt(self.problem.mesh.num_cells())
            ndofs = self.problem.function_space.dim()
            
            results['mesh_sizes'].append(float(h))
            results['dofs'].append(ndofs)
            
            # Compute error if exact solution provided
            if exact_solution is not None:
                # Use first operator for error computation
                if self.problem.equations:
                    eq = self.problem.equations[0]['equation']
                    if hasattr(eq, 'compute_error'):
                        error = eq.compute_error(solution, exact_solution)
                        results['errors'].append(float(error))
            
            logger.info(f"Level {level}: h={h:.3e}, DOFs={ndofs}, "
                       f"error={results['errors'][-1]:.3e}" if results['errors'] else "")
        
        # Compute convergence rates
        if len(results['errors']) > 1:
            for i in range(1, len(results['errors'])):
                rate = np.log(results['errors'][i] / results['errors'][i-1]) / \
                       np.log(results['mesh_sizes'][i] / results['mesh_sizes'][i-1])
                results['convergence_rates'].append(rate)
        
        # Restore original mesh
        self.problem.mesh = original_mesh
        
        return results
    
    def benchmark_performance(self, num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark solver performance.
        
        Parameters
        ----------
        num_runs : int, optional
            Number of benchmark runs, by default 5
            
        Returns
        -------
        Dict[str, Any]
            Performance metrics
        """
        import time
        
        if self.problem is None:
            raise ValueError("Problem must be set for benchmarking")
        
        logger.info(f"Running performance benchmark with {num_runs} runs")
        
        times = []
        memory_usage = []
        
        for run in range(num_runs):
            start_time = time.time()
            
            # Solve problem
            solution = self.solve()
            
            end_time = time.time()
            solve_time = end_time - start_time
            times.append(solve_time)
            
            # Estimate memory usage (simplified)
            if hasattr(solution, 'dat'):
                mem_estimate = solution.dat.data.nbytes / (1024**2)  # MB
                memory_usage.append(mem_estimate)
        
        results = {
            'num_runs': num_runs,
            'solve_times': times,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'memory_usage': memory_usage,
            'mean_memory': np.mean(memory_usage) if memory_usage else 0,
            'dofs': self.problem.function_space.dim() if self.problem else 0
        }
        
        logger.info(f"Benchmark complete: {results['mean_time']:.3f}±{results['std_time']:.3f}s, "
                   f"{results['mean_memory']:.1f}MB")
        
        return results
    
    def save_solution(self, filename: str, solution: Any = None) -> None:
        """Save solution to file.
        
        Parameters
        ----------
        filename : str
            Output filename
        solution : Any, optional
            Solution to save, uses last solution if None
        """
        if not HAS_FIREDRAKE:
            logger.warning("Cannot save solution: Firedrake not available")
            return
        
        sol = solution or (self.solution_history[-1] if self.solution_history else None)
        if sol is None:
            logger.warning("No solution to save")
            return
        
        # Save using Firedrake's built-in functionality
        if filename.endswith('.pvd'):
            output_file = fd.File(filename)
            output_file.write(sol)
        elif filename.endswith('.h5'):
            # HDF5 format for checkpointing
            with fd.CheckpointFile(filename, 'w') as checkpoint:
                checkpoint.save_function(sol)
        else:
            logger.warning(f"Unsupported file format: {filename}")
        
        logger.info(f"Solution saved to {filename}")
    
    def load_solution(self, filename: str) -> Any:
        """Load solution from file.
        
        Parameters
        ----------
        filename : str
            Input filename
            
        Returns
        -------
        Any
            Loaded solution
        """
        if not HAS_FIREDRAKE or self.problem is None:
            logger.warning("Cannot load solution: Firedrake or problem not available")
            return None
        
        if filename.endswith('.h5'):
            with fd.CheckpointFile(filename, 'r') as checkpoint:
                solution = fd.Function(self.problem.function_space)
                checkpoint.load_function(solution)
                return solution
        else:
            logger.warning(f"Unsupported file format for loading: {filename}")
            return None
    
    def __repr__(self) -> str:
        return (f"FEBMLSolver("
                f"backend={self.backend.name}, "
                f"problem={self.problem is not None}, "
                f"solutions={len(self.solution_history)}"
                f")")


class SolverMethod(Enum):
    """Enumeration of solver methods."""
    DIRECT = "direct"
    ITERATIVE = "iterative"
    MULTIGRID = "multigrid"
    NEWTON = "newton"
    BFGS = "bfgs"
    CG = "cg"
    GMRES = "gmres"


class SolverConfig:
    """Configuration for solver parameters."""
    
    def __init__(
        self, 
        method: Union[str, SolverMethod] = SolverMethod.DIRECT,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
        preconditioner: Optional[str] = None,
        linear_solver: Optional[str] = None,
        nonlinear_solver: Optional[str] = None,
        **kwargs
    ):
        self.method = SolverMethod(method) if isinstance(method, str) else method
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.preconditioner = preconditioner
        self.linear_solver = linear_solver
        self.nonlinear_solver = nonlinear_solver
        self.extra_options = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'method': self.method.value,
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations,
            'preconditioner': self.preconditioner,
            'linear_solver': self.linear_solver,
            'nonlinear_solver': self.nonlinear_solver,
            **self.extra_options
        }