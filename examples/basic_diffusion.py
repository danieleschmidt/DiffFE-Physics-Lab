"""Basic diffusion example demonstrating the framework."""

import numpy as np
import logging

# DiffFE-Physics-Lab imports
from src.models import Problem, FEBMLProblem
from src.operators import laplacian
from src.services import FEBMLSolver, OptimizationService
from src.utils import compute_error, generate_manufactured_solution
from src.utils.manufactured_solutions import SolutionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for Firedrake
try:
    import firedrake as fd
    HAS_FIREDRAKE = True
    logger.info("Firedrake available - running full example")
except ImportError:
    HAS_FIREDRAKE = False
    logger.warning("Firedrake not available - running simplified example")


def create_unit_square_mesh(n_elements: int = 32):
    """Create unit square mesh."""
    if not HAS_FIREDRAKE:
        return None
    
    return fd.UnitSquareMesh(n_elements, n_elements)


def solve_basic_diffusion():
    """Solve basic diffusion problem: -∇²u = f on unit square."""
    
    if not HAS_FIREDRAKE:
        logger.info("Firedrake not available - skipping FEM solve")
        return
    
    logger.info("=== Basic Diffusion Problem ===")
    
    # Create mesh and function space
    mesh = create_unit_square_mesh(32)
    V = fd.FunctionSpace(mesh, "CG", 1)
    
    logger.info(f"Created mesh with {mesh.num_cells()} cells, {V.dim()} DOFs")
    
    # Create problem
    problem = Problem(mesh=mesh, function_space=V)
    
    # Define equation: -∇²u = f
    def diffusion_equation(u, v, params):
        return laplacian(u, v, diffusion_coeff=1.0, source=params.get('source'))
    
    problem.add_equation(diffusion_equation, name="diffusion")
    
    # Add boundary conditions: u = 0 on boundary
    problem.add_boundary_condition(
        bc_type='dirichlet',
        boundary_id='on_boundary',
        value=0.0
    )
    
    # Set source term
    def source_function(x):
        return 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    problem.set_parameter('source', source_function)
    
    # Solve problem
    solver = FEBMLSolver(problem)
    solution = solver.solve()
    
    logger.info("Problem solved successfully")
    
    # Exact solution for verification
    def exact_solution(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Compute error
    l2_error = compute_error(solution, exact_solution, 'L2')
    h1_error = compute_error(solution, exact_solution, 'H1')
    
    logger.info(f"L2 error: {l2_error:.6e}")
    logger.info(f"H1 error: {h1_error:.6e}")
    
    # Save solution
    try:
        solver.save_solution("basic_diffusion_solution.pvd", solution)
        logger.info("Solution saved to basic_diffusion_solution.pvd")
    except Exception as e:
        logger.warning(f"Could not save solution: {e}")
    
    return solution, l2_error, h1_error


def solve_with_manufactured_solution():
    """Solve diffusion using manufactured solution."""
    
    if not HAS_FIREDRAKE:
        logger.info("Firedrake not available - skipping MMS example")
        return
    
    logger.info("=== Manufactured Solution Example ===")
    
    # Generate manufactured solution
    mms = generate_manufactured_solution(
        solution_type=SolutionType.TRIGONOMETRIC,
        dimension=2,
        parameters={'frequency': 2.0, 'amplitude': 1.0}
    )
    
    logger.info(f"Generated {mms['type']} manufactured solution")
    
    # Create mesh and function space  
    mesh = create_unit_square_mesh(64)
    V = fd.FunctionSpace(mesh, "CG", 2)  # Higher order for better accuracy
    
    # Create FEBML problem with experiment tracking
    problem = FEBMLProblem(
        mesh=mesh, 
        function_space=V,
        experiment_name="mms_diffusion"
    )
    
    # Add diffusion equation
    def mms_equation(u, v, params):
        return laplacian(u, v, diffusion_coeff=1.0, source=mms['source'])
    
    problem.add_equation(mms_equation)
    
    # Boundary conditions from manufactured solution
    problem.add_boundary_condition(
        bc_type='dirichlet',
        boundary_id='on_boundary', 
        value=mms['solution']
    )
    
    # Solve
    solver = FEBMLSolver(problem)
    solution = solver.solve()
    
    # Compute errors
    l2_error = compute_error(solution, mms['solution'], 'L2')
    h1_error = compute_error(solution, mms['solution'], 'H1')
    
    # Log metrics
    problem.log_metric('L2_error', l2_error)
    problem.log_metric('H1_error', h1_error)
    problem.log_metric('DOFs', V.dim())
    
    logger.info(f"MMS L2 error: {l2_error:.6e}")
    logger.info(f"MMS H1 error: {h1_error:.6e}")
    
    # Checkpoint solution
    problem.checkpoint(solution, "final_solution")
    
    return solution, l2_error, h1_error


def parameter_optimization_example():
    """Example of parameter optimization using inverse problem."""
    
    if not HAS_FIREDRAKE:
        logger.info("Firedrake not available - skipping optimization example")
        return
    
    logger.info("=== Parameter Optimization Example ===")
    
    # Create mesh and function space
    mesh = create_unit_square_mesh(32)
    V = fd.FunctionSpace(mesh, "CG", 1)
    
    # True diffusion coefficient (to be recovered)
    true_diffusion_coeff = 2.5
    
    # Generate synthetic observations
    def true_solution(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    def true_source(x):
        return 2 * true_diffusion_coeff * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Create problem for forward solves
    problem = Problem(mesh=mesh, function_space=V)
    
    def diffusion_eq(u, v, params):
        kappa = params.get('diffusion_coeff', 1.0)
        return laplacian(u, v, diffusion_coeff=kappa, source=params.get('source'))
    
    problem.add_equation(diffusion_eq)
    problem.add_boundary_condition('dirichlet', 'on_boundary', 0.0)
    problem.set_parameter('source', true_source)
    
    # Generate synthetic observations
    problem.set_parameter('diffusion_coeff', true_diffusion_coeff)
    solver = FEBMLSolver(problem)
    true_sol = solver.solve()
    
    # Add noise to observations
    np.random.seed(42)
    observations = true_sol.dat.data + 0.01 * np.random.normal(size=len(true_sol.dat.data))
    
    logger.info(f"Generated {len(observations)} observations with 1% noise")
    
    # Define objective function
    def objective_function(params_dict):
        kappa = params_dict.get('diffusion_coeff', 1.0)
        
        # Solve forward problem
        problem.set_parameter('diffusion_coeff', kappa)
        computed_sol = solver.solve()
        
        # Compute misfit
        misfit = np.sum((computed_sol.dat.data - observations)**2)
        return misfit
    
    # Optimize diffusion coefficient
    opt_service = OptimizationService(problem)
    
    initial_guess = {'diffusion_coeff': 1.0}
    bounds = [(0.1, 10.0)]
    
    logger.info(f"Starting optimization with initial guess: {initial_guess}")
    logger.info(f"True value: {true_diffusion_coeff}")
    
    result = opt_service.minimize_vector(
        objective=objective_function,
        initial_guess=np.array([1.0]),
        bounds=bounds,
        method=opt_service.OptimizationMethod.LBFGS
    )
    
    recovered_coeff = result.optimal_parameters['param_0']
    error = abs(recovered_coeff - true_diffusion_coeff)
    
    logger.info(f"Optimization completed: {result.success}")
    logger.info(f"Recovered coefficient: {recovered_coeff:.6f}")
    logger.info(f"True coefficient: {true_diffusion_coeff:.6f}")
    logger.info(f"Absolute error: {error:.6f}")
    logger.info(f"Relative error: {100*error/true_diffusion_coeff:.2f}%")
    logger.info(f"Function evaluations: {result.function_evaluations}")
    
    return result


def convergence_study():
    """Perform mesh convergence study."""
    
    if not HAS_FIREDRAKE:
        logger.info("Firedrake not available - skipping convergence study")
        return
    
    logger.info("=== Convergence Study ===")
    
    # Exact solution
    def exact_solution(x):
        return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    def source_function(x):
        return 2 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    
    # Test different mesh sizes
    mesh_sizes = [8, 16, 32, 64]
    errors = []
    dofs = []
    
    for n in mesh_sizes:
        logger.info(f"Testing mesh size: {n}x{n}")
        
        # Create mesh and function space
        mesh = create_unit_square_mesh(n)
        V = fd.FunctionSpace(mesh, "CG", 1)
        
        # Create and solve problem
        problem = Problem(mesh=mesh, function_space=V)
        
        def diffusion_eq(u, v, params):
            return laplacian(u, v, diffusion_coeff=1.0, source=params.get('source'))
        
        problem.add_equation(diffusion_eq)
        problem.add_boundary_condition('dirichlet', 'on_boundary', 0.0)
        problem.set_parameter('source', source_function)
        
        solver = FEBMLSolver(problem)
        solution = solver.solve()
        
        # Compute error
        l2_error = compute_error(solution, exact_solution, 'L2')
        errors.append(l2_error)
        dofs.append(V.dim())
        
        logger.info(f"  DOFs: {V.dim()}, L2 error: {l2_error:.6e}")
    
    # Compute convergence rates
    h_values = [1.0/n for n in mesh_sizes]
    
    logger.info("\nConvergence Analysis:")
    logger.info("h\t\tDOFs\t\tL2 Error\t\tRate")
    logger.info("-" * 50)
    
    for i, (h, ndof, err) in enumerate(zip(h_values, dofs, errors)):
        if i == 0:
            logger.info(f"{h:.4f}\t\t{ndof}\t\t{err:.6e}\t\t-")
        else:
            rate = np.log(errors[i] / errors[i-1]) / np.log(h_values[i] / h_values[i-1])
            logger.info(f"{h:.4f}\t\t{ndof}\t\t{err:.6e}\t\t{rate:.2f}")
    
    # Expected rate for P1 elements is 2.0
    if len(errors) >= 2:
        avg_rate = np.mean([
            np.log(errors[i] / errors[i-1]) / np.log(h_values[i] / h_values[i-1])
            for i in range(1, len(errors))
        ])
        logger.info(f"\nAverage convergence rate: {avg_rate:.2f}")
    
    return h_values, errors, dofs


def main():
    """Run all examples."""
    logger.info("Starting DiffFE-Physics-Lab Basic Examples")
    logger.info("=" * 50)
    
    try:
        # Basic diffusion solve
        solve_basic_diffusion()
        logger.info("")
        
        # Manufactured solution verification
        solve_with_manufactured_solution()
        logger.info("")
        
        # Parameter optimization
        parameter_optimization_example()
        logger.info("")
        
        # Convergence study
        convergence_study()
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\nAll examples completed!")


if __name__ == "__main__":
    main()