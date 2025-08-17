#!/usr/bin/env python3
"""
Enhanced API Demonstration - DiffFE-Physics-Lab
===============================================

This example demonstrates the new enhanced API layer that provides
simplified interfaces while maintaining full functionality.
"""

import sys
import os
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_enhanced_febml():
    """Demonstrate enhanced FEBMLProblem API."""
    print("=" * 60)
    print("Enhanced FEBML API Demo")
    print("=" * 60)
    
    from core.enhanced_api import FEBMLProblem, ProblemConfig, quick_solve
    
    # Create problem with configuration
    config = ProblemConfig(
        mesh_size=100,
        element_order=2,
        backend="numpy",
        precision="float64"
    )
    
    problem = FEBMLProblem(config)
    
    # Add physics equations
    problem.add_equation("laplacian", conductivity=1.5)
    problem.add_equation("advection", velocity=[0.1, 0.0])
    
    # Set boundary conditions
    problem.set_boundary_condition("left", "dirichlet", 0.0)
    problem.set_boundary_condition("right", "dirichlet", 1.0)
    problem.set_boundary_condition("top", "neumann", 0.0)
    problem.set_boundary_condition("bottom", "neumann", 0.0)
    
    # Add source term
    problem.add_source_term(lambda x, y: 1.0, region="all")
    
    # Add observers
    problem.add_observer([0.5, 0.5], "center_point")
    problem.add_observer([0.25, 0.25], "quarter_point")
    
    # Solve problem
    solution = problem.solve(method="direct")
    
    print(f"Solution computed: {solution}")
    print(f"Problem metrics: {problem.metrics}")
    
    # Compute error (against itself for demo)
    error = problem.compute_error(solution, norm="L2")
    print(f"L2 error: {error}")
    
    # Export solution
    problem.export_solution("solution.vtk", format="vtk")
    
    return problem

def demo_multiphysics():
    """Demonstrate MultiPhysics API."""
    print("\n" + "=" * 60)
    print("MultiPhysics API Demo")
    print("=" * 60)
    
    from core.enhanced_api import MultiPhysics, FEBMLProblem, ProblemConfig
    
    # Create multi-physics system
    multiphysics = MultiPhysics()
    
    # Create fluid domain
    fluid_config = ProblemConfig(mesh_size=50, backend="jax")
    fluid_problem = FEBMLProblem(fluid_config)
    fluid_problem.add_equation("navier_stokes", reynolds_number=100)
    fluid_problem.add_equation("incompressibility")
    
    multiphysics.add_domain("fluid", fluid_problem, ["navier_stokes", "incompressibility"])
    
    # Create solid domain
    solid_config = ProblemConfig(mesh_size=30, backend="numpy")
    solid_problem = FEBMLProblem(solid_config)
    solid_problem.add_equation("elasticity", youngs_modulus=1e6, poisson_ratio=0.3)
    
    multiphysics.add_domain("solid", solid_problem, ["elasticity"])
    
    # Add interface coupling
    multiphysics.add_interface("fluid_solid", "fluid", "solid", "dirichlet_neumann")
    
    # Solve coupled system
    solutions = multiphysics.solve(coupling_scheme="fixed_point", max_iterations=10)
    
    print(f"Coupled solutions: {solutions}")
    print(f"Domains solved: {list(solutions.keys())}")
    
    return multiphysics

def demo_hybrid_solver():
    """Demonstrate HybridSolver API."""
    print("\n" + "=" * 60)
    print("Hybrid Solver API Demo")
    print("=" * 60)
    
    from core.enhanced_api import HybridSolver, FEBMLProblem, ProblemConfig
    
    # Create FEM component
    fem_config = ProblemConfig(mesh_size=40, backend="jax")
    fem_problem = FEBMLProblem(fem_config)
    fem_problem.add_equation("laplacian", conductivity=1.0)
    fem_problem.set_boundary_condition("boundary", "dirichlet", 0.0)
    
    # Mock neural network (in real implementation, this would be actual NN)
    class MockNeuralNetwork:
        def predict(self, x):
            return x * 0.1  # Simple linear correction
    
    neural_net = MockNeuralNetwork()
    
    # Create hybrid solver
    hybrid = HybridSolver(fem_problem, neural_net)
    hybrid.set_coupling_strength(0.3)  # 30% NN, 70% FEM
    
    # Add training data for NN component
    x_train = [[i/10] for i in range(10)]
    y_train = [x[0]**2 for x in x_train]  # Quadratic target
    
    hybrid.add_training_data(x_train, y_train)
    
    # Train neural component
    training_result = hybrid.train_neural_component(epochs=50, learning_rate=0.001)
    print(f"Training result: {training_result}")
    
    # Solve hybrid problem
    solution = hybrid.solve(
        use_fem_for=["diffusion"],
        use_nn_for=["reaction"],
        tolerance=1e-8
    )
    
    print(f"Hybrid solution: {solution}")
    
    return hybrid

def demo_quick_solve():
    """Demonstrate quick solve functionality."""
    print("\n" + "=" * 60)
    print("Quick Solve Demo")
    print("=" * 60)
    
    from core.enhanced_api import quick_solve
    
    # Simple Laplace equation
    solution1 = quick_solve(
        equation="laplacian",
        boundary_conditions={"left": 0.0, "right": 1.0},
        mesh_size=50,
        conductivity=1.0
    )
    print(f"Laplace solution: {solution1}")
    
    # Heat equation with source
    solution2 = quick_solve(
        equation="heat",
        boundary_conditions={"all_boundaries": 0.0},
        mesh_size=30,
        source_strength=10.0,
        time_step=0.01
    )
    print(f"Heat equation solution: {solution2}")
    
    return solution1, solution2

def demo_factory_pattern():
    """Demonstrate problem factory functionality."""
    print("\n" + "=" * 60)
    print("Problem Factory Demo")
    print("=" * 60)
    
    from core.enhanced_api import create_problem
    
    # Create different problem types
    febml = create_problem("febml", mesh_size=100, backend="jax", gpu_enabled=True)
    print(f"Created FEBML problem: {type(febml).__name__}")
    print(f"Config: mesh_size={febml.config.mesh_size}, backend={febml.config.backend}")
    
    multiphysics = create_problem("multiphysics", mesh_size=200, parallel=True)
    print(f"Created MultiPhysics problem: {type(multiphysics).__name__}")
    print(f"Config: mesh_size={multiphysics.config.mesh_size}, parallel={multiphysics.config.parallel}")
    
    return febml, multiphysics

def run_api_validation():
    """Validate enhanced API functionality."""
    print("\n" + "=" * 60)
    print("API Validation Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: ProblemConfig validation
    total_tests += 1
    try:
        from core.enhanced_api import ProblemConfig
        config = ProblemConfig(mesh_size=50, backend="jax")
        assert config.mesh_size == 50
        assert config.backend == "jax"
        print("✓ Test 1 passed: ProblemConfig creation and validation")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
    
    # Test 2: FEBMLProblem instantiation
    total_tests += 1
    try:
        from core.enhanced_api import FEBMLProblem, ProblemConfig
        problem = FEBMLProblem(ProblemConfig(mesh_size=10))
        problem.add_equation("laplacian", conductivity=1.0)
        assert len(problem.equations) == 1
        print("✓ Test 2 passed: FEBMLProblem equation addition")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
    
    # Test 3: Boundary condition setting
    total_tests += 1
    try:
        from core.enhanced_api import FEBMLProblem
        problem = FEBMLProblem()
        problem.set_boundary_condition("left", "dirichlet", 1.0)
        assert "left" in problem.boundary_conditions
        assert problem.boundary_conditions["left"]["type"] == "dirichlet"
        print("✓ Test 3 passed: Boundary condition setting")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
    
    # Test 4: MultiPhysics domain addition
    total_tests += 1
    try:
        from core.enhanced_api import MultiPhysics, FEBMLProblem
        mp = MultiPhysics()
        problem = FEBMLProblem()
        mp.add_domain("test", problem, ["laplacian"])
        assert "test" in mp.domains
        print("✓ Test 4 passed: MultiPhysics domain addition")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
    
    # Test 5: Problem factory
    total_tests += 1
    try:
        from core.enhanced_api import create_problem
        problem = create_problem("febml", mesh_size=25)
        assert problem.config.mesh_size == 25
        print("✓ Test 5 passed: Problem factory")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
    
    print(f"\nAPI Validation Results: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests

def main():
    """Main demonstration function."""
    print("DiffFE-Physics-Lab Enhanced API Demonstration")
    print("This example shows the new simplified API layer\n")
    
    try:
        # Run all demonstrations
        demo_enhanced_febml()
        demo_multiphysics()
        demo_hybrid_solver()
        demo_quick_solve()
        demo_factory_pattern()
        
        # Validate API
        all_passed = run_api_validation()
        
        print("\n" + "=" * 60)
        print("ENHANCED API SUMMARY")
        print("=" * 60)
        print("✓ Enhanced FEBMLProblem with simplified equation setup")
        print("✓ MultiPhysics coupling with automatic domain management")
        print("✓ HybridSolver combining FEM with neural networks")
        print("✓ Quick solve functions for rapid prototyping")
        print("✓ Factory patterns for consistent problem creation")
        print(f"✓ API validation: {'PASSED' if all_passed else 'FAILED'}")
        
        print("\nKey Enhanced Features:")
        print("- Simplified configuration with ProblemConfig")
        print("- Fluent API for equation and boundary condition setup")
        print("- Built-in validation and error checking")
        print("- Comprehensive logging and metrics collection")
        print("- Factory patterns for consistent object creation")
        print("- Support for quick prototyping workflows")
        
        return 0
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())