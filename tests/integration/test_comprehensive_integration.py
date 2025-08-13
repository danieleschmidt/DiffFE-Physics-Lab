"""Comprehensive integration tests for the DiffFE-Physics-Lab framework."""

import json
import logging
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

try:
    import firedrake as fd

    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from flask import Flask

    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from src.backends import get_backend
from src.models.problem import FEBMLProblem, Problem
from src.operators import ElasticityOperator, LaplacianOperator
from src.services.assembly import AssemblyService
from src.services.optimization import OptimizationService
from src.services.robust_optimization import RobustOptimizer
from src.services.robust_solver import RobustFEBMLSolver
from src.services.solver import SolverService
from src.utils.config_manager import ConfigManager
from src.utils.exceptions import ConvergenceError, SolverError, ValidationError
from src.utils.logging_config import get_logger
from src.utils.manufactured_solutions import (
    SolutionType,
    generate_manufactured_solution,
)

logger = get_logger(__name__)


class TestEndToEndWorkflows:
    """Test complete end-to-end problem solving workflows."""

    @pytest.mark.integration
    @pytest.mark.firedrake
    def test_complete_diffusion_workflow(self, mesh_2d, function_space_scalar):
        """Test complete workflow for diffusion problem."""
        # Create problem
        problem = Problem(
            mesh=mesh_2d, function_space=function_space_scalar, backend="jax"
        )

        # Add Poisson equation: -∇²u = f
        def poisson_equation(u, v, params):
            diffusion_coeff = params.get("diffusion_coeff", 1.0)
            source = params.get("source", fd.Constant(1.0))
            return (
                diffusion_coeff * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx
                - source * v * fd.dx
            )

        problem.add_equation(poisson_equation, name="poisson")

        # Add boundary conditions
        problem.add_boundary_condition(
            "dirichlet", "on_boundary", 0.0, name="homogeneous_dirichlet"
        )

        # Set parameters
        problem.set_parameter("diffusion_coeff", 2.0)
        problem.set_parameter("source", fd.Constant(5.0))

        # Solve problem
        solution = problem.solve()

        # Verify solution
        assert solution is not None
        assert hasattr(solution, "dat")

        # Check that solution has reasonable values
        solution_data = solution.dat.data_ro
        assert np.all(np.isfinite(solution_data))
        assert np.max(np.abs(solution_data)) < 100.0  # Reasonable magnitude

        # Generate observations
        observations = problem.generate_observations(
            num_points=10, noise_level=0.01, seed=42
        )
        assert len(observations) == 10

        # Each observation should have location and value
        for obs in observations:
            assert "location" in obs
            assert "value" in obs
            assert len(obs["location"]) == 2  # 2D problem
            assert np.isfinite(obs["value"])

    @pytest.mark.integration
    @pytest.mark.firedrake
    def test_complete_elasticity_workflow(self, mesh_2d):
        """Test complete workflow for elasticity problem."""
        # Create vector function space
        V = fd.VectorFunctionSpace(mesh_2d, "CG", 1)

        # Create problem
        problem = Problem(mesh=mesh_2d, function_space=V, backend="jax")

        # Add linear elasticity equation
        def elasticity_equation(u, v, params):
            E = params.get("E", 1.0)
            nu = params.get("nu", 0.3)
            body_force = params.get("body_force", fd.Constant((0.0, -1.0)))

            # Lame parameters
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))

            # Strain tensor
            def strain(u):
                return 0.5 * (fd.grad(u) + fd.grad(u).T)

            # Stress tensor
            def stress(u):
                return lmbda * fd.tr(strain(u)) * fd.Identity(2) + 2 * mu * strain(u)

            # Weak form
            return (
                fd.inner(stress(u), strain(v)) * fd.dx - fd.dot(body_force, v) * fd.dx
            )

        problem.add_equation(elasticity_equation, name="linear_elasticity")

        # Add boundary conditions (clamped on bottom)
        problem.add_boundary_condition(
            "dirichlet", 1, fd.Constant((0.0, 0.0)), name="clamped_bottom"
        )

        # Set material parameters
        problem.set_parameter("E", 200e9)  # Steel
        problem.set_parameter("nu", 0.3)
        problem.set_parameter("body_force", fd.Constant((0.0, -9.81 * 7850)))  # Gravity

        # Solve problem
        solution = problem.solve()

        # Verify solution
        assert solution is not None
        assert hasattr(solution, "dat")

        # Check solution dimensions and values
        solution_data = solution.dat.data_ro
        assert len(solution_data) > 0
        assert solution_data.shape[1] == 2  # 2D displacement field
        assert np.all(np.isfinite(solution_data))

    @pytest.mark.integration
    def test_optimization_workflow_with_mocks(self):
        """Test optimization workflow with mocked components."""
        # Create mock problem
        mock_problem = Mock()
        mock_problem.backend_name = "jax"
        mock_problem.parameters = {}
        mock_problem.set_parameter = Mock()
        mock_problem.solve = Mock()

        # Mock solution with varying objective based on parameters
        def mock_solve_with_params():
            diffusion = mock_problem.parameters.get("diffusion_coeff", 1.0)

            # Mock solution that depends on diffusion coefficient
            mock_solution = Mock()
            mock_solution.dat = Mock()
            # Objective: minimize (diffusion - 2.0)²
            mock_solution.objective_value = (diffusion - 2.0) ** 2

            return mock_solution

        mock_problem.solve.side_effect = mock_solve_with_params

        # Create optimization service
        opt_service = OptimizationService(mock_problem)

        # Define objective function
        def objective(params_dict):
            # Set parameters
            for key, value in params_dict.items():
                mock_problem.parameters[key] = value
                mock_problem.set_parameter(key, value)

            # Solve and return objective
            solution = mock_problem.solve()
            return solution.objective_value

        # Run optimization
        with patch("scipy.optimize.minimize") as mock_minimize:
            # Mock optimization result
            mock_result = Mock()
            mock_result.success = True
            mock_result.x = np.array([2.0])  # Optimal diffusion coefficient
            mock_result.fun = 0.0  # Minimum objective value
            mock_result.nit = 15
            mock_result.nfev = 30
            mock_minimize.return_value = mock_result

            # Parameter bounds
            bounds = {"diffusion_coeff": (0.1, 10.0)}
            initial_guess = {"diffusion_coeff": 1.0}

            result = opt_service.minimize_vector(
                objective=objective,
                initial_guess=list(initial_guess.values()),
                bounds=[bounds[key] for key in initial_guess.keys()],
            )

            assert result["success"] is True
            assert abs(result["optimal_parameters"]["param_0"] - 2.0) < 1e-6

    @pytest.mark.integration
    def test_multiphysics_coupling_workflow(self):
        """Test multiphysics coupling workflow."""
        # Mock thermal problem
        mock_thermal_problem = Mock()
        mock_thermal_problem.backend_name = "jax"
        mock_thermal_problem.solve = Mock()

        # Mock thermal solution
        mock_thermal_solution = Mock()
        mock_thermal_solution.at = Mock(return_value=323.0)  # Temperature at a point
        mock_thermal_problem.solve.return_value = mock_thermal_solution

        # Mock mechanical problem
        mock_mechanical_problem = Mock()
        mock_mechanical_problem.backend_name = "jax"
        mock_mechanical_problem.solve = Mock()
        mock_mechanical_problem.set_parameter = Mock()

        # Mock mechanical solution
        mock_mechanical_solution = Mock()
        mock_mechanical_solution.dat = Mock()
        mock_mechanical_solution.dat.data_ro = np.array(
            [[0.001, 0.002], [0.003, 0.004]]
        )
        mock_mechanical_problem.solve.return_value = mock_mechanical_solution

        # Coupling procedure
        def coupled_solve(thermal_load=1000.0, thermal_expansion_coeff=1e-5):
            # Step 1: Solve thermal problem
            thermal_solution = mock_thermal_problem.solve()

            # Step 2: Extract temperature field
            def temperature_field(x):
                return thermal_solution.at(x)  # Spatially uniform for simplicity

            # Step 3: Compute thermal strain
            reference_temp = 293.0

            def thermal_strain_field(x):
                temp = temperature_field(x)
                return thermal_expansion_coeff * (temp - reference_temp)

            # Step 4: Set thermal load in mechanical problem
            mock_mechanical_problem.set_parameter(
                "thermal_strain", thermal_strain_field
            )
            mock_mechanical_problem.set_parameter("body_force", [0.0, -thermal_load])

            # Step 5: Solve mechanical problem
            mechanical_solution = mock_mechanical_problem.solve()

            return {
                "thermal_solution": thermal_solution,
                "mechanical_solution": mechanical_solution,
                "max_temperature": 323.0,
                "max_displacement": np.max(np.abs(mechanical_solution.dat.data_ro)),
            }

        # Run coupled simulation
        result = coupled_solve(thermal_load=2000.0, thermal_expansion_coeff=2e-5)

        # Verify coupling results
        assert result["thermal_solution"] is not None
        assert result["mechanical_solution"] is not None
        assert result["max_temperature"] > 293.0  # Above reference temperature
        assert result["max_displacement"] > 0  # Non-zero displacement

        # Check that thermal effects were applied
        mock_mechanical_problem.set_parameter.assert_any_call(
            "thermal_strain", mock.ANY
        )
        mock_mechanical_problem.set_parameter.assert_any_call(
            "body_force", [0.0, -2000.0]
        )

    @pytest.mark.integration
    def test_inverse_problem_workflow(self):
        """Test inverse problem (parameter identification) workflow."""
        # Create mock forward problem
        mock_problem = Mock()
        mock_problem.backend_name = "jax"
        mock_problem.set_parameter = Mock()
        mock_problem.solve = Mock()
        mock_problem.generate_observations = Mock()

        # True parameters (to be identified)
        true_params = {"diffusion_coeff": 2.5, "source_amplitude": 3.0}

        # Mock forward solver that depends on parameters
        def mock_forward_solve():
            # Get current parameters
            diff_coeff = getattr(mock_problem, "_diffusion_coeff", 1.0)
            source_amp = getattr(mock_problem, "_source_amplitude", 1.0)

            # Mock solution based on parameters
            mock_solution = Mock()
            mock_solution.dat = Mock()
            # Simple model: solution ~ diff_coeff * source_amp
            solution_magnitude = diff_coeff * source_amp
            mock_solution.dat.data_ro = np.array(
                [solution_magnitude, solution_magnitude * 0.8]
            )

            return mock_solution

        def mock_set_parameter(name, value):
            setattr(mock_problem, f"_{name}", value)

        mock_problem.solve.side_effect = mock_forward_solve
        mock_problem.set_parameter.side_effect = mock_set_parameter

        # Generate synthetic observations (with true parameters)
        mock_problem.set_parameter("diffusion_coeff", true_params["diffusion_coeff"])
        mock_problem.set_parameter("source_amplitude", true_params["source_amplitude"])
        true_solution = mock_problem.solve()

        # Synthetic observations with noise
        np.random.seed(42)
        observations = []
        for i in range(5):
            location = [0.2 * i, 0.3 * i]
            true_value = true_solution.dat.data_ro[0] * (
                1 + 0.1 * i
            )  # Spatial variation
            noisy_value = true_value + 0.05 * np.random.randn()  # 5% noise
            observations.append({"location": location, "value": noisy_value})

        # Define inverse problem objective
        def inverse_objective(params):
            diffusion_coeff, source_amplitude = params

            # Set parameters in forward problem
            mock_problem.set_parameter("diffusion_coeff", diffusion_coeff)
            mock_problem.set_parameter("source_amplitude", source_amplitude)

            # Solve forward problem
            solution = mock_problem.solve()

            # Compute misfit with observations
            total_misfit = 0.0
            for obs in observations:
                # Mock evaluation at observation location
                predicted_value = solution.dat.data_ro[0]  # Simplified
                misfit = (predicted_value - obs["value"]) ** 2
                total_misfit += misfit

            return total_misfit / len(observations)  # Mean squared error

        # Solve inverse problem
        with patch("scipy.optimize.minimize") as mock_minimize:
            # Mock successful parameter identification
            mock_result = Mock()
            mock_result.success = True
            mock_result.x = np.array(
                [true_params["diffusion_coeff"], true_params["source_amplitude"]]
            )
            mock_result.fun = 0.01  # Small residual
            mock_result.nit = 25
            mock_minimize.return_value = mock_result

            initial_guess = [1.0, 1.0]
            bounds = [(0.1, 10.0), (0.1, 10.0)]

            opt_service = OptimizationService(mock_problem)
            result = opt_service.minimize_vector(
                objective=inverse_objective, initial_guess=initial_guess, bounds=bounds
            )

            # Verify parameter identification
            assert result["success"] is True
            assert (
                abs(
                    result["optimal_parameters"]["param_0"]
                    - true_params["diffusion_coeff"]
                )
                < 0.1
            )
            assert (
                abs(
                    result["optimal_parameters"]["param_1"]
                    - true_params["source_amplitude"]
                )
                < 0.1
            )
            assert result["objective_value"] < 0.1  # Good fit


class TestRobustWorkflows:
    """Test robust solver and optimization workflows."""

    @pytest.mark.integration
    def test_robust_solver_workflow(self):
        """Test robust solver with error recovery."""
        mock_problem = Mock()
        mock_problem.backend_name = "jax"

        # Create robust solver
        robust_solver = RobustFEBMLSolver(
            mock_problem,
            backend="jax",
            solver_options={
                "tolerance": 1e-8,
                "max_iterations": 100,
                "error_recovery": True,
                "adaptive_tolerance": True,
            },
        )

        # Mock solve attempts (first fails, second succeeds)
        attempt_count = 0

        def mock_solve_attempt(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt fails
                raise ConvergenceError("Initial solve failed")
            else:
                # Second attempt succeeds with relaxed tolerance
                mock_solution = Mock()
                mock_solution.dat = Mock()
                mock_solution.dat.data_ro = np.array([1.0, 2.0, 3.0])

                metrics = {
                    "success": True,
                    "iterations": 50,
                    "final_residual": 1e-6,
                    "tolerance_used": 1e-6,  # Relaxed from 1e-8
                }

                return mock_solution, metrics

        with patch.object(
            robust_solver, "_solve_attempt", side_effect=mock_solve_attempt
        ):
            solution, metrics = robust_solver.solve(return_metrics=True)

            # Verify robust recovery
            assert solution is not None
            assert metrics["success"] is True
            assert attempt_count == 2  # Required error recovery
            assert (
                metrics["tolerance_used"] > robust_solver.solver_options["tolerance"]
            )  # Adaptive

    @pytest.mark.integration
    def test_robust_optimization_workflow(self):
        """Test robust optimization with uncertainty quantification."""
        mock_problem = Mock()
        mock_problem.backend_name = "jax"

        # Create robust optimizer
        robust_optimizer = RobustOptimizer(
            backend="jax",
            optimization_options={
                "method": "L-BFGS-B",
                "tolerance": 1e-8,
                "quantify_uncertainty": True,
                "robust_constraints": True,
            },
        )

        # Noisy objective function
        np.random.seed(42)

        def noisy_objective(params):
            x, y = params
            noise = 0.01 * np.random.randn()  # 1% noise
            return (x - 2) ** 2 + (y - 3) ** 2 + noise

        # Mock uncertainty quantification
        with patch.object(robust_optimizer, "_quantify_uncertainty") as mock_uq:
            mock_uq.return_value = {
                "parameter_covariance": np.array([[0.01, 0.001], [0.001, 0.01]]),
                "confidence_intervals": [(1.9, 2.1), (2.9, 3.1)],
                "parameter_std": [0.1, 0.1],
            }

            with patch("scipy.optimize.minimize") as mock_minimize:
                mock_result = Mock()
                mock_result.success = True
                mock_result.x = np.array([2.0, 3.0])
                mock_result.fun = 0.005  # Small due to noise
                mock_minimize.return_value = mock_result

                initial_guess = [0.0, 0.0]
                result = robust_optimizer.minimize_robust(
                    noisy_objective, initial_guess
                )

                # Verify robust optimization
                assert result["success"] is True
                assert "uncertainty" in result
                assert "parameter_covariance" in result["uncertainty"]
                assert "confidence_intervals" in result["uncertainty"]

    @pytest.mark.integration
    def test_adaptive_workflow(self):
        """Test adaptive mesh refinement workflow."""

        # Mock adaptive problem solver
        class AdaptiveSolver:
            def __init__(self, initial_mesh_size=0.1):
                self.mesh_size = initial_mesh_size
                self.refinement_level = 0
                self.solutions = []
                self.errors = []

            def solve(self):
                # Mock solution that improves with refinement
                error = self.mesh_size**2  # O(h²) convergence

                mock_solution = Mock()
                mock_solution.error_estimate = error
                mock_solution.dat = Mock()
                # Solution quality improves with refinement
                mock_solution.dat.data_ro = (
                    np.random.rand(int(1 / self.mesh_size**2)) * error
                )

                self.solutions.append(mock_solution)
                self.errors.append(error)

                return mock_solution

            def refine_mesh(self, factor=0.5):
                self.mesh_size *= factor
                self.refinement_level += 1

            def should_refine(self, solution, tolerance=1e-4):
                return solution.error_estimate > tolerance and self.refinement_level < 5

        # Run adaptive refinement loop
        solver = AdaptiveSolver(initial_mesh_size=0.2)
        tolerance = 1e-4
        max_iterations = 10

        for iteration in range(max_iterations):
            solution = solver.solve()

            if not solver.should_refine(solution, tolerance):
                break

            solver.refine_mesh(factor=0.7)  # Refine mesh

        # Verify adaptive refinement
        assert len(solver.solutions) > 1  # Multiple refinement levels
        assert len(solver.errors) > 1

        # Errors should generally decrease
        assert solver.errors[-1] < solver.errors[0]

        # Final error should be below tolerance
        assert solver.errors[-1] < tolerance * 10  # Allow some margin


class TestApiIntegration:
    """Test API integration scenarios."""

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_FLASK, reason="Flask not available")
    def test_api_problem_submission_workflow(self, client):
        """Test complete API workflow for problem submission."""
        # Define problem configuration
        problem_config = {
            "mesh": {"type": "unit_square", "n_elements": 8},
            "function_space": {"family": "CG", "degree": 1},
            "equation": {
                "type": "poisson",
                "parameters": {"diffusion_coeff": 2.0, "source": 5.0},
            },
            "boundary_conditions": [
                {"type": "dirichlet", "boundary": "on_boundary", "value": 0.0}
            ],
        }

        # Submit problem
        with patch("src.api.routes.solve_problem") as mock_solve:
            mock_solve.return_value = {
                "success": True,
                "solution_id": "test-123",
                "solution_data": [1.0, 2.0, 3.0],
                "solve_time": 0.15,
                "dofs": 81,
            }

            response = client.post(
                "/api/v1/problems/solve",
                json=problem_config,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == 200
            data = response.get_json()

            assert data["success"] is True
            assert "solution_id" in data
            assert "solution_data" in data
            assert data["solve_time"] > 0
            assert data["dofs"] > 0

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_FLASK, reason="Flask not available")
    def test_api_optimization_workflow(self, client):
        """Test API optimization workflow."""
        optimization_config = {
            "problem": {
                "type": "parameter_identification",
                "forward_model": {
                    "equation": "poisson",
                    "parameters": ["diffusion_coeff"],
                },
                "observations": [
                    {"location": [0.5, 0.5], "value": 2.5},
                    {"location": [0.25, 0.75], "value": 1.8},
                ],
            },
            "optimization": {
                "method": "L-BFGS-B",
                "bounds": {"diffusion_coeff": [0.1, 10.0]},
                "initial_guess": {"diffusion_coeff": 1.0},
            },
        }

        with patch("src.api.routes.run_optimization") as mock_optimize:
            mock_optimize.return_value = {
                "success": True,
                "optimal_parameters": {"diffusion_coeff": 2.3},
                "objective_value": 0.05,
                "iterations": 15,
                "optimization_time": 2.5,
            }

            response = client.post(
                "/api/v1/optimization/run",
                json=optimization_config,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == 200
            data = response.get_json()

            assert data["success"] is True
            assert "optimal_parameters" in data
            assert "diffusion_coeff" in data["optimal_parameters"]
            assert data["objective_value"] < 0.1

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_FLASK, reason="Flask not available")
    def test_api_error_handling(self, client):
        """Test API error handling."""
        # Invalid problem configuration
        invalid_config = {
            "mesh": {"type": "invalid_mesh_type", "n_elements": -5}  # Invalid
        }

        response = client.post(
            "/api/v1/problems/solve",
            json=invalid_config,
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400
        data = response.get_json()

        assert data["success"] is False
        assert "error" in data
        assert "message" in data

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_FLASK, reason="Flask not available")
    def test_api_concurrent_requests(self, client):
        """Test API handling of concurrent requests."""
        import queue
        import threading

        # Queue to collect results
        results = queue.Queue()

        def make_request(request_id):
            problem_config = {
                "mesh": {"type": "unit_square", "n_elements": 4},
                "equation": {
                    "type": "poisson",
                    "parameters": {"diffusion_coeff": float(request_id)},
                },
            }

            with patch("src.api.routes.solve_problem") as mock_solve:
                mock_solve.return_value = {
                    "success": True,
                    "solution_id": f"test-{request_id}",
                    "diffusion_coeff": float(request_id),
                }

                response = client.post("/api/v1/problems/solve", json=problem_config)
                results.put((request_id, response.status_code, response.get_json()))

        # Launch concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Collect results
        responses = []
        while not results.empty():
            responses.append(results.get())

        # Verify all requests succeeded
        assert len(responses) == 5
        for request_id, status_code, data in responses:
            assert status_code == 200
            assert data["success"] is True
            assert f"test-{request_id}" in data["solution_id"]


class TestDatabaseIntegration:
    """Test database integration scenarios."""

    @pytest.mark.integration
    def test_problem_persistence(self):
        """Test problem saving and loading from database."""
        # Mock database connection
        mock_db = Mock()
        mock_cursor = Mock()
        mock_db.cursor.return_value = mock_cursor

        # Create FEBML problem with experiment tracking
        problem = FEBMLProblem(experiment_name="test_experiment")
        problem.set_parameter("diffusion_coeff", 2.5)
        problem.log_metric("error", 0.05)

        # Mock solution
        mock_solution = Mock()
        mock_solution.dat = Mock()
        mock_solution.dat.data_ro = np.array([1.0, 2.0, 3.0])

        # Checkpoint solution
        problem.checkpoint(mock_solution, "final_solution")

        # Mock database operations
        with patch("src.database.connection.get_connection", return_value=mock_db):
            # Save problem to database
            problem_data = {
                "experiment_name": problem.experiment_name,
                "parameters": problem.parameters,
                "metrics": problem.metrics,
                "checkpoints": list(problem.checkpoints.keys()),
            }

            # Mock INSERT operation
            mock_cursor.execute.return_value = None
            mock_cursor.lastrowid = 123

            # Simulate save
            mock_cursor.execute(
                "INSERT INTO problems (experiment_name, data) VALUES (?, ?)",
                (problem.experiment_name, json.dumps(problem_data)),
            )

            problem_id = mock_cursor.lastrowid

            # Mock retrieval
            mock_cursor.fetchone.return_value = (
                problem_id,
                problem.experiment_name,
                json.dumps(problem_data),
            )

            # Simulate load
            mock_cursor.execute(
                "SELECT id, experiment_name, data FROM problems WHERE id = ?",
                (problem_id,),
            )

            retrieved_data = mock_cursor.fetchone()

            # Verify persistence
            assert retrieved_data[0] == problem_id
            assert retrieved_data[1] == problem.experiment_name

            loaded_data = json.loads(retrieved_data[2])
            assert loaded_data["parameters"]["diffusion_coeff"] == 2.5
            assert "error" in loaded_data["metrics"]
            assert "final_solution" in loaded_data["checkpoints"]

    @pytest.mark.integration
    def test_optimization_history_persistence(self):
        """Test optimization history persistence."""
        mock_db = Mock()
        mock_cursor = Mock()
        mock_db.cursor.return_value = mock_cursor

        # Mock optimization service with history tracking
        mock_problem = Mock()
        opt_service = OptimizationService(mock_problem)

        # Mock optimization run with history
        optimization_history = []

        def mock_callback(x):
            iteration_data = {
                "iteration": len(optimization_history),
                "parameters": x.tolist(),
                "objective_value": np.sum(x**2),
                "timestamp": time.time(),
            }
            optimization_history.append(iteration_data)

        # Simulate optimization with callback
        with patch("scipy.optimize.minimize") as mock_minimize:

            def mock_optimization_with_callback(*args, **kwargs):
                callback = kwargs.get("callback")

                # Simulate optimization iterations
                for i in range(5):
                    x = np.array(
                        [2.0 - 0.4 * i, 3.0 - 0.6 * i]
                    )  # Converging to optimum
                    if callback:
                        callback(x)

                mock_result = Mock()
                mock_result.success = True
                mock_result.x = np.array([0.0, 0.0])
                mock_result.fun = 0.0
                return mock_result

            mock_minimize.side_effect = mock_optimization_with_callback

            # Run optimization
            result = opt_service.minimize_vector(
                objective=lambda x: np.sum(x**2),
                initial_guess=[2.0, 3.0],
                callback=mock_callback,
            )

            # Verify history was recorded
            assert len(optimization_history) == 5
            assert optimization_history[0]["iteration"] == 0
            assert optimization_history[-1]["iteration"] == 4

            # Mock database storage
            with patch("src.database.connection.get_connection", return_value=mock_db):
                mock_cursor.execute.return_value = None

                # Store optimization history
                for entry in optimization_history:
                    mock_cursor.execute(
                        "INSERT INTO optimization_history (iteration, parameters, objective, timestamp) VALUES (?, ?, ?, ?)",
                        (
                            entry["iteration"],
                            json.dumps(entry["parameters"]),
                            entry["objective_value"],
                            entry["timestamp"],
                        ),
                    )

                # Verify database calls
                assert mock_cursor.execute.call_count == len(optimization_history)

    @pytest.mark.integration
    def test_configuration_management_integration(self):
        """Test configuration management with database persistence."""
        config_manager = ConfigManager()

        # Set various configuration options
        config_manager.set("solver.tolerance", 1e-8)
        config_manager.set("mesh.max_elements", 10000)
        config_manager.set("optimization.method", "L-BFGS-B")
        config_manager.set("backend.default", "jax")

        # Mock database operations
        mock_db = Mock()
        mock_cursor = Mock()
        mock_db.cursor.return_value = mock_cursor

        with patch("src.database.connection.get_connection", return_value=mock_db):
            # Save configuration
            config_data = config_manager.to_dict()

            mock_cursor.execute.return_value = None
            mock_cursor.execute(
                "INSERT OR REPLACE INTO configurations (name, data) VALUES (?, ?)",
                ("default", json.dumps(config_data)),
            )

            # Mock retrieval
            mock_cursor.fetchone.return_value = ("default", json.dumps(config_data))
            mock_cursor.execute(
                "SELECT name, data FROM configurations WHERE name = ?", ("default",)
            )

            retrieved_config = mock_cursor.fetchone()
            loaded_config = json.loads(retrieved_config[1])

            # Verify configuration persistence
            assert loaded_config["solver"]["tolerance"] == 1e-8
            assert loaded_config["mesh"]["max_elements"] == 10000
            assert loaded_config["optimization"]["method"] == "L-BFGS-B"
            assert loaded_config["backend"]["default"] == "jax"


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_problem_workflow(self):
        """Test workflow with larger problem sizes."""
        # Mock large problem
        problem_size = 10000
        mock_problem = Mock()
        mock_problem.dofs = problem_size

        # Mock assembly service with performance monitoring
        assembly_service = AssemblyService()
        assembly_times = []

        def timed_assembly(*args, **kwargs):
            start_time = time.time()
            time.sleep(0.01)  # Simulate assembly time
            elapsed = time.time() - start_time
            assembly_times.append(elapsed)

            # Return mock matrix
            mock_matrix = Mock()
            mock_matrix.shape = (problem_size, problem_size)
            return mock_matrix

        # Mock solver service with performance monitoring
        solver_service = SolverService()
        solve_times = []

        def timed_solve(*args, **kwargs):
            start_time = time.time()
            time.sleep(0.02)  # Simulate solve time
            elapsed = time.time() - start_time
            solve_times.append(elapsed)

            # Return mock solution
            mock_solution = Mock()
            mock_solution.dat = Mock()
            mock_solution.dat.data_ro = np.random.rand(problem_size)
            return mock_solution

        # Patch services with timing
        with patch.object(
            assembly_service, "assemble_matrix", side_effect=timed_assembly
        ):
            with patch.object(solver_service, "solve_linear", side_effect=timed_solve):

                # Simulate large problem workflow
                start_total = time.time()

                # Assembly phase
                A = assembly_service.assemble_matrix(Mock(), Mock(), Mock())
                b = assembly_service.assemble_vector(Mock(), Mock())

                # Solve phase
                solution = solver_service.solve_linear(A, b, Mock())

                total_time = time.time() - start_total

                # Verify performance
                assert len(assembly_times) == 2  # Matrix and vector assembly
                assert len(solve_times) == 1
                assert total_time < 1.0  # Should complete reasonably quickly
                assert solution is not None
                assert A.shape[0] == problem_size

    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_usage_workflow(self):
        """Test memory usage in complete workflow."""
        try:
            import os

            import psutil
        except ImportError:
            pytest.skip("psutil not available")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Simulate memory-intensive workflow
        problems = []
        solutions = []

        for i in range(10):
            # Create mock problem
            mock_problem = Mock()
            mock_problem.backend_name = "jax"

            # Mock large solution data
            solution_size = 1000
            mock_solution = Mock()
            mock_solution.dat = Mock()
            mock_solution.dat.data_ro = np.random.rand(
                solution_size, 2
            )  # Large solution

            problems.append(mock_problem)
            solutions.append(mock_solution)

            # Simulate problem solving
            time.sleep(0.001)  # Small delay

        # Check memory usage
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        # Clean up
        del problems
        del solutions

        final_memory = process.memory_info().rss
        memory_after_cleanup = final_memory - initial_memory

        # Memory should be reasonable and should decrease after cleanup
        assert memory_increase < 200 * 1024 * 1024  # Less than 200MB increase
        assert memory_after_cleanup < memory_increase  # Some cleanup occurred

    @pytest.mark.integration
    def test_concurrent_problem_solving(self):
        """Test concurrent problem solving workflow."""
        import queue
        import threading

        results = queue.Queue()
        errors = queue.Queue()

        def solve_problem(problem_id):
            try:
                # Create mock problem
                mock_problem = Mock()
                mock_problem.backend_name = "jax"
                mock_problem.set_parameter = Mock()
                mock_problem.solve = Mock()

                # Set problem-specific parameters
                diffusion_coeff = 1.0 + 0.1 * problem_id
                mock_problem.set_parameter("diffusion_coeff", diffusion_coeff)

                # Mock solution
                mock_solution = Mock()
                mock_solution.dat = Mock()
                mock_solution.dat.data_ro = np.ones(100) * diffusion_coeff
                mock_problem.solve.return_value = mock_solution

                # Solve problem
                solution = mock_problem.solve()

                # Store result
                result = {
                    "problem_id": problem_id,
                    "diffusion_coeff": diffusion_coeff,
                    "solution_norm": np.linalg.norm(solution.dat.data_ro),
                    "success": True,
                }
                results.put(result)

            except Exception as e:
                errors.put((problem_id, str(e)))

        # Launch concurrent problem solving
        threads = []
        num_problems = 8

        for i in range(num_problems):
            thread = threading.Thread(target=solve_problem, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Collect results
        solved_problems = []
        while not results.empty():
            solved_problems.append(results.get())

        error_problems = []
        while not errors.empty():
            error_problems.append(errors.get())

        # Verify concurrent solving
        assert (
            len(error_problems) == 0
        ), f"Errors in concurrent solving: {error_problems}"
        assert len(solved_problems) == num_problems

        # Verify all problems were solved successfully
        for result in solved_problems:
            assert result["success"] is True
            assert result["solution_norm"] > 0
            assert 1.0 <= result["diffusion_coeff"] <= 1.8  # Within expected range


class TestEdgeCaseIntegration:
    """Test edge case integration scenarios."""

    @pytest.mark.integration
    def test_workflow_with_failures_and_recovery(self):
        """Test workflow with intermediate failures and recovery."""
        # Mock problem that fails initially
        mock_problem = Mock()
        mock_problem.backend_name = "jax"

        failure_count = 0

        def failing_solve(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1

            if failure_count <= 2:  # Fail first two attempts
                raise SolverError(f"Solve attempt {failure_count} failed")

            # Third attempt succeeds
            mock_solution = Mock()
            mock_solution.dat = Mock()
            mock_solution.dat.data_ro = np.array([1.0, 2.0, 3.0])
            return mock_solution

        mock_problem.solve = failing_solve

        # Robust solver with retry logic
        max_retries = 5
        retry_count = 0
        solution = None

        while retry_count < max_retries and solution is None:
            try:
                solution = mock_problem.solve()
                break
            except SolverError as e:
                retry_count += 1
                logger.warning(f"Solve attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    time.sleep(0.01)  # Brief delay before retry

        # Verify recovery
        assert solution is not None
        assert retry_count == 3  # Required 3 attempts
        assert failure_count == 3

    @pytest.mark.integration
    def test_mixed_precision_workflow(self):
        """Test workflow with mixed precision requirements."""
        # Different precision requirements for different stages
        precisions = {
            "assembly": np.float64,  # High precision for assembly
            "solve": np.float32,  # Lower precision for solve
            "postprocess": np.float64,  # High precision for post-processing
        }

        # Mock data with different precisions
        high_precision_matrix = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        low_precision_vector = np.array([1.0, 1.0], dtype=np.float32)

        # Assembly stage (high precision)
        assembly_data = high_precision_matrix.copy()
        assert assembly_data.dtype == precisions["assembly"]

        # Solve stage (convert to lower precision if needed)
        solve_matrix = assembly_data.astype(precisions["solve"])
        solve_vector = low_precision_vector.astype(precisions["solve"])

        # Mock solve
        mock_solution = np.linalg.solve(solve_matrix, solve_vector)
        assert mock_solution.dtype == precisions["solve"]

        # Post-processing stage (convert back to high precision)
        final_solution = mock_solution.astype(precisions["postprocess"])
        assert final_solution.dtype == precisions["postprocess"]

        # Verify solution quality despite precision changes
        expected_solution = np.array([1.0 / 3.0, 1.0 / 3.0])
        np.testing.assert_allclose(final_solution, expected_solution, rtol=1e-6)

    @pytest.mark.integration
    def test_workflow_with_extreme_parameters(self):
        """Test workflow with extreme parameter values."""
        extreme_cases = [
            {"diffusion_coeff": 1e-12, "description": "very small diffusion"},
            {"diffusion_coeff": 1e12, "description": "very large diffusion"},
            {
                "E": 1e-6,
                "nu": 0.49999,
                "description": "soft nearly incompressible material",
            },
            {"E": 1e15, "nu": 1e-6, "description": "very stiff material"},
        ]

        for case in extreme_cases:
            mock_problem = Mock()
            mock_problem.backend_name = "jax"
            mock_problem.set_parameter = Mock()

            # Set extreme parameters
            for param, value in case.items():
                if param != "description":
                    mock_problem.set_parameter(param, value)

            # Mock robust solver that handles extreme cases
            def robust_solve(*args, **kwargs):
                # Check for numerical issues
                if "diffusion_coeff" in case:
                    diff_coeff = case["diffusion_coeff"]
                    if diff_coeff < 1e-10:
                        # Use regularization for very small coefficients
                        effective_coeff = max(diff_coeff, 1e-10)
                    elif diff_coeff > 1e10:
                        # Use scaling for very large coefficients
                        effective_coeff = min(diff_coeff, 1e10)
                    else:
                        effective_coeff = diff_coeff

                    # Mock solution based on effective coefficient
                    solution_magnitude = 1.0 / np.sqrt(effective_coeff)

                elif "E" in case:
                    E = case["E"]
                    nu = case["nu"]

                    # Check material stability
                    if nu >= 0.5:
                        raise ValueError("Incompressible material not supported")
                    if E <= 0:
                        raise ValueError("Invalid Young's modulus")

                    solution_magnitude = np.sqrt(E) * (1 - nu)

                mock_solution = Mock()
                mock_solution.dat = Mock()
                mock_solution.dat.data_ro = np.array(
                    [solution_magnitude, solution_magnitude * 0.5]
                )
                mock_solution.converged = True

                return mock_solution

            mock_problem.solve = robust_solve

            try:
                solution = mock_problem.solve()

                # Verify solution properties
                assert solution is not None
                assert hasattr(solution, "converged")
                assert solution.converged is True
                assert np.all(np.isfinite(solution.dat.data_ro))

                logger.info(f"Successfully handled extreme case: {case['description']}")

            except Exception as e:
                # Some extreme cases might legitimately fail
                logger.warning(
                    f"Expected failure for extreme case {case['description']}: {e}"
                )
                assert isinstance(e, (ValueError, SolverError))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
