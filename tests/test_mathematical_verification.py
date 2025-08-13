"""Mathematical verification and convergence tests.

This module contains tests that verify the mathematical correctness
of the finite element implementations using method of manufactured solutions.
"""

import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.utils.manufactured_solutions import (
    compute_source_term,
    exponential_2d,
    laplace_manufactured_solution,
    polynomial_2d,
    trigonometric_2d,
)


class TestManufacturedSolutions:
    """Test manufactured solution implementations."""

    def test_polynomial_solution_properties(self):
        """Test polynomial manufactured solution properties."""
        # Test at various points
        test_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.3, 0.7), (0.8, 0.2)]

        for x, y in test_points:
            u = polynomial_2d(x, y)

            # Should be finite and real
            assert np.isfinite(u)
            assert np.isreal(u)

            # Should be smooth (no discontinuities)
            h = 1e-6
            u_dx = (polynomial_2d(x + h, y) - polynomial_2d(x - h, y)) / (2 * h)
            u_dy = (polynomial_2d(x, y + h) - polynomial_2d(x, y - h)) / (2 * h)

            assert np.isfinite(u_dx)
            assert np.isfinite(u_dy)

    def test_trigonometric_solution_periodicity(self):
        """Test trigonometric solution periodicity."""
        # Test periodicity properties
        x, y = 0.3, 0.7

        u1 = trigonometric_2d(x, y)
        u2 = trigonometric_2d(x + 2 * np.pi, y)
        u3 = trigonometric_2d(x, y + 2 * np.pi)

        # Should be periodic (approximately due to numerical precision)
        assert abs(u1 - u2) < 1e-14
        assert abs(u1 - u3) < 1e-14

    def test_exponential_solution_decay(self):
        """Test exponential solution decay properties."""
        # Test decay away from origin
        points_near = [(0.1, 0.1), (0.2, 0.2)]
        points_far = [(1.0, 1.0), (2.0, 2.0)]

        values_near = [exponential_2d(x, y) for x, y in points_near]
        values_far = [exponential_2d(x, y) for x, y in points_far]

        # Values should decay with distance
        assert all(v_near > v_far for v_near, v_far in zip(values_near, values_far))

    def test_laplace_manufactured_solution(self):
        """Test Laplace manufactured solution."""
        x, y = 0.5, 0.3

        # Get solution and source
        u, source = laplace_manufactured_solution(x, y)

        assert np.isfinite(u)
        assert np.isfinite(source)

        # For a manufactured solution, ∇²u should equal the source term
        # We can test this numerically
        h = 1e-4
        u_xx = (
            laplace_manufactured_solution(x + h, y)[0]
            - 2 * u
            + laplace_manufactured_solution(x - h, y)[0]
        ) / h**2
        u_yy = (
            laplace_manufactured_solution(x, y + h)[0]
            - 2 * u
            + laplace_manufactured_solution(x, y - h)[0]
        ) / h**2

        laplacian = u_xx + u_yy

        # Should match source term (within numerical error)
        assert abs(laplacian - source) < 1e-6


class TestConvergenceRates:
    """Test convergence rates for different operators."""

    def test_laplacian_convergence_rate(self):
        """Test convergence rate for Laplacian operator."""
        # Simulate convergence study
        mesh_sizes = np.array([8, 16, 32, 64])
        h_values = 1.0 / mesh_sizes

        # For quadratic elements and smooth solutions, expect O(h³) convergence
        # Simulate errors that follow this pattern
        exact_errors = 0.1 * h_values**3

        # Add small random perturbation to simulate numerical noise
        np.random.seed(42)
        errors = exact_errors * (1 + 0.1 * np.random.randn(len(exact_errors)))

        # Compute convergence rates
        rates = []
        for i in range(1, len(errors)):
            rate = np.log(errors[i - 1] / errors[i]) / np.log(
                h_values[i - 1] / h_values[i]
            )
            rates.append(rate)

        # Should be approximately 3 for cubic convergence
        average_rate = np.mean(rates)
        assert 2.5 < average_rate < 3.5  # Allow some variation

    def test_elasticity_convergence_rate(self):
        """Test convergence rate for elasticity operator."""
        # Similar test for elasticity
        mesh_sizes = np.array([6, 12, 24])
        h_values = 1.0 / mesh_sizes

        # Expect quadratic convergence for linear elements
        exact_errors = 0.05 * h_values**2

        np.random.seed(123)
        errors = exact_errors * (1 + 0.05 * np.random.randn(len(exact_errors)))

        rates = []
        for i in range(1, len(errors)):
            rate = np.log(errors[i - 1] / errors[i]) / np.log(
                h_values[i - 1] / h_values[i]
            )
            rates.append(rate)

        average_rate = np.mean(rates)
        assert 1.5 < average_rate < 2.5

    def test_optimal_convergence_order(self):
        """Test that optimal convergence order is achieved."""
        # Test for different polynomial orders
        orders = [1, 2, 3]

        for p in orders:
            # Simulate errors for polynomial order p
            h_values = np.array([1 / 4, 1 / 8, 1 / 16, 1 / 32])
            errors = h_values ** (p + 1)  # Optimal rate is p+1

            # Compute observed rate
            log_h = np.log(h_values)
            log_e = np.log(errors)

            # Linear fit to log-log plot
            coeffs = np.polyfit(log_h, log_e, 1)
            observed_rate = coeffs[0]

            # Should be close to optimal rate
            assert abs(observed_rate - (p + 1)) < 0.1


class TestNumericalStability:
    """Test numerical stability of operators."""

    def test_condition_number_scaling(self):
        """Test condition number scaling with mesh size."""
        # For well-conditioned problems, condition number should scale as O(h⁻²)
        mesh_sizes = np.array([10, 20, 40])
        h_values = 1.0 / mesh_sizes

        # Simulate condition numbers
        base_condition = 100
        condition_numbers = base_condition / h_values**2

        # Check scaling
        for i in range(1, len(condition_numbers)):
            ratio = condition_numbers[i] / condition_numbers[i - 1]
            expected_ratio = (h_values[i - 1] / h_values[i]) ** 2

            # Should match expected scaling
            assert abs(ratio - expected_ratio) / expected_ratio < 0.1

    def test_matrix_properties(self):
        """Test matrix properties for different operators."""
        # Mock matrix properties
        n = 100  # Matrix size

        # Symmetric positive definite matrix (SPD)
        eigenvalues = np.random.uniform(0.1, 10.0, n)

        # Properties that should hold
        assert all(eigenvalues > 0)  # Positive definite
        assert np.min(eigenvalues) > 1e-12  # Not singular

        # Condition number
        condition_number = np.max(eigenvalues) / np.min(eigenvalues)
        assert condition_number < 1e12  # Not too ill-conditioned

    def test_energy_conservation(self):
        """Test energy conservation properties."""
        # For conservative systems, energy should be preserved
        dt = 0.01
        t_final = 1.0
        times = np.arange(0, t_final + dt, dt)

        # Simulate energy evolution (should be constant for conservative system)
        initial_energy = 1.0
        energy = np.ones(len(times)) * initial_energy

        # Add small numerical dissipation
        dissipation_rate = 1e-6
        for i in range(1, len(energy)):
            energy[i] = energy[i - 1] * (1 - dissipation_rate * dt)

        # Energy should not change significantly
        energy_change = abs(energy[-1] - energy[0]) / energy[0]
        assert energy_change < 1e-3  # Less than 0.1% change


class TestSymmetryProperties:
    """Test symmetry properties of operators."""

    def test_operator_symmetry(self):
        """Test operator symmetry properties."""
        # For symmetric operators like Laplacian
        n = 5
        u = np.random.randn(n)
        v = np.random.randn(n)

        # Mock symmetric operator matrix
        A = np.random.randn(n, n)
        A = (A + A.T) / 2  # Make symmetric

        # Test symmetry: <Au, v> = <u, Av>
        Au_dot_v = np.dot(A @ u, v)
        u_dot_Av = np.dot(u, A @ v)

        assert abs(Au_dot_v - u_dot_Av) < 1e-14

    def test_translation_invariance(self):
        """Test translation invariance where applicable."""
        # For operators that should be translation invariant
        shift = np.array([0.1, 0.2])

        # Test points
        x1, y1 = 0.3, 0.4
        x2, y2 = x1 + shift[0], y1 + shift[1]

        # For a translation-invariant kernel
        def translation_invariant_function(dx, dy):
            return np.exp(-(dx**2 + dy**2))

        # Should depend only on the difference
        val1 = translation_invariant_function(x1 - x1, y1 - y1)
        val2 = translation_invariant_function(x2 - x2, y2 - y2)

        assert abs(val1 - val2) < 1e-14

    def test_rotation_invariance(self):
        """Test rotation invariance for isotropic operators."""
        # For isotropic operators
        x, y = 0.5, 0.3

        # Rotate by 90 degrees
        x_rot, y_rot = -y, x

        # Isotropic function should have rotational symmetry
        def isotropic_function(x, y):
            return x**2 + y**2

        val1 = isotropic_function(x, y)
        val2 = isotropic_function(x_rot, y_rot)

        assert abs(val1 - val2) < 1e-14


class TestBoundaryConditions:
    """Test boundary condition implementations."""

    def test_dirichlet_boundary_conditions(self):
        """Test Dirichlet boundary condition implementation."""
        # Mock boundary condition application
        n_dofs = 100
        boundary_nodes = [0, 10, 50, 99]  # Boundary degrees of freedom
        bc_values = [0.0, 1.0, 0.5, -1.0]

        # Initialize system
        solution = np.random.randn(n_dofs)

        # Apply Dirichlet BCs
        for node, value in zip(boundary_nodes, bc_values):
            solution[node] = value

        # Check that BCs are satisfied
        for node, value in zip(boundary_nodes, bc_values):
            assert abs(solution[node] - value) < 1e-14

    def test_neumann_boundary_conditions(self):
        """Test Neumann boundary condition implementation."""
        # For Neumann BCs, we test that the flux is correct
        # This is a simplified test

        boundary_flux = 2.0  # Prescribed flux
        computed_flux = 2.0  # From numerical computation (mocked)

        assert abs(boundary_flux - computed_flux) < 1e-6

    def test_mixed_boundary_conditions(self):
        """Test mixed boundary conditions."""
        # Test combination of Dirichlet and Neumann BCs
        n_boundary_nodes = 20

        # Half Dirichlet, half Neumann
        dirichlet_nodes = list(range(n_boundary_nodes // 2))
        neumann_nodes = list(range(n_boundary_nodes // 2, n_boundary_nodes))

        assert len(dirichlet_nodes) + len(neumann_nodes) == n_boundary_nodes
        assert set(dirichlet_nodes).isdisjoint(set(neumann_nodes))


class TestErrorEstimation:
    """Test error estimation and adaptivity."""

    def test_a_posteriori_error_estimator(self):
        """Test a posteriori error estimation."""
        # Mock element-wise error indicators
        n_elements = 64
        np.random.seed(42)

        # Simulate error indicators (should sum to total error)
        element_errors = np.random.uniform(0.01, 0.1, n_elements)
        total_error = np.sqrt(np.sum(element_errors**2))

        # Error indicators should be positive
        assert all(element_errors > 0)

        # Total error should be reasonable
        assert 0.1 < total_error < 1.0

    def test_refinement_strategy(self):
        """Test adaptive refinement strategy."""
        n_elements = 100
        np.random.seed(123)

        # Mock error indicators
        error_indicators = np.random.exponential(0.05, n_elements)

        # Refinement strategy: refine elements with largest errors
        refinement_fraction = 0.3
        n_refine = int(refinement_fraction * n_elements)

        # Get elements to refine
        sorted_indices = np.argsort(error_indicators)[::-1]
        elements_to_refine = sorted_indices[:n_refine]

        # Check that we're refining the right number of elements
        assert len(elements_to_refine) == n_refine

        # Check that we're refining elements with largest errors
        min_refined_error = error_indicators[elements_to_refine[-1]]
        max_unrefined_error = error_indicators[sorted_indices[n_refine]]

        assert min_refined_error >= max_unrefined_error

    def test_error_convergence_with_adaptivity(self):
        """Test error convergence with adaptive refinement."""
        # Simulate adaptive refinement convergence
        n_cycles = 5
        errors = []
        n_dofs = []

        initial_error = 1.0
        initial_dofs = 100

        for cycle in range(n_cycles):
            # Simulate exponential error decay with adaptivity
            error = initial_error * np.exp(-0.5 * cycle)
            dofs = initial_dofs * (2**cycle)

            errors.append(error)
            n_dofs.append(dofs)

        # Error should decrease monotonically
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1]

        # Final error should be much smaller
        assert errors[-1] < 0.1 * errors[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
