#!/usr/bin/env python3
"""Basic working example demonstrating DiffFE-Physics-Lab functionality."""

import os
import sys
import numpy as np

# Add src to path for standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src
from src.models import Problem, Field, BoundaryCondition
from src.backends import get_backend


def main():
    """Demonstrate basic functionality without external dependencies."""
    print("🧪 DiffFE-Physics-Lab - Generation 1 Demo")
    print("=" * 50)

    # 1. Backend demonstration
    print("\n📊 Available backends:")
    backend = get_backend("numpy")
    print(f"   - NumPy backend: {backend.name} (available: {backend.is_available})")

    # 2. Basic differentiation
    print("\n🔢 Testing automatic differentiation:")

    def test_function(x):
        return x**2 + 2 * x + 1

    grad_func = backend.grad(test_function)
    x_test = 3.0
    gradient = grad_func(x_test)
    expected = 2 * x_test + 2  # Analytical gradient

    print(f"   f(x) = x² + 2x + 1")
    print(f"   f'({x_test}) = {gradient:.6f} (expected: {expected})")
    print(f"   Error: {abs(gradient - expected):.2e}")

    # 3. Vector differentiation
    print("\n📈 Vector function differentiation:")

    def vector_function(x):
        return np.array([x[0] ** 2, x[1] ** 2, x[0] * x[1]])

    jac_func = backend.jacobian(vector_function)
    x_vec = np.array([2.0, 3.0])
    jacobian = jac_func(x_vec)

    expected_jac = np.array(
        [[2 * x_vec[0], 0], [0, 2 * x_vec[1]], [x_vec[1], x_vec[0]]]
    )

    print(f"   F(x,y) = [x², y², xy]")
    print(f"   Jacobian at {x_vec}:")
    print(f"   Computed:\n{jacobian}")
    print(f"   Expected:\n{expected_jac}")
    print(f"   Max error: {np.max(np.abs(jacobian - expected_jac)):.2e}")

    # 4. Problem setup (minimal)
    print("\n🏗️  Problem setup:")
    problem = Problem(backend="numpy")
    print(f"   Created problem with backend: {problem.backend_name}")

    # 5. Field demonstration
    print("\n🌊 Field operations:")
    # Constant field
    field1 = Field("temperature", values=np.array([25.0, 30.0, 20.0]))
    points = np.array([[0, 0], [1, 0], [0, 1]])
    field_values = field1.evaluate(points)
    print(f"   Field values at 3 points: {field_values}")

    # Function-based field
    field2 = Field("velocity", values=lambda p: p[:, 0] + p[:, 1])
    func_values = field2.evaluate(points)
    print(f"   Function field values: {func_values}")

    # 6. Boundary conditions
    print("\n🔒 Boundary conditions:")
    bc1 = BoundaryCondition("dirichlet", 0.0, "inlet")
    bc2 = BoundaryCondition("neumann", lambda p: np.sin(p[:, 0]), "outlet")

    test_points = np.array([[0, 0], [1, 0]])
    bc1_values = bc1.evaluate(test_points)
    bc2_values = bc2.evaluate(test_points)

    print(f"   Dirichlet BC values: {bc1_values}")
    print(f"   Neumann BC values: {bc2_values}")

    # 7. Basic optimization
    print("\n🎯 Basic optimization:")

    def objective(x):
        return (x[0] - 2) ** 2 + (x[1] - 1) ** 2

    initial = np.array([0.0, 0.0])
    optimal = backend.optimize(objective, initial)

    print(f"   Minimize (x-2)² + (y-1)²")
    print(f"   Starting point: {initial}")
    print(f"   Optimal point: {optimal}")
    print(f"   Expected: [2, 1]")
    print(f"   Error: {np.linalg.norm(optimal - np.array([2, 1])):.3f}")

    print("\n✅ Generation 1 basic functionality working!")
    print("🚀 Ready for Generation 2 robustness improvements")


if __name__ == "__main__":
    main()
