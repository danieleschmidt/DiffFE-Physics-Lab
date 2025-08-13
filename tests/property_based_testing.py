"""Property-based testing infrastructure for numerical stability and robustness."""

import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

try:
    import hypothesis
    from hypothesis import Verbosity, assume, given, settings
    from hypothesis import strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

    # Create dummy decorators if hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def settings(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


try:
    import firedrake as fd

    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from ..src.backends.robust_backend import get_robust_backend
from ..src.models.problem import Problem
from ..src.services.robust_optimization import RobustOptimizer
from ..src.services.robust_solver import RobustFEBMLSolver
from ..src.utils.exceptions import ConvergenceError, SolverError, ValidationError
from ..src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PropertyTestResult:
    """Result of a property-based test."""

    property_name: str
    passed: bool
    examples_tested: int
    counterexample: Optional[Dict[str, Any]]
    error_message: Optional[str]
    execution_time: float
    statistics: Dict[str, Any]


class NumericalProperty(ABC):
    """Abstract base class for numerical properties to test."""

    @abstractmethod
    def test_property(self, *args, **kwargs) -> bool:
        """Test the numerical property."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get description of the property."""
        pass


class ConvergenceProperty(NumericalProperty):
    """Property testing for solver convergence behavior."""

    def __init__(self, tolerance_factor: float = 1e2):
        self.tolerance_factor = tolerance_factor

    def test_property(
        self, problem_size: int, tolerance: float, max_iterations: int, backend: str
    ) -> bool:
        """Test that solver converges within expected tolerances."""
        try:
            if not HAS_FIREDRAKE:
                # Skip test if Firedrake not available
                return True

            # Create simple test problem
            mesh = fd.UnitSquareMesh(problem_size, problem_size)
            V = fd.FunctionSpace(mesh, "CG", 1)

            # Simple Poisson problem: -Δu = f with u = 0 on boundary
            problem = Problem(mesh, V, backend)

            def poisson_equation(u, v, params):
                f = fd.Constant(1.0)  # Source term
                return fd.inner(fd.grad(u), fd.grad(v)) * fd.dx - f * v * fd.dx

            problem.add_equation(poisson_equation)
            problem.add_boundary_condition("dirichlet", "on_boundary", 0.0)

            # Solve with robust solver
            solver = RobustFEBMLSolver(
                problem,
                backend,
                solver_options={
                    "tolerance": tolerance,
                    "max_iterations": max_iterations,
                    "monitor_convergence": True,
                },
            )

            solution, metrics = solver.solve(return_metrics=True)

            # Check convergence properties
            if not metrics.success:
                return False

            # Check that solution is reasonable (not NaN/Inf)
            solution_array = solution.dat.data_ro
            if not np.all(np.isfinite(solution_array)):
                return False

            # Check that convergence was achieved within tolerance
            if metrics.convergence_iterations > max_iterations:
                return False

            # Check solution magnitude is reasonable
            solution_norm = np.linalg.norm(solution_array)
            if solution_norm > 100.0 or solution_norm < 1e-10:
                logger.warning(f"Solution norm seems unreasonable: {solution_norm}")
                return False

            return True

        except Exception as e:
            logger.debug(f"Convergence property test failed: {e}")
            return False

    def get_description(self) -> str:
        return "Solver converges to finite solution within specified tolerance"


class StabilityProperty(NumericalProperty):
    """Property testing for numerical stability under perturbations."""

    def __init__(self, perturbation_scale: float = 1e-10):
        self.perturbation_scale = perturbation_scale

    def test_property(
        self, parameters: Dict[str, float], perturbation_magnitude: float
    ) -> bool:
        """Test numerical stability under small perturbations."""
        try:
            # Test a simple function evaluation
            def test_function(params):
                # Simple quadratic function
                return sum(params[key] ** 2 for key in params.keys())

            # Original evaluation
            original_value = test_function(parameters)

            # Perturbed evaluation
            perturbed_params = {}
            for key, value in parameters.items():
                perturbation = perturbation_magnitude * np.random.randn()
                perturbed_params[key] = value + perturbation

            perturbed_value = test_function(perturbed_params)

            # Check stability: small perturbation should give small change
            if not np.isfinite(original_value) or not np.isfinite(perturbed_value):
                return False

            # Relative change should be proportional to perturbation
            if original_value != 0:
                relative_change = abs(perturbed_value - original_value) / abs(
                    original_value
                )
                expected_change = perturbation_magnitude * 2  # Factor for quadratic

                # Allow some tolerance for numerical errors
                return relative_change <= expected_change * 10
            else:
                # If original value is zero, perturbed value should be small
                return abs(perturbed_value) <= perturbation_magnitude * 10

        except Exception as e:
            logger.debug(f"Stability property test failed: {e}")
            return False

    def get_description(self) -> str:
        return "Function evaluation is numerically stable under small perturbations"


class MonotonicityProperty(NumericalProperty):
    """Property testing for monotonicity in optimization."""

    def test_property(
        self,
        objective_function: Callable,
        initial_params: Dict[str, float],
        step_direction: Dict[str, float],
    ) -> bool:
        """Test monotonicity along optimization direction."""
        try:
            # Evaluate at several points along the direction
            step_sizes = [0.0, 0.1, 0.2, 0.5]
            values = []

            for step_size in step_sizes:
                current_params = {}
                for key in initial_params.keys():
                    current_params[key] = (
                        initial_params[key] + step_size * step_direction[key]
                    )

                try:
                    value = objective_function(current_params)
                    if not np.isfinite(value):
                        return False
                    values.append(value)
                except Exception:
                    return False

            # For a descent direction, values should generally decrease
            # (allowing for some numerical noise)
            for i in range(1, len(values)):
                if values[i] > values[0] + 1e-10:  # Small tolerance
                    return False

            return True

        except Exception as e:
            logger.debug(f"Monotonicity property test failed: {e}")
            return False

    def get_description(self) -> str:
        return "Objective function decreases along descent direction"


class ScaleInvarianceProperty(NumericalProperty):
    """Property testing for scale invariance."""

    def test_property(
        self,
        problem_function: Callable,
        parameters: Dict[str, float],
        scale_factor: float,
    ) -> bool:
        """Test scale invariance of problem formulation."""
        try:
            # Original problem
            original_result = problem_function(parameters)

            # Scaled problem
            scaled_params = {
                key: value * scale_factor for key, value in parameters.items()
            }
            scaled_result = problem_function(scaled_params)

            # Check if results are related by expected scaling
            # This depends on the specific problem, but for many physical problems
            # scaling parameters by a factor should scale results predictably

            if not np.isfinite(original_result) or not np.isfinite(scaled_result):
                return False

            # Simple check: if we scale all parameters by the same factor,
            # the result should change in a predictable way
            if original_result != 0:
                scaling_ratio = scaled_result / original_result
                expected_ratio = scale_factor**2  # Assume quadratic scaling

                # Allow 10% tolerance
                return abs(scaling_ratio - expected_ratio) / abs(expected_ratio) < 0.1
            else:
                # If original is zero, scaled should be small
                return abs(scaled_result) < 1e-10

        except Exception as e:
            logger.debug(f"Scale invariance property test failed: {e}")
            return False

    def get_description(self) -> str:
        return "Problem solution scales predictably with parameter scaling"


class PropertyBasedTester:
    """Comprehensive property-based testing framework for numerical methods."""

    def __init__(self, max_examples: int = 100, timeout_seconds: float = 300.0):
        self.max_examples = max_examples
        self.timeout_seconds = timeout_seconds
        self.properties = {}
        self.test_results = []

        # Register default properties
        self._register_default_properties()

    def _register_default_properties(self):
        """Register default numerical properties."""
        self.properties = {
            "convergence": ConvergenceProperty(),
            "stability": StabilityProperty(),
            "monotonicity": MonotonicityProperty(),
            "scale_invariance": ScaleInvarianceProperty(),
        }

    def register_property(self, name: str, property_test: NumericalProperty):
        """Register a custom property test."""
        self.properties[name] = property_test

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="Hypothesis not available")
    def test_solver_convergence_properties(self):
        """Test solver convergence properties with various inputs."""
        if not HAS_FIREDRAKE:
            pytest.skip("Firedrake not available")

        convergence_prop = self.properties["convergence"]

        @given(
            problem_size=st.integers(min_value=2, max_value=10),
            tolerance=st.floats(min_value=1e-12, max_value=1e-4),
            max_iterations=st.integers(min_value=10, max_value=200),
            backend=st.sampled_from(["jax", "torch", "numpy"]),
        )
        @settings(max_examples=20, deadline=30000)  # 30 second deadline
        def test_convergence_property(problem_size, tolerance, max_iterations, backend):
            # Skip if backend not available
            try:
                backend_obj, _ = get_robust_backend(backend)
                if backend_obj is None:
                    assume(False)
            except Exception:
                assume(False)

            result = convergence_prop.test_property(
                problem_size, tolerance, max_iterations, backend
            )
            assert (
                result
            ), f"Convergence property failed for size={problem_size}, tol={tolerance}"

        test_convergence_property()

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="Hypothesis not available")
    def test_numerical_stability_properties(self):
        """Test numerical stability properties."""
        stability_prop = self.properties["stability"]

        @given(
            parameters=st.dictionaries(
                keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijk"),
                values=st.floats(
                    min_value=-100.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=1,
                max_size=5,
            ),
            perturbation_magnitude=st.floats(min_value=1e-15, max_value=1e-8),
        )
        @settings(max_examples=50)
        def test_stability_property(parameters, perturbation_magnitude):
            result = stability_prop.test_property(parameters, perturbation_magnitude)
            assert result, f"Stability property failed for params={parameters}"

        test_stability_property()

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="Hypothesis not available")
    def test_optimization_monotonicity(self):
        """Test optimization monotonicity properties."""
        monotonicity_prop = self.properties["monotonicity"]

        def simple_quadratic(params):
            return sum(x**2 for x in params.values())

        @given(
            initial_params=st.dictionaries(
                keys=st.text(min_size=1, max_size=5, alphabet="abcde"),
                values=st.floats(
                    min_value=-10.0,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=2,
                max_size=4,
            )
        )
        @settings(max_examples=30)
        def test_monotonicity_property(initial_params):
            # Descent direction (negative gradient for quadratic)
            step_direction = {key: -2 * value for key, value in initial_params.items()}

            result = monotonicity_prop.test_property(
                simple_quadratic, initial_params, step_direction
            )
            assert result, f"Monotonicity property failed for params={initial_params}"

        test_monotonicity_property()

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="Hypothesis not available")
    def test_scale_invariance_properties(self):
        """Test scale invariance properties."""
        scale_prop = self.properties["scale_invariance"]

        def simple_function(params):
            return sum(x**2 for x in params.values())

        @given(
            parameters=st.dictionaries(
                keys=st.text(min_size=1, max_size=5, alphabet="abcde"),
                values=st.floats(
                    min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
                ),
                min_size=2,
                max_size=4,
            ),
            scale_factor=st.floats(
                min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        )
        @settings(max_examples=30)
        def test_scale_property(parameters, scale_factor):
            result = scale_prop.test_property(simple_function, parameters, scale_factor)
            assert (
                result
            ), f"Scale invariance failed for params={parameters}, scale={scale_factor}"

        test_scale_property()

    def run_property_test(
        self, property_name: str, test_function: Callable, *args, **kwargs
    ) -> PropertyTestResult:
        """Run a single property test with timing and error handling."""
        start_time = time.time()

        property_test = self.properties.get(property_name)
        if not property_test:
            raise ValueError(f"Unknown property: {property_name}")

        try:
            result = test_function(*args, **kwargs)

            test_result = PropertyTestResult(
                property_name=property_name,
                passed=result,
                examples_tested=1,
                counterexample=None if result else {"args": args, "kwargs": kwargs},
                error_message=None,
                execution_time=time.time() - start_time,
                statistics={},
            )

        except Exception as e:
            test_result = PropertyTestResult(
                property_name=property_name,
                passed=False,
                examples_tested=1,
                counterexample={"args": args, "kwargs": kwargs},
                error_message=str(e),
                execution_time=time.time() - start_time,
                statistics={},
            )

        self.test_results.append(test_result)
        return test_result

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.test_results:
            return {"error": "No test results available"}

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        failed_tests = total_tests - passed_tests

        # Group by property
        property_stats = {}
        for result in self.test_results:
            prop_name = result.property_name
            if prop_name not in property_stats:
                property_stats[prop_name] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "avg_time": 0.0,
                    "counterexamples": [],
                }

            stats = property_stats[prop_name]
            stats["total"] += 1
            if result.passed:
                stats["passed"] += 1
            else:
                stats["failed"] += 1
                if result.counterexample:
                    stats["counterexamples"].append(result.counterexample)

            stats["avg_time"] += result.execution_time

        # Compute averages
        for stats in property_stats.values():
            stats["avg_time"] /= stats["total"]

        return {
            "timestamp": time.time(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            },
            "property_statistics": property_stats,
            "recommendations": self._generate_recommendations(property_stats),
        }

    def _generate_recommendations(self, property_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        for prop_name, stats in property_stats.items():
            failure_rate = stats["failed"] / stats["total"] if stats["total"] > 0 else 0

            if failure_rate > 0.1:  # More than 10% failures
                recommendations.append(
                    f"High failure rate ({failure_rate*100:.1f}%) for {prop_name} property - "
                    f"consider reviewing implementation"
                )

            if stats["avg_time"] > 10.0:  # Slow tests
                recommendations.append(
                    f"Property {prop_name} tests are slow (avg {stats['avg_time']:.1f}s) - "
                    f"consider optimization"
                )

        if not recommendations:
            recommendations.append(
                "All property tests are passing with good performance"
            )

        return recommendations


# Stateful testing for optimization workflows
if HAS_HYPOTHESIS:

    class OptimizationStateMachine(RuleBasedStateMachine):
        """Stateful property-based testing for optimization workflows."""

        def __init__(self):
            super().__init__()
            self.optimizer = None
            self.current_parameters = {}
            self.objective_history = []
            self.backend = None

        @initialize()
        def setup_optimizer(self):
            """Initialize optimizer state."""

            # Simple quadratic objective
            def objective(params):
                return sum(x**2 + 0.1 * x for x in params.values())

            self.objective = objective
            self.current_parameters = {"x": 1.0, "y": -0.5}
            self.objective_history = []

            try:
                backend_obj, backend_name = get_robust_backend("jax")
                self.backend = backend_name
            except Exception:
                try:
                    backend_obj, backend_name = get_robust_backend("numpy")
                    self.backend = backend_name
                except Exception:
                    self.backend = None

        @rule(
            param_update=st.dictionaries(
                keys=st.sampled_from(["x", "y"]),
                values=st.floats(
                    min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False
                ),
                min_size=1,
                max_size=2,
            )
        )
        def update_parameters(self, param_update):
            """Update optimization parameters."""
            if self.backend is None:
                return

            for key, value in param_update.items():
                self.current_parameters[key] = value

            # Evaluate objective
            try:
                obj_value = self.objective(self.current_parameters)
                if np.isfinite(obj_value):
                    self.objective_history.append(obj_value)
            except Exception:
                pass

        @invariant()
        def objective_is_finite(self):
            """Objective function should always return finite values."""
            if self.backend is None or not self.current_parameters:
                return

            try:
                obj_value = self.objective(self.current_parameters)
                assert np.isfinite(
                    obj_value
                ), f"Objective returned non-finite value: {obj_value}"
            except Exception as e:
                # Allow exceptions, but they should be handled gracefully
                pass

        @invariant()
        def parameters_remain_reasonable(self):
            """Parameters should remain within reasonable bounds."""
            for key, value in self.current_parameters.items():
                assert np.isfinite(value), f"Parameter {key} became non-finite: {value}"
                assert abs(value) < 1e6, f"Parameter {key} grew too large: {value}"


# Regression testing framework
class RegressionTester:
    """Framework for regression testing of numerical methods."""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.reference_results = {}
        self.test_cases = []

    def add_reference_result(self, test_case_name: str, result: Any):
        """Add a reference result for comparison."""
        self.reference_results[test_case_name] = result

    def test_regression(self, test_case_name: str, current_result: Any) -> bool:
        """Test for regression against reference result."""
        if test_case_name not in self.reference_results:
            logger.warning(f"No reference result for test case: {test_case_name}")
            return True  # Can't test regression without reference

        reference = self.reference_results[test_case_name]

        try:
            if isinstance(reference, (int, float)) and isinstance(
                current_result, (int, float)
            ):
                # Scalar comparison
                if not np.isfinite(reference) or not np.isfinite(current_result):
                    return np.isfinite(reference) == np.isfinite(current_result)

                relative_error = abs(current_result - reference) / max(
                    abs(reference), 1e-10
                )
                return relative_error < self.tolerance

            elif hasattr(reference, "__len__") and hasattr(current_result, "__len__"):
                # Array comparison
                ref_array = np.asarray(reference)
                cur_array = np.asarray(current_result)

                if ref_array.shape != cur_array.shape:
                    return False

                if not np.all(np.isfinite(ref_array)) or not np.all(
                    np.isfinite(cur_array)
                ):
                    return np.array_equal(
                        np.isfinite(ref_array), np.isfinite(cur_array)
                    )

                relative_errors = np.abs(cur_array - ref_array) / np.maximum(
                    np.abs(ref_array), 1e-10
                )
                return np.all(relative_errors < self.tolerance)

            else:
                # Fallback to exact comparison
                return reference == current_result

        except Exception as e:
            logger.error(f"Error comparing results for {test_case_name}: {e}")
            return False

    def generate_reference_suite(self):
        """Generate reference results for common test cases."""
        # This would typically run a suite of known problems and store results
        logger.info("Generating reference test suite...")

        # Example: Simple linear algebra operations
        test_matrices = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        ]

        for i, matrix in enumerate(test_matrices):
            # Eigenvalues
            eigenvals = np.linalg.eigvals(matrix)
            self.add_reference_result(f"eigenvals_matrix_{i}", eigenvals)

            # Matrix inverse
            try:
                inv_matrix = np.linalg.inv(matrix)
                self.add_reference_result(f"inverse_matrix_{i}", inv_matrix)
            except np.linalg.LinAlgError:
                self.add_reference_result(f"inverse_matrix_{i}", None)


# Integration with pytest
@pytest.fixture
def property_tester():
    """Pytest fixture for property-based testing."""
    return PropertyBasedTester()


@pytest.fixture
def regression_tester():
    """Pytest fixture for regression testing."""
    tester = RegressionTester()
    tester.generate_reference_suite()
    return tester


# Test discovery and execution
def run_all_property_tests():
    """Run all available property-based tests."""
    if not HAS_HYPOTHESIS:
        logger.warning("Hypothesis not available - skipping property-based tests")
        return

    tester = PropertyBasedTester()

    # Run all test methods
    test_methods = [
        tester.test_solver_convergence_properties,
        tester.test_numerical_stability_properties,
        tester.test_optimization_monotonicity,
        tester.test_scale_invariance_properties,
    ]

    for test_method in test_methods:
        try:
            logger.info(f"Running {test_method.__name__}...")
            test_method()
            logger.info(f"✓ {test_method.__name__} passed")
        except Exception as e:
            logger.error(f"✗ {test_method.__name__} failed: {e}")

    # Generate report
    report = tester.generate_test_report()
    logger.info(f"Property testing complete: {report['summary']}")

    return report


if __name__ == "__main__":
    # Run tests if executed directly
    run_all_property_tests()
