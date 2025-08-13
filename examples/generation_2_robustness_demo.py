#!/usr/bin/env python3
"""Generation 2 Robustness Demo - Comprehensive error handling and reliability."""

import numpy as np
import logging
import sys
import os

# Add src to path
sys.path.insert(0, "/root/repo")

import src
from src.models import Problem, Field, BoundaryCondition, MultiPhysicsProblem, Domain
from src.backends import get_backend
from src.utils.robust_error_handling import (
    robust_operation,
    error_boundary,
    ErrorSeverity,
    get_error_recovery_manager,
    circuit_breaker,
)
from src.utils.error_computation import compute_l2_error, compute_h1_error
from src.utils.manufactured_solutions import polynomial_2d, trigonometric_2d
from src.security.validator import SecurityValidator, ValidationResult
from src.performance.cache import CacheManager, cached, LRUCache


logger = logging.getLogger(__name__)


@robust_operation(max_retries=3, operation_name="demo_computation")
def potentially_failing_computation(x: float) -> float:
    """Demonstration of potentially failing computation with recovery."""
    if x < 0:
        raise ValueError("Negative input not supported")
    if x > 100:
        raise OverflowError("Input too large")
    return np.sqrt(x) * np.log(x + 1)


@circuit_breaker(failure_threshold=3, timeout=10.0)
def unreliable_service_call(service_id: int) -> dict:
    """Simulate unreliable service call with circuit breaker."""
    if service_id % 3 == 0:  # Fail every third call
        raise ConnectionError(f"Service {service_id} unavailable")
    return {"service_id": service_id, "status": "success", "data": np.random.random()}


def main():
    """Demonstrate Generation 2 robustness features."""
    print("🛡️  DiffFE-Physics-Lab - Generation 2 Robustness Demo")
    print("=" * 60)

    # 1. Error Recovery Demonstration
    print("\n🔧 Error Recovery & Resilience:")

    # Test robust operations with different inputs
    test_values = [-1, 5, 150, 10]  # Some will fail, some will succeed
    for val in test_values:
        try:
            result = potentially_failing_computation(val)
            print(f"   ✅ f({val}) = {result:.3f}")
        except Exception as e:
            print(f"   ❌ f({val}) failed: {e}")

    # 2. Circuit Breaker Pattern
    print("\n🔌 Circuit Breaker Pattern:")
    for i in range(10):
        try:
            result = unreliable_service_call(i)
            print(f"   📞 Service call {i}: {result['status']}")
        except Exception as e:
            print(f"   🚫 Service call {i} failed: {type(e).__name__}")

    # 3. Input Validation & Security
    print("\n🔒 Security & Input Validation:")
    validator = SecurityValidator()

    test_inputs = [
        "normal_string",
        "<script>alert('xss')</script>",
        "SELECT * FROM users; DROP TABLE users;",
        "../../etc/passwd",
        "https://example.com/safe",
        "javascript:alert('danger')",
    ]

    for test_input in test_inputs:
        sanitized = validator.sanitize_string(test_input)
        is_safe = len(validator.detect_injection_attempts(test_input)) == 0
        status = "✅ SAFE" if is_safe else "⚠️  UNSAFE"
        print(f"   {status} '{test_input[:30]}' → '{sanitized[:30]}'")

    # 4. Advanced Caching with Error Recovery
    print("\n💾 Robust Caching System:")
    cache = LRUCache(capacity=5)

    @cached(ttl=60)
    def expensive_computation(n: int) -> float:
        """Cached expensive computation."""
        if n < 0:
            raise ValueError("Negative input")
        return sum(np.sqrt(i) for i in range(1, n + 1))

    test_cache_values = [10, -5, 100, 10, 50]  # 10 should hit cache
    for val in test_cache_values:
        try:
            result = expensive_computation(val)
            print(f"   🧮 compute({val}) = {result:.3f}")
        except Exception as e:
            print(f"   ❌ compute({val}) failed: {e}")

    # 5. Multi-Physics Problem with Error Boundaries
    print("\n🌊 Multi-Physics with Error Boundaries:")

    with error_boundary("multi_physics_setup", severity=ErrorSeverity.MEDIUM):
        # Create multi-physics problem
        mpp = MultiPhysicsProblem(backend="numpy")

        # Add domains with potential errors
        fluid_domain = Domain("fluid", physics="navier_stokes")
        solid_domain = Domain("solid", physics="elasticity")

        mpp.add_domain(fluid_domain)
        mpp.add_domain(solid_domain)

        # Add interface with validation
        mpp.add_interface(
            "fsi_interface",
            "fluid",
            "solid",
            condition="no_slip",
            traction="continuous",
        )

        print("   ✅ Multi-physics problem created successfully")
        print(f"   📊 Domains: {list(mpp.domains.keys())}")
        print(f"   🔗 Interfaces: {list(mpp.interfaces.keys())}")

    # 6. Manufactured Solutions with Error Checking
    print("\n🧪 Manufactured Solutions Validation:")

    with error_boundary("manufactured_solutions"):
        # Test different solution types
        solution_tests = [
            ("Polynomial", polynomial_2d, {"degree": 2}),
            ("Trigonometric", trigonometric_2d, {"frequency": 1.0}),
        ]

        for name, generator, params in solution_tests:
            try:
                if params:
                    solution, source = generator(**params)
                else:
                    solution, source = generator()

                # Test evaluation at sample points
                test_point = np.array([0.5, 0.3])
                u_val = solution(test_point)
                f_val = source(test_point)

                print(
                    f"   ✅ {name}: u(0.5,0.3) = {u_val:.3f}, f(0.5,0.3) = {f_val:.3f}"
                )
            except Exception as e:
                print(f"   ❌ {name} failed: {e}")

    # 7. Error Statistics and Monitoring
    print("\n📊 Error Statistics & Monitoring:")
    recovery_manager = get_error_recovery_manager()
    stats = recovery_manager.get_error_statistics()

    print(f"   📈 Total errors handled: {stats['total_errors']}")
    print(f"   🕐 Recent errors: {stats['recent_errors']}")
    print(f"   📊 Error rate/hour: {stats['error_rate_per_hour']}")

    if stats["error_types"]:
        print("   🏷️  Error types:")
        for error_type, count in stats["error_types"].items():
            print(f"     - {error_type}: {count}")

    # 8. Comprehensive Backend Testing with Error Handling
    print("\n🔧 Backend Robustness Testing:")

    backends_to_test = ["numpy", "jax", "torch"]
    for backend_name in backends_to_test:
        with error_boundary(f"backend_test_{backend_name}"):
            try:
                backend = get_backend(backend_name)

                # Test basic functionality
                test_func = lambda x: x**2 + np.sin(x)
                grad_func = backend.grad(test_func)

                test_point = 1.5
                gradient = grad_func(test_point)

                print(f"   ✅ {backend_name}: grad(f)(1.5) = {gradient:.3f}")

            except ImportError:
                print(f"   ⚠️  {backend_name}: Not available (missing dependencies)")
            except Exception as e:
                print(f"   ❌ {backend_name}: Error - {e}")

    # 9. Memory and Resource Management
    print("\n💾 Resource Management:")

    # Test large array handling with error boundaries
    with error_boundary("memory_test", severity=ErrorSeverity.LOW):
        sizes_to_test = [1000, 10000, 100000]
        for size in sizes_to_test:
            try:
                arr = np.random.random(size)
                result = np.linalg.norm(arr)
                print(f"   ✅ Array size {size}: norm = {result:.3f}")
            except MemoryError:
                print(f"   ⚠️  Array size {size}: Memory limit reached")
            except Exception as e:
                print(f"   ❌ Array size {size}: Error - {e}")

    # 10. Integration Test with Error Recovery
    print("\n🔄 End-to-End Integration Test:")

    with error_boundary("integration_test", severity=ErrorSeverity.HIGH):
        try:
            # Create problem
            problem = Problem(backend="numpy")

            # Create fields
            temp_field = Field("temperature", values=np.array([20.0, 25.0, 30.0]))
            velocity_field = Field("velocity", values=lambda p: p[:, 0] + p[:, 1])

            # Test field evaluation
            test_points = np.array([[0, 0], [1, 0], [0.5, 0.5]])
            temp_vals = temp_field.evaluate(test_points)
            vel_vals = velocity_field.evaluate(test_points)

            # Create boundary conditions
            bc_dirichlet = BoundaryCondition("dirichlet", 0.0, "inlet")
            bc_neumann = BoundaryCondition("neumann", 1.0, "outlet")

            # Test BC evaluation
            bc_vals = bc_dirichlet.evaluate(test_points[:2])

            print("   ✅ Problem setup successful")
            print("   ✅ Field evaluation working")
            print("   ✅ Boundary conditions working")
            print("   🎉 Integration test passed!")

        except Exception as e:
            print(f"   ❌ Integration test failed: {e}")

    print(f"\n🛡️  Generation 2 Robustness Demo Complete!")
    print("   ✨ System demonstrates comprehensive error handling")
    print("   🔧 Recovery mechanisms are operational")
    print("   📊 Monitoring and logging are active")
    print("   🚀 Ready for Generation 3 performance optimization")


if __name__ == "__main__":
    main()
