#!/usr/bin/env python3
"""Basic integration test for DiffFE-Physics-Lab."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that core modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test backend imports
        from backends.base import ADBackend, get_backend
        print("✓ Backend base imported successfully")
        
        # Test utility imports (without numpy dependencies)  
        from utils.validation import ValidationError
        print("✓ Validation utilities imported successfully")
        
        print("✓ All core imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        # Test operator registry
        from operators.base import register_operator, get_operator, BaseOperator
        
        @register_operator("test_op")
        class TestOperator(BaseOperator):
            def forward_assembly(self, trial, test, params):
                return f"forward: {params}"
            
            def adjoint_assembly(self, grad_output, trial, test, params):
                return f"adjoint: {params}"
        
        # Test retrieval
        retrieved_op = get_operator("test_op")
        assert retrieved_op == TestOperator
        print("✓ Operator registry working")
        
        # Test operator instantiation
        op = TestOperator(backend='jax', param1=1.0)
        result = op(trial='u', test='v', params={'param2': 2.0})
        expected = "forward: {'param1': 1.0, 'param2': 2.0}"
        assert result == expected
        print("✓ Operator functionality working")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False


def test_manufactured_solutions():
    """Test manufactured solution generation."""
    print("\nTesting manufactured solutions...")
    
    try:
        from utils.manufactured_solutions import generate_manufactured_solution, SolutionType
        import math
        
        # Generate trigonometric solution
        mms = generate_manufactured_solution(
            solution_type=SolutionType.TRIGONOMETRIC,
            dimension=2,
            parameters={'frequency': 1.0, 'amplitude': 1.0}
        )
        
        assert 'solution' in mms
        assert 'source' in mms
        assert callable(mms['solution'])
        assert callable(mms['source'])
        print("✓ MMS generation working")
        
        # Test evaluation
        test_point = [0.5, 0.5]
        sol_val = mms['solution'](test_point)
        src_val = mms['source'](test_point)
        
        # Should be finite numbers
        assert not math.isnan(sol_val) and not math.isinf(sol_val)
        assert not math.isnan(src_val) and not math.isinf(src_val)
        print("✓ MMS evaluation working")
        
        return True
        
    except Exception as e:
        print(f"✗ MMS test failed: {e}")
        return False


def test_error_computation():
    """Test error computation utilities."""
    print("\nTesting error computation...")
    
    try:
        from utils.error_computation import compute_convergence_rate
        
        # Test convergence rate computation
        mesh_sizes = [0.1, 0.05, 0.025]
        errors = [1e-2, 2.5e-3, 6.25e-4]  # Should give rate ≈ 2
        
        result = compute_convergence_rate(mesh_sizes, errors)
        
        assert 'rate' in result
        assert 'error_constant' in result
        assert 'r_squared' in result
        
        # Rate should be approximately 2
        rate = result['rate']
        assert 1.8 <= rate <= 2.2, f"Expected rate ~2, got {rate}"
        print(f"✓ Convergence rate computation working (rate={rate:.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error computation test failed: {e}")
        return False


def test_api_structure():
    """Test API structure without Flask."""
    print("\nTesting API structure...")
    
    try:
        from api.error_handlers import APIException, ValidationError, create_error_response
        
        # Test exception creation
        exc = APIException("Test error", status_code=400, details={'field': 'test'})
        assert exc.message == "Test error"
        assert exc.status_code == 400
        assert exc.details['field'] == 'test'
        print("✓ API exceptions working")
        
        # Test error response creation
        response, status = create_error_response("TestError", "Test message", 400)
        assert response['error'] == "TestError"
        assert response['message'] == "Test message"
        assert status == 400
        print("✓ Error response creation working")
        
        return True
        
    except Exception as e:
        print(f"✗ API structure test failed: {e}")
        return False


def main():
    """Run all basic tests."""
    print("DiffFE-Physics-Lab Basic Integration Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_manufactured_solutions,
        test_error_computation,
        test_api_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())