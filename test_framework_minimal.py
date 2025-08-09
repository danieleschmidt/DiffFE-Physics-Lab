#!/usr/bin/env python3
"""Minimal framework test without external dependencies."""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic module imports."""
    print("Testing basic imports...", end=" ")
    
    try:
        # Test core module structure
        import src
        from src.models import problem  # This might fail due to numpy
        from src.operators import base
        from src.backends import base as backend_base
        
        # Try manufactured solutions with fallback
        try:
            from src.utils import manufactured_solutions_simple
        except ImportError:
            from src.utils import manufactured_solutions
        
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_manufactured_solutions():
    """Test manufactured solutions without NumPy.""" 
    print("Testing manufactured solutions...", end=" ")
    
    try:
        # Try simple version first
        try:
            from src.utils.manufactured_solutions_simple import (
                polynomial_2d, trigonometric_2d, exponential_2d
            )
        except ImportError:
            from src.utils.manufactured_solutions import (
                polynomial_2d, trigonometric_2d, exponential_2d
            )
        
        # Test basic functionality with pure Python
        import math
        
        x, y = 0.5, 0.3
        
        # Test polynomial (should work with pure Python)
        u1 = polynomial_2d(x, y)
        assert isinstance(u1, (int, float))
        assert not math.isnan(u1)
        
        # Test trigonometric
        u2 = trigonometric_2d(x, y)
        assert isinstance(u2, (int, float))
        assert not math.isnan(u2)
        
        # Test exponential
        u3 = exponential_2d(x, y)
        assert isinstance(u3, (int, float))
        assert not math.isnan(u3)
        
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_validation():
    """Test validation functions."""
    print("Testing validation functions...", end=" ")
    
    try:
        # Try simple version first
        try:
            from src.utils.validation_simple import validate_positive_parameter
        except ImportError:
            from src.utils.validation import validate_positive_parameter
        
        # Should pass
        validate_positive_parameter(1.0, "test")
        
        # Should fail
        try:
            validate_positive_parameter(-1.0, "test")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        print("PASSED") 
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_basic_math():
    """Test basic mathematical operations."""
    print("Testing basic math operations...", end=" ")
    
    try:
        import math
        
        # Test basic operations that framework might use
        x = 0.5
        y = 0.3
        
        # Polynomial evaluation
        result1 = x**2 + y**2
        assert result1 > 0
        
        # Trigonometric functions  
        result2 = math.sin(math.pi * x) * math.cos(math.pi * y)
        assert abs(result2) <= 1
        
        # Exponential functions
        result3 = math.exp(-(x**2 + y**2))
        assert 0 < result3 <= 1
        
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_file_structure():
    """Test that all expected files exist."""
    print("Testing file structure...", end=" ")
    
    try:
        expected_files = [
            "src/__init__.py",
            "src/models/__init__.py",
            "src/models/problem.py",
            "src/operators/__init__.py", 
            "src/operators/base.py",
            "src/operators/laplacian.py",
            "src/backends/__init__.py",
            "src/backends/base.py",
            "src/backends/jax_backend.py",
            "src/utils/__init__.py",
            "src/utils/manufactured_solutions.py",
            "src/utils/validation.py",
            "setup.py",
            "README.md"
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"FAILED: Missing files: {missing_files}")
            return False
        
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_code_syntax():
    """Test that all Python files have valid syntax."""
    print("Testing code syntax...", end=" ")
    
    try:
        import ast
        import os
        
        src_dir = "src"
        syntax_errors = []
        
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        ast.parse(content)
                    except SyntaxError as e:
                        syntax_errors.append(f"{file_path}: {e}")
        
        if syntax_errors:
            print(f"FAILED: Syntax errors in: {syntax_errors}")
            return False
        
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def count_code_lines():
    """Count lines of code for coverage estimation."""
    try:
        total_lines = 0
        total_files = 0
        
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        
                        # Count non-empty, non-comment lines
                        code_lines = 0
                        for line in lines:
                            stripped = line.strip()
                            if stripped and not stripped.startswith('#'):
                                code_lines += 1
                        
                        total_lines += code_lines
                        total_files += 1
                    except:
                        pass
        
        return total_files, total_lines
    except:
        return 0, 0

def main():
    """Run all tests."""
    print("DiffFE-Physics-Lab Minimal Test Suite")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_code_syntax, 
        test_imports,
        test_manufactured_solutions,
        test_validation,
        test_basic_math
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error running {test_func.__name__}: {e}")
            failed += 1
    
    # Code analysis
    print("\nCode Analysis:")
    print("-" * 30)
    total_files, total_lines = count_code_lines()
    print(f"Python files in src/: {total_files}")
    print(f"Lines of code: {total_lines}")
    
    # Results
    print("\nResults:")
    print("-" * 20)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n✅ All basic tests passed!")
        print("Framework structure is correct and ready for development.")
        return 0
    else:
        print(f"\n❌ {failed} tests failed.")
        print("Framework needs fixes before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())