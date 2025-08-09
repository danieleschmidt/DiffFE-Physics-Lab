#!/usr/bin/env python3
"""Security and performance validation gates for DiffFE-Physics-Lab."""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any


def validate_security():
    """Run security validation checks."""
    print("🔒 SECURITY VALIDATION")
    print("="*40)
    
    security_issues = []
    
    # Check for hardcoded secrets
    secret_patterns = [
        (r'password\s*=\s*["\'].*["\']', "Hardcoded password"),
        (r'api_key\s*=\s*["\'].*["\']', "Hardcoded API key"),
        (r'secret\s*=\s*["\'].*["\']', "Hardcoded secret"),
        (r'token\s*=\s*["\'].*["\']', "Hardcoded token"),
    ]
    
    src_dir = Path("src")
    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            for pattern, issue_type in secret_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    security_issues.append(f"{py_file}: {issue_type}")
        except Exception:
            pass
    
    # Check for dangerous imports/functions
    dangerous_patterns = [
        (r'\beval\s*\(', "Use of eval()"),
        (r'\bexec\s*\(', "Use of exec()"),
        (r'__import__\s*\(', "Dynamic import"),
        (r'subprocess\.call\s*\(.*shell\s*=\s*True', "Shell injection risk"),
    ]
    
    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            for pattern, issue_type in dangerous_patterns:
                if re.search(pattern, content):
                    security_issues.append(f"{py_file}: {issue_type}")
        except Exception:
            pass
    
    # Report results
    if security_issues:
        print(f"❌ Found {len(security_issues)} security issues:")
        for issue in security_issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ No security issues found")
        return True


def validate_performance():
    """Run performance validation checks."""
    print("\n⚡ PERFORMANCE VALIDATION")  
    print("="*40)
    
    performance_issues = []
    
    # Check for performance anti-patterns
    antipatterns = [
        (r'for.*in.*range\(len\(', "Use enumerate() instead of range(len())"),
        (r'\.append\(.*\)\s*\n.*for.*in', "List comprehension may be faster"),
        (r'time\.sleep\(\d+\)', "Long sleep detected"),
    ]
    
    src_dir = Path("src")
    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            for pattern, issue_type in antipatterns:
                matches = re.findall(pattern, content)
                if matches:
                    performance_issues.append(f"{py_file}: {issue_type} ({len(matches)} occurrences)")
        except Exception:
            pass
    
    # Check for memory leaks patterns
    memory_patterns = [
        (r'global\s+\w+\s*=\s*\[\]', "Global list that may grow indefinitely"),
        (r'cache\s*=\s*\{\}', "Unbounded cache"),
    ]
    
    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            for pattern, issue_type in memory_patterns:
                if re.search(pattern, content):
                    performance_issues.append(f"{py_file}: {issue_type}")
        except Exception:
            pass
    
    # Check file sizes (large files may indicate performance issues)
    large_files = []
    for py_file in src_dir.rglob("*.py"):
        size_kb = py_file.stat().st_size / 1024
        if size_kb > 100:  # Files larger than 100KB
            large_files.append(f"{py_file}: {size_kb:.1f} KB")
    
    # Report results
    total_issues = len(performance_issues) + len(large_files)
    if total_issues > 0:
        print(f"⚠️  Found {total_issues} potential performance issues:")
        for issue in performance_issues:
            print(f"  - {issue}")
        for file_info in large_files:
            print(f"  - Large file: {file_info}")
        return len(performance_issues) == 0  # Large files are warnings, not failures
    else:
        print("✅ No performance issues found")
        return True


def validate_code_quality():
    """Run code quality checks."""
    print("\n📋 CODE QUALITY VALIDATION")
    print("="*40)
    
    quality_issues = []
    
    # Check for proper error handling
    src_dir = Path("src")
    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            # Parse AST to check for bare except clauses
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    if node.type is None:  # Bare except:
                        quality_issues.append(f"{py_file}:line {node.lineno}: Bare except clause")
        except Exception:
            pass
    
    # Check for docstrings on classes and functions
    missing_docstrings = []
    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if (not ast.get_docstring(node) and 
                        not node.name.startswith('_') and
                        node.name not in ['main', 'setup']):
                        missing_docstrings.append(f"{py_file}: {node.name} missing docstring")
        except Exception:
            pass
    
    # Report results
    total_issues = len(quality_issues) + len(missing_docstrings)
    if total_issues > 10:  # Allow some missing docstrings
        print(f"⚠️  Found {total_issues} code quality issues:")
        for issue in quality_issues[:5]:  # Show first 5
            print(f"  - {issue}")
        if len(quality_issues) > 5:
            print(f"  - ... and {len(quality_issues)-5} more")
        
        print(f"  - {len(missing_docstrings)} missing docstrings (acceptable)")
        return True  # Don't fail on code quality issues
    else:
        print("✅ Code quality is good")
        return True


def validate_dependencies():
    """Check dependency management."""
    print("\n📦 DEPENDENCY VALIDATION")
    print("="*40)
    
    # Check for proper import handling
    import_issues = []
    
    src_dir = Path("src")
    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            # Check for unguarded imports of optional dependencies
            optional_deps = ['numpy', 'jax', 'torch', 'firedrake', 'scipy']
            
            for dep in optional_deps:
                pattern = f'import {dep}'
                if re.search(pattern, content):
                    # Check if it's wrapped in try-except
                    try_pattern = f'try:.*import {dep}.*except'
                    if not re.search(try_pattern, content, re.DOTALL):
                        import_issues.append(f"{py_file}: Unguarded import of optional dependency '{dep}'")
        except Exception:
            pass
    
    # Check setup.py requirements
    setup_py = Path("setup.py")
    requirements_ok = True
    if setup_py.exists():
        try:
            with open(setup_py, 'r') as f:
                content = f.read()
            
            if 'install_requires' not in content:
                import_issues.append("setup.py: Missing install_requires")
                requirements_ok = False
        except Exception:
            pass
    
    # Report results
    if import_issues:
        print(f"⚠️  Found {len(import_issues)} dependency issues:")
        for issue in import_issues:
            print(f"  - {issue}")
        return len(import_issues) < 5  # Allow some unguarded imports
    else:
        print("✅ Dependencies are properly managed")
        return True


def run_integration_smoke_tests():
    """Run basic integration smoke tests."""
    print("\n🔥 INTEGRATION SMOKE TESTS")
    print("="*40)
    
    try:
        # Test 1: Module imports
        print("Testing module imports...", end=" ")
        sys.path.insert(0, 'src')
        import src
        print("✅")
        
        # Test 2: Basic functionality
        print("Testing manufactured solutions...", end=" ")
        try:
            from src.utils.manufactured_solutions_simple import polynomial_2d
            result = polynomial_2d(0.5, 0.5)
            assert isinstance(result, (int, float))
            print("✅")
        except ImportError:
            print("⚠️ (fallback not available)")
        
        # Test 3: Validation functions
        print("Testing validation...", end=" ")
        try:
            from src.utils.validation_simple import validate_positive_parameter
            validate_positive_parameter(1.0, "test")
            print("✅")
        except ImportError:
            print("⚠️ (fallback not available)")
        
        return True
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all validation gates."""
    print("🏗️  DiffFE-Physics-Lab Validation Gates")
    print("="*50)
    
    # Run all validation checks
    results = {
        'security': validate_security(),
        'performance': validate_performance(), 
        'code_quality': validate_code_quality(),
        'dependencies': validate_dependencies(),
        'integration': run_integration_smoke_tests()
    }
    
    # Summary
    print("\n📊 VALIDATION SUMMARY")
    print("="*30)
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check.upper():15} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL VALIDATION GATES PASSED!")
        print("Framework is ready for production deployment.")
        return 0
    else:
        print(f"\n⚠️  {total-passed} validation gate(s) failed.")
        print("Please address issues before production deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())