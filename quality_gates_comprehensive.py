#!/usr/bin/env python3
"""Comprehensive Quality Gates and Testing Suite."""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, '/root/repo')

import src
from src.backends import get_backend
from src.models import Problem
from src.utils.robust_error_handling import get_error_recovery_manager


def run_command(cmd: str, description: str = "") -> tuple:
    """Run a shell command and return status and output."""
    print(f"🔧 {description or cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=120,
            cwd='/root/repo'
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        if success:
            print(f"   ✅ PASSED")
        else:
            print(f"   ❌ FAILED (return code: {result.returncode})")
            if output.strip():
                print(f"   Output: {output[:200]}...")
        
        return success, output
        
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT after 120 seconds")
        return False, "Command timed out"
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False, str(e)


def check_code_quality():
    """Check code quality metrics."""
    print("\n📊 Code Quality Checks")
    print("=" * 40)
    
    quality_score = 0
    total_checks = 0
    
    # Check 1: Basic imports work
    total_checks += 1
    try:
        import src
        from src.models import Problem
        from src.backends import get_backend
        print("✅ Core imports: PASSED")
        quality_score += 1
    except Exception as e:
        print(f"❌ Core imports: FAILED - {e}")
    
    # Check 2: Backend functionality
    total_checks += 1
    try:
        backend = get_backend('numpy')
        test_func = lambda x: x**2
        grad_func = backend.grad(test_func)
        result = grad_func(3.0)
        print(f"✅ Backend functionality: PASSED (gradient = {result:.3f})")
        quality_score += 1
    except Exception as e:
        print(f"❌ Backend functionality: FAILED - {e}")
    
    # Check 3: Problem creation
    total_checks += 1
    try:
        problem = Problem(backend='numpy')
        print("✅ Problem creation: PASSED")
        quality_score += 1
    except Exception as e:
        print(f"❌ Problem creation: FAILED - {e}")
    
    # Check 4: Error handling system
    total_checks += 1
    try:
        recovery_manager = get_error_recovery_manager()
        stats = recovery_manager.get_error_statistics()
        print(f"✅ Error handling system: PASSED ({stats['total_errors']} errors handled)")
        quality_score += 1
    except Exception as e:
        print(f"❌ Error handling system: FAILED - {e}")
    
    # Check 5: File structure
    total_checks += 1
    required_files = [
        'src/__init__.py',
        'src/models/problem.py', 
        'src/backends/numpy_backend.py',
        'src/utils/error_computation.py',
        'src/performance/advanced_scaling.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(f'/root/repo/{file_path}').exists():
            missing_files.append(file_path)
    
    if not missing_files:
        print("✅ File structure: PASSED")
        quality_score += 1
    else:
        print(f"❌ File structure: FAILED - Missing: {missing_files}")
    
    print(f"\n📈 Quality Score: {quality_score}/{total_checks} ({quality_score/total_checks*100:.1f}%)")
    return quality_score >= total_checks * 0.8  # 80% threshold


def check_performance():
    """Check performance characteristics."""
    print("\n⚡ Performance Checks")
    print("=" * 40)
    
    perf_score = 0
    total_checks = 0
    
    # Check 1: Import speed
    total_checks += 1
    start_time = time.time()
    try:
        import src.models
        import src.backends  
        import src.performance.advanced_scaling
        import_time = time.time() - start_time
        
        if import_time < 2.0:
            print(f"✅ Import speed: PASSED ({import_time:.3f}s)")
            perf_score += 1
        else:
            print(f"⚠️  Import speed: SLOW ({import_time:.3f}s)")
    except Exception as e:
        print(f"❌ Import speed: FAILED - {e}")
    
    # Check 2: Memory usage
    total_checks += 1
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024**2)
        
        if memory_mb < 200:  # Less than 200MB
            print(f"✅ Memory usage: PASSED ({memory_mb:.1f}MB)")
            perf_score += 1
        else:
            print(f"⚠️  Memory usage: HIGH ({memory_mb:.1f}MB)")
            perf_score += 0.5
    except Exception as e:
        print(f"❌ Memory usage: FAILED - {e}")
    
    # Check 3: Computation speed
    total_checks += 1
    try:
        import numpy as np
        from src.backends import get_backend
        
        backend = get_backend('numpy')
        
        # Test gradient computation speed
        def test_func(x):
            return np.sum(x**2)
        
        grad_func = backend.grad(test_func)
        test_array = np.random.random(1000)
        
        start_time = time.time()
        for _ in range(10):
            result = grad_func(test_array)
        compute_time = time.time() - start_time
        
        if compute_time < 1.0:
            print(f"✅ Computation speed: PASSED ({compute_time:.3f}s for 10 gradients)")
            perf_score += 1
        else:
            print(f"⚠️  Computation speed: SLOW ({compute_time:.3f}s)")
            perf_score += 0.5
    except Exception as e:
        print(f"❌ Computation speed: FAILED - {e}")
    
    print(f"\n⚡ Performance Score: {perf_score}/{total_checks} ({perf_score/total_checks*100:.1f}%)")
    return perf_score >= total_checks * 0.7  # 70% threshold


def check_security():
    """Check security features."""
    print("\n🔒 Security Checks")
    print("=" * 40)
    
    security_score = 0
    total_checks = 0
    
    # Check 1: Input validation
    total_checks += 1
    try:
        from src.security.validator import SecurityValidator
        validator = SecurityValidator()
        
        # Test XSS detection
        malicious_input = "<script>alert('xss')</script>"
        threats = validator.detect_injection_attempts(malicious_input)
        
        if threats:
            print(f"✅ Input validation: PASSED (detected {len(threats)} threats)")
            security_score += 1
        else:
            print("❌ Input validation: FAILED (no threats detected)")
    except Exception as e:
        print(f"❌ Input validation: FAILED - {e}")
    
    # Check 2: Path validation
    total_checks += 1
    try:
        from src.security.validator import SecurityValidator
        validator = SecurityValidator()
        
        # Test path traversal detection
        malicious_path = "../../etc/passwd"
        is_safe = validator.is_safe_path(malicious_path)
        
        if not is_safe:
            print("✅ Path validation: PASSED (blocked traversal)")
            security_score += 1
        else:
            print("❌ Path validation: FAILED (allowed traversal)")
    except Exception as e:
        print(f"❌ Path validation: FAILED - {e}")
    
    # Check 3: Error handling without information leakage
    total_checks += 1
    try:
        from src.utils.robust_error_handling import RobustError, ErrorSeverity
        
        error = RobustError(
            "Test error", 
            severity=ErrorSeverity.LOW,
            context={"sensitive": "data"}
        )
        
        error_str = str(error)
        if "sensitive" not in error_str:
            print("✅ Error handling: PASSED (no information leakage)")
            security_score += 1
        else:
            print("❌ Error handling: FAILED (information leakage)")
    except Exception as e:
        print(f"❌ Error handling: FAILED - {e}")
    
    print(f"\n🔒 Security Score: {security_score}/{total_checks} ({security_score/total_checks*100:.1f}%)")
    return security_score >= total_checks * 0.8  # 80% threshold


def check_examples():
    """Check that examples run successfully."""
    print("\n🧪 Example Checks")
    print("=" * 40)
    
    examples = [
        ('examples/basic_working_example.py', 'Basic working example'),
        ('examples/generation_2_robustness_demo.py', 'Generation 2 robustness demo'),
        ('examples/generation_3_scaling_demo.py', 'Generation 3 scaling demo')
    ]
    
    passed_examples = 0
    
    for example_file, description in examples:
        if Path(f'/root/repo/{example_file}').exists():
            success, output = run_command(f'python3 {example_file}', description)
            if success:
                passed_examples += 1
        else:
            print(f"❌ {description}: FILE NOT FOUND")
    
    print(f"\n🧪 Examples Score: {passed_examples}/{len(examples)} ({passed_examples/len(examples)*100:.1f}%)")
    return passed_examples >= len(examples) * 0.8  # 80% threshold


def main():
    """Run comprehensive quality gates."""
    print("✅ DiffFE-Physics-Lab - Comprehensive Quality Gates")
    print("=" * 60)
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    
    start_time = time.time()
    
    # Run all quality checks
    quality_passed = check_code_quality()
    performance_passed = check_performance()
    security_passed = check_security() 
    examples_passed = check_examples()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Final assessment
    print(f"\n🏁 FINAL QUALITY ASSESSMENT")
    print("=" * 60)
    
    gates = [
        ("Code Quality", quality_passed),
        ("Performance", performance_passed),
        ("Security", security_passed),
        ("Examples", examples_passed)
    ]
    
    passed_gates = sum(1 for _, passed in gates if passed)
    total_gates = len(gates)
    
    for gate_name, passed in gates:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {gate_name}: {status}")
    
    overall_pass = passed_gates >= total_gates * 0.75  # Need 75% to pass
    
    print(f"\n📊 Overall Results:")
    print(f"   Gates passed: {passed_gates}/{total_gates}")
    print(f"   Success rate: {passed_gates/total_gates*100:.1f}%")
    print(f"   Total time: {total_time:.2f} seconds")
    
    if overall_pass:
        print(f"\n🎉 QUALITY GATES: ✅ PASSED")
        print("   System is ready for production deployment!")
    else:
        print(f"\n💥 QUALITY GATES: ❌ FAILED") 
        print("   System requires improvement before deployment.")
    
    # Additional system information
    print(f"\n📋 System Information:")
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   CPU cores: {cpu_count}")
        print(f"   Memory: {memory_gb:.1f}GB")
    except:
        pass
    
    print(f"   Available backends: numpy" + (", jax" if False else "") + (", torch" if False else ""))
    
    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)