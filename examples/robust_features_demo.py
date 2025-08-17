#!/usr/bin/env python3
"""
Robust Features Demonstration - DiffFE-Physics-Lab
=================================================

This example demonstrates the comprehensive error handling, monitoring,
security, and logging features of the robust implementation layer.
"""

import sys
import os
import time
import threading
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_error_handling():
    """Demonstrate robust error handling."""
    print("=" * 60)
    print("Error Handling and Recovery Demo")
    print("=" * 60)
    
    from robust.error_handling import (
        DiffFEError, ValidationError, ConvergenceError,
        robust_execute, retry_with_backoff, error_context,
        ErrorRecovery, CircuitBreaker, validate_positive
    )
    
    # 1. Custom exceptions
    try:
        raise ValidationError("Invalid mesh size", field="mesh_size", value=-10)
    except ValidationError as e:
        print(f"Caught validation error: {e.message}")
        print(f"Error context: {e.context}")
    
    # 2. Robust execution with retries
    def flaky_function(attempt_count=[0]):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ConnectionError(f"Attempt {attempt_count[0]} failed")
        return f"Success on attempt {attempt_count[0]}"
    
    try:
        result = robust_execute(flaky_function, max_retries=3, backoff_factor=0.1)
        print(f"Robust execution result: {result}")
    except Exception as e:
        print(f"Robust execution failed: {e}")
    
    # 3. Retry decorator
    @retry_with_backoff(max_retries=2, backoff_factor=0.1)
    def retry_demo():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise RuntimeError("Random failure")
        return "Success with retry decorator"
    
    try:
        result = retry_demo()
        print(f"Retry decorator result: {result}")
    except Exception as e:
        print(f"Retry decorator failed: {e}")
    
    # 4. Error context manager
    try:
        with error_context("mesh_generation", mesh_type="structured", size=100):
            # Simulate an error in mesh generation
            raise ValueError("Invalid mesh parameters")
    except DiffFEError as e:
        print(f"Error with context: {e.message}")
        print(f"Operation context: {e.context}")
    
    # 5. Error recovery system
    recovery = ErrorRecovery()
    
    def fallback_solver(*args, **kwargs):
        return {"solution": "fallback_result", "method": "simplified"}
    
    recovery.register_fallback(ValueError, fallback_solver)
    
    def problematic_solver():
        raise ValueError("Primary solver failed")
    
    try:
        result = recovery.execute_with_recovery(problematic_solver)
        print(f"Recovery result: {result}")
        print(f"Recovery stats: {recovery.get_recovery_stats()}")
    except Exception as e:
        print(f"Recovery failed: {e}")
    
    # 6. Circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
    
    def unreliable_service(fail_count=[0]):
        fail_count[0] += 1
        if fail_count[0] <= 3:
            raise ConnectionError("Service unavailable")
        return "Service response"
    
    # Demonstrate circuit breaker behavior
    for i in range(6):
        try:
            result = circuit_breaker.call(unreliable_service)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: {type(e).__name__} - {e}")
        
        if i == 2:  # Wait for recovery timeout
            time.sleep(1.1)
    
    print(f"Circuit breaker state: {circuit_breaker.get_state()}")
    
    # 7. Input validation
    try:
        validate_positive(-5, "conductivity")
    except ValidationError as e:
        print(f"Validation failed: {e.message}")
    
    return True

def demo_monitoring():
    """Demonstrate performance monitoring."""
    print("\n" + "=" * 60)
    print("Performance Monitoring Demo")
    print("=" * 60)
    
    from robust.monitoring import (
        PerformanceMonitor, MetricsCollector, HealthChecker,
        resource_monitor, log_performance
    )
    
    # 1. Performance monitoring
    monitor = PerformanceMonitor()
    
    with monitor.measure_operation("simulation", mesh_size=100, solver="direct"):
        # Simulate computational work
        time.sleep(0.1)
        data = [i**2 for i in range(1000)]  # Some computation
    
    with monitor.measure_operation("optimization", algorithm="gradient_descent"):
        time.sleep(0.05)
        result = sum(range(500))
    
    # Get performance summaries
    sim_summary = monitor.get_operation_summary("simulation")
    opt_summary = monitor.get_operation_summary("optimization")
    
    print(f"Simulation performance: {sim_summary}")
    print(f"Optimization performance: {opt_summary}")
    
    # 2. Resource monitoring context manager
    with resource_monitor("matrix_assembly", matrix_size="100x100"):
        # Simulate matrix operations
        matrix = [[i*j for j in range(50)] for i in range(50)]
        determinant = len(matrix)  # Fake calculation
    
    # 3. Performance logging decorator
    @log_performance("data_processing")
    def process_data(data_size):
        # Simulate data processing
        time.sleep(0.02)
        return [x * 2 for x in range(data_size)]
    
    result = process_data(100)
    print(f"Processed {len(result)} data points")
    
    # 4. Metrics collection
    collector = MetricsCollector(collection_interval=0.1)
    collector.start_collection()
    
    # Let it collect for a short time
    time.sleep(0.3)
    
    current_metrics = collector.get_current_metrics()
    print(f"Current system metrics: {current_metrics}")
    
    collector.stop_collection()
    
    # Get metric summaries
    cpu_summary = collector.get_metric_summary("system_cpu_percent")
    memory_summary = collector.get_metric_summary("process_memory_mb")
    
    print(f"CPU usage summary: {cpu_summary}")
    print(f"Memory usage summary: {memory_summary}")
    
    # 5. Health checking
    health_checker = HealthChecker()
    
    # Add custom health check
    def check_solver_availability():
        # Simulate checking if solver is available
        return True
    
    health_checker.register_check("solver_availability", check_solver_availability,
                                 "Check if numerical solver is available")
    
    # Run health checks
    health_results = health_checker.run_health_checks()
    
    print(f"Health check results:")
    print(f"  Overall healthy: {health_results['overall_healthy']}")
    print(f"  Checks passed: {len([c for c in health_results['checks'].values() if c.get('healthy')])}")
    print(f"  Alerts: {len(health_results['alerts'])}")
    
    return True

def demo_security():
    """Demonstrate security features."""
    print("\n" + "=" * 60)
    print("Security and Validation Demo")
    print("=" * 60)
    
    from robust.security import (
        SecurityValidator, InputSanitizer, PermissionChecker,
        SecurityError, require_permission, validate_inputs,
        secure_mode, SecurityContext
    )
    
    # 1. Input validation
    validator = SecurityValidator()
    
    # Test safe input
    try:
        validator.validate_input("conductivity = 1.5", "parameter")
        print("✓ Safe input validated successfully")
    except SecurityError as e:
        print(f"✗ Input validation failed: {e}")
    
    # Test malicious input
    try:
        validator.validate_input("'; DROP TABLE users; --", "sql_input")
        print("✗ Malicious input passed validation")
    except SecurityError as e:
        print(f"✓ Blocked malicious input: {e}")
    
    # Test file path validation
    try:
        validator.validate_file_path("../../../etc/passwd")
        print("✗ Path traversal passed validation")
    except SecurityError as e:
        print(f"✓ Blocked path traversal: {e}")
    
    # 2. Input sanitization
    sanitizer = InputSanitizer()
    
    unsafe_string = "<script>alert('xss')</script>Hello & World"
    safe_string = sanitizer.sanitize_string(unsafe_string)
    print(f"Sanitized string: {safe_string}")
    
    unsafe_filename = "../../malicious<>file.exe"
    safe_filename = sanitizer.sanitize_filename(unsafe_filename)
    print(f"Sanitized filename: {safe_filename}")
    
    # 3. Permission system
    perm_checker = PermissionChecker()
    
    # Create a user session
    session_id = perm_checker.create_session("user123", ["user"])
    print(f"Created session: {session_id}")
    
    # Check permissions
    can_read = perm_checker.check_permission(session_id, "read")
    can_admin = perm_checker.check_permission(session_id, "admin")
    
    print(f"User can read: {can_read}")
    print(f"User can admin: {can_admin}")
    
    # 4. Permission decorator
    @require_permission("execute")
    def sensitive_operation(data):
        return f"Processed sensitive data: {data}"
    
    # Create security context
    security_context = SecurityContext(
        user_id="user123",
        permissions=["read", "write", "execute"],
        session_id=session_id
    )
    
    try:
        result = sensitive_operation("test_data", _security_context=security_context)
        print(f"Sensitive operation result: {result}")
    except SecurityError as e:
        print(f"Sensitive operation blocked: {e}")
    
    # 5. Input validation decorator
    @validate_inputs(
        mesh_size=lambda x: isinstance(x, int) and x > 0,
        conductivity=lambda x: isinstance(x, (int, float)) and x > 0
    )
    def create_simulation(mesh_size, conductivity):
        return f"Simulation created: mesh={mesh_size}, k={conductivity}"
    
    try:
        result = create_simulation(100, 1.5)
        print(f"Simulation created: {result}")
    except SecurityError as e:
        print(f"Simulation creation blocked: {e}")
    
    try:
        create_simulation(-10, 1.5)  # Invalid mesh size
    except SecurityError as e:
        print(f"Invalid parameters blocked: {e}")
    
    # 6. Secure mode context manager
    with secure_mode(security_context):
        print("Operating in secure mode")
        # Perform secure operations here
        secure_result = "secure_computation_result"
    
    print(f"Secure operation completed: {secure_result}")
    
    return True

def demo_logging():
    """Demonstrate logging system."""
    print("\n" + "=" * 60)
    print("Advanced Logging Demo")
    print("=" * 60)
    
    from robust.logging_system import (
        configure_logging, get_logger, AuditLogger,
        PerformanceLogger, LogAnalyzer, log_performance
    )
    
    # 1. Configure structured logging
    configure_logging(level="INFO", enable_structured_logging=True)
    
    # 2. Basic logging
    logger = get_logger("demo")
    logger.info("Starting logging demonstration")
    logger.warning("This is a warning message")
    logger.error("This is an error message", 
                extra={'extra_data': {'error_code': 'DEMO_001', 'component': 'logging'}})
    
    # 3. Audit logging
    audit_logger = AuditLogger()
    
    audit_logger.log_access("user123", "simulation_data", "read", success=True,
                           ip_address="192.168.1.100", session_id="sess_456")
    
    audit_logger.log_security_event("authentication", "medium", 
                                   "Multiple failed login attempts",
                                   user_id="user456", attempts=3)
    
    audit_logger.log_data_operation("create", "mesh", record_count=1,
                                   mesh_size=100, elements=5000)
    
    # 4. Performance logging
    perf_logger = PerformanceLogger()
    
    perf_logger.log_operation("matrix_solve", duration=2.5, memory_delta_mb=15.2,
                             matrix_size=1000, solver="direct", iterations=1)
    
    # 5. Performance logging decorator
    @log_performance("data_analysis")
    def analyze_data(dataset_size):
        time.sleep(0.05)  # Simulate analysis
        return {"mean": 42.0, "std": 5.7, "samples": dataset_size}
    
    analysis_result = analyze_data(1000)
    print(f"Analysis result: {analysis_result}")
    
    # 6. Log analysis
    log_file = "logs/diffhe.log"
    if os.path.exists(log_file):
        analyzer = LogAnalyzer(log_file)
        analysis = analyzer.analyze_log_file(max_lines=100)
        
        print(f"Log analysis for {analysis.get('file', 'unknown')}:")
        print(f"  Total lines analyzed: {analysis.get('total_lines', 0)}")
        print(f"  Pattern counts: {analysis.get('pattern_counts', {})}")
        print(f"  Issues found: {len(analysis.get('issues', []))}")
        print(f"  Recommendations: {len(analysis.get('recommendations', []))}")
        
        for rec in analysis.get('recommendations', [])[:3]:
            print(f"    - {rec}")
    else:
        print("Log file not found for analysis")
    
    return True

def demo_integration():
    """Demonstrate integration of all robust features."""
    print("\n" + "=" * 60)
    print("Integrated Robust Features Demo")
    print("=" * 60)
    
    from robust.error_handling import error_context, ValidationError
    from robust.monitoring import resource_monitor
    from robust.security import SecurityContext, secure_mode
    from robust.logging_system import get_logger
    
    logger = get_logger("integration")
    
    # Simulate a complete computational workflow with all robust features
    security_context = SecurityContext(
        user_id="scientist123",
        permissions=["read", "write", "execute"],
        session_id="sess_789"
    )
    
    with secure_mode(security_context):
        with resource_monitor("integrated_simulation", 
                            workflow="full_simulation", user=security_context.user_id):
            with error_context("simulation_workflow", 
                              user=security_context.user_id, 
                              simulation_type="thermal"):
                
                logger.info("Starting integrated simulation workflow")
                
                # Simulate workflow steps
                steps = [
                    ("mesh_generation", 0.1),
                    ("assembly", 0.15),
                    ("solving", 0.2),
                    ("post_processing", 0.05)
                ]
                
                results = {}
                
                for step_name, duration in steps:
                    logger.info(f"Executing step: {step_name}")
                    
                    with resource_monitor(f"step_{step_name}"):
                        # Simulate step execution
                        time.sleep(duration)
                        
                        # Simulate potential validation
                        if step_name == "mesh_generation":
                            mesh_quality = 0.95
                            if mesh_quality < 0.8:
                                raise ValidationError("Poor mesh quality", 
                                                    field="mesh_quality", 
                                                    value=mesh_quality)
                        
                        results[step_name] = f"{step_name}_completed"
                        logger.info(f"Step {step_name} completed successfully")
                
                logger.info("Integrated simulation workflow completed successfully")
                print(f"Workflow results: {results}")
    
    print("✓ All robust features integrated successfully")
    return True

def run_stress_test():
    """Run stress test to demonstrate robustness under load."""
    print("\n" + "=" * 60)
    print("Robustness Stress Test")
    print("=" * 60)
    
    from robust.monitoring import global_performance_monitor
    from robust.error_handling import robust_execute
    from robust.logging_system import get_logger
    
    logger = get_logger("stress_test")
    
    def stress_operation(operation_id):
        """Simulate a computational operation that might fail."""
        import random
        
        with global_performance_monitor.measure_operation("stress_test", 
                                                         operation_id=operation_id):
            # Random computation time
            time.sleep(random.uniform(0.01, 0.05))
            
            # Random failure
            if random.random() < 0.1:  # 10% failure rate
                raise RuntimeError(f"Random failure in operation {operation_id}")
            
            return f"Result_{operation_id}"
    
    # Run multiple operations concurrently
    import concurrent.futures
    
    num_operations = 20
    successful_operations = 0
    failed_operations = 0
    
    logger.info(f"Starting stress test with {num_operations} operations")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for i in range(num_operations):
            future = executor.submit(
                robust_execute, 
                stress_operation, 
                i, 
                max_retries=2, 
                backoff_factor=0.1
            )
            futures.append(future)
        
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=5.0)
                successful_operations += 1
                logger.debug(f"Operation {i} succeeded: {result}")
            except Exception as e:
                failed_operations += 1
                logger.warning(f"Operation {i} failed: {e}")
    
    # Get performance summary
    stress_summary = global_performance_monitor.get_operation_summary("stress_test")
    
    print(f"Stress test results:")
    print(f"  Total operations: {num_operations}")
    print(f"  Successful: {successful_operations}")
    print(f"  Failed: {failed_operations}")
    print(f"  Success rate: {successful_operations/num_operations*100:.1f}%")
    print(f"  Performance summary: {stress_summary}")
    
    return successful_operations > failed_operations

def main():
    """Main demonstration function."""
    print("DiffFE-Physics-Lab Robust Features Demonstration")
    print("This example shows comprehensive error handling, monitoring, security, and logging\n")
    
    try:
        # Run all demonstrations
        demos = [
            ("Error Handling", demo_error_handling),
            ("Performance Monitoring", demo_monitoring),
            ("Security Features", demo_security),
            ("Advanced Logging", demo_logging),
            ("Integrated Features", demo_integration),
        ]
        
        results = {}
        
        for demo_name, demo_func in demos:
            try:
                results[demo_name] = demo_func()
                print(f"✓ {demo_name} demonstration completed successfully")
            except Exception as e:
                results[demo_name] = False
                print(f"✗ {demo_name} demonstration failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Run stress test
        try:
            stress_result = run_stress_test()
            results["Stress Test"] = stress_result
        except Exception as e:
            results["Stress Test"] = False
            print(f"✗ Stress test failed: {e}")
        
        print("\n" + "=" * 60)
        print("ROBUST FEATURES SUMMARY")
        print("=" * 60)
        
        successful_demos = sum(1 for success in results.values() if success)
        total_demos = len(results)
        
        print(f"Demonstrations completed: {successful_demos}/{total_demos}")
        
        for demo_name, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"  {demo_name}: {status}")
        
        print("\nRobust Features Implemented:")
        print("✓ Comprehensive error handling with custom exceptions")
        print("✓ Automatic retry mechanisms with exponential backoff")
        print("✓ Circuit breaker pattern for fault tolerance")
        print("✓ Real-time performance monitoring and metrics collection")
        print("✓ System health checking and alerting")
        print("✓ Input validation and sanitization")
        print("✓ Role-based permission system")
        print("✓ Structured audit and performance logging")
        print("✓ Security context management")
        print("✓ Resource usage monitoring")
        print("✓ Log analysis and recommendations")
        
        return 0 if successful_demos == total_demos else 1
        
    except Exception as e:
        print(f"Critical error in robust features demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())