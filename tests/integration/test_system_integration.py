"""Integration tests for the complete system."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.models.problem import Problem, FEBMLProblem
from src.services.solver import FEBMLSolver
from src.services.optimization import OptimizationService
from src.operators.laplacian import LaplacianOperator
from src.performance import PerformanceOptimizer, PerformanceMonitor
from src.security.monitor import SecurityMonitor
from src.utils.manufactured_solutions import generate_manufactured_solution, SolutionType


class TestBasicIntegration:
    """Basic integration tests for core components."""
    
    @patch('src.models.problem.HAS_FIREDRAKE', True)
    @patch('src.models.problem.fd')
    def test_problem_solver_integration(self, mock_fd):
        """Test Problem and FEBMLSolver integration."""
        # Mock Firedrake components
        mock_mesh = Mock()
        mock_function_space = Mock()
        mock_function = Mock()
        mock_test_function = Mock()
        
        mock_fd.Function.return_value = mock_function
        mock_fd.TestFunction.return_value = mock_test_function
        mock_fd.solve = Mock()
        
        # Create problem
        problem = Problem(mesh=mock_mesh, function_space=mock_function_space)
        
        # Add equation
        def diffusion_eq(u, v, params):
            return u + v  # Simplified for testing
        
        problem.add_equation(diffusion_eq, name="diffusion")
        problem.add_boundary_condition('dirichlet', 1, 0.0)
        problem.set_parameter('diffusion_coeff', 1.0)
        
        # Create and use solver
        solver = FEBMLSolver(problem)
        solution = solver.solve()
        
        assert solution == mock_function
        mock_fd.solve.assert_called_once()
    
    @patch('src.models.problem.HAS_FIREDRAKE', True)
    @patch('src.models.problem.fd')
    def test_problem_optimization_integration(self, mock_fd):
        """Test Problem with OptimizationService integration."""
        # Mock Firedrake components
        mock_mesh = Mock()
        mock_function_space = Mock()
        mock_solution = Mock()
        mock_solution.dat.data = np.array([1.0, 2.0, 3.0])
        
        mock_fd.Function.return_value = mock_solution
        mock_fd.TestFunction.return_value = Mock()
        mock_fd.solve = Mock()
        
        # Create problem
        problem = Problem(mesh=mock_mesh, function_space=mock_function_space)
        problem.add_equation(lambda u, v, p: u + v)
        problem.add_boundary_condition('dirichlet', 1, 0.0)
        
        # Create optimization service
        opt_service = OptimizationService(problem)
        
        # Define objective function
        def objective(params_dict):
            # Set parameters and solve
            for key, value in params_dict.items():
                problem.set_parameter(key, value)
            
            solver = FEBMLSolver(problem)
            solution = solver.solve()
            
            # Simple objective
            return np.sum(solution.dat.data**2)
        
        # Mock scipy optimization
        with patch('scipy.optimize.minimize') as mock_minimize:
            mock_result = Mock()
            mock_result.success = True
            mock_result.x = np.array([2.0])
            mock_result.fun = 14.0
            mock_result.nit = 10
            mock_result.nfev = 30
            mock_minimize.return_value = mock_result
            
            result = opt_service.minimize_vector(
                objective=objective,
                initial_guess=np.array([1.0])
            )
            
            assert result.success is True
            assert result.objective_value == 14.0


class TestPerformanceIntegration:
    """Integration tests for performance components."""
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring with actual computations."""
        optimizer = PerformanceOptimizer()
        monitor = PerformanceMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        
        try:
            # Perform some computations with optimization
            @optimizer.optimize(strategy="cache")
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n-1) + fibonacci(n-2)
            
            # Log performance metrics
            start_time = time.time()
            result = fibonacci(10)
            end_time = time.time()
            
            monitor.log_application_metric('computation_time', (end_time - start_time) * 1000)
            monitor.log_application_metric('result_value', result)
            
            # Check that metrics were recorded
            current_metrics = monitor.get_current_metrics()
            assert current_metrics['monitoring_active'] is True
            
            # Check optimizer performance
            optimizer_metrics = optimizer.get_metrics()
            assert isinstance(optimizer_metrics.execution_time, float)
            
        finally:
            monitor.stop_monitoring()
    
    def test_performance_profiling_integration(self):
        """Test performance profiling with real functions."""
        from src.performance import profile, get_global_profiler
        
        @profile(include_args=True)
        def matrix_multiplication(size):
            """Test function for profiling."""
            a = np.random.random((size, size))
            b = np.random.random((size, size))
            return np.dot(a, b)
        
        # Execute function multiple times
        for size in [10, 20, 30]:
            result = matrix_multiplication(size)
            assert result.shape == (size, size)
        
        # Check profiling results
        profiler = get_global_profiler()
        stats = profiler.get_stats('matrix_multiplication')
        
        assert stats is not None
        assert stats.call_count == 3
        assert stats.avg_time > 0
        
        # Generate report
        report = profiler.generate_report()
        assert 'matrix_multiplication' in report['function_statistics']


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_security_monitoring_integration(self):
        """Test security monitoring with simulated attacks."""
        from src.security.monitor import SecurityMonitor, ThreatLevel
        from src.security.validator import SecurityValidator
        
        monitor = SecurityMonitor()
        validator = SecurityValidator()
        
        # Set up alert tracking
        alerts_received = []
        
        def alert_handler(event):
            alerts_received.append(event)
        
        monitor.add_alert_callback(alert_handler)
        
        # Simulate a series of security events
        attacker_ip = '192.168.1.100'
        
        # 1. Multiple failed login attempts (brute force)
        for i in range(12):
            monitor.record_event(
                event_type='login_failed',
                threat_level=ThreatLevel.MEDIUM,
                message=f'Failed login attempt {i+1}',
                source_ip=attacker_ip,
                user_id=f'user{i%3}'  # Targeting multiple users
            )
        
        # 2. SQL injection attempts
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'; EXEC xp_cmdshell('dir'); --"
        ]
        
        for malicious_input in malicious_inputs:
            validation_result = validator.validate_input(malicious_input, 'username')
            
            if not validation_result.is_valid:
                monitor.record_event(
                    event_type='sql_injection',
                    threat_level=ThreatLevel.HIGH,
                    message='SQL injection attempt detected',
                    source_ip=attacker_ip,
                    details={'input_sample': malicious_input[:20]}
                )
        
        # 3. Rate limiting test
        monitor.add_rate_limit_rule(
            rule_name='api_test',
            max_requests=3,
            time_window=60,
            block_duration=300
        )
        
        # Trigger rate limiting
        for i in range(5):  # More than the limit
            allowed = monitor.is_request_allowed('api_test', attacker_ip)
            if not allowed:
                break
        
        # Verify security events were recorded
        assert len(monitor.events) > 0
        
        # Check for specific attack patterns
        brute_force_events = [e for e in monitor.events if e.event_type == 'brute_force_attack']
        injection_events = [e for e in monitor.events if e.event_type == 'injection_attack']
        
        assert len(brute_force_events) > 0
        assert len(injection_events) > 0
        
        # Check that IP was blocked
        assert attacker_ip in monitor.blocked_ips
        
        # Generate security report
        report = monitor.generate_security_report(days=1)
        assert report['total_events'] > 0
        assert attacker_ip in [ip for ip, _ in report['top_threat_ips']]


class TestManufacturedSolutionIntegration:
    """Integration tests for manufactured solutions with the complete system."""
    
    @patch('src.models.problem.HAS_FIREDRAKE', True)
    @patch('src.models.problem.fd')
    def test_manufactured_solution_workflow(self, mock_fd):
        """Test complete workflow with manufactured solutions."""
        # Generate manufactured solution
        ms = generate_manufactured_solution(
            solution_type=SolutionType.TRIGONOMETRIC,
            dimension=2,
            parameters={'frequency': 1.0, 'amplitude': 1.0}
        )
        
        # Mock Firedrake components
        mock_mesh = Mock()
        mock_function_space = Mock()
        mock_solution = Mock()
        
        mock_fd.Function.return_value = mock_solution
        mock_fd.TestFunction.return_value = Mock()
        mock_fd.solve = Mock()
        
        # Create FEBML problem
        problem = FEBMLProblem(
            mesh=mock_mesh,
            function_space=mock_function_space,
            experiment_name="manufactured_solution_test"
        )
        
        # Add equation using manufactured solution
        def ms_equation(u, v, params):
            # In reality, this would use the manufactured source
            return u + v  # Simplified for testing
        
        problem.add_equation(ms_equation)
        problem.add_boundary_condition('dirichlet', 'on_boundary', ms['solution'])
        
        # Solve problem
        solver = FEBMLSolver(problem)
        solution = solver.solve()
        
        # Log metrics (simulated error computation)
        test_points = [[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]]
        errors = []
        
        for point in test_points:
            exact_value = ms['solution'](point)
            # Simulated computed value
            computed_value = exact_value + np.random.normal(0, 0.01)
            error = abs(exact_value - computed_value)
            errors.append(error)
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        problem.log_metric('avg_l2_error', avg_error)
        problem.log_metric('max_l2_error', max_error)
        problem.log_metric('test_points', len(test_points))
        
        # Checkpoint solution
        problem.checkpoint(solution, 'final_solution')
        
        # Verify experiment tracking
        assert 'avg_l2_error' in problem.metrics
        assert 'final_solution' in problem.checkpoints
        assert len(problem.metrics['avg_l2_error']) == 1


class TestFullSystemIntegration:
    """Full system integration tests combining all components."""
    
    @patch('src.models.problem.HAS_FIREDRAKE', True)
    @patch('src.models.problem.fd')
    def test_complete_system_workflow(self, mock_fd):
        """Test complete system workflow with all components."""
        from src.performance import get_global_profiler, get_global_optimizer
        from src.security.monitor import SecurityMonitor
        
        # Initialize all systems
        profiler = get_global_profiler()
        optimizer = get_global_optimizer()
        security_monitor = SecurityMonitor()
        
        # Mock Firedrake components
        mock_mesh = Mock()
        mock_function_space = Mock()
        mock_solution = Mock()
        mock_solution.dat.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        mock_fd.Function.return_value = mock_solution
        mock_fd.TestFunction.return_value = Mock()
        mock_fd.solve = Mock()
        
        # Step 1: Create and configure problem
        problem = FEBMLProblem(
            mesh=mock_mesh,
            function_space=mock_function_space,
            experiment_name="full_system_test"
        )
        
        # Add equation with performance monitoring
        @profiler.profile()
        @optimizer.optimize(strategy="cache")
        def create_equation(u, v, params):
            """Profiled and optimized equation creation."""
            time.sleep(0.001)  # Simulate computation
            return u + v
        
        problem.add_equation(create_equation)
        problem.add_boundary_condition('dirichlet', 1, 0.0)
        problem.set_parameter('diffusion_coeff', 1.0)
        
        # Step 2: Solve with performance monitoring
        with profiler.profile_context('problem_solving'):
            solver = FEBMLSolver(problem)
            solution = solver.solve()
        
        # Step 3: Parameter optimization with security monitoring
        opt_service = OptimizationService(problem)
        
        @profiler.profile()
        def secure_objective(params_dict):
            """Objective function with security checks."""
            # Simulate security validation
            for key, value in params_dict.items():
                if not isinstance(value, (int, float)) or value < 0:
                    security_monitor.record_event(
                        'parameter_validation_failed',
                        threat_level=security_monitor.ThreatLevel.MEDIUM,
                        message=f'Invalid parameter: {key}={value}'
                    )
                    return float('inf')  # Invalid parameters
            
            # Set parameters and solve
            for key, value in params_dict.items():
                problem.set_parameter(key, value)
            
            solution = solver.solve()
            return np.sum(solution.dat.data**2)
        
        # Mock optimization
        with patch('scipy.optimize.minimize') as mock_minimize:
            mock_result = Mock()
            mock_result.success = True
            mock_result.x = np.array([1.5])
            mock_result.fun = 55.0
            mock_result.nit = 15
            mock_result.nfev = 45
            mock_minimize.return_value = mock_result
            
            opt_result = opt_service.minimize_vector(
                objective=secure_objective,
                initial_guess=np.array([1.0])
            )
        
        # Step 4: Log comprehensive metrics
        problem.log_metric('optimization_success', float(opt_result.success))
        problem.log_metric('final_objective_value', opt_result.objective_value)
        problem.log_metric('optimization_iterations', opt_result.iterations)
        
        # Step 5: Generate comprehensive reports
        
        # Performance report
        performance_report = profiler.generate_report()
        optimizer_report = optimizer.generate_optimization_report()
        
        # Security report
        security_report = security_monitor.generate_security_report(days=1)
        
        # Experiment report
        problem.checkpoint(solution, 'optimized_solution')
        
        # Step 6: Verify all systems worked together
        
        # Check profiling recorded all operations
        assert 'create_equation' in performance_report['function_statistics']
        assert 'problem_solving' in performance_report['function_statistics']
        assert 'secure_objective' in performance_report['function_statistics']
        
        # Check optimization worked
        assert opt_result.success is True
        assert opt_result.objective_value == 55.0
        
        # Check experiment tracking
        assert 'optimization_success' in problem.metrics
        assert 'optimized_solution' in problem.checkpoints
        
        # Check security monitoring (should have no critical events for valid run)
        critical_events = [e for e in security_monitor.events 
                          if e.threat_level == security_monitor.ThreatLevel.CRITICAL]
        assert len(critical_events) == 0  # No critical security issues
        
        # Verify system health
        assert len(profiler.entries) > 0
        assert len(optimizer._task_history) > 0
        assert isinstance(security_report, dict)
        
        # Final system status
        system_status = {
            'performance': {
                'total_profiled_calls': len(profiler.entries),
                'optimization_tasks': len(optimizer._task_history),
                'cache_hit_rate': len(optimizer._optimization_cache)
            },
            'security': {
                'total_events': len(security_monitor.events),
                'blocked_ips': len(security_monitor.blocked_ips),
                'alert_rules': len(security_monitor.alert_rules)
            },
            'experiment': {
                'metrics_tracked': len(problem.metrics),
                'checkpoints_saved': len(problem.checkpoints),
                'equations_defined': len(problem.equations)
            }
        }
        
        # Verify comprehensive system integration
        assert system_status['performance']['total_profiled_calls'] >= 3
        assert system_status['experiment']['metrics_tracked'] >= 3
        assert system_status['experiment']['checkpoints_saved'] >= 1
        
        print(f"Full system integration test completed successfully!")
        print(f"System Status: {system_status}")


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""
    
    def test_cascading_error_handling(self):
        """Test error handling cascading through multiple components."""
        from src.utils.error_handling import ErrorHandler, ErrorLevel
        from src.security.monitor import SecurityMonitor, ThreatLevel
        
        error_handler = ErrorHandler()
        security_monitor = SecurityMonitor()
        
        # Simulate a cascade of errors
        try:
            # Step 1: Invalid parameter causes validation error
            raise ValueError("Invalid parameter: diffusion_coeff must be positive")
        except ValueError as e:
            error_handler.log_error(ErrorLevel.ERROR, str(e), {'component': 'validation'})
            
            # Step 2: Security monitoring detects repeated validation failures
            security_monitor.record_event(
                'validation_failure',
                ThreatLevel.LOW,
                'Repeated validation failures detected',
                source_ip='192.168.1.100'
            )
        
        try:
            # Step 3: Solver fails due to invalid parameters
            raise RuntimeError("Solver failed: Cannot solve with invalid parameters")
        except RuntimeError as e:
            error_handler.log_error(ErrorLevel.CRITICAL, str(e), {'component': 'solver'})
            
            # Step 4: Security escalation due to critical system failure
            security_monitor.record_event(
                'system_failure',
                ThreatLevel.HIGH,
                'Critical solver failure - possible DoS attack',
                source_ip='192.168.1.100'
            )
        
        # Verify error cascade was properly handled
        assert len(error_handler.errors) == 2
        assert error_handler.has_critical_errors() is True
        
        assert len(security_monitor.events) == 2
        high_threat_events = [e for e in security_monitor.events 
                             if e.threat_level == ThreatLevel.HIGH]
        assert len(high_threat_events) == 1
        
        # Generate combined error report
        error_report = error_handler.generate_error_report()
        security_report = security_monitor.generate_security_report(days=1)
        
        combined_report = {
            'error_summary': error_report,
            'security_summary': security_report,
            'correlation': {
                'total_incidents': len(error_handler.errors) + len(security_monitor.events),
                'critical_level': error_handler.has_critical_errors() or 
                               len(high_threat_events) > 0
            }
        }
        
        assert combined_report['correlation']['critical_level'] is True
        assert combined_report['correlation']['total_incidents'] == 4


class TestResourceManagement:
    """Integration tests for resource management across components."""
    
    def test_memory_management_integration(self):
        """Test memory management across all components."""
        from src.performance import CacheManager, PerformanceOptimizer
        
        # Create components with limited resources
        cache = CacheManager(max_size=10)  # Small cache
        optimizer = PerformanceOptimizer()
        
        # Fill up cache
        for i in range(15):  # More than cache size
            cache.set(f'key_{i}', f'value_{i}')
        
        # Cache should have evicted old entries
        assert len(cache) <= 10
        
        # Test optimizer memory management
        optimizer.optimize_memory_usage(target_mb=50.0)  # Small target
        
        # Verify systems are still functional after resource constraints
        @cache.cached
        def test_function(x):
            return x * 2
        
        result = test_function(5)
        assert result == 10
        
        # Check cache stats
        stats = cache.get_stats()
        assert stats['size'] <= 10
        
        # Verify optimizer metrics
        metrics = optimizer.get_metrics()
        assert isinstance(metrics.memory_usage_mb, float)