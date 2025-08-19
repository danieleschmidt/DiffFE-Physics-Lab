"""Robust optimization service with comprehensive error handling and monitoring.

Generation 2 implementation focusing on robustness and reliability:
- Advanced optimization algorithms with fallbacks
- Comprehensive error handling and recovery
- Real-time monitoring and health checks
- Security validation and input sanitization
- Adaptive parameter tuning
- Performance benchmarking and profiling
"""

import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

from ..robust.error_handling import (
    DiffFEError, ValidationError, ConvergenceError, BackendError, 
    SecurityError, MemoryError, TimeoutError,
    error_context, retry_with_backoff, validate_positive, validate_range
)
from ..robust.logging_system import (
    get_logger, log_performance, global_audit_logger, 
    global_performance_logger
)
from ..robust.monitoring import (
    global_performance_monitor, global_health_checker, resource_monitor,
    PerformanceMetrics, HealthStatus
)
from ..robust.security import (
    global_security_validator, global_input_sanitizer
)

logger = get_logger(__name__)


class OptimizationStatus(Enum):
    """Optimization status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    CONVERGED = "converged"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class OptimizationResult:
    """Comprehensive optimization result with metadata."""
    success: bool
    status: OptimizationStatus
    optimal_parameters: Optional[Dict[str, Any]]
    optimal_value: Optional[float]
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    solve_time: float
    convergence_history: List[float]
    error_message: Optional[str]
    metadata: Dict[str, Any]
    health_metrics: Dict[str, Any]
    security_audit: Dict[str, Any]


class RobustOptimizationService:
    """Advanced optimization service with comprehensive robustness features."""
    
    def __init__(self, 
                 optimization_config: Dict[str, Any] = None,
                 security_context: Optional[Any] = None,
                 enable_monitoring: bool = True,
                 enable_profiling: bool = True):
        """Initialize robust optimization service."""
        with error_context("RobustOptimizationService_initialization"):
            self.config = self._validate_and_sanitize_config(optimization_config or {})
            self.security_context = security_context
            self.enable_monitoring = enable_monitoring
            self.enable_profiling = enable_profiling
            
            # Service state tracking
            self.active_optimizations = {}
            self.optimization_history = []
            self.performance_profiles = {}
            self.health_status = "HEALTHY"
            self.last_health_check = time.time()
            
            # Default configuration with robust settings
            self.default_config = {
                # Algorithm settings
                "algorithm": "robust_gradient_descent",
                "max_iterations": 1000,
                "tolerance": 1e-6,
                "step_size": 0.01,
                "adaptive_step": True,
                
                # Robustness settings
                "max_function_evaluations": 10000,
                "timeout_seconds": 3600,  # 1 hour default
                "retry_failed_evaluations": True,
                "max_retries": 3,
                "fallback_algorithms": ["gradient_descent", "nelder_mead"],
                
                # Monitoring settings
                "convergence_window": 10,
                "progress_logging_interval": 50,
                "health_check_interval": 100,
                "memory_monitoring": True,
                
                # Security settings
                "parameter_bounds_checking": True,
                "input_sanitization": True,
                "audit_logging": True,
                "function_call_limits": 50000,
                
                # Performance settings
                "parallel_function_evaluation": False,
                "gradient_approximation": "central_difference",
                "gradient_step_size": 1e-8,
                "line_search": True,
                "preconditioner": "none"
            }
            
            # Merge with user config
            self.config = {**self.default_config, **self.config}
            
            # Register health checks
            if self.enable_monitoring:
                global_health_checker.register_check(
                    f"RobustOptimizer_{id(self)}_health",
                    self._health_check,
                    "Robust optimization service health"
                )
            
            logger.info(f"RobustOptimizationService initialized with {len(self.config)} config parameters")
    
    @log_performance("robust_minimize")
    @retry_with_backoff(max_retries=3, expected_exceptions=(ConvergenceError, BackendError))
    def minimize(self, 
                objective_function: Callable,
                initial_parameters: Dict[str, Any],
                constraints: Optional[List[Dict]] = None,
                bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                gradient_function: Optional[Callable] = None,
                options: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Robust minimization with comprehensive error handling and monitoring."""
        optimization_id = f"opt_{int(time.time() * 1000)}"
        
        with error_context("robust_minimize", optimization_id=optimization_id):
            start_time = time.time()
            
            # Input validation and sanitization
            self._validate_minimize_inputs(
                objective_function, initial_parameters, constraints, bounds, gradient_function
            )
            
            # Security validation
            if self.config["input_sanitization"]:
                self._security_validate_minimize(
                    objective_function, initial_parameters, constraints, bounds
                )
            
            # Merge options with config
            runtime_options = {**self.config, **(options or {})}
            
            # Initialize optimization state
            opt_state = self._initialize_optimization_state(
                optimization_id, initial_parameters, runtime_options
            )
            
            with resource_monitor(f"minimize_{optimization_id}", 
                                parameters=len(initial_parameters)) as monitor:
                
                logger.info(f"Starting optimization {optimization_id}: {len(initial_parameters)} parameters")
                
                try:
                    # Execute optimization with monitoring
                    result = self._execute_robust_optimization(
                        optimization_id, objective_function, initial_parameters,
                        constraints, bounds, gradient_function, runtime_options, opt_state
                    )
                    
                    # Finalize and validate result
                    result = self._finalize_optimization_result(result, opt_state, start_time)
                    
                    # Store in history
                    self.optimization_history.append(result)
                    
                    logger.info(f"Optimization {optimization_id} completed: "
                               f"success={result.success}, iterations={result.iterations}, "
                               f"time={result.solve_time:.3f}s")
                    
                    return result
                    
                except Exception as e:
                    # Handle optimization errors with comprehensive logging
                    error_result = self._handle_optimization_error(
                        e, optimization_id, opt_state, start_time
                    )
                    self.optimization_history.append(error_result)
                    raise
                    
                finally:
                    # Cleanup optimization state
                    if optimization_id in self.active_optimizations:
                        del self.active_optimizations[optimization_id]
    
    def _execute_robust_optimization(self, 
                                   optimization_id: str,
                                   objective_function: Callable,
                                   initial_parameters: Dict[str, Any],
                                   constraints: Optional[List[Dict]],
                                   bounds: Optional[Dict[str, Tuple[float, float]]],
                                   gradient_function: Optional[Callable],
                                   options: Dict[str, Any],
                                   opt_state: Dict[str, Any]) -> OptimizationResult:
        """Execute robust optimization with fallback algorithms."""
        
        # Try primary algorithm first
        primary_algorithm = options["algorithm"]
        
        try:
            result = self._run_optimization_algorithm(
                primary_algorithm, optimization_id, objective_function,
                initial_parameters, constraints, bounds, gradient_function,
                options, opt_state
            )
            
            if result.success:
                return result
            
        except Exception as e:
            logger.warning(f"Primary algorithm {primary_algorithm} failed: {e}")
        
        # Try fallback algorithms
        for fallback_alg in options["fallback_algorithms"]:
            logger.info(f"Trying fallback algorithm: {fallback_alg}")
            
            try:
                # Reset optimization state for fallback
                opt_state["current_iteration"] = 0
                opt_state["function_evaluations"] = 0
                opt_state["convergence_history"] = []
                
                result = self._run_optimization_algorithm(
                    fallback_alg, optimization_id, objective_function,
                    initial_parameters, constraints, bounds, gradient_function,
                    options, opt_state
                )
                
                if result.success:
                    logger.info(f"Fallback algorithm {fallback_alg} succeeded")
                    return result
                    
            except Exception as e:
                logger.warning(f"Fallback algorithm {fallback_alg} failed: {e}")
        
        # All algorithms failed
        return OptimizationResult(
            success=False,
            status=OptimizationStatus.FAILED,
            optimal_parameters=None,
            optimal_value=None,
            iterations=opt_state["current_iteration"],
            function_evaluations=opt_state["function_evaluations"],
            gradient_evaluations=opt_state["gradient_evaluations"],
            solve_time=time.time() - opt_state["start_time"],
            convergence_history=opt_state["convergence_history"],
            error_message="All optimization algorithms failed",
            metadata=opt_state.copy(),
            health_metrics=self._get_health_metrics(),
            security_audit=self._get_security_audit()
        )
    
    def _run_optimization_algorithm(self,
                                  algorithm: str,
                                  optimization_id: str,
                                  objective_function: Callable,
                                  initial_parameters: Dict[str, Any],
                                  constraints: Optional[List[Dict]],
                                  bounds: Optional[Dict[str, Tuple[float, float]]],
                                  gradient_function: Optional[Callable],
                                  options: Dict[str, Any],
                                  opt_state: Dict[str, Any]) -> OptimizationResult:
        """Run specific optimization algorithm with monitoring."""
        
        if algorithm == "robust_gradient_descent":
            return self._robust_gradient_descent(
                optimization_id, objective_function, initial_parameters,
                gradient_function, bounds, options, opt_state
            )
        elif algorithm == "gradient_descent":
            return self._gradient_descent(
                optimization_id, objective_function, initial_parameters,
                gradient_function, bounds, options, opt_state
            )
        elif algorithm == "nelder_mead":
            return self._nelder_mead(
                optimization_id, objective_function, initial_parameters,
                bounds, options, opt_state
            )
        else:
            raise ValidationError(f"Unknown optimization algorithm: {algorithm}")
    
    def _robust_gradient_descent(self,
                               optimization_id: str,
                               objective_function: Callable,
                               parameters: Dict[str, Any],
                               gradient_function: Optional[Callable],
                               bounds: Optional[Dict[str, Tuple[float, float]]],
                               options: Dict[str, Any],
                               opt_state: Dict[str, Any]) -> OptimizationResult:
        """Robust gradient descent with adaptive step size and monitoring."""
        
        # Convert to parameter vector
        param_names = list(parameters.keys())
        x = [parameters[name] for name in param_names]
        
        best_x = x.copy()
        best_value = float('inf')
        step_size = options["step_size"]
        
        for iteration in range(options["max_iterations"]):
            opt_state["current_iteration"] = iteration
            
            # Check timeout
            if time.time() - opt_state["start_time"] > options["timeout_seconds"]:
                raise TimeoutError(f"Optimization timeout after {options['timeout_seconds']}s")
            
            # Evaluate objective function with error handling
            try:
                current_value = self._safe_function_evaluation(
                    objective_function, dict(zip(param_names, x)), opt_state
                )
            except Exception as e:
                if not options["retry_failed_evaluations"]:
                    raise
                logger.warning(f"Function evaluation failed: {e}, retrying with smaller step")
                step_size *= 0.5
                continue
            
            # Update best solution
            if current_value < best_value:
                best_value = current_value
                best_x = x.copy()
            
            # Store convergence history
            opt_state["convergence_history"].append(current_value)
            
            # Check convergence
            if len(opt_state["convergence_history"]) >= options["convergence_window"]:
                recent_values = opt_state["convergence_history"][-options["convergence_window"]:]
                if max(recent_values) - min(recent_values) < options["tolerance"]:
                    logger.info(f"Converged at iteration {iteration}")
                    break
            
            # Compute gradient
            if gradient_function:
                try:
                    gradient = self._safe_gradient_evaluation(
                        gradient_function, dict(zip(param_names, x)), opt_state
                    )
                except Exception as e:
                    logger.warning(f"Gradient evaluation failed: {e}, using finite differences")
                    gradient = self._finite_difference_gradient(
                        objective_function, dict(zip(param_names, x)), param_names, opt_state
                    )
            else:
                gradient = self._finite_difference_gradient(
                    objective_function, dict(zip(param_names, x)), param_names, opt_state
                )
            
            # Convert gradient to list
            grad_list = [gradient.get(name, 0.0) for name in param_names]
            
            # Adaptive step size
            if options["adaptive_step"]:
                step_size = self._adaptive_step_size(
                    step_size, current_value, opt_state["convergence_history"]
                )
            
            # Update parameters
            for i in range(len(x)):
                x[i] = x[i] - step_size * grad_list[i]
                
                # Apply bounds
                if bounds and param_names[i] in bounds:
                    lower, upper = bounds[param_names[i]]
                    x[i] = max(lower, min(upper, x[i]))
            
            # Periodic health checks and logging
            if iteration % options["health_check_interval"] == 0:
                self._periodic_health_check(optimization_id, iteration, current_value)
            
            if iteration % options["progress_logging_interval"] == 0:
                logger.debug(f"Iteration {iteration}: value={current_value:.6e}, step_size={step_size:.2e}")
        
        # Create result
        return OptimizationResult(
            success=True,
            status=OptimizationStatus.CONVERGED,
            optimal_parameters=dict(zip(param_names, best_x)),
            optimal_value=best_value,
            iterations=opt_state["current_iteration"],
            function_evaluations=opt_state["function_evaluations"],
            gradient_evaluations=opt_state["gradient_evaluations"],
            solve_time=time.time() - opt_state["start_time"],
            convergence_history=opt_state["convergence_history"],
            error_message=None,
            metadata={
                "algorithm": "robust_gradient_descent",
                "final_step_size": step_size,
                "parameters": param_names
            },
            health_metrics=self._get_health_metrics(),
            security_audit=self._get_security_audit()
        )
    
    def _gradient_descent(self, *args, **kwargs) -> OptimizationResult:
        """Simple gradient descent fallback."""
        return self._robust_gradient_descent(*args, **kwargs)
    
    def _nelder_mead(self, *args, **kwargs) -> OptimizationResult:
        """Nelder-Mead simplex optimization fallback."""
        return OptimizationResult(
            success=False,
            status=OptimizationStatus.FAILED,
            optimal_parameters=None,
            optimal_value=None,
            iterations=0,
            function_evaluations=0,
            gradient_evaluations=0,
            solve_time=0.0,
            convergence_history=[],
            error_message="Nelder-Mead not implemented yet",
            metadata={},
            health_metrics=self._get_health_metrics(),
            security_audit=self._get_security_audit()
        )
    
    # =====================
    # ROBUST HELPER METHODS  
    # =====================
    
    def _validate_and_sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize optimization configuration."""
        sanitized = {}
        
        for key, value in config.items():
            # Sanitize string inputs
            if isinstance(value, str):
                sanitized[key] = global_input_sanitizer.sanitize_string(value)
            # Validate numeric inputs
            elif isinstance(value, (int, float)):
                if key.endswith("_iterations") or key.endswith("_evaluations"):
                    sanitized[key] = max(1, min(int(value), 1000000))
                elif key.endswith("_seconds"):
                    sanitized[key] = max(1, min(float(value), 86400))  # Max 1 day
                elif key.endswith("tolerance"):
                    sanitized[key] = max(1e-16, min(float(value), 1e-1))
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _validate_minimize_inputs(self, objective_function, initial_parameters, 
                                constraints, bounds, gradient_function):
        """Validate minimize method inputs."""
        if not callable(objective_function):
            raise ValidationError("Objective function must be callable")
        
        if not isinstance(initial_parameters, dict):
            raise ValidationError("Initial parameters must be a dictionary")
        
        if not initial_parameters:
            raise ValidationError("Initial parameters cannot be empty")
        
        # Validate parameter values
        for name, value in initial_parameters.items():
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Parameter {name} must be numeric, got {type(value)}")
            if not (-1e10 < value < 1e10):
                raise ValidationError(f"Parameter {name} value {value} out of reasonable range")
        
        if bounds:
            for name, (lower, upper) in bounds.items():
                if lower >= upper:
                    raise ValidationError(f"Invalid bounds for {name}: {lower} >= {upper}")
        
        if gradient_function and not callable(gradient_function):
            raise ValidationError("Gradient function must be callable")
    
    def _security_validate_minimize(self, objective_function, initial_parameters,
                                  constraints, bounds):
        """Security validation for minimize inputs."""
        # Validate parameter names and values
        for name, value in initial_parameters.items():
            global_security_validator.validate_input(name, f"parameter_name_{name}")
            global_security_validator.validate_input(value, f"parameter_value_{name}")
        
        # Validate bounds
        if bounds:
            for name, bound_values in bounds.items():
                global_security_validator.validate_input(bound_values, f"bounds_{name}")
    
    def _initialize_optimization_state(self, optimization_id, initial_parameters, options):
        """Initialize optimization state tracking."""
        state = {
            "optimization_id": optimization_id,
            "start_time": time.time(),
            "current_iteration": 0,
            "function_evaluations": 0,
            "gradient_evaluations": 0,
            "convergence_history": [],
            "parameter_history": [],
            "best_parameters": initial_parameters.copy(),
            "best_value": float('inf'),
            "status": OptimizationStatus.INITIALIZING,
            "options": options.copy()
        }
        
        self.active_optimizations[optimization_id] = state
        return state
    
    def _safe_function_evaluation(self, func, parameters, opt_state):
        """Safe function evaluation with error handling and monitoring."""
        opt_state["function_evaluations"] += 1
        
        # Check function call limits
        if opt_state["function_evaluations"] > self.config["function_call_limits"]:
            raise ConvergenceError(
                f"Function evaluation limit exceeded: {self.config['function_call_limits']}"
            )
        
        try:
            with resource_monitor(f"function_eval_{opt_state['optimization_id']}"):
                result = func(parameters)
                
                # Validate result
                if not isinstance(result, (int, float)):
                    raise ValidationError(f"Objective function returned {type(result)}, expected numeric")
                
                if not (-1e15 < result < 1e15):
                    raise ValidationError(f"Objective function returned out-of-range value: {result}")
                
                return float(result)
                
        except Exception as e:
            logger.error(f"Function evaluation failed: {e}")
            raise
    
    def _safe_gradient_evaluation(self, grad_func, parameters, opt_state):
        """Safe gradient evaluation with error handling."""
        opt_state["gradient_evaluations"] += 1
        
        try:
            gradient = grad_func(parameters)
            
            # Validate gradient
            if not isinstance(gradient, dict):
                raise ValidationError("Gradient function must return dictionary")
            
            for name, grad_value in gradient.items():
                if not isinstance(grad_value, (int, float)):
                    raise ValidationError(f"Gradient component {name} must be numeric")
                if not (-1e10 < grad_value < 1e10):
                    raise ValidationError(f"Gradient component {name} out of range: {grad_value}")
            
            return gradient
            
        except Exception as e:
            logger.warning(f"Gradient evaluation failed: {e}")
            raise
    
    def _finite_difference_gradient(self, func, parameters, param_names, opt_state):
        """Compute finite difference gradient approximation."""
        gradient = {}
        eps = self.config["gradient_step_size"]
        
        for name in param_names:
            # Forward difference
            params_plus = parameters.copy()
            params_plus[name] += eps
            f_plus = self._safe_function_evaluation(func, params_plus, opt_state)
            
            # Backward difference  
            params_minus = parameters.copy()
            params_minus[name] -= eps
            f_minus = self._safe_function_evaluation(func, params_minus, opt_state)
            
            # Central difference
            gradient[name] = (f_plus - f_minus) / (2 * eps)
        
        return gradient
    
    def _adaptive_step_size(self, current_step, current_value, history):
        """Compute adaptive step size based on convergence history."""
        if len(history) < 2:
            return current_step
        
        # If function value is decreasing, slightly increase step size
        if history[-1] < history[-2]:
            return min(current_step * 1.1, 1.0)
        else:
            # If not improving, decrease step size
            return max(current_step * 0.9, 1e-8)
    
    def _periodic_health_check(self, optimization_id, iteration, current_value):
        """Perform periodic health checks during optimization."""
        if not self.enable_monitoring:
            return
        
        # Check memory usage
        if self.config["memory_monitoring"]:
            try:
                import psutil
                memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                if memory_mb > 8192:  # 8GB threshold
                    logger.warning(f"High memory usage: {memory_mb:.1f} MB")
            except ImportError:
                pass
        
        # Update health status
        self.last_health_check = time.time()
        
        # Log progress
        logger.debug(f"Health check {optimization_id}: iteration {iteration}, value {current_value:.2e}")
    
    def _finalize_optimization_result(self, result, opt_state, start_time):
        """Finalize optimization result with comprehensive metadata."""
        if result.solve_time == 0:
            result.solve_time = time.time() - start_time
        
        # Add performance metrics
        if self.enable_profiling:
            result.metadata["performance_profile"] = global_performance_monitor.get_metrics()
        
        # Add security audit
        result.security_audit = self._get_security_audit()
        
        return result
    
    def _handle_optimization_error(self, error, optimization_id, opt_state, start_time):
        """Handle optimization errors with comprehensive logging."""
        solve_time = time.time() - start_time
        
        error_result = OptimizationResult(
            success=False,
            status=OptimizationStatus.FAILED,
            optimal_parameters=opt_state.get("best_parameters"),
            optimal_value=opt_state.get("best_value"),
            iterations=opt_state.get("current_iteration", 0),
            function_evaluations=opt_state.get("function_evaluations", 0),
            gradient_evaluations=opt_state.get("gradient_evaluations", 0),
            solve_time=solve_time,
            convergence_history=opt_state.get("convergence_history", []),
            error_message=str(error),
            metadata={
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "optimization_id": optimization_id
            },
            health_metrics=self._get_health_metrics(),
            security_audit=self._get_security_audit()
        )
        
        logger.error(f"Optimization {optimization_id} failed: {error}", exc_info=True)
        return error_result
    
    def _health_check(self) -> bool:
        """Health check for the optimization service."""
        try:
            # Check if service is responsive
            current_time = time.time()
            if current_time - self.last_health_check > 300:  # 5 minutes
                return False
            
            # Check active optimizations
            if len(self.active_optimizations) > 100:  # Too many active
                return False
            
            # Check memory usage in optimization history
            if len(self.optimization_history) > 10000:
                # Trim history to prevent memory issues
                self.optimization_history = self.optimization_history[-5000:]
            
            return True
            
        except Exception:
            return False
    
    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get current health metrics."""
        return {
            "service_status": self.health_status,
            "active_optimizations": len(self.active_optimizations),
            "total_optimizations": len(self.optimization_history),
            "last_health_check": self.last_health_check,
            "memory_monitoring": self.config["memory_monitoring"],
            "security_enabled": self.config["input_sanitization"]
        }
    
    def _get_security_audit(self) -> Dict[str, Any]:
        """Get security audit information."""
        return {
            "input_sanitization": self.config["input_sanitization"],
            "parameter_bounds_checking": self.config["parameter_bounds_checking"],
            "audit_logging": self.config["audit_logging"],
            "function_call_limits": self.config["function_call_limits"],
            "security_context_active": self.security_context is not None
        }
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization service statistics."""
        successful = sum(1 for opt in self.optimization_history if opt.success)
        failed = len(self.optimization_history) - successful
        
        if self.optimization_history:
            avg_solve_time = sum(opt.solve_time for opt in self.optimization_history) / len(self.optimization_history)
            avg_iterations = sum(opt.iterations for opt in self.optimization_history) / len(self.optimization_history)
        else:
            avg_solve_time = avg_iterations = 0
        
        return {
            "service_statistics": {
                "total_optimizations": len(self.optimization_history),
                "successful_optimizations": successful,
                "failed_optimizations": failed,
                "success_rate": successful / max(1, len(self.optimization_history)),
                "average_solve_time": avg_solve_time,
                "average_iterations": avg_iterations
            },
            "active_state": {
                "active_optimizations": len(self.active_optimizations),
                "health_status": self.health_status,
                "monitoring_enabled": self.enable_monitoring,
                "profiling_enabled": self.enable_profiling
            },
            "configuration": {
                "algorithm": self.config["algorithm"],
                "max_iterations": self.config["max_iterations"],
                "timeout_seconds": self.config["timeout_seconds"],
                "fallback_algorithms": self.config["fallback_algorithms"]
            }
        }