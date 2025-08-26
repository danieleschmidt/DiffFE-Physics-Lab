"""Advanced Error Recovery System - Generation 2 Enhancement.

This module provides sophisticated error recovery, circuit breakers,
self-healing capabilities, and robust failure handling for the DiffFE system.
"""

import time
import traceback
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
from pathlib import Path


class ErrorSeverity(Enum):
    """Error severity levels for intelligent handling."""
    LOW = "low"           # Recoverable, log and continue
    MEDIUM = "medium"     # Recoverable with retry
    HIGH = "high"         # Requires fallback strategy
    CRITICAL = "critical" # System-level failure


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    SELF_HEALING = "self_healing"


@dataclass
class ErrorContext:
    """Comprehensive error context for intelligent recovery."""
    error_type: str
    severity: ErrorSeverity
    timestamp: float
    operation: str
    parameters: Dict[str, Any]
    stack_trace: str
    system_state: Dict[str, Any]
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


@dataclass 
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    success_threshold: int = 3
    monitoring_window: float = 300.0  # 5 minutes


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.failure_timestamps = []
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker."""
        current_time = time.time()
        
        # Clean old failure timestamps
        self._clean_old_failures(current_time)
        
        if self.state == CircuitBreakerState.OPEN:
            if current_time - self.last_failure_time >= self.config.timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                logging.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(current_time)
            raise e
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logging.info(f"Circuit breaker {self.name} recovered to CLOSED")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)  # Gradually reduce failure count
    
    def _on_failure(self, current_time: float):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = current_time
        self.failure_timestamps.append(current_time)
        self.success_count = 0
        
        if (self.state == CircuitBreakerState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            logging.error(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logging.error(f"Circuit breaker {self.name} failed during recovery test")
    
    def _clean_old_failures(self, current_time: float):
        """Remove failure timestamps outside monitoring window."""
        cutoff_time = current_time - self.config.monitoring_window
        self.failure_timestamps = [t for t in self.failure_timestamps if t > cutoff_time]
        
        # Update failure count based on recent failures
        recent_failures = len(self.failure_timestamps)
        if recent_failures < self.failure_count:
            self.failure_count = recent_failures


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class AdvancedErrorRecoverySystem:
    """Comprehensive error recovery and self-healing system."""
    
    def __init__(self):
        """Initialize error recovery system."""
        self.error_history = []
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self.fallback_handlers = {}
        self.system_health_metrics = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0.0,
            'system_uptime': time.time(),
            'last_critical_error': None
        }
        
        # Initialize default recovery strategies
        self._setup_default_recovery_strategies()
        
        logging.info("🛡️ Advanced Error Recovery System initialized")
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies for common error types."""
        
        # Timeout errors - retry with exponential backoff
        self.register_recovery_strategy(
            "TimeoutError",
            ErrorSeverity.MEDIUM,
            RecoveryStrategy.RETRY,
            max_attempts=3,
            backoff_factor=2.0
        )
        
        # Memory errors - graceful degradation
        self.register_recovery_strategy(
            "MemoryError",
            ErrorSeverity.HIGH,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            degradation_level=0.5
        )
        
        # Connection errors - circuit breaker
        self.register_recovery_strategy(
            "ConnectionError",
            ErrorSeverity.HIGH,
            RecoveryStrategy.CIRCUIT_BREAKER,
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30.0)
        )
        
        # Validation errors - fallback to safe defaults
        self.register_recovery_strategy(
            "ValidationError",
            ErrorSeverity.MEDIUM,
            RecoveryStrategy.FALLBACK,
            fallback_safe=True
        )
    
    def register_recovery_strategy(self, error_type: str, severity: ErrorSeverity,
                                  strategy: RecoveryStrategy, **config):
        """Register a recovery strategy for specific error type."""
        self.recovery_strategies[error_type] = {
            'severity': severity,
            'strategy': strategy,
            'config': config
        }
        
        # Initialize circuit breaker if needed
        if strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            cb_config = config.get('circuit_breaker_config', CircuitBreakerConfig())
            self.circuit_breakers[error_type] = CircuitBreaker(cb_config, error_type)
        
        logging.info(f"Registered recovery strategy for {error_type}: {strategy.value}")
    
    def register_fallback_handler(self, operation: str, handler: Callable):
        """Register fallback handler for specific operation."""
        self.fallback_handlers[operation] = handler
        logging.info(f"Registered fallback handler for operation: {operation}")
    
    async def execute_with_recovery(self, operation: str, func: Callable, 
                                   *args, **kwargs) -> Dict[str, Any]:
        """Execute function with automatic error recovery."""
        start_time = time.time()
        error_context = None
        
        try:
            # Check if operation has circuit breaker
            if operation in self.circuit_breakers:
                cb = self.circuit_breakers[operation]
                if cb.state == CircuitBreakerState.OPEN:
                    return await self._handle_circuit_breaker_open(operation, func, *args, **kwargs)
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - update circuit breaker if applicable
            if operation in self.circuit_breakers:
                self.circuit_breakers[operation]._on_success()
            
            return {
                'success': True,
                'result': result,
                'execution_time': time.time() - start_time,
                'operation': operation
            }
            
        except Exception as e:
            # Create error context
            error_context = ErrorContext(
                error_type=type(e).__name__,
                severity=self._determine_error_severity(e),
                timestamp=time.time(),
                operation=operation,
                parameters={'args': args, 'kwargs': kwargs},
                stack_trace=traceback.format_exc(),
                system_state=self._get_system_state()
            )
            
            # Attempt recovery
            recovery_result = await self._attempt_recovery(error_context, func, *args, **kwargs)
            
            if recovery_result['recovered']:
                self.system_health_metrics['recovered_errors'] += 1
                return {
                    'success': True,
                    'result': recovery_result['result'],
                    'execution_time': time.time() - start_time,
                    'operation': operation,
                    'recovery_applied': recovery_result['strategy'],
                    'recovery_attempts': recovery_result['attempts']
                }
            else:
                self.system_health_metrics['failed_recoveries'] += 1
                if error_context.severity == ErrorSeverity.CRITICAL:
                    self.system_health_metrics['last_critical_error'] = error_context.timestamp
                
                return {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'execution_time': time.time() - start_time,
                    'operation': operation,
                    'recovery_attempted': True,
                    'recovery_strategy': recovery_result.get('strategy', 'none'),
                    'severity': error_context.severity.value
                }
        finally:
            self.system_health_metrics['total_errors'] += 1
            if error_context:
                self.error_history.append(error_context)
                self._cleanup_error_history()
    
    async def _attempt_recovery(self, error_context: ErrorContext, func: Callable,
                               *args, **kwargs) -> Dict[str, Any]:
        """Attempt to recover from error using registered strategies."""
        error_type = error_context.error_type
        
        if error_type not in self.recovery_strategies:
            logging.warning(f"No recovery strategy for error type: {error_type}")
            return {'recovered': False, 'strategy': 'none'}
        
        strategy_config = self.recovery_strategies[error_type]
        strategy = strategy_config['strategy']
        config = strategy_config['config']
        
        logging.info(f"Attempting recovery for {error_type} using {strategy.value}")
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._retry_with_backoff(error_context, func, config, *args, **kwargs)
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._execute_fallback(error_context, func, config, *args, **kwargs)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._handle_circuit_breaker(error_context, func, config, *args, **kwargs)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._graceful_degradation(error_context, func, config, *args, **kwargs)
            elif strategy == RecoveryStrategy.SELF_HEALING:
                return await self._self_healing_recovery(error_context, func, config, *args, **kwargs)
            else:
                logging.error(f"Unknown recovery strategy: {strategy}")
                return {'recovered': False, 'strategy': strategy.value}
        
        except Exception as recovery_error:
            logging.error(f"Recovery strategy {strategy.value} failed: {recovery_error}")
            return {
                'recovered': False,
                'strategy': strategy.value,
                'recovery_error': str(recovery_error)
            }
    
    async def _retry_with_backoff(self, error_context: ErrorContext, func: Callable,
                                 config: Dict, *args, **kwargs) -> Dict[str, Any]:
        """Retry with exponential backoff."""
        max_attempts = config.get('max_attempts', 3)
        backoff_factor = config.get('backoff_factor', 2.0)
        base_delay = config.get('base_delay', 1.0)
        
        for attempt in range(max_attempts):
            if attempt > 0:  # Don't delay on first retry
                delay = base_delay * (backoff_factor ** (attempt - 1))
                logging.info(f"Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_attempts})")
                await asyncio.sleep(delay)
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                logging.info(f"Recovery successful on attempt {attempt + 1}")
                return {
                    'recovered': True,
                    'result': result,
                    'strategy': 'retry',
                    'attempts': attempt + 1
                }
            
            except Exception as retry_error:
                error_context.recovery_attempts += 1
                if attempt == max_attempts - 1:  # Last attempt
                    logging.error(f"All retry attempts failed: {retry_error}")
                else:
                    logging.warning(f"Retry attempt {attempt + 1} failed: {retry_error}")
        
        return {'recovered': False, 'strategy': 'retry', 'attempts': max_attempts}
    
    async def _execute_fallback(self, error_context: ErrorContext, func: Callable,
                               config: Dict, *args, **kwargs) -> Dict[str, Any]:
        """Execute fallback strategy."""
        operation = error_context.operation
        
        if operation in self.fallback_handlers:
            logging.info(f"Executing registered fallback handler for {operation}")
            try:
                fallback_func = self.fallback_handlers[operation]
                if asyncio.iscoroutinefunction(fallback_func):
                    result = await fallback_func(*args, **kwargs)
                else:
                    result = fallback_func(*args, **kwargs)
                
                return {
                    'recovered': True,
                    'result': result,
                    'strategy': 'fallback',
                    'fallback_used': True
                }
            except Exception as fallback_error:
                logging.error(f"Fallback handler failed: {fallback_error}")
        
        # Use safe defaults if configured
        if config.get('fallback_safe', False):
            safe_result = self._generate_safe_fallback_result(error_context)
            return {
                'recovered': True,
                'result': safe_result,
                'strategy': 'fallback',
                'safe_defaults_used': True
            }
        
        return {'recovered': False, 'strategy': 'fallback'}
    
    async def _handle_circuit_breaker(self, error_context: ErrorContext, func: Callable,
                                     config: Dict, *args, **kwargs) -> Dict[str, Any]:
        """Handle circuit breaker recovery."""
        error_type = error_context.error_type
        
        if error_type in self.circuit_breakers:
            cb = self.circuit_breakers[error_type]
            cb._on_failure(error_context.timestamp)
            
            if cb.state == CircuitBreakerState.OPEN:
                return await self._handle_circuit_breaker_open(error_context.operation, func, *args, **kwargs)
        
        return {'recovered': False, 'strategy': 'circuit_breaker'}
    
    async def _handle_circuit_breaker_open(self, operation: str, func: Callable,
                                          *args, **kwargs) -> Dict[str, Any]:
        """Handle operations when circuit breaker is open."""
        # Try fallback if available
        if operation in self.fallback_handlers:
            try:
                fallback_func = self.fallback_handlers[operation]
                if asyncio.iscoroutinefunction(fallback_func):
                    result = await fallback_func(*args, **kwargs)
                else:
                    result = fallback_func(*args, **kwargs)
                
                return {
                    'recovered': True,
                    'result': result,
                    'strategy': 'circuit_breaker_fallback',
                    'circuit_breaker_open': True
                }
            except Exception as fallback_error:
                logging.error(f"Circuit breaker fallback failed: {fallback_error}")
        
        # Return fast failure
        return {
            'recovered': False,
            'strategy': 'circuit_breaker',
            'circuit_breaker_open': True,
            'fast_fail': True
        }
    
    async def _graceful_degradation(self, error_context: ErrorContext, func: Callable,
                                   config: Dict, *args, **kwargs) -> Dict[str, Any]:
        """Implement graceful degradation strategy."""
        degradation_level = config.get('degradation_level', 0.5)
        
        # Modify parameters for reduced resource usage
        degraded_kwargs = kwargs.copy()
        
        # Common degradation strategies
        if 'mesh_size' in degraded_kwargs:
            degraded_kwargs['mesh_size'] = int(degraded_kwargs['mesh_size'] * degradation_level)
        
        if 'max_iterations' in degraded_kwargs:
            degraded_kwargs['max_iterations'] = int(degraded_kwargs['max_iterations'] * degradation_level)
        
        if 'precision' in degraded_kwargs and degraded_kwargs['precision'] == 'float64':
            degraded_kwargs['precision'] = 'float32'
        
        logging.info(f"Attempting graceful degradation with level {degradation_level}")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **degraded_kwargs)
            else:
                result = func(*args, **degraded_kwargs)
            
            return {
                'recovered': True,
                'result': result,
                'strategy': 'graceful_degradation',
                'degradation_level': degradation_level,
                'modified_parameters': list(degraded_kwargs.keys())
            }
        
        except Exception as degraded_error:
            logging.error(f"Graceful degradation failed: {degraded_error}")
            return {'recovered': False, 'strategy': 'graceful_degradation'}
    
    async def _self_healing_recovery(self, error_context: ErrorContext, func: Callable,
                                    config: Dict, *args, **kwargs) -> Dict[str, Any]:
        """Implement self-healing recovery strategy."""
        # This would implement more sophisticated self-healing
        # For now, combine retry with parameter adjustment
        
        logging.info("Attempting self-healing recovery")
        
        # First try graceful degradation
        degradation_result = await self._graceful_degradation(error_context, func, 
                                                            {'degradation_level': 0.7}, 
                                                            *args, **kwargs)
        
        if degradation_result['recovered']:
            return {
                'recovered': True,
                'result': degradation_result['result'],
                'strategy': 'self_healing',
                'healing_method': 'graceful_degradation'
            }
        
        # If that fails, try with even more conservative settings
        conservative_config = {'degradation_level': 0.3}
        conservative_result = await self._graceful_degradation(error_context, func,
                                                             conservative_config,
                                                             *args, **kwargs)
        
        if conservative_result['recovered']:
            return {
                'recovered': True,
                'result': conservative_result['result'],
                'strategy': 'self_healing',
                'healing_method': 'conservative_degradation'
            }
        
        return {'recovered': False, 'strategy': 'self_healing'}
    
    def _determine_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type and context."""
        error_type = type(error).__name__
        
        # Critical errors that affect system stability
        if error_type in ['SystemExit', 'KeyboardInterrupt', 'SystemError']:
            return ErrorSeverity.CRITICAL
        
        # High severity errors that significantly impact functionality
        elif error_type in ['MemoryError', 'ConnectionError', 'OSError']:
            return ErrorSeverity.HIGH
        
        # Medium severity errors that can be recovered from
        elif error_type in ['TimeoutError', 'ValueError', 'TypeError', 'ValidationError']:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors that don't significantly impact operation
        else:
            return ErrorSeverity.LOW
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for error context."""
        return {
            'timestamp': time.time(),
            'total_errors': self.system_health_metrics['total_errors'],
            'recent_error_rate': self._calculate_recent_error_rate(),
            'circuit_breaker_states': {
                name: cb.state.value for name, cb in self.circuit_breakers.items()
            }
        }
    
    def _calculate_recent_error_rate(self) -> float:
        """Calculate recent error rate (errors per minute)."""
        current_time = time.time()
        recent_errors = [
            err for err in self.error_history
            if current_time - err.timestamp < 300  # Last 5 minutes
        ]
        
        return len(recent_errors) / 5.0  # errors per minute
    
    def _generate_safe_fallback_result(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Generate safe fallback result for failed operations."""
        return {
            'fallback_result': True,
            'operation': error_context.operation,
            'message': f'Safe fallback result for {error_context.operation}',
            'timestamp': time.time()
        }
    
    def _cleanup_error_history(self):
        """Clean up old error history entries."""
        current_time = time.time()
        # Keep only last 24 hours of error history
        cutoff_time = current_time - (24 * 60 * 60)
        
        self.error_history = [
            err for err in self.error_history
            if err.timestamp > cutoff_time
        ]
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        current_time = time.time()
        uptime = current_time - self.system_health_metrics['system_uptime']
        
        total_errors = self.system_health_metrics['total_errors']
        recovered_errors = self.system_health_metrics['recovered_errors']
        recovery_rate = (recovered_errors / max(1, total_errors)) * 100
        
        # Calculate error rate trends
        recent_errors = [
            err for err in self.error_history
            if current_time - err.timestamp < 3600  # Last hour
        ]
        
        error_types_frequency = {}
        for error in recent_errors:
            error_type = error.error_type
            error_types_frequency[error_type] = error_types_frequency.get(error_type, 0) + 1
        
        return {
            'system_uptime_seconds': uptime,
            'total_errors': total_errors,
            'recovered_errors': recovered_errors,
            'recovery_rate_percent': recovery_rate,
            'recent_error_rate': self._calculate_recent_error_rate(),
            'circuit_breaker_status': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb.failure_count,
                    'success_count': cb.success_count
                }
                for name, cb in self.circuit_breakers.items()
            },
            'error_types_frequency': error_types_frequency,
            'registered_strategies': list(self.recovery_strategies.keys()),
            'fallback_handlers': list(self.fallback_handlers.keys()),
            'last_critical_error': self.system_health_metrics['last_critical_error']
        }


# Global instance for easy access
global_error_recovery_system = AdvancedErrorRecoverySystem()


def with_error_recovery(operation: str):
    """Decorator for automatic error recovery."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            return await global_error_recovery_system.execute_with_recovery(
                operation, func, *args, **kwargs
            )
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage and demonstration
async def demo_error_recovery():
    """Demonstrate error recovery capabilities."""
    print("🛡️ Starting Error Recovery System Demonstration")
    
    recovery_system = AdvancedErrorRecoverySystem()
    
    # Register a custom fallback handler
    def safe_computation_fallback(*args, **kwargs):
        return {"result": "safe_fallback_value", "fallback": True}
    
    recovery_system.register_fallback_handler("test_computation", safe_computation_fallback)
    
    # Test function that can fail
    failure_count = 0
    
    async def unreliable_function(should_fail: bool = False):
        nonlocal failure_count
        if should_fail or failure_count < 2:
            failure_count += 1
            if failure_count == 1:
                raise TimeoutError("Simulated timeout error")
            elif failure_count == 2:
                raise MemoryError("Simulated memory error")
        return {"result": "success", "attempts": failure_count + 1}
    
    # Test error recovery
    test_cases = [
        {"should_fail": True, "description": "TimeoutError with retry"},
        {"should_fail": True, "description": "MemoryError with degradation"},
        {"should_fail": False, "description": "Success after retries"}
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} error recovery scenarios:")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['description']} ---")
        
        result = await recovery_system.execute_with_recovery(
            "test_computation", 
            unreliable_function,
            should_fail=test_case["should_fail"]
        )
        
        if result['success']:
            print(f"✅ Recovery successful!")
            if 'recovery_applied' in result:
                print(f"   Strategy: {result['recovery_applied']}")
                print(f"   Attempts: {result['recovery_attempts']}")
        else:
            print(f"❌ Recovery failed: {result.get('error', 'Unknown error')}")
        
        print(f"   Execution time: {result['execution_time']:.4f}s")
    
    # Generate health report
    print(f"\n📊 System Health Report:")
    health = recovery_system.get_health_report()
    print(f"   System uptime: {health['system_uptime_seconds']:.1f}s")
    print(f"   Total errors: {health['total_errors']}")
    print(f"   Recovery rate: {health['recovery_rate_percent']:.1f}%")
    print(f"   Recent error rate: {health['recent_error_rate']:.2f} errors/min")
    
    return recovery_system


if __name__ == "__main__":
    # Run demonstration
    recovery_system = asyncio.run(demo_error_recovery())
    print(f"\n🎉 Advanced Error Recovery Generation 2 Enhancement Complete!")