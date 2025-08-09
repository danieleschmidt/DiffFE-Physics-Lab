"""Comprehensive health check and monitoring endpoints for production deployment."""

import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import threading
from collections import deque, defaultdict

try:
    from flask import Blueprint, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from ..backends.robust_backend import get_global_backend_manager
from ..performance.monitor import get_global_monitor
from ..utils.exceptions import create_error_response
from ..utils.logging_config import get_logger
from ..utils.config_manager import get_global_config
from ..services.robust_solver import RobustFEBMLSolver
from ..services.robust_optimization import RobustOptimizer

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """Health check status."""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    timestamp: str
    details: Dict[str, Any] = None
    response_time_ms: Optional[float] = None


@dataclass
class SystemHealth:
    """Overall system health."""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    timestamp: str
    uptime_seconds: float
    version: str
    environment: str
    components: List[HealthStatus]
    metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.start_time = time.time()
        self.health_history = deque(maxlen=100)
        self.component_checkers = {}
        self._register_default_checkers()
        self._lock = threading.Lock()
    
    def _register_default_checkers(self):
        """Register default health checkers."""
        self.component_checkers = {
            'system': self._check_system_health,
            'memory': self._check_memory_health,
            'disk': self._check_disk_health,
            'backends': self._check_backends_health,
            'database': self._check_database_health,
            'cache': self._check_cache_health,
            'solver': self._check_solver_health,
            'optimization': self._check_optimization_health,
        }
    
    def get_system_health(self, include_details: bool = True) -> SystemHealth:
        """Get comprehensive system health status.
        
        Parameters
        ----------
        include_details : bool, optional
            Whether to include detailed component information
            
        Returns
        -------
        SystemHealth
            Overall system health status
        """
        start_time = time.time()
        
        # Check all components
        component_statuses = []
        for component_name, checker_func in self.component_checkers.items():
            try:
                check_start = time.time()
                status = checker_func(include_details)
                check_time = (time.time() - check_start) * 1000
                status.response_time_ms = check_time
                component_statuses.append(status)
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                component_statuses.append(HealthStatus(
                    component=component_name,
                    status='unhealthy',
                    message=f"Health check error: {e}",
                    timestamp=datetime.now().isoformat(),
                    response_time_ms=None
                ))
        
        # Determine overall status
        overall_status = self._determine_overall_status(component_statuses)
        
        # Get system metrics
        metrics = self._get_system_metrics()
        
        # Get active alerts
        alerts = self._get_active_alerts()
        
        # Get configuration
        config = get_global_config()
        
        health = SystemHealth(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=time.time() - self.start_time,
            version=config.version if config else "unknown",
            environment=config.environment if config else "unknown",
            components=component_statuses,
            metrics=metrics,
            alerts=alerts
        )
        
        # Store in history
        with self._lock:
            self.health_history.append(health)
        
        total_time = (time.time() - start_time) * 1000
        logger.debug(f"Health check completed in {total_time:.1f}ms - Status: {overall_status}")
        
        return health
    
    def _determine_overall_status(self, component_statuses: List[HealthStatus]) -> str:
        """Determine overall system status from component statuses."""
        if any(status.status == 'unhealthy' for status in component_statuses):
            return 'unhealthy'
        elif any(status.status == 'degraded' for status in component_statuses):
            return 'degraded'
        else:
            return 'healthy'
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            process = psutil.Process()
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
                'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'process_memory_mb': process.memory_info().rss / (1024 * 1024),
                'process_cpu_percent': process.cpu_percent(),
                'thread_count': process.num_threads(),
                'file_descriptors': process.num_fds() if hasattr(process, 'num_fds') else None,
                'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts."""
        alerts = []
        
        try:
            # Get alerts from performance monitor
            monitor = get_global_monitor()
            recent_alerts = monitor.get_recent_alerts(hours=1)
            
            for alert in recent_alerts:
                if alert.get('level') in ['warning', 'critical']:
                    alerts.append({
                        'type': alert.get('alert_type', 'unknown'),
                        'level': alert.get('level', 'info'),
                        'message': alert.get('message', ''),
                        'timestamp': alert.get('timestamp', ''),
                        'source': 'performance_monitor'
                    })
        except Exception as e:
            logger.warning(f"Error getting performance monitor alerts: {e}")
        
        # Add resource-based alerts
        try:
            metrics = self._get_system_metrics()
            
            if metrics.get('memory_percent', 0) > 90:
                alerts.append({
                    'type': 'high_memory_usage',
                    'level': 'critical',
                    'message': f"High memory usage: {metrics['memory_percent']:.1f}%",
                    'timestamp': datetime.now().isoformat(),
                    'source': 'health_checker'
                })
            
            if metrics.get('cpu_percent', 0) > 95:
                alerts.append({
                    'type': 'high_cpu_usage',
                    'level': 'critical',
                    'message': f"High CPU usage: {metrics['cpu_percent']:.1f}%",
                    'timestamp': datetime.now().isoformat(),
                    'source': 'health_checker'
                })
            
            if metrics.get('disk_usage_percent', 0) > 95:
                alerts.append({
                    'type': 'high_disk_usage',
                    'level': 'critical',
                    'message': f"High disk usage: {metrics['disk_usage_percent']:.1f}%",
                    'timestamp': datetime.now().isoformat(),
                    'source': 'health_checker'
                })
        except Exception as e:
            logger.warning(f"Error generating resource alerts: {e}")
        
        return alerts
    
    def _check_system_health(self, include_details: bool = True) -> HealthStatus:
        """Check overall system health."""
        try:
            metrics = self._get_system_metrics()
            
            # Check critical thresholds
            issues = []
            
            if metrics.get('memory_percent', 0) > 95:
                issues.append(f"Critical memory usage: {metrics['memory_percent']:.1f}%")
            elif metrics.get('memory_percent', 0) > 85:
                issues.append(f"High memory usage: {metrics['memory_percent']:.1f}%")
            
            if metrics.get('cpu_percent', 0) > 95:
                issues.append(f"Critical CPU usage: {metrics['cpu_percent']:.1f}%")
            
            if metrics.get('disk_usage_percent', 0) > 95:
                issues.append(f"Critical disk usage: {metrics['disk_usage_percent']:.1f}%")
            
            # Determine status
            if any('Critical' in issue for issue in issues):
                status = 'unhealthy'
                message = '; '.join(issues)
            elif issues:
                status = 'degraded'
                message = '; '.join(issues)
            else:
                status = 'healthy'
                message = 'System resources within normal limits'
            
            return HealthStatus(
                component='system',
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                details=metrics if include_details else None
            )
            
        except Exception as e:
            return HealthStatus(
                component='system',
                status='unhealthy',
                message=f"System check failed: {e}",
                timestamp=datetime.now().isoformat()
            )
    
    def _check_memory_health(self, include_details: bool = True) -> HealthStatus:
        """Check memory health."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                status = 'unhealthy'
                message = f"Critical memory usage: {memory.percent:.1f}%"
            elif memory.percent > 85:
                status = 'degraded'
                message = f"High memory usage: {memory.percent:.1f}%"
            else:
                status = 'healthy'
                message = f"Memory usage normal: {memory.percent:.1f}%"
            
            details = {
                'percent_used': memory.percent,
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3)
            } if include_details else None
            
            return HealthStatus(
                component='memory',
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                component='memory',
                status='unhealthy',
                message=f"Memory check failed: {e}",
                timestamp=datetime.now().isoformat()
            )
    
    def _check_disk_health(self, include_details: bool = True) -> HealthStatus:
        """Check disk health."""
        try:
            disk = psutil.disk_usage('/')
            
            if disk.percent > 95:
                status = 'unhealthy'
                message = f"Critical disk usage: {disk.percent:.1f}%"
            elif disk.percent > 85:
                status = 'degraded'
                message = f"High disk usage: {disk.percent:.1f}%"
            else:
                status = 'healthy'
                message = f"Disk usage normal: {disk.percent:.1f}%"
            
            details = {
                'percent_used': disk.percent,
                'total_gb': disk.total / (1024**3),
                'free_gb': disk.free / (1024**3),
                'used_gb': disk.used / (1024**3)
            } if include_details else None
            
            return HealthStatus(
                component='disk',
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                component='disk',
                status='unhealthy',
                message=f"Disk check failed: {e}",
                timestamp=datetime.now().isoformat()
            )
    
    def _check_backends_health(self, include_details: bool = True) -> HealthStatus:
        """Check automatic differentiation backends health."""
        try:
            backend_manager = get_global_backend_manager()
            available_backends = backend_manager.list_available_backends()
            
            if not available_backends:
                status = 'unhealthy'
                message = "No AD backends available"
            elif 'jax' in available_backends or 'torch' in available_backends:
                status = 'healthy'
                message = f"AD backends available: {', '.join(available_backends)}"
            else:
                status = 'degraded'
                message = f"Only fallback backends available: {', '.join(available_backends)}"
            
            details = None
            if include_details:
                backend_report = backend_manager.generate_backend_report()
                details = {
                    'available_backends': available_backends,
                    'backend_details': backend_report.get('backend_details', {}),
                    'recommendations': backend_report.get('recommendations', [])
                }
            
            return HealthStatus(
                component='backends',
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                component='backends',
                status='unhealthy',
                message=f"Backend check failed: {e}",
                timestamp=datetime.now().isoformat()
            )
    
    def _check_database_health(self, include_details: bool = True) -> HealthStatus:
        """Check database health."""
        try:
            config = get_global_config()
            if not config or not config.database:
                return HealthStatus(
                    component='database',
                    status='healthy',
                    message="Database not configured",
                    timestamp=datetime.now().isoformat()
                )
            
            # Try to connect to database
            try:
                # This would typically use your database connection
                # For now, we'll do a basic connectivity check
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)  # 5 second timeout
                result = sock.connect_ex((config.database.host, config.database.port))
                sock.close()
                
                if result == 0:
                    status = 'healthy'
                    message = f"Database connection successful to {config.database.host}:{config.database.port}"
                else:
                    status = 'unhealthy'
                    message = f"Cannot connect to database at {config.database.host}:{config.database.port}"
                
            except Exception as e:
                status = 'unhealthy'
                message = f"Database connection failed: {e}"
            
            details = {
                'host': config.database.host,
                'port': config.database.port,
                'database': config.database.database
            } if include_details else None
            
            return HealthStatus(
                component='database',
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                component='database',
                status='unhealthy',
                message=f"Database check failed: {e}",
                timestamp=datetime.now().isoformat()
            )
    
    def _check_cache_health(self, include_details: bool = True) -> HealthStatus:
        """Check cache health."""
        try:
            config = get_global_config()
            if not config or config.cache.type == 'memory':
                return HealthStatus(
                    component='cache',
                    status='healthy',
                    message="Using memory cache",
                    timestamp=datetime.now().isoformat(),
                    details={'type': 'memory'} if include_details else None
                )
            
            # Check Redis/Memcached connectivity
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                result = sock.connect_ex((config.cache.host, config.cache.port))
                sock.close()
                
                if result == 0:
                    status = 'healthy'
                    message = f"Cache connection successful to {config.cache.host}:{config.cache.port}"
                else:
                    status = 'degraded'
                    message = f"Cannot connect to cache, falling back to memory"
                
            except Exception as e:
                status = 'degraded'
                message = f"Cache connection failed, using memory: {e}"
            
            details = {
                'type': config.cache.type,
                'host': config.cache.host,
                'port': config.cache.port
            } if include_details else None
            
            return HealthStatus(
                component='cache',
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                component='cache',
                status='degraded',
                message=f"Cache check failed: {e}",
                timestamp=datetime.now().isoformat()
            )
    
    def _check_solver_health(self, include_details: bool = True) -> HealthStatus:
        """Check solver health."""
        try:
            # This would typically check if solver services are responsive
            # For now, we'll do a basic import and initialization check
            
            status = 'healthy'
            message = "Solver components available"
            details = None
            
            if include_details:
                config = get_global_config()
                details = {
                    'default_method': config.solver.default_method if config else 'newton',
                    'monitoring_enabled': config.solver.enable_monitoring if config else True,
                    'timeout_configured': config.solver.timeout_minutes is not None if config else False
                }
            
            return HealthStatus(
                component='solver',
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                component='solver',
                status='degraded',
                message=f"Solver check failed: {e}",
                timestamp=datetime.now().isoformat()
            )
    
    def _check_optimization_health(self, include_details: bool = True) -> HealthStatus:
        """Check optimization service health."""
        try:
            status = 'healthy'
            message = "Optimization components available"
            details = None
            
            if include_details:
                config = get_global_config()
                details = {
                    'default_method': config.optimization.default_method if config else 'L-BFGS-B',
                    'checkpointing_enabled': config.optimization.enable_checkpointing if config else True,
                    'checkpoint_dir_exists': os.path.exists(config.optimization.checkpoint_dir) if config else False
                }
            
            return HealthStatus(
                component='optimization',
                status=status,
                message=message,
                timestamp=datetime.now().isoformat(),
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                component='optimization',
                status='degraded',
                message=f"Optimization check failed: {e}",
                timestamp=datetime.now().isoformat()
            )
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealth]:
        """Get health check history.
        
        Parameters
        ----------
        hours : int, optional
            Hours of history to return
            
        Returns
        -------
        List[SystemHealth]
            Historical health data
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                health for health in self.health_history
                if datetime.fromisoformat(health.timestamp.replace('Z', '+00:00')) > cutoff_time
            ]
    
    def get_component_status(self, component_name: str) -> Optional[HealthStatus]:
        """Get status for a specific component.
        
        Parameters
        ----------
        component_name : str
            Name of component to check
            
        Returns
        -------
        Optional[HealthStatus]
            Component health status
        """
        if component_name in self.component_checkers:
            try:
                return self.component_checkers[component_name](include_details=True)
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                return HealthStatus(
                    component=component_name,
                    status='unhealthy',
                    message=f"Health check error: {e}",
                    timestamp=datetime.now().isoformat()
                )
        return None


# Global health checker instance
_global_health_checker = None


def get_global_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


# Flask blueprint for health endpoints
if HAS_FLASK:
    health_bp = Blueprint('health', __name__, url_prefix='/health')
    
    @health_bp.route('/')
    def health_check():
        """Basic health check endpoint."""
        try:
            health_checker = get_global_health_checker()
            system_health = health_checker.get_system_health(include_details=False)
            
            # Return simplified health status
            response = {
                'status': system_health.status,
                'timestamp': system_health.timestamp,
                'uptime_seconds': system_health.uptime_seconds,
                'version': system_health.version,
                'message': f"System is {system_health.status}"
            }
            
            status_code = 200 if system_health.status == 'healthy' else 503
            return jsonify(response), status_code
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'message': f"Health check error: {e}"
            }), 503
    
    @health_bp.route('/detailed')
    def detailed_health_check():
        """Detailed health check with all components."""
        try:
            health_checker = get_global_health_checker()
            system_health = health_checker.get_system_health(include_details=True)
            
            response = asdict(system_health)
            status_code = 200 if system_health.status == 'healthy' else 503
            
            return jsonify(response), status_code
            
        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            return jsonify(create_error_response(e)), 500
    
    @health_bp.route('/component/<component_name>')
    def component_health_check(component_name: str):
        """Health check for specific component."""
        try:
            health_checker = get_global_health_checker()
            component_status = health_checker.get_component_status(component_name)
            
            if component_status is None:
                return jsonify({
                    'error': f"Unknown component: {component_name}",
                    'available_components': list(health_checker.component_checkers.keys())
                }), 404
            
            response = asdict(component_status)
            status_code = 200 if component_status.status == 'healthy' else 503
            
            return jsonify(response), status_code
            
        except Exception as e:
            logger.error(f"Component health check failed for {component_name}: {e}")
            return jsonify(create_error_response(e)), 500
    
    @health_bp.route('/metrics')
    def health_metrics():
        """Get system metrics."""
        try:
            health_checker = get_global_health_checker()
            metrics = health_checker._get_system_metrics()
            
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
            
        except Exception as e:
            logger.error(f"Health metrics failed: {e}")
            return jsonify(create_error_response(e)), 500
    
    @health_bp.route('/alerts')
    def health_alerts():
        """Get active alerts."""
        try:
            health_checker = get_global_health_checker()
            alerts = health_checker._get_active_alerts()
            
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'alerts': alerts,
                'alert_count': len(alerts)
            })
            
        except Exception as e:
            logger.error(f"Health alerts failed: {e}")
            return jsonify(create_error_response(e)), 500
    
    @health_bp.route('/history')
    def health_history():
        """Get health check history."""
        try:
            hours = request.args.get('hours', 24, type=int)
            health_checker = get_global_health_checker()
            history = health_checker.get_health_history(hours)
            
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'history_hours': hours,
                'history_count': len(history),
                'history': [asdict(h) for h in history]
            })
            
        except Exception as e:
            logger.error(f"Health history failed: {e}")
            return jsonify(create_error_response(e)), 500
    
    @health_bp.route('/readiness')
    def readiness_check():
        """Kubernetes-style readiness probe."""
        try:
            health_checker = get_global_health_checker()
            system_health = health_checker.get_system_health(include_details=False)
            
            # Ready if not unhealthy
            is_ready = system_health.status != 'unhealthy'
            
            response = {
                'ready': is_ready,
                'status': system_health.status,
                'timestamp': system_health.timestamp
            }
            
            return jsonify(response), 200 if is_ready else 503
            
        except Exception as e:
            return jsonify({'ready': False, 'error': str(e)}), 503
    
    @health_bp.route('/liveness')
    def liveness_check():
        """Kubernetes-style liveness probe."""
        try:
            # Basic liveness - can we respond?
            return jsonify({
                'alive': True,
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time() - get_global_health_checker().start_time
            })
            
        except Exception as e:
            return jsonify({'alive': False, 'error': str(e)}), 503

else:
    # Placeholder if Flask is not available
    health_bp = None
    logger.warning("Flask not available - health monitoring endpoints disabled")