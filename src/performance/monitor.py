"""Performance monitoring and alerting system."""

import time
import threading
import logging
import psutil
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Performance alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance alert representation."""
    alert_type: str
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[float] = None
    actual_value: Optional[float] = None


@dataclass
class SystemMetrics:
    """System performance metrics snapshot."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_bytes_sent: float = 0.0
    network_bytes_recv: float = 0.0
    process_count: int = 0
    thread_count: int = 0
    load_average: List[float] = field(default_factory=list)


@dataclass
class ApplicationMetrics:
    """Application-specific performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    request_count: int = 0
    error_count: int = 0
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    active_connections: int = 0
    cache_hit_rate: float = 0.0
    database_query_time: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class PerformanceMonitor:
    """Comprehensive performance monitoring system.
    
    Provides real-time monitoring and alerting for:
    - System resource usage (CPU, Memory, Disk, Network)
    - Application performance metrics
    - Custom performance indicators
    - Automated alerting and notifications
    - Performance trend analysis
    
    Examples
    --------
    >>> monitor = PerformanceMonitor()
    >>> monitor.start_monitoring()
    >>> 
    >>> # Add custom alert
    >>> monitor.add_alert_rule('high_cpu', 'cpu_percent', threshold=80.0)
    >>> 
    >>> # Log application metrics
    >>> monitor.log_application_metric('response_time', 150.0)
    >>> 
    >>> # Get current metrics
    >>> metrics = monitor.get_current_metrics()
    """
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 3600,
        enable_system_monitoring: bool = True,
        enable_application_monitoring: bool = True
    ):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_system_monitoring = enable_system_monitoring
        self.enable_application_monitoring = enable_application_monitoring
        
        # Data storage
        self.system_metrics_history = deque(maxlen=history_size)
        self.application_metrics_history = deque(maxlen=history_size)
        self.alerts_history = deque(maxlen=1000)
        
        # Alert configuration
        self.alert_rules = {}
        self.alert_callbacks = []
        self.alert_thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 90.0,
            'response_time_avg': 1000.0,  # ms
            'error_rate': 5.0  # %
        }
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread = None
        self._lock = threading.RLock()
        
        # Performance baselines
        self._baselines = {}
        self._trend_data = defaultdict(list)
        
        # Custom metrics
        self._custom_metrics = {}
        self._metric_aggregators = {}
        
        logger.info(f"Performance monitor initialized (interval={monitoring_interval}s, history={history_size})")
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self._monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")
        
        while self._monitoring_active:
            try:
                start_time = time.time()
                
                # Collect system metrics
                if self.enable_system_monitoring:
                    system_metrics = self._collect_system_metrics()
                    if system_metrics:
                        with self._lock:
                            self.system_metrics_history.append(system_metrics)
                            self._check_system_alerts(system_metrics)
                
                # Collect application metrics
                if self.enable_application_monitoring:
                    app_metrics = self._collect_application_metrics()
                    if app_metrics:
                        with self._lock:
                            self.application_metrics_history.append(app_metrics)
                            self._check_application_alerts(app_metrics)
                
                # Update trends
                self._update_trends()
                
                # Sleep for remaining interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
        
        logger.info("Monitoring loop stopped")
    
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0.0
            disk_io_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0.0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_bytes_sent = net_io.bytes_sent if net_io else 0.0
            network_bytes_recv = net_io.bytes_recv if net_io else 0.0
            
            # Process information
            process_count = len(psutil.pids())
            
            # Thread count for current process
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            # Load average (Unix-like systems)
            load_average = []
            try:
                load_average = list(os.getloadavg())
            except (OSError, AttributeError):
                load_average = [0.0, 0.0, 0.0]
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                thread_count=thread_count,
                load_average=load_average
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    def _collect_application_metrics(self) -> Optional[ApplicationMetrics]:
        """Collect application performance metrics."""
        try:
            # Get metrics from custom collectors
            metrics = ApplicationMetrics()
            
            # Aggregate custom metrics
            with self._lock:
                for metric_name, values in self._custom_metrics.items():
                    if values:
                        if metric_name in self._metric_aggregators:
                            aggregator = self._metric_aggregators[metric_name]
                            aggregated_value = aggregator(values)
                        else:
                            # Default to average
                            aggregated_value = sum(values) / len(values)
                        
                        metrics.custom_metrics[metric_name] = aggregated_value
                
                # Clear custom metrics after aggregation
                self._custom_metrics.clear()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return None
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics against alert thresholds."""
        alerts = []
        
        # CPU alerts
        if metrics.cpu_percent > self.alert_thresholds.get('cpu_percent', 85.0):
            alerts.append(PerformanceAlert(
                alert_type='high_cpu_usage',
                level=AlertLevel.CRITICAL if metrics.cpu_percent > 95.0 else AlertLevel.WARNING,
                message=f'High CPU usage: {metrics.cpu_percent:.1f}%',
                threshold=self.alert_thresholds.get('cpu_percent'),
                actual_value=metrics.cpu_percent,
                metrics={'cpu_percent': metrics.cpu_percent}
            ))
        
        # Memory alerts
        if metrics.memory_percent > self.alert_thresholds.get('memory_percent', 90.0):
            alerts.append(PerformanceAlert(
                alert_type='high_memory_usage',
                level=AlertLevel.CRITICAL if metrics.memory_percent > 95.0 else AlertLevel.WARNING,
                message=f'High memory usage: {metrics.memory_percent:.1f}%',
                threshold=self.alert_thresholds.get('memory_percent'),
                actual_value=metrics.memory_percent,
                metrics={'memory_percent': metrics.memory_percent, 'memory_used_mb': metrics.memory_used_mb}
            ))
        
        # Disk usage alerts
        if metrics.disk_usage_percent > 85.0:
            alerts.append(PerformanceAlert(
                alert_type='high_disk_usage',
                level=AlertLevel.WARNING if metrics.disk_usage_percent < 95.0 else AlertLevel.CRITICAL,
                message=f'High disk usage: {metrics.disk_usage_percent:.1f}%',
                threshold=85.0,
                actual_value=metrics.disk_usage_percent,
                metrics={'disk_usage_percent': metrics.disk_usage_percent}
            ))
        
        # Load average alerts (for Unix-like systems)
        if metrics.load_average and len(metrics.load_average) > 0:
            load_1min = metrics.load_average[0]
            cpu_count = psutil.cpu_count()
            
            if load_1min > cpu_count * 2:  # Load > 2x CPU count
                alerts.append(PerformanceAlert(
                    alert_type='high_load_average',
                    level=AlertLevel.WARNING,
                    message=f'High load average: {load_1min:.2f} (CPUs: {cpu_count})',
                    threshold=cpu_count * 2,
                    actual_value=load_1min,
                    metrics={'load_1min': load_1min, 'cpu_count': cpu_count}
                ))
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _check_application_alerts(self, metrics: ApplicationMetrics):
        """Check application metrics against alert thresholds."""
        alerts = []
        
        # Response time alerts
        if metrics.response_time_avg > self.alert_thresholds.get('response_time_avg', 1000.0):
            alerts.append(PerformanceAlert(
                alert_type='slow_response_time',
                level=AlertLevel.WARNING if metrics.response_time_avg < 2000.0 else AlertLevel.CRITICAL,
                message=f'Slow response time: {metrics.response_time_avg:.1f}ms',
                threshold=self.alert_thresholds.get('response_time_avg'),
                actual_value=metrics.response_time_avg,
                metrics={'response_time_avg': metrics.response_time_avg}
            ))
        
        # Error rate alerts
        if metrics.request_count > 0:
            error_rate = (metrics.error_count / metrics.request_count) * 100
            if error_rate > self.alert_thresholds.get('error_rate', 5.0):
                alerts.append(PerformanceAlert(
                    alert_type='high_error_rate',
                    level=AlertLevel.CRITICAL if error_rate > 20.0 else AlertLevel.WARNING,
                    message=f'High error rate: {error_rate:.1f}%',
                    threshold=self.alert_thresholds.get('error_rate'),
                    actual_value=error_rate,
                    metrics={'error_rate': error_rate, 'error_count': metrics.error_count, 'request_count': metrics.request_count}
                ))
        
        # Custom metric alerts
        for rule_name, rule_config in self.alert_rules.items():
            metric_name = rule_config['metric']
            threshold = rule_config['threshold']
            
            if metric_name in metrics.custom_metrics:
                value = metrics.custom_metrics[metric_name]
                
                if value > threshold:
                    alerts.append(PerformanceAlert(
                        alert_type=f'custom_{rule_name}',
                        level=rule_config.get('level', AlertLevel.WARNING),
                        message=f'Custom alert {rule_name}: {metric_name} = {value:.2f} (threshold: {threshold})',
                        threshold=threshold,
                        actual_value=value,
                        metrics={metric_name: value}
                    ))
        
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger performance alert."""
        with self._lock:
            self.alerts_history.append(alert)
        
        # Log alert
        log_func = logger.critical if alert.level == AlertLevel.CRITICAL else logger.warning
        log_func(f"Performance Alert: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _update_trends(self):
        """Update performance trend data."""
        try:
            with self._lock:
                # System trends
                if self.system_metrics_history:
                    latest_system = self.system_metrics_history[-1]
                    self._trend_data['cpu_percent'].append(latest_system.cpu_percent)
                    self._trend_data['memory_percent'].append(latest_system.memory_percent)
                    self._trend_data['disk_usage_percent'].append(latest_system.disk_usage_percent)
                
                # Application trends
                if self.application_metrics_history:
                    latest_app = self.application_metrics_history[-1]
                    self._trend_data['response_time_avg'].append(latest_app.response_time_avg)
                    
                    for metric_name, value in latest_app.custom_metrics.items():
                        self._trend_data[f'custom_{metric_name}'].append(value)
                
                # Limit trend data size
                max_trend_size = 1000
                for trend_name in self._trend_data:
                    if len(self._trend_data[trend_name]) > max_trend_size:
                        self._trend_data[trend_name] = self._trend_data[trend_name][-max_trend_size:]
                        
        except Exception as e:
            logger.error(f"Failed to update trends: {e}")
    
    def add_alert_rule(self, rule_name: str, metric_name: str, threshold: float, level: AlertLevel = AlertLevel.WARNING):
        """Add custom alert rule.
        
        Parameters
        ----------
        rule_name : str
            Name of the alert rule
        metric_name : str
            Name of the metric to monitor
        threshold : float
            Threshold value for triggering alert
        level : AlertLevel, optional
            Alert severity level
        """
        self.alert_rules[rule_name] = {
            'metric': metric_name,
            'threshold': threshold,
            'level': level
        }
        
        logger.info(f"Added alert rule '{rule_name}': {metric_name} > {threshold}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for performance alerts.
        
        Parameters
        ----------
        callback : Callable[[PerformanceAlert], None]
            Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
        logger.info("Added performance alert callback")
    
    def set_alert_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for a metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric
        threshold : float
            Threshold value
        """
        self.alert_thresholds[metric_name] = threshold
        logger.info(f"Set alert threshold {metric_name} = {threshold}")
    
    def log_application_metric(self, metric_name: str, value: float, aggregator: Optional[Callable] = None):
        """Log application performance metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric
        value : float
            Metric value
        aggregator : Callable, optional
            Function to aggregate multiple values (default: average)
        """
        with self._lock:
            if metric_name not in self._custom_metrics:
                self._custom_metrics[metric_name] = []
            
            self._custom_metrics[metric_name].append(value)
            
            if aggregator:
                self._metric_aggregators[metric_name] = aggregator
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns
        -------
        Dict[str, Any]
            Current system and application metrics
        """
        with self._lock:
            current_system = self.system_metrics_history[-1] if self.system_metrics_history else None
            current_app = self.application_metrics_history[-1] if self.application_metrics_history else None
            
            return {
                'system': current_system.__dict__ if current_system else {},
                'application': current_app.__dict__ if current_app else {},
                'alerts_count': len(self.alerts_history),
                'monitoring_active': self._monitoring_active
            }
    
    def get_metrics_history(self, hours: int = 1) -> Dict[str, List[Dict]]:
        """Get metrics history for specified time period.
        
        Parameters
        ----------
        hours : int, optional
            Number of hours of history to return
            
        Returns
        -------
        Dict[str, List[Dict]]
            Historical metrics data
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            # Filter by time
            system_history = [
                m.__dict__ for m in self.system_metrics_history
                if m.timestamp > cutoff_time
            ]
            
            app_history = [
                m.__dict__ for m in self.application_metrics_history
                if m.timestamp > cutoff_time
            ]
            
            return {
                'system': system_history,
                'application': app_history
            }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts.
        
        Parameters
        ----------
        hours : int, optional
            Number of hours to look back
            
        Returns
        -------
        List[Dict[str, Any]]
            Recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_alerts = [
                {
                    'alert_type': alert.alert_type,
                    'level': alert.level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'threshold': alert.threshold,
                    'actual_value': alert.actual_value,
                    'metrics': alert.metrics
                }
                for alert in self.alerts_history
                if alert.timestamp > cutoff_time
            ]
            
            return sorted(recent_alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_trend_analysis(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get trend analysis for a specific metric.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric to analyze
        window_minutes : int, optional
            Time window for analysis in minutes
            
        Returns
        -------
        Dict[str, Any]
            Trend analysis results
        """
        if metric_name not in self._trend_data:
            return {'error': f'No trend data available for {metric_name}'}
        
        with self._lock:
            data = self._trend_data[metric_name]
            
            if len(data) < 2:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Get recent data points
            window_size = min(window_minutes, len(data))
            recent_data = data[-window_size:]
            
            # Calculate statistics
            import numpy as np
            
            mean_value = np.mean(recent_data)
            std_value = np.std(recent_data)
            min_value = np.min(recent_data)
            max_value = np.max(recent_data)
            
            # Calculate trend direction
            if len(recent_data) >= 10:
                # Linear regression for trend
                x = np.arange(len(recent_data))
                y = np.array(recent_data)
                trend_slope = np.polyfit(x, y, 1)[0]
                
                if trend_slope > 0.1:
                    trend_direction = 'increasing'
                elif trend_slope < -0.1:
                    trend_direction = 'decreasing'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'unknown'
            
            return {
                'metric_name': metric_name,
                'window_minutes': window_minutes,
                'data_points': len(recent_data),
                'statistics': {
                    'mean': mean_value,
                    'std': std_value,
                    'min': min_value,
                    'max': max_value
                },
                'trend_direction': trend_direction,
                'current_value': recent_data[-1] if recent_data else None
            }
    
    def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Parameters
        ----------
        hours : int, optional
            Time period for report in hours
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive performance report
        """
        current_metrics = self.get_current_metrics()
        metrics_history = self.get_metrics_history(hours)
        recent_alerts = self.get_recent_alerts(hours)
        
        # Alert summary
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert['level']] += 1
        
        # Top metrics by value
        trend_summaries = {}
        for metric_name in ['cpu_percent', 'memory_percent', 'response_time_avg']:
            trend_summaries[metric_name] = self.get_trend_analysis(metric_name)
        
        report = {
            'report_period': f'{hours} hours',
            'generated_at': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'alert_summary': {
                'total_alerts': len(recent_alerts),
                'by_level': dict(alert_counts),
                'recent_alerts': recent_alerts[:10]  # Top 10 recent alerts
            },
            'trend_analysis': trend_summaries,
            'recommendations': self._generate_performance_recommendations(current_metrics, recent_alerts),
            'system_health': self._assess_system_health(current_metrics, recent_alerts)
        }
        
        return report
    
    def _generate_performance_recommendations(self, current_metrics: Dict, recent_alerts: List) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        system_metrics = current_metrics.get('system', {})
        
        # CPU recommendations
        cpu_percent = system_metrics.get('cpu_percent', 0)
        if cpu_percent > 80:
            recommendations.append(f"High CPU usage ({cpu_percent:.1f}%) - consider optimizing CPU-intensive operations")
        
        # Memory recommendations
        memory_percent = system_metrics.get('memory_percent', 0)
        if memory_percent > 85:
            recommendations.append(f"High memory usage ({memory_percent:.1f}%) - consider memory optimization or scaling")
        
        # Alert-based recommendations
        critical_alerts = [a for a in recent_alerts if a['level'] == 'critical']
        if critical_alerts:
            recommendations.append(f"Address {len(critical_alerts)} critical alerts immediately")
        
        # Trend-based recommendations
        if 'cpu_percent' in self._trend_data and len(self._trend_data['cpu_percent']) > 10:
            cpu_trend = self.get_trend_analysis('cpu_percent')
            if cpu_trend.get('trend_direction') == 'increasing':
                recommendations.append("CPU usage is trending upward - monitor for potential scaling needs")
        
        return recommendations
    
    def _assess_system_health(self, current_metrics: Dict, recent_alerts: List) -> str:
        """Assess overall system health."""
        system_metrics = current_metrics.get('system', {})
        
        # Count critical issues
        critical_issues = 0
        
        if system_metrics.get('cpu_percent', 0) > 90:
            critical_issues += 1
        
        if system_metrics.get('memory_percent', 0) > 95:
            critical_issues += 1
        
        critical_alerts = len([a for a in recent_alerts if a['level'] == 'critical'])
        
        if critical_issues > 0 or critical_alerts > 5:
            return 'critical'
        elif system_metrics.get('cpu_percent', 0) > 70 or system_metrics.get('memory_percent', 0) > 80:
            return 'warning'
        else:
            return 'healthy'
    
    def export_metrics(self, file_path: str, hours: int = 24):
        """Export metrics to JSON file.
        
        Parameters
        ----------
        file_path : str
            Path to export file
        hours : int, optional
            Hours of data to export
        """
        try:
            report = self.generate_performance_report(hours)
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance metrics exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def shutdown(self):
        """Shutdown monitor and clean up resources."""
        logger.info("Shutting down performance monitor...")
        
        self.stop_monitoring()
        
        with self._lock:
            self.system_metrics_history.clear()
            self.application_metrics_history.clear()
            self.alerts_history.clear()
            self._trend_data.clear()
            self._custom_metrics.clear()
        
        logger.info("Performance monitor shut down successfully")


# Global monitor instance
_global_monitor = None


def get_global_monitor() -> PerformanceMonitor:
    """Get global monitor instance.
    
    Returns
    -------
    PerformanceMonitor
        Global monitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def log_metric(metric_name: str, value: float):
    """Log application metric to global monitor.
    
    Parameters
    ----------
    metric_name : str
        Name of the metric
    value : float
        Metric value
    """
    get_global_monitor().log_application_metric(metric_name, value)


def add_alert_rule(rule_name: str, metric_name: str, threshold: float, level: AlertLevel = AlertLevel.WARNING):
    """Add alert rule to global monitor.
    
    Parameters
    ----------
    rule_name : str
        Name of the alert rule
    metric_name : str
        Name of the metric to monitor
    threshold : float
        Threshold value for triggering alert
    level : AlertLevel, optional
        Alert severity level
    """
    get_global_monitor().add_alert_rule(rule_name, metric_name, threshold, level)