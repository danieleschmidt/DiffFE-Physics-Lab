"""Real-time performance dashboard and monitoring system."""

import asyncio
import json
import time
import threading
import logging
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
from collections import deque, defaultdict

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WEBSOCKETS_AVAILABLE = False

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    Flask = None
    SocketIO = None
    FLASK_AVAILABLE = False

try:
    import plotly
    import plotly.graph_objs as go
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    plotly = None
    go = None
    PlotlyJSONEncoder = None
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetric:
    """Real-time dashboard metric."""
    name: str
    value: float
    unit: str
    timestamp: float
    category: str = "general"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AlertEvent:
    """Dashboard alert event."""
    id: str
    type: str
    severity: str  # info, warning, error, critical
    message: str
    timestamp: float
    source: str
    metadata: Dict[str, Any] = None
    resolved: bool = False
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MetricsCollector:
    """Collect and aggregate metrics for dashboard."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics_store = defaultdict(deque)  # metric_name -> deque of values
        self.aggregated_metrics = {}
        self.metric_callbacks = {}  # metric_name -> callback function
        self.collection_lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Metrics collector initialized (retention: {retention_hours}h)")
    
    def record_metric(self, metric: DashboardMetric):
        """Record a new metric value."""
        with self.collection_lock:
            # Store metric
            self.metrics_store[metric.name].append({
                'value': metric.value,
                'timestamp': metric.timestamp,
                'unit': metric.unit,
                'category': metric.category,
                'metadata': metric.metadata
            })
            
            # Trigger callback if registered
            if metric.name in self.metric_callbacks:
                try:
                    self.metric_callbacks[metric.name](metric)
                except Exception as e:
                    logger.error(f"Metric callback error for {metric.name}: {e}")
            
            # Update aggregated metrics
            self._update_aggregated_metrics(metric.name)
    
    def _update_aggregated_metrics(self, metric_name: str):
        """Update aggregated metrics for a given metric."""
        if metric_name not in self.metrics_store:
            return
        
        values = [m['value'] for m in self.metrics_store[metric_name]]
        if not values:
            return
        
        recent_values = values[-100:]  # Last 100 data points
        
        self.aggregated_metrics[metric_name] = {
            'current': values[-1],
            'avg': sum(recent_values) / len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'count': len(values),
            'trend': self._calculate_trend(values[-20:]) if len(values) >= 20 else 'stable'
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[Dict]:
        """Get metric history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.collection_lock:
            if metric_name not in self.metrics_store:
                return []
            
            return [
                m for m in self.metrics_store[metric_name]
                if m['timestamp'] > cutoff_time
            ]
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        with self.collection_lock:
            return {
                'total_metrics': len(self.metrics_store),
                'aggregated_metrics': dict(self.aggregated_metrics),
                'metric_names': list(self.metrics_store.keys())
            }
    
    def register_metric_callback(self, metric_name: str, callback: Callable):
        """Register callback for metric updates."""
        self.metric_callbacks[metric_name] = callback
        logger.debug(f"Registered callback for metric: {metric_name}")
    
    def _cleanup_loop(self):
        """Cleanup old metrics data."""
        while True:
            try:
                cutoff_time = time.time() - (self.retention_hours * 3600)
                
                with self.collection_lock:
                    for metric_name, metric_deque in self.metrics_store.items():
                        # Remove old entries
                        while (metric_deque and 
                               metric_deque[0]['timestamp'] < cutoff_time):
                            metric_deque.popleft()
                
                time.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
                time.sleep(3600)


class AlertManager:
    """Manage dashboard alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self.active_alerts = {}  # alert_id -> AlertEvent
        self.alert_history = deque(maxlen=max_alerts)
        self.alert_rules = {}  # rule_name -> rule_config
        self.subscribers = set()  # WebSocket connections or callbacks
        self.alert_lock = threading.RLock()
        
        logger.info("Alert manager initialized")
    
    def add_alert_rule(
        self,
        rule_name: str,
        condition: Callable[[DashboardMetric], bool],
        severity: str = "warning",
        message_template: str = "Alert: {metric_name} = {value}"
    ):
        """Add alert rule."""
        self.alert_rules[rule_name] = {
            'condition': condition,
            'severity': severity,
            'message_template': message_template,
            'enabled': True
        }
        logger.info(f"Added alert rule: {rule_name}")
    
    def check_metric_alerts(self, metric: DashboardMetric):
        """Check metric against alert rules."""
        for rule_name, rule in self.alert_rules.items():
            if not rule['enabled']:
                continue
            
            try:
                if rule['condition'](metric):
                    self._trigger_alert(
                        alert_type=f"metric_{rule_name}",
                        severity=rule['severity'],
                        message=rule['message_template'].format(
                            metric_name=metric.name,
                            value=metric.value,
                            unit=metric.unit
                        ),
                        source=f"metric:{metric.name}",
                        metadata={'metric': asdict(metric), 'rule': rule_name}
                    )
            except Exception as e:
                logger.error(f"Alert rule {rule_name} evaluation error: {e}")
    
    def _trigger_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        source: str,
        metadata: Dict = None
    ):
        """Trigger new alert."""
        alert_id = str(uuid.uuid4())
        
        alert = AlertEvent(
            id=alert_id,
            type=alert_type,
            severity=severity,
            message=message,
            timestamp=time.time(),
            source=source,
            metadata=metadata or {}
        )
        
        with self.alert_lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
        
        # Notify subscribers
        self._notify_subscribers(alert)
        
        logger.warning(f"Alert triggered: {message}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        with self.alert_lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert_id}")
                return True
            return False
    
    def get_active_alerts(self) -> List[AlertEvent]:
        """Get all active alerts."""
        with self.alert_lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[AlertEvent]:
        """Get alert history."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.alert_lock:
            return [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_time
            ]
    
    def subscribe_to_alerts(self, callback: Callable[[AlertEvent], None]):
        """Subscribe to alert notifications."""
        self.subscribers.add(callback)
    
    def unsubscribe_from_alerts(self, callback: Callable):
        """Unsubscribe from alert notifications."""
        self.subscribers.discard(callback)
    
    def _notify_subscribers(self, alert: AlertEvent):
        """Notify all subscribers of new alert."""
        for subscriber in self.subscribers.copy():
            try:
                subscriber(alert)
            except Exception as e:
                logger.error(f"Alert notification error: {e}")
                self.subscribers.discard(subscriber)


class WebDashboard:
    """Web-based performance dashboard."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        debug: bool = False
    ):
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask not available. Install flask and flask-socketio")
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder=None,
                        static_folder=None)
        self.app.config['SECRET_KEY'] = 'dashboard_secret_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # Connected clients
        self.connected_clients = set()
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        
        # Register alert callback for real-time updates
        self.alert_manager.subscribe_to_alerts(self._broadcast_alert)
        
        # Register metrics callback
        self.metrics_collector.register_metric_callback('*', self._broadcast_metric)
        
        logger.info(f"Web dashboard initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return self._get_dashboard_html()
        
        @self.app.route('/api/metrics')
        def get_metrics():
            return jsonify(self.metrics_collector.get_all_metrics_summary())
        
        @self.app.route('/api/metrics/<metric_name>')
        def get_metric_history(metric_name):
            hours = request.args.get('hours', 1, type=int)
            history = self.metrics_collector.get_metric_history(metric_name, hours)
            return jsonify(history)
        
        @self.app.route('/api/alerts')
        def get_alerts():
            active_alerts = [asdict(alert) for alert in self.alert_manager.get_active_alerts()]
            return jsonify(active_alerts)
        
        @self.app.route('/api/alerts/history')
        def get_alert_history():
            hours = request.args.get('hours', 24, type=int)
            history = [asdict(alert) for alert in self.alert_manager.get_alert_history(hours)]
            return jsonify(history)
        
        @self.app.route('/api/charts/<metric_name>')
        def get_metric_chart(metric_name):
            if not PLOTLY_AVAILABLE:
                return jsonify({'error': 'Plotly not available'})
            
            hours = request.args.get('hours', 1, type=int)
            history = self.metrics_collector.get_metric_history(metric_name, hours)
            
            if not history:
                return jsonify({'error': 'No data available'})
            
            # Create Plotly chart
            timestamps = [datetime.fromtimestamp(m['timestamp']) for m in history]
            values = [m['value'] for m in history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name=metric_name,
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title=f'{metric_name} - Last {hours} Hours',
                xaxis_title='Time',
                yaxis_title=history[0]['unit'] if history else 'Value',
                template='plotly_white'
            )
            
            return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _setup_socket_handlers(self):
        """Setup Socket.IO handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.connected_clients.add(request.sid)
            logger.debug(f"Client connected: {request.sid}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.connected_clients.discard(request.sid)
            logger.debug(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_metrics')
        def handle_subscribe_metrics(data):
            # Send current metrics summary
            summary = self.metrics_collector.get_all_metrics_summary()
            emit('metrics_update', summary)
        
        @self.socketio.on('resolve_alert')
        def handle_resolve_alert(data):
            alert_id = data.get('alert_id')
            if alert_id:
                success = self.alert_manager.resolve_alert(alert_id)
                emit('alert_resolved', {'alert_id': alert_id, 'success': success})
    
    def _broadcast_metric(self, metric: DashboardMetric):
        """Broadcast metric update to connected clients."""
        if self.connected_clients:
            self.socketio.emit('metric_update', {
                'name': metric.name,
                'value': metric.value,
                'unit': metric.unit,
                'timestamp': metric.timestamp,
                'category': metric.category
            })
        
        # Check alerts
        self.alert_manager.check_metric_alerts(metric)
    
    def _broadcast_alert(self, alert: AlertEvent):
        """Broadcast alert to connected clients."""
        if self.connected_clients:
            self.socketio.emit('new_alert', asdict(alert))
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Performance Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .widget { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }
        .metric:last-child { border-bottom: none; }
        .metric-value { font-weight: bold; color: #007bff; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
        .alert-critical { background: #f8d7da; border-left: 4px solid #dc3545; }
        .alert-warning { background: #fff3cd; border-left: 4px solid #ffc107; }
        .alert-info { background: #d1ecf1; border-left: 4px solid #17a2b8; }
        .chart-container { height: 400px; }
        h2 { margin-top: 0; color: #333; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
    </style>
</head>
<body>
    <h1>Performance Dashboard</h1>
    
    <div class="dashboard">
        <div class="widget">
            <h2>System Metrics</h2>
            <div id="metrics-list"></div>
        </div>
        
        <div class="widget">
            <h2>Active Alerts</h2>
            <div id="alerts-list"></div>
        </div>
        
        <div class="widget">
            <h2>CPU Usage</h2>
            <div id="cpu-chart" class="chart-container"></div>
        </div>
        
        <div class="widget">
            <h2>Memory Usage</h2>
            <div id="memory-chart" class="chart-container"></div>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Connect to dashboard
        socket.on('connect', function() {
            console.log('Connected to dashboard');
            socket.emit('subscribe_metrics');
        });
        
        // Handle metrics updates
        socket.on('metrics_update', function(data) {
            updateMetrics(data);
        });
        
        socket.on('metric_update', function(metric) {
            updateSingleMetric(metric);
        });
        
        // Handle alerts
        socket.on('new_alert', function(alert) {
            addAlert(alert);
        });
        
        function updateMetrics(data) {
            const metricsList = document.getElementById('metrics-list');
            metricsList.innerHTML = '';
            
            Object.entries(data.aggregated_metrics || {}).forEach(([name, stats]) => {
                const div = document.createElement('div');
                div.className = 'metric';
                div.innerHTML = `
                    <span>${name}</span>
                    <span class="metric-value">${stats.current?.toFixed(2) || 'N/A'}</span>
                `;
                metricsList.appendChild(div);
            });
        }
        
        function updateSingleMetric(metric) {
            // Update individual metric display
            console.log('Metric update:', metric.name, metric.value);
        }
        
        function addAlert(alert) {
            const alertsList = document.getElementById('alerts-list');
            const div = document.createElement('div');
            div.className = `alert alert-${alert.severity}`;
            div.innerHTML = `
                <strong>${alert.type}</strong>: ${alert.message}
                <small style="float: right;">${new Date(alert.timestamp * 1000).toLocaleTimeString()}</small>
            `;
            alertsList.insertBefore(div, alertsList.firstChild);
            
            // Keep only last 10 alerts
            while (alertsList.children.length > 10) {
                alertsList.removeChild(alertsList.lastChild);
            }
        }
        
        // Load initial data
        fetch('/api/alerts')
            .then(response => response.json())
            .then(alerts => {
                alerts.forEach(alert => addAlert(alert));
            });
        
        // Load charts
        loadChart('cpu_percent', 'cpu-chart');
        loadChart('memory_percent', 'memory-chart');
        
        function loadChart(metricName, containerId) {
            fetch(`/api/charts/${metricName}?hours=1`)
                .then(response => response.json())
                .then(chartData => {
                    if (chartData.error) {
                        document.getElementById(containerId).innerHTML = 
                            `<p>No data available for ${metricName}</p>`;
                    } else {
                        Plotly.newPlot(containerId, chartData.data, chartData.layout);
                    }
                })
                .catch(error => {
                    console.error(`Failed to load chart for ${metricName}:`, error);
                });
        }
        
        // Auto-refresh charts every 30 seconds
        setInterval(() => {
            loadChart('cpu_percent', 'cpu-chart');
            loadChart('memory_percent', 'memory-chart');
        }, 30000);
    </script>
</body>
</html>
"""
    
    def record_metric(self, name: str, value: float, unit: str = "", category: str = "general"):
        """Record a metric for dashboard display."""
        metric = DashboardMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            category=category
        )
        self.metrics_collector.record_metric(metric)
    
    def add_alert_rule(self, rule_name: str, condition: Callable, severity: str = "warning"):
        """Add alert rule to dashboard."""
        self.alert_manager.add_alert_rule(rule_name, condition, severity)
    
    def run(self):
        """Run the dashboard server."""
        logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug)
    
    def run_async(self):
        """Run dashboard in background thread."""
        dashboard_thread = threading.Thread(
            target=self.run,
            daemon=True
        )
        dashboard_thread.start()
        logger.info("Dashboard started in background thread")
        return dashboard_thread


# Global dashboard instance
_global_dashboard = None


def get_dashboard(host: str = "0.0.0.0", port: int = 8080) -> WebDashboard:
    """Get global dashboard instance."""
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = WebDashboard(host=host, port=port)
    return _global_dashboard


def record_dashboard_metric(name: str, value: float, unit: str = "", category: str = "general"):
    """Record metric to global dashboard."""
    dashboard = get_dashboard()
    dashboard.record_metric(name, value, unit, category)


def add_dashboard_alert_rule(rule_name: str, condition: Callable, severity: str = "warning"):
    """Add alert rule to global dashboard."""
    dashboard = get_dashboard()
    dashboard.add_alert_rule(rule_name, condition, severity)


def start_dashboard(host: str = "0.0.0.0", port: int = 8080, background: bool = True):
    """Start the performance dashboard."""
    dashboard = get_dashboard(host, port)
    
    # Add some default alert rules
    dashboard.add_alert_rule(
        "high_cpu",
        lambda m: m.name == "cpu_percent" and m.value > 80,
        "warning"
    )
    
    dashboard.add_alert_rule(
        "high_memory",
        lambda m: m.name == "memory_percent" and m.value > 90,
        "critical"
    )
    
    if background:
        return dashboard.run_async()
    else:
        dashboard.run()
        return None