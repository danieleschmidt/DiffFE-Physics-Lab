#!/usr/bin/env python3
"""
Gunicorn configuration for Physics-Informed Sentiment Analyzer production deployment.
Optimized for high-performance, concurrent request handling.
"""

import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WEB_CONCURRENCY', multiprocessing.cpu_count() * 2 + 1))
worker_class = "gevent"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings
timeout = 30
keepalive = 60
graceful_timeout = 30

# Logging
access_logfile = "-"
error_logfile = "-"
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'physics-sentiment-analyzer'

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Server mechanics
preload_app = True
daemon = False
pidfile = '/tmp/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = os.getenv('SSL_KEYFILE')
certfile = os.getenv('SSL_CERTFILE')

# Worker process configuration
worker_tmp_dir = '/dev/shm'  # Use memory for worker tmp files

# Application specific settings
raw_env = [
    'DIFFHE_ENV=production',
    f'DIFFHE_WORKERS={workers}',
    'PYTHONPATH=/app',
]

# Hooks for graceful startup/shutdown
def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Physics-Informed Sentiment Analyzer server is ready. Accepting connections.")
    
def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info("Worker received SIGINT or SIGQUIT. Shutting down gracefully.")

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    worker.log.info(f"Worker pid {worker.pid} was aborted")

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, re-executing.")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.debug(f"{req.method} {req.path}")

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    pass

# Memory optimization
def max_memory_per_child():
    """Set maximum memory per worker to prevent memory leaks."""
    return int(os.getenv('MAX_MEMORY_PER_WORKER', 1024 * 1024 * 1024))  # 1GB default

# Custom application configuration
def application(environ, start_response):
    """WSGI application callable."""
    # This is handled by Flask app factory
    pass

# Environment-specific overrides
env = os.getenv('DIFFHE_ENV', 'production')

if env == 'development':
    workers = 1
    reload = True
    loglevel = 'debug'
elif env == 'testing':
    workers = 2
    timeout = 60
elif env == 'production':
    # Production optimizations
    preload_app = True
    
    # Performance tuning
    worker_connections = min(1000, int(os.getenv('MAX_WORKER_CONNECTIONS', 1000)))
    
    # Security hardening
    forwarded_allow_ips = os.getenv('FORWARDED_ALLOW_IPS', '127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16')
    proxy_allow_ips = forwarded_allow_ips
    
    # Enable access log in production
    access_logfile = os.getenv('ACCESS_LOG_FILE', '-')
    
# Auto-scaling based on load
def auto_scale_workers():
    """Dynamically adjust worker count based on CPU usage."""
    try:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=1)
        
        if cpu_usage > 80:
            # Increase workers if CPU usage is high
            return min(workers + 1, multiprocessing.cpu_count() * 4)
        elif cpu_usage < 30 and workers > 1:
            # Decrease workers if CPU usage is low
            return max(workers - 1, 1)
    except ImportError:
        pass
    
    return workers

# Health check endpoint integration
def health_check(environ, start_response):
    """Built-in health check for load balancers."""
    if environ['PATH_INFO'] == '/health':
        status = '200 OK'
        headers = [('Content-Type', 'application/json')]
        start_response(status, headers)
        return [b'{"status": "healthy", "server": "gunicorn"}']

# Graceful shutdown handling
def on_exit(server):
    """Called just before the master process exits."""
    server.log.info("Physics-Informed Sentiment Analyzer server shutting down gracefully.")

# Error handling
def on_reload(server):
    """Called when configuration is reloaded."""
    server.log.info("Configuration reloaded.")

# Development vs Production settings
if os.getenv('FLASK_ENV') == 'development':
    reload = True
    loglevel = 'debug'
    workers = 1
else:
    reload = False
    
# Resource limits (prevent memory leaks)
rlimit_nofile = 65535  # Maximum number of open files
rlimit_core = 0        # Core dump size limit

# Statistics and monitoring
statsd_host = os.getenv('STATSD_HOST')
statsd_prefix = os.getenv('STATSD_PREFIX', 'sentiment_analyzer')

print(f"Gunicorn configured with {workers} workers for {env} environment")