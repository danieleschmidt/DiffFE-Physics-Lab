# Production Deployment Guide

## Overview

This guide covers deploying the DiffFE Sentiment Analysis Pro framework to production environments with comprehensive security, monitoring, and scaling considerations.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Load Balancer │────▶│  API Gateway    │────▶│ Application     │
│   (nginx/traefik)│     │  (Rate Limiting)│     │ Servers         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                 │                        │
                                 ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   Security      │     │    Cache        │
                        │   Monitor       │     │   (Redis)       │
                        └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │   Database      │
                                                │  (PostgreSQL)   │
                                                └─────────────────┘
```

## Prerequisites

### System Requirements

- **CPU**: 4+ cores (8+ recommended for production)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 50GB+ SSD storage
- **Network**: 1Gbps+ bandwidth
- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+

### Software Dependencies

- Python 3.10+
- Docker & Docker Compose
- nginx (for reverse proxy)
- Redis (for caching)
- PostgreSQL (for persistent data)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/danieleschmidt/sentiment-analyzer-pro.git
cd sentiment-analyzer-pro
```

### 2. Environment Setup

Create production environment file:

```bash
cp .env.example .env.production
```

Edit `.env.production`:

```env
# Application
ENVIRONMENT=production
DEBUG=False
SECRET_KEY=your-super-secret-key-here
FLASK_APP=src.api.app:create_app

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/sentiment_db

# Cache
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=your-jwt-secret-key
ENCRYPTION_KEY=your-encryption-key

# Performance
MAX_WORKERS=4
CACHE_TTL=3600
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
METRICS_ENABLED=True

# External Services
EMBEDDING_SERVICE_URL=https://api.embeddings.com
GPU_ACCELERATION=True
```

### 3. Docker Deployment

#### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    env_file:
      - .env.production
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/app/logs
      - ./cache:/app/cache
    restart: always
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: always

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: always
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: sentiment_db
      POSTGRES_USER: sentiment_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: always

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    restart: always

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin_password
    volumes:
      - grafana_data:/var/lib/grafana
    restart: always

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

#### Dockerfile.prod

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["gunicorn", "--config", "gunicorn.conf.py", "src.api.app:create_app()"]
```

### 4. nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    server {
        listen 80;
        server_name sentiment-api.example.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name sentiment-api.example.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 300s;
        }

        # Health check
        location /health {
            proxy_pass http://app/health;
            access_log off;
        }

        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

### 5. Gunicorn Configuration

Create `gunicorn.conf.py`:

```python
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = int(os.environ.get('MAX_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "gevent"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# Timeout
timeout = 30
keepalive = 2

# Logging
accesslog = "/app/logs/access.log"
errorlog = "/app/logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'sentiment-api'

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Performance
worker_tmp_dir = '/dev/shm'
```

## Security Configuration

### 1. SSL/TLS Setup

Generate SSL certificates:

```bash
# Using Let's Encrypt
sudo certbot certonly --webroot -w /var/www/html -d sentiment-api.example.com

# Or self-signed for testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/key.pem -out ssl/cert.pem
```

### 2. Firewall Configuration

```bash
# Ubuntu UFW
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS

# Block direct access to application ports
sudo ufw deny 8000/tcp
sudo ufw deny 6379/tcp
sudo ufw deny 5432/tcp
```

### 3. Environment Security

Set proper file permissions:

```bash
chmod 600 .env.production
chmod 600 ssl/key.pem
chmod 644 ssl/cert.pem

# Create secure directories
mkdir -p logs cache
chmod 755 logs cache
chown app:app logs cache
```

### 4. Database Security

PostgreSQL security configuration:

```sql
-- Create database and user
CREATE DATABASE sentiment_db;
CREATE USER sentiment_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE sentiment_db TO sentiment_user;

-- Security settings
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
SELECT pg_reload_conf();
```

## Monitoring & Logging

### 1. Application Logs

Configure structured logging in `src/utils/logging.py`:

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
            
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/application.log'),
        logging.StreamHandler()
    ]
)

# Add JSON formatter to file handler
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setFormatter(JSONFormatter())
```

### 2. Prometheus Metrics

Configure `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'sentiment-api'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
```

### 3. Grafana Dashboards

Import pre-built dashboards for:
- Application performance metrics
- API response times and error rates
- Resource utilization (CPU, memory, disk)
- Security threat monitoring
- Cache hit rates and performance

## Scaling

### 1. Horizontal Scaling

Scale application containers:

```bash
docker-compose -f docker-compose.prod.yml up --scale app=4
```

Update nginx upstream configuration:

```nginx
upstream app {
    server app_1:8000;
    server app_2:8000;
    server app_3:8000;
    server app_4:8000;
}
```

### 2. Database Scaling

Configure read replicas:

```yaml
# Add to docker-compose.prod.yml
postgres-replica:
  image: postgres:15-alpine
  environment:
    POSTGRES_DB: sentiment_db
    POSTGRES_USER: sentiment_user
    POSTGRES_PASSWORD: secure_password
    POSTGRES_MASTER_SERVICE: postgres
  command: |
    bash -c '
    until pg_basebackup -h postgres -D /var/lib/postgresql/data -U replicator -v -P; do
      echo "Waiting for master to connect..."
      sleep 1s
    done
    echo "standby_mode = on" >> /var/lib/postgresql/data/recovery.conf
    echo "primary_conninfo = \"host=postgres port=5432 user=replicator\"" >> /var/lib/postgresql/data/recovery.conf
    postgres
    '
```

### 3. Cache Scaling

Redis Cluster configuration:

```yaml
redis-cluster:
  image: redis:7-alpine
  command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 --appendonly yes
  ports:
    - "7000-7005:7000-7005"
```

### 4. Load Testing

Run performance tests:

```bash
# Install load testing tools
pip install locust

# Run load test
locust -f tests/load_test.py --host=https://sentiment-api.example.com
```

Example load test script:

```python
from locust import HttpUser, task, between

class SentimentAnalysisUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Setup
        pass
    
    @task(3)
    def analyze_sentiment(self):
        payload = {
            "texts": ["This is a test sentence for sentiment analysis."],
            "options": {
                "embedding_method": "tfidf",
                "return_diagnostics": False
            }
        }
        self.client.post("/api/v1/sentiment/analyze", json=payload)
    
    @task(1)
    def health_check(self):
        self.client.get("/health")
```

## Backup & Recovery

### 1. Database Backups

Automated backup script:

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="sentiment_db"

# Create backup
pg_dump -h localhost -U sentiment_user -d $DB_NAME -f "${BACKUP_DIR}/sentiment_db_${DATE}.sql"

# Compress backup
gzip "${BACKUP_DIR}/sentiment_db_${DATE}.sql"

# Remove backups older than 30 days
find $BACKUP_DIR -name "sentiment_db_*.sql.gz" -mtime +30 -delete

echo "Backup completed: sentiment_db_${DATE}.sql.gz"
```

### 2. Application Data Backup

```bash
#!/bin/bash
# app_backup.sh

BACKUP_DIR="/backups/app"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup cache data
tar -czf "${BACKUP_DIR}/cache_${DATE}.tar.gz" /app/cache/

# Backup logs
tar -czf "${BACKUP_DIR}/logs_${DATE}.tar.gz" /app/logs/

# Backup configuration
tar -czf "${BACKUP_DIR}/config_${DATE}.tar.gz" /app/config/
```

### 3. Automated Backup Schedule

Add to crontab:

```bash
# Database backup daily at 2 AM
0 2 * * * /home/app/backup.sh

# Application backup weekly at 3 AM Sunday
0 3 * * 0 /home/app/app_backup.sh
```

## Health Checks & Alerting

### 1. Health Check Endpoint

Implement comprehensive health checks:

```python
@app.route('/health')
def health_check():
    checks = {
        'database': check_database_connection(),
        'cache': check_redis_connection(),
        'external_services': check_external_services(),
        'disk_space': check_disk_space(),
        'memory': check_memory_usage()
    }
    
    overall_status = 'healthy' if all(checks.values()) else 'unhealthy'
    
    return jsonify({
        'status': overall_status,
        'timestamp': datetime.utcnow().isoformat(),
        'checks': checks
    }), 200 if overall_status == 'healthy' else 503
```

### 2. Alerting Configuration

Configure alerts for:
- High error rates (>5%)
- High response times (>1s)
- Memory usage (>80%)
- Disk usage (>85%)
- Database connection failures
- Security threat spikes

### 3. Log Aggregation

Set up centralized logging with ELK stack:

```yaml
# Add to docker-compose.prod.yml
elasticsearch:
  image: elasticsearch:8.8.0
  environment:
    - discovery.type=single-node
    - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
  ports:
    - "9200:9200"

logstash:
  image: logstash:8.8.0
  volumes:
    - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf:ro
    - ./logs:/logs:ro

kibana:
  image: kibana:8.8.0
  ports:
    - "5601:5601"
  environment:
    ELASTICSEARCH_HOSTS: http://elasticsearch:9200
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   
   # Optimize cache settings
   redis-cli CONFIG SET maxmemory 512mb
   redis-cli CONFIG SET maxmemory-policy allkeys-lru
   ```

2. **Database Connection Issues**
   ```bash
   # Check PostgreSQL connections
   docker exec -it postgres psql -U sentiment_user -d sentiment_db -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Increase connection limit
   docker exec -it postgres psql -U postgres -c "ALTER SYSTEM SET max_connections = 200;"
   ```

3. **SSL Certificate Issues**
   ```bash
   # Renew Let's Encrypt certificates
   sudo certbot renew
   
   # Reload nginx
   docker exec nginx nginx -s reload
   ```

### Performance Optimization

1. **Database Optimization**
   ```sql
   -- Add indexes for common queries
   CREATE INDEX idx_sentiment_timestamp ON sentiment_results(timestamp);
   CREATE INDEX idx_sentiment_user_id ON sentiment_results(user_id);
   
   -- Analyze query performance
   EXPLAIN ANALYZE SELECT * FROM sentiment_results WHERE timestamp > '2024-01-01';
   ```

2. **Cache Optimization**
   ```bash
   # Monitor cache performance
   redis-cli INFO stats
   
   # Adjust cache settings
   redis-cli CONFIG SET timeout 300
   ```

3. **Application Optimization**
   - Enable gzip compression
   - Optimize embedding cache hit rates
   - Use connection pooling
   - Implement request deduplication

## Security Checklist

- [ ] SSL/TLS certificates installed and configured
- [ ] Firewall rules configured properly
- [ ] Database access restricted
- [ ] Application secrets secured
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Input validation implemented
- [ ] Security monitoring active
- [ ] Regular security updates scheduled
- [ ] Backup and recovery tested

## Maintenance

### Regular Tasks

- **Daily**: Monitor logs and alerts
- **Weekly**: Review security reports and update dependencies
- **Monthly**: Performance analysis and optimization
- **Quarterly**: Full security audit and penetration testing

### Update Procedures

1. Test updates in staging environment
2. Schedule maintenance window
3. Create backup before deployment
4. Deploy updates with zero-downtime strategy
5. Monitor system health post-deployment
6. Rollback if issues detected

This deployment guide provides a comprehensive foundation for running the sentiment analysis framework in production with enterprise-grade security, monitoring, and scalability.