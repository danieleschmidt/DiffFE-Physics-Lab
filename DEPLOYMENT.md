# 🚀 DiffFE-Physics-Lab Deployment Guide

This guide provides comprehensive instructions for deploying the DiffFE-Physics-Lab framework across different environments and platforms.

## 📋 Quick Start

### Docker Deployment (Recommended for Development)

```bash
# Basic deployment
python deploy.py deploy --type docker --environment production

# With monitoring stack
python deploy.py deploy --type docker --monitoring

# With GPU support
python deploy.py deploy --type docker --gpu

# Development environment
python deploy.py deploy --type docker --environment development
```

### Kubernetes Deployment (Recommended for Production)

```bash
# Production deployment
python deploy.py deploy --type kubernetes --environment production --nodes 5

# With monitoring
python deploy.py deploy --type kubernetes --monitoring --nodes 3

# Check status
python deploy.py status --type kubernetes
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │    API Gateway  │    │   Monitoring    │
│    (Nginx)      │────│   (FastAPI)     │────│  (Prometheus)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │ Compute Engine  │    │     Database    │
│   (React/Vue)   │    │  (JAX/PyTorch)  │    │  (PostgreSQL)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      Cache      │    │   File Storage  │    │     Logging     │
│     (Redis)     │    │     (S3/NFS)    │    │   (ELK Stack)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Prerequisites

### System Requirements

- **CPU**: Minimum 4 cores, recommended 8+ cores
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **Storage**: Minimum 50GB, recommended 200GB+ for data
- **Network**: High-speed internet for package installation

### Software Dependencies

#### For Docker Deployment:
- Docker Engine 20.10+
- Docker Compose 2.0+
- Python 3.10+

#### For Kubernetes Deployment:
- kubectl 1.24+
- Kubernetes cluster 1.24+
- Helm 3.8+ (optional)

#### For Cloud Deployment:
- **AWS**: AWS CLI v2, eksctl
- **Azure**: Azure CLI 2.40+
- **GCP**: Google Cloud SDK 400.0+

## 🐳 Docker Deployment

### Development Environment

```bash
# Clone repository
git clone https://github.com/danieleschmidt/DiffFE-Physics-Lab.git
cd DiffFE-Physics-Lab

# Create requirements
python deploy.py requirements

# Deploy development stack
docker-compose -f docker/docker-compose.yml up -d diffhe-dev

# Access services
echo "API: http://localhost:8080"
echo "Jupyter: http://localhost:8888"
```

### Production Environment

```bash
# Build production image
docker build -f docker/Dockerfile --target production -t diffhe-physics-lab:1.0.0 .

# Deploy production stack
docker-compose -f docker/docker-compose.yml up -d

# Verify deployment
docker-compose -f docker/docker-compose.yml ps
```

### GPU-Enabled Deployment

```bash
# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Deploy with GPU profile
docker-compose -f docker/docker-compose.yml --profile gpu up -d diffhe-gpu
```

## ☸️ Kubernetes Deployment

### Local Development (minikube)

```bash
# Start minikube with sufficient resources
minikube start --cpus 4 --memory 8192 --disk-size 50g

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server

# Deploy
kubectl apply -f kubernetes/deployment.yaml

# Port forward for testing
kubectl port-forward -n diffhe-physics-lab svc/diffhe-app-service 8000:8000
```

### Production Cluster

```bash
# Apply namespace and RBAC
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/rbac.yaml

# Apply storage classes
kubectl apply -f kubernetes/storage.yaml

# Deploy application
kubectl apply -f kubernetes/deployment.yaml

# Deploy monitoring (optional)
kubectl apply -f kubernetes/monitoring.yaml

# Configure ingress
kubectl apply -f kubernetes/ingress.yaml
```

### Scaling and Monitoring

```bash
# Scale deployment
kubectl scale deployment diffhe-app --replicas=5 -n diffhe-physics-lab

# Monitor pods
kubectl get pods -n diffhe-physics-lab -w

# Check HPA status
kubectl get hpa -n diffhe-physics-lab

# View logs
kubectl logs -f deployment/diffhe-app -n diffhe-physics-lab
```

## ☁️ Cloud Deployment

### Amazon Web Services (AWS)

#### EKS Deployment

```bash
# Deploy to EKS
python deploy.py deploy --type aws --environment production --nodes 5

# Alternative manual setup
eksctl create cluster \
  --name diffhe-physics-lab-prod \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --ssh-access \
  --ssh-public-key my-key

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name diffhe-physics-lab-prod
```

#### ECS Deployment

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name diffhe-physics-lab

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
  --cluster diffhe-physics-lab \
  --service-name diffhe-app \
  --task-definition diffhe-app:1 \
  --desired-count 3
```

### Microsoft Azure (AKS)

```bash
# Deploy to AKS
python deploy.py deploy --type azure --environment production --nodes 3

# Manual setup
az group create --name diffhe-physics-lab-rg --location eastus

az aks create \
  --resource-group diffhe-physics-lab-rg \
  --name diffhe-physics-lab-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

az aks get-credentials --resource-group diffhe-physics-lab-rg --name diffhe-physics-lab-cluster
```

### Google Cloud Platform (GKE)

```bash
# Deploy to GKE
python deploy.py deploy --type gcp --environment production --project-id my-project

# Manual setup
gcloud container clusters create diffhe-physics-lab-cluster \
  --project my-project-id \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type e2-standard-4 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

gcloud container clusters get-credentials diffhe-physics-lab-cluster \
  --zone us-central1-a \
  --project my-project-id
```

## 📊 Monitoring and Observability

### Prometheus + Grafana Stack

```bash
# Deploy monitoring stack
kubectl apply -f kubernetes/monitoring.yaml

# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000

# Default credentials: admin/admin123
```

### Custom Metrics

The application exposes metrics at `/metrics` endpoint:

- `diffhe_requests_total`: Total HTTP requests
- `diffhe_request_duration_seconds`: Request duration
- `diffhe_active_problems`: Number of active problems being solved
- `diffhe_memory_usage_bytes`: Memory usage
- `diffhe_cpu_usage_percent`: CPU usage percentage

### Logging

Logs are collected using structured JSON format:

```json
{
  "timestamp": "2025-01-09T10:00:00Z",
  "level": "INFO",
  "logger": "diffhe.solver",
  "message": "Problem solved successfully",
  "execution_time": 1.23,
  "problem_id": "prob_123"
}
```

## 🔧 Configuration Management

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DIFFHE_ENV` | `production` | Environment (development/staging/production) |
| `DIFFHE_LOG_LEVEL` | `INFO` | Logging level |
| `DIFFHE_BACKEND` | `numpy` | Compute backend (numpy/jax/torch) |
| `DIFFHE_MAX_WORKERS` | `4` | Maximum worker processes |
| `DIFFHE_CACHE_SIZE` | `1000` | Cache size |
| `DIFFHE_DB_URL` | - | Database connection URL |
| `DIFFHE_REDIS_URL` | - | Redis connection URL |

### Configuration Files

- `config/production.yaml`: Production configuration
- `config/development.yaml`: Development configuration  
- `config/staging.yaml`: Staging configuration

### Secrets Management

#### Kubernetes Secrets

```bash
# Create secrets
kubectl create secret generic diffhe-secrets \
  --from-literal=postgres-password=mysecretpassword \
  --from-literal=redis-password=mysecretredis \
  --from-literal=jwt-secret=mysecretjwt \
  -n diffhe-physics-lab
```

#### Docker Secrets

```bash
# Using Docker secrets
echo "mysecretpassword" | docker secret create postgres_password -
echo "mysecretredis" | docker secret create redis_password -
```

## 🛡️ Security

### Network Security

- All services communicate over encrypted channels
- Network policies restrict inter-pod communication
- Ingress configured with TLS termination
- Rate limiting enabled on all endpoints

### Container Security

- Non-root user execution
- Read-only root filesystem
- Capability dropping
- Resource limits enforced

### Authentication & Authorization

- JWT-based authentication
- Role-based access control (RBAC)
- API key management for external integrations

## 📈 Performance Tuning

### Resource Allocation

#### Development Environment
- CPU: 2 cores
- Memory: 4GB RAM
- Storage: 20GB

#### Production Environment
- CPU: 8+ cores
- Memory: 16GB+ RAM  
- Storage: 200GB+ SSD

### Scaling Guidelines

#### Horizontal Scaling
- Target: 70% CPU utilization
- Min replicas: 2
- Max replicas: 10
- Scale up: 50% every 60s
- Scale down: 25% every 180s

#### Vertical Scaling
- Monitor memory usage patterns
- Adjust resource requests/limits based on workload
- Use VPA (Vertical Pod Autoscaler) for automatic sizing

### Database Optimization

```sql
-- Recommended PostgreSQL settings
shared_preload_libraries = 'pg_stat_statements'
max_connections = 100
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

```yaml
name: Deploy DiffFE-Physics-Lab
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run tests
      run: python run_tests_standalone.py
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: docker build -t diffhe-physics-lab:${{ github.sha }} .
    
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: python deploy.py deploy --type kubernetes --environment production
```

## 🚨 Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod status
kubectl describe pod <pod-name> -n diffhe-physics-lab

# Check logs
kubectl logs <pod-name> -n diffhe-physics-lab

# Check resource usage
kubectl top pod <pod-name> -n diffhe-physics-lab
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- bash
psql -h diffhe-postgres -U diffhe -d diffhe_physics_lab
```

#### Performance Issues
```bash
# Check HPA status
kubectl get hpa -n diffhe-physics-lab

# Monitor resource usage
kubectl top pods -n diffhe-physics-lab

# Check application metrics
curl http://localhost:8000/metrics
```

### Log Analysis

```bash
# Get recent logs
kubectl logs --since=1h deployment/diffhe-app -n diffhe-physics-lab

# Follow logs in real-time
kubectl logs -f deployment/diffhe-app -n diffhe-physics-lab

# Search for errors
kubectl logs deployment/diffhe-app -n diffhe-physics-lab | grep ERROR
```

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Framework API Documentation](./docs/api.md)
- [Performance Tuning Guide](./docs/performance.md)

## 🆘 Support

For deployment issues and support:

- 📧 Email: support@diffhe-physics.org
- 🐛 Issues: [GitHub Issues](https://github.com/danieleschmidt/DiffFE-Physics-Lab/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/danieleschmidt/DiffFE-Physics-Lab/discussions)
- 📖 Documentation: [ReadTheDocs](https://diffhe-physics-lab.readthedocs.io/)

---

*Last updated: January 9, 2025*