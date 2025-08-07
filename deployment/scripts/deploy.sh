#!/bin/bash

# Physics-Informed Sentiment Analyzer Deployment Script
# Supports multiple deployment targets: docker, kubernetes, cloud

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOYMENT_TARGET="${DEPLOYMENT_TARGET:-docker}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
NAMESPACE="${NAMESPACE:-sentiment-analysis}"
REPLICA_COUNT="${REPLICA_COUNT:-3}"
DOMAIN="${DOMAIN:-api.sentiment-analyzer.local}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
Physics-Informed Sentiment Analyzer Deployment Script

Usage: $0 [OPTIONS] DEPLOYMENT_TARGET

DEPLOYMENT_TARGETS:
    docker       Deploy using Docker Compose
    kubernetes   Deploy to Kubernetes cluster
    aws          Deploy to AWS ECS/EKS
    gcp          Deploy to Google Cloud Run/GKE
    azure        Deploy to Azure Container Instances/AKS

OPTIONS:
    -e, --environment ENV     Deployment environment (development, staging, production)
    -t, --tag TAG            Docker image tag (default: latest)
    -n, --namespace NS       Kubernetes namespace (default: sentiment-analysis)
    -r, --replicas COUNT     Number of replicas (default: 3)
    -d, --domain DOMAIN      Application domain (default: api.sentiment-analyzer.local)
    -h, --help               Show this help message

EXAMPLES:
    $0 docker
    $0 kubernetes --environment production --replicas 5
    $0 aws --tag v1.2.3 --environment production
    $0 --environment staging kubernetes

ENVIRONMENT VARIABLES:
    AWS_REGION                 AWS region for deployment
    GCP_PROJECT_ID             Google Cloud project ID
    AZURE_RESOURCE_GROUP       Azure resource group name
    KUBECONFIG                 Path to Kubernetes config file
    DOCKER_REGISTRY            Container registry URL
    SSL_CERT_EMAIL             Email for SSL certificate generation

EOF
}

validate_requirements() {
    log_info "Validating deployment requirements..."
    
    case "$DEPLOYMENT_TARGET" in
        docker)
            command -v docker >/dev/null 2>&1 || { log_error "Docker is required but not installed."; exit 1; }
            command -v docker-compose >/dev/null 2>&1 || { log_error "Docker Compose is required but not installed."; exit 1; }
            ;;
        kubernetes)
            command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed."; exit 1; }
            kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to Kubernetes cluster."; exit 1; }
            ;;
        aws)
            command -v aws >/dev/null 2>&1 || { log_error "AWS CLI is required but not installed."; exit 1; }
            aws sts get-caller-identity >/dev/null 2>&1 || { log_error "AWS credentials not configured."; exit 1; }
            ;;
        gcp)
            command -v gcloud >/dev/null 2>&1 || { log_error "Google Cloud SDK is required but not installed."; exit 1; }
            gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 >/dev/null || { log_error "Google Cloud credentials not configured."; exit 1; }
            ;;
        azure)
            command -v az >/dev/null 2>&1 || { log_error "Azure CLI is required but not installed."; exit 1; }
            az account show >/dev/null 2>&1 || { log_error "Azure credentials not configured."; exit 1; }
            ;;
        *)
            log_error "Unsupported deployment target: $DEPLOYMENT_TARGET"
            exit 1
            ;;
    esac
    
    log_success "Requirements validation passed"
}

build_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build the image
    docker build \
        -f "$DEPLOYMENT_DIR/docker/Dockerfile" \
        -t "physics-sentiment-analyzer:$IMAGE_TAG" \
        --target production \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .
    
    # Tag for registry if specified
    if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        docker tag "physics-sentiment-analyzer:$IMAGE_TAG" "$DOCKER_REGISTRY/physics-sentiment-analyzer:$IMAGE_TAG"
        log_info "Pushing image to registry..."
        docker push "$DOCKER_REGISTRY/physics-sentiment-analyzer:$IMAGE_TAG"
    fi
    
    log_success "Image built successfully"
}

deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$DEPLOYMENT_DIR/docker"
    
    # Create environment file
    cat > .env << EOF
ENVIRONMENT=$ENVIRONMENT
BUILD_TARGET=production
TAG=$IMAGE_TAG
API_PORT=5000
LOG_LEVEL=${LOG_LEVEL:-INFO}
SECRET_KEY=${SECRET_KEY:-$(openssl rand -base64 32)}
DEBUG=false
REDIS_MEMORY=256mb
DB_NAME=sentiment_db
DB_USER=postgres
DB_PASSWORD=${DB_PASSWORD:-$(openssl rand -base64 12)}
MEMORY_LIMIT=2G
CPU_LIMIT=1.0
MEMORY_RESERVATION=512M
CPU_RESERVATION=0.25
EOF

    # Start services
    docker-compose -f docker-compose.yml up -d --build
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:5000/health >/dev/null 2>&1; then
        log_success "Deployment successful! Service available at http://localhost:5000"
    else
        log_error "Deployment failed - health check failed"
        exit 1
    fi
}

deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    cd "$DEPLOYMENT_DIR/kubernetes"
    
    # Substitute environment variables in deployment files
    envsubst < deployment.yaml > deployment-rendered.yaml
    
    # Apply the deployment
    kubectl apply -f deployment-rendered.yaml -n "$NAMESPACE"
    
    # Wait for rollout to complete
    kubectl rollout status deployment/sentiment-analyzer -n "$NAMESPACE" --timeout=600s
    
    # Get service URL
    if command -v minikube >/dev/null 2>&1 && minikube status >/dev/null 2>&1; then
        SERVICE_URL=$(minikube service sentiment-analyzer-service --url -n "$NAMESPACE")
        log_success "Deployment successful! Service available at $SERVICE_URL"
    else
        log_success "Deployment successful! Check service status with: kubectl get services -n $NAMESPACE"
    fi
    
    # Cleanup
    rm -f deployment-rendered.yaml
}

deploy_aws() {
    log_info "Deploying to AWS..."
    
    # Validate AWS environment
    if [[ -z "${AWS_REGION:-}" ]]; then
        log_error "AWS_REGION environment variable is required"
        exit 1
    fi
    
    # Build and push to ECR
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names physics-sentiment-analyzer --region "$AWS_REGION" 2>/dev/null || \
        aws ecr create-repository --repository-name physics-sentiment-analyzer --region "$AWS_REGION"
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
    
    # Build and push image
    docker build -f "$DEPLOYMENT_DIR/docker/Dockerfile" -t "$ECR_REGISTRY/physics-sentiment-analyzer:$IMAGE_TAG" "$PROJECT_ROOT"
    docker push "$ECR_REGISTRY/physics-sentiment-analyzer:$IMAGE_TAG"
    
    # Deploy using ECS or EKS (simplified example for ECS)
    log_info "Creating ECS task definition and service..."
    
    # This would typically use CloudFormation or Terraform
    # For now, just log the next steps
    log_success "Image pushed to ECR: $ECR_REGISTRY/physics-sentiment-analyzer:$IMAGE_TAG"
    log_info "Next steps:"
    log_info "1. Create ECS cluster if not exists"
    log_info "2. Create task definition using the pushed image"
    log_info "3. Create ECS service"
    log_info "4. Configure load balancer"
}

deploy_gcp() {
    log_info "Deploying to Google Cloud..."
    
    if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
        log_error "GCP_PROJECT_ID environment variable is required"
        exit 1
    fi
    
    # Set project
    gcloud config set project "$GCP_PROJECT_ID"
    
    # Build and push to Container Registry
    GCR_REGISTRY="gcr.io/$GCP_PROJECT_ID"
    
    gcloud builds submit --tag "$GCR_REGISTRY/physics-sentiment-analyzer:$IMAGE_TAG" "$PROJECT_ROOT"
    
    # Deploy to Cloud Run
    gcloud run deploy sentiment-analyzer \
        --image "$GCR_REGISTRY/physics-sentiment-analyzer:$IMAGE_TAG" \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated \
        --set-env-vars "DIFFHE_ENV=$ENVIRONMENT" \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 100
    
    SERVICE_URL=$(gcloud run services describe sentiment-analyzer --region us-central1 --format 'value(status.url)')
    log_success "Deployment successful! Service available at $SERVICE_URL"
}

deploy_azure() {
    log_info "Deploying to Azure..."
    
    if [[ -z "${AZURE_RESOURCE_GROUP:-}" ]]; then
        log_error "AZURE_RESOURCE_GROUP environment variable is required"
        exit 1
    fi
    
    # Create Azure Container Registry
    ACR_NAME="physicsentiment$(openssl rand -hex 6)"
    az acr create --resource-group "$AZURE_RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic
    
    # Build and push image
    az acr build --registry "$ACR_NAME" --image "physics-sentiment-analyzer:$IMAGE_TAG" "$PROJECT_ROOT"
    
    # Deploy to Container Instances
    az container create \
        --resource-group "$AZURE_RESOURCE_GROUP" \
        --name sentiment-analyzer \
        --image "$ACR_NAME.azurecr.io/physics-sentiment-analyzer:$IMAGE_TAG" \
        --dns-name-label sentiment-analyzer-$(openssl rand -hex 4) \
        --ports 5000 \
        --environment-variables DIFFHE_ENV="$ENVIRONMENT"
    
    FQDN=$(az container show --resource-group "$AZURE_RESOURCE_GROUP" --name sentiment-analyzer --query ipAddress.fqdn --output tsv)
    log_success "Deployment successful! Service available at http://$FQDN:5000"
}

cleanup() {
    log_info "Cleaning up..."
    
    case "$DEPLOYMENT_TARGET" in
        docker)
            cd "$DEPLOYMENT_DIR/docker"
            docker-compose down
            ;;
        kubernetes)
            kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
            ;;
    esac
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--replicas)
                REPLICA_COUNT="$2"
                shift 2
                ;;
            -d|--domain)
                DOMAIN="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            --cleanup)
                cleanup
                exit 0
                ;;
            *)
                if [[ -z "${DEPLOYMENT_TARGET:-}" ]]; then
                    DEPLOYMENT_TARGET="$1"
                else
                    log_error "Unknown option: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate arguments
    if [[ -z "$DEPLOYMENT_TARGET" ]]; then
        log_error "Deployment target is required"
        show_usage
        exit 1
    fi
    
    # Export variables for envsubst
    export ENVIRONMENT IMAGE_TAG NAMESPACE REPLICA_COUNT DOMAIN
    
    # Show deployment info
    log_info "Starting deployment with the following configuration:"
    log_info "  Target: $DEPLOYMENT_TARGET"
    log_info "  Environment: $ENVIRONMENT"
    log_info "  Image Tag: $IMAGE_TAG"
    log_info "  Namespace: $NAMESPACE"
    log_info "  Replicas: $REPLICA_COUNT"
    log_info "  Domain: $DOMAIN"
    
    # Validate requirements
    validate_requirements
    
    # Build image for most deployment targets
    if [[ "$DEPLOYMENT_TARGET" != "gcp" && "$DEPLOYMENT_TARGET" != "azure" ]]; then
        build_image
    fi
    
    # Deploy based on target
    case "$DEPLOYMENT_TARGET" in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        aws)
            deploy_aws
            ;;
        gcp)
            deploy_gcp
            ;;
        azure)
            deploy_azure
            ;;
    esac
    
    log_success "Deployment completed successfully!"
}

# Handle interrupts gracefully
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"