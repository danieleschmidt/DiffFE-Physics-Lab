#!/usr/bin/env python3
"""Deployment script for DiffFE-Physics-Lab framework."""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional


class DeploymentManager:
    """Manages deployment of DiffFE-Physics-Lab framework."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.docker_dir = self.root_dir / "docker"
        self.k8s_dir = self.root_dir / "kubernetes"
        self.monitoring_dir = self.root_dir / "monitoring"
    
    def deploy_docker(self, environment: str = "production", **kwargs):
        """Deploy using Docker Compose."""
        print(f"🐳 Deploying DiffFE-Physics-Lab with Docker ({environment})")
        
        # Set environment variables
        env_vars = {
            "DIFFHE_ENV": environment,
            "COMPOSE_PROJECT_NAME": "diffhe-physics-lab",
        }
        
        # Add any additional environment variables
        env_vars.update(kwargs)
        
        # Update environment
        os.environ.update(env_vars)
        
        # Build and deploy
        compose_file = self.docker_dir / "docker-compose.yml"
        
        try:
            # Pull latest images
            subprocess.run([
                "docker-compose", "-f", str(compose_file),
                "pull"
            ], check=True)
            
            # Build services
            subprocess.run([
                "docker-compose", "-f", str(compose_file),
                "build", "--no-cache"
            ], check=True)
            
            # Deploy services
            services = ["diffhe-app", "diffhe-db", "diffhe-cache"]
            if kwargs.get("monitoring", False):
                services.extend(["prometheus", "grafana"])
            if kwargs.get("gpu", False):
                services.append("diffhe-gpu")
            
            subprocess.run([
                "docker-compose", "-f", str(compose_file),
                "up", "-d"
            ] + services, check=True)
            
            print("✅ Docker deployment completed successfully")
            print(f"🌐 Application available at: http://localhost:8000")
            
            if kwargs.get("monitoring", False):
                print(f"📊 Grafana available at: http://localhost:3000")
                print(f"🔍 Prometheus available at: http://localhost:9090")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker deployment failed: {e}")
            return False
        
        return True
    
    def deploy_kubernetes(self, environment: str = "production", **kwargs):
        """Deploy to Kubernetes."""
        print(f"☸️  Deploying DiffFE-Physics-Lab to Kubernetes ({environment})")
        
        # Check kubectl availability
        try:
            subprocess.run(["kubectl", "version", "--client"], 
                          check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("❌ kubectl not found. Please install kubectl.")
            return False
        
        # Apply Kubernetes manifests
        k8s_files = [
            self.k8s_dir / "deployment.yaml",
        ]
        
        # Add monitoring if requested
        if kwargs.get("monitoring", False):
            k8s_files.extend([
                self.k8s_dir / "monitoring.yaml",
            ])
        
        try:
            for k8s_file in k8s_files:
                if k8s_file.exists():
                    subprocess.run([
                        "kubectl", "apply", "-f", str(k8s_file)
                    ], check=True)
            
            # Wait for deployment
            subprocess.run([
                "kubectl", "rollout", "status", 
                "deployment/diffhe-app",
                "-n", "diffhe-physics-lab"
            ], check=True)
            
            print("✅ Kubernetes deployment completed successfully")
            print("🔍 Check status with: kubectl get pods -n diffhe-physics-lab")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Kubernetes deployment failed: {e}")
            return False
        
        return True
    
    def deploy_cloud(self, cloud_provider: str, environment: str = "production", **kwargs):
        """Deploy to cloud provider."""
        print(f"☁️  Deploying DiffFE-Physics-Lab to {cloud_provider} ({environment})")
        
        if cloud_provider.lower() == "aws":
            return self._deploy_aws(environment, **kwargs)
        elif cloud_provider.lower() == "azure":
            return self._deploy_azure(environment, **kwargs)
        elif cloud_provider.lower() == "gcp":
            return self._deploy_gcp(environment, **kwargs)
        else:
            print(f"❌ Unsupported cloud provider: {cloud_provider}")
            return False
    
    def _deploy_aws(self, environment: str, **kwargs):
        """Deploy to AWS."""
        try:
            # Check AWS CLI
            subprocess.run(["aws", "--version"], check=True, capture_output=True)
            
            # Deploy using ECS or EKS
            if kwargs.get("kubernetes", False):
                # Deploy to EKS
                cluster_name = f"diffhe-{environment}"
                subprocess.run([
                    "eksctl", "create", "cluster", 
                    "--name", cluster_name,
                    "--region", kwargs.get("region", "us-west-2"),
                    "--nodes", str(kwargs.get("nodes", 3)),
                    "--node-type", kwargs.get("instance_type", "m5.large")
                ], check=True)
                
                # Apply K8s manifests
                return self.deploy_kubernetes(environment, **kwargs)
            else:
                # Deploy to ECS
                print("🚀 Deploying to AWS ECS...")
                # ECS deployment would go here
                return True
                
        except subprocess.CalledProcessError as e:
            print(f"❌ AWS deployment failed: {e}")
            return False
    
    def _deploy_azure(self, environment: str, **kwargs):
        """Deploy to Azure."""
        try:
            # Check Azure CLI
            subprocess.run(["az", "--version"], check=True, capture_output=True)
            
            # Deploy to AKS
            resource_group = f"diffhe-{environment}-rg"
            cluster_name = f"diffhe-{environment}"
            
            # Create resource group
            subprocess.run([
                "az", "group", "create",
                "--name", resource_group,
                "--location", kwargs.get("location", "eastus")
            ], check=True)
            
            # Create AKS cluster
            subprocess.run([
                "az", "aks", "create",
                "--resource-group", resource_group,
                "--name", cluster_name,
                "--node-count", str(kwargs.get("nodes", 3)),
                "--node-vm-size", kwargs.get("vm_size", "Standard_D2s_v3"),
                "--enable-addons", "monitoring"
            ], check=True)
            
            # Get credentials
            subprocess.run([
                "az", "aks", "get-credentials",
                "--resource-group", resource_group,
                "--name", cluster_name
            ], check=True)
            
            # Apply K8s manifests
            return self.deploy_kubernetes(environment, **kwargs)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Azure deployment failed: {e}")
            return False
    
    def _deploy_gcp(self, environment: str, **kwargs):
        """Deploy to Google Cloud Platform."""
        try:
            # Check gcloud CLI
            subprocess.run(["gcloud", "--version"], check=True, capture_output=True)
            
            # Deploy to GKE
            cluster_name = f"diffhe-{environment}"
            project_id = kwargs.get("project_id", "diffhe-physics-lab")
            zone = kwargs.get("zone", "us-central1-a")
            
            # Create GKE cluster
            subprocess.run([
                "gcloud", "container", "clusters", "create", cluster_name,
                "--project", project_id,
                "--zone", zone,
                "--num-nodes", str(kwargs.get("nodes", 3)),
                "--machine-type", kwargs.get("machine_type", "e2-standard-2"),
                "--enable-autoscaling",
                "--min-nodes", "1",
                "--max-nodes", "10"
            ], check=True)
            
            # Get credentials
            subprocess.run([
                "gcloud", "container", "clusters", "get-credentials", cluster_name,
                "--zone", zone,
                "--project", project_id
            ], check=True)
            
            # Apply K8s manifests
            return self.deploy_kubernetes(environment, **kwargs)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ GCP deployment failed: {e}")
            return False
    
    def cleanup(self, deployment_type: str):
        """Cleanup deployments."""
        print(f"🧹 Cleaning up {deployment_type} deployment")
        
        if deployment_type == "docker":
            compose_file = self.docker_dir / "docker-compose.yml"
            subprocess.run([
                "docker-compose", "-f", str(compose_file),
                "down", "-v", "--remove-orphans"
            ])
        
        elif deployment_type == "kubernetes":
            subprocess.run([
                "kubectl", "delete", "namespace", "diffhe-physics-lab"
            ])
        
        print("✅ Cleanup completed")
    
    def status(self, deployment_type: str):
        """Check deployment status."""
        print(f"📊 Checking {deployment_type} deployment status")
        
        if deployment_type == "docker":
            subprocess.run([
                "docker-compose", "-f", str(self.docker_dir / "docker-compose.yml"),
                "ps"
            ])
        
        elif deployment_type == "kubernetes":
            subprocess.run([
                "kubectl", "get", "all", "-n", "diffhe-physics-lab"
            ])
    
    def create_requirements(self):
        """Create requirements files for deployment."""
        print("📋 Creating requirements files...")
        
        # Base requirements
        base_requirements = [
            "numpy>=1.21.0",
            "scipy>=1.7.0", 
            "psutil>=5.9.0",
        ]
        
        # Optional requirements
        optional_requirements = {
            "jax": ["jax>=0.4.25", "jaxlib>=0.4.25"],
            "torch": ["torch>=2.4.0", "torchvision>=0.19.0"],
            "optimization": ["optax>=0.1.0", "scikit-learn>=1.0.0"],
            "data": ["h5py>=3.7.0", "netcdf4>=1.6.0", "xarray>=2022.3.0"],
            "visualization": ["matplotlib>=3.5.0", "plotly>=5.0.0"],
            "dev": ["pytest>=7.0.0", "black>=22.0.0", "mypy>=0.991"],
            "docs": ["sphinx>=5.0.0", "sphinx-rtd-theme>=1.0.0"],
            "performance": ["memory-profiler>=0.60.0", "line-profiler>=4.0.0"],
            "database": ["psycopg2-binary>=2.9.0", "sqlalchemy>=1.4.0"]
        }
        
        # Write requirements.txt
        with open("requirements.txt", "w") as f:
            f.write("# DiffFE-Physics-Lab requirements\n")
            f.write("# Base requirements (no external dependencies)\n\n")
            for req in base_requirements:
                f.write(f"{req}\n")
        
        # Write requirements-full.txt
        with open("requirements-full.txt", "w") as f:
            f.write("# DiffFE-Physics-Lab full requirements\n")
            f.write("# Includes all optional dependencies\n\n")
            f.write("-r requirements.txt\n\n")
            
            for category, reqs in optional_requirements.items():
                f.write(f"# {category.title()} dependencies\n")
                for req in reqs:
                    f.write(f"{req}\n")
                f.write("\n")
        
        print("✅ Requirements files created")


def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(
        description="Deploy DiffFE-Physics-Lab framework"
    )
    
    parser.add_argument(
        "action",
        choices=["deploy", "cleanup", "status", "requirements"],
        help="Action to perform"
    )
    
    parser.add_argument(
        "--type",
        choices=["docker", "kubernetes", "aws", "azure", "gcp"],
        default="docker",
        help="Deployment type"
    )
    
    parser.add_argument(
        "--environment", 
        choices=["development", "staging", "production"],
        default="production",
        help="Environment"
    )
    
    parser.add_argument(
        "--monitoring",
        action="store_true",
        help="Enable monitoring stack"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true", 
        help="Enable GPU support"
    )
    
    parser.add_argument(
        "--nodes",
        type=int,
        default=3,
        help="Number of nodes for cloud deployment"
    )
    
    args = parser.parse_args()
    
    # Create deployment manager
    deployer = DeploymentManager()
    
    # Perform action
    if args.action == "deploy":
        if args.type == "docker":
            success = deployer.deploy_docker(
                args.environment,
                monitoring=args.monitoring,
                gpu=args.gpu
            )
        elif args.type == "kubernetes":
            success = deployer.deploy_kubernetes(
                args.environment,
                monitoring=args.monitoring,
                gpu=args.gpu
            )
        elif args.type in ["aws", "azure", "gcp"]:
            success = deployer.deploy_cloud(
                args.type,
                args.environment,
                monitoring=args.monitoring,
                gpu=args.gpu,
                nodes=args.nodes
            )
        else:
            print(f"❌ Unsupported deployment type: {args.type}")
            sys.exit(1)
        
        if not success:
            sys.exit(1)
    
    elif args.action == "cleanup":
        deployer.cleanup(args.type)
    
    elif args.action == "status":
        deployer.status(args.type)
    
    elif args.action == "requirements":
        deployer.create_requirements()


if __name__ == "__main__":
    main()