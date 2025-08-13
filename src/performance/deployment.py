"""Multi-region deployment and scaling configuration system."""

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

try:
    import kubernetes
    from kubernetes import client, config

    K8S_AVAILABLE = True
except ImportError:
    kubernetes = None
    client = None
    config = None
    K8S_AVAILABLE = False

try:
    import boto3

    AWS_AVAILABLE = True
except ImportError:
    boto3 = None
    AWS_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.containerinstance import ContainerInstanceManagementClient

    AZURE_AVAILABLE = True
except ImportError:
    DefaultAzureCredential = None
    ContainerInstanceManagementClient = None
    AZURE_AVAILABLE = False

try:
    from google.cloud import container_v1

    GCP_AVAILABLE = True
except ImportError:
    container_v1 = None
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status enumeration."""

    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    SCALING = "scaling"
    ERROR = "error"
    TERMINATED = "terminated"


class CloudProvider(Enum):
    """Supported cloud providers."""

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    KUBERNETES = "kubernetes"
    LOCAL = "local"


@dataclass
class RegionConfig:
    """Regional deployment configuration."""

    name: str
    cloud_provider: CloudProvider
    region: str
    availability_zones: List[str] = field(default_factory=list)
    instance_types: List[str] = field(default_factory=list)
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    network_config: Dict[str, Any] = field(default_factory=dict)
    storage_config: Dict[str, Any] = field(default_factory=dict)
    security_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentInstance:
    """Individual deployment instance."""

    id: str
    region_name: str
    cloud_provider: CloudProvider
    instance_type: str
    status: DeploymentStatus
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_health_check: float = 0.0
    health_status: str = "unknown"
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadBalancingConfig:
    """Load balancing configuration."""

    algorithm: str = (
        "round_robin"  # round_robin, least_connections, weighted, geographic
    )
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    failover_threshold: int = 3
    sticky_sessions: bool = False
    geographic_routing: bool = True
    latency_threshold_ms: float = 100.0


class MultiRegionDeploymentManager:
    """Manage multi-region deployments with auto-scaling and load balancing."""

    def __init__(
        self,
        config_file: Optional[str] = None,
        enable_auto_scaling: bool = True,
        enable_load_balancing: bool = True,
    ):
        self.config_file = config_file
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_load_balancing = enable_load_balancing

        # Deployment state
        self.regions: Dict[str, RegionConfig] = {}
        self.instances: Dict[str, DeploymentInstance] = {}
        self.deployment_history: List[Dict] = []

        # Load balancing
        self.load_balancer_config = LoadBalancingConfig()
        self.active_endpoints = {}  # region -> list of healthy endpoints

        # Monitoring
        self.health_monitor_active = False
        self.health_monitor_thread = None

        # Cloud provider clients
        self.cloud_clients = {}

        # Load configuration
        if config_file:
            self.load_configuration(config_file)

        logger.info("Multi-region deployment manager initialized")

    def load_configuration(self, config_file: str):
        """Load deployment configuration from file."""
        try:
            config_path = Path(config_file)

            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_file}")
                return

            with open(config_path, "r") as f:
                if (
                    config_path.suffix.lower() == ".yaml"
                    or config_path.suffix.lower() == ".yml"
                ):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            # Parse regions
            for region_data in config_data.get("regions", []):
                region = RegionConfig(
                    name=region_data["name"],
                    cloud_provider=CloudProvider(region_data["cloud_provider"]),
                    region=region_data["region"],
                    availability_zones=region_data.get("availability_zones", []),
                    instance_types=region_data.get("instance_types", ["t3.medium"]),
                    min_instances=region_data.get("min_instances", 1),
                    max_instances=region_data.get("max_instances", 10),
                    target_cpu_utilization=region_data.get(
                        "target_cpu_utilization", 70.0
                    ),
                    target_memory_utilization=region_data.get(
                        "target_memory_utilization", 80.0
                    ),
                    network_config=region_data.get("network_config", {}),
                    storage_config=region_data.get("storage_config", {}),
                    security_config=region_data.get("security_config", {}),
                )
                self.regions[region.name] = region

            # Parse load balancer configuration
            lb_config = config_data.get("load_balancer", {})
            self.load_balancer_config = LoadBalancingConfig(
                algorithm=lb_config.get("algorithm", "round_robin"),
                health_check_interval=lb_config.get("health_check_interval", 30.0),
                health_check_timeout=lb_config.get("health_check_timeout", 5.0),
                failover_threshold=lb_config.get("failover_threshold", 3),
                sticky_sessions=lb_config.get("sticky_sessions", False),
                geographic_routing=lb_config.get("geographic_routing", True),
                latency_threshold_ms=lb_config.get("latency_threshold_ms", 100.0),
            )

            logger.info(
                f"Loaded configuration for {len(self.regions)} regions from {config_file}"
            )

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")

    def save_configuration(self, config_file: str):
        """Save deployment configuration to file."""
        try:
            config_data = {
                "regions": [asdict(region) for region in self.regions.values()],
                "load_balancer": asdict(self.load_balancer_config),
            }

            # Convert enums to strings
            for region in config_data["regions"]:
                region["cloud_provider"] = region["cloud_provider"].value

            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w") as f:
                if (
                    config_path.suffix.lower() == ".yaml"
                    or config_path.suffix.lower() == ".yml"
                ):
                    yaml.dump(config_data, f, indent=2)
                else:
                    json.dump(config_data, f, indent=2)

            logger.info(f"Configuration saved to {config_file}")

        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")

    def initialize_cloud_clients(self):
        """Initialize cloud provider clients."""
        for region in self.regions.values():
            if region.cloud_provider == CloudProvider.AWS and AWS_AVAILABLE:
                self._initialize_aws_client(region)
            elif region.cloud_provider == CloudProvider.AZURE and AZURE_AVAILABLE:
                self._initialize_azure_client(region)
            elif region.cloud_provider == CloudProvider.GCP and GCP_AVAILABLE:
                self._initialize_gcp_client(region)
            elif region.cloud_provider == CloudProvider.KUBERNETES and K8S_AVAILABLE:
                self._initialize_k8s_client(region)

    def _initialize_aws_client(self, region: RegionConfig):
        """Initialize AWS client for region."""
        try:
            session = boto3.Session(region_name=region.region)
            self.cloud_clients[region.name] = {
                "ec2": session.client("ec2"),
                "ecs": session.client("ecs"),
                "elb": session.client("elbv2"),
                "autoscaling": session.client("autoscaling"),
            }
            logger.info(f"AWS client initialized for region {region.name}")
        except Exception as e:
            logger.error(f"Failed to initialize AWS client for {region.name}: {e}")

    def _initialize_azure_client(self, region: RegionConfig):
        """Initialize Azure client for region."""
        try:
            credential = DefaultAzureCredential()
            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")

            self.cloud_clients[region.name] = {
                "container_client": ContainerInstanceManagementClient(
                    credential, subscription_id
                ),
                "credential": credential,
            }
            logger.info(f"Azure client initialized for region {region.name}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure client for {region.name}: {e}")

    def _initialize_gcp_client(self, region: RegionConfig):
        """Initialize GCP client for region."""
        try:
            self.cloud_clients[region.name] = {
                "container_client": container_v1.ClusterManagerClient()
            }
            logger.info(f"GCP client initialized for region {region.name}")
        except Exception as e:
            logger.error(f"Failed to initialize GCP client for {region.name}: {e}")

    def _initialize_k8s_client(self, region: RegionConfig):
        """Initialize Kubernetes client for region."""
        try:
            config.load_incluster_config()  # Try in-cluster config first
        except:
            try:
                config.load_kube_config()  # Fall back to local config
            except Exception as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                return

        self.cloud_clients[region.name] = {
            "core_v1": client.CoreV1Api(),
            "apps_v1": client.AppsV1Api(),
            "autoscaling_v1": client.AutoscalingV1Api(),
        }
        logger.info(f"Kubernetes client initialized for region {region.name}")

    async def deploy_to_region(
        self,
        region_name: str,
        application_config: Dict[str, Any],
        num_instances: int = 1,
    ) -> List[str]:
        """Deploy application to specific region."""
        if region_name not in self.regions:
            raise ValueError(f"Region {region_name} not configured")

        region = self.regions[region_name]
        deployed_instances = []

        logger.info(f"Deploying {num_instances} instances to region {region_name}")

        for i in range(num_instances):
            try:
                instance = await self._deploy_single_instance(
                    region, application_config, i
                )
                deployed_instances.append(instance.id)
                self.instances[instance.id] = instance

                logger.info(f"Instance {instance.id} deployed to {region_name}")

            except Exception as e:
                logger.error(f"Failed to deploy instance {i} to {region_name}: {e}")

        # Update active endpoints
        self._update_active_endpoints(region_name)

        # Record deployment
        self.deployment_history.append(
            {
                "timestamp": time.time(),
                "action": "deploy",
                "region": region_name,
                "instances_deployed": len(deployed_instances),
                "instance_ids": deployed_instances,
            }
        )

        return deployed_instances

    async def _deploy_single_instance(
        self,
        region: RegionConfig,
        application_config: Dict[str, Any],
        instance_index: int,
    ) -> DeploymentInstance:
        """Deploy single instance to region."""
        instance_id = f"{region.name}-instance-{instance_index}-{int(time.time())}"

        instance = DeploymentInstance(
            id=instance_id,
            region_name=region.name,
            cloud_provider=region.cloud_provider,
            instance_type=(
                region.instance_types[0] if region.instance_types else "default"
            ),
            status=DeploymentStatus.DEPLOYING,
        )

        # Simulate deployment based on cloud provider
        if region.cloud_provider == CloudProvider.AWS:
            await self._deploy_aws_instance(region, instance, application_config)
        elif region.cloud_provider == CloudProvider.KUBERNETES:
            await self._deploy_k8s_instance(region, instance, application_config)
        elif region.cloud_provider == CloudProvider.LOCAL:
            await self._deploy_local_instance(region, instance, application_config)
        else:
            # Simulate deployment for other providers
            await asyncio.sleep(2)  # Simulate deployment time
            instance.status = DeploymentStatus.RUNNING
            instance.public_ip = f"192.168.{instance_index}.100"
            instance.private_ip = f"10.0.{instance_index}.100"

        return instance

    async def _deploy_aws_instance(
        self, region: RegionConfig, instance: DeploymentInstance, config: Dict[str, Any]
    ):
        """Deploy instance to AWS."""
        # Simulate AWS deployment
        await asyncio.sleep(3)  # Simulate AWS deployment time
        instance.status = DeploymentStatus.RUNNING
        instance.public_ip = f"52.{region.region.split('-')[1]}.{hash(instance.id) % 255}.{hash(instance.id) % 255}"
        instance.private_ip = (
            f"172.31.{hash(instance.id) % 255}.{hash(instance.id) % 255}"
        )

    async def _deploy_k8s_instance(
        self, region: RegionConfig, instance: DeploymentInstance, config: Dict[str, Any]
    ):
        """Deploy instance to Kubernetes."""
        if region.name not in self.cloud_clients:
            raise ValueError(f"Kubernetes client not initialized for {region.name}")

        # Simulate Kubernetes deployment
        await asyncio.sleep(2)
        instance.status = DeploymentStatus.RUNNING
        instance.private_ip = (
            f"10.244.{hash(instance.id) % 255}.{hash(instance.id) % 255}"
        )

    async def _deploy_local_instance(
        self, region: RegionConfig, instance: DeploymentInstance, config: Dict[str, Any]
    ):
        """Deploy instance locally (for testing)."""
        await asyncio.sleep(1)  # Simulate local deployment time
        instance.status = DeploymentStatus.RUNNING
        instance.public_ip = "127.0.0.1"
        instance.private_ip = "127.0.0.1"

    def scale_region(self, region_name: str, target_instances: int):
        """Scale region to target number of instances."""
        if region_name not in self.regions:
            raise ValueError(f"Region {region_name} not configured")

        region = self.regions[region_name]
        current_instances = [
            inst
            for inst in self.instances.values()
            if inst.region_name == region_name
            and inst.status == DeploymentStatus.RUNNING
        ]

        current_count = len(current_instances)

        logger.info(
            f"Scaling {region_name}: {current_count} -> {target_instances} instances"
        )

        if target_instances > current_count:
            # Scale up
            scale_up_count = target_instances - current_count
            asyncio.create_task(self._scale_up_region(region_name, scale_up_count))
        elif target_instances < current_count:
            # Scale down
            scale_down_count = current_count - target_instances
            self._scale_down_region(region_name, scale_down_count)

    async def _scale_up_region(self, region_name: str, count: int):
        """Scale up region by adding instances."""
        region = self.regions[region_name]

        for i in range(count):
            try:
                instance = await self._deploy_single_instance(
                    region, {}, len(self.instances) + i
                )
                self.instances[instance.id] = instance

                logger.info(f"Scaled up: added instance {instance.id} to {region_name}")

            except Exception as e:
                logger.error(f"Failed to scale up instance in {region_name}: {e}")

        self._update_active_endpoints(region_name)

    def _scale_down_region(self, region_name: str, count: int):
        """Scale down region by removing instances."""
        region_instances = [
            inst
            for inst in self.instances.values()
            if inst.region_name == region_name
            and inst.status == DeploymentStatus.RUNNING
        ]

        # Remove oldest instances first
        instances_to_remove = sorted(region_instances, key=lambda x: x.created_at)[
            :count
        ]

        for instance in instances_to_remove:
            self._terminate_instance(instance.id)
            logger.info(
                f"Scaled down: removed instance {instance.id} from {region_name}"
            )

        self._update_active_endpoints(region_name)

    def _terminate_instance(self, instance_id: str):
        """Terminate a specific instance."""
        if instance_id not in self.instances:
            return

        instance = self.instances[instance_id]
        instance.status = DeploymentStatus.TERMINATED

        # Remove from instances after a delay (to keep history)
        def delayed_removal():
            time.sleep(300)  # Keep terminated instance info for 5 minutes
            if instance_id in self.instances:
                del self.instances[instance_id]

        threading.Thread(target=delayed_removal, daemon=True).start()

    def _update_active_endpoints(self, region_name: str):
        """Update active endpoints for load balancing."""
        healthy_instances = [
            inst
            for inst in self.instances.values()
            if (
                inst.region_name == region_name
                and inst.status == DeploymentStatus.RUNNING
                and inst.health_status in ["healthy", "unknown"]
            )
        ]

        endpoints = []
        for instance in healthy_instances:
            endpoint = {
                "instance_id": instance.id,
                "ip": instance.public_ip or instance.private_ip,
                "port": 8080,  # Default port
                "weight": 1.0,
                "health_status": instance.health_status,
            }
            endpoints.append(endpoint)

        self.active_endpoints[region_name] = endpoints
        logger.debug(f"Updated {region_name}: {len(endpoints)} active endpoints")

    def start_health_monitoring(self):
        """Start health monitoring for all instances."""
        if self.health_monitor_active:
            return

        self.health_monitor_active = True
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop, daemon=True
        )
        self.health_monitor_thread.start()

        logger.info("Health monitoring started")

    def stop_health_monitoring(self):
        """Stop health monitoring."""
        self.health_monitor_active = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5.0)

        logger.info("Health monitoring stopped")

    def _health_monitor_loop(self):
        """Health monitoring loop."""
        while self.health_monitor_active:
            try:
                for instance in self.instances.values():
                    if instance.status == DeploymentStatus.RUNNING:
                        self._check_instance_health(instance)

                # Update active endpoints based on health status
                for region_name in self.regions:
                    self._update_active_endpoints(region_name)

                time.sleep(self.load_balancer_config.health_check_interval)

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(30)

    def _check_instance_health(self, instance: DeploymentInstance):
        """Check health of a single instance."""
        try:
            # Simulate health check
            import random

            # 95% chance of healthy status
            if random.random() < 0.95:
                instance.health_status = "healthy"
                instance.last_health_check = time.time()

                # Update metrics
                instance.metrics.update(
                    {
                        "cpu_usage": random.uniform(20, 80),
                        "memory_usage": random.uniform(30, 70),
                        "response_time": random.uniform(50, 200),
                    }
                )
            else:
                instance.health_status = "unhealthy"

        except Exception as e:
            instance.health_status = "unhealthy"
            logger.error(f"Health check failed for {instance.id}: {e}")

    def get_region_status(self, region_name: str) -> Dict[str, Any]:
        """Get status of a specific region."""
        if region_name not in self.regions:
            return {"error": f"Region {region_name} not found"}

        region_instances = [
            inst for inst in self.instances.values() if inst.region_name == region_name
        ]

        running_instances = [
            inst for inst in region_instances if inst.status == DeploymentStatus.RUNNING
        ]

        healthy_instances = [
            inst for inst in running_instances if inst.health_status == "healthy"
        ]

        return {
            "region_name": region_name,
            "total_instances": len(region_instances),
            "running_instances": len(running_instances),
            "healthy_instances": len(healthy_instances),
            "active_endpoints": len(self.active_endpoints.get(region_name, [])),
            "average_cpu": (
                sum(inst.metrics.get("cpu_usage", 0) for inst in running_instances)
                / len(running_instances)
                if running_instances
                else 0
            ),
            "average_memory": (
                sum(inst.metrics.get("memory_usage", 0) for inst in running_instances)
                / len(running_instances)
                if running_instances
                else 0
            ),
            "instances": [asdict(inst) for inst in region_instances],
        }

    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get overall deployment summary."""
        total_instances = len(self.instances)
        running_instances = len(
            [
                inst
                for inst in self.instances.values()
                if inst.status == DeploymentStatus.RUNNING
            ]
        )
        healthy_instances = len(
            [
                inst
                for inst in self.instances.values()
                if inst.status == DeploymentStatus.RUNNING
                and inst.health_status == "healthy"
            ]
        )

        region_summaries = {}
        for region_name in self.regions:
            region_summaries[region_name] = self.get_region_status(region_name)

        return {
            "total_regions": len(self.regions),
            "total_instances": total_instances,
            "running_instances": running_instances,
            "healthy_instances": healthy_instances,
            "health_monitoring_active": self.health_monitor_active,
            "auto_scaling_enabled": self.enable_auto_scaling,
            "load_balancing_enabled": self.enable_load_balancing,
            "region_summaries": region_summaries,
            "deployment_history_count": len(self.deployment_history),
            "load_balancer_config": asdict(self.load_balancer_config),
        }


# Global deployment manager
_global_deployment_manager = None


def get_deployment_manager(config_file: str = None) -> MultiRegionDeploymentManager:
    """Get global deployment manager instance."""
    global _global_deployment_manager
    if _global_deployment_manager is None:
        _global_deployment_manager = MultiRegionDeploymentManager(
            config_file=config_file
        )
    return _global_deployment_manager


def create_sample_deployment_config(output_file: str = "deployment_config.yaml"):
    """Create sample deployment configuration file."""
    sample_config = {
        "regions": [
            {
                "name": "us-east-1",
                "cloud_provider": "aws",
                "region": "us-east-1",
                "availability_zones": ["us-east-1a", "us-east-1b", "us-east-1c"],
                "instance_types": ["t3.medium", "t3.large"],
                "min_instances": 2,
                "max_instances": 10,
                "target_cpu_utilization": 70.0,
                "target_memory_utilization": 80.0,
                "network_config": {
                    "vpc_id": "vpc-12345",
                    "subnet_ids": ["subnet-123", "subnet-456"],
                },
                "storage_config": {"volume_size": 20, "volume_type": "gp3"},
                "security_config": {
                    "security_groups": ["sg-12345"],
                    "key_pair": "my-key-pair",
                },
            },
            {
                "name": "eu-west-1",
                "cloud_provider": "aws",
                "region": "eu-west-1",
                "availability_zones": ["eu-west-1a", "eu-west-1b", "eu-west-1c"],
                "instance_types": ["t3.medium", "t3.large"],
                "min_instances": 1,
                "max_instances": 8,
                "target_cpu_utilization": 75.0,
                "target_memory_utilization": 85.0,
            },
            {
                "name": "local-dev",
                "cloud_provider": "local",
                "region": "local",
                "instance_types": ["local"],
                "min_instances": 1,
                "max_instances": 3,
                "target_cpu_utilization": 80.0,
                "target_memory_utilization": 90.0,
            },
        ],
        "load_balancer": {
            "algorithm": "round_robin",
            "health_check_interval": 30.0,
            "health_check_timeout": 5.0,
            "failover_threshold": 3,
            "sticky_sessions": False,
            "geographic_routing": True,
            "latency_threshold_ms": 100.0,
        },
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(sample_config, f, indent=2)

    logger.info(f"Sample deployment configuration created: {output_file}")
    return output_file
