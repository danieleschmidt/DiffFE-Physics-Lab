"""Comprehensive scaling configuration and orchestration system."""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

# Import all performance components
from .advanced_cache import get_adaptive_cache
from .advanced_optimization import (
    get_async_io,
    get_batch_processor,
    get_jax_engine,
    get_mesh_refinement,
)
from .dashboard import get_dashboard
from .deployment import get_deployment_manager
from .ml_acceleration import get_fusion_engine, get_nas_engine
from .parallel_processing import (
    get_autoscaling_manager,
    get_distributed_manager,
    get_parallel_engine,
    get_resource_monitor,
)
from .profiling import get_profiler

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Scaling operation modes."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class PerformanceProfile(Enum):
    """Performance optimization profiles."""

    MEMORY_OPTIMIZED = "memory_optimized"
    CPU_OPTIMIZED = "cpu_optimized"
    GPU_OPTIMIZED = "gpu_optimized"
    BALANCED = "balanced"
    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"


@dataclass
class ScalingConfiguration:
    """Complete scaling configuration."""

    # Environment
    mode: ScalingMode = ScalingMode.DEVELOPMENT
    profile: PerformanceProfile = PerformanceProfile.BALANCED

    # Caching configuration
    cache_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_size": 1000,
            "max_memory_mb": 100,
            "enable_distributed": False,
            "redis_config": None,
            "enable_assembly_cache": True,
            "assembly_cache_memory_gb": 2.0,
        }
    )

    # Parallel processing configuration
    parallel_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_threads": None,  # Auto-detect
            "enable_gpu": True,
            "enable_distributed": False,
            "distributed_backend": "auto",
            "enable_auto_scaling": True,
            "min_workers": 1,
            "max_workers": 16,
        }
    )

    # Optimization configuration
    optimization_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_mesh_refinement": True,
            "refinement_threshold": 0.1,
            "enable_jit_compilation": True,
            "enable_operator_fusion": True,
            "enable_mixed_precision": False,
            "enable_gradient_checkpointing": True,
        }
    )

    # ML acceleration configuration
    ml_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_nas": False,
            "nas_generations": 20,
            "nas_population_size": 10,
            "enable_neural_operators": False,
        }
    )

    # Profiling configuration
    profiling_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_profiling": False,
            "profiler_types": ["cProfile"],
            "enable_flamegraphs": True,
            "auto_save_reports": False,
            "output_dir": "./profiling_results",
        }
    )

    # Dashboard configuration
    dashboard_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_dashboard": False,
            "host": "0.0.0.0",
            "port": 8080,
            "enable_real_time": True,
        }
    )

    # Deployment configuration
    deployment_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_multi_region": False,
            "config_file": None,
            "regions": [],
            "enable_load_balancing": True,
        }
    )

    # Monitoring configuration
    monitoring_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_resource_monitoring": True,
            "monitoring_interval": 1.0,
            "enable_alerts": True,
            "alert_thresholds": {
                "cpu_percent": 85.0,
                "memory_percent": 90.0,
                "response_time_ms": 1000.0,
            },
        }
    )

    # Async I/O configuration
    async_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_async_io": False,
            "max_concurrent_operations": 100,
            "enable_database_pool": False,
            "database_url": None,
        }
    )

    # Batch processing configuration
    batch_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_batch_processing": False,
            "batch_size": 10,
            "max_workers": 4,
            "enable_async": True,
        }
    )


class ScalingOrchestrator:
    """Central orchestrator for all scaling and performance components."""

    def __init__(
        self,
        config: Optional[ScalingConfiguration] = None,
        config_file: Optional[str] = None,
    ):
        self.config = config or ScalingConfiguration()
        self.config_file = config_file

        # Component instances
        self.components = {}
        self.initialized_components = set()

        # State tracking
        self.orchestrator_active = False
        self.performance_metrics = {}
        self.scaling_history = []

        # Load configuration from file if provided
        if config_file:
            self.load_configuration(config_file)

        # Apply profile optimizations
        self._apply_profile_optimizations()

        logger.info(
            f"Scaling orchestrator initialized (mode: {self.config.mode.value}, profile: {self.config.profile.value})"
        )

    def _apply_profile_optimizations(self):
        """Apply optimizations based on performance profile."""
        profile = self.config.profile

        if profile == PerformanceProfile.MEMORY_OPTIMIZED:
            self.config.cache_config["max_memory_mb"] = 50
            self.config.parallel_config["max_workers"] = 8
            self.config.optimization_config["enable_gradient_checkpointing"] = True
            self.config.optimization_config["enable_mixed_precision"] = True

        elif profile == PerformanceProfile.CPU_OPTIMIZED:
            self.config.parallel_config["num_threads"] = os.cpu_count()
            self.config.parallel_config["max_workers"] = os.cpu_count() * 2
            self.config.optimization_config["enable_jit_compilation"] = True
            self.config.optimization_config["enable_operator_fusion"] = True

        elif profile == PerformanceProfile.GPU_OPTIMIZED:
            self.config.parallel_config["enable_gpu"] = True
            self.config.optimization_config["enable_mixed_precision"] = True
            self.config.ml_config["enable_neural_operators"] = True

        elif profile == PerformanceProfile.LATENCY_OPTIMIZED:
            self.config.cache_config["max_size"] = 5000
            self.config.cache_config["enable_distributed"] = True
            self.config.parallel_config["enable_auto_scaling"] = True
            self.config.optimization_config["enable_jit_compilation"] = True

        elif profile == PerformanceProfile.THROUGHPUT_OPTIMIZED:
            self.config.parallel_config["max_workers"] = 32
            self.config.batch_config["batch_size"] = 100
            self.config.batch_config["enable_batch_processing"] = True
            self.config.async_config["enable_async_io"] = True

        logger.info(f"Applied {profile.value} optimizations")

    def load_configuration(self, config_file: str):
        """Load configuration from file."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_file}")
                return

            with open(config_path, "r") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            # Update configuration
            if "mode" in config_data:
                self.config.mode = ScalingMode(config_data["mode"])
            if "profile" in config_data:
                self.config.profile = PerformanceProfile(config_data["profile"])

            # Update component configurations
            for component_name in [
                "cache_config",
                "parallel_config",
                "optimization_config",
                "ml_config",
                "profiling_config",
                "dashboard_config",
                "deployment_config",
                "monitoring_config",
                "async_config",
                "batch_config",
            ]:
                if component_name in config_data:
                    component_config = getattr(self.config, component_name)
                    component_config.update(config_data[component_name])

            logger.info(f"Configuration loaded from {config_file}")

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")

    def save_configuration(self, config_file: str):
        """Save current configuration to file."""
        try:
            config_data = {
                "mode": self.config.mode.value,
                "profile": self.config.profile.value,
                "cache_config": self.config.cache_config,
                "parallel_config": self.config.parallel_config,
                "optimization_config": self.config.optimization_config,
                "ml_config": self.config.ml_config,
                "profiling_config": self.config.profiling_config,
                "dashboard_config": self.config.dashboard_config,
                "deployment_config": self.config.deployment_config,
                "monitoring_config": self.config.monitoring_config,
                "async_config": self.config.async_config,
                "batch_config": self.config.batch_config,
            }

            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(config_data, f, indent=2)
                else:
                    json.dump(config_data, f, indent=2)

            logger.info(f"Configuration saved to {config_file}")

        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")

    def initialize_all_components(self):
        """Initialize all enabled components."""
        logger.info("Initializing scaling components...")

        # Initialize caching
        if self.config.cache_config.get("max_size", 0) > 0:
            self._initialize_caching()

        # Initialize parallel processing
        if self.config.parallel_config.get("num_threads", 0) != 0:
            self._initialize_parallel_processing()

        # Initialize optimization
        if any(self.config.optimization_config.values()):
            self._initialize_optimization()

        # Initialize ML acceleration
        if any(self.config.ml_config.values()):
            self._initialize_ml_acceleration()

        # Initialize profiling
        if self.config.profiling_config.get("enable_profiling", False):
            self._initialize_profiling()

        # Initialize dashboard
        if self.config.dashboard_config.get("enable_dashboard", False):
            self._initialize_dashboard()

        # Initialize deployment
        if self.config.deployment_config.get("enable_multi_region", False):
            self._initialize_deployment()

        # Initialize monitoring
        if self.config.monitoring_config.get("enable_resource_monitoring", True):
            self._initialize_monitoring()

        # Initialize async I/O
        if self.config.async_config.get("enable_async_io", False):
            self._initialize_async_io()

        # Initialize batch processing
        if self.config.batch_config.get("enable_batch_processing", False):
            self._initialize_batch_processing()

        logger.info(f"Initialized {len(self.initialized_components)} components")

    def _initialize_caching(self):
        """Initialize caching components."""
        try:
            cache_config = self.config.cache_config

            cache_manager = get_adaptive_cache()
            cache_manager.max_size = cache_config.get("max_size", 1000)
            cache_manager.max_memory_mb = cache_config.get("max_memory_mb", 100)

            if cache_config.get("enable_distributed", False) and cache_config.get(
                "redis_config"
            ):
                # Initialize distributed caching if configured
                pass

            self.components["cache"] = cache_manager
            self.initialized_components.add("cache")

            logger.info("Caching components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize caching: {e}")

    def _initialize_parallel_processing(self):
        """Initialize parallel processing components."""
        try:
            parallel_config = self.config.parallel_config

            # Resource monitor
            resource_monitor = get_resource_monitor()
            resource_monitor.start_monitoring()

            # Parallel engine
            parallel_engine = get_parallel_engine()

            # Distributed computing
            if parallel_config.get("enable_distributed", False):
                distributed_manager = get_distributed_manager(
                    backend=parallel_config.get("distributed_backend", "auto")
                )
                self.components["distributed"] = distributed_manager

            # Auto-scaling
            if parallel_config.get("enable_auto_scaling", False):
                autoscaling_manager = get_autoscaling_manager()
                autoscaling_manager.min_workers = parallel_config.get("min_workers", 1)
                autoscaling_manager.max_workers = parallel_config.get("max_workers", 16)
                autoscaling_manager.start_auto_scaling()
                self.components["autoscaling"] = autoscaling_manager

            self.components["resource_monitor"] = resource_monitor
            self.components["parallel_engine"] = parallel_engine
            self.initialized_components.add("parallel_processing")

            logger.info("Parallel processing components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize parallel processing: {e}")

    def _initialize_optimization(self):
        """Initialize optimization components."""
        try:
            opt_config = self.config.optimization_config

            # Mesh refinement
            if opt_config.get("enable_mesh_refinement", True):
                mesh_refinement = get_mesh_refinement()
                mesh_refinement.refinement_threshold = opt_config.get(
                    "refinement_threshold", 0.1
                )
                self.components["mesh_refinement"] = mesh_refinement

            # JAX optimization engine
            if opt_config.get("enable_jit_compilation", True):
                jax_engine = get_jax_engine()
                jax_engine.enable_checkpointing = opt_config.get(
                    "enable_gradient_checkpointing", True
                )
                jax_engine.enable_mixed_precision = opt_config.get(
                    "enable_mixed_precision", False
                )
                self.components["jax_engine"] = jax_engine

            self.initialized_components.add("optimization")

            logger.info("Optimization components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize optimization: {e}")

    def _initialize_ml_acceleration(self):
        """Initialize ML acceleration components."""
        try:
            ml_config = self.config.ml_config

            # Operator fusion
            if ml_config.get("enable_operator_fusion", True):
                fusion_engine = get_fusion_engine()
                self.components["fusion_engine"] = fusion_engine

            # Neural architecture search
            if ml_config.get("enable_nas", False):
                nas_engine = get_nas_engine(
                    generations=ml_config.get("nas_generations", 20),
                    population_size=ml_config.get("nas_population_size", 10),
                )
                self.components["nas_engine"] = nas_engine

            self.initialized_components.add("ml_acceleration")

            logger.info("ML acceleration components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ML acceleration: {e}")

    def _initialize_profiling(self):
        """Initialize profiling components."""
        try:
            profiler = get_profiler()
            self.components["profiler"] = profiler
            self.initialized_components.add("profiling")

            logger.info("Profiling components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize profiling: {e}")

    def _initialize_dashboard(self):
        """Initialize dashboard components."""
        try:
            dashboard_config = self.config.dashboard_config

            dashboard = get_dashboard(
                host=dashboard_config.get("host", "0.0.0.0"),
                port=dashboard_config.get("port", 8080),
            )

            if dashboard_config.get("enable_real_time", True):
                dashboard.run_async()

            self.components["dashboard"] = dashboard
            self.initialized_components.add("dashboard")

            logger.info("Dashboard components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")

    def _initialize_deployment(self):
        """Initialize deployment components."""
        try:
            deployment_config = self.config.deployment_config

            deployment_manager = get_deployment_manager(
                config_file=deployment_config.get("config_file")
            )

            deployment_manager.initialize_cloud_clients()

            if deployment_config.get("enable_load_balancing", True):
                deployment_manager.start_health_monitoring()

            self.components["deployment"] = deployment_manager
            self.initialized_components.add("deployment")

            logger.info("Deployment components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize deployment: {e}")

    def _initialize_monitoring(self):
        """Initialize monitoring components."""
        try:
            monitoring_config = self.config.monitoring_config

            # Resource monitoring already handled in parallel processing
            if "resource_monitor" not in self.components:
                resource_monitor = get_resource_monitor()
                resource_monitor.monitoring_interval = monitoring_config.get(
                    "monitoring_interval", 1.0
                )
                resource_monitor.start_monitoring()
                self.components["resource_monitor"] = resource_monitor

            self.initialized_components.add("monitoring")

            logger.info("Monitoring components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")

    def _initialize_async_io(self):
        """Initialize async I/O components."""
        try:
            async_config = self.config.async_config

            async_io_manager = get_async_io()
            async_io_manager.max_concurrent_operations = async_config.get(
                "max_concurrent_operations", 100
            )

            self.components["async_io"] = async_io_manager
            self.initialized_components.add("async_io")

            logger.info("Async I/O components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize async I/O: {e}")

    def _initialize_batch_processing(self):
        """Initialize batch processing components."""
        try:
            batch_config = self.config.batch_config

            batch_processor = get_batch_processor()
            batch_processor.batch_size = batch_config.get("batch_size", 10)
            batch_processor.max_workers = batch_config.get("max_workers", 4)

            self.components["batch_processor"] = batch_processor
            self.initialized_components.add("batch_processing")

            logger.info("Batch processing components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize batch processing: {e}")

    def start_orchestration(self):
        """Start the orchestration system."""
        if self.orchestrator_active:
            return

        self.orchestrator_active = True

        # Initialize components if not already done
        if not self.initialized_components:
            self.initialize_all_components()

        # Start orchestration monitoring thread
        self.orchestration_thread = threading.Thread(
            target=self._orchestration_loop, daemon=True
        )
        self.orchestration_thread.start()

        logger.info("Scaling orchestration started")

    def stop_orchestration(self):
        """Stop the orchestration system."""
        if not self.orchestrator_active:
            return

        self.orchestrator_active = False

        # Stop component services
        self._stop_all_components()

        logger.info("Scaling orchestration stopped")

    def _orchestration_loop(self):
        """Main orchestration monitoring loop."""
        while self.orchestrator_active:
            try:
                # Collect performance metrics
                self._collect_performance_metrics()

                # Make scaling decisions
                self._evaluate_scaling_decisions()

                # Update components
                self._update_components()

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                time.sleep(30)

    def _collect_performance_metrics(self):
        """Collect performance metrics from all components."""
        metrics = {}

        # Cache metrics
        if "cache" in self.components:
            metrics["cache"] = self.components["cache"].get_comprehensive_stats()

        # Resource metrics
        if "resource_monitor" in self.components:
            metrics["resources"] = self.components[
                "resource_monitor"
            ].get_current_metrics()

        # Parallel processing metrics
        if "parallel_engine" in self.components:
            metrics["parallel"] = self.components[
                "parallel_engine"
            ].get_assembly_stats()

        # Auto-scaling metrics
        if "autoscaling" in self.components:
            metrics["autoscaling"] = self.components["autoscaling"].get_scaling_stats()

        # Deployment metrics
        if "deployment" in self.components:
            metrics["deployment"] = self.components[
                "deployment"
            ].get_deployment_summary()

        self.performance_metrics = metrics

    def _evaluate_scaling_decisions(self):
        """Evaluate and make scaling decisions."""
        # Simple scaling logic based on resource usage
        if "resources" in self.performance_metrics:
            resource_metrics = self.performance_metrics["resources"]
            system_metrics = resource_metrics.get("system", {})

            cpu_usage = system_metrics.get("cpu_percent", 0)
            memory_usage = system_metrics.get("memory_percent", 0)

            # Auto-scaling based on resource usage
            if "autoscaling" in self.components:
                autoscaling = self.components["autoscaling"]

                if cpu_usage > 85 or memory_usage > 90:
                    # Scale up if high resource usage
                    current_workers = autoscaling.current_workers
                    if current_workers < autoscaling.max_workers:
                        new_workers = min(current_workers + 1, autoscaling.max_workers)
                        autoscaling.manual_scale(new_workers)

                        logger.info(
                            f"Auto-scaled up: {current_workers} -> {new_workers} workers"
                        )

                elif cpu_usage < 30 and memory_usage < 40:
                    # Scale down if low resource usage
                    current_workers = autoscaling.current_workers
                    if current_workers > autoscaling.min_workers:
                        new_workers = max(current_workers - 1, autoscaling.min_workers)
                        autoscaling.manual_scale(new_workers)

                        logger.info(
                            f"Auto-scaled down: {current_workers} -> {new_workers} workers"
                        )

    def _update_components(self):
        """Update component configurations based on current performance."""
        # Dashboard updates
        if "dashboard" in self.components and "resources" in self.performance_metrics:
            dashboard = self.components["dashboard"]
            resource_metrics = self.performance_metrics["resources"]
            system_metrics = resource_metrics.get("system", {})

            # Record metrics to dashboard
            for metric_name, value in system_metrics.items():
                if isinstance(value, (int, float)):
                    dashboard.record_metric(metric_name, value, unit="%")

    def _stop_all_components(self):
        """Stop all component services."""
        # Stop monitoring
        if "resource_monitor" in self.components:
            self.components["resource_monitor"].stop_monitoring()

        # Stop auto-scaling
        if "autoscaling" in self.components:
            self.components["autoscaling"].stop_auto_scaling()

        # Stop deployment health monitoring
        if "deployment" in self.components:
            self.components["deployment"].stop_health_monitoring()

        # Shutdown batch processor
        if "batch_processor" in self.components:
            self.components["batch_processor"].shutdown()

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "orchestrator_active": self.orchestrator_active,
            "configuration": {
                "mode": self.config.mode.value,
                "profile": self.config.profile.value,
            },
            "initialized_components": list(self.initialized_components),
            "performance_metrics": self.performance_metrics,
            "component_count": len(self.components),
            "scaling_history_count": len(self.scaling_history),
        }

    def create_deployment_package(self, output_dir: str = "./deployment_package"):
        """Create complete deployment package with all configurations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main configuration
        self.save_configuration(str(output_path / "scaling_config.yaml"))

        # Create deployment configuration if multi-region is enabled
        if self.config.deployment_config.get("enable_multi_region", False):
            from .deployment import create_sample_deployment_config

            create_sample_deployment_config(str(output_path / "deployment_config.yaml"))

        # Create Docker configuration
        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8080 8000

# Start command
CMD ["python", "-m", "src.performance.scaling_config", "--config", "scaling_config.yaml"]
"""

        with open(output_path / "Dockerfile", "w") as f:
            f.write(dockerfile_content)

        # Create requirements.txt
        requirements = [
            "jax[cpu]>=0.4.0",
            "jaxlib>=0.4.0",
            "optax>=0.1.0",
            "flax>=0.6.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "psutil>=5.8.0",
            "flask>=2.0.0",
            "flask-socketio>=5.0.0",
            "plotly>=5.0.0",
            "pyyaml>=6.0",
            "redis",
            "pymemcache",
            "asyncio",
            "aiofiles",
            "aiohttp",
            "asyncpg",
        ]

        with open(output_path / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))

        # Create deployment scripts
        start_script = """#!/bin/bash
# Start the scaling system

echo "Starting PDE Solver Scaling System..."

# Check for configuration file
if [ ! -f "scaling_config.yaml" ]; then
    echo "Configuration file not found. Creating default configuration..."
    python -c "
from src.performance.scaling_config import ScalingOrchestrator
orchestrator = ScalingOrchestrator()
orchestrator.save_configuration('scaling_config.yaml')
"
fi

# Start the orchestrator
python -m src.performance.scaling_config --config scaling_config.yaml --start

echo "Scaling system started successfully!"
"""

        with open(output_path / "start.sh", "w") as f:
            f.write(start_script)

        # Make script executable
        import stat

        os.chmod(output_path / "start.sh", stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

        # Create Kubernetes deployment manifest
        k8s_manifest = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pde-solver-scaling
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pde-solver-scaling
  template:
    metadata:
      labels:
        app: pde-solver-scaling
    spec:
      containers:
      - name: pde-solver
        image: pde-solver-scaling:latest
        ports:
        - containerPort: 8080
        - containerPort: 8000
        env:
        - name: SCALING_MODE
          value: "production"
        - name: PERFORMANCE_PROFILE
          value: "balanced"
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: pde-solver-service
spec:
  selector:
    app: pde-solver-scaling
  ports:
    - name: dashboard
      protocol: TCP
      port: 8080
      targetPort: 8080
    - name: api
      protocol: TCP
      port: 8000
      targetPort: 8000
  type: LoadBalancer
"""

        with open(output_path / "kubernetes.yaml", "w") as f:
            f.write(k8s_manifest)

        logger.info(f"Deployment package created in {output_dir}")
        return str(output_path)


# Global orchestrator instance
_global_orchestrator = None


def get_orchestrator(config_file: str = None) -> ScalingOrchestrator:
    """Get global scaling orchestrator."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = ScalingOrchestrator(config_file=config_file)
    return _global_orchestrator


def create_sample_scaling_config(output_file: str = "scaling_config.yaml"):
    """Create sample scaling configuration."""
    config = ScalingConfiguration(
        mode=ScalingMode.PRODUCTION, profile=PerformanceProfile.BALANCED
    )

    orchestrator = ScalingOrchestrator(config=config)
    orchestrator.save_configuration(output_file)

    logger.info(f"Sample scaling configuration created: {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDE Solver Scaling System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument(
        "--start", action="store_true", help="Start the orchestration system"
    )
    parser.add_argument("--create-config", help="Create sample configuration file")
    parser.add_argument("--create-package", help="Create deployment package")

    args = parser.parse_args()

    if args.create_config:
        create_sample_scaling_config(args.create_config)
    elif args.create_package:
        orchestrator = get_orchestrator(args.config)
        orchestrator.create_deployment_package(args.create_package)
    elif args.start:
        orchestrator = get_orchestrator(args.config)
        orchestrator.start_orchestration()

        # Keep running
        try:
            while True:
                time.sleep(60)
                status = orchestrator.get_system_status()
                logger.info(
                    f"System status: {status['component_count']} components active"
                )
        except KeyboardInterrupt:
            logger.info("Stopping orchestration system...")
            orchestrator.stop_orchestration()
    else:
        print("Use --help for usage information")
