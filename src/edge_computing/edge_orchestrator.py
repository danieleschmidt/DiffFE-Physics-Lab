"""Edge Computing Orchestration for Distributed Physics Simulation.

Advanced orchestration system for managing distributed PDE solving across
heterogeneous edge devices with intelligent load balancing and fault tolerance.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import logging
import time
import asyncio
import threading
import json
import socket
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import uuid

from .real_time_solver import RealTimeSolver, RealTimeConfig
from ..utils.validation import validate_device_capabilities


class DeviceType(Enum):
    """Types of edge devices."""
    SMARTPHONE = "smartphone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    EDGE_SERVER = "edge_server"
    IOT_DEVICE = "iot_device"
    GPU_CLUSTER = "gpu_cluster"
    CUSTOM = "custom"


class ComputeCapability(Enum):
    """Compute capability levels."""
    MINIMAL = "minimal"      # Basic arithmetic only
    LOW = "low"              # Simple PDE solving
    MEDIUM = "medium"        # Standard FEM problems
    HIGH = "high"            # Complex multi-physics
    EXTREME = "extreme"      # Large-scale simulations


@dataclass
class DeviceProfile:
    """Profile of edge device capabilities."""
    device_id: str
    device_type: DeviceType
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    network_bandwidth_mbps: float
    compute_capability: ComputeCapability
    battery_level: Optional[float] = None  # For mobile devices
    power_consumption_watts: float = 100.0
    thermal_state: str = "normal"  # normal, warm, hot, critical
    availability: float = 1.0  # 0.0 to 1.0
    reliability_score: float = 1.0  # Historical reliability
    current_load: float = 0.0  # 0.0 to 1.0
    specializations: List[str] = None  # e.g., ["fluid_dynamics", "heat_transfer"]
    
    def __post_init__(self):
        if self.specializations is None:
            self.specializations = []


@dataclass
class ComputeTask:
    """Computational task for edge device."""
    task_id: str
    problem_type: str
    priority: int  # 1-10, higher is more important
    deadline_ms: float
    estimated_compute_time_ms: float
    memory_requirement_mb: float
    data_size_mb: float
    dependencies: List[str]  # Task IDs this depends on
    preferred_device_types: List[DeviceType]
    problem_data: Dict[str, Any]
    created_time: float
    assigned_device: Optional[str] = None
    status: str = "pending"  # pending, assigned, running, completed, failed


class ComputeNode:
    """Individual compute node (edge device) in the distributed system."""
    
    def __init__(self, device_profile: DeviceProfile):
        self.profile = device_profile
        self.solver = None
        self.is_active = True
        self.current_tasks = {}
        self.completed_tasks = []
        self.performance_history = []
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.heartbeat_interval = 5.0  # seconds
        self.last_heartbeat = time.time()
        
        # Performance monitoring
        self.total_tasks_completed = 0
        self.total_compute_time = 0.0
        self.average_task_time = 0.0
        self.error_count = 0
        
        # Initialize solver with device-appropriate configuration
        self._initialize_solver()
        
        logging.info(f"Compute node {self.profile.device_id} initialized")
    
    def _initialize_solver(self):
        """Initialize solver with device-specific configuration."""
        config = RealTimeConfig()
        
        # Adjust configuration based on device capabilities
        if self.profile.compute_capability == ComputeCapability.MINIMAL:
            config.target_latency_ms = 50.0
            config.precision_tolerance = 1e-2
            config.memory_budget_mb = 64
        elif self.profile.compute_capability == ComputeCapability.LOW:
            config.target_latency_ms = 20.0
            config.precision_tolerance = 1e-3
            config.memory_budget_mb = 256
        elif self.profile.compute_capability == ComputeCapability.MEDIUM:
            config.target_latency_ms = 10.0
            config.precision_tolerance = 1e-4
            config.memory_budget_mb = 512
        elif self.profile.compute_capability == ComputeCapability.HIGH:
            config.target_latency_ms = 5.0
            config.precision_tolerance = 1e-5
            config.memory_budget_mb = 1024
        else:  # EXTREME
            config.target_latency_ms = 1.0
            config.precision_tolerance = 1e-6
            config.memory_budget_mb = 2048
        
        config.cpu_cores = self.profile.cpu_cores
        config.gpu_available = self.profile.gpu_available
        
        self.solver = RealTimeSolver(config)
    
    async def process_task(self, task: ComputeTask) -> Tuple[Any, Dict[str, Any]]:
        """Process computational task on this node."""
        start_time = time.time()
        task.status = "running"
        task.assigned_device = self.profile.device_id
        
        try:
            # Update current load
            self.profile.current_load = len(self.current_tasks) / self.profile.cpu_cores
            
            # Solve the problem
            solution, metadata = self.solver.solve_real_time(
                task.problem_data, 
                deadline_ms=task.deadline_ms
            )
            
            # Record performance
            compute_time = (time.time() - start_time) * 1000  # ms
            self._record_task_performance(task, compute_time, True)
            
            task.status = "completed"
            metadata.update({
                "device_id": self.profile.device_id,
                "actual_compute_time_ms": compute_time,
                "device_type": self.profile.device_type.value,
            })
            
            return solution, metadata
            
        except Exception as e:
            # Handle errors gracefully
            compute_time = (time.time() - start_time) * 1000
            self._record_task_performance(task, compute_time, False)
            
            task.status = "failed"
            error_metadata = {
                "error": str(e),
                "device_id": self.profile.device_id,
                "compute_time_ms": compute_time,
            }
            
            logging.error(f"Task {task.task_id} failed on device {self.profile.device_id}: {e}")
            
            return None, error_metadata
    
    def _record_task_performance(self, task: ComputeTask, compute_time_ms: float, success: bool):
        """Record task performance metrics."""
        self.performance_history.append({
            "task_id": task.task_id,
            "compute_time_ms": compute_time_ms,
            "success": success,
            "problem_type": task.problem_type,
            "timestamp": time.time(),
        })
        
        # Update statistics
        if success:
            self.total_tasks_completed += 1
            self.total_compute_time += compute_time_ms
            self.average_task_time = self.total_compute_time / self.total_tasks_completed
        else:
            self.error_count += 1
        
        # Update reliability score
        total_tasks = self.total_tasks_completed + self.error_count
        if total_tasks > 0:
            self.profile.reliability_score = self.total_tasks_completed / total_tasks
        
        # Maintain performance history size
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for this node."""
        return {
            "device_id": self.profile.device_id,
            "total_tasks_completed": self.total_tasks_completed,
            "average_task_time_ms": self.average_task_time,
            "error_rate": self.error_count / max(1, self.total_tasks_completed + self.error_count),
            "reliability_score": self.profile.reliability_score,
            "current_load": self.profile.current_load,
            "solver_performance": self.solver.get_performance_summary() if self.solver else {},
        }
    
    def can_handle_task(self, task: ComputeTask) -> bool:
        """Check if this node can handle the given task."""
        # Check memory requirements
        if task.memory_requirement_mb > self.profile.memory_gb * 1024:
            return False
        
        # Check device type preferences
        if task.preferred_device_types and self.profile.device_type not in task.preferred_device_types:
            return False
        
        # Check current load
        if self.profile.current_load > 0.9:  # 90% load threshold
            return False
        
        # Check specializations
        if task.problem_type in self.profile.specializations:
            return True  # Prefer specialized devices
        
        # Check compute capability
        required_capability = self._estimate_required_capability(task)
        return self.profile.compute_capability.value >= required_capability.value
    
    def _estimate_required_capability(self, task: ComputeTask) -> ComputeCapability:
        """Estimate required compute capability for task."""
        if task.estimated_compute_time_ms > 1000:  # > 1 second
            return ComputeCapability.EXTREME
        elif task.estimated_compute_time_ms > 100:  # > 100ms
            return ComputeCapability.HIGH
        elif task.estimated_compute_time_ms > 20:   # > 20ms
            return ComputeCapability.MEDIUM
        elif task.estimated_compute_time_ms > 5:    # > 5ms
            return ComputeCapability.LOW
        else:
            return ComputeCapability.MINIMAL
    
    async def send_heartbeat(self) -> Dict[str, Any]:
        """Send heartbeat with current status."""
        self.last_heartbeat = time.time()
        
        # Update system metrics
        self.profile.current_load = psutil.cpu_percent() / 100.0
        memory_info = psutil.virtual_memory()
        self.profile.memory_gb = memory_info.total / (1024**3)
        
        # Update thermal state (simplified)
        cpu_temp = self._get_cpu_temperature()
        if cpu_temp > 80:
            self.profile.thermal_state = "critical"
        elif cpu_temp > 70:
            self.profile.thermal_state = "hot"
        elif cpu_temp > 60:
            self.profile.thermal_state = "warm"
        else:
            self.profile.thermal_state = "normal"
        
        heartbeat_data = {
            "device_id": self.profile.device_id,
            "timestamp": self.last_heartbeat,
            "status": "active" if self.is_active else "inactive",
            "current_load": self.profile.current_load,
            "thermal_state": self.profile.thermal_state,
            "active_tasks": len(self.current_tasks),
            "performance_metrics": self.get_performance_metrics(),
        }
        
        return heartbeat_data
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (simplified implementation)."""
        try:
            # This would use platform-specific temperature sensors
            # For now, return simulated temperature based on load
            return 40 + self.profile.current_load * 40  # 40-80°C range
        except:
            return 50.0  # Default temperature


class DeviceManager:
    """Manager for discovering and managing edge devices."""
    
    def __init__(self):
        self.devices: Dict[str, ComputeNode] = {}
        self.device_discovery_active = False
        self.heartbeat_timeout = 30.0  # seconds
        
        # Device clustering for locality
        self.device_clusters = {}
        self.network_topology = {}
        
        logging.info("Device manager initialized")
    
    def register_device(self, device_profile: DeviceProfile) -> ComputeNode:
        """Register a new edge device."""
        node = ComputeNode(device_profile)
        self.devices[device_profile.device_id] = node
        
        # Add to appropriate cluster
        self._cluster_device(node)
        
        logging.info(f"Registered device {device_profile.device_id} "
                    f"({device_profile.device_type.value}, "
                    f"{device_profile.compute_capability.value})")
        
        return node
    
    def unregister_device(self, device_id: str):
        """Unregister an edge device."""
        if device_id in self.devices:
            device = self.devices[device_id]
            device.is_active = False
            del self.devices[device_id]
            logging.info(f"Unregistered device {device_id}")
    
    def _cluster_device(self, node: ComputeNode):
        """Organize devices into clusters for efficient task distribution."""
        cluster_key = f"{node.profile.device_type.value}_{node.profile.compute_capability.value}"
        
        if cluster_key not in self.device_clusters:
            self.device_clusters[cluster_key] = []
        
        self.device_clusters[cluster_key].append(node.profile.device_id)
    
    def discover_devices(self) -> List[DeviceProfile]:
        """Discover available edge devices on the network."""
        discovered_devices = []
        
        # This would implement network discovery protocols
        # For now, return simulated devices
        simulated_devices = self._generate_simulated_devices()
        
        for device_profile in simulated_devices:
            self.register_device(device_profile)
            discovered_devices.append(device_profile)
        
        return discovered_devices
    
    def _generate_simulated_devices(self) -> List[DeviceProfile]:
        """Generate simulated edge devices for testing."""
        devices = []
        
        # High-performance edge server
        devices.append(DeviceProfile(
            device_id="edge_server_001",
            device_type=DeviceType.EDGE_SERVER,
            cpu_cores=16,
            memory_gb=64,
            gpu_available=True,
            gpu_memory_gb=24,
            network_bandwidth_mbps=1000,
            compute_capability=ComputeCapability.EXTREME,
            power_consumption_watts=500,
            specializations=["fluid_dynamics", "electromagnetics"]
        ))
        
        # Medium laptop
        devices.append(DeviceProfile(
            device_id="laptop_001",
            device_type=DeviceType.LAPTOP,
            cpu_cores=8,
            memory_gb=16,
            gpu_available=True,
            gpu_memory_gb=4,
            network_bandwidth_mbps=100,
            compute_capability=ComputeCapability.HIGH,
            battery_level=0.8,
            power_consumption_watts=65,
            specializations=["heat_transfer"]
        ))
        
        # Smartphone
        devices.append(DeviceProfile(
            device_id="smartphone_001",
            device_type=DeviceType.SMARTPHONE,
            cpu_cores=8,
            memory_gb=8,
            gpu_available=True,
            gpu_memory_gb=2,
            network_bandwidth_mbps=50,
            compute_capability=ComputeCapability.MEDIUM,
            battery_level=0.6,
            power_consumption_watts=5,
        ))
        
        # IoT device
        devices.append(DeviceProfile(
            device_id="iot_001",
            device_type=DeviceType.IOT_DEVICE,
            cpu_cores=2,
            memory_gb=1,
            gpu_available=False,
            gpu_memory_gb=0,
            network_bandwidth_mbps=10,
            compute_capability=ComputeCapability.LOW,
            power_consumption_watts=2,
        ))
        
        return devices
    
    def get_available_devices(self, task_requirements: Optional[Dict] = None) -> List[ComputeNode]:
        """Get list of available devices that meet task requirements."""
        available = []
        
        for device in self.devices.values():
            if not device.is_active:
                continue
            
            # Check if device has sent heartbeat recently
            if time.time() - device.last_heartbeat > self.heartbeat_timeout:
                device.is_active = False
                continue
            
            # Check task requirements if provided
            if task_requirements:
                dummy_task = ComputeTask(
                    task_id="dummy",
                    problem_type=task_requirements.get("problem_type", "generic"),
                    priority=1,
                    deadline_ms=task_requirements.get("deadline_ms", 1000),
                    estimated_compute_time_ms=task_requirements.get("compute_time_ms", 100),
                    memory_requirement_mb=task_requirements.get("memory_mb", 256),
                    data_size_mb=task_requirements.get("data_mb", 10),
                    dependencies=[],
                    preferred_device_types=task_requirements.get("device_types", []),
                    problem_data={}
                )
                
                if device.can_handle_task(dummy_task):
                    available.append(device)
            else:
                available.append(device)
        
        return available
    
    def get_device_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed devices."""
        total_devices = len(self.devices)
        active_devices = sum(1 for d in self.devices.values() if d.is_active)
        
        device_type_counts = {}
        capability_counts = {}
        
        for device in self.devices.values():
            device_type = device.profile.device_type.value
            capability = device.profile.compute_capability.value
            
            device_type_counts[device_type] = device_type_counts.get(device_type, 0) + 1
            capability_counts[capability] = capability_counts.get(capability, 0) + 1
        
        total_compute_cores = sum(d.profile.cpu_cores for d in self.devices.values())
        total_memory_gb = sum(d.profile.memory_gb for d in self.devices.values())
        
        return {
            "total_devices": total_devices,
            "active_devices": active_devices,
            "device_type_distribution": device_type_counts,
            "capability_distribution": capability_counts,
            "total_compute_cores": total_compute_cores,
            "total_memory_gb": total_memory_gb,
            "clusters": {k: len(v) for k, v in self.device_clusters.items()},
        }


class LoadBalancer:
    """Intelligent load balancer for distributing tasks across edge devices."""
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.balancing_strategy = "adaptive"  # round_robin, least_loaded, adaptive, ml_based
        self.task_history = []
        self.performance_predictor = None
        
        # Load balancing metrics
        self.total_tasks_distributed = 0
        self.load_balancing_decisions = []
        
        logging.info("Load balancer initialized")
    
    def select_device(self, task: ComputeTask) -> Optional[ComputeNode]:
        """Select optimal device for task execution."""
        available_devices = self.device_manager.get_available_devices()
        
        # Filter devices that can handle the task
        capable_devices = [d for d in available_devices if d.can_handle_task(task)]
        
        if not capable_devices:
            logging.warning(f"No capable devices found for task {task.task_id}")
            return None
        
        # Select device based on strategy
        if self.balancing_strategy == "round_robin":
            selected_device = self._round_robin_selection(capable_devices)
        elif self.balancing_strategy == "least_loaded":
            selected_device = self._least_loaded_selection(capable_devices)
        elif self.balancing_strategy == "adaptive":
            selected_device = self._adaptive_selection(capable_devices, task)
        elif self.balancing_strategy == "ml_based":
            selected_device = self._ml_based_selection(capable_devices, task)
        else:
            # Default to least loaded
            selected_device = self._least_loaded_selection(capable_devices)
        
        # Record decision
        self._record_balancing_decision(task, selected_device, capable_devices)
        
        return selected_device
    
    def _round_robin_selection(self, devices: List[ComputeNode]) -> ComputeNode:
        """Simple round-robin device selection."""
        if not devices:
            return None
        
        # Use task count modulo for round-robin
        index = self.total_tasks_distributed % len(devices)
        return devices[index]
    
    def _least_loaded_selection(self, devices: List[ComputeNode]) -> ComputeNode:
        """Select device with lowest current load."""
        if not devices:
            return None
        
        return min(devices, key=lambda d: d.profile.current_load)
    
    def _adaptive_selection(self, devices: List[ComputeNode], task: ComputeTask) -> ComputeNode:
        """Adaptive selection considering multiple factors."""
        if not devices:
            return None
        
        scores = []
        
        for device in devices:
            score = self._compute_device_score(device, task)
            scores.append((device, score))
        
        # Select device with highest score
        best_device, best_score = max(scores, key=lambda x: x[1])
        
        logging.debug(f"Selected device {best_device.profile.device_id} "
                     f"with score {best_score:.3f} for task {task.task_id}")
        
        return best_device
    
    def _compute_device_score(self, device: ComputeNode, task: ComputeTask) -> float:
        """Compute selection score for device-task pair."""
        score = 0.0
        
        # Load factor (prefer less loaded devices)
        load_factor = 1.0 - device.profile.current_load
        score += load_factor * 0.3
        
        # Capability factor (prefer appropriate capability level)
        required_cap = device._estimate_required_capability(task)
        device_cap = device.profile.compute_capability
        
        if device_cap == required_cap:
            capability_factor = 1.0  # Perfect match
        elif device_cap.value > required_cap.value:
            capability_factor = 0.8  # Overqualified (wastes resources)
        else:
            capability_factor = 0.2  # Underqualified (may fail)
        
        score += capability_factor * 0.25
        
        # Reliability factor
        score += device.profile.reliability_score * 0.2
        
        # Specialization factor
        if task.problem_type in device.profile.specializations:
            score += 0.15
        
        # Performance history factor
        if device.performance_history:
            recent_performance = device.performance_history[-10:]  # Last 10 tasks
            avg_task_time = np.mean([p["compute_time_ms"] for p in recent_performance])
            
            # Prefer faster devices
            if avg_task_time < task.estimated_compute_time_ms:
                score += 0.1
        
        # Power efficiency factor (for mobile devices)
        if device.profile.battery_level is not None:
            if device.profile.battery_level > 0.3:  # Good battery level
                score += 0.05
            else:
                score -= 0.1  # Low battery penalty
        
        # Thermal state factor
        thermal_penalties = {
            "normal": 0.0,
            "warm": -0.05,
            "hot": -0.15,
            "critical": -0.5
        }
        score += thermal_penalties.get(device.profile.thermal_state, 0.0)
        
        return max(0.0, score)  # Ensure non-negative score
    
    def _ml_based_selection(self, devices: List[ComputeNode], task: ComputeTask) -> ComputeNode:
        """ML-based device selection using performance prediction."""
        # This would use a trained model to predict task completion time
        # For now, fall back to adaptive selection
        return self._adaptive_selection(devices, task)
    
    def _record_balancing_decision(self, task: ComputeTask, selected_device: ComputeNode,
                                 available_devices: List[ComputeNode]):
        """Record load balancing decision for analysis."""
        decision_record = {
            "task_id": task.task_id,
            "selected_device_id": selected_device.profile.device_id if selected_device else None,
            "num_available_devices": len(available_devices),
            "task_priority": task.priority,
            "task_deadline_ms": task.deadline_ms,
            "timestamp": time.time(),
            "strategy": self.balancing_strategy,
        }
        
        self.load_balancing_decisions.append(decision_record)
        self.total_tasks_distributed += 1
        
        # Maintain decision history size
        if len(self.load_balancing_decisions) > 10000:
            self.load_balancing_decisions = self.load_balancing_decisions[-5000:]
    
    def analyze_balancing_performance(self) -> Dict[str, Any]:
        """Analyze load balancing performance and effectiveness."""
        if not self.load_balancing_decisions:
            return {"error": "No balancing decisions recorded"}
        
        # Device utilization analysis
        device_task_counts = {}
        for decision in self.load_balancing_decisions:
            device_id = decision.get("selected_device_id")
            if device_id:
                device_task_counts[device_id] = device_task_counts.get(device_id, 0) + 1
        
        # Load distribution fairness (coefficient of variation)
        if device_task_counts:
            task_counts = list(device_task_counts.values())
            load_mean = np.mean(task_counts)
            load_std = np.std(task_counts)
            load_fairness = load_std / load_mean if load_mean > 0 else 0.0
        else:
            load_fairness = 0.0
        
        # Recent performance trends
        recent_decisions = self.load_balancing_decisions[-100:]  # Last 100 decisions
        
        analysis = {
            "total_tasks_distributed": self.total_tasks_distributed,
            "unique_devices_used": len(device_task_counts),
            "load_distribution_fairness": load_fairness,
            "device_utilization": device_task_counts,
            "strategy": self.balancing_strategy,
            "recent_decision_count": len(recent_decisions),
        }
        
        return analysis
    
    def optimize_strategy(self):
        """Optimize load balancing strategy based on historical performance."""
        analysis = self.analyze_balancing_performance()
        
        # Simple optimization rules
        if analysis.get("load_distribution_fairness", 0) > 0.5:
            # High variation in load distribution - switch to least_loaded
            self.balancing_strategy = "least_loaded"
            logging.info("Optimized strategy to least_loaded due to load imbalance")
        
        elif len(analysis.get("device_utilization", {})) > 10:
            # Many devices - use adaptive strategy
            self.balancing_strategy = "adaptive"
            logging.info("Optimized strategy to adaptive for multi-device environment")


class EdgeOrchestrator:
    """Main orchestrator for edge computing distributed PDE solving."""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.load_balancer = LoadBalancer(self.device_manager)
        
        # Task management
        self.pending_tasks = []
        self.running_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        
        # Orchestration state
        self.is_running = False
        self.task_executor = ThreadPoolExecutor(max_workers=50)
        
        # Performance monitoring
        self.total_throughput = 0.0  # tasks per second
        self.average_latency = 0.0   # milliseconds
        self.system_efficiency = 0.0  # 0.0 to 1.0
        
        logging.info("Edge orchestrator initialized")
    
    async def start(self):
        """Start the edge orchestration system."""
        self.is_running = True
        
        # Discover and register devices
        discovered_devices = self.device_manager.discover_devices()
        logging.info(f"Discovered {len(discovered_devices)} edge devices")
        
        # Start background tasks
        orchestration_tasks = [
            asyncio.create_task(self._task_scheduler()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._performance_optimizer()),
        ]
        
        # Wait for all background tasks
        await asyncio.gather(*orchestration_tasks)
    
    def stop(self):
        """Stop the edge orchestration system."""
        self.is_running = False
        self.task_executor.shutdown(wait=True)
        logging.info("Edge orchestrator stopped")
    
    def submit_task(self, problem: Dict[str, Any], priority: int = 5,
                   deadline_ms: float = 1000.0) -> str:
        """Submit a computational task to the edge system."""
        task_id = str(uuid.uuid4())
        
        # Estimate task requirements
        estimated_compute_time = self._estimate_compute_time(problem)
        memory_requirement = self._estimate_memory_requirement(problem)
        data_size = self._estimate_data_size(problem)
        
        task = ComputeTask(
            task_id=task_id,
            problem_type=problem.get("type", "generic"),
            priority=priority,
            deadline_ms=deadline_ms,
            estimated_compute_time_ms=estimated_compute_time,
            memory_requirement_mb=memory_requirement,
            data_size_mb=data_size,
            dependencies=[],
            preferred_device_types=[],
            problem_data=problem,
            created_time=time.time()
        )
        
        self.pending_tasks.append(task)
        logging.debug(f"Submitted task {task_id} with priority {priority}")
        
        return task_id
    
    def _estimate_compute_time(self, problem: Dict[str, Any]) -> float:
        """Estimate computational time for problem."""
        problem_type = problem.get("type", "generic")
        
        # Simple heuristics based on problem type and size
        if "matrix" in problem:
            n = problem["matrix"].shape[0]
            base_time = n**1.5 / 1000.0  # Rough estimate
        elif "grid_size" in problem:
            grid_size = problem["grid_size"]
            base_time = grid_size**2 / 10000.0
        else:
            base_time = 10.0  # Default 10ms
        
        # Adjust based on problem type
        type_multipliers = {
            "heat_equation": 1.0,
            "wave_equation": 1.2,
            "laplacian": 0.8,
            "navier_stokes": 3.0,
            "electromagnetics": 2.5,
        }
        
        multiplier = type_multipliers.get(problem_type, 1.0)
        return base_time * multiplier
    
    def _estimate_memory_requirement(self, problem: Dict[str, Any]) -> float:
        """Estimate memory requirement in MB."""
        if "matrix" in problem:
            n = problem["matrix"].shape[0]
            # Rough estimate: matrix storage + workspace
            return n * n * 8 / (1024**2) * 3  # 8 bytes per float64, 3x for workspace
        else:
            return 64.0  # Default 64MB
    
    def _estimate_data_size(self, problem: Dict[str, Any]) -> float:
        """Estimate data transfer size in MB."""
        # This would calculate serialized problem size
        # For now, use simple estimate
        return 1.0  # Default 1MB
    
    async def _task_scheduler(self):
        """Background task scheduler."""
        while self.is_running:
            if not self.pending_tasks:
                await asyncio.sleep(0.1)  # 100ms polling
                continue
            
            # Sort tasks by priority and deadline
            self.pending_tasks.sort(key=lambda t: (-t.priority, t.deadline_ms))
            
            # Process tasks that can be scheduled
            scheduled_count = 0
            tasks_to_remove = []
            
            for task in self.pending_tasks:
                # Check if task has exceeded deadline
                elapsed_time = (time.time() - task.created_time) * 1000
                if elapsed_time > task.deadline_ms:
                    task.status = "failed"
                    self.failed_tasks.append(task)
                    tasks_to_remove.append(task)
                    logging.warning(f"Task {task.task_id} exceeded deadline")
                    continue
                
                # Try to schedule task
                selected_device = self.load_balancer.select_device(task)
                if selected_device:
                    # Submit task to device
                    future = self.task_executor.submit(
                        self._execute_task_on_device, task, selected_device)
                    
                    self.running_tasks[task.task_id] = {
                        "task": task,
                        "device": selected_device,
                        "future": future,
                        "start_time": time.time()
                    }
                    
                    tasks_to_remove.append(task)
                    scheduled_count += 1
                    
                    logging.debug(f"Scheduled task {task.task_id} "
                                f"on device {selected_device.profile.device_id}")
            
            # Remove scheduled tasks from pending list
            for task in tasks_to_remove:
                self.pending_tasks.remove(task)
            
            if scheduled_count > 0:
                logging.info(f"Scheduled {scheduled_count} tasks")
            
            await asyncio.sleep(0.01)  # 10ms scheduling loop
    
    def _execute_task_on_device(self, task: ComputeTask, device: ComputeNode) -> Tuple[Any, Dict[str, Any]]:
        """Execute task on selected device."""
        try:
            # This would be async in a real implementation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            solution, metadata = loop.run_until_complete(device.process_task(task))
            
            # Move task to completed
            self.completed_tasks.append(task)
            
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            return solution, metadata
            
        except Exception as e:
            task.status = "failed"
            self.failed_tasks.append(task)
            
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            logging.error(f"Task execution failed: {e}")
            return None, {"error": str(e)}
    
    async def _health_monitor(self):
        """Monitor health of edge devices and system."""
        while self.is_running:
            # Collect heartbeats from all devices
            for device in self.device_manager.devices.values():
                try:
                    heartbeat = await device.send_heartbeat()
                    # Process heartbeat data for health monitoring
                    
                    # Check for unhealthy conditions
                    if heartbeat["thermal_state"] == "critical":
                        logging.warning(f"Device {device.profile.device_id} "
                                      f"in critical thermal state")
                        device.profile.availability = 0.5  # Reduce availability
                    
                    elif heartbeat["current_load"] > 0.95:
                        logging.warning(f"Device {device.profile.device_id} "
                                      f"at {heartbeat['current_load']:.1%} load")
                
                except Exception as e:
                    logging.error(f"Health check failed for device "
                                f"{device.profile.device_id}: {e}")
                    device.is_active = False
            
            await asyncio.sleep(5.0)  # Health check every 5 seconds
    
    async def _performance_optimizer(self):
        """Optimize system performance based on metrics."""
        while self.is_running:
            # Update system performance metrics
            self._update_performance_metrics()
            
            # Optimize load balancing strategy
            self.load_balancer.optimize_strategy()
            
            # Optimize device configurations
            self._optimize_device_configurations()
            
            await asyncio.sleep(30.0)  # Optimize every 30 seconds
    
    def _update_performance_metrics(self):
        """Update system-wide performance metrics."""
        # Calculate throughput
        completed_count = len(self.completed_tasks)
        if completed_count > 0:
            recent_completions = [t for t in self.completed_tasks 
                                if time.time() - t.created_time < 60.0]  # Last 60 seconds
            self.total_throughput = len(recent_completions) / 60.0
        
        # Calculate average latency
        if self.completed_tasks:
            recent_tasks = self.completed_tasks[-100:]  # Last 100 tasks
            latencies = [(time.time() - t.created_time) * 1000 
                        for t in recent_tasks]
            self.average_latency = np.mean(latencies)
        
        # Calculate system efficiency
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        if total_tasks > 0:
            self.system_efficiency = len(self.completed_tasks) / total_tasks
    
    def _optimize_device_configurations(self):
        """Optimize individual device configurations."""
        for device in self.device_manager.devices.values():
            if not device.is_active:
                continue
            
            # Optimize solver configuration based on performance
            performance_metrics = device.get_performance_metrics()
            avg_task_time = performance_metrics.get("average_task_time_ms", 0)
            
            if avg_task_time > device.solver.config.target_latency_ms * 1.5:
                # Tasks taking too long - reduce precision
                device.solver.config.precision_tolerance *= 1.2
                logging.debug(f"Reduced precision for device {device.profile.device_id}")
            
            elif avg_task_time < device.solver.config.target_latency_ms * 0.5:
                # Tasks completing quickly - increase precision
                device.solver.config.precision_tolerance *= 0.9
                logging.debug(f"Increased precision for device {device.profile.device_id}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        device_stats = self.device_manager.get_device_statistics()
        load_balancing_stats = self.load_balancer.analyze_balancing_performance()
        
        status = {
            "system_state": "running" if self.is_running else "stopped",
            "devices": device_stats,
            "load_balancing": load_balancing_stats,
            "tasks": {
                "pending": len(self.pending_tasks),
                "running": len(self.running_tasks),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
            },
            "performance": {
                "throughput_tasks_per_second": self.total_throughput,
                "average_latency_ms": self.average_latency,
                "system_efficiency": self.system_efficiency,
            },
            "timestamp": time.time(),
        }
        
        return status