"""Auto-scaling and resource management for DiffFE-Physics-Lab."""

import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    active_workers: int
    queue_length: int
    timestamp: float


class AutoScaler:
    """Automatic scaling system."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 16):
        """Initialize auto-scaler."""
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.scale_up_threshold = 80.0
        self.scale_down_threshold = 20.0
        self.cooldown_period = 30.0
        self.last_scale_time = 0.0
    
    def should_scale_up(self, metrics: ResourceMetrics) -> bool:
        """Check if should scale up."""
        return (metrics.cpu_percent > self.scale_up_threshold and
                self.current_workers < self.max_workers and
                time.time() - self.last_scale_time > self.cooldown_period)
    
    def should_scale_down(self, metrics: ResourceMetrics) -> bool:
        """Check if should scale down."""
        return (metrics.cpu_percent < self.scale_down_threshold and
                self.current_workers > self.min_workers and
                time.time() - self.last_scale_time > self.cooldown_period)
    
    def scale(self, metrics: ResourceMetrics) -> Optional[str]:
        """Perform scaling decision."""
        if self.should_scale_up(metrics):
            self.current_workers = min(self.max_workers, self.current_workers * 2)
            self.last_scale_time = time.time()
            return "scaled_up"
        elif self.should_scale_down(metrics):
            self.current_workers = max(self.min_workers, self.current_workers // 2)
            self.last_scale_time = time.time()
            return "scaled_down"
        return None


class ResourceManager:
    """Resource allocation and management."""
    
    def __init__(self):
        """Initialize resource manager."""
        self.resources = {"workers": 2, "memory_mb": 1000}
        self.allocations = {}
    
    def allocate(self, resource_type: str, amount: int) -> bool:
        """Allocate resources."""
        if self.resources.get(resource_type, 0) >= amount:
            self.resources[resource_type] -= amount
            return True
        return False
    
    def deallocate(self, resource_type: str, amount: int):
        """Deallocate resources."""
        self.resources[resource_type] += amount


class LoadBalancer:
    """Load balancing system."""
    
    def __init__(self):
        """Initialize load balancer."""
        self.workers = []
        self.current_index = 0
    
    def add_worker(self, worker_id: str):
        """Add worker to pool."""
        self.workers.append(worker_id)
    
    def get_next_worker(self) -> Optional[str]:
        """Get next worker using round-robin."""
        if not self.workers:
            return None
        
        worker = self.workers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.workers)
        return worker


def scale_workers(current_count: int, target_count: int) -> int:
    """Scale worker count."""
    return target_count


def adaptive_resources(metrics: Dict[str, float]) -> Dict[str, int]:
    """Adapt resource allocation based on metrics."""
    return {"workers": 2, "memory_mb": 1000}