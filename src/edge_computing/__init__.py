"""Edge Computing Module for DiffFE-Physics-Lab.

High-performance edge computing capabilities for real-time physics simulation,
distributed processing, and adaptive computation across heterogeneous devices.
"""

from .real_time_solver import RealTimeSolver, StreamingPDESolver, AdaptiveTimestepping
from .edge_orchestrator import EdgeOrchestrator, DeviceManager, ComputeNode
from .distributed_fem import DistributedAssembly, MeshPartitioner, LoadBalancer
from .streaming_data import DataStreamer, ResultAggregator, CompressionEngine
from .adaptive_precision import PrecisionManager, DynamicPrecision, ErrorBudget
from .mobile_solver import MobileFEMSolver, ResourceConstrainedSolver

__all__ = [
    # Real-time solving
    "RealTimeSolver",
    "StreamingPDESolver", 
    "AdaptiveTimestepping",
    
    # Edge orchestration
    "EdgeOrchestrator",
    "DeviceManager",
    "ComputeNode",
    
    # Distributed computing
    "DistributedAssembly",
    "MeshPartitioner",
    "LoadBalancer",
    
    # Data streaming
    "DataStreamer",
    "ResultAggregator",
    "CompressionEngine",
    
    # Adaptive precision
    "PrecisionManager",
    "DynamicPrecision",
    "ErrorBudget",
    
    # Mobile computing
    "MobileFEMSolver",
    "ResourceConstrainedSolver",
]

__version__ = "1.0.0-edge-dev"