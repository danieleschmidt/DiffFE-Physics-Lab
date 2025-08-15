"""ML Acceleration Module for DiffFE-Physics-Lab.

Advanced machine learning acceleration capabilities including neural operators,
physics-informed neural networks, and hybrid ML-physics solvers.
"""

from .neural_operators import FourierNeuralOperator, DeepONet, GraphNeuralOperator
from .physics_informed import PINNSolver, PhysicsLoss, AutomaticDifferentiation
from .hybrid_solvers import MLPhysicsHybrid, NeuralPreconditioner, AdaptiveMeshML
from .acceleration_engines import TensorRTEngine, ONNXOptimizer, MLCompiler
from .transfer_learning import PhysicsTransferLearning, DomainAdaptation, FewShotPDE

__all__ = [
    # Neural Operators
    "FourierNeuralOperator",
    "DeepONet", 
    "GraphNeuralOperator",
    
    # Physics-Informed ML
    "PINNSolver",
    "PhysicsLoss",
    "AutomaticDifferentiation",
    
    # Hybrid Solvers
    "MLPhysicsHybrid",
    "NeuralPreconditioner",
    "AdaptiveMeshML",
    
    # Acceleration Engines
    "TensorRTEngine",
    "ONNXOptimizer",
    "MLCompiler",
    
    # Transfer Learning
    "PhysicsTransferLearning",
    "DomainAdaptation",
    "FewShotPDE",
]

__version__ = "1.0.0-ml-dev"