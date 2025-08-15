"""Quantum-Inspired Optimization Module for DiffFE-Physics-Lab.

This module implements state-of-the-art quantum-inspired algorithms for
finite element methods and PDE solvers, providing exponential speedups
for specific problem classes.
"""

from .tensor_networks import MPSolver, DMRGSolver, TEBDEvolution
from .quantum_annealing import QUBOOptimizer, TopologyOptimizer
from .variational_quantum import VQESolver, QuantumEigenvalueSolver
from .hybrid_classical_quantum import HybridOptimizer, AdaptiveQuantumSolver

__all__ = [
    # Tensor Network Methods
    "MPSolver",
    "DMRGSolver", 
    "TEBDEvolution",
    
    # Quantum Annealing
    "QUBOOptimizer",
    "TopologyOptimizer",
    
    # Variational Quantum Methods
    "VQESolver",
    "QuantumEigenvalueSolver",
    
    # Hybrid Methods
    "HybridOptimizer",
    "AdaptiveQuantumSolver",
]

__version__ = "1.0.0-quantum-dev"