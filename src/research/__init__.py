"""Research-level algorithms and methodologies for DiffFE-Physics-Lab.

This module contains cutting-edge research implementations including:
- Novel adaptive optimization algorithms with convergence guarantees
- Quantum-inspired variational methods for large-scale problems
- Statistical validation frameworks for algorithm comparison
- Machine learning acceleration techniques with neural operators
"""

from .adaptive_algorithms import (
    AdaptiveOptimizerBase,
    PhysicsInformedAdaptiveOptimizer,
    MultiScaleAdaptiveOptimizer,
    BayesianAdaptiveOptimizer,
    OptimizationMetrics,
    compare_optimizers,
    research_optimization_experiment
)

__all__ = [
    "AdaptiveOptimizerBase",
    "PhysicsInformedAdaptiveOptimizer", 
    "MultiScaleAdaptiveOptimizer",
    "BayesianAdaptiveOptimizer",
    "OptimizationMetrics",
    "compare_optimizers",
    "research_optimization_experiment",
]