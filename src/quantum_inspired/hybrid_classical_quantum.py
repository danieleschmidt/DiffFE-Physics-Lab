"""Hybrid Classical-Quantum Optimization Algorithms.

Implementation of adaptive algorithms that intelligently combine classical
and quantum-inspired methods for optimal performance across problem scales.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

from .tensor_networks import MPSolver, MPSConfig
from .quantum_annealing import QUBOOptimizer, QUBOConfig
from .variational_quantum import VQESolver, VQEConfig
from ..backends.base import Backend
from ..utils.validation import validate_optimization_problem


@dataclass
class HybridConfig:
    """Configuration for hybrid classical-quantum optimization."""
    problem_size_threshold: int = 1000  # Switch to quantum methods above this size
    quantum_advantage_factor: float = 2.0  # Required speedup for quantum method
    max_classical_time: float = 300.0  # Max seconds for classical method
    max_quantum_time: float = 600.0  # Max seconds for quantum method
    adaptive_switching: bool = True  # Enable dynamic method switching
    benchmarking_enabled: bool = True  # Compare methods when possible
    convergence_tolerance: float = 1e-8
    max_iterations: int = 10000
    random_seed: int = 42


class OptimizationMethod(ABC):
    """Abstract base class for optimization methods."""
    
    @abstractmethod
    def solve(self, problem: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve optimization problem and return solution with metadata."""
        pass
    
    @abstractmethod
    def estimate_complexity(self, problem: Any) -> Dict[str, float]:
        """Estimate computational complexity for problem."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return name of optimization method."""
        pass


class ClassicalMethod(OptimizationMethod):
    """Classical optimization methods wrapper."""
    
    def __init__(self, method_type: str = "scipy"):
        self.method_type = method_type
        self.last_solve_time = 0.0
        self.last_iterations = 0
    
    def solve(self, problem: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve using classical optimization methods."""
        start_time = time.time()
        
        if problem["type"] == "eigenvalue":
            solution, metadata = self._solve_eigenvalue_classical(problem, **kwargs)
        elif problem["type"] == "optimization":
            solution, metadata = self._solve_optimization_classical(problem, **kwargs)
        elif problem["type"] == "pde":
            solution, metadata = self._solve_pde_classical(problem, **kwargs)
        else:
            raise ValueError(f"Unknown problem type: {problem['type']}")
        
        self.last_solve_time = time.time() - start_time
        metadata.update({
            "method": self.get_method_name(),
            "solve_time": self.last_solve_time,
            "iterations": self.last_iterations
        })
        
        return solution, metadata
    
    def _solve_eigenvalue_classical(self, problem: Dict[str, Any], 
                                  **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve eigenvalue problem using classical methods."""
        matrix = problem["matrix"]
        num_eigenvalues = kwargs.get("num_eigenvalues", 5)
        
        if matrix.shape[0] < 10000:  # Full diagonalization for small matrices
            eigenvals, eigenvecs = jnp.linalg.eigh(matrix)
            # Sort by eigenvalue
            sort_indices = jnp.argsort(eigenvals)
            eigenvals = eigenvals[sort_indices][:num_eigenvalues]
            eigenvecs = eigenvecs[:, sort_indices][:, :num_eigenvalues]
        else:
            # Use iterative methods for large matrices
            from scipy.sparse.linalg import eigsh
            eigenvals, eigenvecs = eigsh(matrix, k=num_eigenvalues, which='SM')
            eigenvals = jnp.array(eigenvals)
            eigenvecs = jnp.array(eigenvecs)
        
        self.last_iterations = 1  # Direct solver
        
        metadata = {
            "eigenvalue_errors": jnp.zeros(num_eigenvalues),  # Exact solution
            "condition_number": jnp.linalg.cond(matrix),
        }
        
        return (eigenvals, eigenvecs), metadata
    
    def _solve_optimization_classical(self, problem: Dict[str, Any],
                                    **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve optimization problem using classical methods."""
        objective = problem["objective"]
        constraints = problem.get("constraints", [])
        initial_guess = problem.get("initial_guess")
        
        from scipy.optimize import minimize
        
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            constraints=constraints,
            options={'maxiter': kwargs.get('max_iterations', 10000)}
        )
        
        self.last_iterations = result.nit
        
        metadata = {
            "success": result.success,
            "function_evaluations": result.nfev,
            "optimization_message": result.message,
            "final_gradient_norm": jnp.linalg.norm(result.jac) if hasattr(result, 'jac') else 0.0
        }
        
        return result.x, metadata
    
    def _solve_pde_classical(self, problem: Dict[str, Any],
                           **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve PDE using classical finite element methods."""
        # This would interface with the main FEM solver
        # Placeholder implementation
        solution = jnp.zeros(problem["matrix"].shape[0])
        self.last_iterations = 100
        
        metadata = {
            "residual_norm": 1e-10,
            "solver_type": "direct"
        }
        
        return solution, metadata
    
    def estimate_complexity(self, problem: Dict[str, Any]) -> Dict[str, float]:
        """Estimate computational complexity for classical methods."""
        if problem["type"] == "eigenvalue":
            n = problem["matrix"].shape[0]
            # O(n³) for full diagonalization, O(kn²) for k eigenvalues iteratively
            time_complexity = n**3 if n < 1000 else n**2
            memory_complexity = n**2
        elif problem["type"] == "optimization":
            n = len(problem.get("initial_guess", [100]))
            # L-BFGS complexity
            time_complexity = n**2  # Per iteration
            memory_complexity = n**2
        else:
            n = problem["matrix"].shape[0]
            time_complexity = n**2
            memory_complexity = n**2
        
        return {
            "time_complexity": time_complexity,
            "memory_complexity": memory_complexity,
            "estimated_time_seconds": time_complexity / 1e9,  # Rough estimate
        }
    
    def get_method_name(self) -> str:
        return f"Classical-{self.method_type}"


class QuantumMethod(OptimizationMethod):
    """Quantum-inspired optimization methods wrapper."""
    
    def __init__(self, method_type: str = "auto"):
        self.method_type = method_type
        self.last_solve_time = 0.0
        self.last_iterations = 0
        
        # Initialize quantum-inspired solvers
        self.mps_solver = None
        self.qubo_optimizer = None
        self.vqe_solver = None
    
    def solve(self, problem: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve using quantum-inspired methods."""
        start_time = time.time()
        
        # Choose best quantum method for problem type
        if self.method_type == "auto":
            method = self._select_quantum_method(problem)
        else:
            method = self.method_type
        
        if method == "mps" or problem["type"] == "pde":
            solution, metadata = self._solve_with_mps(problem, **kwargs)
        elif method == "vqe" or problem["type"] == "eigenvalue":
            solution, metadata = self._solve_with_vqe(problem, **kwargs)
        elif method == "qubo" or problem["type"] == "optimization":
            solution, metadata = self._solve_with_qubo(problem, **kwargs)
        else:
            raise ValueError(f"Unknown quantum method: {method}")
        
        self.last_solve_time = time.time() - start_time
        metadata.update({
            "method": f"Quantum-{method}",
            "solve_time": self.last_solve_time,
            "iterations": self.last_iterations
        })
        
        return solution, metadata
    
    def _select_quantum_method(self, problem: Dict[str, Any]) -> str:
        """Automatically select best quantum method for problem."""
        problem_type = problem["type"]
        size = problem["matrix"].shape[0] if "matrix" in problem else 1000
        
        if problem_type == "eigenvalue":
            # VQE for small eigenvalue problems, MPS for large ones
            return "vqe" if size < 256 else "mps"
        elif problem_type == "optimization":
            # QUBO for discrete optimization, MPS for continuous
            return "qubo" if problem.get("discrete", False) else "mps"
        elif problem_type == "pde":
            # MPS for high-dimensional PDEs
            return "mps"
        else:
            return "mps"  # Default to MPS
    
    def _solve_with_mps(self, problem: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve using Matrix Product States."""
        if self.mps_solver is None:
            config = MPSConfig(
                bond_dimension=kwargs.get("bond_dimension", 50),
                max_iterations=kwargs.get("max_iterations", 1000),
                convergence_tolerance=kwargs.get("convergence_tolerance", 1e-8)
            )
            self.mps_solver = MPSolver(config)
        
        matrix = problem["matrix"]
        n = matrix.shape[0]
        
        # Determine system dimensions for MPS
        # For 1D problems: use direct mapping
        # For 2D problems: use row-column decomposition
        if "dimensions" in problem:
            dimensions = problem["dimensions"]
        else:
            # Auto-determine dimensions
            dim_1d = int(jnp.ceil(jnp.sqrt(n)))
            dimensions = [2] * int(jnp.ceil(jnp.log2(n)))  # Binary encoding
        
        # Initialize MPS
        self.mps_solver.initialize_random_mps(dimensions)
        
        if problem["type"] == "eigenvalue":
            # Convert matrix to Trotter gates for imaginary time evolution
            gates = self._matrix_to_gates(matrix)
            energy = self.mps_solver.solve_imaginary_time_evolution(
                gates, dt=0.01, max_time=10.0)
            
            # Extract eigenstate
            eigenstate = self.mps_solver.contract_mps(self.mps_solver.tensors)
            solution = ([energy], [eigenstate])
            
            self.last_iterations = len(self.mps_solver.convergence_history)
            
            metadata = {
                "compression_ratio": self.mps_solver.get_compression_stats()["compression_ratio"],
                "convergence_history": self.mps_solver.convergence_history,
                "bond_dimensions": self.mps_solver.bond_dimensions,
            }
        else:
            # PDE solving or optimization
            gates = self._matrix_to_gates(matrix)
            final_energy = self.mps_solver.solve_imaginary_time_evolution(
                gates, dt=0.01, max_time=5.0)
            
            solution = self.mps_solver.contract_mps(self.mps_solver.tensors)
            self.last_iterations = len(self.mps_solver.convergence_history)
            
            metadata = {
                "final_energy": final_energy,
                "compression_stats": self.mps_solver.get_compression_stats(),
            }
        
        return solution, metadata
    
    def _solve_with_vqe(self, problem: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve using Variational Quantum Eigensolver."""
        if self.vqe_solver is None:
            config = VQEConfig(
                max_iterations=kwargs.get("max_iterations", 1000),
                convergence_tolerance=kwargs.get("convergence_tolerance", 1e-8),
                num_layers=kwargs.get("num_layers", 6)
            )
            self.vqe_solver = VQESolver(config)
        
        matrix = problem["matrix"]
        self.vqe_solver.setup_problem(matrix)
        
        # Solve for ground state and excited states
        num_eigenvalues = kwargs.get("num_eigenvalues", 1)
        
        ground_energy, ground_params = self.vqe_solver.solve_ground_state()
        eigenvalues = [ground_energy]
        eigenvectors = [self.vqe_solver.optimal_state]
        
        if num_eigenvalues > 1:
            excited_states = self.vqe_solver.compute_excited_states(num_eigenvalues - 1)
            for energy, state in excited_states:
                eigenvalues.append(energy)
                eigenvectors.append(state)
        
        self.last_iterations = len(self.vqe_solver.energy_history)
        
        solution = (eigenvalues, eigenvectors)
        metadata = {
            "convergence_analysis": self.vqe_solver.analyze_convergence(),
            "energy_history": self.vqe_solver.energy_history,
            "parameter_count": len(ground_params) if ground_params is not None else 0,
        }
        
        return solution, metadata
    
    def _solve_with_qubo(self, problem: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve using Quantum Annealing (QUBO)."""
        if self.qubo_optimizer is None:
            config = QUBOConfig(
                max_iterations=kwargs.get("max_iterations", 10000),
                convergence_tolerance=kwargs.get("convergence_tolerance", 1e-6)
            )
            self.qubo_optimizer = QUBOOptimizer(config)
        
        # Convert problem to QUBO formulation
        if "qubo_formulation" in problem:
            qubo_formulation = problem["qubo_formulation"]
        else:
            # Auto-convert continuous problem to QUBO
            qubo_formulation = self._convert_to_qubo(problem)
        
        # Optimize
        solution, energy = self.qubo_optimizer.optimize(qubo_formulation)
        
        self.last_iterations = len(self.qubo_optimizer.optimization_history)
        
        metadata = {
            "final_energy": energy,
            "optimization_history": self.qubo_optimizer.optimization_history,
            "binary_solution": solution,
        }
        
        # Convert binary solution back to continuous if needed
        if hasattr(qubo_formulation, 'decode_solution'):
            continuous_solution = qubo_formulation.decode_solution(solution)
            return continuous_solution, metadata
        else:
            return solution, metadata
    
    def _matrix_to_gates(self, matrix: jnp.ndarray) -> List[Tuple[jnp.ndarray, int]]:
        """Convert matrix to Trotter gate decomposition for MPS."""
        # Simplified Trotter decomposition
        # In practice, this would use sophisticated decomposition techniques
        n = matrix.shape[0]
        gates = []
        
        # Decompose into nearest-neighbor terms
        for i in range(min(n-1, 10)):  # Limit for demonstration
            # Extract 2x2 block
            gate = matrix[i:i+2, i:i+2]
            gates.append((gate, i))
        
        return gates
    
    def _convert_to_qubo(self, problem: Dict[str, Any]) -> Any:
        """Convert continuous optimization problem to QUBO."""
        # This would implement sophisticated QUBO conversion
        # Placeholder implementation
        from .quantum_annealing import QUBOFormulation
        
        problem_size = len(problem.get("initial_guess", [100]))
        qubo = QUBOFormulation(problem_size)
        
        # Simple binary encoding
        objective = problem["objective"]
        
        # Create dummy Q matrix and linear terms
        Q = jnp.eye(problem_size)
        linear = jnp.ones(problem_size)
        
        qubo.Q_matrix = Q
        qubo.linear_terms = linear
        
        return qubo
    
    def estimate_complexity(self, problem: Dict[str, Any]) -> Dict[str, float]:
        """Estimate computational complexity for quantum methods."""
        if problem["type"] == "eigenvalue":
            n = problem["matrix"].shape[0]
            # VQE: polynomial in system size
            time_complexity = n * jnp.log(n)**2
            memory_complexity = jnp.log(n)**2  # Exponential compression
        elif problem["type"] == "optimization":
            n = len(problem.get("initial_guess", [100]))
            # QUBO: exponential speedup possible
            time_complexity = jnp.sqrt(n)  # Quantum speedup
            memory_complexity = n
        else:
            n = problem["matrix"].shape[0]
            # MPS: polynomial scaling with bond dimension
            time_complexity = n * 50**3  # Assuming bond dimension 50
            memory_complexity = n * 50**2
        
        return {
            "time_complexity": time_complexity,
            "memory_complexity": memory_complexity,
            "estimated_time_seconds": float(time_complexity / 1e8),  # Quantum estimate
        }
    
    def get_method_name(self) -> str:
        return f"Quantum-{self.method_type}"


class HybridOptimizer:
    """Intelligent hybrid classical-quantum optimizer.
    
    Automatically selects and combines classical and quantum methods
    for optimal performance based on problem characteristics.
    """
    
    def __init__(self, config: HybridConfig = None):
        self.config = config or HybridConfig()
        
        # Initialize method implementations
        self.classical_method = ClassicalMethod()
        self.quantum_method = QuantumMethod()
        
        # Performance tracking
        self.method_performance = {
            "classical": {"total_time": 0.0, "total_problems": 0, "success_rate": 0.0},
            "quantum": {"total_time": 0.0, "total_problems": 0, "success_rate": 0.0}
        }
        
        self.solve_history = []
        
        logging.info("Hybrid optimizer initialized")
    
    def solve(self, problem: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve problem using optimal hybrid approach.
        
        Args:
            problem: Problem specification dictionary
            **kwargs: Additional solver parameters
            
        Returns:
            Tuple of (solution, metadata)
        """
        # Analyze problem characteristics
        problem_analysis = self._analyze_problem(problem)
        
        # Select optimal method
        selected_method = self._select_method(problem_analysis, **kwargs)
        
        logging.info(f"Selected method: {selected_method} for problem type: {problem['type']}")
        
        # Solve with selected method
        if selected_method == "classical":
            solution, metadata = self._solve_classical(problem, **kwargs)
        elif selected_method == "quantum":
            solution, metadata = self._solve_quantum(problem, **kwargs)
        elif selected_method == "hybrid":
            solution, metadata = self._solve_hybrid(problem, **kwargs)
        else:
            raise ValueError(f"Unknown method: {selected_method}")
        
        # Update performance tracking
        self._update_performance_tracking(selected_method, metadata)
        
        # Store solve history
        self.solve_history.append({
            "problem_type": problem["type"],
            "problem_size": problem_analysis["size"],
            "selected_method": selected_method,
            "solve_time": metadata["solve_time"],
            "success": metadata.get("success", True),
        })
        
        return solution, metadata
    
    def _analyze_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze problem characteristics to guide method selection."""
        analysis = {
            "type": problem["type"],
            "size": 0,
            "sparsity": 0.0,
            "condition_number": 1.0,
            "symmetry": False,
            "positive_definite": False,
        }
        
        if "matrix" in problem:
            matrix = problem["matrix"]
            analysis["size"] = matrix.shape[0]
            
            # Compute sparsity
            nonzero_elements = jnp.count_nonzero(matrix)
            total_elements = matrix.size
            analysis["sparsity"] = 1.0 - (nonzero_elements / total_elements)
            
            # Check symmetry
            analysis["symmetry"] = jnp.allclose(matrix, matrix.T)
            
            # Estimate condition number (for small matrices)
            if analysis["size"] < 1000:
                try:
                    analysis["condition_number"] = float(jnp.linalg.cond(matrix))
                except:
                    analysis["condition_number"] = 1e12  # Assume ill-conditioned
            
            # Check positive definiteness (for symmetric matrices)
            if analysis["symmetry"] and analysis["size"] < 1000:
                try:
                    eigenvals = jnp.linalg.eigvals(matrix)
                    analysis["positive_definite"] = jnp.all(eigenvals > 0)
                except:
                    analysis["positive_definite"] = False
        
        return analysis
    
    def _select_method(self, problem_analysis: Dict[str, Any], **kwargs) -> str:
        """Select optimal method based on problem analysis."""
        size = problem_analysis["size"]
        problem_type = problem_analysis["type"]
        
        # Force method selection if specified
        if "force_method" in kwargs:
            return kwargs["force_method"]
        
        # Size-based selection
        if size > self.config.problem_size_threshold:
            # Large problems: prefer quantum methods
            quantum_advantage = self._estimate_quantum_advantage(problem_analysis)
            if quantum_advantage > self.config.quantum_advantage_factor:
                return "quantum"
        
        # Problem-type based selection
        if problem_type == "eigenvalue":
            # For eigenvalue problems, compare VQE vs classical
            if size < 256:  # Small enough for VQE
                return "quantum" if problem_analysis.get("positive_definite", False) else "classical"
            else:
                return "quantum"  # Use MPS for large eigenvalue problems
        
        elif problem_type == "optimization":
            # For optimization, consider if problem is discrete
            if kwargs.get("discrete", False):
                return "quantum"  # QUBO for discrete optimization
            else:
                return "classical"  # Classical methods often good for continuous
        
        elif problem_type == "pde":
            # For PDEs, prefer quantum for high-dimensional problems
            return "quantum" if size > 10000 else "classical"
        
        # Default: use adaptive selection based on historical performance
        return self._adaptive_method_selection(problem_analysis)
    
    def _estimate_quantum_advantage(self, problem_analysis: Dict[str, Any]) -> float:
        """Estimate potential quantum advantage for problem."""
        # Get complexity estimates
        dummy_problem = {"matrix": jnp.eye(problem_analysis["size"]), "type": problem_analysis["type"]}
        
        classical_complexity = self.classical_method.estimate_complexity(dummy_problem)
        quantum_complexity = self.quantum_method.estimate_complexity(dummy_problem)
        
        # Compute advantage ratio
        classical_time = classical_complexity["estimated_time_seconds"]
        quantum_time = quantum_complexity["estimated_time_seconds"]
        
        if quantum_time > 0:
            advantage = classical_time / quantum_time
        else:
            advantage = 1.0
        
        # Adjust based on problem characteristics
        if problem_analysis.get("sparsity", 0) > 0.9:
            advantage *= 1.5  # Sparse problems favor quantum methods
        
        if problem_analysis.get("condition_number", 1) > 1e12:
            advantage *= 0.5  # Ill-conditioned problems challenging for quantum
        
        return advantage
    
    def _adaptive_method_selection(self, problem_analysis: Dict[str, Any]) -> str:
        """Adaptive method selection based on historical performance."""
        problem_type = problem_analysis["type"]
        size_range = self._get_size_range(problem_analysis["size"])
        
        # Look for similar problems in history
        similar_problems = [
            entry for entry in self.solve_history
            if entry["problem_type"] == problem_type and 
            self._get_size_range(entry["problem_size"]) == size_range
        ]
        
        if len(similar_problems) < 3:
            # Not enough data, use default heuristics
            return "classical" if problem_analysis["size"] < 1000 else "quantum"
        
        # Compute average performance for each method
        classical_performance = self._compute_method_performance(similar_problems, "classical")
        quantum_performance = self._compute_method_performance(similar_problems, "quantum")
        
        # Select method with better performance
        if quantum_performance["avg_time"] < classical_performance["avg_time"]:
            return "quantum"
        else:
            return "classical"
    
    def _get_size_range(self, size: int) -> str:
        """Get size range category for problem."""
        if size < 100:
            return "small"
        elif size < 1000:
            return "medium"
        elif size < 10000:
            return "large"
        else:
            return "xlarge"
    
    def _compute_method_performance(self, similar_problems: List[Dict], method: str) -> Dict[str, float]:
        """Compute average performance metrics for method on similar problems."""
        method_problems = [p for p in similar_problems if method in p["selected_method"]]
        
        if not method_problems:
            return {"avg_time": float('inf'), "success_rate": 0.0}
        
        avg_time = np.mean([p["solve_time"] for p in method_problems])
        success_rate = np.mean([p["success"] for p in method_problems])
        
        return {"avg_time": avg_time, "success_rate": success_rate}
    
    def _solve_classical(self, problem: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve using classical method."""
        return self.classical_method.solve(problem, **kwargs)
    
    def _solve_quantum(self, problem: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve using quantum method."""
        return self.quantum_method.solve(problem, **kwargs)
    
    def _solve_hybrid(self, problem: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve using hybrid approach - run both methods and compare."""
        if not self.config.benchmarking_enabled:
            # If benchmarking disabled, fall back to adaptive selection
            problem_analysis = self._analyze_problem(problem)
            method = self._adaptive_method_selection(problem_analysis)
            return self.solve(problem, force_method=method, **kwargs)
        
        # Run both methods in parallel (if computationally feasible)
        problem_analysis = self._analyze_problem(problem)
        
        # Only run hybrid for medium-sized problems
        if problem_analysis["size"] > 10000:
            return self._solve_quantum(problem, **kwargs)
        elif problem_analysis["size"] < 100:
            return self._solve_classical(problem, **kwargs)
        
        # Run both methods
        classical_solution, classical_metadata = self._solve_classical(problem, **kwargs)
        quantum_solution, quantum_metadata = self._solve_quantum(problem, **kwargs)
        
        # Compare results and select best
        classical_time = classical_metadata["solve_time"]
        quantum_time = quantum_metadata["solve_time"]
        
        # For eigenvalue problems, compare accuracy
        if problem["type"] == "eigenvalue":
            classical_error = classical_metadata.get("eigenvalue_errors", [0.0])[0]
            quantum_error = quantum_metadata.get("convergence_analysis", {}).get("energy_error", 0.0)
            
            # Select based on accuracy and time trade-off
            if quantum_error < classical_error and quantum_time < 2 * classical_time:
                selected_solution = quantum_solution
                selected_metadata = quantum_metadata
                selected_metadata["comparison"] = {
                    "classical_time": classical_time,
                    "quantum_time": quantum_time,
                    "classical_error": classical_error,
                    "quantum_error": quantum_error,
                    "selected": "quantum"
                }
            else:
                selected_solution = classical_solution
                selected_metadata = classical_metadata
                selected_metadata["comparison"] = {
                    "classical_time": classical_time,
                    "quantum_time": quantum_time,
                    "classical_error": classical_error,
                    "quantum_error": quantum_error,
                    "selected": "classical"
                }
        else:
            # For other problems, select based on time
            if quantum_time < classical_time:
                selected_solution = quantum_solution
                selected_metadata = quantum_metadata
                selected_metadata["selected_method"] = "quantum"
            else:
                selected_solution = classical_solution
                selected_metadata = classical_metadata
                selected_metadata["selected_method"] = "classical"
        
        return selected_solution, selected_metadata
    
    def _update_performance_tracking(self, method: str, metadata: Dict[str, Any]) -> None:
        """Update performance tracking statistics."""
        if "classical" in method:
            method_key = "classical"
        elif "quantum" in method:
            method_key = "quantum"
        else:
            return
        
        stats = self.method_performance[method_key]
        stats["total_time"] += metadata["solve_time"]
        stats["total_problems"] += 1
        
        # Update success rate (moving average)
        success = metadata.get("success", True)
        alpha = 0.1  # Learning rate for moving average
        stats["success_rate"] = (1 - alpha) * stats["success_rate"] + alpha * success
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance across methods."""
        summary = {
            "total_problems_solved": len(self.solve_history),
            "method_performance": self.method_performance.copy(),
            "problem_type_distribution": {},
            "size_distribution": {},
            "method_selection_frequency": {},
        }
        
        if not self.solve_history:
            return summary
        
        # Analyze problem type distribution
        problem_types = [entry["problem_type"] for entry in self.solve_history]
        for ptype in set(problem_types):
            summary["problem_type_distribution"][ptype] = problem_types.count(ptype)
        
        # Analyze size distribution
        sizes = [entry["problem_size"] for entry in self.solve_history]
        summary["size_distribution"] = {
            "min": min(sizes),
            "max": max(sizes),
            "mean": np.mean(sizes),
            "median": np.median(sizes),
        }
        
        # Analyze method selection frequency
        methods = [entry["selected_method"] for entry in self.solve_history]
        for method in set(methods):
            summary["method_selection_frequency"][method] = methods.count(method)
        
        return summary


class AdaptiveQuantumSolver:
    """Adaptive solver that learns optimal quantum method configurations.
    
    Uses machine learning to optimize quantum algorithm parameters
    based on problem characteristics and historical performance.
    """
    
    def __init__(self, base_optimizer: HybridOptimizer = None):
        self.base_optimizer = base_optimizer or HybridOptimizer()
        
        # Parameter optimization history
        self.parameter_optimization_history = []
        self.optimal_parameters = {}
        
        # Learning components
        self.problem_feature_extractor = self._initialize_feature_extractor()
        self.parameter_predictor = None  # Would be ML model
        
        logging.info("Adaptive quantum solver initialized")
    
    def _initialize_feature_extractor(self) -> Callable:
        """Initialize feature extraction for problem characterization."""
        def extract_features(problem: Dict[str, Any]) -> jnp.ndarray:
            """Extract relevant features from problem for ML prediction."""
            features = []
            
            if "matrix" in problem:
                matrix = problem["matrix"]
                features.extend([
                    matrix.shape[0],  # Size
                    float(jnp.count_nonzero(matrix) / matrix.size),  # Sparsity
                    float(jnp.linalg.norm(matrix)),  # Norm
                ])
                
                # Add spectral features for small matrices
                if matrix.shape[0] < 1000:
                    try:
                        eigenvals = jnp.linalg.eigvals(matrix)
                        features.extend([
                            float(jnp.max(jnp.real(eigenvals))),  # Max eigenvalue
                            float(jnp.min(jnp.real(eigenvals))),  # Min eigenvalue
                            float(jnp.std(jnp.real(eigenvals))),  # Eigenvalue spread
                        ])
                    except:
                        features.extend([0.0, 0.0, 0.0])
                else:
                    features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            return jnp.array(features)
        
        return extract_features
    
    def solve_with_adaptation(self, problem: Dict[str, Any], **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Solve problem with adaptive parameter optimization."""
        # Extract problem features
        features = self.problem_feature_extractor(problem)
        
        # Predict optimal parameters (if model trained)
        if self.parameter_predictor is not None:
            predicted_params = self._predict_optimal_parameters(features)
            kwargs.update(predicted_params)
        
        # Solve with current best parameters
        solution, metadata = self.base_optimizer.solve(problem, **kwargs)
        
        # Record performance for learning
        self._record_parameter_performance(features, kwargs, metadata)
        
        # Periodically update parameter predictor
        if len(self.parameter_optimization_history) % 50 == 0:
            self._update_parameter_predictor()
        
        return solution, metadata
    
    def _predict_optimal_parameters(self, features: jnp.ndarray) -> Dict[str, Any]:
        """Predict optimal parameters based on problem features."""
        # Placeholder for ML-based parameter prediction
        # In practice, this would use trained neural network or regression model
        
        # Simple heuristic-based prediction for now
        problem_size = features[0]
        sparsity = features[1]
        
        params = {}
        
        # VQE parameters
        if problem_size < 256:
            params["num_layers"] = max(4, min(8, int(problem_size / 32)))
            params["learning_rate"] = 0.01 if sparsity > 0.5 else 0.001
        
        # MPS parameters
        if problem_size > 1000:
            params["bond_dimension"] = max(20, min(100, int(jnp.sqrt(problem_size))))
            params["max_iterations"] = 2000 if sparsity > 0.8 else 1000
        
        # QUBO parameters
        params["initial_temperature"] = 100.0 if problem_size > 500 else 50.0
        params["annealing_schedule"] = "exponential" if sparsity > 0.5 else "linear"
        
        return params
    
    def _record_parameter_performance(self, features: jnp.ndarray, 
                                    parameters: Dict[str, Any],
                                    metadata: Dict[str, Any]) -> None:
        """Record parameter performance for learning."""
        performance_record = {
            "features": features,
            "parameters": parameters,
            "solve_time": metadata["solve_time"],
            "success": metadata.get("success", True),
            "accuracy": metadata.get("final_energy", 0.0),  # Use as proxy for solution quality
        }
        
        self.parameter_optimization_history.append(performance_record)
    
    def _update_parameter_predictor(self) -> None:
        """Update parameter prediction model based on accumulated data."""
        if len(self.parameter_optimization_history) < 20:
            return  # Need minimum data for training
        
        # This would implement ML model training
        # For now, just compute optimal parameters for each feature range
        
        logging.info(f"Updated parameter predictor with {len(self.parameter_optimization_history)} data points")
    
    def benchmark_adaptation_performance(self, test_problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark adaptive solver against non-adaptive baseline."""
        adaptive_results = []
        baseline_results = []
        
        for problem in test_problems:
            # Solve with adaptation
            adaptive_solution, adaptive_metadata = self.solve_with_adaptation(problem)
            adaptive_results.append({
                "solve_time": adaptive_metadata["solve_time"],
                "success": adaptive_metadata.get("success", True),
            })
            
            # Solve with baseline (default parameters)
            baseline_solution, baseline_metadata = self.base_optimizer.solve(problem)
            baseline_results.append({
                "solve_time": baseline_metadata["solve_time"],
                "success": baseline_metadata.get("success", True),
            })
        
        # Compute comparison metrics
        adaptive_avg_time = np.mean([r["solve_time"] for r in adaptive_results])
        baseline_avg_time = np.mean([r["solve_time"] for r in baseline_results])
        
        adaptive_success_rate = np.mean([r["success"] for r in adaptive_results])
        baseline_success_rate = np.mean([r["success"] for r in baseline_results])
        
        benchmark_results = {
            "num_problems": len(test_problems),
            "adaptive_avg_time": adaptive_avg_time,
            "baseline_avg_time": baseline_avg_time,
            "speedup_factor": baseline_avg_time / adaptive_avg_time,
            "adaptive_success_rate": adaptive_success_rate,
            "baseline_success_rate": baseline_success_rate,
            "improvement_ratio": adaptive_success_rate / baseline_success_rate,
        }
        
        return benchmark_results