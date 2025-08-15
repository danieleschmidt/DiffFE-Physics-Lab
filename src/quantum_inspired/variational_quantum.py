"""Variational Quantum Eigensolver (VQE) for PDE Eigenvalue Problems.

Implementation of quantum-inspired variational algorithms for solving
eigenvalue problems arising from PDE discretizations.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import List, Tuple, Optional, Callable, Dict, Any
import logging
from dataclasses import dataclass
from functools import partial

from ..backends.base import Backend
from ..utils.validation import validate_eigenvalue_problem


@dataclass
class VQEConfig:
    """Configuration for Variational Quantum Eigensolver."""
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-8
    learning_rate: float = 0.01
    optimizer: str = "adam"  # adam, sgd, lbfgs
    num_layers: int = 6
    entangling_gates: str = "cnot"  # cnot, cz, iswap
    measurement_shots: int = 10000
    error_mitigation: bool = True
    classical_optimizer_steps: int = 100
    random_seed: int = 42


class QuantumCircuit:
    """Quantum circuit representation for VQE ansatz.
    
    Implements hardware-efficient ansatz with parameterized gates
    suitable for eigenvalue problems from PDE discretization.
    """
    
    def __init__(self, num_qubits: int, num_layers: int, 
                 gate_type: str = "cnot"):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.gate_type = gate_type
        
        # Parameter layout: [layer][qubit][gate_param]
        # Each layer has rotation + entangling gates
        params_per_layer = num_qubits * 3  # 3 rotation parameters per qubit
        self.num_parameters = params_per_layer * num_layers
        
        logging.info(f"Quantum circuit initialized: {num_qubits} qubits, "
                    f"{num_layers} layers, {self.num_parameters} parameters")
    
    def initialize_parameters(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Initialize circuit parameters randomly."""
        # Initialize with small random values for gradient-based optimization
        return jax.random.normal(key, (self.num_parameters,)) * 0.1
    
    @partial(jit, static_argnums=(0,))
    def apply_circuit(self, params: jnp.ndarray, 
                     initial_state: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply parameterized quantum circuit to initial state.
        
        Args:
            params: Circuit parameters
            initial_state: Initial quantum state (default |0⟩)
            
        Returns:
            Output quantum state vector
        """
        if initial_state is None:
            # Start with |0⟩ state
            state = jnp.zeros(2**self.num_qubits)
            state = state.at[0].set(1.0)
        else:
            state = initial_state
        
        # Reshape parameters for layer-wise application
        layer_params = params.reshape(self.num_layers, self.num_qubits, 3)
        
        for layer in range(self.num_layers):
            # Apply rotation gates to each qubit
            for qubit in range(self.num_qubits):
                rx_angle, ry_angle, rz_angle = layer_params[layer, qubit]
                state = self._apply_single_qubit_rotation(state, qubit, 
                                                        rx_angle, ry_angle, rz_angle)
            
            # Apply entangling gates
            state = self._apply_entangling_layer(state)
        
        return state
    
    @partial(jit, static_argnums=(0, 2))
    def _apply_single_qubit_rotation(self, state: jnp.ndarray, qubit: int,
                                   rx_angle: float, ry_angle: float, 
                                   rz_angle: float) -> jnp.ndarray:
        """Apply parameterized single-qubit rotations."""
        # Create rotation matrices
        cos_rx, sin_rx = jnp.cos(rx_angle/2), jnp.sin(rx_angle/2)
        cos_ry, sin_ry = jnp.cos(ry_angle/2), jnp.sin(ry_angle/2)
        cos_rz, sin_rz = jnp.cos(rz_angle/2), jnp.sin(rz_angle/2)
        
        # RX gate
        rx_matrix = jnp.array([[cos_rx, -1j*sin_rx],
                              [-1j*sin_rx, cos_rx]])
        
        # RY gate
        ry_matrix = jnp.array([[cos_ry, -sin_ry],
                              [sin_ry, cos_ry]])
        
        # RZ gate
        rz_matrix = jnp.array([[jnp.exp(-1j*rz_angle/2), 0],
                              [0, jnp.exp(1j*rz_angle/2)]])
        
        # Combined rotation
        rotation = rz_matrix @ ry_matrix @ rx_matrix
        
        # Apply to state vector
        return self._apply_single_qubit_gate(state, qubit, rotation)
    
    @partial(jit, static_argnums=(0, 2))
    def _apply_single_qubit_gate(self, state: jnp.ndarray, qubit: int,
                               gate_matrix: jnp.ndarray) -> jnp.ndarray:
        """Apply single-qubit gate to state vector."""
        n_qubits = self.num_qubits
        state_reshaped = state.reshape([2] * n_qubits)
        
        # Contract gate with state along specified qubit axis
        axes = ([1], [qubit])
        result = jnp.tensordot(gate_matrix, state_reshaped, axes=axes)
        
        # Move the result axis to the correct position
        result = jnp.moveaxis(result, 0, qubit)
        
        return result.flatten()
    
    @partial(jit, static_argnums=(0,))
    def _apply_entangling_layer(self, state: jnp.ndarray) -> jnp.ndarray:
        """Apply layer of entangling gates."""
        if self.gate_type == "cnot":
            # Linear connectivity CNOT ladder
            for i in range(self.num_qubits - 1):
                state = self._apply_cnot(state, i, i + 1)
        elif self.gate_type == "cz":
            # Controlled-Z gates
            for i in range(self.num_qubits - 1):
                state = self._apply_cz(state, i, i + 1)
        
        return state
    
    @partial(jit, static_argnums=(0, 2, 3))
    def _apply_cnot(self, state: jnp.ndarray, control: int, target: int) -> jnp.ndarray:
        """Apply CNOT gate between control and target qubits."""
        cnot_matrix = jnp.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0]], dtype=complex)
        
        return self._apply_two_qubit_gate(state, control, target, cnot_matrix)
    
    @partial(jit, static_argnums=(0, 2, 3))
    def _apply_cz(self, state: jnp.ndarray, control: int, target: int) -> jnp.ndarray:
        """Apply controlled-Z gate between control and target qubits."""
        cz_matrix = jnp.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, -1]], dtype=complex)
        
        return self._apply_two_qubit_gate(state, control, target, cz_matrix)
    
    @partial(jit, static_argnums=(0, 2, 3))
    def _apply_two_qubit_gate(self, state: jnp.ndarray, qubit1: int, qubit2: int,
                            gate_matrix: jnp.ndarray) -> jnp.ndarray:
        """Apply two-qubit gate to state vector."""
        n_qubits = self.num_qubits
        state_reshaped = state.reshape([2] * n_qubits)
        
        # Ensure qubits are ordered
        if qubit1 > qubit2:
            qubit1, qubit2 = qubit2, qubit1
            # Swap rows/columns of gate matrix accordingly
            gate_matrix = gate_matrix[[0, 2, 1, 3]][:, [0, 2, 1, 3]]
        
        # Reshape gate for tensor contraction
        gate_reshaped = gate_matrix.reshape(2, 2, 2, 2)
        
        # Contract gate with state
        axes = ([2, 3], [qubit1, qubit2])
        result = jnp.tensordot(gate_reshaped, state_reshaped, axes=axes)
        
        # Move result axes to correct positions
        result = jnp.moveaxis(result, [0, 1], [qubit1, qubit2])
        
        return result.flatten()
    
    def measure_expectation(self, state: jnp.ndarray, 
                          observable: jnp.ndarray) -> float:
        """Compute expectation value of observable in given state."""
        return jnp.real(jnp.conj(state) @ observable @ state)


class VQESolver:
    """Variational Quantum Eigensolver for PDE eigenvalue problems.
    
    Solves eigenvalue problems of the form Hψ = λψ using variational
    quantum circuits optimized with classical algorithms.
    """
    
    def __init__(self, config: VQEConfig = None):
        self.config = config or VQEConfig()
        self.circuit: Optional[QuantumCircuit] = None
        self.hamiltonian: Optional[jnp.ndarray] = None
        
        # Optimization history
        self.energy_history = []
        self.parameter_history = []
        self.gradient_norms = []
        
        # Current best solution
        self.optimal_parameters: Optional[jnp.ndarray] = None
        self.optimal_energy: float = float('inf')
        self.optimal_state: Optional[jnp.ndarray] = None
        
        logging.info("VQE solver initialized")
    
    def setup_problem(self, hamiltonian: jnp.ndarray, 
                     num_qubits: Optional[int] = None) -> None:
        """Setup eigenvalue problem from Hamiltonian matrix.
        
        Args:
            hamiltonian: Hermitian matrix representing the Hamiltonian
            num_qubits: Number of qubits (auto-determined if None)
        """
        # Validate Hamiltonian
        if not jnp.allclose(hamiltonian, jnp.conj(hamiltonian.T)):
            logging.warning("Hamiltonian is not Hermitian - symmetrizing")
            hamiltonian = (hamiltonian + jnp.conj(hamiltonian.T)) / 2
        
        self.hamiltonian = hamiltonian
        
        # Determine number of qubits
        if num_qubits is None:
            matrix_size = hamiltonian.shape[0]
            num_qubits = int(jnp.ceil(jnp.log2(matrix_size)))
        
        # Pad Hamiltonian to match qubit space if necessary
        target_size = 2**num_qubits
        if hamiltonian.shape[0] < target_size:
            padded_hamiltonian = jnp.zeros((target_size, target_size), dtype=hamiltonian.dtype)
            original_size = hamiltonian.shape[0]
            padded_hamiltonian = padded_hamiltonian.at[:original_size, :original_size].set(hamiltonian)
            self.hamiltonian = padded_hamiltonian
        
        # Initialize quantum circuit
        self.circuit = QuantumCircuit(
            num_qubits=num_qubits,
            num_layers=self.config.num_layers,
            gate_type=self.config.entangling_gates
        )
        
        logging.info(f"VQE problem setup: {num_qubits} qubits, "
                    f"Hamiltonian size {self.hamiltonian.shape}")
    
    @partial(jit, static_argnums=(0,))
    def energy_expectation(self, params: jnp.ndarray) -> float:
        """Compute energy expectation value for given parameters."""
        state = self.circuit.apply_circuit(params)
        energy = self.circuit.measure_expectation(state, self.hamiltonian)
        return jnp.real(energy)
    
    def solve_ground_state(self, initial_params: Optional[jnp.ndarray] = None) -> Tuple[float, jnp.ndarray]:
        """Solve for ground state using VQE optimization.
        
        Args:
            initial_params: Initial circuit parameters (random if None)
            
        Returns:
            Tuple of (ground_state_energy, optimal_parameters)
        """
        if self.circuit is None or self.hamiltonian is None:
            raise ValueError("Must call setup_problem() first")
        
        # Initialize parameters
        if initial_params is None:
            key = jax.random.PRNGKey(self.config.random_seed)
            initial_params = self.circuit.initialize_parameters(key)
        
        # Setup optimization
        if self.config.optimizer == "adam":
            optimizer = self._setup_adam_optimizer()
        elif self.config.optimizer == "sgd":
            optimizer = self._setup_sgd_optimizer()
        elif self.config.optimizer == "lbfgs":
            return self._solve_with_lbfgs(initial_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Optimization loop
        params = initial_params
        opt_state = optimizer.init(params)
        
        # JIT-compiled update function
        @jit
        def update_step(params, opt_state):
            energy = self.energy_expectation(params)
            grads = grad(self.energy_expectation)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, energy, grads
        
        # Import optax for optimizers
        try:
            import optax
        except ImportError:
            logging.error("optax required for VQE optimization")
            raise
        
        for iteration in range(self.config.max_iterations):
            params, opt_state, energy, grads = update_step(params, opt_state)
            
            # Track optimization progress
            self.energy_history.append(float(energy))
            self.parameter_history.append(params.copy())
            grad_norm = jnp.linalg.norm(grads)
            self.gradient_norms.append(float(grad_norm))
            
            # Update best solution
            if energy < self.optimal_energy:
                self.optimal_energy = float(energy)
                self.optimal_parameters = params.copy()
                self.optimal_state = self.circuit.apply_circuit(params)
            
            # Log progress
            if iteration % 100 == 0:
                logging.info(f"VQE iteration {iteration}: energy = {energy:.8f}, "
                           f"grad_norm = {grad_norm:.6f}")
            
            # Check convergence
            if grad_norm < self.config.convergence_tolerance:
                logging.info(f"VQE converged at iteration {iteration}")
                break
        
        return self.optimal_energy, self.optimal_parameters
    
    def _setup_adam_optimizer(self):
        """Setup Adam optimizer."""
        try:
            import optax
            return optax.adam(learning_rate=self.config.learning_rate)
        except ImportError:
            raise ImportError("optax required for Adam optimizer")
    
    def _setup_sgd_optimizer(self):
        """Setup SGD optimizer."""
        try:
            import optax
            return optax.sgd(learning_rate=self.config.learning_rate)
        except ImportError:
            raise ImportError("optax required for SGD optimizer")
    
    def _solve_with_lbfgs(self, initial_params: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """Solve using L-BFGS optimization."""
        from scipy.optimize import minimize
        
        # Convert JAX functions to numpy for scipy
        def energy_func(params):
            return float(self.energy_expectation(jnp.array(params)))
        
        def grad_func(params):
            return np.array(grad(self.energy_expectation)(jnp.array(params)))
        
        # Optimize with L-BFGS
        result = minimize(
            energy_func,
            np.array(initial_params),
            method='L-BFGS-B',
            jac=grad_func,
            options={'maxiter': self.config.max_iterations,
                    'ftol': self.config.convergence_tolerance}
        )
        
        optimal_params = jnp.array(result.x)
        optimal_energy = result.fun
        
        # Update solver state
        self.optimal_energy = optimal_energy
        self.optimal_parameters = optimal_params
        self.optimal_state = self.circuit.apply_circuit(optimal_params)
        
        logging.info(f"L-BFGS optimization completed: energy = {optimal_energy:.8f}")
        
        return optimal_energy, optimal_params
    
    def compute_excited_states(self, num_states: int = 3) -> List[Tuple[float, jnp.ndarray]]:
        """Compute excited states using subspace expansion.
        
        Args:
            num_states: Number of excited states to compute
            
        Returns:
            List of (energy, state) tuples for excited states
        """
        if self.optimal_state is None:
            raise ValueError("Must solve ground state first")
        
        excited_states = []
        orthogonal_subspace = [self.optimal_state]
        
        for state_idx in range(num_states):
            # Create projector onto orthogonal subspace
            def orthogonal_energy(params):
                state = self.circuit.apply_circuit(params)
                
                # Project out previous states
                for prev_state in orthogonal_subspace:
                    overlap = jnp.conj(prev_state) @ state
                    state = state - overlap * prev_state
                
                # Normalize
                norm = jnp.linalg.norm(state)
                if norm > 1e-10:
                    state = state / norm
                
                # Compute energy
                return self.circuit.measure_expectation(state, self.hamiltonian)
            
            # Optimize for excited state
            key = jax.random.PRNGKey(self.config.random_seed + state_idx + 1)
            initial_params = self.circuit.initialize_parameters(key)
            
            # Use L-BFGS for excited state optimization
            from scipy.optimize import minimize
            
            def energy_func(params):
                return float(orthogonal_energy(jnp.array(params)))
            
            def grad_func(params):
                return np.array(grad(orthogonal_energy)(jnp.array(params)))
            
            result = minimize(
                energy_func,
                np.array(initial_params),
                method='L-BFGS-B',
                jac=grad_func,
                options={'maxiter': self.config.max_iterations}
            )
            
            excited_params = jnp.array(result.x)
            excited_energy = result.fun
            excited_state = self.circuit.apply_circuit(excited_params)
            
            # Add to orthogonal subspace
            orthogonal_subspace.append(excited_state)
            excited_states.append((excited_energy, excited_state))
            
            logging.info(f"Excited state {state_idx + 1}: energy = {excited_energy:.8f}")
        
        return excited_states
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze optimization convergence and circuit performance."""
        if not self.energy_history:
            return {}
        
        energy_array = jnp.array(self.energy_history)
        grad_array = jnp.array(self.gradient_norms)
        
        # Compute exact solution for comparison
        exact_eigenvalues = jnp.linalg.eigvals(self.hamiltonian)
        exact_ground_energy = jnp.min(jnp.real(exact_eigenvalues))
        
        analysis = {
            'final_energy': self.optimal_energy,
            'exact_ground_energy': float(exact_ground_energy),
            'energy_error': abs(self.optimal_energy - exact_ground_energy),
            'convergence_iterations': len(self.energy_history),
            'final_gradient_norm': float(grad_array[-1]) if len(grad_array) > 0 else 0.0,
            'energy_variance': float(jnp.var(energy_array[-100:])),  # Last 100 iterations
            'optimization_efficiency': self._compute_optimization_efficiency(),
        }
        
        return analysis
    
    def _compute_optimization_efficiency(self) -> float:
        """Compute optimization efficiency metric."""
        if len(self.energy_history) < 10:
            return 0.0
        
        initial_energy = self.energy_history[0]
        final_energy = self.energy_history[-1]
        energy_improvement = initial_energy - final_energy
        
        # Efficiency = improvement per iteration
        efficiency = energy_improvement / len(self.energy_history)
        return float(efficiency)


class QuantumEigenvalueSolver:
    """High-level interface for quantum eigenvalue problems from PDEs.
    
    Integrates VQE with PDE discretization for practical eigenvalue problems.
    """
    
    def __init__(self, config: VQEConfig = None):
        self.config = config or VQEConfig()
        self.vqe_solver = VQESolver(config)
        
        # Problem-specific data
        self.differential_operator: Optional[jnp.ndarray] = None
        self.boundary_conditions: Optional[Dict] = None
        self.eigenvalues: List[float] = []
        self.eigenvectors: List[jnp.ndarray] = []
    
    def setup_pde_eigenvalue_problem(self, operator_matrix: jnp.ndarray,
                                   boundary_conditions: Dict = None) -> None:
        """Setup eigenvalue problem from PDE discretization.
        
        Args:
            operator_matrix: Discretized differential operator
            boundary_conditions: Boundary condition specifications
        """
        self.differential_operator = operator_matrix
        self.boundary_conditions = boundary_conditions or {}
        
        # Setup VQE problem
        self.vqe_solver.setup_problem(operator_matrix)
        
        logging.info(f"PDE eigenvalue problem setup: matrix size {operator_matrix.shape}")
    
    def solve_modes(self, num_modes: int = 5) -> Tuple[List[float], List[jnp.ndarray]]:
        """Solve for lowest eigenvalue modes.
        
        Args:
            num_modes: Number of eigenvalue modes to compute
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Solve ground state
        ground_energy, ground_params = self.vqe_solver.solve_ground_state()
        ground_state = self.vqe_solver.optimal_state
        
        self.eigenvalues = [ground_energy]
        self.eigenvectors = [ground_state]
        
        # Solve excited states
        if num_modes > 1:
            excited_states = self.vqe_solver.compute_excited_states(num_modes - 1)
            
            for energy, state in excited_states:
                self.eigenvalues.append(energy)
                self.eigenvectors.append(state)
        
        logging.info(f"Solved {len(self.eigenvalues)} eigenvalue modes")
        
        return self.eigenvalues, self.eigenvectors
    
    def benchmark_against_classical(self) -> Dict[str, Any]:
        """Benchmark VQE results against classical eigenvalue solver."""
        if self.differential_operator is None:
            raise ValueError("Must setup problem first")
        
        # Classical solution
        classical_eigenvals, classical_eigenvecs = jnp.linalg.eigh(self.differential_operator)
        
        # Sort by eigenvalue
        sort_indices = jnp.argsort(classical_eigenvals)
        classical_eigenvals = classical_eigenvals[sort_indices]
        classical_eigenvecs = classical_eigenvecs[:, sort_indices]
        
        # Compare with VQE results
        num_compare = min(len(self.eigenvalues), len(classical_eigenvals))
        
        eigenvalue_errors = []
        overlap_fidelities = []
        
        for i in range(num_compare):
            # Eigenvalue error
            vqe_eigenval = self.eigenvalues[i]
            classical_eigenval = classical_eigenvals[i]
            eigenvalue_error = abs(vqe_eigenval - classical_eigenval)
            eigenvalue_errors.append(eigenvalue_error)
            
            # State overlap fidelity
            vqe_state = self.eigenvectors[i]
            classical_state = classical_eigenvecs[:, i]
            
            # Trim classical state to match VQE state size if necessary
            if len(classical_state) > len(vqe_state):
                classical_state = classical_state[:len(vqe_state)]
            elif len(classical_state) < len(vqe_state):
                padded_classical = jnp.zeros(len(vqe_state), dtype=classical_state.dtype)
                padded_classical = padded_classical.at[:len(classical_state)].set(classical_state)
                classical_state = padded_classical
            
            overlap = abs(jnp.conj(vqe_state) @ classical_state)
            overlap_fidelities.append(overlap)
        
        benchmark_results = {
            'num_modes_compared': num_compare,
            'eigenvalue_errors': eigenvalue_errors,
            'overlap_fidelities': overlap_fidelities,
            'mean_eigenvalue_error': float(jnp.mean(jnp.array(eigenvalue_errors))),
            'mean_overlap_fidelity': float(jnp.mean(jnp.array(overlap_fidelities))),
            'classical_eigenvalues': classical_eigenvals[:num_compare],
            'vqe_eigenvalues': self.eigenvalues[:num_compare],
        }
        
        return benchmark_results


# Example applications and utility functions
def create_harmonic_oscillator_problem(n_qubits: int = 4) -> jnp.ndarray:
    """Create quantum harmonic oscillator Hamiltonian for testing."""
    n_levels = 2**n_qubits
    
    # Harmonic oscillator: H = ℏω(a†a + 1/2)
    # In position basis: H = -d²/dx² + x²
    
    # Simple finite difference discretization
    dx = 0.1
    x_max = 5.0
    x_points = jnp.linspace(-x_max, x_max, n_levels)
    
    # Kinetic energy term: -d²/dx²
    kinetic = jnp.zeros((n_levels, n_levels))
    for i in range(1, n_levels - 1):
        kinetic = kinetic.at[i, i-1].set(-1/(dx**2))
        kinetic = kinetic.at[i, i].set(2/(dx**2))
        kinetic = kinetic.at[i, i+1].set(-1/(dx**2))
    
    # Potential energy term: x²
    potential = jnp.diag(x_points**2)
    
    hamiltonian = kinetic + potential
    
    return hamiltonian


def create_laplacian_eigenvalue_problem(grid_size: int = 8) -> jnp.ndarray:
    """Create 2D Laplacian eigenvalue problem for testing."""
    # 2D Laplacian with Dirichlet boundary conditions
    # -∇²u = λu on square domain
    
    n_total = grid_size * grid_size
    hamiltonian = jnp.zeros((n_total, n_total))
    
    dx = 1.0 / (grid_size + 1)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            
            # Diagonal term
            hamiltonian = hamiltonian.at[idx, idx].set(4 / (dx**2))
            
            # Off-diagonal terms (nearest neighbors)
            if i > 0:  # Left neighbor
                left_idx = (i-1) * grid_size + j
                hamiltonian = hamiltonian.at[idx, left_idx].set(-1 / (dx**2))
            
            if i < grid_size - 1:  # Right neighbor
                right_idx = (i+1) * grid_size + j
                hamiltonian = hamiltonian.at[idx, right_idx].set(-1 / (dx**2))
            
            if j > 0:  # Bottom neighbor
                bottom_idx = i * grid_size + (j-1)
                hamiltonian = hamiltonian.at[idx, bottom_idx].set(-1 / (dx**2))
            
            if j < grid_size - 1:  # Top neighbor
                top_idx = i * grid_size + (j+1)
                hamiltonian = hamiltonian.at[idx, top_idx].set(-1 / (dx**2))
    
    return hamiltonian