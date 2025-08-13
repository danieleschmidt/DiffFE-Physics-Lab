"""Machine learning acceleration features for PINNs and operator optimization."""

import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import grad, jit, pmap, vmap
from jax.experimental import optimizers

logger = logging.getLogger(__name__)


@dataclass
class OperatorFusionPattern:
    """Pattern for fusing multiple operators."""

    name: str
    operators: List[str]
    fusion_func: Callable
    expected_speedup: float = 1.0
    memory_reduction: float = 0.0


@dataclass
class NASArchitecture:
    """Neural Architecture Search architecture configuration."""

    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]
    parameters_count: int = 0
    flops: int = 0
    validation_score: float = 0.0
    training_time: float = 0.0


class OperatorFusionEngine:
    """Operator fusion engine for performance optimization."""

    def __init__(self):
        self.fusion_patterns = {}
        self.fused_operators = {}
        self.performance_cache = {}
        self.fusion_stats = {
            "fusion_attempts": 0,
            "successful_fusions": 0,
            "performance_improvements": [],
        }

        # Register common fusion patterns
        self._register_default_patterns()

        logger.info("Operator fusion engine initialized")

    def _register_default_patterns(self):
        """Register common operator fusion patterns."""

        # Laplacian + reaction fusion
        def fused_laplacian_reaction(u, diffusion_coeff, reaction_coeff):
            laplacian = self._compute_laplacian_jax(u)
            reaction = reaction_coeff * u
            return diffusion_coeff * laplacian + reaction

        self.register_fusion_pattern(
            OperatorFusionPattern(
                name="laplacian_reaction",
                operators=["laplacian", "reaction"],
                fusion_func=jit(fused_laplacian_reaction),
                expected_speedup=1.8,
                memory_reduction=0.3,
            )
        )

        # Gradient + divergence fusion
        def fused_gradient_divergence(u):
            grad_u = jnp.gradient(u)
            div_u = jnp.sum(
                jnp.array(
                    [jnp.gradient(grad_u[i], axis=i) for i in range(len(grad_u))]
                ),
                axis=0,
            )
            return grad_u, div_u

        self.register_fusion_pattern(
            OperatorFusionPattern(
                name="gradient_divergence",
                operators=["gradient", "divergence"],
                fusion_func=jit(fused_gradient_divergence),
                expected_speedup=1.5,
                memory_reduction=0.2,
            )
        )

        # Convection + diffusion fusion
        def fused_convection_diffusion(u, velocity, diffusion_coeff):
            grad_u = jnp.gradient(u)
            convection = jnp.sum(velocity * jnp.array(grad_u), axis=0)
            laplacian = self._compute_laplacian_jax(u)
            diffusion = diffusion_coeff * laplacian
            return convection + diffusion

        self.register_fusion_pattern(
            OperatorFusionPattern(
                name="convection_diffusion",
                operators=["convection", "diffusion"],
                fusion_func=jit(fused_convection_diffusion),
                expected_speedup=2.1,
                memory_reduction=0.4,
            )
        )

    def _compute_laplacian_jax(self, u):
        """Compute Laplacian using JAX."""
        if u.ndim == 1:
            return jnp.gradient(jnp.gradient(u))
        elif u.ndim == 2:
            d2u_dx2 = jnp.gradient(jnp.gradient(u, axis=0), axis=0)
            d2u_dy2 = jnp.gradient(jnp.gradient(u, axis=1), axis=1)
            return d2u_dx2 + d2u_dy2
        elif u.ndim == 3:
            d2u_dx2 = jnp.gradient(jnp.gradient(u, axis=0), axis=0)
            d2u_dy2 = jnp.gradient(jnp.gradient(u, axis=1), axis=1)
            d2u_dz2 = jnp.gradient(jnp.gradient(u, axis=2), axis=2)
            return d2u_dx2 + d2u_dy2 + d2u_dz2
        else:
            raise ValueError(f"Unsupported dimensionality: {u.ndim}")

    def register_fusion_pattern(self, pattern: OperatorFusionPattern):
        """Register a new fusion pattern."""
        self.fusion_patterns[pattern.name] = pattern
        logger.debug(f"Registered fusion pattern: {pattern.name}")

    def detect_fusion_opportunities(
        self, operator_sequence: List[str]
    ) -> List[OperatorFusionPattern]:
        """Detect fusion opportunities in operator sequence."""
        opportunities = []

        for pattern_name, pattern in self.fusion_patterns.items():
            # Check if pattern operators appear in sequence
            pattern_ops = set(pattern.operators)
            sequence_ops = set(operator_sequence)

            if pattern_ops.issubset(sequence_ops):
                # Check if operators are in reasonable proximity
                indices = [operator_sequence.index(op) for op in pattern.operators]
                if max(indices) - min(indices) <= len(pattern.operators) + 2:
                    opportunities.append(pattern)

        return opportunities

    def fuse_operators(
        self,
        pattern: OperatorFusionPattern,
        operator_functions: Dict[str, Callable],
        benchmark: bool = True,
    ) -> Callable:
        """Fuse operators according to pattern."""
        fusion_key = f"{pattern.name}_{hash(tuple(sorted(operator_functions.keys())))}"

        if fusion_key in self.fused_operators:
            return self.fused_operators[fusion_key]

        self.fusion_stats["fusion_attempts"] += 1

        try:
            # Create fused function
            fused_func = pattern.fusion_func

            # Benchmark if requested
            if benchmark:
                speedup = self._benchmark_fusion(
                    pattern, operator_functions, fused_func
                )
                if speedup > 1.0:
                    self.fusion_stats["successful_fusions"] += 1
                    self.fusion_stats["performance_improvements"].append(speedup)

                    logger.info(
                        f"Successful fusion '{pattern.name}': {speedup:.2f}x speedup"
                    )
                else:
                    logger.warning(
                        f"Fusion '{pattern.name}' provided no speedup: {speedup:.2f}x"
                    )

            # Cache fused operator
            self.fused_operators[fusion_key] = fused_func
            return fused_func

        except Exception as e:
            logger.error(f"Failed to fuse operators for pattern '{pattern.name}': {e}")
            raise

    def _benchmark_fusion(
        self,
        pattern: OperatorFusionPattern,
        operator_functions: Dict[str, Callable],
        fused_func: Callable,
    ) -> float:
        """Benchmark fused operator against individual operators."""
        # Create test data
        test_data = jnp.ones((100, 100))  # 2D test case

        # Benchmark individual operators
        start_time = time.time()
        for _ in range(10):
            # Simulate running individual operators
            temp_result = test_data
            for op_name in pattern.operators:
                if op_name in operator_functions:
                    temp_result = operator_functions[op_name](temp_result)
        individual_time = time.time() - start_time

        # Benchmark fused operator
        start_time = time.time()
        for _ in range(10):
            if pattern.name == "laplacian_reaction":
                fused_result = fused_func(test_data, 1.0, 0.1)
            elif pattern.name == "gradient_divergence":
                fused_result = fused_func(test_data)
            elif pattern.name == "convection_diffusion":
                velocity = jnp.ones((2,) + test_data.shape)
                fused_result = fused_func(test_data, velocity, 1.0)
            else:
                fused_result = fused_func(test_data)
        fused_time = time.time() - start_time

        speedup = individual_time / fused_time if fused_time > 0 else 0.0
        return speedup

    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get operator fusion statistics."""
        avg_speedup = (
            sum(self.fusion_stats["performance_improvements"])
            / len(self.fusion_stats["performance_improvements"])
            if self.fusion_stats["performance_improvements"]
            else 0.0
        )

        return {
            "registered_patterns": len(self.fusion_patterns),
            "fused_operators": len(self.fused_operators),
            "fusion_attempts": self.fusion_stats["fusion_attempts"],
            "successful_fusions": self.fusion_stats["successful_fusions"],
            "success_rate": (
                self.fusion_stats["successful_fusions"]
                / self.fusion_stats["fusion_attempts"]
                if self.fusion_stats["fusion_attempts"] > 0
                else 0.0
            ),
            "average_speedup": avg_speedup,
            "best_speedup": (
                max(self.fusion_stats["performance_improvements"])
                if self.fusion_stats["performance_improvements"]
                else 0.0
            ),
        }


class PINNArchitecture(nn.Module):
    """Physics-Informed Neural Network architecture."""

    features: List[int]
    activation: str = "tanh"
    use_batch_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Get activation function
        if self.activation == "tanh":
            activation_fn = nn.tanh
        elif self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "gelu":
            activation_fn = nn.gelu
        elif self.activation == "swish":
            activation_fn = nn.swish
        else:
            activation_fn = nn.tanh

        # Forward pass through layers
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat)(x)

            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training)(x)

            x = activation_fn(x)

            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Output layer (no activation)
        x = nn.Dense(self.features[-1])(x)

        return x


class NeuralArchitectureSearch:
    """Neural Architecture Search for PINNs optimization."""

    def __init__(
        self,
        search_space: Dict[str, List],
        population_size: int = 20,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Search state
        self.population = []
        self.best_architecture = None
        self.search_history = []
        self.evaluation_cache = {}

        # Default search space if not provided
        if not search_space:
            self.search_space = {
                "hidden_layers": [1, 2, 3, 4, 5, 6],
                "hidden_units": [16, 32, 64, 128, 256, 512],
                "activation": ["tanh", "relu", "gelu", "swish"],
                "batch_norm": [True, False],
                "dropout_rate": [0.0, 0.1, 0.2, 0.3],
            }

        logger.info(
            f"NAS initialized: pop_size={population_size}, generations={generations}"
        )

    def generate_random_architecture(self) -> NASArchitecture:
        """Generate a random architecture from search space."""
        import random

        # Sample architecture parameters
        num_layers = random.choice(self.search_space["hidden_layers"])
        hidden_units = [
            random.choice(self.search_space["hidden_units"]) for _ in range(num_layers)
        ]
        activation = random.choice(self.search_space["activation"])
        batch_norm = random.choice(self.search_space["batch_norm"])
        dropout_rate = random.choice(self.search_space["dropout_rate"])

        # Add input and output layers
        features = [2] + hidden_units + [1]  # 2D input, 1D output for typical PDE

        layers = []
        for i, feat in enumerate(features):
            layer_config = {
                "type": "dense",
                "units": feat,
                "activation": activation if i < len(features) - 1 else None,
                "batch_norm": batch_norm if i < len(features) - 1 else False,
                "dropout_rate": dropout_rate if i < len(features) - 1 else 0.0,
            }
            layers.append(layer_config)

        # Simple sequential connections
        connections = [(i, i + 1) for i in range(len(layers) - 1)]

        # Estimate parameters count
        params_count = 0
        for i in range(len(features) - 1):
            params_count += (
                features[i] * features[i + 1] + features[i + 1]
            )  # weights + biases

        return NASArchitecture(
            layers=layers, connections=connections, parameters_count=params_count
        )

    def initialize_population(self):
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            arch = self.generate_random_architecture()
            self.population.append(arch)

        logger.info(f"Initialized population of {len(self.population)} architectures")

    def evaluate_architecture(
        self,
        architecture: NASArchitecture,
        train_data: Tuple[jnp.ndarray, jnp.ndarray],
        validation_data: Tuple[jnp.ndarray, jnp.ndarray],
        physics_loss_func: Callable,
        max_epochs: int = 100,
    ) -> float:
        """Evaluate architecture performance."""
        arch_key = self._architecture_hash(architecture)

        # Check cache
        if arch_key in self.evaluation_cache:
            return self.evaluation_cache[arch_key]

        start_time = time.time()

        try:
            # Create PINN model
            features = [layer["units"] for layer in architecture.layers]
            activation = architecture.layers[0].get("activation", "tanh")
            batch_norm = architecture.layers[0].get("batch_norm", False)
            dropout_rate = architecture.layers[0].get("dropout_rate", 0.0)

            model = PINNArchitecture(
                features=features,
                activation=activation,
                use_batch_norm=batch_norm,
                dropout_rate=dropout_rate,
            )

            # Initialize parameters
            rng = jax.random.PRNGKey(42)
            x_sample = jnp.ones((1, features[0]))
            params = model.init(rng, x_sample)

            # Train model (simplified)
            optimizer = optax.adam(1e-3)
            opt_state = optimizer.init(params)

            X_train, y_train = train_data
            X_val, y_val = validation_data

            def loss_fn(params, x, y):
                pred = model.apply(params, x, training=True)
                data_loss = jnp.mean((pred - y) ** 2)
                physics_loss = physics_loss_func(params, model, x)
                return data_loss + 0.1 * physics_loss

            # Training loop (abbreviated for performance)
            train_epochs = min(max_epochs, 20)  # Limit epochs for NAS

            for epoch in range(train_epochs):
                loss, grads = jax.value_and_grad(loss_fn)(params, X_train, y_train)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

            # Validation score
            val_pred = model.apply(params, X_val, training=False)
            val_loss = jnp.mean((val_pred - y_val) ** 2)

            # Score combines accuracy and efficiency
            training_time = time.time() - start_time
            architecture.training_time = training_time

            # Lower is better for score
            score = (
                float(val_loss)
                + 0.001 * architecture.parameters_count / 10000
                + 0.01 * training_time
            )
            architecture.validation_score = score

            # Cache result
            self.evaluation_cache[arch_key] = score

            return score

        except Exception as e:
            logger.error(f"Architecture evaluation failed: {e}")
            # Return poor score for failed architectures
            return float("inf")

    def _architecture_hash(self, architecture: NASArchitecture) -> str:
        """Generate hash for architecture caching."""
        arch_str = str(architecture.layers) + str(architecture.connections)
        return str(hash(arch_str))

    def mutate_architecture(self, architecture: NASArchitecture) -> NASArchitecture:
        """Mutate architecture."""
        import copy
        import random

        mutated = copy.deepcopy(architecture)

        # Randomly mutate one aspect
        mutation_type = random.choice(
            [
                "add_layer",
                "remove_layer",
                "change_units",
                "change_activation",
                "change_dropout",
            ]
        )

        if mutation_type == "add_layer" and len(mutated.layers) < 10:
            # Add a layer
            insert_idx = random.randint(1, len(mutated.layers) - 1)
            units = random.choice(self.search_space["hidden_units"])
            new_layer = {
                "type": "dense",
                "units": units,
                "activation": mutated.layers[0]["activation"],
                "batch_norm": mutated.layers[0]["batch_norm"],
                "dropout_rate": mutated.layers[0]["dropout_rate"],
            }
            mutated.layers.insert(insert_idx, new_layer)

        elif mutation_type == "remove_layer" and len(mutated.layers) > 3:
            # Remove a layer
            remove_idx = random.randint(1, len(mutated.layers) - 2)
            mutated.layers.pop(remove_idx)

        elif mutation_type == "change_units":
            # Change number of units in a layer
            layer_idx = random.randint(1, len(mutated.layers) - 2)
            mutated.layers[layer_idx]["units"] = random.choice(
                self.search_space["hidden_units"]
            )

        elif mutation_type == "change_activation":
            # Change activation function
            new_activation = random.choice(self.search_space["activation"])
            for layer in mutated.layers[:-1]:  # Exclude output layer
                layer["activation"] = new_activation

        elif mutation_type == "change_dropout":
            # Change dropout rate
            new_dropout = random.choice(self.search_space["dropout_rate"])
            for layer in mutated.layers[:-1]:  # Exclude output layer
                layer["dropout_rate"] = new_dropout

        # Update connections (simple sequential)
        mutated.connections = [(i, i + 1) for i in range(len(mutated.layers) - 1)]

        # Recalculate parameters count
        features = [layer["units"] for layer in mutated.layers]
        params_count = 0
        for i in range(len(features) - 1):
            params_count += features[i] * features[i + 1] + features[i + 1]
        mutated.parameters_count = params_count

        return mutated

    def crossover_architectures(
        self, parent1: NASArchitecture, parent2: NASArchitecture
    ) -> Tuple[NASArchitecture, NASArchitecture]:
        """Crossover two architectures."""
        import copy
        import random

        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Exchange some layers
        if len(parent1.layers) > 2 and len(parent2.layers) > 2:
            # Find common layer range
            min_layers = min(len(parent1.layers), len(parent2.layers))

            # Exchange middle layers
            start_idx = 1
            end_idx = min(min_layers - 1, start_idx + 2)

            if end_idx > start_idx:
                # Swap layer sections
                child1.layers[start_idx:end_idx] = parent2.layers[start_idx:end_idx]
                child2.layers[start_idx:end_idx] = parent1.layers[start_idx:end_idx]

        # Update connections
        child1.connections = [(i, i + 1) for i in range(len(child1.layers) - 1)]
        child2.connections = [(i, i + 1) for i in range(len(child2.layers) - 1)]

        # Recalculate parameters
        for child in [child1, child2]:
            features = [layer["units"] for layer in child.layers]
            params_count = 0
            for i in range(len(features) - 1):
                params_count += features[i] * features[i + 1] + features[i + 1]
            child.parameters_count = params_count

        return child1, child2

    def search_optimal_architecture(
        self,
        train_data: Tuple[jnp.ndarray, jnp.ndarray],
        validation_data: Tuple[jnp.ndarray, jnp.ndarray],
        physics_loss_func: Callable,
    ) -> NASArchitecture:
        """Run neural architecture search."""
        logger.info("Starting neural architecture search...")

        # Initialize population
        self.initialize_population()

        # Evolution loop
        for generation in range(self.generations):
            logger.info(f"Generation {generation + 1}/{self.generations}")

            # Evaluate population
            scores = []
            for i, arch in enumerate(self.population):
                score = self.evaluate_architecture(
                    arch, train_data, validation_data, physics_loss_func
                )
                scores.append((score, i))
                arch.validation_score = score

            # Sort by score (lower is better)
            scores.sort(key=lambda x: x[0])

            # Update best architecture
            best_score, best_idx = scores[0]
            if (
                self.best_architecture is None
                or best_score < self.best_architecture.validation_score
            ):
                self.best_architecture = self.population[best_idx]
                logger.info(f"New best architecture found: score={best_score:.6f}")

            # Record generation stats
            avg_score = sum(score for score, _ in scores) / len(scores)
            self.search_history.append(
                {
                    "generation": generation,
                    "best_score": best_score,
                    "average_score": avg_score,
                    "best_params_count": self.population[best_idx].parameters_count,
                }
            )

            # Selection and reproduction
            if generation < self.generations - 1:
                new_population = []

                # Elite selection (keep top 20%)
                elite_count = max(1, self.population_size // 5)
                for _, idx in scores[:elite_count]:
                    new_population.append(self.population[idx])

                # Generate offspring
                while len(new_population) < self.population_size:
                    # Tournament selection
                    parent1_idx = self._tournament_selection(scores)
                    parent2_idx = self._tournament_selection(scores)

                    parent1 = self.population[parent1_idx]
                    parent2 = self.population[parent2_idx]

                    # Crossover
                    if np.random.random() < self.crossover_rate:
                        child1, child2 = self.crossover_architectures(parent1, parent2)
                    else:
                        child1, child2 = parent1, parent2

                    # Mutation
                    if np.random.random() < self.mutation_rate:
                        child1 = self.mutate_architecture(child1)
                    if np.random.random() < self.mutation_rate:
                        child2 = self.mutate_architecture(child2)

                    new_population.extend([child1, child2])

                # Keep population size constant
                self.population = new_population[: self.population_size]

        logger.info(
            f"NAS completed. Best architecture score: {self.best_architecture.validation_score:.6f}"
        )
        return self.best_architecture

    def _tournament_selection(
        self, scores: List[Tuple[float, int]], tournament_size: int = 3
    ) -> int:
        """Tournament selection for parent selection."""
        import random

        tournament = random.sample(scores, min(tournament_size, len(scores)))
        winner_score, winner_idx = min(tournament, key=lambda x: x[0])
        return winner_idx

    def get_search_stats(self) -> Dict[str, Any]:
        """Get neural architecture search statistics."""
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "evaluations_cached": len(self.evaluation_cache),
            "best_score": (
                self.best_architecture.validation_score
                if self.best_architecture
                else None
            ),
            "best_params_count": (
                self.best_architecture.parameters_count
                if self.best_architecture
                else None
            ),
            "search_history": self.search_history[-10:] if self.search_history else [],
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
        }


# Global instances
_global_fusion_engine = None
_global_nas_engine = None


def get_fusion_engine() -> OperatorFusionEngine:
    """Get global operator fusion engine."""
    global _global_fusion_engine
    if _global_fusion_engine is None:
        _global_fusion_engine = OperatorFusionEngine()
    return _global_fusion_engine


def get_nas_engine(**kwargs) -> NeuralArchitectureSearch:
    """Get global neural architecture search engine."""
    global _global_nas_engine
    if _global_nas_engine is None:
        _global_nas_engine = NeuralArchitectureSearch(**kwargs)
    return _global_nas_engine


def operator_fusion(operator_sequence: List[str], enable_benchmarking: bool = True):
    """Decorator for operator fusion optimization."""

    def decorator(compute_func):
        @wraps(compute_func)
        def wrapper(*args, **kwargs):
            fusion_engine = get_fusion_engine()

            # Detect fusion opportunities
            opportunities = fusion_engine.detect_fusion_opportunities(operator_sequence)

            if opportunities:
                logger.info(f"Found {len(opportunities)} fusion opportunities")

                # Apply best fusion pattern
                best_pattern = max(opportunities, key=lambda p: p.expected_speedup)

                # Create dummy operator functions for benchmarking
                dummy_ops = {op: lambda x: x for op in best_pattern.operators}

                # Fuse operators
                fused_func = fusion_engine.fuse_operators(
                    best_pattern, dummy_ops, benchmark=enable_benchmarking
                )

                # Execute with fused function
                return compute_func(*args, fused_operators=fused_func, **kwargs)
            else:
                # No fusion opportunities, execute normally
                return compute_func(*args, **kwargs)

        return wrapper

    return decorator


def nas_optimized(search_generations: int = 20, population_size: int = 10):
    """Decorator for neural architecture search optimization."""

    def decorator(pinn_training_func):
        @wraps(pinn_training_func)
        def wrapper(train_data, validation_data, physics_loss_func, *args, **kwargs):
            nas_engine = get_nas_engine(
                search_space={},  # Use defaults
                population_size=population_size,
                generations=search_generations,
            )

            # Run architecture search
            best_arch = nas_engine.search_optimal_architecture(
                train_data, validation_data, physics_loss_func
            )

            # Train final model with best architecture
            return pinn_training_func(
                train_data,
                validation_data,
                physics_loss_func,
                architecture=best_arch,
                *args,
                **kwargs,
            )

        return wrapper

    return decorator


def mixed_precision_training(precision_policy: str = "mixed_float16"):
    """Decorator for mixed precision training."""

    def decorator(training_func):
        @wraps(training_func)
        def wrapper(*args, **kwargs):
            # Set JAX default dtype for mixed precision
            original_default = jnp.float32

            if precision_policy == "mixed_float16":
                jax.config.update("jax_enable_x64", False)
                mixed_dtype = jnp.float16
            elif precision_policy == "mixed_bfloat16":
                mixed_dtype = jnp.bfloat16
            else:
                mixed_dtype = jnp.float32

            try:
                # Execute training with mixed precision
                result = training_func(
                    *args, mixed_precision_dtype=mixed_dtype, **kwargs
                )
                return result
            finally:
                # Restore original precision
                jax.config.update("jax_enable_x64", True)

        return wrapper

    return decorator
