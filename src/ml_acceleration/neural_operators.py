"""Neural Operators for Learning Solution Operators of PDEs.

Implementation of state-of-the-art neural operators including Fourier Neural
Operators (FNO), DeepONet, and Graph Neural Operators for learning mappings
between function spaces.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, List, Tuple, Optional, Callable, Any, Sequence
import logging
from dataclasses import dataclass
from functools import partial

from ..backends.base import Backend
from ..utils.validation import validate_neural_operator_input


@dataclass
class FNOConfig:
    """Configuration for Fourier Neural Operator."""
    modes: Sequence[int] = (16, 16)  # Fourier modes to keep
    width: int = 64                  # Channel width
    n_layers: int = 4               # Number of FNO layers
    activation: str = "gelu"        # Activation function
    dropout_rate: float = 0.0       # Dropout rate
    use_batch_norm: bool = False    # Use batch normalization
    padding_mode: str = "circular"  # Padding mode for FFT
    factorization: str = "dense"    # dense, cp, tucker
    rank: Optional[int] = None      # Rank for tensor decomposition


class SpectralConv(nn.Module):
    """Spectral convolution layer using FFT."""
    
    modes: Sequence[int]
    width: int
    factorization: str = "dense"
    rank: Optional[int] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor of shape (batch, spatial_dims..., channels)
        Returns:
            Output tensor of same shape
        """
        batch_size = x.shape[0]
        spatial_dims = x.shape[1:-1]
        in_channels = x.shape[-1]
        
        # Take FFT
        x_ft = jnp.fft.rfftn(x, axes=range(1, len(spatial_dims) + 1))
        
        # Initialize spectral weights
        if self.factorization == "dense":
            # Dense spectral weights
            weight_shape = self.modes + (in_channels, self.width)
            weights = self.param('spectral_weights',
                               nn.initializers.he_normal(),
                               weight_shape,
                               jnp.complex64)
        elif self.factorization == "cp":
            # CP decomposition for parameter efficiency
            rank = self.rank or min(in_channels, self.width) // 2
            weight_factors = []
            for i, mode in enumerate(self.modes):
                factor = self.param(f'cp_factor_{i}',
                                  nn.initializers.he_normal(),
                                  (mode, rank),
                                  jnp.complex64)
                weight_factors.append(factor)
            
            # Channel factors
            factor_in = self.param('cp_factor_in',
                                 nn.initializers.he_normal(),
                                 (in_channels, rank),
                                 jnp.complex64)
            factor_out = self.param('cp_factor_out',
                                  nn.initializers.he_normal(),
                                  (self.width, rank),
                                  jnp.complex64)
            
            # Reconstruct weights using CP decomposition
            weights = self._reconstruct_cp_weights(weight_factors, factor_in, factor_out)
        
        # Apply spectral convolution
        out_ft = self._spectral_multiply(x_ft, weights)
        
        # Take inverse FFT
        out = jnp.fft.irfftn(out_ft, s=spatial_dims, axes=range(1, len(spatial_dims) + 1))
        
        return out
    
    def _spectral_multiply(self, x_ft: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """Multiply in spectral domain with mode truncation."""
        # Extract relevant modes
        if len(self.modes) == 1:
            x_ft_modes = x_ft[:, :self.modes[0], :]
        elif len(self.modes) == 2:
            x_ft_modes = x_ft[:, :self.modes[0], :self.modes[1], :]
        elif len(self.modes) == 3:
            x_ft_modes = x_ft[:, :self.modes[0], :self.modes[1], :self.modes[2], :]
        else:
            raise NotImplementedError(f"Modes {self.modes} not supported")
        
        # Spectral convolution: pointwise multiplication in frequency domain
        out_ft_modes = jnp.einsum('b...i,....ij->b...j', x_ft_modes, weights)
        
        # Pad back to original size
        pad_widths = []
        pad_widths.append((0, 0))  # Batch dimension
        
        for i, (mode, size) in enumerate(zip(self.modes, x_ft.shape[1:-1])):
            if i < len(self.modes) - 1:
                pad_widths.append((0, size - mode))
            else:
                # Last dimension (for rfft) needs special handling
                pad_widths.append((0, size - mode))
        
        pad_widths.append((0, 0))  # Channel dimension
        
        out_ft = jnp.pad(out_ft_modes, pad_widths, mode='constant')
        
        return out_ft
    
    def _reconstruct_cp_weights(self, spatial_factors: List[jnp.ndarray],
                               factor_in: jnp.ndarray, factor_out: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct weights from CP decomposition."""
        # Start with channel factors
        weights = jnp.einsum('ir,jr->ij', factor_in, factor_out)
        
        # Add spatial dimensions
        for factor in spatial_factors:
            weights = jnp.einsum('...ij,kr->...krij', weights, factor)
        
        return weights


class FNOLayer(nn.Module):
    """Single FNO layer with spectral and local convolutions."""
    
    modes: Sequence[int]
    width: int
    activation: Callable = nn.gelu
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Spectral convolution
        spectral_out = SpectralConv(modes=self.modes, width=self.width)(x)
        
        # Local convolution (1x1 conv)
        local_out = nn.Conv(features=self.width, kernel_size=(1,) * (len(x.shape) - 2))(x)
        
        # Combine spectral and local
        out = spectral_out + local_out
        
        # Batch normalization
        if self.use_batch_norm:
            out = nn.BatchNorm(use_running_average=not training)(out)
        
        # Activation
        out = self.activation(out)
        
        # Dropout
        if self.dropout_rate > 0.0:
            out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(out)
        
        return out


class FourierNeuralOperator(nn.Module):
    """Fourier Neural Operator for learning solution operators.
    
    FNO learns mappings between function spaces by parameterizing
    integral operators in Fourier space, enabling resolution-invariant
    learning of PDE solution operators.
    """
    
    config: FNOConfig
    output_dim: int = 1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input function samples, shape (batch, spatial_dims..., input_channels)
            training: Whether in training mode
        Returns:
            Output function samples, shape (batch, spatial_dims..., output_dim)
        """
        # Input projection
        x = nn.Conv(features=self.config.width, 
                   kernel_size=(1,) * (len(x.shape) - 2))(x)
        
        # FNO layers
        for _ in range(self.config.n_layers):
            residual = x
            x = FNOLayer(modes=self.config.modes,
                        width=self.config.width,
                        activation=getattr(nn, self.config.activation),
                        dropout_rate=self.config.dropout_rate,
                        use_batch_norm=self.config.use_batch_norm)(x, training)
            
            # Residual connection
            x = x + residual
        
        # Output projection
        x = nn.Conv(features=128, kernel_size=(1,) * (len(x.shape) - 2))(x)
        x = getattr(nn, self.config.activation)(x)
        x = nn.Conv(features=self.output_dim, kernel_size=(1,) * (len(x.shape) - 2))(x)
        
        return x


@dataclass
class DeepONetConfig:
    """Configuration for DeepONet."""
    branch_layers: Sequence[int] = (128, 128, 128)
    trunk_layers: Sequence[int] = (128, 128, 128)
    activation: str = "tanh"
    use_bias: bool = True
    dropout_rate: float = 0.0
    output_transform: Optional[str] = None  # None, "sigmoid", "tanh"


class DeepONet(nn.Module):
    """Deep Operator Network for learning nonlinear operators.
    
    DeepONet learns operators by using two neural networks:
    - Branch net: processes input functions
    - Trunk net: processes coordinates
    The output is their dot product, enabling operator learning.
    """
    
    config: DeepONetConfig
    output_dim: int = 1
    
    def setup(self):
        # Branch network (processes input functions)
        branch_layers = []
        for width in self.config.branch_layers:
            branch_layers.extend([
                nn.Dense(width, use_bias=self.config.use_bias),
                getattr(nn, self.config.activation),
            ])
        if self.config.dropout_rate > 0:
            branch_layers.append(nn.Dropout(self.config.dropout_rate))
        
        branch_layers.append(nn.Dense(self.config.trunk_layers[-1] * self.output_dim))
        self.branch_net = nn.Sequential(branch_layers)
        
        # Trunk network (processes coordinates)
        trunk_layers = []
        for width in self.config.trunk_layers[:-1]:
            trunk_layers.extend([
                nn.Dense(width, use_bias=self.config.use_bias),
                getattr(nn, self.config.activation),
            ])
        if self.config.dropout_rate > 0:
            trunk_layers.append(nn.Dropout(self.config.dropout_rate))
        
        trunk_layers.append(nn.Dense(self.config.trunk_layers[-1]))
        self.trunk_net = nn.Sequential(trunk_layers)
        
        # Bias term
        if self.config.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (self.output_dim,))
    
    def __call__(self, branch_input: jnp.ndarray, trunk_input: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        """
        Args:
            branch_input: Input functions, shape (batch, n_sensors, input_dim)
            trunk_input: Coordinate points, shape (batch, n_points, coord_dim)
            training: Whether in training mode
        Returns:
            Operator output, shape (batch, n_points, output_dim)
        """
        # Flatten branch input (function sensors)
        branch_flat = branch_input.reshape(branch_input.shape[0], -1)
        
        # Process through networks
        branch_out = self.branch_net(branch_flat, training=training)
        trunk_out = self.trunk_net(trunk_input, training=training)
        
        # Reshape branch output for dot product
        branch_out = branch_out.reshape(
            branch_out.shape[0], self.config.trunk_layers[-1], self.output_dim)
        
        # Compute dot product
        output = jnp.einsum('bto,bpt->bpo', branch_out, trunk_out)
        
        # Add bias
        if self.config.use_bias:
            output = output + self.bias
        
        # Apply output transform
        if self.config.output_transform == "sigmoid":
            output = nn.sigmoid(output)
        elif self.config.output_transform == "tanh":
            output = nn.tanh(output)
        
        return output


@dataclass
class GNOConfig:
    """Configuration for Graph Neural Operator."""
    hidden_dim: int = 64
    n_layers: int = 4
    message_passing_steps: int = 3
    activation: str = "relu"
    use_edge_features: bool = True
    use_global_features: bool = False
    aggregation: str = "mean"  # mean, sum, max
    dropout_rate: float = 0.0


class MessagePassingLayer(nn.Module):
    """Message passing layer for graph neural networks."""
    
    hidden_dim: int
    activation: Callable = nn.relu
    use_edge_features: bool = True
    aggregation: str = "mean"
    
    @nn.compact
    def __call__(self, node_features: jnp.ndarray, edge_features: jnp.ndarray,
                 adjacency: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            node_features: Node features, shape (n_nodes, node_dim)
            edge_features: Edge features, shape (n_edges, edge_dim)
            adjacency: Adjacency matrix, shape (n_nodes, n_nodes)
            training: Whether in training mode
        Returns:
            Updated node features, shape (n_nodes, hidden_dim)
        """
        n_nodes = node_features.shape[0]
        
        # Message computation
        message_mlp = nn.Sequential([
            nn.Dense(self.hidden_dim),
            self.activation,
            nn.Dense(self.hidden_dim),
        ])
        
        # Prepare messages
        messages = jnp.zeros((n_nodes, n_nodes, self.hidden_dim))
        
        # For each edge, compute message
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adjacency[i, j] > 0:  # Edge exists
                    # Concatenate node features
                    edge_input = jnp.concatenate([node_features[i], node_features[j]])
                    
                    # Add edge features if available
                    if self.use_edge_features and edge_features is not None:
                        edge_idx = i * n_nodes + j  # Simplified edge indexing
                        if edge_idx < edge_features.shape[0]:
                            edge_input = jnp.concatenate([edge_input, edge_features[edge_idx]])
                    
                    # Compute message
                    message = message_mlp(edge_input)
                    messages = messages.at[i, j].set(message)
        
        # Aggregate messages
        if self.aggregation == "mean":
            # Compute degree for normalization
            degrees = jnp.sum(adjacency, axis=1, keepdims=True)
            degrees = jnp.where(degrees == 0, 1, degrees)  # Avoid division by zero
            aggregated = jnp.sum(messages, axis=1) / degrees
        elif self.aggregation == "sum":
            aggregated = jnp.sum(messages, axis=1)
        elif self.aggregation == "max":
            aggregated = jnp.max(messages, axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Update function
        update_mlp = nn.Sequential([
            nn.Dense(self.hidden_dim),
            self.activation,
            nn.Dense(self.hidden_dim),
        ])
        
        # Combine node features with aggregated messages
        combined = jnp.concatenate([node_features, aggregated], axis=-1)
        updated_features = update_mlp(combined)
        
        return updated_features


class GraphNeuralOperator(nn.Module):
    """Graph Neural Operator for learning operators on irregular geometries.
    
    GNO extends neural operators to handle irregular meshes and geometries
    by using graph neural networks to process spatial relationships.
    """
    
    config: GNOConfig
    output_dim: int = 1
    
    def setup(self):
        # Input embedding
        self.input_embedding = nn.Dense(self.config.hidden_dim)
        
        # Message passing layers
        self.mp_layers = [
            MessagePassingLayer(
                hidden_dim=self.config.hidden_dim,
                activation=getattr(nn, self.config.activation),
                use_edge_features=self.config.use_edge_features,
                aggregation=self.config.aggregation
            )
            for _ in range(self.config.n_layers)
        ]
        
        # Output projection
        self.output_projection = nn.Sequential([
            nn.Dense(self.config.hidden_dim),
            getattr(nn, self.config.activation),
            nn.Dense(self.output_dim),
        ])
    
    def __call__(self, node_features: jnp.ndarray, edge_features: jnp.ndarray,
                 adjacency: jnp.ndarray, coordinates: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        """
        Args:
            node_features: Node features, shape (n_nodes, node_dim)
            edge_features: Edge features, shape (n_edges, edge_dim) 
            adjacency: Adjacency matrix, shape (n_nodes, n_nodes)
            coordinates: Node coordinates, shape (n_nodes, coord_dim)
            training: Whether in training mode
        Returns:
            Node outputs, shape (n_nodes, output_dim)
        """
        # Embed input features
        x = self.input_embedding(node_features)
        
        # Add coordinate information
        coord_embedding = nn.Dense(self.config.hidden_dim)(coordinates)
        x = x + coord_embedding
        
        # Message passing
        for mp_layer in self.mp_layers:
            residual = x
            x = mp_layer(x, edge_features, adjacency, training)
            
            # Residual connection
            if x.shape == residual.shape:
                x = x + residual
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class NeuralOperatorTrainer:
    """Trainer for neural operators with physics-informed losses."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Initialize training state
        self.state = None
        self.metrics_history = []
        
        # Loss functions
        self.data_loss_fn = self._create_data_loss()
        self.physics_loss_fn = self._create_physics_loss()
        
    def _create_data_loss(self) -> Callable:
        """Create data fitting loss function."""
        def data_loss(params, batch):
            predictions = self.model.apply(params, **batch['inputs'], training=True)
            targets = batch['targets']
            
            # L2 loss
            loss = jnp.mean((predictions - targets) ** 2)
            
            return loss, {'data_loss': loss}
        
        return jit(data_loss)
    
    def _create_physics_loss(self) -> Callable:
        """Create physics-informed loss function."""
        def physics_loss(params, batch):
            # This would implement PDE residual computation
            # For now, return zero loss
            return 0.0, {'physics_loss': 0.0}
        
        return jit(physics_loss)
    
    def create_train_state(self, rng: jax.random.PRNGKey, 
                          sample_input: Dict[str, jnp.ndarray]) -> train_state.TrainState:
        """Create initial training state."""
        # Initialize parameters
        params = self.model.init(rng, **sample_input, training=True)
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=self.config.get('learning_rate', 1e-3))
        
        # Create training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )
        
        return self.state
    
    @partial(jit, static_argnums=(0,))
    def train_step(self, state: train_state.TrainState, batch: Dict[str, Any]) -> Tuple[train_state.TrainState, Dict[str, Any]]:
        """Single training step."""
        def loss_fn(params):
            # Data loss
            data_loss, data_metrics = self.data_loss_fn(params, batch)
            
            # Physics loss (if applicable)
            physics_loss, physics_metrics = self.physics_loss_fn(params, batch)
            
            # Total loss
            total_loss = data_loss + self.config.get('physics_weight', 0.0) * physics_loss
            
            metrics = {**data_metrics, **physics_metrics, 'total_loss': total_loss}
            return total_loss, metrics
        
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, metrics
    
    def train(self, train_dataset: Any, val_dataset: Any, 
              num_epochs: int) -> Dict[str, List[float]]:
        """Train the neural operator."""
        if self.state is None:
            raise ValueError("Training state not initialized. Call create_train_state first.")
        
        train_metrics_history = []
        val_metrics_history = []
        
        for epoch in range(num_epochs):
            # Training
            epoch_train_metrics = []
            for batch in train_dataset:
                self.state, metrics = self.train_step(self.state, batch)
                epoch_train_metrics.append(metrics)
            
            # Average training metrics
            avg_train_metrics = {}
            for key in epoch_train_metrics[0].keys():
                avg_train_metrics[key] = jnp.mean(
                    jnp.array([m[key] for m in epoch_train_metrics]))
            
            train_metrics_history.append(avg_train_metrics)
            
            # Validation
            if val_dataset is not None:
                val_metrics = self.evaluate(val_dataset)
                val_metrics_history.append(val_metrics)
            
            # Logging
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: train_loss = {avg_train_metrics['total_loss']:.6f}")
                if val_dataset is not None:
                    logging.info(f"  val_loss = {val_metrics['total_loss']:.6f}")
        
        return {
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history
        }
    
    def evaluate(self, dataset: Any) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        metrics_list = []
        
        for batch in dataset:
            # Data loss
            data_loss, data_metrics = self.data_loss_fn(self.state.params, batch)
            physics_loss, physics_metrics = self.physics_loss_fn(self.state.params, batch)
            
            total_loss = data_loss + self.config.get('physics_weight', 0.0) * physics_loss
            
            metrics = {**data_metrics, **physics_metrics, 'total_loss': total_loss}
            metrics_list.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[key] = jnp.mean(jnp.array([m[key] for m in metrics_list]))
        
        return avg_metrics
    
    def predict(self, inputs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Make predictions with the trained model."""
        return self.model.apply(self.state.params, **inputs, training=False)


# Utility functions for neural operators
def generate_training_data_pde(pde_solver: Callable, n_samples: int = 1000,
                              input_dim: int = 64) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate training data for neural operators using PDE solver."""
    key = random.PRNGKey(42)
    
    inputs = []
    outputs = []
    
    for _ in range(n_samples):
        key, subkey = random.split(key)
        
        # Generate random input function (e.g., initial condition)
        input_func = random.normal(subkey, (input_dim, input_dim))
        
        # Solve PDE to get output function
        output_func = pde_solver(input_func)
        
        inputs.append(input_func)
        outputs.append(output_func)
    
    return jnp.array(inputs), jnp.array(outputs)


def create_mesh_graph(coordinates: jnp.ndarray, 
                     connectivity: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create graph representation from mesh."""
    n_nodes = coordinates.shape[0]
    
    # Create adjacency matrix
    adjacency = jnp.zeros((n_nodes, n_nodes))
    
    # Fill adjacency based on connectivity
    for element in connectivity:
        for i in range(len(element)):
            for j in range(i + 1, len(element)):
                node_i, node_j = element[i], element[j]
                adjacency = adjacency.at[node_i, node_j].set(1.0)
                adjacency = adjacency.at[node_j, node_i].set(1.0)
    
    # Create edge features (distances)
    edge_features = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adjacency[i, j] > 0:
                distance = jnp.linalg.norm(coordinates[i] - coordinates[j])
                edge_features.append(distance)
    
    edge_features = jnp.array(edge_features).reshape(-1, 1)
    
    return adjacency, edge_features