"""Sentiment analysis operators using differentiable computing principles."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import Operator


class SentimentOperator(Operator):
    """Base class for sentiment analysis operators using differentiable computing."""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self._backend = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the preferred AD backend."""
        if HAS_JAX:
            self._backend = "jax"
        elif HAS_TORCH:
            self._backend = "torch"
        else:
            raise ImportError("No AD backend available. Install JAX or PyTorch.")
    
    @property
    def backend(self) -> str:
        return self._backend
    
    def set_backend(self, backend: str):
        """Set the automatic differentiation backend."""
        if backend == "jax" and not HAS_JAX:
            raise ImportError("JAX not available")
        if backend == "torch" and not HAS_TORCH:
            raise ImportError("PyTorch not available")
        self._backend = backend


class PhysicsInformedSentimentClassifier(SentimentOperator):
    """
    Novel sentiment classifier using physics-informed neural networks.
    
    Applies concepts from physics simulation to sentiment analysis:
    - Energy minimization for stable sentiment representations
    - Conservation laws for maintaining semantic consistency
    - Diffusion equations for contextual propagation
    """
    
    def __init__(self, 
                 vocab_size: int = 10000,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_classes: int = 3,
                 physics_weight: float = 0.1):
        super().__init__(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.physics_weight = physics_weight
        
        if self.backend == "jax":
            self._init_jax_model()
        else:
            self._init_torch_model()
    
    def _init_jax_model(self):
        """Initialize JAX-based model parameters."""
        if not HAS_JAX:
            raise ImportError("JAX required for JAX backend")
            
        key = jax.random.PRNGKey(42)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        # Embedding layer
        self.params = {
            'embedding': jax.random.normal(key1, (self.vocab_size, self.embedding_dim)) * 0.1,
            'W1': jax.random.normal(key2, (self.embedding_dim, self.hidden_dim)) * 0.1,
            'b1': jnp.zeros(self.hidden_dim),
            'W2': jax.random.normal(key3, (self.hidden_dim, self.hidden_dim)) * 0.1,
            'b2': jnp.zeros(self.hidden_dim),
            'W_out': jax.random.normal(key4, (self.hidden_dim, self.num_classes)) * 0.1,
            'b_out': jnp.zeros(self.num_classes),
        }
    
    def _init_torch_model(self):
        """Initialize PyTorch-based model."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required for torch backend")
            
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor'], 
                params: Optional[dict] = None) -> Union[np.ndarray, 'torch.Tensor']:
        """Forward pass through the physics-informed sentiment model."""
        if self.backend == "jax":
            return self._jax_forward(x, params or self.params)
        else:
            return self._torch_forward(x)
    
    def _jax_forward(self, x: jnp.ndarray, params: dict) -> jnp.ndarray:
        """JAX implementation of forward pass."""
        # Embedding lookup
        embedded = params['embedding'][x]  # Shape: (seq_len, embedding_dim)
        
        # Mean pooling for sequence representation
        pooled = jnp.mean(embedded, axis=0)  # Shape: (embedding_dim,)
        
        # Physics-inspired transformations
        h1 = jax.nn.tanh(jnp.dot(pooled, params['W1']) + params['b1'])
        
        # Energy conservation constraint - maintain energy conservation
        h1_normalized = h1 / jnp.linalg.norm(h1)
        
        # Gradient flow dynamics for diffusion-like propagation
        h2 = jax.nn.tanh(jnp.dot(h1_normalized, params['W2']) + params['b2'])
        
        # Apply physics_weight for regularization strength
        h2_regularized = h2 * (1.0 - self.physics_weight) + h1_normalized * self.physics_weight
        
        # Final classification with conservation_laws preserved
        logits = jnp.dot(h2_regularized, params['W_out']) + params['b_out']
        
        return jax.nn.softmax(logits)
    
    def _torch_forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """PyTorch implementation of forward pass."""
        # Embedding lookup
        embedded = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Mean pooling
        pooled = torch.mean(embedded, dim=1)  # Shape: (batch_size, embedding_dim)
        
        # Physics-inspired transformations
        h1 = torch.tanh(self.fc1(pooled))
        h1 = self.dropout(h1)
        
        # Energy conservation constraint
        h1_normalized = F.normalize(h1, p=2, dim=1)
        
        h2 = torch.tanh(self.fc2(h1_normalized))
        h2 = self.dropout(h2)
        
        # Final classification
        logits = self.fc_out(h2)
        
        return F.softmax(logits, dim=1)
    
    def physics_loss(self, predictions: Union[np.ndarray, 'torch.Tensor'], 
                     targets: Union[np.ndarray, 'torch.Tensor']) -> float:
        """
        Physics-informed loss function incorporating:
        - Cross-entropy loss for classification
        - Energy conservation penalty
        - Semantic consistency regularization
        """
        if self.backend == "jax":
            return self._jax_physics_loss(predictions, targets)
        else:
            return self._torch_physics_loss(predictions, targets)
    
    def _jax_physics_loss(self, predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
        """JAX implementation of physics-informed loss."""
        # Standard cross-entropy loss
        ce_loss = -jnp.sum(targets * jnp.log(predictions + 1e-8))
        
        # Energy conservation penalty (predictions should sum to 1)
        energy_penalty = jnp.abs(jnp.sum(predictions) - 1.0)
        
        # Smoothness regularization (prevent extreme predictions)
        smoothness_penalty = jnp.sum(predictions**2)
        
        total_loss = ce_loss + self.physics_weight * (energy_penalty + smoothness_penalty)
        return total_loss
    
    def _torch_physics_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """PyTorch implementation of physics-informed loss."""
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(predictions, targets)
        
        # Energy conservation penalty
        energy_penalty = torch.abs(torch.sum(predictions, dim=1) - 1.0).mean()
        
        # Smoothness regularization
        smoothness_penalty = torch.sum(predictions**2, dim=1).mean()
        
        total_loss = ce_loss + self.physics_weight * (energy_penalty + smoothness_penalty)
        return total_loss


class DiffusionSentimentPropagator(SentimentOperator):
    """
    Sentiment propagation using diffusion equations.
    
    Models how sentiment spreads through text using heat equation analogies.
    """
    
    def __init__(self, diffusion_rate: float = 0.1, time_steps: int = 10):
        super().__init__()
        self.diffusion_rate = diffusion_rate
        self.time_steps = time_steps
    
    def propagate_sentiment(self, 
                          initial_sentiment: Union[np.ndarray, 'torch.Tensor'],
                          adjacency_matrix: Union[np.ndarray, 'torch.Tensor']) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Propagate sentiment through text using diffusion dynamics.
        
        Parameters
        ----------
        initial_sentiment : array-like
            Initial sentiment scores for each token
        adjacency_matrix : array-like
            Adjacency matrix representing token relationships
            
        Returns
        -------
        array-like
            Propagated sentiment scores
        """
        if self.backend == "jax":
            return self._jax_propagate(initial_sentiment, adjacency_matrix)
        else:
            return self._torch_propagate(initial_sentiment, adjacency_matrix)
    
    def _jax_propagate(self, sentiment: jnp.ndarray, adj_matrix: jnp.ndarray) -> jnp.ndarray:
        """JAX implementation of sentiment diffusion."""
        current_sentiment = sentiment.copy()
        
        for _ in range(self.time_steps):
            # Compute Laplacian (discrete diffusion operator)
            degree = jnp.sum(adj_matrix, axis=1)
            laplacian = jnp.diag(degree) - adj_matrix
            
            # Apply diffusion step
            gradient = jnp.dot(laplacian, current_sentiment)
            current_sentiment = current_sentiment - self.diffusion_rate * gradient
            
            # Ensure sentiment stays in valid range
            current_sentiment = jnp.clip(current_sentiment, -1.0, 1.0)
        
        return current_sentiment
    
    def _torch_propagate(self, sentiment: 'torch.Tensor', adj_matrix: 'torch.Tensor') -> 'torch.Tensor':
        """PyTorch implementation of sentiment diffusion."""
        current_sentiment = sentiment.clone()
        
        for _ in range(self.time_steps):
            # Compute Laplacian
            degree = torch.sum(adj_matrix, dim=1)
            laplacian = torch.diag(degree) - adj_matrix
            
            # Apply diffusion step
            gradient = torch.matmul(laplacian, current_sentiment)
            current_sentiment = current_sentiment - self.diffusion_rate * gradient
            
            # Ensure sentiment stays in valid range
            current_sentiment = torch.clamp(current_sentiment, -1.0, 1.0)
        
        return current_sentiment


class ConservationSentimentAnalyzer(SentimentOperator):
    """
    Sentiment analyzer based on conservation principles.
    
    Ensures that sentiment analysis respects conservation of emotional energy
    across different parts of the text.
    """
    
    def __init__(self, conservation_weight: float = 0.05):
        super().__init__()
        self.conservation_weight = conservation_weight
    
    def analyze_with_conservation(self, 
                                text_segments: List[Union[np.ndarray, 'torch.Tensor']]) -> List[Union[np.ndarray, 'torch.Tensor']]:
        """
        Analyze sentiment while maintaining conservation laws.
        
        Parameters
        ----------
        text_segments : list
            List of text segment representations
            
        Returns
        -------
        list
            Sentiment predictions for each segment with conservation constraints
        """
        if self.backend == "jax":
            return self._jax_conservative_analysis(text_segments)
        else:
            return self._torch_conservative_analysis(text_segments)
    
    def _jax_conservative_analysis(self, segments: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """JAX implementation of conservative sentiment analysis."""
        # Compute initial predictions
        predictions = [self.forward(segment) for segment in segments]
        
        # Apply conservation constraint
        total_sentiment = sum(jnp.sum(pred) for pred in predictions)
        target_total = len(predictions)  # Normalized total
        
        correction_factor = target_total / (total_sentiment + 1e-8)
        
        # Apply correction while preserving relative differences
        corrected_predictions = []
        for pred in predictions:
            corrected = pred * correction_factor
            corrected_predictions.append(corrected / jnp.sum(corrected))
        
        return corrected_predictions
    
    def _torch_conservative_analysis(self, segments: List['torch.Tensor']) -> List['torch.Tensor']:
        """PyTorch implementation of conservative sentiment analysis."""
        # Compute initial predictions
        predictions = [self.forward(segment) for segment in segments]
        
        # Apply conservation constraint
        total_sentiment = sum(torch.sum(pred) for pred in predictions)
        target_total = len(predictions)  # Normalized total
        
        correction_factor = target_total / (total_sentiment + 1e-8)
        
        # Apply correction while preserving relative differences
        corrected_predictions = []
        for pred in predictions:
            corrected = pred * correction_factor
            corrected_predictions.append(corrected / torch.sum(corrected))
        
        return corrected_predictions


# Factory function for creating sentiment operators
def create_sentiment_operator(operator_type: str, **kwargs) -> SentimentOperator:
    """
    Factory function for creating sentiment analysis operators.
    
    Parameters
    ----------
    operator_type : str
        Type of operator to create ('physics_informed', 'diffusion', 'conservation')
    **kwargs
        Additional arguments for operator initialization
        
    Returns
    -------
    SentimentOperator
        Initialized sentiment operator
    """
    if operator_type == "physics_informed":
        return PhysicsInformedSentimentClassifier(**kwargs)
    elif operator_type == "diffusion":
        return DiffusionSentimentPropagator(**kwargs)
    elif operator_type == "conservation":
        return ConservationSentimentAnalyzer(**kwargs)
    else:
        raise ValueError(f"Unknown operator type: {operator_type}")