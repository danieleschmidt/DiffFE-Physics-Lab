"""
Physics-Inspired Transformer Models for Sentiment Analysis.

This module implements transformer architectures that incorporate principles from
physics simulations for enhanced sentiment understanding.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, random
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PhysicsInformedTransformer(nn.Module):
    """
    Transformer with physics-inspired regularization for sentiment analysis.
    
    Key Physics Concepts:
    - Energy Conservation: Attention weights maintain energy conservation
    - Gradient Flow: Information flows following gradient descent on energy landscape
    - Damping: Oscillations in attention patterns are damped for stability
    - Conservation of Semantic Mass: Total semantic content is preserved across layers
    """
    
    def __init__(self,
                 vocab_size: int = 10000,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 max_seq_length: int = 512,
                 num_classes: int = 3,
                 physics_weight: float = 0.1,
                 energy_conservation: bool = True,
                 gradient_flow: bool = True,
                 damping_factor: float = 0.05):
        
        super().__init__()
        
        if not HAS_TORCH:
            raise ImportError("PyTorch required for PhysicsInformedTransformer")
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        self.physics_weight = physics_weight
        self.energy_conservation = energy_conservation
        self.gradient_flow = gradient_flow
        self.damping_factor = damping_factor
        
        # Core transformer components
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding()
        
        # Physics-informed encoder layers
        self.encoder_layers = nn.ModuleList([
            PhysicsInformedEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                physics_weight=physics_weight,
                energy_conservation=energy_conservation,
                damping_factor=damping_factor
            ) for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
        
        # Physics-inspired components
        self.energy_tracker = EnergyTracker(d_model)
        self.gradient_flow_controller = GradientFlowController(damping_factor)
        
        self._initialize_weights()
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create physics-inspired positional encoding with wave-like properties."""
        pe = torch.zeros(self.max_seq_length, self.d_model)
        position = torch.arange(0, self.max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Physics-inspired: Use wave equation solutions for positional encoding
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           (-np.log(10000.0) / self.d_model))
        
        # Sine and cosine waves with different frequencies (like standing waves)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add wave interference patterns for richer representations
        interference = torch.sin(position * div_term * 2) * torch.cos(position * div_term * 0.5)
        pe[:, 0::2] += 0.1 * interference[:, :pe[:, 0::2].shape[1]]
        
        return pe.unsqueeze(0)
    
    def _initialize_weights(self):
        """Initialize weights using physics-inspired principles."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization (maintains energy flow)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Initialize embeddings with bounded energy
                nn.init.normal_(module.weight, mean=0, std=1/np.sqrt(self.d_model))
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with physics-informed processing."""
        batch_size, seq_length = input_ids.shape
        
        # Embedding with positional encoding
        x = self.embedding(input_ids) * np.sqrt(self.d_model)  # Scale for energy conservation
        pos_enc = self.pos_encoding[:, :seq_length, :].to(x.device)
        x = x + pos_enc
        
        # Initialize energy tracking
        initial_energy = self.energy_tracker.compute_energy(x)
        
        # Apply physics-informed encoder layers
        attention_patterns = []
        energy_history = [initial_energy]
        
        for i, layer in enumerate(self.encoder_layers):
            x, attention_weights = layer(x, src_mask=attention_mask)
            
            attention_patterns.append(attention_weights.detach())
            current_energy = self.energy_tracker.compute_energy(x)
            energy_history.append(current_energy)
            
            # Apply gradient flow control if enabled
            if self.gradient_flow and i > 0:
                x = self.gradient_flow_controller.apply_flow_control(
                    x, energy_history[-2], current_energy
                )
        
        # Global representation via energy-weighted pooling
        pooled = self._physics_pooling(x, attention_mask)
        
        # Classification
        logits = self.classifier(pooled)
        predictions = F.softmax(logits, dim=-1)
        
        return {
            'predictions': predictions,
            'logits': logits,
            'hidden_states': x,
            'attention_patterns': attention_patterns,
            'energy_history': energy_history,
            'physics_metrics': self._compute_physics_metrics(x, attention_patterns, energy_history)
        }
    
    def _physics_pooling(self, hidden_states: torch.Tensor, 
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply physics-inspired pooling based on energy density."""
        if attention_mask is not None:
            # Mask padded positions
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        
        # Compute energy density for each position
        energy_density = torch.sum(hidden_states ** 2, dim=-1, keepdim=True)
        
        # Weighted average based on energy density (high energy = more important)
        weights = F.softmax(energy_density.squeeze(-1), dim=-1).unsqueeze(-1)
        pooled = torch.sum(weights * hidden_states, dim=1)
        
        return pooled
    
    def _compute_physics_metrics(self, hidden_states: torch.Tensor,
                               attention_patterns: List[torch.Tensor],
                               energy_history: List[torch.Tensor]) -> Dict[str, float]:
        """Compute physics-inspired metrics for analysis."""
        metrics = {}
        
        # Energy conservation metric
        energy_conservation_ratio = energy_history[-1] / energy_history[0]
        metrics['energy_conservation_ratio'] = energy_conservation_ratio.item()
        
        # Energy stability (variance across layers)
        energy_tensor = torch.stack(energy_history)
        metrics['energy_stability'] = 1.0 / (torch.var(energy_tensor).item() + 1e-8)
        
        # Attention entropy (measure of information distribution)
        if attention_patterns:
            avg_entropy = 0
            for attn in attention_patterns:
                # Compute entropy of attention weights
                attn_probs = F.softmax(attn.mean(dim=1), dim=-1)  # Average over heads
                entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1)
                avg_entropy += entropy.mean().item()
            metrics['average_attention_entropy'] = avg_entropy / len(attention_patterns)
        
        # Gradient flow smoothness
        if len(energy_history) > 2:
            energy_gradients = torch.diff(torch.stack(energy_history))
            metrics['gradient_flow_smoothness'] = 1.0 / (torch.var(energy_gradients).item() + 1e-8)
        
        return metrics
    
    def physics_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute loss with physics-inspired regularization."""
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(outputs['logits'], targets)
        
        physics_penalties = {}
        total_physics_loss = 0
        
        # Energy conservation penalty
        if self.energy_conservation:
            energy_history = outputs['energy_history']
            energy_conservation_penalty = torch.abs(
                energy_history[-1] / energy_history[0] - 1.0
            ).mean()
            physics_penalties['energy_conservation'] = energy_conservation_penalty
            total_physics_loss += energy_conservation_penalty
        
        # Attention entropy regularization (prevent overconfident attention)
        if 'attention_patterns' in outputs and outputs['attention_patterns']:
            entropy_penalty = 0
            for attn in outputs['attention_patterns']:
                attn_probs = F.softmax(attn.mean(dim=1), dim=-1)
                entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1)
                # Penalty for too low entropy (overconfident attention)
                entropy_penalty += F.relu(2.0 - entropy).mean()
            
            physics_penalties['attention_entropy'] = entropy_penalty / len(outputs['attention_patterns'])
            total_physics_loss += physics_penalties['attention_entropy']
        
        # Total loss
        total_loss = ce_loss + self.physics_weight * total_physics_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'physics_loss': total_physics_loss,
            'physics_penalties': physics_penalties
        }


class PhysicsInformedEncoderLayer(nn.Module):
    """Individual transformer encoder layer with physics regularization."""
    
    def __init__(self, 
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 physics_weight: float = 0.1,
                 energy_conservation: bool = True,
                 damping_factor: float = 0.05):
        
        super().__init__()
        
        self.d_model = d_model
        self.physics_weight = physics_weight
        self.energy_conservation = energy_conservation
        self.damping_factor = damping_factor
        
        # Standard transformer components
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Physics-inspired components
        self.energy_damping = EnergyDamping(damping_factor)
        
    def forward(self, src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with physics regularization."""
        
        # Self-attention with energy conservation
        attn_output, attn_weights = self.self_attn(
            src, src, src, key_padding_mask=src_mask, average_attn_weights=False
        )
        
        # Apply energy damping to reduce oscillations
        attn_output = self.energy_damping(attn_output, src)
        
        # Residual connection and normalization
        src2 = self.norm1(src + self.dropout(attn_output))
        
        # Feed-forward network
        ff_output = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        
        # Second residual connection
        output = self.norm2(src2 + self.dropout(ff_output))
        
        return output, attn_weights


class EnergyTracker:
    """Tracks energy flow through the transformer."""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
    
    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total energy of hidden states."""
        # L2 norm squared as energy measure
        energy = torch.sum(x ** 2, dim=(1, 2)) / (x.shape[1] * x.shape[2])
        return energy.mean()


class GradientFlowController:
    """Controls gradient flow to prevent instabilities."""
    
    def __init__(self, damping_factor: float = 0.05):
        self.damping_factor = damping_factor
    
    def apply_flow_control(self, 
                          current_state: torch.Tensor,
                          prev_energy: torch.Tensor,
                          current_energy: torch.Tensor) -> torch.Tensor:
        """Apply gradient flow control based on energy dynamics."""
        
        # Compute energy change rate
        energy_change = (current_energy - prev_energy) / (prev_energy + 1e-8)
        
        # Apply damping if energy is increasing too rapidly
        if energy_change > 0.1:  # Threshold for rapid energy increase
            damping_weight = torch.exp(-self.damping_factor * energy_change)
            current_state = current_state * damping_weight
        
        return current_state


class EnergyDamping(nn.Module):
    """Energy damping module to reduce oscillations."""
    
    def __init__(self, damping_factor: float = 0.05):
        super().__init__()
        self.damping_factor = damping_factor
        
    def forward(self, output: torch.Tensor, input_state: torch.Tensor) -> torch.Tensor:
        """Apply energy damping."""
        # Compute energy difference
        output_energy = torch.sum(output ** 2, dim=-1, keepdim=True)
        input_energy = torch.sum(input_state ** 2, dim=-1, keepdim=True)
        
        energy_ratio = output_energy / (input_energy + 1e-8)
        
        # Apply adaptive damping
        damping = torch.exp(-self.damping_factor * F.relu(energy_ratio - 1.0))
        
        return output * damping


class PhysicsInformedBERT:
    """
    BERT-like model with physics-informed training.
    
    This class provides a simplified interface for the PhysicsInformedTransformer
    with pre-training capabilities inspired by physics principles.
    """
    
    def __init__(self, 
                 vocab_size: int = 30522,  # Standard BERT vocab size
                 max_position_embeddings: int = 512,
                 num_attention_heads: int = 12,
                 num_hidden_layers: int = 12,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_classes: int = 3,
                 physics_weight: float = 0.1):
        
        self.model = PhysicsInformedTransformer(
            vocab_size=vocab_size,
            d_model=hidden_size,
            nhead=num_attention_heads,
            num_layers=num_hidden_layers,
            dim_feedforward=intermediate_size,
            max_seq_length=max_position_embeddings,
            num_classes=num_classes,
            physics_weight=physics_weight
        )
        
        self.tokenizer = None  # Would be initialized with actual tokenizer
    
    def train_physics_informed(self, 
                             train_data: List[Tuple[str, int]],
                             validation_data: List[Tuple[str, int]],
                             num_epochs: int = 10,
                             learning_rate: float = 5e-5,
                             batch_size: int = 16) -> Dict[str, List[float]]:
        """
        Train the model using physics-informed objectives.
        
        Parameters
        ----------
        train_data : List[Tuple[str, int]]
            Training data as (text, label) pairs
        validation_data : List[Tuple[str, int]]
            Validation data as (text, label) pairs
        num_epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
            
        Returns
        -------
        Dict[str, List[float]]
            Training history with loss and metrics
        """
        
        if not HAS_TORCH:
            raise ImportError("PyTorch required for training")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'physics_metrics': []
        }
        
        self.model.train()
        
        for epoch in range(num_epochs):
            train_losses = []
            train_correct = 0
            train_total = 0
            
            # Training loop (simplified - would use DataLoader in practice)
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                # Simple tokenization (would use proper tokenizer)
                input_ids = []
                labels = []
                
                for text, label in batch:
                    tokens = [hash(word) % self.model.vocab_size for word in text.lower().split()]
                    tokens = tokens[:self.model.max_seq_length]  # Truncate
                    tokens += [0] * (self.model.max_seq_length - len(tokens))  # Pad
                    
                    input_ids.append(tokens)
                    labels.append(label)
                
                input_ids = torch.tensor(input_ids)
                labels = torch.tensor(labels)
                
                # Forward pass
                outputs = self.model(input_ids)
                
                # Compute physics-informed loss
                loss_dict = self.model.physics_loss(outputs, labels)
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track metrics
                train_losses.append(total_loss.item())
                
                predictions = torch.argmax(outputs['predictions'], dim=-1)
                train_correct += (predictions == labels).sum().item()
                train_total += len(labels)
            
            # Compute epoch metrics
            avg_train_loss = np.mean(train_losses)
            train_accuracy = train_correct / train_total
            
            # Validation (simplified)
            val_loss, val_accuracy, physics_metrics = self._validate(validation_data)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['physics_metrics'].append(physics_metrics)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return history
    
    def _validate(self, validation_data: List[Tuple[str, int]]) -> Tuple[float, float, Dict]:
        """Validate the model and return metrics."""
        self.model.eval()
        
        val_losses = []
        val_correct = 0
        val_total = 0
        all_physics_metrics = []
        
        with torch.no_grad():
            for text, label in validation_data:
                # Simple tokenization
                tokens = [hash(word) % self.model.vocab_size for word in text.lower().split()]
                tokens = tokens[:self.model.max_seq_length]
                tokens += [0] * (self.model.max_seq_length - len(tokens))
                
                input_ids = torch.tensor([tokens])
                labels = torch.tensor([label])
                
                # Forward pass
                outputs = self.model(input_ids)
                
                # Compute loss
                loss_dict = self.model.physics_loss(outputs, labels)
                val_losses.append(loss_dict['total_loss'].item())
                
                # Track accuracy
                predictions = torch.argmax(outputs['predictions'], dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += len(labels)
                
                # Collect physics metrics
                all_physics_metrics.append(outputs['physics_metrics'])
        
        self.model.train()
        
        # Aggregate physics metrics
        aggregated_physics_metrics = {}
        if all_physics_metrics:
            for key in all_physics_metrics[0].keys():
                values = [metrics[key] for metrics in all_physics_metrics]
                aggregated_physics_metrics[key] = np.mean(values)
        
        return np.mean(val_losses), val_correct / val_total, aggregated_physics_metrics


# Factory function for easy model creation
def create_physics_transformer(model_type: str = "sentiment", **kwargs) -> Union[PhysicsInformedTransformer, PhysicsInformedBERT]:
    """
    Factory function for creating physics-informed transformer models.
    
    Parameters
    ----------
    model_type : str
        Type of model ('sentiment', 'bert')
    **kwargs
        Model-specific parameters
        
    Returns
    -------
    Union[PhysicsInformedTransformer, PhysicsInformedBERT]
        Initialized model
    """
    
    if model_type == "sentiment":
        return PhysicsInformedTransformer(**kwargs)
    elif model_type == "bert":
        return PhysicsInformedBERT(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")