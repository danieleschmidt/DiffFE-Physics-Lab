"""Sentiment analysis operators for physics-informed sentiment modeling."""

from typing import Dict, Any, Optional, Callable, Union, List, Tuple
import numpy as np
try:
    import jax.numpy as jnp
    import jax
    from jax import grad, jit, vmap, jacfwd, jacrev
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import Operator


class SentimentDiffusionOperator(Operator):
    """Diffusion operator for sentiment in semantic space.
    
    Models sentiment diffusion as a heat equation in embedding space,
    where sentiment flows from high to low concentration regions.
    """
    
    def __init__(self, diffusivity: float = 1.0, kernel_type: str = 'gaussian'):
        """Initialize sentiment diffusion operator.
        
        Parameters
        ----------
        diffusivity : float, optional
            Diffusion coefficient, by default 1.0
        kernel_type : str, optional
            Diffusion kernel type ('gaussian', 'laplacian'), by default 'gaussian'
        """
        super().__init__()
        self.diffusivity = diffusivity
        self.kernel_type = kernel_type
        self.is_linear = True
        
    def apply(
        self, 
        sentiment_field: np.ndarray, 
        semantic_distances: np.ndarray,
        **params
    ) -> np.ndarray:
        """Apply diffusion operator to sentiment field.
        
        Parameters
        ----------
        sentiment_field : np.ndarray
            Current sentiment values
        semantic_distances : np.ndarray
            Pairwise semantic distances between embeddings
        **params : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Diffused sentiment field
        """
        # Create diffusion kernel
        if self.kernel_type == 'gaussian':
            kernel = np.exp(-semantic_distances**2 / (2 * self.diffusivity))
        elif self.kernel_type == 'laplacian':
            kernel = np.exp(-np.abs(semantic_distances) / self.diffusivity)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
        # Normalize kernel
        kernel = kernel / np.sum(kernel, axis=1, keepdims=True)
        
        # Apply diffusion
        diffused = np.dot(kernel, sentiment_field)
        
        return self.diffusivity * (diffused - sentiment_field)
        
    def jacobian(
        self, 
        sentiment_field: np.ndarray,
        semantic_distances: np.ndarray,
        **params
    ) -> np.ndarray:
        """Compute Jacobian of diffusion operator.
        
        For linear operators, Jacobian is independent of the field.
        """
        n = len(sentiment_field)
        
        # Create diffusion kernel
        if self.kernel_type == 'gaussian':
            kernel = np.exp(-semantic_distances**2 / (2 * self.diffusivity))
        elif self.kernel_type == 'laplacian':
            kernel = np.exp(-np.abs(semantic_distances) / self.diffusivity)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
            
        # Normalize kernel
        kernel = kernel / np.sum(kernel, axis=1, keepdims=True)
        
        # Jacobian is diffusivity * (K - I)
        jacobian = self.diffusivity * (kernel - np.eye(n))
        
        return jacobian


class SentimentReactionOperator(Operator):
    """Reaction operator for sentiment dynamics.
    
    Models sentiment evolution using bistable reaction kinetics,
    creating stable positive/negative sentiment states.
    """
    
    def __init__(self, reaction_strength: float = 1.0, bistable: bool = True):
        """Initialize sentiment reaction operator.
        
        Parameters
        ----------
        reaction_strength : float, optional
            Reaction rate constant, by default 1.0
        bistable : bool, optional
            Use bistable kinetics (double-well potential), by default True
        """
        super().__init__()
        self.reaction_strength = reaction_strength
        self.bistable = bistable
        self.is_linear = False
        
    def apply(
        self, 
        sentiment_field: np.ndarray,
        **params
    ) -> np.ndarray:
        """Apply reaction operator to sentiment field.
        
        Parameters
        ----------
        sentiment_field : np.ndarray
            Current sentiment values
        **params : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Reaction term for sentiment evolution
        """
        if self.bistable:
            # Bistable reaction: f(u) = u - u^3 (double-well potential)
            reaction = self.reaction_strength * sentiment_field * (1 - sentiment_field**2)
        else:
            # Logistic reaction: f(u) = u(1 - u) (single-well)
            reaction = self.reaction_strength * sentiment_field * (1 - sentiment_field)
            
        return reaction
        
    def jacobian(
        self, 
        sentiment_field: np.ndarray,
        **params
    ) -> np.ndarray:
        """Compute Jacobian of reaction operator."""
        n = len(sentiment_field)
        jacobian = np.zeros((n, n))
        
        if self.bistable:
            # d/du [u - u^3] = 1 - 3u^2
            diagonal = self.reaction_strength * (1 - 3 * sentiment_field**2)
        else:
            # d/du [u(1-u)] = 1 - 2u
            diagonal = self.reaction_strength * (1 - 2 * sentiment_field)
            
        np.fill_diagonal(jacobian, diagonal)
        
        return jacobian


class SentimentGradientOperator(Operator):
    """Gradient operator for sentiment in semantic space.
    
    Computes gradients of sentiment with respect to semantic coordinates,
    enabling gradient-based optimization and flow analysis.
    """
    
    def __init__(self, gradient_type: str = 'finite_difference'):
        """Initialize sentiment gradient operator.
        
        Parameters
        ----------
        gradient_type : str, optional
            Gradient computation method, by default 'finite_difference'
        """
        super().__init__()
        self.gradient_type = gradient_type
        self.is_linear = True
        
    def apply(
        self,
        sentiment_field: np.ndarray,
        text_embeddings: np.ndarray,
        **params
    ) -> np.ndarray:
        """Compute gradients of sentiment field.
        
        Parameters
        ----------
        sentiment_field : np.ndarray
            Current sentiment values
        text_embeddings : np.ndarray
            Text embeddings (coordinates)
        **params : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Gradient vectors for each point
        """
        n_points, embedding_dim = text_embeddings.shape
        gradients = np.zeros((n_points, embedding_dim))
        
        if self.gradient_type == 'finite_difference':
            # Compute gradients using finite differences with k-nearest neighbors
            k = min(5, n_points - 1)  # Use 5 nearest neighbors
            
            for i in range(n_points):
                # Find k nearest neighbors
                distances = np.linalg.norm(
                    text_embeddings - text_embeddings[i], axis=1
                )
                neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self
                
                # Compute gradient using weighted finite differences
                for j in neighbor_indices:
                    direction = text_embeddings[j] - text_embeddings[i]
                    distance = np.linalg.norm(direction)
                    
                    if distance > 1e-10:  # Avoid division by zero
                        direction_normalized = direction / distance
                        sentiment_diff = sentiment_field[j] - sentiment_field[i]
                        weight = np.exp(-distance)  # Distance-based weight
                        
                        gradients[i] += weight * sentiment_diff * direction_normalized / distance
                        
                # Normalize by sum of weights
                total_weight = np.sum(np.exp(-np.linalg.norm(
                    text_embeddings[neighbor_indices] - text_embeddings[i], axis=1
                )))
                if total_weight > 1e-10:
                    gradients[i] /= total_weight
                    
        elif self.gradient_type == 'rbf':
            # Radial basis function gradient estimation
            self._compute_rbf_gradients(sentiment_field, text_embeddings, gradients)
        else:
            raise ValueError(f"Unknown gradient type: {self.gradient_type}")
            
        return gradients
        
    def _compute_rbf_gradients(
        self,
        sentiment_field: np.ndarray,
        text_embeddings: np.ndarray,
        gradients: np.ndarray
    ):
        """Compute gradients using RBF interpolation."""
        # This is a simplified RBF gradient computation
        # In practice, would use more sophisticated methods
        n_points, embedding_dim = text_embeddings.shape
        
        # Use Gaussian RBF with automatic bandwidth selection
        pairwise_distances = np.linalg.norm(
            text_embeddings[:, None, :] - text_embeddings[None, :, :], axis=2
        )
        
        # Automatic bandwidth (median heuristic)
        bandwidth = np.median(pairwise_distances[pairwise_distances > 0])
        
        # RBF matrix
        rbf_matrix = np.exp(-pairwise_distances**2 / (2 * bandwidth**2))
        
        # Solve for RBF weights
        try:
            weights = np.linalg.solve(rbf_matrix + 1e-10 * np.eye(n_points), sentiment_field)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(rbf_matrix + 1e-10 * np.eye(n_points), sentiment_field, rcond=None)[0]
        
        # Compute gradients
        for i in range(n_points):
            gradient = np.zeros(embedding_dim)
            for j in range(n_points):
                if i != j:
                    diff = text_embeddings[i] - text_embeddings[j]
                    distance_sq = np.sum(diff**2)
                    rbf_value = np.exp(-distance_sq / (2 * bandwidth**2))
                    gradient += weights[j] * rbf_value * (-diff / bandwidth**2)
            gradients[i] = gradient


class SentimentLaplacianOperator(Operator):
    """Laplacian operator for sentiment in semantic space.
    
    Computes the Laplacian (divergence of gradient) for sentiment diffusion.
    This is the key operator for sentiment heat equation dynamics.
    """
    
    def __init__(self, kernel_bandwidth: Optional[float] = None):
        """Initialize sentiment Laplacian operator.
        
        Parameters
        ----------
        kernel_bandwidth : float, optional
            Kernel bandwidth for Laplacian computation, by default None (auto)
        """
        super().__init__()
        self.kernel_bandwidth = kernel_bandwidth
        self.is_linear = True
        
    def apply(
        self,
        sentiment_field: np.ndarray,
        text_embeddings: np.ndarray,
        **params
    ) -> np.ndarray:
        """Apply Laplacian operator to sentiment field.
        
        Parameters
        ----------
        sentiment_field : np.ndarray
            Current sentiment values
        text_embeddings : np.ndarray
            Text embeddings (coordinates)
        **params : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Laplacian values for each point
        """
        n_points = len(sentiment_field)
        laplacian = np.zeros(n_points)
        
        # Compute pairwise distances
        pairwise_distances = np.linalg.norm(
            text_embeddings[:, None, :] - text_embeddings[None, :, :], axis=2
        )
        
        # Automatic bandwidth selection if not provided
        if self.kernel_bandwidth is None:
            bandwidth = np.median(pairwise_distances[pairwise_distances > 0]) * 0.5
        else:
            bandwidth = self.kernel_bandwidth
            
        # Compute Laplacian using Gaussian kernel approximation
        for i in range(n_points):
            # Find neighbors within bandwidth
            neighbor_mask = (pairwise_distances[i] < 3 * bandwidth) & (pairwise_distances[i] > 0)
            neighbor_indices = np.where(neighbor_mask)[0]
            
            if len(neighbor_indices) == 0:
                continue
                
            # Compute weighted Laplacian
            total_weight = 0
            for j in neighbor_indices:
                distance = pairwise_distances[i, j]
                weight = np.exp(-distance**2 / (2 * bandwidth**2))
                total_weight += weight
                laplacian[i] += weight * (sentiment_field[j] - sentiment_field[i])
                
            if total_weight > 1e-10:
                laplacian[i] /= (total_weight * bandwidth**2)
                
        return laplacian
        
    def compute_laplacian_matrix(
        self,
        text_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute the Laplacian matrix for the embedding space.
        
        Parameters
        ----------
        text_embeddings : np.ndarray
            Text embeddings (coordinates)
            
        Returns
        -------
        np.ndarray
            Laplacian matrix
        """
        n_points = len(text_embeddings)
        laplacian_matrix = np.zeros((n_points, n_points))
        
        # Compute pairwise distances
        pairwise_distances = np.linalg.norm(
            text_embeddings[:, None, :] - text_embeddings[None, :, :], axis=2
        )
        
        # Automatic bandwidth selection
        if self.kernel_bandwidth is None:
            bandwidth = np.median(pairwise_distances[pairwise_distances > 0]) * 0.5
        else:
            bandwidth = self.kernel_bandwidth
            
        # Build Laplacian matrix
        for i in range(n_points):
            total_weight = 0
            for j in range(n_points):
                if i != j:
                    distance = pairwise_distances[i, j]
                    if distance < 3 * bandwidth:  # Local support
                        weight = np.exp(-distance**2 / (2 * bandwidth**2))
                        laplacian_matrix[i, j] = weight / bandwidth**2
                        total_weight += weight / bandwidth**2
                        
            laplacian_matrix[i, i] = -total_weight
            
        return laplacian_matrix


class SentimentAdvectionOperator(Operator):
    """Advection operator for sentiment transport.
    
    Models directed sentiment flow based on semantic gradients
    and external driving forces.
    """
    
    def __init__(self, velocity_field: Optional[np.ndarray] = None):
        """Initialize sentiment advection operator.
        
        Parameters
        ----------
        velocity_field : np.ndarray, optional
            Velocity field for advection, by default None
        """
        super().__init__()
        self.velocity_field = velocity_field
        self.is_linear = True
        
    def apply(
        self,
        sentiment_field: np.ndarray,
        text_embeddings: np.ndarray,
        velocity_field: Optional[np.ndarray] = None,
        **params
    ) -> np.ndarray:
        """Apply advection operator to sentiment field.
        
        Parameters
        ----------
        sentiment_field : np.ndarray
            Current sentiment values
        text_embeddings : np.ndarray
            Text embeddings (coordinates)
        velocity_field : np.ndarray, optional
            Velocity field for advection
        **params : dict
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Advection term for sentiment evolution
        """
        # Use provided velocity field or default
        if velocity_field is not None:
            velocity = velocity_field
        elif self.velocity_field is not None:
            velocity = self.velocity_field
        else:
            # Default: no advection
            return np.zeros_like(sentiment_field)
            
        # Compute sentiment gradients
        gradient_op = SentimentGradientOperator()
        gradients = gradient_op.apply(sentiment_field, text_embeddings)
        
        # Compute advection: -v · ∇u
        advection = np.zeros(len(sentiment_field))
        for i in range(len(sentiment_field)):
            advection[i] = -np.dot(velocity[i], gradients[i])
            
        return advection


class CompositeSentimentOperator(Operator):
    """Composite operator combining diffusion, reaction, and advection.
    
    This is the main operator for physics-informed sentiment analysis,
    combining multiple physical processes.
    """
    
    def __init__(
        self,
        diffusion_coeff: float = 1.0,
        reaction_strength: float = 0.5,
        advection_strength: float = 0.0,
        **kwargs
    ):
        """Initialize composite sentiment operator.
        
        Parameters
        ----------
        diffusion_coeff : float, optional
            Diffusion coefficient, by default 1.0
        reaction_strength : float, optional
            Reaction strength, by default 0.5
        advection_strength : float, optional
            Advection strength, by default 0.0
        """
        super().__init__()
        
        # Create sub-operators
        self.diffusion_op = SentimentDiffusionOperator(diffusivity=diffusion_coeff)
        self.reaction_op = SentimentReactionOperator(reaction_strength=reaction_strength)
        self.laplacian_op = SentimentLaplacianOperator()
        
        self.advection_strength = advection_strength
        self.is_linear = False  # Due to reaction term
        
    def apply(
        self,
        sentiment_field: np.ndarray,
        text_embeddings: np.ndarray,
        semantic_distances: Optional[np.ndarray] = None,
        velocity_field: Optional[np.ndarray] = None,
        **params
    ) -> np.ndarray:
        """Apply composite operator to sentiment field.
        
        Computes: ∂u/∂t = D∇²u + f(u) - v·∇u
        where D is diffusion, f(u) is reaction, v·∇u is advection
        """
        result = np.zeros_like(sentiment_field)
        
        # Diffusion term: D∇²u
        if semantic_distances is not None:
            diffusion_term = self.diffusion_op.apply(sentiment_field, semantic_distances)
        else:
            # Use Laplacian approximation
            diffusion_term = self.laplacian_op.apply(sentiment_field, text_embeddings)
        result += diffusion_term
        
        # Reaction term: f(u)
        reaction_term = self.reaction_op.apply(sentiment_field)
        result += reaction_term
        
        # Advection term: -v·∇u (if enabled)
        if self.advection_strength > 0 and velocity_field is not None:
            advection_op = SentimentAdvectionOperator(velocity_field)
            advection_term = advection_op.apply(sentiment_field, text_embeddings)
            result += self.advection_strength * advection_term
            
        return result
        
    def jacobian(
        self,
        sentiment_field: np.ndarray,
        text_embeddings: np.ndarray,
        **params
    ) -> np.ndarray:
        """Compute Jacobian of composite operator."""
        n = len(sentiment_field)
        
        # Diffusion Jacobian (linear)
        diffusion_jac = self.laplacian_op.compute_laplacian_matrix(text_embeddings)
        
        # Reaction Jacobian (nonlinear, diagonal)
        reaction_jac = self.reaction_op.jacobian(sentiment_field)
        
        # Combined Jacobian
        jacobian = diffusion_jac + reaction_jac
        
        return jacobian