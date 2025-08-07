"""Sentiment analysis problem definition for physics-informed sentiment modeling."""

from typing import Dict, Any, Optional, Callable, Union, List, Tuple
import numpy as np
try:
    import jax.numpy as jnp
    import jax
    from jax import grad, jit, vmap
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .problem import Problem
from ..operators.base import Operator
from ..utils.validation import validate_text_input, validate_sentiment_score


class SentimentProblem(Problem):
    """Physics-informed sentiment analysis problem.
    
    This class extends the base Problem class to handle sentiment analysis
    using physics-informed neural networks and finite element methods.
    Sentiment is modeled as a diffusion-reaction system in semantic space.
    
    Parameters
    ----------
    text_embeddings : array_like
        Text embeddings representing semantic space coordinates
    sentiment_field : array_like, optional
        Initial sentiment field values
    backend : str, optional
        AD backend ('jax' or 'torch'), by default 'jax'
    temperature : float, optional
        Temperature parameter for sentiment diffusion, by default 1.0
    reaction_strength : float, optional
        Reaction strength for sentiment dynamics, by default 0.5
    """
    
    def __init__(
        self,
        text_embeddings: np.ndarray,
        sentiment_field: Optional[np.ndarray] = None,
        backend: str = 'jax',
        temperature: float = 1.0,
        reaction_strength: float = 0.5,
        **kwargs
    ):
        # Initialize base problem without mesh (semantic space is abstract)
        super().__init__(mesh=None, function_space=None, backend=backend, **kwargs)
        
        # Validate inputs
        if not isinstance(text_embeddings, np.ndarray):
            raise ValueError("text_embeddings must be numpy array")
        if text_embeddings.ndim != 2:
            raise ValueError("text_embeddings must be 2D array (n_samples, embedding_dim)")
            
        self.text_embeddings = text_embeddings
        self.n_samples, self.embedding_dim = text_embeddings.shape
        
        # Initialize sentiment field if not provided
        if sentiment_field is None:
            self.sentiment_field = np.zeros(self.n_samples)
        else:
            if len(sentiment_field) != self.n_samples:
                raise ValueError("sentiment_field length must match number of samples")
            self.sentiment_field = np.array(sentiment_field)
            
        # Physics parameters
        self.temperature = temperature
        self.reaction_strength = reaction_strength
        
        # Semantic space metrics
        self._compute_semantic_distances()
        
        # Initialize backend-specific components
        self._setup_backend_functions()
        
    def _compute_semantic_distances(self):
        """Compute pairwise semantic distances for diffusion kernel."""
        # Euclidean distance in embedding space
        distances = np.linalg.norm(
            self.text_embeddings[:, None, :] - self.text_embeddings[None, :, :], 
            axis=2
        )
        
        # Convert to similarity kernel (higher for closer embeddings)
        self.semantic_kernel = np.exp(-distances / self.temperature)
        
        # Normalize to ensure proper diffusion
        self.semantic_kernel = self.semantic_kernel / np.sum(self.semantic_kernel, axis=1, keepdims=True)
        
    def _setup_backend_functions(self):
        """Setup backend-specific functions for sentiment analysis."""
        if self.backend_name == 'jax' and HAS_JAX:
            self._setup_jax_functions()
        elif self.backend_name == 'torch' and HAS_TORCH:
            self._setup_torch_functions()
        else:
            self._setup_numpy_functions()
            
    def _setup_jax_functions(self):
        """Setup JAX-specific optimized functions."""
        # Convert to JAX arrays
        self.text_embeddings_jax = jnp.array(self.text_embeddings)
        self.semantic_kernel_jax = jnp.array(self.semantic_kernel)
        
        @jit
        def sentiment_diffusion_step(sentiment, dt=0.01):
            """Single diffusion step in semantic space."""
            diffused = jnp.dot(self.semantic_kernel_jax, sentiment)
            return sentiment + dt * (diffused - sentiment)
            
        @jit 
        def sentiment_reaction_step(sentiment, dt=0.01):
            """Reaction step for sentiment dynamics."""
            # Logistic reaction: sentiment tends toward stable states
            reaction = self.reaction_strength * sentiment * (1 - sentiment**2)
            return sentiment + dt * reaction
            
        @jit
        def compute_sentiment_energy(sentiment):
            """Compute total energy of sentiment field."""
            # Kinetic energy from gradients + potential from sentiment values
            kinetic = 0.5 * jnp.sum((jnp.dot(self.semantic_kernel_jax, sentiment) - sentiment)**2)
            potential = 0.25 * self.reaction_strength * jnp.sum(sentiment**4 - sentiment**2)
            return kinetic + potential
            
        self.diffusion_step = sentiment_diffusion_step
        self.reaction_step = sentiment_reaction_step  
        self.compute_energy = compute_sentiment_energy
        
        # Gradient functions
        self.energy_gradient = jit(grad(compute_sentiment_energy))
        
    def _setup_torch_functions(self):
        """Setup PyTorch-specific functions."""
        # Convert to torch tensors
        self.text_embeddings_torch = torch.tensor(self.text_embeddings, dtype=torch.float32)
        self.semantic_kernel_torch = torch.tensor(self.semantic_kernel, dtype=torch.float32)
        
        def sentiment_diffusion_step(sentiment, dt=0.01):
            """Single diffusion step in semantic space."""
            diffused = torch.mm(self.semantic_kernel_torch, sentiment.unsqueeze(1)).squeeze()
            return sentiment + dt * (diffused - sentiment)
            
        def sentiment_reaction_step(sentiment, dt=0.01):
            """Reaction step for sentiment dynamics."""
            reaction = self.reaction_strength * sentiment * (1 - sentiment**2)
            return sentiment + dt * reaction
            
        def compute_sentiment_energy(sentiment):
            """Compute total energy of sentiment field."""
            diffused = torch.mm(self.semantic_kernel_torch, sentiment.unsqueeze(1)).squeeze()
            kinetic = 0.5 * torch.sum((diffused - sentiment)**2)
            potential = 0.25 * self.reaction_strength * torch.sum(sentiment**4 - sentiment**2)
            return kinetic + potential
            
        self.diffusion_step = sentiment_diffusion_step
        self.reaction_step = sentiment_reaction_step
        self.compute_energy = compute_sentiment_energy
        
    def _setup_numpy_functions(self):
        """Setup NumPy fallback functions."""
        def sentiment_diffusion_step(sentiment, dt=0.01):
            """Single diffusion step in semantic space."""
            diffused = np.dot(self.semantic_kernel, sentiment)
            return sentiment + dt * (diffused - sentiment)
            
        def sentiment_reaction_step(sentiment, dt=0.01):
            """Reaction step for sentiment dynamics."""
            reaction = self.reaction_strength * sentiment * (1 - sentiment**2)
            return sentiment + dt * reaction
            
        def compute_sentiment_energy(sentiment):
            """Compute total energy of sentiment field."""
            diffused = np.dot(self.semantic_kernel, sentiment)
            kinetic = 0.5 * np.sum((diffused - sentiment)**2)
            potential = 0.25 * self.reaction_strength * np.sum(sentiment**4 - sentiment**2)
            return kinetic + potential
            
        self.diffusion_step = sentiment_diffusion_step
        self.reaction_step = sentiment_reaction_step
        self.compute_energy = compute_sentiment_energy
        
    def analyze_sentiment(
        self, 
        texts: List[str], 
        num_steps: int = 100,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Analyze sentiment using physics-informed dynamics.
        
        Parameters
        ----------
        texts : List[str]
            Input texts to analyze
        num_steps : int, optional
            Number of integration steps, by default 100
        dt : float, optional
            Time step size, by default 0.01
            
        Returns
        -------
        sentiments : np.ndarray
            Final sentiment scores [-1, 1]
        info : Dict[str, Any]
            Analysis information and diagnostics
        """
        if len(texts) != self.n_samples:
            raise ValueError(f"Expected {self.n_samples} texts, got {len(texts)}")
            
        # Initialize sentiment field
        sentiment = self.sentiment_field.copy()
        
        # Track evolution
        energies = []
        sentiment_history = [sentiment.copy()]
        
        # Physics-informed evolution
        for step in range(num_steps):
            # Diffusion step (semantic space smoothing)
            sentiment = self.diffusion_step(sentiment, dt)
            
            # Reaction step (sentiment dynamics)
            sentiment = self.reaction_step(sentiment, dt)
            
            # Clamp to valid range
            sentiment = np.clip(sentiment, -1, 1)
            
            # Track energy
            if step % 10 == 0:
                energy = float(self.compute_energy(sentiment))
                energies.append(energy)
                sentiment_history.append(sentiment.copy())
                
        # Final sentiment scores
        final_sentiments = np.tanh(sentiment)  # Ensure [-1, 1] range
        
        # Analysis diagnostics
        info = {
            'num_steps': num_steps,
            'final_energy': energies[-1] if energies else None,
            'energy_history': energies,
            'sentiment_evolution': sentiment_history,
            'converged': len(energies) > 1 and abs(energies[-1] - energies[-2]) < 1e-6,
            'mean_sentiment': np.mean(final_sentiments),
            'sentiment_variance': np.var(final_sentiments),
            'extreme_sentiments': np.sum(np.abs(final_sentiments) > 0.8)
        }
        
        return final_sentiments, info
        
    def train_on_labeled_data(
        self,
        labeled_texts: List[str],
        true_sentiments: np.ndarray,
        num_epochs: int = 50,
        learning_rate: float = 0.01
    ) -> Dict[str, Any]:
        """Train sentiment model on labeled data using physics-informed loss.
        
        Parameters
        ----------
        labeled_texts : List[str]
            Training texts
        true_sentiments : np.ndarray
            True sentiment labels [-1, 1]
        num_epochs : int, optional
            Number of training epochs, by default 50
        learning_rate : float, optional
            Learning rate, by default 0.01
            
        Returns
        -------
        training_info : Dict[str, Any]
            Training diagnostics and final parameters
        """
        if len(labeled_texts) != len(true_sentiments):
            raise ValueError("Number of texts and labels must match")
            
        # Training history
        losses = []
        best_loss = float('inf')
        best_params = None
        
        for epoch in range(num_epochs):
            # Forward pass
            predicted_sentiments, _ = self.analyze_sentiment(labeled_texts)
            
            # Physics-informed loss: MSE + energy regularization
            mse_loss = np.mean((predicted_sentiments - true_sentiments)**2)
            energy_reg = self.compute_energy(predicted_sentiments)
            total_loss = mse_loss + 0.01 * energy_reg
            
            losses.append(float(total_loss))
            
            # Track best parameters
            if total_loss < best_loss:
                best_loss = total_loss
                best_params = {
                    'temperature': self.temperature,
                    'reaction_strength': self.reaction_strength,
                    'sentiment_field': self.sentiment_field.copy()
                }
                
            # Simple parameter updates (gradient-free for simplicity)
            if epoch > 0 and total_loss > losses[-2]:
                # If loss increased, reduce learning rate and revert
                learning_rate *= 0.9
                if best_params:
                    self.temperature = best_params['temperature']
                    self.reaction_strength = best_params['reaction_strength']
                    self.sentiment_field = best_params['sentiment_field']
            else:
                # Update parameters based on loss gradients (simplified)
                temp_grad = (total_loss - (losses[-2] if len(losses) > 1 else total_loss))
                self.temperature = np.clip(self.temperature - learning_rate * temp_grad, 0.1, 10.0)
                self.reaction_strength = np.clip(self.reaction_strength - learning_rate * temp_grad, 0.1, 2.0)
                
        training_info = {
            'num_epochs': num_epochs,
            'final_loss': losses[-1],
            'best_loss': best_loss,
            'loss_history': losses,
            'final_temperature': self.temperature,
            'final_reaction_strength': self.reaction_strength,
            'converged': len(losses) > 10 and np.std(losses[-10:]) < 1e-4
        }
        
        return training_info
        
    def get_sentiment_gradients(self, texts: List[str]) -> np.ndarray:
        """Compute sentiment gradients in embedding space.
        
        Parameters
        ----------
        texts : List[str]
            Input texts
            
        Returns
        -------
        gradients : np.ndarray
            Sentiment gradients for each text
        """
        if self.backend_name == 'jax' and hasattr(self, 'energy_gradient'):
            sentiments, _ = self.analyze_sentiment(texts)
            return np.array(self.energy_gradient(sentiments))
        else:
            # Finite difference approximation
            sentiments, _ = self.analyze_sentiment(texts)
            gradients = np.zeros_like(sentiments)
            
            eps = 1e-6
            for i in range(len(sentiments)):
                sentiment_plus = sentiments.copy()
                sentiment_plus[i] += eps
                energy_plus = self.compute_energy(sentiment_plus)
                
                sentiment_minus = sentiments.copy()
                sentiment_minus[i] -= eps
                energy_minus = self.compute_energy(sentiment_minus)
                
                gradients[i] = (energy_plus - energy_minus) / (2 * eps)
                
            return gradients


class TextEmbeddingSentimentProblem(SentimentProblem):
    """Sentiment problem with built-in text embedding generation.
    
    This class automatically generates embeddings from raw text using
    various embedding methods (TF-IDF, Word2Vec, BERT, etc.)
    """
    
    def __init__(
        self,
        texts: List[str],
        embedding_method: str = 'tfidf',
        embedding_dim: int = 300,
        **kwargs
    ):
        """Initialize with automatic text embedding generation.
        
        Parameters
        ----------
        texts : List[str]
            Input texts to analyze
        embedding_method : str, optional
            Embedding method ('tfidf', 'word2vec', 'bert'), by default 'tfidf'
        embedding_dim : int, optional
            Embedding dimension, by default 300
        """
        self.texts = texts
        self.embedding_method = embedding_method
        self.embedding_dim = embedding_dim
        
        # Generate embeddings
        text_embeddings = self._generate_embeddings(texts, embedding_method, embedding_dim)
        
        # Initialize parent
        super().__init__(text_embeddings=text_embeddings, **kwargs)
        
    def _generate_embeddings(
        self, 
        texts: List[str], 
        method: str, 
        dim: int
    ) -> np.ndarray:
        """Generate text embeddings using specified method."""
        
        if method == 'tfidf':
            return self._generate_tfidf_embeddings(texts, dim)
        elif method == 'word2vec':
            return self._generate_word2vec_embeddings(texts, dim)
        elif method == 'bert':
            return self._generate_bert_embeddings(texts)
        else:
            raise ValueError(f"Unknown embedding method: {method}")
            
    def _generate_tfidf_embeddings(self, texts: List[str], max_features: int) -> np.ndarray:
        """Generate TF-IDF embeddings."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )
            
            embeddings = vectorizer.fit_transform(texts).toarray()
            return embeddings.astype(np.float32)
            
        except ImportError:
            # Fallback: simple word frequency embeddings
            return self._generate_simple_embeddings(texts, max_features)
            
    def _generate_simple_embeddings(self, texts: List[str], dim: int) -> np.ndarray:
        """Generate simple frequency-based embeddings as fallback."""
        import re
        from collections import Counter
        
        # Tokenize all texts
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
            
        # Get most common words
        vocab = [word for word, _ in Counter(all_words).most_common(dim)]
        
        # Generate embeddings
        embeddings = np.zeros((len(texts), len(vocab)))
        for i, text in enumerate(texts):
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts = Counter(words)
            for j, word in enumerate(vocab):
                embeddings[i, j] = word_counts.get(word, 0)
                
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        return embeddings.astype(np.float32)
        
    def _generate_word2vec_embeddings(self, texts: List[str], dim: int) -> np.ndarray:
        """Generate Word2Vec embeddings (placeholder for now)."""
        # For now, return random embeddings with proper structure
        # In production, would use actual Word2Vec model
        np.random.seed(42)  # For reproducibility
        embeddings = np.random.normal(0, 1, (len(texts), dim))
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        return embeddings.astype(np.float32)
        
    def _generate_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings (placeholder for now)."""
        # For now, return structured embeddings
        # In production, would use actual BERT model
        dim = 768  # Standard BERT embedding dimension
        np.random.seed(42)
        embeddings = np.random.normal(0, 0.1, (len(texts), dim))
        
        return embeddings.astype(np.float32)