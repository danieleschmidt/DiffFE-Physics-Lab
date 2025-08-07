"""High-level sentiment analysis service using physics-informed methods."""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from ..models.sentiment_problem import SentimentProblem, TextEmbeddingSentimentProblem
from ..operators.sentiment import CompositeSentimentOperator
from ..performance.monitor import PerformanceMonitor
from ..performance.sentiment_cache import get_global_cache, SentimentAnalysisCache
from ..utils.validation import validate_text_input


@dataclass
class SentimentAnalysisResult:
    """Results from sentiment analysis."""
    
    # Core results
    sentiments: np.ndarray  # Final sentiment scores [-1, 1]
    confidence_scores: np.ndarray  # Confidence in predictions [0, 1]
    
    # Analysis metadata  
    processing_time: float
    num_texts: int
    embedding_method: str
    physics_parameters: Dict[str, float]
    
    # Detailed diagnostics
    convergence_info: Dict[str, Any]
    energy_evolution: List[float]
    sentiment_evolution: List[np.ndarray]
    
    # Performance metrics
    tokens_per_second: float
    memory_usage_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result_dict = asdict(self)
        
        # Convert numpy arrays to lists for JSON serialization
        result_dict['sentiments'] = self.sentiments.tolist()
        result_dict['confidence_scores'] = self.confidence_scores.tolist()
        result_dict['sentiment_evolution'] = [arr.tolist() for arr in self.sentiment_evolution]
        
        return result_dict
        
    def save(self, filepath: Union[str, Path]):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SentimentAnalysisResult':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Convert lists back to numpy arrays
        data['sentiments'] = np.array(data['sentiments'])
        data['confidence_scores'] = np.array(data['confidence_scores'])
        data['sentiment_evolution'] = [np.array(arr) for arr in data['sentiment_evolution']]
        
        return cls(**data)


class SentimentAnalyzer:
    """High-level sentiment analysis service.
    
    Provides a simple interface for physics-informed sentiment analysis
    with automatic parameter tuning, performance monitoring, and result caching.
    """
    
    def __init__(
        self,
        embedding_method: str = 'tfidf',
        embedding_dim: int = 300,
        backend: str = 'jax',
        cache_embeddings: bool = True,
        performance_monitoring: bool = True,
        cache_instance: Optional[SentimentAnalysisCache] = None
    ):
        """Initialize sentiment analyzer.
        
        Parameters
        ----------
        embedding_method : str, optional
            Text embedding method ('tfidf', 'word2vec', 'bert'), by default 'tfidf'
        embedding_dim : int, optional
            Embedding dimension, by default 300
        backend : str, optional
            AD backend ('jax', 'torch'), by default 'jax'
        cache_embeddings : bool, optional
            Cache embeddings for repeated analysis, by default True
        performance_monitoring : bool, optional
            Enable performance monitoring, by default True
        cache_instance : SentimentAnalysisCache, optional
            Custom cache instance, uses global cache if None
        """
        self.embedding_method = embedding_method
        self.embedding_dim = embedding_dim
        self.backend = backend
        self.cache_embeddings = cache_embeddings
        
        # Performance monitoring
        if performance_monitoring:
            self.monitor = PerformanceMonitor(
                monitor_memory=True,
                monitor_cpu=True,
                alert_memory_threshold_mb=1000
            )
        else:
            self.monitor = None
            
        # Advanced caching system
        if cache_embeddings:
            self.cache = cache_instance or get_global_cache()
        else:
            self.cache = None
        
        # Default physics parameters (auto-tuned)
        self.default_params = {
            'temperature': 1.0,
            'reaction_strength': 0.5,
            'diffusion_coeff': 1.0,
            'num_steps': 100,
            'dt': 0.01
        }
        
    def analyze(
        self,
        texts: List[str],
        physics_params: Optional[Dict[str, float]] = None,
        return_diagnostics: bool = False,
        auto_tune_params: bool = True
    ) -> Union[np.ndarray, SentimentAnalysisResult]:
        """Analyze sentiment of input texts.
        
        Parameters
        ----------
        texts : List[str]
            Input texts to analyze
        physics_params : Dict[str, float], optional
            Physics parameters for analysis
        return_diagnostics : bool, optional
            Return detailed diagnostics, by default False
        auto_tune_params : bool, optional
            Automatically tune physics parameters, by default True
            
        Returns
        -------
        Union[np.ndarray, SentimentAnalysisResult]
            Sentiment scores or detailed results
        """
        if self.monitor:
            self.monitor.start_measurement()
            
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(texts)
            
            # Use provided parameters or defaults
            params = physics_params or self.default_params.copy()
            
            # Auto-tune parameters if requested
            if auto_tune_params and len(texts) > 10:
                params = self._auto_tune_parameters(texts, params)
                
            # Create or retrieve problem from cache
            problem = self._get_problem(texts, params)
            
            # Analyze sentiment
            sentiments, analysis_info = problem.analyze_sentiment(
                texts,
                num_steps=int(params['num_steps']),
                dt=params['dt']
            )
            
            # Compute confidence scores
            confidence_scores = self._compute_confidence_scores(sentiments, analysis_info)
            
            processing_time = time.time() - start_time
            
            if return_diagnostics:
                # Create detailed result object
                result = SentimentAnalysisResult(
                    sentiments=sentiments,
                    confidence_scores=confidence_scores,
                    processing_time=processing_time,
                    num_texts=len(texts),
                    embedding_method=self.embedding_method,
                    physics_parameters=params,
                    convergence_info=analysis_info,
                    energy_evolution=analysis_info.get('energy_history', []),
                    sentiment_evolution=analysis_info.get('sentiment_evolution', []),
                    tokens_per_second=self._estimate_tokens_per_second(texts, processing_time),
                    memory_usage_mb=self.monitor.get_current_memory_usage() if self.monitor else 0.0
                )
                
                return result
            else:
                return sentiments
                
        finally:
            if self.monitor:
                self.monitor.stop_measurement()
                
    def analyze_batch(
        self,
        text_batches: List[List[str]],
        physics_params: Optional[Dict[str, float]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[np.ndarray]:
        """Analyze sentiment for multiple batches of texts.
        
        Parameters
        ----------
        text_batches : List[List[str]]
            Batches of texts to analyze
        physics_params : Dict[str, float], optional
            Physics parameters for analysis
        progress_callback : callable, optional
            Callback function for progress updates
            
        Returns
        -------
        List[np.ndarray]
            Sentiment scores for each batch
        """
        results = []
        total_batches = len(text_batches)
        
        for i, texts in enumerate(text_batches):
            sentiments = self.analyze(texts, physics_params, return_diagnostics=False)
            results.append(sentiments)
            
            if progress_callback:
                progress_callback(i + 1, total_batches)
                
        return results
        
    def train_on_labeled_data(
        self,
        texts: List[str],
        labels: np.ndarray,
        validation_split: float = 0.2,
        num_epochs: int = 50,
        learning_rate: float = 0.01
    ) -> Dict[str, Any]:
        """Train sentiment analyzer on labeled data.
        
        Parameters
        ----------
        texts : List[str]
            Training texts
        labels : np.ndarray
            True sentiment labels [-1, 1]
        validation_split : float, optional
            Fraction of data for validation, by default 0.2
        num_epochs : int, optional
            Number of training epochs, by default 50
        learning_rate : float, optional
            Learning rate, by default 0.01
            
        Returns
        -------
        Dict[str, Any]
            Training results and metrics
        """
        # Validate inputs
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
            
        # Split data
        n_val = int(len(texts) * validation_split)
        indices = np.random.permutation(len(texts))
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        train_texts = [texts[i] for i in train_indices]
        train_labels = labels[train_indices]
        val_texts = [texts[i] for i in val_indices]
        val_labels = labels[val_indices]
        
        # Create problem
        problem = self._get_problem(train_texts)
        
        # Train model
        training_info = problem.train_on_labeled_data(
            train_texts,
            train_labels,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        # Validate on holdout set
        if n_val > 0:
            val_problem = self._get_problem(val_texts)
            val_sentiments = val_problem.analyze_sentiment(val_texts)[0]
            val_mse = np.mean((val_sentiments - val_labels)**2)
            val_mae = np.mean(np.abs(val_sentiments - val_labels))
            val_accuracy = np.mean((val_sentiments > 0) == (val_labels > 0))
            
            training_info.update({
                'validation_mse': val_mse,
                'validation_mae': val_mae, 
                'validation_accuracy': val_accuracy
            })
            
        # Update default parameters with trained values
        self.default_params.update({
            'temperature': problem.temperature,
            'reaction_strength': problem.reaction_strength
        })
        
        return training_info
        
    def get_sentiment_explanations(
        self,
        texts: List[str],
        sentiments: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Get explanations for sentiment predictions.
        
        Parameters
        ----------
        texts : List[str]
            Input texts
        sentiments : np.ndarray, optional
            Precomputed sentiments (if None, will analyze)
            
        Returns
        -------
        List[Dict[str, Any]]
            Explanations for each text
        """
        if sentiments is None:
            sentiments = self.analyze(texts)
            
        problem = self._get_problem(texts)
        gradients = problem.get_sentiment_gradients(texts)
        
        explanations = []
        for i, (text, sentiment, gradient) in enumerate(zip(texts, sentiments, gradients)):
            # Analyze gradient magnitude and direction
            gradient_magnitude = np.linalg.norm(gradient)
            
            # Find most influential embedding dimensions
            top_dims = np.argsort(np.abs(gradient))[-5:]  # Top 5 dimensions
            
            explanation = {
                'text': text,
                'sentiment_score': float(sentiment),
                'sentiment_category': self._categorize_sentiment(sentiment),
                'confidence': float(self._compute_confidence_scores([sentiment], {})[0]),
                'gradient_magnitude': float(gradient_magnitude),
                'stability': 'high' if gradient_magnitude < 0.1 else 'medium' if gradient_magnitude < 0.5 else 'low',
                'influential_dimensions': top_dims.tolist(),
                'text_length': len(text),
                'processing_notes': self._generate_processing_notes(text, sentiment, gradient)
            }
            
            explanations.append(explanation)
            
        return explanations
        
    def benchmark_performance(
        self,
        test_sizes: List[int] = [10, 50, 100, 500, 1000],
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Benchmark performance across different input sizes.
        
        Parameters
        ----------
        test_sizes : List[int], optional
            Test sizes to benchmark
        num_runs : int, optional
            Number of runs per size for averaging
            
        Returns
        -------
        Dict[str, Any]
            Benchmark results
        """
        results = {
            'test_sizes': test_sizes,
            'processing_times': [],
            'tokens_per_second': [],
            'memory_usage': [],
            'accuracy_metrics': []
        }
        
        for size in test_sizes:
            # Generate test texts
            test_texts = [f"This is test text number {i} with some sentiment." for i in range(size)]
            
            size_results = {
                'times': [],
                'tokens_per_sec': [],
                'memory': []
            }
            
            for run in range(num_runs):
                start_time = time.time()
                
                if self.monitor:
                    self.monitor.start_measurement()
                    
                sentiments = self.analyze(test_texts, return_diagnostics=False)
                
                processing_time = time.time() - start_time
                tokens_per_sec = self._estimate_tokens_per_second(test_texts, processing_time)
                
                size_results['times'].append(processing_time)
                size_results['tokens_per_sec'].append(tokens_per_sec)
                
                if self.monitor:
                    memory_usage = self.monitor.get_current_memory_usage()
                    size_results['memory'].append(memory_usage)
                    self.monitor.stop_measurement()
                    
            # Average results for this size
            results['processing_times'].append(np.mean(size_results['times']))
            results['tokens_per_second'].append(np.mean(size_results['tokens_per_sec']))
            if size_results['memory']:
                results['memory_usage'].append(np.mean(size_results['memory']))
                
        return results
        
    def _validate_inputs(self, texts: List[str]):
        """Validate input texts."""
        if not texts:
            raise ValueError("Empty text list provided")
            
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All inputs must be strings")
            
        if any(len(text.strip()) == 0 for text in texts):
            raise ValueError("Empty texts are not allowed")
            
    def _get_problem(
        self,
        texts: List[str],
        params: Optional[Dict[str, float]] = None
    ) -> TextEmbeddingSentimentProblem:
        """Get or create sentiment problem for given texts."""
        problem_params = params or self.default_params
        
        # Try to get embeddings from cache first
        embeddings = None
        if self.cache is not None:
            embeddings = self.cache.get_embeddings(texts, self.embedding_method, self.embedding_dim)
        
        if embeddings is not None:
            # Create problem with cached embeddings
            problem = SentimentProblem(
                text_embeddings=embeddings,
                backend=self.backend,
                temperature=problem_params.get('temperature', 1.0),
                reaction_strength=problem_params.get('reaction_strength', 0.5)
            )
            # Store original texts for reference
            problem._original_texts = texts
        else:
            # Create new problem (will generate embeddings)
            problem = TextEmbeddingSentimentProblem(
                texts=texts,
                embedding_method=self.embedding_method,
                embedding_dim=self.embedding_dim,
                backend=self.backend,
                temperature=problem_params.get('temperature', 1.0),
                reaction_strength=problem_params.get('reaction_strength', 0.5)
            )
            
            # Cache embeddings if caching enabled
            if self.cache is not None:
                self.cache.put_embeddings(
                    texts, self.embedding_method, self.embedding_dim, 
                    problem.text_embeddings
                )
        
        return problem
        
    def _auto_tune_parameters(
        self,
        texts: List[str],
        base_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Automatically tune physics parameters based on text characteristics."""
        # Analyze text characteristics
        text_lengths = [len(text.split()) for text in texts]
        avg_length = np.mean(text_lengths)
        length_std = np.std(text_lengths)
        
        # Adjust parameters based on text characteristics
        tuned_params = base_params.copy()
        
        # Longer texts may need more diffusion
        if avg_length > 50:
            tuned_params['temperature'] *= 1.2
            tuned_params['num_steps'] = int(tuned_params['num_steps'] * 1.5)
            
        # High variance in length suggests need for more reaction
        if length_std > 20:
            tuned_params['reaction_strength'] *= 1.1
            
        # Adjust time step for stability
        if len(texts) > 100:
            tuned_params['dt'] *= 0.8  # Smaller time steps for larger problems
            
        return tuned_params
        
    def _compute_confidence_scores(
        self,
        sentiments: np.ndarray,
        analysis_info: Dict[str, Any]
    ) -> np.ndarray:
        """Compute confidence scores for sentiment predictions."""
        # Base confidence from sentiment magnitude
        base_confidence = np.abs(sentiments)
        
        # Adjust based on convergence
        if analysis_info.get('converged', False):
            convergence_bonus = 0.1
        else:
            convergence_bonus = -0.05
            
        # Adjust based on energy evolution (stability)
        energy_history = analysis_info.get('energy_history', [])
        if len(energy_history) > 1:
            energy_stability = 1.0 / (1.0 + abs(energy_history[-1] - energy_history[-2]))
        else:
            energy_stability = 0.5
            
        confidence = np.clip(base_confidence + convergence_bonus + 0.1 * energy_stability, 0, 1)
        
        return confidence
        
    def _estimate_tokens_per_second(self, texts: List[str], processing_time: float) -> float:
        """Estimate processing speed in tokens per second."""
        total_tokens = sum(len(text.split()) for text in texts)
        return total_tokens / max(processing_time, 1e-6)
        
    def _categorize_sentiment(self, sentiment_score: float) -> str:
        """Categorize sentiment score into readable label."""
        if sentiment_score > 0.5:
            return 'very positive'
        elif sentiment_score > 0.2:
            return 'positive'  
        elif sentiment_score > -0.2:
            return 'neutral'
        elif sentiment_score > -0.5:
            return 'negative'
        else:
            return 'very negative'
            
    def _generate_processing_notes(
        self,
        text: str,
        sentiment: float,
        gradient: np.ndarray
    ) -> List[str]:
        """Generate human-readable processing notes."""
        notes = []
        
        # Text length notes
        if len(text.split()) < 5:
            notes.append("Short text - may have limited context")
        elif len(text.split()) > 100:
            notes.append("Long text - averaged sentiment across content")
            
        # Sentiment strength notes
        if abs(sentiment) < 0.1:
            notes.append("Neutral sentiment - check for ambiguous language")
        elif abs(sentiment) > 0.9:
            notes.append("Strong sentiment - high confidence prediction")
            
        # Gradient notes
        if np.linalg.norm(gradient) > 0.5:
            notes.append("High gradient - sentiment sensitive to small changes")
        else:
            notes.append("Stable prediction - robust to small perturbations")
            
        return notes