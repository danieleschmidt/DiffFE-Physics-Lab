"""Unit tests for sentiment analysis components."""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os

# Mock dependencies that might not be available in all environments
with patch.dict('sys.modules', {
    'jax': MagicMock(),
    'jax.numpy': MagicMock(),
    'torch': MagicMock(),
    'sklearn': MagicMock(),
    'sklearn.feature_extraction': MagicMock(),
    'sklearn.feature_extraction.text': MagicMock(),
}):
    from src.models.sentiment_problem import SentimentProblem, TextEmbeddingSentimentProblem
    from src.operators.sentiment import (
        SentimentDiffusionOperator, SentimentReactionOperator,
        SentimentLaplacianOperator, CompositeSentimentOperator
    )
    from src.services.sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult
    from src.utils.sentiment_validation import (
        validate_text_input, validate_sentiment_score, 
        validate_embeddings, SentimentValidationError
    )


class TestSentimentProblem(unittest.TestCase):
    """Test cases for SentimentProblem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.text_embeddings = np.random.randn(10, 50)
        self.sentiment_field = np.random.uniform(-0.5, 0.5, 10)
        
    def test_initialization(self):
        """Test SentimentProblem initialization."""
        problem = SentimentProblem(
            text_embeddings=self.text_embeddings,
            sentiment_field=self.sentiment_field,
            temperature=1.0,
            reaction_strength=0.5
        )
        
        self.assertEqual(problem.n_samples, 10)
        self.assertEqual(problem.embedding_dim, 50)
        self.assertEqual(problem.temperature, 1.0)
        self.assertEqual(problem.reaction_strength, 0.5)
        np.testing.assert_array_equal(problem.text_embeddings, self.text_embeddings)
        np.testing.assert_array_equal(problem.sentiment_field, self.sentiment_field)
        
    def test_invalid_initialization(self):
        """Test SentimentProblem with invalid inputs."""
        # Wrong shape for embeddings
        with self.assertRaises(ValueError):
            SentimentProblem(text_embeddings=np.random.randn(10))
            
        # Mismatched dimensions
        with self.assertRaises(ValueError):
            SentimentProblem(
                text_embeddings=self.text_embeddings,
                sentiment_field=np.random.randn(5)  # Wrong size
            )
            
        # Non-array input
        with self.assertRaises(ValueError):
            SentimentProblem(text_embeddings="not an array")
            
    def test_semantic_distance_computation(self):
        """Test semantic distance computation."""
        problem = SentimentProblem(
            text_embeddings=self.text_embeddings,
            temperature=1.0
        )
        
        # Check kernel properties
        self.assertEqual(problem.semantic_kernel.shape, (10, 10))
        
        # Kernel should be positive
        self.assertTrue(np.all(problem.semantic_kernel >= 0))
        
        # Diagonal should be maximum (self-similarity)
        diagonal = np.diag(problem.semantic_kernel)
        for i in range(10):
            self.assertTrue(np.all(problem.semantic_kernel[i, :] <= diagonal[i]))
            
        # Rows should sum to 1 (normalization)
        row_sums = np.sum(problem.semantic_kernel, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(10), decimal=10)
        
    def test_analyze_sentiment(self):
        """Test sentiment analysis functionality."""
        problem = SentimentProblem(
            text_embeddings=self.text_embeddings,
            temperature=1.0,
            reaction_strength=0.5
        )
        
        test_texts = [f"Text {i}" for i in range(10)]
        
        sentiments, info = problem.analyze_sentiment(
            test_texts,
            num_steps=50,
            dt=0.01
        )
        
        # Check output format
        self.assertEqual(len(sentiments), 10)
        self.assertTrue(np.all(np.abs(sentiments) <= 1))  # Valid range
        
        # Check info dictionary
        required_keys = ['num_steps', 'converged', 'mean_sentiment', 'sentiment_variance']
        for key in required_keys:
            self.assertIn(key, info)
            
        self.assertEqual(info['num_steps'], 50)
        self.assertIsInstance(info['converged'], bool)
        
    def test_backend_functions(self):
        """Test backend-specific function setup."""
        problem = SentimentProblem(
            text_embeddings=self.text_embeddings,
            backend='jax'
        )
        
        # Check that functions are callable
        self.assertTrue(callable(problem.diffusion_step))
        self.assertTrue(callable(problem.reaction_step))
        self.assertTrue(callable(problem.compute_energy))
        
        # Test function calls
        test_sentiment = np.random.uniform(-0.5, 0.5, 10)
        
        diffused = problem.diffusion_step(test_sentiment, dt=0.01)
        self.assertEqual(len(diffused), 10)
        
        reaction = problem.reaction_step(test_sentiment, dt=0.01)
        self.assertEqual(len(reaction), 10)
        
        energy = problem.compute_energy(test_sentiment)
        self.assertIsInstance(energy, (float, np.float64))


class TestTextEmbeddingSentimentProblem(unittest.TestCase):
    """Test cases for TextEmbeddingSentimentProblem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_texts = [
            "This is a positive text",
            "This is a negative text", 
            "This is neutral text",
            "Another positive example",
            "Another negative example"
        ]
        
    def test_initialization_with_tfidf(self):
        """Test initialization with TF-IDF embeddings."""
        with patch('src.models.sentiment_problem.TfidfVectorizer') as mock_vectorizer:
            mock_instance = Mock()
            mock_instance.fit_transform.return_value.toarray.return_value = np.random.randn(5, 100)
            mock_vectorizer.return_value = mock_instance
            
            problem = TextEmbeddingSentimentProblem(
                texts=self.test_texts,
                embedding_method='tfidf',
                embedding_dim=100
            )
            
            self.assertEqual(problem.n_samples, 5)
            self.assertEqual(problem.embedding_dim, 100)
            self.assertEqual(len(problem.texts), 5)
            
    def test_simple_embeddings_fallback(self):
        """Test fallback to simple embeddings when sklearn unavailable."""
        problem = TextEmbeddingSentimentProblem(
            texts=self.test_texts,
            embedding_method='tfidf',
            embedding_dim=50
        )
        
        # Should create embeddings without error
        self.assertEqual(problem.n_samples, 5)
        self.assertGreaterEqual(problem.embedding_dim, 1)
        
    def test_word2vec_placeholder(self):
        """Test Word2Vec embedding placeholder."""
        problem = TextEmbeddingSentimentProblem(
            texts=self.test_texts,
            embedding_method='word2vec',
            embedding_dim=100
        )
        
        self.assertEqual(problem.n_samples, 5)
        self.assertEqual(problem.embedding_dim, 100)
        
    def test_bert_placeholder(self):
        """Test BERT embedding placeholder."""
        problem = TextEmbeddingSentimentProblem(
            texts=self.test_texts,
            embedding_method='bert'
        )
        
        self.assertEqual(problem.n_samples, 5)
        self.assertEqual(problem.embedding_dim, 768)  # Standard BERT dimension


class TestSentimentOperators(unittest.TestCase):
    """Test cases for sentiment operators."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sentiment_field = np.random.uniform(-0.5, 0.5, 20)
        self.text_embeddings = np.random.randn(20, 10)
        self.semantic_distances = np.random.uniform(0, 2, (20, 20))
        
    def test_diffusion_operator(self):
        """Test SentimentDiffusionOperator."""
        op = SentimentDiffusionOperator(diffusivity=1.0, kernel_type='gaussian')
        
        self.assertTrue(op.is_linear)
        self.assertEqual(op.diffusivity, 1.0)
        self.assertEqual(op.kernel_type, 'gaussian')
        
        # Test apply method
        result = op.apply(self.sentiment_field, self.semantic_distances)
        self.assertEqual(len(result), 20)
        
        # Test Jacobian
        jacobian = op.jacobian(self.sentiment_field, self.semantic_distances)
        self.assertEqual(jacobian.shape, (20, 20))
        
    def test_reaction_operator(self):
        """Test SentimentReactionOperator."""
        op = SentimentReactionOperator(reaction_strength=0.5, bistable=True)
        
        self.assertFalse(op.is_linear)
        self.assertEqual(op.reaction_strength, 0.5)
        self.assertTrue(op.bistable)
        
        # Test apply method
        result = op.apply(self.sentiment_field)
        self.assertEqual(len(result), 20)
        
        # Test Jacobian
        jacobian = op.jacobian(self.sentiment_field)
        self.assertEqual(jacobian.shape, (20, 20))
        
        # Jacobian should be diagonal for reaction operator
        non_diagonal = jacobian - np.diag(np.diag(jacobian))
        np.testing.assert_array_almost_equal(non_diagonal, np.zeros((20, 20)))
        
    def test_laplacian_operator(self):
        """Test SentimentLaplacianOperator."""
        op = SentimentLaplacianOperator()
        
        self.assertTrue(op.is_linear)
        
        # Test apply method
        result = op.apply(self.sentiment_field, self.text_embeddings)
        self.assertEqual(len(result), 20)
        
        # Test Laplacian matrix computation
        laplacian_matrix = op.compute_laplacian_matrix(self.text_embeddings)
        self.assertEqual(laplacian_matrix.shape, (20, 20))
        
        # Laplacian matrix should have zero row sums (conservation)
        row_sums = np.sum(laplacian_matrix, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.zeros(20), decimal=10)
        
    def test_composite_operator(self):
        """Test CompositeSentimentOperator."""
        op = CompositeSentimentOperator(
            diffusion_coeff=1.0,
            reaction_strength=0.5,
            advection_strength=0.1
        )
        
        self.assertFalse(op.is_linear)  # Due to reaction term
        
        # Test apply method
        result = op.apply(self.sentiment_field, self.text_embeddings, self.semantic_distances)
        self.assertEqual(len(result), 20)
        
        # Test Jacobian
        jacobian = op.jacobian(self.sentiment_field, self.text_embeddings)
        self.assertEqual(jacobian.shape, (20, 20))


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special.",
            "Amazing quality and service!",
            "Disappointed with the results."
        ]
        
    def test_initialization(self):
        """Test SentimentAnalyzer initialization."""
        analyzer = SentimentAnalyzer(
            embedding_method='tfidf',
            embedding_dim=100,
            backend='jax',
            cache_embeddings=True,
            performance_monitoring=False
        )
        
        self.assertEqual(analyzer.embedding_method, 'tfidf')
        self.assertEqual(analyzer.embedding_dim, 100)
        self.assertEqual(analyzer.backend, 'jax')
        self.assertTrue(analyzer.cache_embeddings)
        
    def test_analyze_basic(self):
        """Test basic sentiment analysis."""
        analyzer = SentimentAnalyzer(performance_monitoring=False)
        
        sentiments = analyzer.analyze(
            self.test_texts,
            return_diagnostics=False,
            auto_tune_params=False
        )
        
        self.assertEqual(len(sentiments), 5)
        self.assertTrue(np.all(np.abs(sentiments) <= 1))
        
    def test_analyze_with_diagnostics(self):
        """Test sentiment analysis with detailed diagnostics."""
        analyzer = SentimentAnalyzer(performance_monitoring=False)
        
        result = analyzer.analyze(
            self.test_texts,
            return_diagnostics=True,
            auto_tune_params=False
        )
        
        self.assertIsInstance(result, SentimentAnalysisResult)
        self.assertEqual(len(result.sentiments), 5)
        self.assertEqual(len(result.confidence_scores), 5)
        self.assertGreater(result.processing_time, 0)
        self.assertEqual(result.num_texts, 5)
        
    def test_batch_processing(self):
        """Test batch processing."""
        analyzer = SentimentAnalyzer(performance_monitoring=False)
        
        batches = [
            self.test_texts[:2],
            self.test_texts[2:4],
            self.test_texts[4:]
        ]
        
        results = analyzer.analyze_batch(batches)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(len(results[1]), 2) 
        self.assertEqual(len(results[2]), 1)
        
    def test_explanations(self):
        """Test sentiment explanations."""
        analyzer = SentimentAnalyzer(performance_monitoring=False)
        
        explanations = analyzer.get_sentiment_explanations(self.test_texts[:3])
        
        self.assertEqual(len(explanations), 3)
        
        for explanation in explanations:
            required_keys = ['text', 'sentiment_score', 'sentiment_category', 'confidence']
            for key in required_keys:
                self.assertIn(key, explanation)


class TestSentimentValidation(unittest.TestCase):
    """Test cases for sentiment validation utilities."""
    
    def test_text_validation_valid(self):
        """Test text validation with valid inputs."""
        texts = ["This is a valid text", "Another valid text"]
        
        result = validate_text_input(texts)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.text_statistics['total_texts'], 2)
        
    def test_text_validation_invalid(self):
        """Test text validation with invalid inputs."""
        texts = ["", "Valid text", "A" * 15000]  # Empty, valid, too long
        
        result = validate_text_input(
            texts,
            min_length=1,
            max_length=10000,
            allow_empty=False
        )
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        self.assertEqual(result.text_statistics['empty_texts'], 1)
        self.assertEqual(result.text_statistics['very_long_texts'], 1)
        
    def test_sentiment_score_validation(self):
        """Test sentiment score validation."""
        # Valid single score
        self.assertTrue(validate_sentiment_score(0.5))
        self.assertTrue(validate_sentiment_score(-0.8))
        
        # Valid array
        scores = np.array([0.1, -0.3, 0.9, -1.0, 1.0])
        self.assertTrue(validate_sentiment_score(scores))
        
        # Invalid single score
        with self.assertRaises(SentimentValidationError):
            validate_sentiment_score(2.0)  # Out of range
            
        with self.assertRaises(SentimentValidationError):
            validate_sentiment_score(np.nan)  # NaN
            
        # Invalid array
        with self.assertRaises(SentimentValidationError):
            scores_invalid = np.array([0.1, 2.0, -0.3])  # Out of range
            validate_sentiment_score(scores_invalid)
            
    def test_embeddings_validation(self):
        """Test embeddings validation."""
        # Valid embeddings
        embeddings = np.random.randn(10, 50)
        result = validate_embeddings(embeddings)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['statistics']['shape'], (10, 50))
        
        # Invalid embeddings
        with self.assertRaises(SentimentValidationError):
            validate_embeddings(np.array([]))  # Empty array
            
        with self.assertRaises(SentimentValidationError):
            validate_embeddings(np.random.randn(10))  # Wrong dimensions


class TestSentimentAnalysisResult(unittest.TestCase):
    """Test cases for SentimentAnalysisResult dataclass."""
    
    def test_result_creation(self):
        """Test SentimentAnalysisResult creation."""
        result = SentimentAnalysisResult(
            sentiments=np.array([0.5, -0.3, 0.1]),
            confidence_scores=np.array([0.8, 0.7, 0.6]),
            processing_time=0.123,
            num_texts=3,
            embedding_method='tfidf',
            physics_parameters={'temperature': 1.0},
            convergence_info={'converged': True},
            energy_evolution=[1.0, 0.8, 0.6],
            sentiment_evolution=[np.array([0.1, 0.2, 0.3])],
            tokens_per_second=100.0,
            memory_usage_mb=50.0
        )
        
        self.assertEqual(len(result.sentiments), 3)
        self.assertEqual(result.num_texts, 3)
        self.assertEqual(result.embedding_method, 'tfidf')
        
    def test_result_serialization(self):
        """Test result serialization to dictionary."""
        result = SentimentAnalysisResult(
            sentiments=np.array([0.5, -0.3]),
            confidence_scores=np.array([0.8, 0.7]),
            processing_time=0.123,
            num_texts=2,
            embedding_method='tfidf',
            physics_parameters={},
            convergence_info={},
            energy_evolution=[],
            sentiment_evolution=[np.array([0.1, 0.2])],
            tokens_per_second=100.0,
            memory_usage_mb=50.0
        )
        
        result_dict = result.to_dict()
        
        # Check that numpy arrays are converted to lists
        self.assertIsInstance(result_dict['sentiments'], list)
        self.assertIsInstance(result_dict['confidence_scores'], list)
        self.assertIsInstance(result_dict['sentiment_evolution'][0], list)
        
    def test_result_save_load(self):
        """Test saving and loading results."""
        result = SentimentAnalysisResult(
            sentiments=np.array([0.5, -0.3]),
            confidence_scores=np.array([0.8, 0.7]),
            processing_time=0.123,
            num_texts=2,
            embedding_method='tfidf',
            physics_parameters={},
            convergence_info={},
            energy_evolution=[],
            sentiment_evolution=[np.array([0.1, 0.2])],
            tokens_per_second=100.0,
            memory_usage_mb=50.0
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            result.save(f.name)
            
            # Load and verify
            loaded_result = SentimentAnalysisResult.load(f.name)
            
            np.testing.assert_array_equal(result.sentiments, loaded_result.sentiments)
            np.testing.assert_array_equal(result.confidence_scores, loaded_result.confidence_scores)
            self.assertEqual(result.processing_time, loaded_result.processing_time)
            self.assertEqual(result.num_texts, loaded_result.num_texts)
            
        # Cleanup
        os.unlink(f.name)


if __name__ == '__main__':
    unittest.main()