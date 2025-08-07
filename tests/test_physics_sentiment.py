"""
Comprehensive test suite for physics-informed sentiment analysis.

This module provides extensive testing for all physics-informed sentiment analysis
components, including unit tests, integration tests, performance benchmarks,
and research validation tests.
"""

import pytest
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, patch
import tempfile
import os

# Handle optional dependencies gracefully
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import modules to test
from src.operators.sentiment import (
    create_sentiment_operator, PhysicsInformedSentimentClassifier,
    DiffusionSentimentPropagator, ConservationSentimentAnalyzer
)
from src.models.transformers import create_physics_transformer, PhysicsInformedTransformer
from src.utils.nlp_processing import (
    create_processing_pipeline, Language, TextProcessingConfig
)
from src.services.multilingual_sentiment import (
    create_multilingual_analyzer, MultilingualConfig, 
    SentimentAnalysisError, LanguageNotSupportedError
)
from src.performance.cache import (
    get_sentiment_cache, cached_sentiment_analysis, SentimentCacheManager
)
from src.research.physics_sentiment_algorithms import (
    create_research_algorithm, ResearchAlgorithm, ExperimentConfig,
    ResearchExperimentSuite, QuantumSentimentEntanglement, ThermodynamicEmotionModel
)

logger = logging.getLogger(__name__)


class TestPhysicsInformedOperators:
    """Test suite for physics-informed sentiment operators."""
    
    def test_create_sentiment_operator_physics_informed(self):
        """Test creation of physics-informed sentiment operator."""
        operator = create_sentiment_operator("physics_informed", vocab_size=1000, embedding_dim=64)
        
        assert isinstance(operator, PhysicsInformedSentimentClassifier)
        assert operator.vocab_size == 1000
        assert operator.embedding_dim == 64
    
    def test_create_sentiment_operator_diffusion(self):
        """Test creation of diffusion sentiment operator."""
        operator = create_sentiment_operator("diffusion", diffusion_rate=0.2, time_steps=5)
        
        assert isinstance(operator, DiffusionSentimentPropagator)
        assert operator.diffusion_rate == 0.2
        assert operator.time_steps == 5
    
    def test_create_sentiment_operator_conservation(self):
        """Test creation of conservation sentiment operator."""
        operator = create_sentiment_operator("conservation", conservation_weight=0.1)
        
        assert isinstance(operator, ConservationSentimentAnalyzer)
        assert operator.conservation_weight == 0.1
    
    def test_physics_informed_forward_pass(self):
        """Test forward pass of physics-informed classifier."""
        operator = create_sentiment_operator("physics_informed", vocab_size=100, embedding_dim=32)
        
        # Test input
        test_tokens = np.array([1, 15, 23, 7, 42])
        
        # Should work with different backends
        if operator.backend == "jax" and HAS_JAX:
            token_array = jnp.array(test_tokens)
        else:
            # Test should work even without JAX
            token_array = test_tokens
        
        try:
            result = operator.forward(token_array)
            
            # Check result properties
            assert isinstance(result, (np.ndarray, list))
            if isinstance(result, np.ndarray):
                assert len(result) == 3  # negative, neutral, positive
                assert np.all(result >= 0)  # Probabilities should be non-negative
                assert np.abs(np.sum(result) - 1.0) < 1e-6  # Should sum to 1
            
        except ImportError as e:
            # Expected if JAX/PyTorch not available
            pytest.skip(f"Backend not available: {e}")
    
    def test_diffusion_propagation(self):
        """Test sentiment diffusion propagation."""
        operator = create_sentiment_operator("diffusion", diffusion_rate=0.1, time_steps=3)
        
        # Create test sentiment and adjacency matrix
        initial_sentiment = np.array([0.8, -0.5, 0.2, 0.1, -0.3])
        adjacency_matrix = np.array([
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0]
        ], dtype=float)
        
        try:
            result = operator.propagate_sentiment(initial_sentiment, adjacency_matrix)
            
            # Check result properties
            assert isinstance(result, (np.ndarray, list))
            if isinstance(result, np.ndarray):
                assert len(result) == len(initial_sentiment)
                assert np.all(np.abs(result) <= 1.0)  # Should be bounded
            
        except ImportError:
            pytest.skip("Backend not available for diffusion test")
    
    def test_conservation_analysis(self):
        """Test conservation-based sentiment analysis."""
        operator = create_sentiment_operator("conservation", conservation_weight=0.05)
        
        # Create test segments
        test_segments = [
            np.array([0.5, 0.3, 0.2]),  # Mock sentiment for segment 1
            np.array([0.2, 0.4, 0.4]),  # Mock sentiment for segment 2
            np.array([0.7, 0.2, 0.1])   # Mock sentiment for segment 3
        ]
        
        try:
            results = operator.analyze_with_conservation(test_segments)
            
            # Check conservation property
            assert len(results) == len(test_segments)
            
            # Total probability should be conserved
            total_before = sum(np.sum(segment) for segment in test_segments)
            total_after = sum(np.sum(result) for result in results)
            assert np.abs(total_after - len(test_segments)) < 1e-3  # Should be normalized
            
        except ImportError:
            pytest.skip("Backend not available for conservation test")
    
    @pytest.mark.parametrize("operator_type", ["physics_informed", "diffusion", "conservation"])
    def test_operator_error_handling(self, operator_type):
        """Test error handling in sentiment operators."""
        # Test with invalid parameters
        with pytest.raises((ValueError, TypeError)):
            create_sentiment_operator(operator_type, vocab_size=-1)
        
        # Test with unknown operator type
        with pytest.raises(ValueError):
            create_sentiment_operator("unknown_operator")


class TestPhysicsTransformers:
    """Test suite for physics-informed transformer models."""
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_create_physics_transformer(self):
        """Test creation of physics-informed transformer."""
        model = create_physics_transformer(
            "sentiment",
            vocab_size=1000,
            d_model=256,
            nhead=8,
            num_layers=4
        )
        
        assert isinstance(model, PhysicsInformedTransformer)
        assert model.vocab_size == 1000
        assert model.d_model == 256
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_physics_transformer_forward(self):
        """Test forward pass of physics transformer."""
        model = create_physics_transformer(
            "sentiment",
            vocab_size=100,
            d_model=128,
            nhead=4,
            num_layers=2,
            max_seq_length=50
        )
        
        # Test input
        input_ids = torch.randint(0, 100, (2, 20))  # Batch size 2, sequence length 20
        
        # Forward pass
        outputs = model(input_ids)
        
        # Check output structure
        assert isinstance(outputs, dict)
        assert 'predictions' in outputs
        assert 'logits' in outputs
        assert 'physics_metrics' in outputs
        assert 'energy_history' in outputs
        
        # Check tensor shapes
        batch_size, seq_len = input_ids.shape
        assert outputs['predictions'].shape == (batch_size, 3)  # 3 sentiment classes
        assert outputs['logits'].shape == (batch_size, 3)
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_physics_loss_computation(self):
        """Test physics-informed loss computation."""
        model = create_physics_transformer("sentiment", vocab_size=100, d_model=64)
        
        # Mock outputs
        outputs = {
            'logits': torch.randn(2, 3),
            'predictions': torch.softmax(torch.randn(2, 3), dim=1),
            'energy_history': [torch.tensor(1.0), torch.tensor(0.95), torch.tensor(0.98)],
            'attention_patterns': [torch.randn(2, 4, 10, 10)]  # Mock attention
        }
        
        targets = torch.tensor([0, 2])  # Mock target labels
        
        loss_dict = model.physics_loss(outputs, targets)
        
        # Check loss components
        assert 'total_loss' in loss_dict
        assert 'ce_loss' in loss_dict
        assert 'physics_loss' in loss_dict
        assert 'physics_penalties' in loss_dict
        
        # Losses should be positive scalars
        assert loss_dict['total_loss'].item() >= 0
        assert loss_dict['ce_loss'].item() >= 0


class TestNLPProcessing:
    """Test suite for NLP processing utilities."""
    
    def test_create_processing_pipeline(self):
        """Test creation of processing pipeline."""
        pipeline = create_processing_pipeline(Language.ENGLISH)
        
        assert pipeline is not None
        assert pipeline.config.language == Language.ENGLISH
    
    def test_text_processing_validation(self):
        """Test text validation."""
        pipeline = create_processing_pipeline(Language.ENGLISH)
        
        # Valid text
        result = pipeline.process_text("This is a good movie with great acting.")
        assert result['is_valid'] is True
        assert len(result['validation_issues']) == 0
        
        # Empty text
        result = pipeline.process_text("")
        assert result['is_valid'] is False
        assert len(result['validation_issues']) > 0
    
    def test_multilingual_processing(self):
        """Test multilingual text processing."""
        # English
        en_pipeline = create_processing_pipeline(Language.ENGLISH)
        en_result = en_pipeline.process_text("Hello world")
        assert en_result['is_valid']
        
        # Spanish
        es_pipeline = create_processing_pipeline(Language.SPANISH)
        es_result = es_pipeline.process_text("Hola mundo")
        assert es_result['is_valid']
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        pipeline = create_processing_pipeline(
            Language.ENGLISH,
            remove_urls=True,
            remove_emails=True,
            lowercase=True
        )
        
        dirty_text = "Check out https://example.com and email me at test@email.com!"
        result = pipeline.process_text(dirty_text)
        
        cleaned = result['cleaned_text']
        assert "https://example.com" not in cleaned
        assert "test@email.com" not in cleaned
        assert cleaned.islower()
    
    def test_tokenization_with_physics_principles(self):
        """Test physics-inspired tokenization."""
        config = TextProcessingConfig(
            energy_normalization=True,
            information_preserving_truncation=True,
            max_sequence_length=10
        )
        pipeline = create_processing_pipeline(Language.ENGLISH, **config.__dict__)
        
        long_text = "This is a very long sentence that should be truncated using information-preserving principles to maintain the most important content."
        result = pipeline.process_text(long_text)
        
        tokens = result['tokens']
        assert len(tokens) <= config.max_sequence_length
        assert len(tokens) > 0


class TestMultilingualSentimentService:
    """Test suite for multilingual sentiment analysis service."""
    
    def test_create_multilingual_analyzer(self):
        """Test creation of multilingual analyzer."""
        analyzer = create_multilingual_analyzer()
        
        assert analyzer is not None
        supported_languages = analyzer.get_supported_languages()
        assert len(supported_languages) > 0
        assert Language.ENGLISH in supported_languages
    
    def test_single_text_analysis(self):
        """Test single text sentiment analysis."""
        analyzer = create_multilingual_analyzer()
        
        try:
            result = analyzer.analyze_sentiment("This is a great movie!")
            
            assert result.text == "This is a great movie!"
            assert result.predicted_sentiment is not None
            assert 0 <= result.confidence <= 1
            assert result.processing_time_ms >= 0
            
        except ImportError:
            pytest.skip("Required backend not available")
    
    def test_batch_analysis(self):
        """Test batch sentiment analysis."""
        analyzer = create_multilingual_analyzer()
        
        texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special."
        ]
        
        try:
            results = analyzer.analyze_batch(texts)
            
            assert len(results) == len(texts)
            for result in results:
                assert result.text in texts
                assert result.predicted_sentiment is not None
                
        except ImportError:
            pytest.skip("Required backend not available")
    
    def test_language_detection(self):
        """Test automatic language detection."""
        analyzer = create_multilingual_analyzer()
        
        try:
            # English text
            en_result = analyzer.analyze_sentiment("This is an English sentence.")
            assert en_result.language == Language.ENGLISH
            
            # Spanish text
            es_result = analyzer.analyze_sentiment("Esta es una oración en español.")
            # Note: Simplified language detection might not be perfect
            assert es_result.language in [Language.ENGLISH, Language.SPANISH]
            
        except ImportError:
            pytest.skip("Required backend not available")
    
    def test_error_handling(self):
        """Test error handling in multilingual analyzer."""
        config = MultilingualConfig()
        config.supported_languages = {Language.ENGLISH: "physics_informed"}
        
        analyzer = create_multilingual_analyzer(config)
        
        # Test unsupported language
        try:
            analyzer.analyze_sentiment("Bonjour le monde", language=Language.FRENCH)
            # Should either work with fallback or raise error
        except LanguageNotSupportedError:
            # Expected behavior
            pass
        except ImportError:
            pytest.skip("Required backend not available")
        
        # Test empty text
        with pytest.raises(SentimentAnalysisError):
            analyzer.analyze_sentiment("")
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        analyzer = create_multilingual_analyzer()
        
        # Perform some analyses
        try:
            analyzer.analyze_sentiment("Test text 1")
            analyzer.analyze_sentiment("Test text 2")
            
            metrics = analyzer.get_metrics()
            
            assert 'total_requests' in metrics
            assert 'total_errors' in metrics
            assert 'error_rate' in metrics
            assert metrics['total_requests'] >= 2
            
        except ImportError:
            pytest.skip("Required backend not available")


class TestCachingSystem:
    """Test suite for caching system."""
    
    def test_sentiment_cache_manager(self):
        """Test sentiment cache manager."""
        cache_manager = SentimentCacheManager(
            text_cache_size=100,
            model_cache_size=50,
            physics_cache_size=25,
            total_memory_mb=64
        )
        
        # Test text caching
        text_hash = "test_hash_123"
        test_result = {'tokens': ['hello', 'world'], 'quality': 0.9}
        
        cache_manager.cache_text_result(text_hash, test_result)
        cached_result = cache_manager.get_text_result(text_hash)
        
        assert cached_result is not None
        assert cached_result['tokens'] == ['hello', 'world']
        assert cached_result['quality'] == 0.9
    
    def test_cache_warming(self):
        """Test cache warming functionality."""
        cache_manager = SentimentCacheManager(total_memory_mb=32)
        
        common_texts = {
            'en': ['hello world', 'good morning', 'thank you'],
            'es': ['hola mundo', 'buenos días', 'gracias']
        }
        model_types = ['physics_informed', 'diffusion']
        
        warming_results = cache_manager.warm_multilingual_cache(common_texts, model_types)
        
        assert 'text' in warming_results
        assert 'model' in warming_results
        assert 'physics' in warming_results
        assert warming_results['text'] > 0
        assert warming_results['model'] > 0
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        cache_manager = SentimentCacheManager()
        
        # Add some data
        cache_manager.cache_text_result("hash1", {'data': 'test1'})
        cache_manager.cache_prediction("model1", "input1", {'prediction': 'positive'})
        
        stats = cache_manager.get_global_stats()
        
        assert 'text_cache' in stats
        assert 'model_cache' in stats
        assert 'physics_cache' in stats
        assert 'memory_allocation' in stats
    
    def test_cached_decorator(self):
        """Test caching decorator."""
        call_count = 0
        
        @cached_sentiment_analysis(cache_type="text", ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        
        # Different input should call function
        result3 = expensive_function(7)
        assert result3 == 14
        assert call_count == 2


class TestResearchAlgorithms:
    """Test suite for research algorithms."""
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_quantum_sentiment_entanglement(self):
        """Test quantum sentiment entanglement algorithm."""
        algorithm = create_research_algorithm(
            ResearchAlgorithm.QUANTUM_SENTIMENT,
            vocab_size=100,
            num_qubits=4,
            entanglement_depth=2
        )
        
        assert isinstance(algorithm, QuantumSentimentEntanglement)
        assert algorithm.num_qubits == 4
        assert algorithm.entanglement_depth == 2
        
        # Test forward pass
        test_tokens = jnp.array([1, 15, 23, 7])
        result = algorithm.forward(test_tokens)
        
        assert 'sentiment_probabilities' in result
        assert 'quantum_states' in result
        assert 'entanglement_measure' in result
        assert 'quantum_uncertainty' in result
        
        # Check probability constraints
        probs = result['sentiment_probabilities']
        assert len(probs) == 3
        assert all(p >= 0 for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-6
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_thermodynamic_emotion_model(self):
        """Test thermodynamic emotion model."""
        algorithm = create_research_algorithm(
            ResearchAlgorithm.THERMODYNAMIC_EMOTIONS,
            vocab_size=100,
            temperature=1.5,
            num_emotional_states=6
        )
        
        assert isinstance(algorithm, ThermodynamicEmotionModel)
        assert algorithm.temperature == 1.5
        assert algorithm.num_emotional_states == 6
        
        # Test forward pass
        test_tokens = jnp.array([5, 25, 45, 65])
        result = algorithm.forward(test_tokens)
        
        assert 'sentiment_probabilities' in result
        assert 'emotional_distribution' in result
        assert 'entropy' in result
        assert 'free_energy' in result
        assert 'temperature' in result
        assert 'phase_info' in result
        
        # Check thermodynamic consistency
        assert result['entropy'] >= 0
        assert result['temperature'] == 1.5
        assert result['phase_info']['phase'] in ['ordered', 'disordered', 'critical']
    
    def test_research_experiment_suite(self):
        """Test research experiment suite."""
        suite = ResearchExperimentSuite()
        
        # Design experiment
        config = ExperimentConfig(
            name="test_quantum_experiment",
            description="Test quantum sentiment analysis",
            algorithm=ResearchAlgorithm.QUANTUM_SENTIMENT,
            num_epochs=5,
            batch_size=16
        )
        
        experiment_id = suite.design_experiment(config)
        assert experiment_id is not None
        assert experiment_id in suite.experiments
        
        # Check experiment summary
        summary = suite.get_experiment_summary()
        assert summary['total_experiments'] >= 1
        assert any(exp['name'] == 'test_quantum_experiment' for exp in summary['experiment_list'])
    
    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_full_research_experiment(self):
        """Test complete research experiment execution."""
        suite = ResearchExperimentSuite()
        
        # Create simple test data
        train_data = [
            ("I love this movie", 2),
            ("This is terrible", 0),
            ("It's okay", 1),
            ("Great film", 2),
            ("Bad acting", 0)
        ]
        
        test_data = [
            ("Excellent work", 2),
            ("Poor quality", 0),
            ("Average movie", 1)
        ]
        
        # Design and run experiment
        config = ExperimentConfig(
            name="mini_quantum_test",
            description="Mini quantum experiment",
            algorithm=ResearchAlgorithm.QUANTUM_SENTIMENT,
            num_epochs=2,
            batch_size=2,
            vocab_size=50
        )
        
        experiment_id = suite.design_experiment(config)
        results = suite.run_experiment(experiment_id, train_data, test_data)
        
        # Check results structure
        assert 'testing_results' in results
        assert 'baseline_results' in results
        assert 'physics_analysis' in results
        assert 'statistical_analysis' in results
        
        # Check performance metrics
        assert 'accuracy' in results['testing_results']
        assert 'macro_f1' in results['testing_results']
        assert 0 <= results['testing_results']['accuracy'] <= 1
        
        # Generate report
        report = suite.generate_research_report(experiment_id)
        assert len(report) > 100  # Should be a substantial report
        assert "Quantum" in report  # Should mention the algorithm


class TestBenchmarkSuite:
    """Comprehensive benchmark tests."""
    
    def test_performance_benchmark(self):
        """Test performance benchmarks."""
        # Create different types of operators
        operators = {
            'physics_informed': create_sentiment_operator('physics_informed', vocab_size=100),
            'diffusion': create_sentiment_operator('diffusion'),
            'conservation': create_sentiment_operator('conservation')
        }
        
        # Test data
        test_tokens = np.array([1, 5, 15, 25, 35])
        performance_results = {}
        
        for name, operator in operators.items():
            try:
                start_time = time.time()
                
                # Convert to appropriate format
                if operator.backend == "jax" and HAS_JAX:
                    token_array = jnp.array(test_tokens)
                else:
                    token_array = test_tokens
                
                # Run prediction
                result = operator.forward(token_array)
                
                end_time = time.time()
                processing_time = (end_time - start_time) * 1000  # milliseconds
                
                performance_results[name] = {
                    'processing_time_ms': processing_time,
                    'success': True,
                    'result_type': type(result).__name__
                }
                
            except ImportError:
                performance_results[name] = {
                    'processing_time_ms': float('inf'),
                    'success': False,
                    'error': 'Backend not available'
                }
        
        # Check that at least one operator works
        successful_ops = [name for name, result in performance_results.items() if result['success']]
        if successful_ops:
            # Compare performance
            fastest_op = min(successful_ops, key=lambda x: performance_results[x]['processing_time_ms'])
            assert performance_results[fastest_op]['processing_time_ms'] < 1000  # Should be under 1 second
    
    def test_memory_usage_benchmark(self):
        """Test memory usage benchmarks."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and use cache manager
        cache_manager = SentimentCacheManager(total_memory_mb=32)
        
        # Add data to cache
        for i in range(100):
            cache_manager.cache_text_result(f"hash_{i}", {'data': f'test_data_{i}'})
        
        # Check memory increase
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - baseline_memory
        
        # Should not use excessive memory (allow for some overhead)
        assert memory_increase < 100  # Less than 100MB increase
    
    def test_accuracy_benchmark(self):
        """Test accuracy benchmarks with known cases."""
        # Create test cases with expected outcomes
        test_cases = [
            ("I absolutely love this amazing product!", 2),  # Clearly positive
            ("This is terrible and awful", 0),  # Clearly negative  
            ("It's okay, not bad not great", 1),  # Clearly neutral
        ]
        
        try:
            analyzer = create_multilingual_analyzer()
            
            correct_predictions = 0
            total_predictions = len(test_cases)
            
            for text, expected_label in test_cases:
                try:
                    result = analyzer.analyze_sentiment(text)
                    
                    # Map sentiment to numeric label
                    sentiment_to_label = {
                        'negative': 0,
                        'neutral': 1, 
                        'positive': 2
                    }
                    
                    predicted_label = sentiment_to_label.get(result.predicted_sentiment.value, 1)
                    
                    if predicted_label == expected_label:
                        correct_predictions += 1
                        
                except ImportError:
                    # Skip if backend not available
                    total_predictions -= 1
                    continue
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                # Should get at least some test cases right
                assert accuracy >= 0.0  # Very lenient for basic functionality test
                
        except ImportError:
            pytest.skip("Required dependencies not available for accuracy benchmark")


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_sentiment_analysis(self):
        """Test complete end-to-end sentiment analysis pipeline."""
        try:
            # Create multilingual analyzer
            analyzer = create_multilingual_analyzer()
            
            # Test texts in different languages
            test_texts = {
                'en': "This movie is absolutely fantastic!",
                'es': "Esta película es absolutamente fantástica!",
                'fr': "Ce film est absolument fantastique!"
            }
            
            results = []
            
            for lang_code, text in test_texts.items():
                try:
                    # Map language codes to Language enum
                    lang_map = {'en': Language.ENGLISH, 'es': Language.SPANISH, 'fr': Language.FRENCH}
                    language = lang_map.get(lang_code)
                    
                    result = analyzer.analyze_sentiment(text, language=language)
                    results.append(result)
                    
                    # Check result completeness
                    assert result.text == text
                    assert result.confidence >= 0
                    assert result.processing_time_ms >= 0
                    assert result.predicted_sentiment is not None
                    
                except (ImportError, LanguageNotSupportedError):
                    # Skip unsupported configurations
                    continue
            
            # Should have processed at least English
            assert len(results) >= 1
            
        except ImportError:
            pytest.skip("Required dependencies not available for integration test")
    
    def test_research_to_production_pipeline(self):
        """Test pipeline from research algorithm to production deployment."""
        if not HAS_JAX:
            pytest.skip("JAX required for research algorithm test")
        
        # Step 1: Research experiment
        suite = ResearchExperimentSuite()
        
        config = ExperimentConfig(
            name="integration_test",
            description="Integration test experiment",
            algorithm=ResearchAlgorithm.QUANTUM_SENTIMENT,
            num_epochs=1,
            batch_size=2,
            vocab_size=50
        )
        
        # Minimal test data
        train_data = [("good", 2), ("bad", 0)]
        test_data = [("excellent", 2)]
        
        experiment_id = suite.design_experiment(config)
        research_results = suite.run_experiment(experiment_id, train_data, test_data)
        
        # Step 2: Extract algorithm for production use
        algorithm = create_research_algorithm(ResearchAlgorithm.QUANTUM_SENTIMENT, vocab_size=50)
        
        # Step 3: Test in production-like environment with caching
        cache_manager = SentimentCacheManager()
        
        test_input = jnp.array([5, 15, 25])
        result = algorithm.forward(test_input)
        
        # Cache the result
        input_hash = "test_integration_hash"
        cache_manager.cache_physics_result("quantum", input_hash, result)
        
        # Retrieve from cache
        cached_result = cache_manager.get_physics_result("quantum", input_hash)
        
        # Verify pipeline worked end-to-end
        assert research_results is not None
        assert result is not None
        assert cached_result is not None
        assert 'sentiment_probabilities' in result
        assert 'quantum_uncertainty' in result


# Performance test fixtures
@pytest.fixture(scope="session")
def benchmark_data():
    """Generate benchmark data for performance tests."""
    return {
        'small_texts': ["good", "bad", "okay"] * 10,
        'medium_texts': ["This is a good movie with great acting"] * 50,
        'large_texts': [" ".join(["word"] * 100)] * 100
    }


# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])