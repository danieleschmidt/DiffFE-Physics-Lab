"""
Multilingual sentiment analysis service with robust error handling.

This service provides comprehensive multilingual sentiment analysis capabilities
with physics-informed models, advanced error handling, and production-ready features.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..utils.nlp_processing import (
    Language, TextProcessingConfig, TextProcessingPipeline,
    create_processing_pipeline
)
from ..operators.sentiment import create_sentiment_operator, SentimentOperator
from ..models.transformers import create_physics_transformer

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Sentiment classification labels."""
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SentimentResult:
    """Structured sentiment analysis result."""
    
    # Input information
    text: str
    language: Language
    text_length: int
    
    # Prediction results
    predicted_sentiment: SentimentLabel
    confidence: float
    confidence_level: ConfidenceLevel
    sentiment_scores: Dict[str, float]
    
    # Model information
    model_type: str
    backend: str
    processing_time_ms: float
    
    # Physics-informed metrics
    physics_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Quality indicators
    text_quality_score: float = 1.0
    prediction_reliability: float = 1.0
    
    # Metadata
    timestamp: str = field(default_factory=lambda: str(time.time()))
    model_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'text': self.text,
            'language': self.language.value,
            'text_length': self.text_length,
            'predicted_sentiment': self.predicted_sentiment.value,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'sentiment_scores': self.sentiment_scores,
            'model_type': self.model_type,
            'backend': self.backend,
            'processing_time_ms': self.processing_time_ms,
            'physics_metrics': self.physics_metrics,
            'text_quality_score': self.text_quality_score,
            'prediction_reliability': self.prediction_reliability,
            'timestamp': self.timestamp,
            'model_version': self.model_version
        }


@dataclass
class MultilingualConfig:
    """Configuration for multilingual sentiment analysis."""
    
    # Supported languages and their models
    supported_languages: Dict[Language, str] = field(default_factory=lambda: {
        Language.ENGLISH: "physics_informed",
        Language.SPANISH: "physics_informed",
        Language.FRENCH: "physics_informed",
        Language.GERMAN: "physics_informed",
        Language.ITALIAN: "conservation",
        Language.PORTUGUESE: "conservation",
        Language.RUSSIAN: "diffusion",
        Language.CHINESE: "diffusion",
        Language.JAPANESE: "diffusion",
        Language.KOREAN: "diffusion"
    })
    
    # Default model for unsupported languages
    default_model: str = "physics_informed"
    fallback_language: Language = Language.ENGLISH
    
    # Processing settings
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    retry_attempts: int = 3
    
    # Quality thresholds
    min_confidence_threshold: float = 0.3
    high_confidence_threshold: float = 0.8
    min_text_quality_score: float = 0.5
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    enable_batch_processing: bool = True
    max_batch_size: int = 100
    
    # Monitoring and logging
    enable_metrics: bool = True
    log_predictions: bool = False
    enable_audit_trail: bool = True


class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis errors."""
    
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR", 
                 details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class LanguageNotSupportedError(SentimentAnalysisError):
    """Raised when requested language is not supported."""
    
    def __init__(self, language: Language, supported_languages: List[Language]):
        super().__init__(
            f"Language {language.value} not supported. Supported: {[l.value for l in supported_languages]}",
            "LANGUAGE_NOT_SUPPORTED",
            {"requested_language": language.value, "supported_languages": [l.value for l in supported_languages]}
        )


class TextProcessingError(SentimentAnalysisError):
    """Raised when text processing fails."""
    
    def __init__(self, message: str, validation_issues: List[str]):
        super().__init__(
            f"Text processing failed: {message}",
            "TEXT_PROCESSING_ERROR", 
            {"validation_issues": validation_issues}
        )


class ModelLoadError(SentimentAnalysisError):
    """Raised when model loading fails."""
    
    def __init__(self, model_type: str, language: Language, original_error: str):
        super().__init__(
            f"Failed to load model {model_type} for language {language.value}: {original_error}",
            "MODEL_LOAD_ERROR",
            {"model_type": model_type, "language": language.value, "original_error": original_error}
        )


class PredictionError(SentimentAnalysisError):
    """Raised when prediction fails."""
    
    def __init__(self, message: str, model_info: Dict):
        super().__init__(
            f"Prediction failed: {message}",
            "PREDICTION_ERROR",
            {"model_info": model_info}
        )


class MultilingualSentimentAnalyzer:
    """
    Production-ready multilingual sentiment analyzer with physics-informed models.
    
    Features:
    - Multi-language support with automatic language detection
    - Robust error handling and fallback mechanisms
    - Physics-informed regularization for improved accuracy
    - Comprehensive logging and monitoring
    - Batch processing and caching capabilities
    - Quality assessment and reliability scoring
    """
    
    def __init__(self, config: MultilingualConfig):
        self.config = config
        
        # Initialize components
        self._models = {}  # Lazy loading
        self._processing_pipelines = {}  # Lazy loading
        self._cache = {} if config.enable_caching else None
        self._metrics = {'requests': 0, 'errors': 0, 'cache_hits': 0}
        
        # Thread pool for concurrent processing
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        
        logger.info(f"Initialized MultilingualSentimentAnalyzer with {len(config.supported_languages)} languages")
    
    def analyze_sentiment(self, 
                         text: str, 
                         language: Optional[Language] = None,
                         model_type: Optional[str] = None) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Parameters
        ----------
        text : str
            Text to analyze
        language : Optional[Language]
            Specific language (auto-detected if None)
        model_type : Optional[str]
            Specific model type to use
            
        Returns
        -------
        SentimentResult
            Comprehensive sentiment analysis result
            
        Raises
        ------
        SentimentAnalysisError
            If analysis fails
        """
        start_time = time.time()
        self._metrics['requests'] += 1
        
        try:
            # Input validation
            if not text or not text.strip():
                raise TextProcessingError("Empty or whitespace-only text", ["empty_text"])
            
            # Check cache
            cache_key = self._generate_cache_key(text, language, model_type)
            if self._cache and cache_key in self._cache:
                self._metrics['cache_hits'] += 1
                cached_result = self._cache[cache_key]
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached_result
            
            # Language detection if not specified
            if language is None:
                language = self._detect_language(text)
                logger.debug(f"Detected language: {language.value}")
            
            # Validate language support
            if language not in self.config.supported_languages:
                if self.config.fallback_language in self.config.supported_languages:
                    logger.warning(f"Language {language.value} not supported, using fallback {self.config.fallback_language.value}")
                    language = self.config.fallback_language
                else:
                    raise LanguageNotSupportedError(language, list(self.config.supported_languages.keys()))
            
            # Determine model type
            if model_type is None:
                model_type = self.config.supported_languages.get(language, self.config.default_model)
            
            # Process text
            processing_result = self._process_text(text, language)
            if not processing_result['is_valid'] and processing_result['validation_issues']:
                logger.warning(f"Text validation issues: {processing_result['validation_issues']}")
            
            # Get or create model
            model = self._get_model(language, model_type)
            
            # Make prediction
            prediction_result = self._predict_sentiment(
                processing_result, model, model_type
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = SentimentResult(
                text=text,
                language=language,
                text_length=len(text),
                predicted_sentiment=prediction_result['sentiment'],
                confidence=prediction_result['confidence'],
                confidence_level=self._determine_confidence_level(prediction_result['confidence']),
                sentiment_scores=prediction_result['scores'],
                model_type=model_type,
                backend=getattr(model, 'backend', 'unknown'),
                processing_time_ms=processing_time_ms,
                physics_metrics=prediction_result.get('physics_metrics', {}),
                text_quality_score=self._assess_text_quality(processing_result),
                prediction_reliability=self._assess_prediction_reliability(prediction_result)
            )
            
            # Cache result
            if self._cache:
                self._cache[cache_key] = result
            
            # Log if enabled
            if self.config.log_predictions:
                logger.info(f"Prediction: {result.predicted_sentiment.value} ({result.confidence:.3f}) for '{text[:50]}...'")
            
            return result
            
        except SentimentAnalysisError:
            self._metrics['errors'] += 1
            raise
        except Exception as e:
            self._metrics['errors'] += 1
            logger.error(f"Unexpected error in sentiment analysis: {str(e)}\n{traceback.format_exc()}")
            raise SentimentAnalysisError(
                f"Unexpected error: {str(e)}",
                "UNEXPECTED_ERROR",
                {"original_error": str(e), "traceback": traceback.format_exc()}
            )
    
    def analyze_batch(self, 
                     texts: List[str],
                     language: Optional[Language] = None,
                     model_type: Optional[str] = None,
                     max_workers: Optional[int] = None) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts concurrently.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to analyze
        language : Optional[Language]
            Language for all texts (auto-detected per text if None)
        model_type : Optional[str]
            Model type to use for all texts
        max_workers : Optional[int]
            Maximum number of concurrent workers
            
        Returns
        -------
        List[SentimentResult]
            List of sentiment analysis results
        """
        if not self.config.enable_batch_processing:
            # Process sequentially
            return [self.analyze_sentiment(text, language, model_type) for text in texts]
        
        # Limit batch size
        if len(texts) > self.config.max_batch_size:
            logger.warning(f"Batch size {len(texts)} exceeds maximum {self.config.max_batch_size}, processing in chunks")
            
            results = []
            for i in range(0, len(texts), self.config.max_batch_size):
                chunk = texts[i:i + self.config.max_batch_size]
                chunk_results = self.analyze_batch(chunk, language, model_type, max_workers)
                results.extend(chunk_results)
            return results
        
        # Process concurrently
        max_workers = max_workers or min(len(texts), self.config.max_concurrent_requests)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_text = {
                executor.submit(self.analyze_sentiment, text, language, model_type): text
                for text in texts
            }
            
            # Collect results in order
            results = [None] * len(texts)
            text_to_index = {text: i for i, text in enumerate(texts)}
            
            for future in as_completed(future_to_text, timeout=self.config.request_timeout_seconds):
                text = future_to_text[future]
                index = text_to_index[text]
                
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Batch processing failed for text '{text[:50]}...': {str(e)}")
                    # Create error result
                    results[index] = self._create_error_result(text, str(e))
        
        return results
    
    def get_supported_languages(self) -> List[Language]:
        """Get list of supported languages."""
        return list(self.config.supported_languages.keys())
    
    def get_model_info(self, language: Language) -> Dict[str, Any]:
        """Get information about the model for a specific language."""
        if language not in self.config.supported_languages:
            raise LanguageNotSupportedError(language, list(self.config.supported_languages.keys()))
        
        model_type = self.config.supported_languages[language]
        
        return {
            'language': language.value,
            'model_type': model_type,
            'is_loaded': self._is_model_loaded(language, model_type),
            'supports_physics_regularization': model_type in ['physics_informed', 'conservation', 'diffusion']
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        cache_hit_rate = self._metrics['cache_hits'] / max(self._metrics['requests'], 1)
        error_rate = self._metrics['errors'] / max(self._metrics['requests'], 1)
        
        return {
            'total_requests': self._metrics['requests'],
            'total_errors': self._metrics['errors'],
            'cache_hits': self._metrics['cache_hits'],
            'cache_hit_rate': cache_hit_rate,
            'error_rate': error_rate,
            'supported_languages': len(self.config.supported_languages),
            'loaded_models': len(self._models)
        }
    
    def clear_cache(self):
        """Clear the prediction cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def _detect_language(self, text: str) -> Language:
        """Detect the language of the text."""
        # Simplified language detection - in production use proper language detector
        text_lower = text.lower()
        
        # Language indicators
        indicators = {
            Language.SPANISH: ['el', 'la', 'y', 'de', 'que', 'es', 'en', 'un', 'una', 'con', 'por', 'para'],
            Language.FRENCH: ['le', 'de', 'et', 'à', 'un', 'une', 'ce', 'qui', 'que', 'dans', 'pour', 'avec'],
            Language.GERMAN: ['der', 'die', 'das', 'und', 'ein', 'eine', 'ist', 'sind', 'war', 'waren', 'mit', 'für'],
            Language.ITALIAN: ['il', 'lo', 'la', 'e', 'di', 'che', 'è', 'in', 'un', 'una', 'con', 'per'],
            Language.PORTUGUESE: ['o', 'a', 'e', 'de', 'que', 'é', 'em', 'um', 'uma', 'com', 'por', 'para'],
            Language.ENGLISH: ['the', 'and', 'a', 'an', 'is', 'are', 'was', 'were', 'have', 'has', 'with', 'for']
        }
        
        # Score each language
        scores = {}
        words = text_lower.split()[:50]  # Check first 50 words
        
        for lang, lang_indicators in indicators.items():
            score = sum(1 for word in words if word in lang_indicators)
            scores[lang] = score
        
        # Return language with highest score
        if scores:
            detected = max(scores, key=scores.get)
            if scores[detected] > 0:
                return detected
        
        return Language.ENGLISH  # Default fallback
    
    def _process_text(self, text: str, language: Language) -> Dict:
        """Process text using language-specific pipeline."""
        pipeline_key = language.value
        
        if pipeline_key not in self._processing_pipelines:
            try:
                self._processing_pipelines[pipeline_key] = create_processing_pipeline(language)
                logger.debug(f"Created processing pipeline for {language.value}")
            except Exception as e:
                logger.error(f"Failed to create processing pipeline for {language.value}: {str(e)}")
                raise TextProcessingError(f"Failed to create processing pipeline: {str(e)}", [])
        
        pipeline = self._processing_pipelines[pipeline_key]
        
        try:
            result = pipeline.process_text(text)
            return result
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            raise TextProcessingError(str(e), [str(e)])
    
    def _get_model(self, language: Language, model_type: str) -> SentimentOperator:
        """Get or create model for language and type."""
        model_key = f"{language.value}_{model_type}"
        
        if model_key not in self._models:
            try:
                self._models[model_key] = create_sentiment_operator(model_type)
                logger.debug(f"Loaded model {model_type} for {language.value}")
            except Exception as e:
                logger.error(f"Failed to load model {model_type} for {language.value}: {str(e)}")
                raise ModelLoadError(model_type, language, str(e))
        
        return self._models[model_key]
    
    def _predict_sentiment(self, 
                          processing_result: Dict,
                          model: SentimentOperator,
                          model_type: str) -> Dict:
        """Make sentiment prediction."""
        try:
            token_ids = processing_result['token_ids']
            if not token_ids:
                raise PredictionError("No tokens to process", {"model_type": model_type})
            
            # Convert to appropriate tensor format
            if model.backend == "jax":
                import jax.numpy as jnp
                token_array = jnp.array(token_ids[:100])  # Limit sequence length
            else:
                import torch
                token_array = torch.tensor(token_ids[:100], dtype=torch.long)
            
            # Get prediction
            prediction = model.forward(token_array)
            
            # Convert to standard format
            if model.backend == "jax":
                sentiment_scores = prediction.tolist()
            else:
                sentiment_scores = prediction.detach().numpy().tolist() if hasattr(prediction, 'detach') else prediction.tolist()
            
            # Ensure we have 3 scores (negative, neutral, positive)
            while len(sentiment_scores) < 3:
                sentiment_scores.append(0.0)
            sentiment_scores = sentiment_scores[:3]
            
            # Normalize scores
            score_sum = sum(sentiment_scores)
            if score_sum > 0:
                sentiment_scores = [score / score_sum for score in sentiment_scores]
            else:
                sentiment_scores = [1/3, 1/3, 1/3]  # Uniform distribution
            
            # Determine predicted sentiment
            max_score = max(sentiment_scores)
            predicted_idx = sentiment_scores.index(max_score)
            
            sentiment_labels = [SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL, SentimentLabel.POSITIVE]
            predicted_sentiment = sentiment_labels[predicted_idx]
            
            return {
                'sentiment': predicted_sentiment,
                'confidence': max_score,
                'scores': {
                    'negative': sentiment_scores[0],
                    'neutral': sentiment_scores[1],
                    'positive': sentiment_scores[2]
                },
                'physics_metrics': self._extract_physics_metrics(model, model_type)
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(str(e), {"model_type": model_type, "backend": getattr(model, 'backend', 'unknown')})
    
    def _extract_physics_metrics(self, model: SentimentOperator, model_type: str) -> Dict[str, float]:
        """Extract physics-specific metrics from model."""
        metrics = {}
        
        if model_type == "physics_informed":
            metrics['energy_conservation'] = 0.95  # Placeholder
            metrics['gradient_flow_smoothness'] = 0.87
        elif model_type == "diffusion":
            metrics['diffusion_stability'] = 0.92
        elif model_type == "conservation":
            metrics['semantic_conservation'] = 0.91
        
        return metrics
    
    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level based on score."""
        if confidence >= self.config.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= self.config.min_confidence_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _assess_text_quality(self, processing_result: Dict) -> float:
        """Assess the quality of the input text."""
        stats = processing_result.get('processing_stats', {})
        
        # Factors affecting quality
        compression_ratio = stats.get('compression_ratio', 1.0)
        vocab_coverage = stats.get('vocab_coverage', 1.0)
        avg_token_length = stats.get('avg_token_length', 5.0)
        
        # Normalize factors
        compression_score = min(compression_ratio * 1.5, 1.0)  # Penalize excessive compression
        vocab_score = vocab_coverage
        length_score = min(avg_token_length / 5.0, 1.0)  # Optimal around 5 chars per token
        
        # Weighted average
        quality_score = (compression_score * 0.3 + vocab_score * 0.5 + length_score * 0.2)
        
        return max(0.0, min(1.0, quality_score))
    
    def _assess_prediction_reliability(self, prediction_result: Dict) -> float:
        """Assess the reliability of the prediction."""
        confidence = prediction_result['confidence']
        
        # Base reliability on confidence
        reliability = confidence
        
        # Adjust based on score distribution
        scores = list(prediction_result['scores'].values())
        score_entropy = -sum(p * np.log(p + 1e-8) for p in scores if p > 0) if HAS_NUMPY else 0.5
        
        # Higher entropy (more uncertain) = lower reliability
        max_entropy = np.log(3) if HAS_NUMPY else 1.1  # log(3) for 3 classes
        entropy_penalty = score_entropy / max_entropy
        
        reliability = reliability * (1 - entropy_penalty * 0.2)
        
        return max(0.0, min(1.0, reliability))
    
    def _generate_cache_key(self, text: str, language: Optional[Language], model_type: Optional[str]) -> str:
        """Generate cache key for request."""
        import hashlib
        key_components = [
            text,
            language.value if language else "auto",
            model_type or "default"
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_model_loaded(self, language: Language, model_type: str) -> bool:
        """Check if model is loaded."""
        model_key = f"{language.value}_{model_type}"
        return model_key in self._models
    
    def _create_error_result(self, text: str, error_message: str) -> SentimentResult:
        """Create a result object for failed predictions."""
        return SentimentResult(
            text=text,
            language=Language.ENGLISH,  # Default
            text_length=len(text),
            predicted_sentiment=SentimentLabel.UNKNOWN,
            confidence=0.0,
            confidence_level=ConfidenceLevel.LOW,
            sentiment_scores={'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
            model_type="error",
            backend="none",
            processing_time_ms=0.0,
            physics_metrics={},
            text_quality_score=0.0,
            prediction_reliability=0.0
        )


# Factory function for creating service
def create_multilingual_analyzer(config: Optional[MultilingualConfig] = None) -> MultilingualSentimentAnalyzer:
    """Create a multilingual sentiment analyzer with default or custom configuration."""
    if config is None:
        config = MultilingualConfig()
    
    return MultilingualSentimentAnalyzer(config)