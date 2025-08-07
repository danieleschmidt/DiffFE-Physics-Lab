"""REST API routes for sentiment analysis endpoints."""

from flask import Blueprint, request, jsonify, current_app
from functools import wraps
import time
import traceback
import numpy as np
from typing import Dict, Any, List, Optional

from ..services.sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult
from ..utils.sentiment_validation import (
    validate_text_input, 
    validate_physics_parameters,
    create_validation_report,
    SentimentValidationError
)
from .error_handlers import APIError, rate_limit
from ..security.validator import validate_request_size, sanitize_text


# Create blueprint for sentiment analysis routes
sentiment_bp = Blueprint('sentiment', __name__, url_prefix='/api/v1/sentiment')

# Global analyzer instance (consider using dependency injection in production)
_analyzer_cache = {}


def get_analyzer(
    embedding_method: str = 'tfidf',
    embedding_dim: int = 300,
    backend: str = 'jax'
) -> SentimentAnalyzer:
    """Get or create sentiment analyzer instance with caching."""
    cache_key = f"{embedding_method}_{embedding_dim}_{backend}"
    
    if cache_key not in _analyzer_cache:
        _analyzer_cache[cache_key] = SentimentAnalyzer(
            embedding_method=embedding_method,
            embedding_dim=embedding_dim,
            backend=backend,
            cache_embeddings=True,
            performance_monitoring=True
        )
        
    return _analyzer_cache[cache_key]


def require_json(f):
    """Decorator to ensure request has JSON content."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            raise APIError(
                message="Request must be JSON",
                status_code=400,
                error_code="INVALID_CONTENT_TYPE"
            )
        return f(*args, **kwargs)
    return decorated_function


def validate_request_data(f):
    """Decorator to validate common request data."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.get_json()
        
        # Validate request size
        validate_request_size(request)
        
        # Check for required fields
        if 'texts' not in data:
            raise APIError(
                message="Missing required field: texts",
                status_code=400,
                error_code="MISSING_FIELD"
            )
            
        texts = data['texts']
        
        # Validate texts format
        if not isinstance(texts, list):
            raise APIError(
                message="Field 'texts' must be a list",
                status_code=400,
                error_code="INVALID_FORMAT"
            )
            
        if len(texts) == 0:
            raise APIError(
                message="Field 'texts' cannot be empty",
                status_code=400,
                error_code="EMPTY_INPUT"
            )
            
        if len(texts) > 1000:  # Limit batch size
            raise APIError(
                message="Maximum 1000 texts per request",
                status_code=413,
                error_code="REQUEST_TOO_LARGE"
            )
            
        # Sanitize texts
        try:
            sanitized_texts = [sanitize_text(text) for text in texts]
            data['texts'] = sanitized_texts
        except Exception as e:
            raise APIError(
                message=f"Text sanitization failed: {str(e)}",
                status_code=400,
                error_code="SANITIZATION_ERROR"
            )
            
        return f(*args, **kwargs)
    return decorated_function


@sentiment_bp.route('/analyze', methods=['POST'])
@rate_limit(requests_per_minute=60)
@require_json
@validate_request_data
def analyze_sentiment():
    """
    Analyze sentiment of input texts using physics-informed methods.
    
    Request Body:
    {
        "texts": ["Text 1", "Text 2", ...],
        "options": {
            "embedding_method": "tfidf|word2vec|bert",
            "embedding_dim": 300,
            "backend": "jax|torch",
            "physics_params": {
                "temperature": 1.0,
                "reaction_strength": 0.5,
                "num_steps": 100,
                "dt": 0.01
            },
            "return_diagnostics": false,
            "auto_tune_params": true,
            "validate_inputs": true
        }
    }
    
    Response:
    {
        "success": true,
        "data": {
            "sentiments": [...],
            "confidence_scores": [...],
            "processing_time": 0.123,
            "diagnostics": {...}  // if requested
        }
    }
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        texts = data['texts']
        options = data.get('options', {})
        
        # Extract options with defaults
        embedding_method = options.get('embedding_method', 'tfidf')
        embedding_dim = options.get('embedding_dim', 300)
        backend = options.get('backend', 'jax')
        physics_params = options.get('physics_params', {})
        return_diagnostics = options.get('return_diagnostics', False)
        auto_tune_params = options.get('auto_tune_params', True)
        validate_inputs = options.get('validate_inputs', True)
        
        # Input validation if requested
        validation_report = None
        if validate_inputs:
            try:
                text_validation = validate_text_input(
                    texts,
                    min_length=1,
                    max_length=5000,
                    allow_empty=False,
                    check_encoding=True
                )
                
                if not text_validation.is_valid:
                    return jsonify({
                        "success": False,
                        "error": {
                            "message": "Input validation failed",
                            "code": "VALIDATION_ERROR",
                            "details": {
                                "errors": text_validation.errors,
                                "warnings": text_validation.warnings
                            }
                        }
                    }), 400
                    
                # Validate physics parameters
                if physics_params:
                    physics_validation = validate_physics_parameters(physics_params)
                    physics_params = physics_validation['corrected_params']
                    
            except SentimentValidationError as e:
                raise APIError(
                    message=f"Validation error: {str(e)}",
                    status_code=400,
                    error_code="VALIDATION_ERROR"
                )
        
        # Get analyzer
        try:
            analyzer = get_analyzer(embedding_method, embedding_dim, backend)
        except Exception as e:
            raise APIError(
                message=f"Failed to initialize analyzer: {str(e)}",
                status_code=500,
                error_code="ANALYZER_ERROR"
            )
        
        # Perform analysis
        try:
            if return_diagnostics:
                result: SentimentAnalysisResult = analyzer.analyze(
                    texts,
                    physics_params=physics_params if physics_params else None,
                    return_diagnostics=True,
                    auto_tune_params=auto_tune_params
                )
                
                response_data = {
                    "sentiments": result.sentiments.tolist(),
                    "confidence_scores": result.confidence_scores.tolist(),
                    "processing_time": result.processing_time,
                    "diagnostics": {
                        "num_texts": result.num_texts,
                        "embedding_method": result.embedding_method,
                        "physics_parameters": result.physics_parameters,
                        "convergence_info": result.convergence_info,
                        "tokens_per_second": result.tokens_per_second,
                        "memory_usage_mb": result.memory_usage_mb
                    }
                }
                
            else:
                sentiments = analyzer.analyze(
                    texts,
                    physics_params=physics_params if physics_params else None,
                    return_diagnostics=False,
                    auto_tune_params=auto_tune_params
                )
                
                response_data = {
                    "sentiments": sentiments.tolist(),
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            current_app.logger.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
            raise APIError(
                message=f"Sentiment analysis failed: {str(e)}",
                status_code=500,
                error_code="ANALYSIS_ERROR"
            )
        
        return jsonify({
            "success": True,
            "data": response_data
        })
        
    except APIError:
        raise
    except Exception as e:
        current_app.logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise APIError(
            message="Internal server error",
            status_code=500,
            error_code="INTERNAL_ERROR"
        )


@sentiment_bp.route('/analyze/batch', methods=['POST'])
@rate_limit(requests_per_minute=30)
@require_json
def analyze_batch():
    """
    Analyze sentiment for multiple batches of texts.
    
    Request Body:
    {
        "batches": [
            ["Text 1a", "Text 1b"],
            ["Text 2a", "Text 2b", "Text 2c"],
            ...
        ],
        "options": {
            "embedding_method": "tfidf",
            "physics_params": {...}
        }
    }
    """
    try:
        data = request.get_json()
        
        if 'batches' not in data:
            raise APIError(
                message="Missing required field: batches",
                status_code=400,
                error_code="MISSING_FIELD"
            )
            
        batches = data['batches']
        options = data.get('options', {})
        
        # Validate batches
        if not isinstance(batches, list) or len(batches) == 0:
            raise APIError(
                message="Field 'batches' must be a non-empty list",
                status_code=400,
                error_code="INVALID_FORMAT"
            )
            
        total_texts = sum(len(batch) for batch in batches)
        if total_texts > 2000:
            raise APIError(
                message="Maximum 2000 total texts across all batches",
                status_code=413,
                error_code="REQUEST_TOO_LARGE"
            )
        
        # Get analyzer
        analyzer = get_analyzer(
            options.get('embedding_method', 'tfidf'),
            options.get('embedding_dim', 300),
            options.get('backend', 'jax')
        )
        
        # Process batches
        results = analyzer.analyze_batch(
            batches,
            physics_params=options.get('physics_params')
        )
        
        # Convert to serializable format
        serialized_results = [batch_result.tolist() for batch_result in results]
        
        return jsonify({
            "success": True,
            "data": {
                "batch_results": serialized_results,
                "num_batches": len(batches),
                "total_texts": total_texts
            }
        })
        
    except APIError:
        raise
    except Exception as e:
        current_app.logger.error(f"Batch analysis failed: {str(e)}")
        raise APIError(
            message=f"Batch analysis failed: {str(e)}",
            status_code=500,
            error_code="BATCH_ERROR"
        )


@sentiment_bp.route('/train', methods=['POST'])
@rate_limit(requests_per_minute=10)
@require_json
def train_analyzer():
    """
    Train sentiment analyzer on labeled data.
    
    Request Body:
    {
        "texts": ["Text 1", "Text 2", ...],
        "labels": [-0.8, 0.6, 0.1, ...],
        "options": {
            "validation_split": 0.2,
            "num_epochs": 50,
            "learning_rate": 0.01,
            "embedding_method": "tfidf"
        }
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if 'texts' not in data or 'labels' not in data:
            raise APIError(
                message="Missing required fields: texts, labels",
                status_code=400,
                error_code="MISSING_FIELD"
            )
            
        texts = data['texts']
        labels = np.array(data['labels'])
        options = data.get('options', {})
        
        # Validate data consistency
        if len(texts) != len(labels):
            raise APIError(
                message="Number of texts and labels must match",
                status_code=400,
                error_code="DATA_MISMATCH"
            )
            
        if len(texts) < 10:
            raise APIError(
                message="Minimum 10 training samples required",
                status_code=400,
                error_code="INSUFFICIENT_DATA"
            )
        
        # Get analyzer
        analyzer = get_analyzer(
            options.get('embedding_method', 'tfidf'),
            options.get('embedding_dim', 300),
            options.get('backend', 'jax')
        )
        
        # Train model
        training_info = analyzer.train_on_labeled_data(
            texts=texts,
            labels=labels,
            validation_split=options.get('validation_split', 0.2),
            num_epochs=options.get('num_epochs', 50),
            learning_rate=options.get('learning_rate', 0.01)
        )
        
        return jsonify({
            "success": True,
            "data": {
                "training_info": training_info,
                "message": "Training completed successfully"
            }
        })
        
    except APIError:
        raise
    except Exception as e:
        current_app.logger.error(f"Training failed: {str(e)}")
        raise APIError(
            message=f"Training failed: {str(e)}",
            status_code=500,
            error_code="TRAINING_ERROR"
        )


@sentiment_bp.route('/explain', methods=['POST'])
@rate_limit(requests_per_minute=100)
@require_json
@validate_request_data
def explain_sentiment():
    """
    Get explanations for sentiment predictions.
    
    Request Body:
    {
        "texts": ["Text 1", "Text 2", ...],
        "options": {
            "embedding_method": "tfidf",
            "include_gradients": true
        }
    }
    """
    try:
        data = request.get_json()
        texts = data['texts']
        options = data.get('options', {})
        
        # Get analyzer
        analyzer = get_analyzer(
            options.get('embedding_method', 'tfidf'),
            options.get('embedding_dim', 300),
            options.get('backend', 'jax')
        )
        
        # Get explanations
        explanations = analyzer.get_sentiment_explanations(texts)
        
        return jsonify({
            "success": True,
            "data": {
                "explanations": explanations,
                "num_texts": len(texts)
            }
        })
        
    except APIError:
        raise
    except Exception as e:
        current_app.logger.error(f"Explanation failed: {str(e)}")
        raise APIError(
            message=f"Explanation failed: {str(e)}",
            status_code=500,
            error_code="EXPLANATION_ERROR"
        )


@sentiment_bp.route('/benchmark', methods=['POST'])
@rate_limit(requests_per_minute=5)
@require_json
def benchmark_performance():
    """
    Benchmark sentiment analysis performance.
    
    Request Body:
    {
        "test_sizes": [10, 50, 100, 500],
        "num_runs": 3,
        "embedding_method": "tfidf"
    }
    """
    try:
        data = request.get_json()
        
        test_sizes = data.get('test_sizes', [10, 50, 100, 500])
        num_runs = data.get('num_runs', 3)
        embedding_method = data.get('embedding_method', 'tfidf')
        
        # Validate parameters
        if max(test_sizes) > 1000:
            raise APIError(
                message="Maximum test size is 1000",
                status_code=400,
                error_code="SIZE_TOO_LARGE"
            )
            
        if num_runs > 5:
            raise APIError(
                message="Maximum 5 runs per benchmark",
                status_code=400,
                error_code="TOO_MANY_RUNS"
            )
        
        # Get analyzer
        analyzer = get_analyzer(embedding_method, 300, 'jax')
        
        # Run benchmark
        benchmark_results = analyzer.benchmark_performance(test_sizes, num_runs)
        
        return jsonify({
            "success": True,
            "data": {
                "benchmark_results": benchmark_results,
                "embedding_method": embedding_method
            }
        })
        
    except APIError:
        raise
    except Exception as e:
        current_app.logger.error(f"Benchmark failed: {str(e)}")
        raise APIError(
            message=f"Benchmark failed: {str(e)}",
            status_code=500,
            error_code="BENCHMARK_ERROR"
        )


@sentiment_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for sentiment analysis service."""
    try:
        # Quick test of analyzer functionality
        analyzer = get_analyzer('tfidf', 50, 'jax')  # Small embedding for speed
        test_result = analyzer.analyze(["Test text"], return_diagnostics=False)
        
        return jsonify({
            "success": True,
            "status": "healthy",
            "service": "sentiment_analysis",
            "analyzers_cached": len(_analyzer_cache),
            "test_result": float(test_result[0]) if len(test_result) > 0 else None
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "status": "unhealthy",
            "service": "sentiment_analysis",
            "error": str(e)
        }), 503


@sentiment_bp.route('/info', methods=['GET'])
def service_info():
    """Get information about the sentiment analysis service."""
    return jsonify({
        "success": True,
        "service": {
            "name": "Physics-Informed Sentiment Analysis",
            "version": "1.0.0",
            "description": "Sentiment analysis using differentiable physics operators",
            "supported_methods": {
                "embedding": ["tfidf", "word2vec", "bert"],
                "backends": ["jax", "torch"],
                "physics_operators": [
                    "diffusion", "reaction", "advection", 
                    "laplacian", "gradient"
                ]
            },
            "limits": {
                "max_texts_per_request": 1000,
                "max_text_length": 5000,
                "max_embedding_dim": 1000,
                "rate_limits": {
                    "analyze": "60/minute",
                    "batch": "30/minute", 
                    "train": "10/minute",
                    "explain": "100/minute",
                    "benchmark": "5/minute"
                }
            },
            "performance": {
                "typical_latency_ms": "10-100",
                "throughput_texts_per_second": "100-1000",
                "memory_usage_mb": "50-500"
            }
        }
    })