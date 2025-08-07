"""Validation utilities for sentiment analysis components."""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import warnings
from dataclasses import dataclass

from .validation import ValidationError


@dataclass
class TextValidationResult:
    """Results from text validation."""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    preprocessing_suggestions: List[str]
    text_statistics: Dict[str, Any]


class SentimentValidationError(ValidationError):
    """Custom exception for sentiment analysis validation errors."""
    pass


def validate_text_input(
    texts: Union[str, List[str]],
    min_length: int = 1,
    max_length: int = 10000,
    allow_empty: bool = False,
    check_encoding: bool = True,
    language_check: bool = False
) -> TextValidationResult:
    """Validate text input for sentiment analysis.
    
    Parameters
    ----------
    texts : Union[str, List[str]]
        Input text(s) to validate
    min_length : int, optional
        Minimum text length in characters, by default 1
    max_length : int, optional
        Maximum text length in characters, by default 10000
    allow_empty : bool, optional
        Whether to allow empty texts, by default False
    check_encoding : bool, optional
        Check for encoding issues, by default True
    language_check : bool, optional
        Perform basic language detection, by default False
        
    Returns
    -------
    TextValidationResult
        Validation results with warnings and suggestions
        
    Raises
    ------
    SentimentValidationError
        If critical validation errors are found
    """
    # Convert single string to list
    if isinstance(texts, str):
        texts = [texts]
        
    if not isinstance(texts, list):
        raise SentimentValidationError("Input must be string or list of strings")
        
    warnings_list = []
    errors = []
    suggestions = []
    statistics = {
        'total_texts': len(texts),
        'total_characters': 0,
        'total_words': 0,
        'avg_text_length': 0,
        'empty_texts': 0,
        'very_short_texts': 0,
        'very_long_texts': 0,
        'special_char_ratio': 0,
        'numeric_ratio': 0
    }
    
    # Validate each text
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            errors.append(f"Text {i}: Must be string, got {type(text)}")
            continue
            
        # Length checks
        text_len = len(text)
        statistics['total_characters'] += text_len
        
        if text_len == 0:
            statistics['empty_texts'] += 1
            if not allow_empty:
                errors.append(f"Text {i}: Empty text not allowed")
            else:
                warnings_list.append(f"Text {i}: Empty text may produce unreliable results")
                
        elif text_len < min_length:
            statistics['very_short_texts'] += 1
            warnings_list.append(f"Text {i}: Very short ({text_len} chars), may lack context")
            
        elif text_len > max_length:
            statistics['very_long_texts'] += 1
            errors.append(f"Text {i}: Too long ({text_len} chars > {max_length})")
            
        # Skip further validation for empty texts
        if text_len == 0:
            continue
            
        # Word count
        words = text.split()
        statistics['total_words'] += len(words)
        
        # Check for suspicious patterns
        if len(words) == 1 and text_len > 50:
            warnings_list.append(f"Text {i}: Single very long word - may be corrupted data")
            
        # Encoding validation
        if check_encoding:
            try:
                text.encode('utf-8').decode('utf-8')
            except UnicodeError:
                errors.append(f"Text {i}: Contains invalid UTF-8 characters")
                
        # Character composition analysis
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        numeric_chars = len(re.findall(r'\d', text))
        
        if text_len > 0:
            special_ratio = special_chars / text_len
            numeric_ratio = numeric_chars / text_len
            
            statistics['special_char_ratio'] += special_ratio
            statistics['numeric_ratio'] += numeric_ratio
            
            if special_ratio > 0.5:
                warnings_list.append(f"Text {i}: High special character ratio ({special_ratio:.1%})")
                suggestions.append(f"Text {i}: Consider text preprocessing to handle special characters")
                
            if numeric_ratio > 0.7:
                warnings_list.append(f"Text {i}: Mostly numeric content ({numeric_ratio:.1%})")
                suggestions.append(f"Text {i}: Sentiment analysis may not be meaningful for numeric data")
                
        # Language detection (basic)
        if language_check and text_len > 10:
            if _detect_non_english(text):
                warnings_list.append(f"Text {i}: May contain non-English text")
                suggestions.append(f"Text {i}: Consider language-specific preprocessing")
                
        # HTML/XML detection
        if re.search(r'<[^>]+>', text):
            warnings_list.append(f"Text {i}: Contains HTML/XML tags")
            suggestions.append(f"Text {i}: Consider removing HTML tags before analysis")
            
        # URL detection
        if re.search(r'http[s]?://\S+', text):
            warnings_list.append(f"Text {i}: Contains URLs")
            suggestions.append(f"Text {i}: Consider URL handling in preprocessing")
            
        # Email detection
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            warnings_list.append(f"Text {i}: Contains email addresses")
            suggestions.append(f"Text {i}: Consider masking email addresses")
    
    # Compute final statistics
    if statistics['total_texts'] > 0:
        statistics['avg_text_length'] = statistics['total_characters'] / statistics['total_texts']
        statistics['special_char_ratio'] /= statistics['total_texts']
        statistics['numeric_ratio'] /= statistics['total_texts']
        
    # Overall validation result
    is_valid = len(errors) == 0
    
    # Generate additional suggestions based on statistics
    if statistics['very_short_texts'] > statistics['total_texts'] * 0.5:
        suggestions.append("Many texts are very short - consider combining related texts")
        
    if statistics['empty_texts'] > 0:
        suggestions.append(f"Found {statistics['empty_texts']} empty texts - filter before analysis")
        
    return TextValidationResult(
        is_valid=is_valid,
        warnings=warnings_list,
        errors=errors,
        preprocessing_suggestions=suggestions,
        text_statistics=statistics
    )


def validate_sentiment_score(
    score: Union[float, np.ndarray],
    min_val: float = -1.0,
    max_val: float = 1.0,
    allow_nan: bool = False
) -> bool:
    """Validate sentiment score(s).
    
    Parameters
    ----------
    score : Union[float, np.ndarray]
        Sentiment score(s) to validate
    min_val : float, optional
        Minimum valid value, by default -1.0
    max_val : float, optional
        Maximum valid value, by default 1.0
    allow_nan : bool, optional
        Whether to allow NaN values, by default False
        
    Returns
    -------
    bool
        True if valid, False otherwise
        
    Raises
    ------
    SentimentValidationError
        If critical validation errors are found
    """
    if isinstance(score, np.ndarray):
        # Validate array of scores
        if score.size == 0:
            raise SentimentValidationError("Empty score array")
            
        if not allow_nan and np.any(np.isnan(score)):
            raise SentimentValidationError("NaN values found in sentiment scores")
            
        if np.any(np.isinf(score)):
            raise SentimentValidationError("Infinite values found in sentiment scores")
            
        if np.any(score < min_val) or np.any(score > max_val):
            invalid_indices = np.where((score < min_val) | (score > max_val))[0]
            raise SentimentValidationError(
                f"Sentiment scores out of range [{min_val}, {max_val}] at indices: {invalid_indices}"
            )
            
    else:
        # Validate single score
        if not isinstance(score, (int, float)):
            raise SentimentValidationError(f"Sentiment score must be numeric, got {type(score)}")
            
        if not allow_nan and np.isnan(score):
            raise SentimentValidationError("NaN sentiment score")
            
        if np.isinf(score):
            raise SentimentValidationError("Infinite sentiment score")
            
        if score < min_val or score > max_val:
            raise SentimentValidationError(
                f"Sentiment score {score} out of range [{min_val}, {max_val}]"
            )
            
    return True


def validate_embeddings(
    embeddings: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    check_normalization: bool = True,
    check_finite: bool = True
) -> Dict[str, Any]:
    """Validate text embeddings for sentiment analysis.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Text embeddings to validate
    expected_shape : Tuple[int, ...], optional
        Expected shape of embeddings
    check_normalization : bool, optional
        Check if embeddings are normalized, by default True
    check_finite : bool, optional
        Check for finite values, by default True
        
    Returns
    -------
    Dict[str, Any]
        Validation results with statistics and warnings
        
    Raises
    ------
    SentimentValidationError
        If critical validation errors are found
    """
    if not isinstance(embeddings, np.ndarray):
        raise SentimentValidationError("Embeddings must be numpy array")
        
    if embeddings.size == 0:
        raise SentimentValidationError("Empty embeddings array")
        
    if embeddings.ndim != 2:
        raise SentimentValidationError(f"Embeddings must be 2D array, got {embeddings.ndim}D")
        
    n_samples, embedding_dim = embeddings.shape
    
    # Shape validation
    if expected_shape is not None:
        if embeddings.shape != expected_shape:
            raise SentimentValidationError(
                f"Shape mismatch: expected {expected_shape}, got {embeddings.shape}"
            )
            
    # Finite value check
    if check_finite:
        if not np.all(np.isfinite(embeddings)):
            nan_count = np.sum(np.isnan(embeddings))
            inf_count = np.sum(np.isinf(embeddings))
            raise SentimentValidationError(
                f"Non-finite values found: {nan_count} NaN, {inf_count} infinite"
            )
            
    # Statistics
    stats = {
        'shape': embeddings.shape,
        'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
        'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
        'min_value': np.min(embeddings),
        'max_value': np.max(embeddings),
        'mean_value': np.mean(embeddings),
        'std_value': np.std(embeddings),
        'zero_embeddings': np.sum(np.all(embeddings == 0, axis=1)),
        'duplicate_embeddings': 0
    }
    
    warnings_list = []
    
    # Normalization check
    if check_normalization:
        norms = np.linalg.norm(embeddings, axis=1)
        norm_variance = np.var(norms)
        
        if norm_variance > 0.1:  # High variance in norms
            warnings_list.append(f"High variance in embedding norms: {norm_variance:.3f}")
            
        mean_norm = np.mean(norms)
        if abs(mean_norm - 1.0) > 0.1:  # Not unit normalized
            warnings_list.append(f"Embeddings not unit normalized: mean norm = {mean_norm:.3f}")
            
    # Zero embedding check
    if stats['zero_embeddings'] > 0:
        warnings_list.append(f"Found {stats['zero_embeddings']} zero embeddings")
        
    # Duplicate detection (expensive, so sample-based)
    if n_samples <= 1000:  # Only for smaller arrays
        unique_embeddings = np.unique(embeddings, axis=0)
        stats['duplicate_embeddings'] = n_samples - len(unique_embeddings)
        if stats['duplicate_embeddings'] > 0:
            warnings_list.append(f"Found {stats['duplicate_embeddings']} duplicate embeddings")
            
    # Range checks
    if stats['max_value'] > 10 or stats['min_value'] < -10:
        warnings_list.append("Embeddings have unusual value range")
        
    return {
        'is_valid': True,
        'warnings': warnings_list,
        'statistics': stats
    }


def validate_physics_parameters(
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate physics parameters for sentiment analysis.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Physics parameters to validate
        
    Returns
    -------
    Dict[str, Any]
        Validation results with corrected parameters if needed
        
    Raises
    ------
    SentimentValidationError
        If critical validation errors are found
    """
    if not isinstance(params, dict):
        raise SentimentValidationError("Physics parameters must be dictionary")
        
    # Expected parameter ranges
    param_ranges = {
        'temperature': (0.01, 100.0),
        'reaction_strength': (0.0, 10.0),
        'diffusion_coeff': (0.0, 100.0),
        'num_steps': (1, 10000),
        'dt': (1e-6, 1.0),
        'advection_strength': (0.0, 10.0)
    }
    
    warnings_list = []
    corrected_params = params.copy()
    
    for param_name, (min_val, max_val) in param_ranges.items():
        if param_name in params:
            value = params[param_name]
            
            # Type check
            if not isinstance(value, (int, float)):
                raise SentimentValidationError(
                    f"Parameter '{param_name}' must be numeric, got {type(value)}"
                )
                
            # Finite check
            if not np.isfinite(value):
                raise SentimentValidationError(
                    f"Parameter '{param_name}' must be finite, got {value}"
                )
                
            # Range check with auto-correction
            if value < min_val:
                warnings_list.append(
                    f"Parameter '{param_name}' = {value} below minimum {min_val}, "
                    f"corrected to {min_val}"
                )
                corrected_params[param_name] = min_val
                
            elif value > max_val:
                warnings_list.append(
                    f"Parameter '{param_name}' = {value} above maximum {max_val}, "
                    f"corrected to {max_val}"
                )
                corrected_params[param_name] = max_val
                
    # Physics consistency checks
    if 'dt' in params and 'num_steps' in params:
        total_time = params['dt'] * params['num_steps']
        if total_time > 100:
            warnings_list.append(
                f"Very long integration time: {total_time:.1f} time units"
            )
        elif total_time < 0.01:
            warnings_list.append(
                f"Very short integration time: {total_time:.4f} time units"
            )
            
    # Stability analysis
    if 'dt' in params and 'diffusion_coeff' in params:
        # CFL-like condition for diffusion
        max_stable_dt = 0.25 / max(params['diffusion_coeff'], 1e-10)
        if params['dt'] > max_stable_dt:
            warnings_list.append(
                f"Time step {params['dt']} may be too large for stability "
                f"(recommended max: {max_stable_dt:.4f})"
            )
            
    return {
        'is_valid': True,
        'warnings': warnings_list,
        'original_params': params,
        'corrected_params': corrected_params
    }


def _detect_non_english(text: str) -> bool:
    """Basic non-English text detection."""
    # Count ASCII vs non-ASCII characters
    ascii_chars = sum(ord(char) < 128 for char in text)
    total_chars = len(text)
    
    if total_chars == 0:
        return False
        
    ascii_ratio = ascii_chars / total_chars
    
    # If less than 80% ASCII, likely non-English
    return ascii_ratio < 0.8


def create_validation_report(
    text_validation: TextValidationResult,
    embedding_validation: Dict[str, Any],
    physics_validation: Dict[str, Any]
) -> str:
    """Create comprehensive validation report.
    
    Parameters
    ----------
    text_validation : TextValidationResult
        Text validation results
    embedding_validation : Dict[str, Any]
        Embedding validation results
    physics_validation : Dict[str, Any]
        Physics parameter validation results
        
    Returns
    -------
    str
        Formatted validation report
    """
    report = []
    report.append("SENTIMENT ANALYSIS VALIDATION REPORT")
    report.append("=" * 45)
    
    # Text validation section
    report.append("\n📝 TEXT VALIDATION")
    report.append("-" * 20)
    
    if text_validation.is_valid:
        report.append("✅ Status: PASSED")
    else:
        report.append("❌ Status: FAILED")
        
    if text_validation.errors:
        report.append(f"\n❌ Errors ({len(text_validation.errors)}):")
        for error in text_validation.errors[:5]:  # Show max 5 errors
            report.append(f"  • {error}")
        if len(text_validation.errors) > 5:
            report.append(f"  • ... and {len(text_validation.errors) - 5} more")
            
    if text_validation.warnings:
        report.append(f"\n⚠️  Warnings ({len(text_validation.warnings)}):")
        for warning in text_validation.warnings[:3]:  # Show max 3 warnings
            report.append(f"  • {warning}")
        if len(text_validation.warnings) > 3:
            report.append(f"  • ... and {len(text_validation.warnings) - 3} more")
            
    # Text statistics
    stats = text_validation.text_statistics
    report.append(f"\n📊 Text Statistics:")
    report.append(f"  • Total texts: {stats['total_texts']}")
    report.append(f"  • Average length: {stats['avg_text_length']:.1f} characters")
    report.append(f"  • Total words: {stats['total_words']}")
    report.append(f"  • Empty texts: {stats['empty_texts']}")
    report.append(f"  • Very short texts: {stats['very_short_texts']}")
    report.append(f"  • Very long texts: {stats['very_long_texts']}")
    
    # Embedding validation section
    report.append("\n🔢 EMBEDDING VALIDATION")
    report.append("-" * 25)
    
    if embedding_validation['is_valid']:
        report.append("✅ Status: PASSED")
    else:
        report.append("❌ Status: FAILED")
        
    if embedding_validation['warnings']:
        report.append(f"\n⚠️  Warnings ({len(embedding_validation['warnings'])}):")
        for warning in embedding_validation['warnings']:
            report.append(f"  • {warning}")
            
    embed_stats = embedding_validation['statistics']
    report.append(f"\n📊 Embedding Statistics:")
    report.append(f"  • Shape: {embed_stats['shape']}")
    report.append(f"  • Mean norm: {embed_stats['mean_norm']:.3f}")
    report.append(f"  • Value range: [{embed_stats['min_value']:.3f}, {embed_stats['max_value']:.3f}]")
    report.append(f"  • Zero embeddings: {embed_stats['zero_embeddings']}")
    
    # Physics validation section
    report.append("\n⚛️  PHYSICS VALIDATION")
    report.append("-" * 23)
    
    if physics_validation['is_valid']:
        report.append("✅ Status: PASSED")
    else:
        report.append("❌ Status: FAILED")
        
    if physics_validation['warnings']:
        report.append(f"\n⚠️  Warnings ({len(physics_validation['warnings'])}):")
        for warning in physics_validation['warnings']:
            report.append(f"  • {warning}")
            
    # Suggestions section
    all_suggestions = text_validation.preprocessing_suggestions
    if all_suggestions:
        report.append("\n💡 SUGGESTIONS")
        report.append("-" * 15)
        for suggestion in all_suggestions[:5]:
            report.append(f"  • {suggestion}")
            
    report.append("\n" + "=" * 45)
    
    return "\n".join(report)