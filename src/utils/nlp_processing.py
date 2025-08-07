"""
Advanced NLP preprocessing and validation for physics-informed sentiment analysis.

This module provides robust text processing capabilities with validation,
multilingual support, and physics-inspired normalization techniques.
"""

import re
import string
import unicodedata
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages for multilingual processing."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"


@dataclass
class TextProcessingConfig:
    """Configuration for text processing pipeline."""
    
    # Language and encoding
    language: Language = Language.ENGLISH
    encoding: str = "utf-8"
    
    # Tokenization settings
    max_sequence_length: int = 512
    vocab_size: int = 10000
    min_token_length: int = 1
    max_token_length: int = 50
    
    # Normalization settings
    lowercase: bool = True
    remove_accents: bool = False
    normalize_unicode: bool = True
    remove_extra_whitespace: bool = True
    
    # Cleaning settings
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_special_chars: bool = False
    preserve_emoticons: bool = True
    preserve_hashtags: bool = True
    preserve_mentions: bool = True
    
    # Physics-inspired settings
    energy_normalization: bool = True
    semantic_conservation: bool = True
    information_preserving_truncation: bool = True
    
    # Validation settings
    min_text_length: int = 3
    max_text_length: int = 10000
    allowed_languages: Set[Language] = field(default_factory=lambda: {Language.ENGLISH})
    reject_spam: bool = True
    reject_toxic: bool = True
    
    # Performance settings
    use_cache: bool = True
    parallel_processing: bool = False
    batch_size: int = 1000


class TextValidator:
    """Validates text input with comprehensive checks."""
    
    def __init__(self, config: TextProcessingConfig):
        self.config = config
        
        # Compile regex patterns for efficiency
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self._phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self._html_pattern = re.compile(r'<[^<]+?>')
        
        # Spam/toxic detection patterns (simplified - would use ML models in production)
        self._spam_patterns = [
            r'\b(viagra|cialis|lottery|winner|million\s+dollars)\b',
            r'click\s+here',
            r'free\s+money',
            r'nigerian\s+prince'
        ]
        self._spam_regex = re.compile('|'.join(self._spam_patterns), re.IGNORECASE)
        
        # Toxic content patterns (simplified)
        self._toxic_patterns = [
            r'\b(hate|kill|die)\s+(you|them|him|her)\b',
            r'\b(stupid|idiot|moron)\b',
        ]
        self._toxic_regex = re.compile('|'.join(self._toxic_patterns), re.IGNORECASE)
    
    def validate_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate text input and return validation result.
        
        Parameters
        ----------
        text : str
            Input text to validate
            
        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check text length
        if len(text) < self.config.min_text_length:
            issues.append(f"Text too short (min: {self.config.min_text_length})")
        
        if len(text) > self.config.max_text_length:
            issues.append(f"Text too long (max: {self.config.max_text_length})")
        
        # Check encoding
        try:
            text.encode(self.config.encoding)
        except UnicodeEncodeError:
            issues.append(f"Text contains characters not encodable in {self.config.encoding}")
        
        # Language detection (simplified - would use proper language detection)
        detected_lang = self._detect_language(text)
        if detected_lang not in self.config.allowed_languages:
            issues.append(f"Language {detected_lang.value} not in allowed languages")
        
        # Spam detection
        if self.config.reject_spam and self._spam_regex.search(text):
            issues.append("Text appears to be spam")
        
        # Toxic content detection
        if self.config.reject_toxic and self._toxic_regex.search(text):
            issues.append("Text contains potentially toxic content")
        
        # Check for excessive repetition
        if self._has_excessive_repetition(text):
            issues.append("Text contains excessive character repetition")
        
        # Check for valid characters
        if self._has_invalid_characters(text):
            issues.append("Text contains invalid characters")
        
        return len(issues) == 0, issues
    
    def _detect_language(self, text: str) -> Language:
        """Simplified language detection. In production, use proper language detector."""
        # Count character frequency to make basic language guess
        text_lower = text.lower()
        
        # English indicators
        english_words = {'the', 'and', 'a', 'an', 'is', 'are', 'was', 'were', 'have', 'has'}
        english_count = sum(1 for word in english_words if word in text_lower)
        
        # Spanish indicators
        spanish_words = {'el', 'la', 'y', 'de', 'que', 'es', 'en', 'un', 'una', 'con'}
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        
        # Simple heuristic
        if spanish_count > english_count:
            return Language.SPANISH
        else:
            return Language.ENGLISH
    
    def _has_excessive_repetition(self, text: str, threshold: int = 5) -> bool:
        """Check for excessive character or substring repetition."""
        # Check for repeated characters
        for i in range(len(text) - threshold):
            if text[i] == text[i+1] == text[i+2] == text[i+3] == text[i+4]:
                return True
        
        # Check for repeated short substrings
        words = text.split()
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True
        
        return False
    
    def _has_invalid_characters(self, text: str) -> bool:
        """Check for invalid or suspicious characters."""
        # Count control characters (excluding whitespace)
        control_chars = sum(1 for c in text if unicodedata.category(c).startswith('C') 
                           and c not in '\n\r\t')
        
        # Reject if more than 5% control characters
        return control_chars > len(text) * 0.05


class TextCleaner:
    """Cleans and normalizes text using configurable rules."""
    
    def __init__(self, config: TextProcessingConfig):
        self.config = config
        
        # Compile regex patterns
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self._phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self._html_pattern = re.compile(r'<[^<]+?>')
        self._whitespace_pattern = re.compile(r'\s+')
        
        # Emoticon patterns (preserve these)
        self._emoticon_pattern = re.compile(r'[:;=8][-o\*\']?[\)\]\(\[dDpP/:\}\{@\|\\]')
        
        # Hashtag and mention patterns
        self._hashtag_pattern = re.compile(r'#\w+')
        self._mention_pattern = re.compile(r'@\w+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean text according to configuration.
        
        Parameters
        ----------
        text : str
            Input text to clean
            
        Returns
        -------
        str
            Cleaned text
        """
        # Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove HTML tags
        if self.config.remove_html:
            text = self._html_pattern.sub(' ', text)
        
        # Handle URLs
        if self.config.remove_urls:
            text = self._url_pattern.sub(' [URL] ', text)
        
        # Handle emails
        if self.config.remove_emails:
            text = self._email_pattern.sub(' [EMAIL] ', text)
        
        # Handle phone numbers
        if self.config.remove_phone_numbers:
            text = self._phone_pattern.sub(' [PHONE] ', text)
        
        # Preserve important patterns before general cleaning
        preserved_emoticons = []
        preserved_hashtags = []
        preserved_mentions = []
        
        if self.config.preserve_emoticons:
            preserved_emoticons = self._emoticon_pattern.findall(text)
            text = self._emoticon_pattern.sub(' [EMOTICON] ', text)
        
        if self.config.preserve_hashtags:
            preserved_hashtags = self._hashtag_pattern.findall(text)
            text = self._hashtag_pattern.sub(' [HASHTAG] ', text)
        
        if self.config.preserve_mentions:
            preserved_mentions = self._mention_pattern.findall(text)
            text = self._mention_pattern.sub(' [MENTION] ', text)
        
        # Remove special characters if requested
        if self.config.remove_special_chars:
            # Keep letters, numbers, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:\[\]\(\)]', ' ', text)
        
        # Remove accents
        if self.config.remove_accents:
            text = ''.join(c for c in unicodedata.normalize('NFD', text)
                          if unicodedata.category(c) != 'Mn')
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Normalize whitespace
        if self.config.remove_extra_whitespace:
            text = self._whitespace_pattern.sub(' ', text).strip()
        
        # Restore preserved patterns
        if preserved_emoticons:
            for emoticon in preserved_emoticons:
                text = text.replace('[EMOTICON]', emoticon, 1)
        
        if preserved_hashtags:
            for hashtag in preserved_hashtags:
                text = text.replace('[HASHTAG]', hashtag, 1)
        
        if preserved_mentions:
            for mention in preserved_mentions:
                text = text.replace('[MENTION]', mention, 1)
        
        return text


class PhysicsInspiredTokenizer:
    """
    Tokenizer that applies physics-inspired principles to text segmentation.
    
    Key Physics Concepts:
    - Energy Conservation: Preserve semantic energy during tokenization
    - Information Density: Prioritize tokens with higher information content
    - Gradient Flow: Smooth transitions between tokens
    """
    
    def __init__(self, config: TextProcessingConfig, vocabulary: Optional[Dict[str, int]] = None):
        self.config = config
        self.vocabulary = vocabulary or {}
        self.inverse_vocab = {v: k for k, v in self.vocabulary.items()}
        
        # Special tokens
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        
        # Initialize vocabulary with special tokens
        if not self.vocabulary:
            self.vocabulary.update(self.special_tokens)
            self.inverse_vocab.update({v: k for k, v in self.special_tokens.items()})
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using physics-inspired segmentation.
        
        Parameters
        ----------
        text : str
            Input text to tokenize
            
        Returns
        -------
        List[str]
            List of tokens
        """
        # Basic word tokenization
        words = text.split()
        
        tokens = []
        for word in words:
            # Apply subword tokenization for unknown words
            if word in self.vocabulary:
                tokens.append(word)
            else:
                # Simple subword tokenization (in production, use BPE or SentencePiece)
                subtokens = self._subword_tokenize(word)
                tokens.extend(subtokens)
        
        # Apply physics-inspired filtering
        if self.config.energy_normalization:
            tokens = self._apply_energy_normalization(tokens)
        
        # Truncate with information preservation
        if len(tokens) > self.config.max_sequence_length:
            if self.config.information_preserving_truncation:
                tokens = self._information_preserving_truncate(tokens)
            else:
                tokens = tokens[:self.config.max_sequence_length]
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Parameters
        ----------
        text : str
            Input text to encode
            
        Returns
        -------
        List[int]
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        token_ids = []
        for token in tokens:
            if token in self.vocabulary:
                token_ids.append(self.vocabulary[token])
            else:
                token_ids.append(self.special_tokens['[UNK]'])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Parameters
        ----------
        token_ids : List[int]
            List of token IDs to decode
            
        Returns
        -------
        str
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if token not in ['[PAD]', '[CLS]', '[SEP]']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def _subword_tokenize(self, word: str) -> List[str]:
        """Simple subword tokenization."""
        if len(word) <= 4:
            return [word]
        
        # Break into subwords at morphological boundaries (simplified)
        subwords = []
        
        # Common prefixes
        prefixes = ['un', 'pre', 're', 'anti', 'de', 'dis', 'over', 'under']
        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                subwords.append(prefix)
                word = word[len(prefix):]
                break
        
        # Common suffixes
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                subwords.extend([word[:-len(suffix)], suffix])
                return subwords if subwords[0] != word[:-len(suffix)] else [word]
        
        # If no affixes found, add the remaining word
        if subwords:
            subwords.append(word)
        else:
            subwords = [word]
        
        return subwords
    
    def _apply_energy_normalization(self, tokens: List[str]) -> List[str]:
        """Apply energy normalization to token sequence."""
        if not HAS_NUMPY:
            return tokens
        
        # Compute token "energy" based on length and frequency
        energies = []
        for token in tokens:
            # Length-based energy
            length_energy = len(token) / 10.0
            
            # Frequency-based energy (inverse frequency = higher energy)
            # In production, would use actual corpus statistics
            freq_energy = 1.0 / (len(token) + 1)  # Simplified
            
            total_energy = length_energy + freq_energy
            energies.append(total_energy)
        
        # Normalize energies to maintain conservation
        if energies:
            energies = np.array(energies)
            energies = energies / np.sum(energies) * len(energies)
            
            # Filter tokens with very low energy (but keep minimum diversity)
            keep_indices = np.argsort(energies)[::-1][:max(len(tokens) // 2, 10)]
            keep_indices = sorted(keep_indices)
            
            return [tokens[i] for i in keep_indices]
        
        return tokens
    
    def _information_preserving_truncate(self, tokens: List[str]) -> List[str]:
        """Truncate while preserving maximum information."""
        if not HAS_NUMPY:
            return tokens[:self.config.max_sequence_length]
        
        # Compute information content for each token
        info_scores = []
        for i, token in enumerate(tokens):
            # Position-based scoring (beginning and end are important)
            position_score = 1.0
            if i < 3:  # Beginning tokens
                position_score += 0.5
            if i >= len(tokens) - 3:  # End tokens
                position_score += 0.3
            
            # Length-based scoring
            length_score = min(len(token) / 5.0, 1.0)
            
            # Uniqueness score
            uniqueness_score = 1.0 / (tokens[:i].count(token) + 1)
            
            total_score = position_score * length_score * uniqueness_score
            info_scores.append(total_score)
        
        # Select top scoring tokens while preserving order
        target_length = self.config.max_sequence_length - 2  # Reserve space for special tokens
        
        if len(tokens) <= target_length:
            return tokens
        
        # Always keep first and last few tokens
        keep_indices = set(range(min(3, len(tokens))))
        keep_indices.update(range(max(0, len(tokens) - 3), len(tokens)))
        
        # Add highest scoring tokens from the middle
        remaining_slots = target_length - len(keep_indices)
        middle_indices = range(3, len(tokens) - 3)
        middle_scores = [(info_scores[i], i) for i in middle_indices if i not in keep_indices]
        middle_scores.sort(reverse=True)
        
        for _, idx in middle_scores[:remaining_slots]:
            keep_indices.add(idx)
        
        # Return tokens in original order
        keep_indices = sorted(keep_indices)
        return [tokens[i] for i in keep_indices]


class TextProcessingPipeline:
    """Complete text processing pipeline with validation, cleaning, and tokenization."""
    
    def __init__(self, config: TextProcessingConfig):
        self.config = config
        self.validator = TextValidator(config)
        self.cleaner = TextCleaner(config)
        self.tokenizer = PhysicsInspiredTokenizer(config)
        
        # Processing cache for efficiency
        self._cache = {} if config.use_cache else None
    
    def process_text(self, text: str) -> Dict[str, Union[bool, str, List[str], List[int]]]:
        """
        Process text through the complete pipeline.
        
        Parameters
        ----------
        text : str
            Input text to process
            
        Returns
        -------
        Dict[str, Union[bool, str, List[str], List[int]]]
            Processing results with validation, cleaning, and tokenization
        """
        # Check cache
        if self._cache is not None and text in self._cache:
            return self._cache[text]
        
        result = {
            'original_text': text,
            'is_valid': False,
            'validation_issues': [],
            'cleaned_text': '',
            'tokens': [],
            'token_ids': [],
            'processing_stats': {}
        }
        
        try:
            # Validation
            is_valid, issues = self.validator.validate_text(text)
            result['is_valid'] = is_valid
            result['validation_issues'] = issues
            
            if not is_valid:
                logger.warning(f"Text validation failed: {issues}")
                # Still attempt processing for analysis
            
            # Cleaning
            cleaned_text = self.cleaner.clean_text(text)
            result['cleaned_text'] = cleaned_text
            
            # Tokenization
            tokens = self.tokenizer.tokenize(cleaned_text)
            token_ids = self.tokenizer.encode(cleaned_text)
            
            result['tokens'] = tokens
            result['token_ids'] = token_ids
            
            # Processing statistics
            result['processing_stats'] = {
                'original_length': len(text),
                'cleaned_length': len(cleaned_text),
                'compression_ratio': len(cleaned_text) / len(text) if text else 0,
                'num_tokens': len(tokens),
                'avg_token_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
                'vocab_coverage': sum(1 for t in tokens if t in self.tokenizer.vocabulary) / len(tokens) if tokens else 0
            }
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            result['validation_issues'].append(f"Processing error: {str(e)}")
        
        # Cache result
        if self._cache is not None:
            self._cache[text] = result
        
        return result
    
    def process_batch(self, texts: List[str]) -> List[Dict]:
        """Process multiple texts efficiently."""
        results = []
        
        if self.config.parallel_processing:
            # Would implement parallel processing here
            pass
        
        for text in texts:
            result = self.process_text(text)
            results.append(result)
        
        return results
    
    def clear_cache(self):
        """Clear the processing cache."""
        if self._cache is not None:
            self._cache.clear()


# Factory function for creating processing pipeline
def create_processing_pipeline(language: Language = Language.ENGLISH,
                             **config_kwargs) -> TextProcessingPipeline:
    """
    Create a text processing pipeline with sensible defaults.
    
    Parameters
    ----------
    language : Language
        Primary language for processing
    **config_kwargs
        Additional configuration parameters
        
    Returns
    -------
    TextProcessingPipeline
        Configured processing pipeline
    """
    
    config = TextProcessingConfig(
        language=language,
        allowed_languages={language},
        **config_kwargs
    )
    
    return TextProcessingPipeline(config)