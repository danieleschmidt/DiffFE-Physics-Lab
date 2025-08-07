"""Internationalization (i18n) utilities for sentiment analysis framework."""

import json
import os
import re
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import threading


@dataclass
class TranslationEntry:
    """Translation entry with metadata."""
    
    key: str
    value: str
    language: str
    context: Optional[str] = None
    plural_forms: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TranslationManager:
    """Manager for handling translations and internationalization."""
    
    def __init__(self, default_language: str = 'en', translations_dir: Optional[Path] = None):
        """Initialize translation manager.
        
        Parameters
        ----------
        default_language : str, optional
            Default language code, by default 'en'
        translations_dir : Path, optional
            Directory containing translation files
        """
        self.default_language = default_language
        self.current_language = default_language
        self.translations_dir = translations_dir or Path(__file__).parent.parent.parent / 'translations'
        self.translations: Dict[str, Dict[str, TranslationEntry]] = {}
        self._lock = threading.RLock()
        
        # Language metadata
        self.language_info = {
            'en': {'name': 'English', 'direction': 'ltr', 'plural_rules': 'english'},
            'es': {'name': 'Español', 'direction': 'ltr', 'plural_rules': 'romance'},
            'fr': {'name': 'Français', 'direction': 'ltr', 'plural_rules': 'romance'},
            'de': {'name': 'Deutsch', 'direction': 'ltr', 'plural_rules': 'germanic'},
            'zh': {'name': '中文', 'direction': 'ltr', 'plural_rules': 'chinese'},
            'ja': {'name': '日本語', 'direction': 'ltr', 'plural_rules': 'japanese'},
            'ar': {'name': 'العربية', 'direction': 'rtl', 'plural_rules': 'arabic'},
            'hi': {'name': 'हिन्दी', 'direction': 'ltr', 'plural_rules': 'hindi'},
            'pt': {'name': 'Português', 'direction': 'ltr', 'plural_rules': 'romance'},
            'ru': {'name': 'Русский', 'direction': 'ltr', 'plural_rules': 'slavic'}
        }
        
        # Load translations
        self._load_translations()
        
    def _load_translations(self):
        """Load all translation files."""
        if not self.translations_dir.exists():
            self.translations_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_translations()
            
        for lang_file in self.translations_dir.glob('*.json'):
            lang_code = lang_file.stem
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._load_language_data(lang_code, data)
            except Exception as e:
                print(f"Error loading translations for {lang_code}: {e}")
                
    def _load_language_data(self, lang_code: str, data: Dict[str, Any]):
        """Load translation data for a specific language."""
        with self._lock:
            self.translations[lang_code] = {}
            
            for key, value in data.items():
                if isinstance(value, str):
                    # Simple string translation
                    self.translations[lang_code][key] = TranslationEntry(
                        key=key,
                        value=value,
                        language=lang_code
                    )
                elif isinstance(value, dict):
                    # Complex translation with metadata
                    entry = TranslationEntry(
                        key=key,
                        value=value.get('value', ''),
                        language=lang_code,
                        context=value.get('context'),
                        plural_forms=value.get('plural_forms'),
                        metadata=value.get('metadata', {})
                    )
                    self.translations[lang_code][key] = entry
                    
    def _create_default_translations(self):
        """Create default translation files."""
        
        # English (base language)
        en_translations = {
            # General UI
            'app.name': 'Sentiment Analysis Pro',
            'app.description': 'Physics-informed sentiment analysis framework',
            'app.version': 'Version',
            
            # API Messages
            'api.success': 'Operation completed successfully',
            'api.error.general': 'An error occurred while processing your request',
            'api.error.validation': 'Validation error in request data',
            'api.error.rate_limit': 'Rate limit exceeded. Please try again later',
            'api.error.authentication': 'Authentication failed',
            'api.error.authorization': 'Insufficient permissions',
            'api.error.not_found': 'Requested resource not found',
            'api.error.internal': 'Internal server error',
            
            # Sentiment Analysis
            'sentiment.positive': 'Positive',
            'sentiment.negative': 'Negative',
            'sentiment.neutral': 'Neutral',
            'sentiment.very_positive': 'Very Positive',
            'sentiment.very_negative': 'Very Negative',
            'sentiment.confidence': 'Confidence',
            'sentiment.score': 'Sentiment Score',
            'sentiment.analysis_complete': 'Sentiment analysis completed',
            'sentiment.processing': 'Processing sentiment analysis...',
            
            # Physics Terms
            'physics.diffusion': 'Diffusion',
            'physics.reaction': 'Reaction',
            'physics.temperature': 'Temperature',
            'physics.energy': 'Energy',
            'physics.convergence': 'Convergence',
            'physics.stability': 'Stability',
            
            # Validation Messages
            'validation.text.empty': 'Text cannot be empty',
            'validation.text.too_long': 'Text exceeds maximum length of {max_length} characters',
            'validation.batch.too_large': 'Batch size exceeds maximum of {max_size} items',
            'validation.parameter.invalid': 'Invalid parameter value: {parameter}',
            'validation.format.invalid': 'Invalid data format',
            
            # Performance Messages
            'performance.processing_time': 'Processing Time',
            'performance.throughput': 'Throughput',
            'performance.memory_usage': 'Memory Usage',
            'performance.cache_hit_rate': 'Cache Hit Rate',
            'performance.optimization.enabled': 'Performance optimization enabled',
            
            # Security Messages
            'security.threat_detected': 'Security threat detected',
            'security.access_denied': 'Access denied',
            'security.rate_limited': 'Rate limit exceeded',
            'security.invalid_input': 'Invalid or potentially malicious input detected',
            
            # Units
            'units.seconds': 'seconds',
            'units.milliseconds': 'milliseconds',
            'units.requests_per_second': 'requests/second',
            'units.texts_per_second': 'texts/second',
            'units.megabytes': 'MB',
            'units.percentage': '%'
        }
        
        # Spanish translations
        es_translations = {
            'app.name': 'Análisis de Sentimiento Pro',
            'app.description': 'Marco de análisis de sentimiento informado por física',
            'app.version': 'Versión',
            
            'api.success': 'Operación completada exitosamente',
            'api.error.general': 'Ocurrió un error al procesar su solicitud',
            'api.error.validation': 'Error de validación en los datos de solicitud',
            'api.error.rate_limit': 'Límite de tasa excedido. Intente nuevamente más tarde',
            
            'sentiment.positive': 'Positivo',
            'sentiment.negative': 'Negativo',
            'sentiment.neutral': 'Neutral',
            'sentiment.very_positive': 'Muy Positivo',
            'sentiment.very_negative': 'Muy Negativo',
            'sentiment.confidence': 'Confianza',
            'sentiment.score': 'Puntuación de Sentimiento',
            
            'physics.diffusion': 'Difusión',
            'physics.reaction': 'Reacción',
            'physics.temperature': 'Temperatura',
            'physics.energy': 'Energía',
            'physics.convergence': 'Convergencia',
            'physics.stability': 'Estabilidad',
            
            'validation.text.empty': 'El texto no puede estar vacío',
            'validation.text.too_long': 'El texto excede la longitud máxima de {max_length} caracteres',
            
            'units.seconds': 'segundos',
            'units.milliseconds': 'milisegundos',
            'units.megabytes': 'MB',
            'units.percentage': '%'
        }
        
        # French translations
        fr_translations = {
            'app.name': 'Analyse de Sentiment Pro',
            'app.description': 'Cadre d\'analyse de sentiment informé par la physique',
            'app.version': 'Version',
            
            'sentiment.positive': 'Positif',
            'sentiment.negative': 'Négatif',
            'sentiment.neutral': 'Neutre',
            'sentiment.very_positive': 'Très Positif',
            'sentiment.very_negative': 'Très Négatif',
            'sentiment.confidence': 'Confiance',
            'sentiment.score': 'Score de Sentiment',
            
            'physics.diffusion': 'Diffusion',
            'physics.reaction': 'Réaction',
            'physics.temperature': 'Température',
            'physics.energy': 'Énergie',
            'physics.convergence': 'Convergence',
            'physics.stability': 'Stabilité',
            
            'validation.text.empty': 'Le texte ne peut pas être vide',
            'validation.text.too_long': 'Le texte dépasse la longueur maximale de {max_length} caractères',
            
            'units.seconds': 'secondes',
            'units.milliseconds': 'millisecondes',
            'units.megabytes': 'Mo',
            'units.percentage': '%'
        }
        
        # German translations
        de_translations = {
            'app.name': 'Sentiment Analyse Pro',
            'app.description': 'Physik-informiertes Sentiment-Analyse-Framework',
            'app.version': 'Version',
            
            'sentiment.positive': 'Positiv',
            'sentiment.negative': 'Negativ',
            'sentiment.neutral': 'Neutral',
            'sentiment.very_positive': 'Sehr Positiv',
            'sentiment.very_negative': 'Sehr Negativ',
            'sentiment.confidence': 'Vertrauen',
            'sentiment.score': 'Sentiment-Bewertung',
            
            'physics.diffusion': 'Diffusion',
            'physics.reaction': 'Reaktion',
            'physics.temperature': 'Temperatur',
            'physics.energy': 'Energie',
            'physics.convergence': 'Konvergenz',
            'physics.stability': 'Stabilität',
            
            'validation.text.empty': 'Text darf nicht leer sein',
            'validation.text.too_long': 'Text überschreitet maximale Länge von {max_length} Zeichen',
            
            'units.seconds': 'Sekunden',
            'units.milliseconds': 'Millisekunden',
            'units.megabytes': 'MB',
            'units.percentage': '%'
        }
        
        # Chinese (Simplified) translations
        zh_translations = {
            'app.name': '情感分析专业版',
            'app.description': '基于物理的情感分析框架',
            'app.version': '版本',
            
            'sentiment.positive': '正面',
            'sentiment.negative': '负面',
            'sentiment.neutral': '中性',
            'sentiment.very_positive': '非常正面',
            'sentiment.very_negative': '非常负面',
            'sentiment.confidence': '置信度',
            'sentiment.score': '情感得分',
            
            'physics.diffusion': '扩散',
            'physics.reaction': '反应',
            'physics.temperature': '温度',
            'physics.energy': '能量',
            'physics.convergence': '收敛',
            'physics.stability': '稳定性',
            
            'validation.text.empty': '文本不能为空',
            'validation.text.too_long': '文本长度超过最大限制 {max_length} 个字符',
            
            'units.seconds': '秒',
            'units.milliseconds': '毫秒',
            'units.megabytes': 'MB',
            'units.percentage': '%'
        }
        
        # Japanese translations
        ja_translations = {
            'app.name': 'センチメント分析プロ',
            'app.description': '物理学に基づくセンチメント分析フレームワーク',
            'app.version': 'バージョン',
            
            'sentiment.positive': 'ポジティブ',
            'sentiment.negative': 'ネガティブ',
            'sentiment.neutral': 'ニュートラル',
            'sentiment.very_positive': '非常にポジティブ',
            'sentiment.very_negative': '非常にネガティブ',
            'sentiment.confidence': '信頼度',
            'sentiment.score': 'センチメントスコア',
            
            'physics.diffusion': '拡散',
            'physics.reaction': '反応',
            'physics.temperature': '温度',
            'physics.energy': 'エネルギー',
            'physics.convergence': '収束',
            'physics.stability': '安定性',
            
            'validation.text.empty': 'テキストは空にできません',
            'validation.text.too_long': 'テキストが最大長 {max_length} 文字を超えています',
            
            'units.seconds': '秒',
            'units.milliseconds': 'ミリ秒',
            'units.megabytes': 'MB',
            'units.percentage': '%'
        }
        
        # Save translation files
        translations = {
            'en': en_translations,
            'es': es_translations,
            'fr': fr_translations,
            'de': de_translations,
            'zh': zh_translations,
            'ja': ja_translations
        }
        
        for lang, trans in translations.items():
            file_path = self.translations_dir / f'{lang}.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(trans, f, ensure_ascii=False, indent=2)
                
    def set_language(self, language: str):
        """Set current language.
        
        Parameters
        ----------
        language : str
            Language code to set
        """
        if language in self.translations or language == self.default_language:
            self.current_language = language
        else:
            print(f"Warning: Language {language} not available, falling back to {self.default_language}")
            
    def translate(
        self, 
        key: str, 
        language: Optional[str] = None,
        default: Optional[str] = None,
        **kwargs
    ) -> str:
        """Translate a key to the current or specified language.
        
        Parameters
        ----------
        key : str
            Translation key
        language : str, optional
            Target language, uses current language if None
        default : str, optional
            Default value if translation not found
        **kwargs
            Variables for string formatting
            
        Returns
        -------
        str
            Translated text
        """
        target_lang = language or self.current_language
        
        # Try target language
        if target_lang in self.translations and key in self.translations[target_lang]:
            translation = self.translations[target_lang][key].value
        # Fall back to default language
        elif self.default_language in self.translations and key in self.translations[self.default_language]:
            translation = self.translations[self.default_language][key].value
        # Use default or key
        else:
            translation = default or key
            
        # Apply string formatting
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError):
                # If formatting fails, return unformatted string
                pass
                
        return translation
        
    def translate_dict(
        self, 
        data: Dict[str, Any], 
        language: Optional[str] = None,
        key_prefix: str = ''
    ) -> Dict[str, Any]:
        """Translate dictionary values recursively.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary to translate
        language : str, optional
            Target language
        key_prefix : str, optional
            Prefix for translation keys
            
        Returns
        -------
        Dict[str, Any]
            Translated dictionary
        """
        translated = {}
        
        for key, value in data.items():
            translation_key = f"{key_prefix}.{key}" if key_prefix else key
            
            if isinstance(value, str):
                translated[key] = self.translate(translation_key, language, default=value)
            elif isinstance(value, dict):
                translated[key] = self.translate_dict(value, language, translation_key)
            else:
                translated[key] = value
                
        return translated
        
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes.
        
        Returns
        -------
        List[str]
            Available language codes
        """
        return list(self.translations.keys())
        
    def get_language_info(self, language: str) -> Dict[str, Any]:
        """Get information about a language.
        
        Parameters
        ----------
        language : str
            Language code
            
        Returns
        -------
        Dict[str, Any]
            Language information
        """
        return self.language_info.get(language, {
            'name': language,
            'direction': 'ltr',
            'plural_rules': 'english'
        })
        
    def add_translation(
        self, 
        key: str, 
        value: str, 
        language: str,
        context: Optional[str] = None
    ):
        """Add or update a translation.
        
        Parameters
        ----------
        key : str
            Translation key
        value : str
            Translation value
        language : str
            Language code
        context : str, optional
            Context information
        """
        with self._lock:
            if language not in self.translations:
                self.translations[language] = {}
                
            self.translations[language][key] = TranslationEntry(
                key=key,
                value=value,
                language=language,
                context=context
            )
            
    def export_translations(self, language: str, file_path: Path):
        """Export translations for a language to file.
        
        Parameters
        ----------
        language : str
            Language code
        file_path : Path
            Output file path
        """
        if language not in self.translations:
            raise ValueError(f"Language {language} not found")
            
        translations_dict = {}
        for key, entry in self.translations[language].items():
            translations_dict[key] = entry.value
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(translations_dict, f, ensure_ascii=False, indent=2)


# Global translation manager
_global_translation_manager = None


def get_translation_manager() -> TranslationManager:
    """Get global translation manager instance."""
    global _global_translation_manager
    
    if _global_translation_manager is None:
        _global_translation_manager = TranslationManager()
        
    return _global_translation_manager


def t(key: str, language: Optional[str] = None, default: Optional[str] = None, **kwargs) -> str:
    """Shorthand function for translation.
    
    Parameters
    ----------
    key : str
        Translation key
    language : str, optional
        Target language
    default : str, optional
        Default value
    **kwargs
        Formatting variables
        
    Returns
    -------
    str
        Translated text
    """
    return get_translation_manager().translate(key, language, default, **kwargs)


def set_language(language: str):
    """Set global language.
    
    Parameters
    ----------
    language : str
        Language code
    """
    get_translation_manager().set_language(language)


def get_available_languages() -> List[str]:
    """Get available languages.
    
    Returns
    -------
    List[str]
        Available language codes
    """
    return get_translation_manager().get_available_languages()


class LocalizationMiddleware:
    """Middleware for handling localization in web requests."""
    
    def __init__(self, app, translation_manager: TranslationManager = None):
        """Initialize localization middleware.
        
        Parameters
        ----------
        app
            Flask/web application instance
        translation_manager : TranslationManager, optional
            Translation manager to use
        """
        self.app = app
        self.translation_manager = translation_manager or get_translation_manager()
        
    def __call__(self, environ, start_response):
        """WSGI middleware call."""
        # Extract language preference from headers
        accept_language = environ.get('HTTP_ACCEPT_LANGUAGE', '')
        preferred_language = self._parse_accept_language(accept_language)
        
        # Set language for request
        if preferred_language in self.translation_manager.get_available_languages():
            self.translation_manager.set_language(preferred_language)
            
        return self.app(environ, start_response)
        
    def _parse_accept_language(self, accept_language: str) -> str:
        """Parse Accept-Language header.
        
        Parameters
        ----------
        accept_language : str
            Accept-Language header value
            
        Returns
        -------
        str
            Preferred language code
        """
        if not accept_language:
            return self.translation_manager.default_language
            
        # Parse language preferences
        languages = []
        for lang in accept_language.split(','):
            parts = lang.strip().split(';')
            if len(parts) > 1 and parts[1].startswith('q='):
                try:
                    quality = float(parts[1][2:])
                except ValueError:
                    quality = 1.0
            else:
                quality = 1.0
                
            lang_code = parts[0].strip().lower()
            # Handle language-country codes (e.g., en-US -> en)
            if '-' in lang_code:
                lang_code = lang_code.split('-')[0]
                
            languages.append((quality, lang_code))
            
        # Sort by quality and return highest preference
        languages.sort(reverse=True, key=lambda x: x[0])
        
        for quality, lang_code in languages:
            if lang_code in self.translation_manager.get_available_languages():
                return lang_code
                
        return self.translation_manager.default_language