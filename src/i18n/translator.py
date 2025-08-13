"""Multi-language translation support for DiffFE-Physics-Lab."""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


class Translator:
    """Multi-language translator with fallback support."""

    def __init__(self, default_language: Language = Language.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = {}
        self.load_default_translations()

    def load_default_translations(self):
        """Load built-in translations."""
        self.translations = {
            Language.ENGLISH: {
                # Core messages
                "error.general": "An error occurred",
                "error.invalid_input": "Invalid input provided",
                "error.computation_failed": "Computation failed",
                "error.memory_insufficient": "Insufficient memory",
                "error.backend_unavailable": "Backend unavailable",
                # Status messages
                "status.initializing": "Initializing system",
                "status.computing": "Computing solution",
                "status.converged": "Solution converged",
                "status.failed": "Operation failed",
                "status.completed": "Operation completed successfully",
                # Progress messages
                "progress.iteration": "Iteration {iteration} of {total}",
                "progress.percentage": "{percentage:.1f}% complete",
                "progress.time_remaining": "Estimated time remaining: {time}",
                # Physics terms
                "physics.temperature": "Temperature",
                "physics.pressure": "Pressure",
                "physics.velocity": "Velocity",
                "physics.displacement": "Displacement",
                "physics.stress": "Stress",
                "physics.strain": "Strain",
                # Mathematical terms
                "math.gradient": "Gradient",
                "math.jacobian": "Jacobian",
                "math.hessian": "Hessian",
                "math.eigenvalue": "Eigenvalue",
                "math.convergence": "Convergence",
                "math.residual": "Residual",
                # Units
                "units.celsius": "°C",
                "units.kelvin": "K",
                "units.pascal": "Pa",
                "units.meter_per_second": "m/s",
                "units.newton": "N",
                "units.joule": "J",
            },
            Language.SPANISH: {
                # Core messages
                "error.general": "Ocurrió un error",
                "error.invalid_input": "Entrada inválida proporcionada",
                "error.computation_failed": "El cálculo falló",
                "error.memory_insufficient": "Memoria insuficiente",
                "error.backend_unavailable": "Backend no disponible",
                # Status messages
                "status.initializing": "Inicializando sistema",
                "status.computing": "Calculando solución",
                "status.converged": "Solución convergida",
                "status.failed": "Operación falló",
                "status.completed": "Operación completada exitosamente",
                # Progress messages
                "progress.iteration": "Iteración {iteration} de {total}",
                "progress.percentage": "{percentage:.1f}% completo",
                "progress.time_remaining": "Tiempo estimado restante: {time}",
                # Physics terms
                "physics.temperature": "Temperatura",
                "physics.pressure": "Presión",
                "physics.velocity": "Velocidad",
                "physics.displacement": "Desplazamiento",
                "physics.stress": "Esfuerzo",
                "physics.strain": "Deformación",
                # Mathematical terms
                "math.gradient": "Gradiente",
                "math.jacobian": "Jacobiano",
                "math.hessian": "Hessiano",
                "math.eigenvalue": "Valor propio",
                "math.convergence": "Convergencia",
                "math.residual": "Residual",
                # Units (mostly same)
                "units.celsius": "°C",
                "units.kelvin": "K",
                "units.pascal": "Pa",
                "units.meter_per_second": "m/s",
                "units.newton": "N",
                "units.joule": "J",
            },
            Language.FRENCH: {
                # Core messages
                "error.general": "Une erreur s'est produite",
                "error.invalid_input": "Entrée invalide fournie",
                "error.computation_failed": "Le calcul a échoué",
                "error.memory_insufficient": "Mémoire insuffisante",
                "error.backend_unavailable": "Backend indisponible",
                # Status messages
                "status.initializing": "Initialisation du système",
                "status.computing": "Calcul de la solution",
                "status.converged": "Solution convergée",
                "status.failed": "Opération échouée",
                "status.completed": "Opération terminée avec succès",
                # Physics terms
                "physics.temperature": "Température",
                "physics.pressure": "Pression",
                "physics.velocity": "Vitesse",
                "physics.displacement": "Déplacement",
                "physics.stress": "Contrainte",
                "physics.strain": "Déformation",
                # Mathematical terms
                "math.gradient": "Gradient",
                "math.jacobian": "Jacobien",
                "math.hessian": "Hessien",
                "math.eigenvalue": "Valeur propre",
                "math.convergence": "Convergence",
                "math.residual": "Résiduel",
            },
            Language.GERMAN: {
                # Core messages
                "error.general": "Ein Fehler ist aufgetreten",
                "error.invalid_input": "Ungültige Eingabe bereitgestellt",
                "error.computation_failed": "Berechnung fehlgeschlagen",
                "error.memory_insufficient": "Unzureichender Speicher",
                "error.backend_unavailable": "Backend nicht verfügbar",
                # Status messages
                "status.initializing": "System wird initialisiert",
                "status.computing": "Lösung wird berechnet",
                "status.converged": "Lösung konvergiert",
                "status.failed": "Operation fehlgeschlagen",
                "status.completed": "Operation erfolgreich abgeschlossen",
                # Physics terms
                "physics.temperature": "Temperatur",
                "physics.pressure": "Druck",
                "physics.velocity": "Geschwindigkeit",
                "physics.displacement": "Verschiebung",
                "physics.stress": "Spannung",
                "physics.strain": "Dehnung",
                # Mathematical terms
                "math.gradient": "Gradient",
                "math.jacobian": "Jacobi-Matrix",
                "math.hessian": "Hesse-Matrix",
                "math.eigenvalue": "Eigenwert",
                "math.convergence": "Konvergenz",
                "math.residual": "Residuum",
            },
            Language.JAPANESE: {
                # Core messages
                "error.general": "エラーが発生しました",
                "error.invalid_input": "無効な入力が提供されました",
                "error.computation_failed": "計算に失敗しました",
                "error.memory_insufficient": "メモリ不足",
                "error.backend_unavailable": "バックエンドが利用できません",
                # Status messages
                "status.initializing": "システム初期化中",
                "status.computing": "解計算中",
                "status.converged": "解が収束しました",
                "status.failed": "操作が失敗しました",
                "status.completed": "操作が正常に完了しました",
                # Physics terms
                "physics.temperature": "温度",
                "physics.pressure": "圧力",
                "physics.velocity": "速度",
                "physics.displacement": "変位",
                "physics.stress": "応力",
                "physics.strain": "ひずみ",
                # Mathematical terms
                "math.gradient": "勾配",
                "math.jacobian": "ヤコビ行列",
                "math.hessian": "ヘシアン行列",
                "math.eigenvalue": "固有値",
                "math.convergence": "収束",
                "math.residual": "残差",
            },
            Language.CHINESE: {
                # Core messages
                "error.general": "发生错误",
                "error.invalid_input": "提供的输入无效",
                "error.computation_failed": "计算失败",
                "error.memory_insufficient": "内存不足",
                "error.backend_unavailable": "后端不可用",
                # Status messages
                "status.initializing": "正在初始化系统",
                "status.computing": "正在计算解",
                "status.converged": "解已收敛",
                "status.failed": "操作失败",
                "status.completed": "操作成功完成",
                # Physics terms
                "physics.temperature": "温度",
                "physics.pressure": "压力",
                "physics.velocity": "速度",
                "physics.displacement": "位移",
                "physics.stress": "应力",
                "physics.strain": "应变",
                # Mathematical terms
                "math.gradient": "梯度",
                "math.jacobian": "雅可比矩阵",
                "math.hessian": "黑塞矩阵",
                "math.eigenvalue": "特征值",
                "math.convergence": "收敛",
                "math.residual": "残差",
            },
        }

    def set_language(self, language: Language):
        """Set the current language."""
        self.current_language = language
        logger.info(f"Language set to {language.value}")

    def translate(self, key: str, **kwargs) -> str:
        """Translate a message key with optional formatting."""
        # Try current language
        if self.current_language in self.translations:
            if key in self.translations[self.current_language]:
                message = self.translations[self.current_language][key]
                try:
                    return message.format(**kwargs) if kwargs else message
                except KeyError as e:
                    logger.warning(f"Missing format parameter {e} for key '{key}'")
                    return message

        # Fall back to default language
        if self.default_language in self.translations:
            if key in self.translations[self.default_language]:
                message = self.translations[self.default_language][key]
                try:
                    return message.format(**kwargs) if kwargs else message
                except KeyError as e:
                    logger.warning(f"Missing format parameter {e} for key '{key}'")
                    return message

        # Final fallback - return the key itself
        logger.warning(
            f"Translation not found for key '{key}' in {self.current_language.value}"
        )
        return key

    def t(self, key: str, **kwargs) -> str:
        """Shorthand for translate."""
        return self.translate(key, **kwargs)

    def add_custom_translations(self, language: Language, translations: Dict[str, str]):
        """Add custom translations for a language."""
        if language not in self.translations:
            self.translations[language] = {}

        self.translations[language].update(translations)
        logger.info(
            f"Added {len(translations)} custom translations for {language.value}"
        )

    def get_available_languages(self) -> list:
        """Get list of available languages."""
        return list(self.translations.keys())

    def load_translations_from_file(self, file_path: Path, language: Language):
        """Load translations from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                translations = json.load(f)

            self.add_custom_translations(language, translations)
            logger.info(f"Loaded translations from {file_path} for {language.value}")

        except Exception as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")


# Global translator instance
_global_translator = Translator()


def get_translator() -> Translator:
    """Get the global translator instance."""
    return _global_translator


def set_language(language: Language):
    """Set the global language."""
    _global_translator.set_language(language)


def t(key: str, **kwargs) -> str:
    """Global translation function."""
    return _global_translator.translate(key, **kwargs)


# Convenience functions for specific domains
def error_message(key: str, **kwargs) -> str:
    """Get error message in current language."""
    return t(f"error.{key}", **kwargs)


def status_message(key: str, **kwargs) -> str:
    """Get status message in current language."""
    return t(f"status.{key}", **kwargs)


def physics_term(key: str, **kwargs) -> str:
    """Get physics term in current language."""
    return t(f"physics.{key}", **kwargs)


def math_term(key: str, **kwargs) -> str:
    """Get mathematical term in current language."""
    return t(f"math.{key}", **kwargs)


def unit_symbol(key: str) -> str:
    """Get unit symbol (typically language-independent)."""
    return t(f"units.{key}")
