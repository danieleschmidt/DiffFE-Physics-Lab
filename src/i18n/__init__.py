"""Internationalization and localization support."""

from .compliance import (
    ComplianceConfig,
    ComplianceFramework,
    ComplianceManager,
    get_compliance_manager,
)
from .translator import (
    Language,
    Translator,
    error_message,
    get_translator,
    math_term,
    physics_term,
    set_language,
    status_message,
    t,
)

__all__ = [
    "Language",
    "Translator",
    "get_translator",
    "set_language",
    "t",
    "error_message",
    "status_message",
    "physics_term",
    "math_term",
    "ComplianceManager",
    "ComplianceFramework",
    "ComplianceConfig",
    "get_compliance_manager",
]
