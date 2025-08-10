"""Internationalization and localization support."""

from .translator import (
    Language, 
    Translator, 
    get_translator, 
    set_language,
    t,
    error_message,
    status_message,
    physics_term,
    math_term
)
from .compliance import (
    ComplianceManager, 
    ComplianceFramework,
    ComplianceConfig,
    get_compliance_manager
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
    "get_compliance_manager"
]