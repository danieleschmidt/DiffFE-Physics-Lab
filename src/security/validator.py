"""Input validation and sanitization for security."""

import re
import html
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Validator for input sanitization and security checks.
    
    Provides comprehensive input validation and sanitization to prevent
    security vulnerabilities including injection attacks, path traversal,
    and malformed input handling.
    
    Examples
    --------
    >>> validator = SecurityValidator()
    >>> clean_data = validator.sanitize_dict(user_input)
    >>> if validator.is_safe_path(file_path):
    ...     process_file(file_path)
    """
    
    def __init__(self, max_string_length: int = 10000):
        self.max_string_length = max_string_length
        self.dangerous_patterns = self._initialize_dangerous_patterns()
    
    def _initialize_dangerous_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize patterns for detecting dangerous input."""
        return {
            'sql_injection': re.compile(
                r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|OR|AND)\b)|'
                r'(--|#|/\*|\*/|;|\'\s*OR\s*\'|"\s*OR\s*")',
                re.IGNORECASE
            ),
            'xss_script': re.compile(
                r'<script[^>]*>.*?</script>|javascript:|vbscript:|onload=|onerror=',
                re.IGNORECASE | re.DOTALL
            ),
            'command_injection': re.compile(
                r'[;&|`$(){}[\]\\]|(\.\./)|(\\\.\.)|(^|[^a-zA-Z0-9])(cat|ls|ps|whoami|id|uname)\s',
                re.IGNORECASE
            ),
            'path_traversal': re.compile(
                r'(\.\.[/\\])|(\.\.\\\)|(/etc/)|(/proc/)|(/sys/)|(\\\\\\.\\)|(\\\\.\\\\)',
                re.IGNORECASE
            ),
            'ldap_injection': re.compile(
                r'[()!&|*]|(\bAND\b)|(\bOR\b)|(\bNOT\b)',
                re.IGNORECASE
            )
        }
    
    def sanitize_string(self, value: str, allow_html: bool = False, max_length: Optional[int] = None) -> str:
        """Sanitize string input.
        
        Parameters
        ----------
        value : str
            Input string to sanitize
        allow_html : bool, optional
            Whether to allow HTML tags, by default False
        max_length : int, optional
            Maximum allowed length, uses default if None
            
        Returns
        -------
        str
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)
        
        # Limit length
        max_len = max_length or self.max_string_length
        if len(value) > max_len:
            logger.warning(f"String truncated from {len(value)} to {max_len} characters")
            value = value[:max_len]
        
        # Remove null bytes and control characters
        value = value.replace('\x00', '').replace('\r\n', '\n')
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n')
        
        # HTML escape if not allowing HTML
        if not allow_html:
            value = html.escape(value, quote=True)
        
        # Strip leading/trailing whitespace
        value = value.strip()
        
        return value
    
    def sanitize_dict(self, data: Dict[str, Any], max_depth: int = 10) -> Dict[str, Any]:
        """Recursively sanitize dictionary data.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary to sanitize
        max_depth : int, optional
            Maximum recursion depth, by default 10
            
        Returns
        -------
        Dict[str, Any]
            Sanitized dictionary
        """
        if max_depth <= 0:
            logger.warning("Maximum recursion depth reached during sanitization")
            return {}
        
        if not isinstance(data, dict):
            return {}
        
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            if not isinstance(key, str):
                key = str(key)
            
            # Skip private/dangerous keys
            if key.startswith('_') or key.startswith('__'):
                logger.debug(f"Skipping private key: {key}")
                continue
            
            clean_key = self.sanitize_string(key, max_length=100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                clean_value = self.sanitize_string(value)
            elif isinstance(value, dict):
                clean_value = self.sanitize_dict(value, max_depth - 1)
            elif isinstance(value, list):
                clean_value = self.sanitize_list(value, max_depth - 1)
            elif isinstance(value, (int, float, bool, type(None))):
                clean_value = value
            else:
                # Convert unknown types to string and sanitize
                clean_value = self.sanitize_string(str(value))
            
            if clean_key:  # Only add non-empty keys
                sanitized[clean_key] = clean_value
        
        return sanitized
    
    def sanitize_list(self, data: List[Any], max_depth: int = 10) -> List[Any]:
        """Recursively sanitize list data.
        
        Parameters
        ----------
        data : List[Any]
            List to sanitize
        max_depth : int, optional
            Maximum recursion depth, by default 10
            
        Returns
        -------
        List[Any]
            Sanitized list
        """
        if max_depth <= 0:
            logger.warning("Maximum recursion depth reached during list sanitization")
            return []
        
        if not isinstance(data, list):
            return []
        
        # Limit list size
        max_items = 1000
        if len(data) > max_items:
            logger.warning(f"List truncated from {len(data)} to {max_items} items")
            data = data[:max_items]
        
        sanitized = []
        
        for item in data:
            if isinstance(item, str):
                clean_item = self.sanitize_string(item)
            elif isinstance(item, dict):
                clean_item = self.sanitize_dict(item, max_depth - 1)
            elif isinstance(item, list):
                clean_item = self.sanitize_list(item, max_depth - 1)
            elif isinstance(item, (int, float, bool, type(None))):
                clean_item = item
            else:
                clean_item = self.sanitize_string(str(item))
            
            sanitized.append(clean_item)
        
        return sanitized
    
    def validate_json(self, json_str: str, max_size: int = 1024 * 1024) -> Optional[Dict[str, Any]]:
        """Validate and parse JSON safely.
        
        Parameters
        ----------
        json_str : str
            JSON string to validate
        max_size : int, optional
            Maximum JSON size in bytes, by default 1MB
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Parsed JSON data or None if invalid
        """
        if not isinstance(json_str, str):
            return None
        
        # Check size
        if len(json_str.encode('utf-8')) > max_size:
            logger.warning(f"JSON too large: {len(json_str)} bytes > {max_size}")
            return None
        
        try:
            data = json.loads(json_str)
            return self.sanitize_dict(data) if isinstance(data, dict) else None
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Invalid JSON: {e}")
            return None
    
    def is_safe_path(self, path: str, allowed_dirs: Optional[List[str]] = None) -> bool:
        """Check if file path is safe (no traversal attacks).
        
        Parameters
        ----------
        path : str
            File path to validate
        allowed_dirs : List[str], optional
            List of allowed base directories
            
        Returns
        -------
        bool
            True if path is safe
        """
        if not isinstance(path, str):
            return False
        
        # Check for dangerous patterns
        if self.dangerous_patterns['path_traversal'].search(path):
            logger.warning(f"Path traversal detected in: {path}")
            return False
        
        try:
            # Resolve path to prevent traversal
            resolved_path = Path(path).resolve()
            path_str = str(resolved_path)
        except (OSError, ValueError):
            logger.warning(f"Invalid path: {path}")
            return False
        
        # Check against allowed directories
        if allowed_dirs:
            allowed = False
            for allowed_dir in allowed_dirs:
                try:
                    allowed_resolved = Path(allowed_dir).resolve()
                    if resolved_path.is_relative_to(allowed_resolved):
                        allowed = True
                        break
                except (OSError, ValueError):
                    continue
            
            if not allowed:
                logger.warning(f"Path not in allowed directories: {path}")
                return False
        
        # Check for suspicious paths
        suspicious_dirs = {
            '/etc', '/proc', '/sys', '/root', '/home',
            'C:\\Windows', 'C:\\System32', 'C:\\Users'
        }
        
        for sus_dir in suspicious_dirs:
            if path_str.startswith(sus_dir):
                logger.warning(f"Access to suspicious directory: {path}")
                return False
        
        return True
    
    def is_safe_url(self, url: str, allowed_schemes: Optional[List[str]] = None) -> bool:
        """Check if URL is safe.
        
        Parameters
        ----------
        url : str
            URL to validate
        allowed_schemes : List[str], optional
            Allowed URL schemes, defaults to ['http', 'https']
            
        Returns
        -------
        bool
            True if URL is safe
        """
        if not isinstance(url, str):
            return False
        
        if len(url) > 2000:  # URLs should not be too long
            return False
        
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        
        # Check scheme
        allowed = allowed_schemes or ['http', 'https']
        if parsed.scheme.lower() not in allowed:
            return False
        
        # Check for dangerous URLs
        dangerous_hosts = {
            'localhost', '127.0.0.1', '0.0.0.0',
            '::1', '169.254.169.254'  # AWS metadata service
        }
        
        if parsed.hostname and parsed.hostname.lower() in dangerous_hosts:
            logger.warning(f"Dangerous hostname in URL: {parsed.hostname}")
            return False
        
        return True
    
    def detect_injection_attempts(self, value: str) -> List[str]:
        """Detect potential injection attempts in input.
        
        Parameters
        ----------
        value : str
            Input to check
            
        Returns
        -------
        List[str]
            List of detected attack types
        """
        if not isinstance(value, str):
            return []
        
        detected = []
        
        for attack_type, pattern in self.dangerous_patterns.items():
            if pattern.search(value):
                detected.append(attack_type)
                logger.warning(f"Potential {attack_type} detected in input")
        
        return detected
    
    def validate_filename(self, filename: str) -> bool:
        """Validate filename for safety.
        
        Parameters
        ----------
        filename : str
            Filename to validate
            
        Returns
        -------
        bool
            True if filename is safe
        """
        if not isinstance(filename, str):
            return False
        
        # Length check
        if len(filename) > 255:
            return False
        
        # Character whitelist
        allowed_pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
        if not allowed_pattern.match(filename):
            return False
        
        # Dangerous filenames
        dangerous_names = {
            'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3',
            'lpt1', 'lpt2', 'lpt3', '.htaccess', 'web.config'
        }
        
        if filename.lower() in dangerous_names:
            return False
        
        # No leading/trailing dots or spaces
        if filename.startswith('.') or filename.endswith('.') or filename.startswith(' ') or filename.endswith(' '):
            return False
        
        return True
    
    def sanitize_sql_identifier(self, identifier: str) -> Optional[str]:
        """Sanitize SQL identifier (table/column name).
        
        Parameters
        ----------
        identifier : str
            SQL identifier to sanitize
            
        Returns
        -------
        Optional[str]
            Sanitized identifier or None if invalid
        """
        if not isinstance(identifier, str):
            return None
        
        # SQL identifiers should only contain alphanumeric and underscore
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            return None
        
        # Limit length
        if len(identifier) > 64:
            return None
        
        # Check against SQL keywords
        sql_keywords = {
            'select', 'insert', 'update', 'delete', 'drop', 'create',
            'alter', 'table', 'index', 'view', 'database', 'schema',
            'from', 'where', 'order', 'group', 'having', 'union',
            'join', 'inner', 'outer', 'left', 'right', 'on'
        }
        
        if identifier.lower() in sql_keywords:
            return None
        
        return identifier
    
    def validate_email(self, email: str) -> bool:
        """Validate email address format.
        
        Parameters
        ----------
        email : str
            Email address to validate
            
        Returns
        -------
        bool
            True if email format is valid
        """
        if not isinstance(email, str):
            return False
        
        # Length check
        if len(email) > 254:
            return False
        
        # Basic regex for email validation
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        return bool(email_pattern.match(email))
    
    def validate_numeric_range(
        self,
        value: Union[int, float],
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None
    ) -> bool:
        """Validate numeric value is within range.
        
        Parameters
        ----------
        value : Union[int, float]
            Value to validate
        min_val : Union[int, float], optional
            Minimum allowed value
        max_val : Union[int, float], optional
            Maximum allowed value
            
        Returns
        -------
        bool
            True if value is within range
        """
        if not isinstance(value, (int, float)):
            return False
        
        if min_val is not None and value < min_val:
            return False
        
        if max_val is not None and value > max_val:
            return False
        
        return True
    
    def create_security_headers(self) -> Dict[str, str]:
        """Create security headers for HTTP responses.
        
        Returns
        -------
        Dict[str, str]
            Dictionary of security headers
        """
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            ),
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=()"
            )
        }