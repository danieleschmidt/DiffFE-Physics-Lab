"""Security validation and protection for DiffFE-Physics-Lab."""

import os
import re
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: Optional[str] = None
    permissions: List[str] = None
    session_id: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.timestamp is None:
            self.timestamp = time.time()


class SecurityValidator:
    """Comprehensive security validation system."""
    
    def __init__(self):
        """Initialize security validator."""
        self.blocked_patterns = [
            r'\.\./',  # Path traversal
            r'[;&|`$]',  # Command injection
            r'<script',  # XSS
            r'union\s+select',  # SQL injection
            r'exec\s*\(',  # Code execution
            r'eval\s*\(',  # Code evaluation
            r'import\s+os',  # Dangerous imports
            r'__import__',  # Dynamic imports
        ]
        self.max_input_size = 1024 * 1024  # 1MB
        self.allowed_file_extensions = {'.py', '.txt', '.json', '.yaml', '.yml', '.md'}
        self.blocked_file_extensions = {'.exe', '.bat', '.sh', '.cmd', '.ps1'}
        
        logger.info("Security validator initialized")
    
    def validate_input(self, input_data: Any, name: str = "input") -> bool:
        """Validate input data for security threats.
        
        Args:
            input_data: Data to validate
            name: Name of the input for logging
            
        Returns:
            True if input is safe
            
        Raises:
            SecurityError: If input contains threats
        """
        # Convert to string for pattern matching
        if isinstance(input_data, (str, bytes)):
            text = str(input_data)
        else:
            text = str(input_data)
        
        # Check input size
        if len(text) > self.max_input_size:
            raise SecurityError(f"Input {name} exceeds maximum size limit", 
                              input_size=len(text), max_size=self.max_input_size)
        
        # Check for malicious patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Blocked pattern '{pattern}' found in {name}")
                raise SecurityError(f"Malicious pattern detected in {name}", 
                                  pattern=pattern, input_name=name)
        
        logger.debug(f"Input validation passed for {name}")
        return True
    
    def validate_file_path(self, file_path: Union[str, Path]) -> bool:
        """Validate file path for security.
        
        Args:
            file_path: File path to validate
            
        Returns:
            True if path is safe
            
        Raises:
            SecurityError: If path is unsafe
        """
        path = Path(file_path).resolve()
        
        # Check for path traversal
        if '..' in str(path):
            raise SecurityError(f"Path traversal detected in {file_path}")
        
        # Check file extension
        if path.suffix.lower() in self.blocked_file_extensions:
            raise SecurityError(f"Blocked file extension: {path.suffix}")
        
        # Check if path is within allowed directories
        cwd = Path.cwd().resolve()
        try:
            path.relative_to(cwd)
        except ValueError:
            raise SecurityError(f"Path outside allowed directory: {path}")
        
        logger.debug(f"File path validation passed for {path}")
        return True
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameter dictionary.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if parameters are safe
        """
        for key, value in params.items():
            # Validate key
            self.validate_input(key, f"parameter_key_{key}")
            
            # Validate value
            if isinstance(value, (str, int, float, bool)):
                self.validate_input(value, f"parameter_{key}")
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    self.validate_input(item, f"parameter_{key}[{i}]")
            elif isinstance(value, dict):
                self.validate_parameters(value)
        
        logger.debug(f"Parameter validation passed for {len(params)} parameters")
        return True


class InputSanitizer:
    """Input sanitization and normalization."""
    
    def __init__(self):
        """Initialize input sanitizer."""
        self.html_escape_map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
        }
    
    def sanitize_string(self, text: str) -> str:
        """Sanitize string input.
        
        Args:
            text: String to sanitize
            
        Returns:
            Sanitized string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Limit length
        max_length = 10000
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"String truncated to {max_length} characters")
        
        # Escape HTML characters
        for char, escape in self.html_escape_map.items():
            text = text.replace(char, escape)
        
        return text
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure it's not empty
        if not filename:
            filename = 'unnamed_file'
        
        return filename
    
    def sanitize_numeric(self, value: Any, min_val: Optional[float] = None,
                        max_val: Optional[float] = None) -> float:
        """Sanitize numeric input.
        
        Args:
            value: Value to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Sanitized numeric value
        """
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid numeric value: {value}, using 0.0")
            return 0.0
        
        # Check for special values
        if not (-float('inf') < num_value < float('inf')):
            logger.warning(f"Invalid numeric value: {num_value}, using 0.0")
            return 0.0
        
        # Apply bounds
        if min_val is not None and num_value < min_val:
            logger.warning(f"Value {num_value} below minimum {min_val}, clamping")
            num_value = min_val
        
        if max_val is not None and num_value > max_val:
            logger.warning(f"Value {num_value} above maximum {max_val}, clamping")
            num_value = max_val
        
        return num_value


class PermissionChecker:
    """Permission and access control system."""
    
    def __init__(self):
        """Initialize permission checker."""
        self.permissions = {}
        self.roles = {}
        self.sessions = {}
        
    def define_permission(self, name: str, description: str = ""):
        """Define a new permission.
        
        Args:
            name: Permission name
            description: Permission description
        """
        self.permissions[name] = description
        logger.info(f"Defined permission: {name}")
    
    def define_role(self, role_name: str, permissions: List[str]):
        """Define a role with permissions.
        
        Args:
            role_name: Role name
            permissions: List of permission names
        """
        # Validate permissions exist
        for perm in permissions:
            if perm not in self.permissions:
                raise SecurityError(f"Unknown permission: {perm}")
        
        self.roles[role_name] = permissions
        logger.info(f"Defined role '{role_name}' with {len(permissions)} permissions")
    
    def create_session(self, user_id: str, roles: List[str]) -> str:
        """Create a user session.
        
        Args:
            user_id: User identifier
            roles: List of role names
            
        Returns:
            Session ID
        """
        # Validate roles exist
        for role in roles:
            if role not in self.roles:
                raise SecurityError(f"Unknown role: {role}")
        
        # Collect all permissions from roles
        user_permissions = set()
        for role in roles:
            user_permissions.update(self.roles[role])
        
        # Generate session ID
        session_data = f"{user_id}:{time.time()}:{hash(tuple(sorted(user_permissions)))}"
        session_id = hashlib.sha256(session_data.encode()).hexdigest()[:16]
        
        # Store session
        self.sessions[session_id] = {
            "user_id": user_id,
            "roles": roles,
            "permissions": list(user_permissions),
            "created_at": time.time(),
            "last_used": time.time()
        }
        
        logger.info(f"Created session for user {user_id} with {len(user_permissions)} permissions")
        return session_id
    
    def check_permission(self, session_id: str, permission: str) -> bool:
        """Check if session has permission.
        
        Args:
            session_id: Session identifier
            permission: Permission to check
            
        Returns:
            True if permission granted
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        # Update last used time
        session["last_used"] = time.time()
        
        has_permission = permission in session["permissions"]
        
        if not has_permission:
            logger.warning(f"Permission denied: {permission} for session {session_id}")
        
        return has_permission
    
    def cleanup_expired_sessions(self, max_age: float = 3600.0):
        """Clean up expired sessions.
        
        Args:
            max_age: Maximum session age in seconds
        """
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_used"] > max_age:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class SecurityError(Exception):
    """Security-related error."""
    
    def __init__(self, message: str, **context):
        """Initialize security error.
        
        Args:
            message: Error message
            **context: Additional context
        """
        super().__init__(message)
        self.context = context
        logger.error(f"Security error: {message}")


# Global instances
global_security_validator = SecurityValidator()
global_input_sanitizer = InputSanitizer()
global_permission_checker = PermissionChecker()


def require_permission(permission: str):
    """Decorator to require permission for function execution.
    
    Args:
        permission: Required permission name
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Look for security context in kwargs
            security_context = kwargs.pop('_security_context', None)
            
            if security_context and hasattr(security_context, 'session_id'):
                if not global_permission_checker.check_permission(
                    security_context.session_id, permission
                ):
                    raise SecurityError(f"Permission required: {permission}")
            else:
                logger.warning(f"No security context provided for {func.__name__}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_inputs(**validators):
    """Decorator to validate function inputs.
    
    Args:
        **validators: Mapping of parameter names to validator functions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        if not validator(value):
                            raise SecurityError(f"Validation failed for parameter {param_name}")
                    except Exception as e:
                        raise SecurityError(f"Validation error for {param_name}: {e}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def secure_mode(security_context: SecurityContext):
    """Context manager for secure operations.
    
    Args:
        security_context: Security context for the operation
    """
    logger.debug(f"Entering secure mode for user {security_context.user_id}")
    
    try:
        yield security_context
    except SecurityError:
        logger.error("Security violation in secure mode")
        raise
    except Exception as e:
        logger.error(f"Error in secure mode: {e}")
        raise
    finally:
        logger.debug("Exiting secure mode")


# Define default permissions
def setup_default_permissions():
    """Set up default permissions and roles."""
    # Define basic permissions
    global_permission_checker.define_permission("read", "Read access to data")
    global_permission_checker.define_permission("write", "Write access to data")
    global_permission_checker.define_permission("execute", "Execute operations")
    global_permission_checker.define_permission("admin", "Administrative access")
    
    # Define default roles
    global_permission_checker.define_role("viewer", ["read"])
    global_permission_checker.define_role("user", ["read", "write", "execute"])
    global_permission_checker.define_role("admin", ["read", "write", "execute", "admin"])
    
    logger.info("Default permissions and roles set up")


# Initialize default security setup
setup_default_permissions()