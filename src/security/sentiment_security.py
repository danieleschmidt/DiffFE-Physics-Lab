"""Security module for sentiment analysis API and services."""

import hashlib
import hmac
import time
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import threading
from pathlib import Path


@dataclass
class SecurityThreat:
    """Security threat detection result."""
    
    threat_type: str
    severity: str  # low, medium, high, critical
    description: str
    source_ip: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'threat_type': self.threat_type,
            'severity': self.severity,
            'description': self.description,
            'source_ip': self.source_ip,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class TextInjectionDetector:
    """Detector for malicious text injection attacks."""
    
    def __init__(self):
        """Initialize text injection detector."""
        # Common injection patterns
        self.sql_patterns = [
            r'(?i)(union\s+select|select\s+.*\s+from|drop\s+table|delete\s+from)',
            r'(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)',
            r'(?i)(exec\s*\(|xp_cmdshell|sp_executesql)',
            r'(?i)(insert\s+into|update\s+.*\s+set)'
        ]
        
        self.xss_patterns = [
            r'(?i)<script[^>]*>.*?</script>',
            r'(?i)javascript:',
            r'(?i)on\w+\s*=\s*["\'][^"\']*["\']',
            r'(?i)<iframe[^>]*>.*?</iframe>',
            r'(?i)eval\s*\(',
            r'(?i)document\.cookie',
            r'(?i)alert\s*\(',
        ]
        
        self.command_patterns = [
            r'(?:^|\s)(?:cat|ls|pwd|whoami|id|uname)\s',
            r'(?:^|\s)(?:curl|wget|nc|netcat)\s',
            r'(?:^|\s)(?:rm|mv|cp|chmod|sudo)\s',
            r'(?:[;&|]|\|\||&&).*(?:cat|ls|pwd)',
            r'`[^`]*`',  # Command substitution
            r'\$\([^)]*\)',  # Command substitution
        ]
        
        self.path_traversal_patterns = [
            r'\.\./',
            r'\.\.\\',
            r'/etc/passwd',
            r'/etc/shadow',
            r'c:\\windows\\system32',
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = {
            'sql': [re.compile(p) for p in self.sql_patterns],
            'xss': [re.compile(p) for p in self.xss_patterns],
            'command': [re.compile(p) for p in self.command_patterns],
            'path_traversal': [re.compile(p) for p in self.path_traversal_patterns]
        }
        
    def detect_injection(self, text: str) -> List[SecurityThreat]:
        """Detect injection attacks in text.
        
        Parameters
        ----------
        text : str
            Text to analyze for injection attacks
            
        Returns
        -------
        List[SecurityThreat]
            List of detected threats
        """
        threats = []
        
        # Check for SQL injection
        for pattern in self.compiled_patterns['sql']:
            if pattern.search(text):
                threats.append(SecurityThreat(
                    threat_type='sql_injection',
                    severity='high',
                    description=f'SQL injection pattern detected: {pattern.pattern}',
                    metadata={'matched_pattern': pattern.pattern, 'text_sample': text[:100]}
                ))
                
        # Check for XSS
        for pattern in self.compiled_patterns['xss']:
            if pattern.search(text):
                threats.append(SecurityThreat(
                    threat_type='xss_injection',
                    severity='high',
                    description=f'XSS injection pattern detected: {pattern.pattern}',
                    metadata={'matched_pattern': pattern.pattern, 'text_sample': text[:100]}
                ))
                
        # Check for command injection
        for pattern in self.compiled_patterns['command']:
            if pattern.search(text):
                threats.append(SecurityThreat(
                    threat_type='command_injection',
                    severity='critical',
                    description=f'Command injection pattern detected: {pattern.pattern}',
                    metadata={'matched_pattern': pattern.pattern, 'text_sample': text[:100]}
                ))
                
        # Check for path traversal
        for pattern in self.compiled_patterns['path_traversal']:
            if pattern.search(text):
                threats.append(SecurityThreat(
                    threat_type='path_traversal',
                    severity='medium',
                    description=f'Path traversal pattern detected: {pattern.pattern}',
                    metadata={'matched_pattern': pattern.pattern, 'text_sample': text[:100]}
                ))
                
        return threats
        
    def sanitize_text(self, text: str) -> str:
        """Sanitize text by removing potentially malicious content.
        
        Parameters
        ----------
        text : str
            Text to sanitize
            
        Returns
        -------
        str
            Sanitized text
        """
        # Remove script tags
        text = re.sub(r'(?i)<script[^>]*>.*?</script>', '', text)
        
        # Remove iframe tags
        text = re.sub(r'(?i)<iframe[^>]*>.*?</iframe>', '', text)
        
        # Remove javascript: URLs
        text = re.sub(r'(?i)javascript:[^"\'\s]*', '', text)
        
        # Remove event handlers
        text = re.sub(r'(?i)on\w+\s*=\s*["\'][^"\']*["\']', '', text)
        
        # Remove path traversal sequences
        text = re.sub(r'\.\./', '', text)
        text = re.sub(r'\.\.\\', '', text)
        
        # Remove command substitution
        text = re.sub(r'`[^`]*`', '', text)
        text = re.sub(r'\$\([^)]*\)', '', text)
        
        return text.strip()


class RateLimitManager:
    """Advanced rate limiting for API requests."""
    
    def __init__(self):
        """Initialize rate limit manager."""
        self.request_history = defaultdict(deque)  # IP -> deque of timestamps
        self.blocked_ips = set()
        self.block_duration = 3600  # 1 hour
        self.ip_blocks = {}  # IP -> block_timestamp
        self._lock = threading.RLock()
        
        # Rate limit tiers
        self.rate_limits = {
            'analyze': {'requests': 60, 'window': 60},      # 60 requests per minute
            'train': {'requests': 10, 'window': 60},        # 10 requests per minute
            'batch': {'requests': 30, 'window': 60},        # 30 requests per minute
            'explain': {'requests': 100, 'window': 60},     # 100 requests per minute
            'benchmark': {'requests': 5, 'window': 60},     # 5 requests per minute
        }
        
    def is_rate_limited(self, ip: str, endpoint: str) -> Tuple[bool, Optional[SecurityThreat]]:
        """Check if IP is rate limited for endpoint.
        
        Parameters
        ----------
        ip : str
            Client IP address
        endpoint : str
            API endpoint name
            
        Returns
        -------
        Tuple[bool, Optional[SecurityThreat]]
            (is_limited, threat_info)
        """
        with self._lock:
            current_time = time.time()
            
            # Check if IP is blocked
            if ip in self.blocked_ips:
                block_time = self.ip_blocks.get(ip, 0)
                if current_time - block_time < self.block_duration:
                    threat = SecurityThreat(
                        threat_type='blocked_ip',
                        severity='high',
                        description=f'Blocked IP {ip} attempting access',
                        source_ip=ip,
                        metadata={'block_remaining': self.block_duration - (current_time - block_time)}
                    )
                    return True, threat
                else:
                    # Unblock expired IPs
                    self.blocked_ips.discard(ip)
                    if ip in self.ip_blocks:
                        del self.ip_blocks[ip]
                        
            # Get rate limit for endpoint
            limit_config = self.rate_limits.get(endpoint, self.rate_limits['analyze'])
            max_requests = limit_config['requests']
            time_window = limit_config['window']
            
            # Clean old requests
            requests = self.request_history[ip]
            while requests and current_time - requests[0] > time_window:
                requests.popleft()
                
            # Check rate limit
            if len(requests) >= max_requests:
                # Rate limit exceeded
                threat = SecurityThreat(
                    threat_type='rate_limit_exceeded',
                    severity='medium',
                    description=f'Rate limit exceeded for {ip} on {endpoint}',
                    source_ip=ip,
                    metadata={
                        'endpoint': endpoint,
                        'request_count': len(requests),
                        'limit': max_requests,
                        'window': time_window
                    }
                )
                
                # Check for potential abuse (multiple rate limit violations)
                if len(requests) > max_requests * 2:
                    self.blocked_ips.add(ip)
                    self.ip_blocks[ip] = current_time
                    threat.severity = 'high'
                    threat.description += ' - IP blocked for abuse'
                    
                return True, threat
                
            # Record request
            requests.append(current_time)
            return False, None
            
    def get_rate_limit_status(self, ip: str, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit status for IP and endpoint.
        
        Parameters
        ----------
        ip : str
            Client IP address
        endpoint : str
            API endpoint name
            
        Returns
        -------
        Dict[str, Any]
            Rate limit status information
        """
        with self._lock:
            current_time = time.time()
            limit_config = self.rate_limits.get(endpoint, self.rate_limits['analyze'])
            
            # Clean old requests
            requests = self.request_history[ip]
            while requests and current_time - requests[0] > limit_config['window']:
                requests.popleft()
                
            return {
                'requests_made': len(requests),
                'requests_limit': limit_config['requests'],
                'time_window': limit_config['window'],
                'requests_remaining': max(0, limit_config['requests'] - len(requests)),
                'reset_time': current_time + limit_config['window'] if requests else current_time,
                'is_blocked': ip in self.blocked_ips
            }


class ContentSecurityValidator:
    """Validator for content security policies."""
    
    def __init__(self):
        """Initialize content security validator."""
        self.max_text_length = 10000
        self.max_batch_size = 1000
        self.max_embedding_dim = 2000
        self.allowed_embedding_methods = {'tfidf', 'word2vec', 'bert'}
        self.allowed_backends = {'jax', 'torch', 'numpy'}
        
        # Suspicious content patterns
        self.suspicious_patterns = [
            r'(?i)password\s*[:=]\s*[\'""][^\'"]{8,}[\'\""]',  # Password in text
            r'(?i)api[_-]?key\s*[:=]\s*[\'""][^\'"]{16,}[\'\""]',  # API key
            r'(?i)secret\s*[:=]\s*[\'""][^\'"]{8,}[\'\""]',  # Secret
            r'(?i)token\s*[:=]\s*[\'""][^\'"]{20,}[\'\""]',  # Token
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card pattern
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        ]
        
        self.compiled_suspicious = [re.compile(p) for p in self.suspicious_patterns]
        
    def validate_request(self, request_data: Dict[str, Any]) -> List[SecurityThreat]:
        """Validate API request for security issues.
        
        Parameters
        ----------
        request_data : Dict[str, Any]
            Request data to validate
            
        Returns
        -------
        List[SecurityThreat]
            List of security threats found
        """
        threats = []
        
        # Validate text content
        texts = request_data.get('texts', [])
        if isinstance(texts, list):
            for i, text in enumerate(texts):
                if isinstance(text, str):
                    # Check text length
                    if len(text) > self.max_text_length:
                        threats.append(SecurityThreat(
                            threat_type='oversized_content',
                            severity='medium',
                            description=f'Text {i} exceeds maximum length ({len(text)} > {self.max_text_length})',
                            metadata={'text_index': i, 'text_length': len(text)}
                        ))
                        
                    # Check for suspicious content
                    for pattern in self.compiled_suspicious:
                        if pattern.search(text):
                            threats.append(SecurityThreat(
                                threat_type='suspicious_content',
                                severity='high',
                                description=f'Suspicious content detected in text {i}: {pattern.pattern}',
                                metadata={'text_index': i, 'pattern': pattern.pattern}
                            ))
                            
        # Validate batch size
        if len(texts) > self.max_batch_size:
            threats.append(SecurityThreat(
                threat_type='oversized_batch',
                severity='medium',
                description=f'Batch size exceeds limit ({len(texts)} > {self.max_batch_size})',
                metadata={'batch_size': len(texts)}
            ))
            
        # Validate options
        options = request_data.get('options', {})
        if isinstance(options, dict):
            # Check embedding method
            embedding_method = options.get('embedding_method')
            if embedding_method and embedding_method not in self.allowed_embedding_methods:
                threats.append(SecurityThreat(
                    threat_type='invalid_parameter',
                    severity='low',
                    description=f'Invalid embedding method: {embedding_method}',
                    metadata={'parameter': 'embedding_method', 'value': embedding_method}
                ))
                
            # Check backend
            backend = options.get('backend')
            if backend and backend not in self.allowed_backends:
                threats.append(SecurityThreat(
                    threat_type='invalid_parameter',
                    severity='low',
                    description=f'Invalid backend: {backend}',
                    metadata={'parameter': 'backend', 'value': backend}
                ))
                
            # Check embedding dimension
            embedding_dim = options.get('embedding_dim')
            if embedding_dim and isinstance(embedding_dim, (int, float)):
                if embedding_dim > self.max_embedding_dim or embedding_dim < 1:
                    threats.append(SecurityThreat(
                        threat_type='invalid_parameter',
                        severity='medium',
                        description=f'Invalid embedding dimension: {embedding_dim}',
                        metadata={'parameter': 'embedding_dim', 'value': embedding_dim}
                    ))
                    
        return threats


class APISecurityManager:
    """Comprehensive security manager for sentiment analysis API."""
    
    def __init__(self, secret_key: str = None):
        """Initialize API security manager.
        
        Parameters
        ----------
        secret_key : str, optional
            Secret key for HMAC signatures
        """
        self.secret_key = secret_key or 'default-secret-key'
        self.injection_detector = TextInjectionDetector()
        self.rate_limiter = RateLimitManager()
        self.content_validator = ContentSecurityValidator()
        self.threat_log = []
        self._lock = threading.RLock()
        
    def validate_request_security(
        self,
        request_data: Dict[str, Any],
        client_ip: str,
        endpoint: str,
        headers: Dict[str, str] = None
    ) -> Tuple[bool, List[SecurityThreat]]:
        """Comprehensive request security validation.
        
        Parameters
        ----------
        request_data : Dict[str, Any]
            Request data to validate
        client_ip : str
            Client IP address
        endpoint : str
            API endpoint name
        headers : Dict[str, str], optional
            Request headers
            
        Returns
        -------
        Tuple[bool, List[SecurityThreat]]
            (is_valid, list of threats)
        """
        all_threats = []
        
        # Rate limiting check
        is_limited, rate_threat = self.rate_limiter.is_rate_limited(client_ip, endpoint)
        if is_limited and rate_threat:
            all_threats.append(rate_threat)
            
        # Content validation
        content_threats = self.content_validator.validate_request(request_data)
        all_threats.extend(content_threats)
        
        # Text injection detection
        texts = request_data.get('texts', [])
        if isinstance(texts, list):
            for text in texts:
                if isinstance(text, str):
                    injection_threats = self.injection_detector.detect_injection(text)
                    for threat in injection_threats:
                        threat.source_ip = client_ip
                    all_threats.extend(injection_threats)
                    
        # HMAC signature validation (if headers provided)
        if headers and 'X-Signature' in headers:
            signature_valid = self.validate_hmac_signature(
                request_data, headers['X-Signature']
            )
            if not signature_valid:
                all_threats.append(SecurityThreat(
                    threat_type='invalid_signature',
                    severity='high',
                    description='Invalid HMAC signature',
                    source_ip=client_ip,
                    metadata={'endpoint': endpoint}
                ))
                
        # Log threats
        with self._lock:
            self.threat_log.extend(all_threats)
            
        # Determine if request should be allowed
        critical_threats = [t for t in all_threats if t.severity in ['high', 'critical']]
        is_valid = len(critical_threats) == 0 and not is_limited
        
        return is_valid, all_threats
        
    def sanitize_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data to remove malicious content.
        
        Parameters
        ----------
        request_data : Dict[str, Any]
            Request data to sanitize
            
        Returns
        -------
        Dict[str, Any]
            Sanitized request data
        """
        sanitized = request_data.copy()
        
        # Sanitize text content
        texts = sanitized.get('texts', [])
        if isinstance(texts, list):
            sanitized_texts = []
            for text in texts:
                if isinstance(text, str):
                    sanitized_text = self.injection_detector.sanitize_text(text)
                    sanitized_texts.append(sanitized_text)
                else:
                    sanitized_texts.append(text)
            sanitized['texts'] = sanitized_texts
            
        return sanitized
        
    def generate_hmac_signature(self, data: Dict[str, Any]) -> str:
        """Generate HMAC signature for data.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data to sign
            
        Returns
        -------
        str
            HMAC signature
        """
        data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        signature = hmac.new(
            self.secret_key.encode(),
            data_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    def validate_hmac_signature(self, data: Dict[str, Any], signature: str) -> bool:
        """Validate HMAC signature for data.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Data to validate
        signature : str
            Provided signature
            
        Returns
        -------
        bool
            True if signature is valid
        """
        expected_signature = self.generate_hmac_signature(data)
        return hmac.compare_digest(signature, expected_signature)
        
    def get_threat_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get threat summary for specified time period.
        
        Parameters
        ----------
        hours : int, optional
            Hours to look back, by default 24
            
        Returns
        -------
        Dict[str, Any]
            Threat summary statistics
        """
        with self._lock:
            cutoff_time = time.time() - (hours * 3600)
            recent_threats = [t for t in self.threat_log if t.timestamp > cutoff_time]
            
            # Categorize threats
            threats_by_type = defaultdict(int)
            threats_by_severity = defaultdict(int)
            threats_by_ip = defaultdict(int)
            
            for threat in recent_threats:
                threats_by_type[threat.threat_type] += 1
                threats_by_severity[threat.severity] += 1
                if threat.source_ip:
                    threats_by_ip[threat.source_ip] += 1
                    
            return {
                'total_threats': len(recent_threats),
                'time_period_hours': hours,
                'threats_by_type': dict(threats_by_type),
                'threats_by_severity': dict(threats_by_severity),
                'top_threat_sources': dict(sorted(
                    threats_by_ip.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                'most_common_threat': max(threats_by_type.items(), key=lambda x: x[1])[0] if threats_by_type else None
            }
            
    def export_threat_log(self, filepath: Path):
        """Export threat log to file.
        
        Parameters
        ----------
        filepath : Path
            File path to export to
        """
        with self._lock:
            threat_data = {
                'export_timestamp': time.time(),
                'total_threats': len(self.threat_log),
                'threats': [threat.to_dict() for threat in self.threat_log]
            }
            
            with open(filepath, 'w') as f:
                json.dump(threat_data, f, indent=2)


# Security decorator for API endpoints
def require_security_validation(security_manager: APISecurityManager):
    """Decorator to require security validation for API endpoints.
    
    Parameters
    ----------
    security_manager : APISecurityManager
        Security manager instance
        
    Returns
    -------
    Callable
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be integrated with Flask request context
            # For demonstration purposes, we'll show the pattern
            
            # Extract request info (would come from Flask in practice)
            request_data = kwargs.get('request_data', {})
            client_ip = kwargs.get('client_ip', '127.0.0.1')
            endpoint = func.__name__
            headers = kwargs.get('headers', {})
            
            # Validate security
            is_valid, threats = security_manager.validate_request_security(
                request_data, client_ip, endpoint, headers
            )
            
            if not is_valid:
                # Return security error (would be proper HTTP response in practice)
                return {
                    'success': False,
                    'error': 'Security validation failed',
                    'threats': [threat.to_dict() for threat in threats]
                }, 403
                
            # Sanitize request data
            sanitized_data = security_manager.sanitize_request_data(request_data)
            kwargs['request_data'] = sanitized_data
            
            # Call original function
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


# Global security manager instance
_global_security_manager = None


def get_global_security_manager() -> APISecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    
    if _global_security_manager is None:
        _global_security_manager = APISecurityManager()
        
    return _global_security_manager