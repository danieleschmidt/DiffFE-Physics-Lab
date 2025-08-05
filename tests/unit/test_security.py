"""Unit tests for security module."""

import pytest
import time
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta

from src.security.scanner import SecurityScanner, SecurityIssue, Severity
from src.security.validator import SecurityValidator, ValidationResult
from src.security.monitor import SecurityMonitor, SecurityEvent, ThreatLevel, RateLimitRule


class TestSecurityScanner:
    """Test cases for SecurityScanner."""
    
    def test_scanner_init(self):
        """Test scanner initialization."""
        scanner = SecurityScanner()
        
        assert scanner.issues == []
        assert len(scanner.rules) > 0  # Should have built-in rules
    
    def test_scan_file_secrets(self):
        """Test scanning for hardcoded secrets."""
        scanner = SecurityScanner()
        
        # Mock file with secrets
        content = '''
api_key = "sk-1234567890abcdef"
password = "super_secret_password"
token = "ghp_abcdef123456"
normal_var = "this is fine"
'''
        
        with patch('builtins.open', mock_open(read_data=content)):
            issues = scanner.scan_file('test.py')
        
        secret_issues = [i for i in issues if i.issue_type == 'hardcoded_secret']
        assert len(secret_issues) >= 2  # Should find api_key and password
        
        # Check issue details
        api_key_issue = next((i for i in secret_issues if 'api_key' in i.message), None)
        assert api_key_issue is not None
        assert api_key_issue.severity == Severity.HIGH
    
    def test_scan_file_sql_injection(self):
        """Test scanning for SQL injection vulnerabilities."""
        scanner = SecurityScanner()
        
        content = '''
def bad_query(user_input):
    query = "SELECT * FROM users WHERE id = " + user_input
    return db.execute(query)

def good_query(user_input):
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (user_input,))
'''
        
        with patch('builtins.open', mock_open(read_data=content)):
            issues = scanner.scan_file('test.py')
        
        sql_issues = [i for i in issues if i.issue_type == 'sql_injection']
        assert len(sql_issues) >= 1
        
        sql_issue = sql_issues[0]
        assert 'user_input' in sql_issue.message
        assert sql_issue.severity == Severity.HIGH
    
    def test_scan_file_path_traversal(self):
        """Test scanning for path traversal vulnerabilities."""
        scanner = SecurityScanner()
        
        content = '''
def bad_file_access(filename):
    with open("/uploads/" + filename, 'r') as f:
        return f.read()

def better_file_access(filename):
    safe_path = os.path.join("/uploads", os.path.basename(filename))
    with open(safe_path, 'r') as f:
        return f.read()
'''
        
        with patch('builtins.open', mock_open(read_data=content)):
            issues = scanner.scan_file('test.py')
        
        path_issues = [i for i in issues if i.issue_type == 'path_traversal']
        assert len(path_issues) >= 1
        
        path_issue = path_issues[0]
        assert 'filename' in path_issue.message
        assert path_issue.severity in [Severity.HIGH, Severity.MEDIUM]
    
    def test_scan_directory(self):
        """Test scanning a directory."""
        scanner = SecurityScanner()
        
        # Mock file system
        files = ['file1.py', 'file2.py', 'README.md']
        
        with patch('os.listdir', return_value=files):
            with patch('os.path.isfile', return_value=True):
                with patch('builtins.open', mock_open(read_data='safe_code = "hello"')):
                    issues = scanner.scan_directory('/test/dir')
        
        # Should have scanned Python files but not README.md
        assert isinstance(issues, list)
    
    def test_add_custom_rule(self):
        """Test adding custom security rules."""
        scanner = SecurityScanner()
        initial_rule_count = len(scanner.rules)
        
        def custom_rule(content, filepath):
            issues = []
            if 'dangerous_function()' in content:
                issues.append(SecurityIssue(
                    issue_type='custom_danger',
                    severity=Severity.MEDIUM,
                    message='Dangerous function detected',
                    filepath=filepath,
                    line_number=1
                ))
            return issues
        
        scanner.add_rule(custom_rule)
        assert len(scanner.rules) == initial_rule_count + 1
        
        # Test the custom rule
        content = 'result = dangerous_function()'
        with patch('builtins.open', mock_open(read_data=content)):
            issues = scanner.scan_file('test.py')
        
        custom_issues = [i for i in issues if i.issue_type == 'custom_danger']
        assert len(custom_issues) == 1
    
    def test_generate_report(self):
        """Test generating security report."""
        scanner = SecurityScanner()
        
        # Add some mock issues
        scanner.issues = [
            SecurityIssue('secret', Severity.HIGH, 'API key found', 'file1.py', 10),
            SecurityIssue('sql_injection', Severity.HIGH, 'SQL injection risk', 'file2.py', 25),
            SecurityIssue('weak_crypto', Severity.MEDIUM, 'Weak encryption', 'file3.py', 15)
        ]
        
        report = scanner.generate_report()
        
        assert 'summary' in report
        assert 'issues_by_severity' in report
        assert 'issues_by_type' in report
        assert 'files_scanned' in report
        
        assert report['summary']['total_issues'] == 3
        assert report['summary']['high_severity'] == 2
        assert report['summary']['medium_severity'] == 1
        assert report['summary']['low_severity'] == 0


class TestSecurityValidator:
    """Test cases for SecurityValidator."""
    
    def test_validator_init(self):
        """Test validator initialization."""
        validator = SecurityValidator()
        assert validator is not None
    
    def test_validate_input_sql_injection(self):
        """Test SQL injection validation."""
        validator = SecurityValidator()
        
        # Safe input
        safe_input = "user123"
        result = validator.validate_input(safe_input, 'username')
        assert result.is_valid is True
        assert len(result.issues) == 0
        
        # Dangerous input
        dangerous_input = "'; DROP TABLE users; --"
        result = validator.validate_input(dangerous_input, 'username')
        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any('SQL injection' in issue for issue in result.issues)
    
    def test_validate_input_xss(self):
        """Test XSS validation."""
        validator = SecurityValidator()
        
        # Safe input
        safe_input = "Hello world"
        result = validator.validate_input(safe_input, 'comment')
        assert result.is_valid is True
        
        # XSS attempt
        xss_input = "<script>alert('xss')</script>"
        result = validator.validate_input(xss_input, 'comment')
        assert result.is_valid is False
        assert any('XSS' in issue or 'script' in issue for issue in result.issues)
    
    def test_validate_input_command_injection(self):
        """Test command injection validation."""
        validator = SecurityValidator()
        
        # Safe input
        safe_input = "filename.txt"
        result = validator.validate_input(safe_input, 'filename')
        assert result.is_valid is True
        
        # Command injection attempt
        dangerous_input = "file.txt; rm -rf /"
        result = validator.validate_input(dangerous_input, 'filename')
        assert result.is_valid is False
        assert len(result.issues) > 0
    
    def test_validate_file_upload(self):
        """Test file upload validation."""
        validator = SecurityValidator()
        
        # Valid file
        valid_result = validator.validate_file_upload('document.pdf', b'%PDF-1.4', 1024)
        assert valid_result.is_valid is True
        
        # Invalid extension
        invalid_ext_result = validator.validate_file_upload('script.exe', b'MZ\x90\x00', 1024)
        assert invalid_ext_result.is_valid is False
        
        # File too large
        large_file_result = validator.validate_file_upload('large.pdf', b'%PDF-1.4', 50 * 1024 * 1024)
        assert large_file_result.is_valid is False
        
        # Suspicious content
        suspicious_result = validator.validate_file_upload('doc.pdf', b'<script>alert()</script>', 1024)
        assert suspicious_result.is_valid is False
    
    def test_sanitize_input(self):
        """Test input sanitization."""
        validator = SecurityValidator()
        
        # HTML sanitization
        html_input = "<p>Hello <script>alert('xss')</script> world</p>"
        sanitized = validator.sanitize_input(html_input, 'html')
        assert '<script>' not in sanitized
        assert 'Hello' in sanitized
        assert 'world' in sanitized
        
        # SQL sanitization
        sql_input = "O'Reilly"
        sanitized_sql = validator.sanitize_input(sql_input, 'sql')
        assert "O''Reilly" in sanitized_sql or "O\\'Reilly" in sanitized_sql
        
        # Filename sanitization
        filename_input = "../../../etc/passwd"
        sanitized_filename = validator.sanitize_input(filename_input, 'filename')
        assert '../' not in sanitized_filename
    
    def test_create_security_headers(self):
        """Test security header creation."""
        validator = SecurityValidator()
        
        headers = validator.create_security_headers()
        
        expected_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options', 
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy'
        ]
        
        for header in expected_headers:
            assert header in headers
        
        # Check specific values
        assert headers['X-Content-Type-Options'] == 'nosniff'
        assert headers['X-Frame-Options'] == 'DENY'
    
    def test_validate_password_strength(self):
        """Test password strength validation."""
        validator = SecurityValidator()
        
        # Weak passwords
        weak_passwords = ['123456', 'password', 'abc', 'P@ss']
        for pwd in weak_passwords:
            result = validator.validate_password_strength(pwd)
            assert result.is_valid is False
        
        # Strong password
        strong_password = 'MyStr0ng!P@ssw0rd123'
        result = validator.validate_password_strength(strong_password)
        assert result.is_valid is True
    
    def test_validate_email_format(self):
        """Test email format validation."""
        validator = SecurityValidator()
        
        # Valid emails
        valid_emails = ['user@example.com', 'test.email+tag@domain.co.uk']
        for email in valid_emails:
            result = validator.validate_input(email, 'email')
            assert result.is_valid is True
        
        # Invalid emails
        invalid_emails = ['invalid.email', '@domain.com', 'user@', 'user name@domain.com']
        for email in invalid_emails:
            result = validator.validate_input(email, 'email')
            assert result.is_valid is False


class TestSecurityMonitor:
    """Test cases for SecurityMonitor."""
    
    def test_monitor_init(self):
        """Test monitor initialization."""
        monitor = SecurityMonitor(max_events=1000)
        
        assert monitor.max_events == 1000
        assert len(monitor.events) == 0
        assert len(monitor.rate_limits) == 0
        assert len(monitor.blocked_ips) == 0
    
    def test_record_event(self):
        """Test recording security events."""
        monitor = SecurityMonitor()
        
        monitor.record_event(
            event_type='login_failed',
            threat_level=ThreatLevel.MEDIUM,
            message='Failed login attempt',
            source_ip='192.168.1.100',
            user_id='testuser'
        )
        
        assert len(monitor.events) == 1
        event = monitor.events[0]
        assert event.event_type == 'login_failed'
        assert event.threat_level == ThreatLevel.MEDIUM
        assert event.source_ip == '192.168.1.100'
        assert event.user_id == 'testuser'
    
    def test_add_rate_limit_rule(self):
        """Test adding rate limiting rules."""
        monitor = SecurityMonitor()
        
        monitor.add_rate_limit_rule(
            rule_name='api_requests',
            max_requests=100,
            time_window=60,
            block_duration=300
        )
        
        assert 'api_requests' in monitor.rate_limits
        rule_data = monitor.rate_limits['api_requests']
        assert rule_data['rule'].max_requests == 100
        assert rule_data['rule'].time_window == 60
        assert rule_data['rule'].block_duration == 300
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        monitor = SecurityMonitor()
        
        # Add a strict rate limit for testing
        monitor.add_rate_limit_rule(
            rule_name='test_api',
            max_requests=2,
            time_window=60,
            block_duration=300
        )
        
        client_ip = '192.168.1.100'
        
        # First two requests should be allowed
        assert monitor.is_request_allowed('test_api', client_ip) is True
        assert monitor.is_request_allowed('test_api', client_ip) is True
        
        # Third request should be blocked
        assert monitor.is_request_allowed('test_api', client_ip) is False
        
        # Should have recorded rate limit event
        rate_limit_events = [e for e in monitor.events if e.event_type == 'rate_limit_exceeded']
        assert len(rate_limit_events) >= 1
    
    def test_ip_blocking(self):
        """Test IP blocking functionality."""
        monitor = SecurityMonitor()
        
        # Add rate limit rule that causes IP blocking
        monitor.add_rate_limit_rule(
            rule_name='strict_api',
            max_requests=1,
            time_window=60,
            block_duration=10  # 10 second block
        )
        
        client_ip = '192.168.1.100'
        
        # Trigger rate limit multiple times to cause IP block
        monitor.is_request_allowed('strict_api', client_ip)  # Allowed
        monitor.is_request_allowed('strict_api', client_ip)  # Rate limited
        monitor.is_request_allowed('strict_api', client_ip)  # Rate limited
        monitor.is_request_allowed('strict_api', client_ip)  # Should trigger IP block
        
        # Now all requests should be blocked
        assert monitor.is_request_allowed('strict_api', client_ip) is False
        
        # Check that IP is in blocked list
        assert client_ip in monitor.blocked_ips
    
    def test_brute_force_detection(self):
        """Test brute force attack detection."""
        monitor = SecurityMonitor()
        
        client_ip = '192.168.1.100'
        
        # Simulate multiple failed login attempts
        for i in range(12):  # More than the threshold of 10
            monitor.record_event(
                event_type='login_failed',
                threat_level=ThreatLevel.MEDIUM,
                message=f'Failed login attempt {i+1}',
                source_ip=client_ip
            )
        
        # Should have detected brute force attack
        brute_force_events = [e for e in monitor.events if e.event_type == 'brute_force_attack']
        assert len(brute_force_events) >= 1
        
        # IP should be automatically blocked
        assert client_ip in monitor.blocked_ips
    
    def test_scan_attack_detection(self):
        """Test scanning attack detection."""
        monitor = SecurityMonitor()
        
        client_ip = '192.168.1.100'
        
        # Simulate multiple 404 errors (scanning behavior)
        for i in range(25):  # More than the threshold of 20
            monitor.record_event(
                event_type='404_error',
                threat_level=ThreatLevel.LOW,
                message=f'404 error for path /admin/{i}',
                source_ip=client_ip,
                endpoint=f'/admin/{i}'
            )
        
        # Should have detected scanning attack
        scan_events = [e for e in monitor.events if e.event_type == 'scan_attack']
        assert len(scan_events) >= 1
    
    def test_injection_attack_detection(self):
        """Test injection attack detection."""
        monitor = SecurityMonitor()
        
        client_ip = '192.168.1.100'
        
        # Simulate multiple injection attempts
        for i in range(5):  # More than the threshold of 3
            monitor.record_event(
                event_type='sql_injection',
                threat_level=ThreatLevel.HIGH,
                message=f'SQL injection attempt {i+1}',
                source_ip=client_ip
            )
        
        # Should have detected injection attack
        injection_events = [e for e in monitor.events if e.event_type == 'injection_attack']
        assert len(injection_events) >= 1
        
        # IP should be immediately blocked for injection attacks
        assert client_ip in monitor.blocked_ips
    
    def test_alert_handlers(self):
        """Test alert handler functionality."""
        monitor = SecurityMonitor()
        
        alerts_received = []
        
        def test_alert_handler(event):
            alerts_received.append(event)
        
        monitor.add_alert_handler(test_alert_handler)
        
        # Record a high-severity event
        monitor.record_event(
            event_type='critical_breach',
            threat_level=ThreatLevel.CRITICAL,
            message='Critical security breach detected',
            source_ip='192.168.1.100'
        )
        
        # Alert handler should have been called
        assert len(alerts_received) == 1
        assert alerts_received[0].event_type == 'critical_breach'
    
    def test_get_security_metrics(self):
        """Test getting security metrics."""
        monitor = SecurityMonitor()
        
        # Add some test data
        monitor.record_event('test_event', ThreatLevel.MEDIUM, 'Test message', '192.168.1.100')
        monitor.blocked_ips['192.168.1.100'] = time.time() + 300  # Blocked for 5 minutes
        
        metrics = monitor.get_security_metrics()
        
        assert 'total_events' in metrics
        assert 'events_last_hour' in metrics
        assert 'events_last_day' in metrics
        assert 'blocked_ips' in metrics
        assert 'active_blocks' in metrics
        assert 'top_attacking_ips' in metrics
        
        assert metrics['total_events'] >= 1
        assert metrics['blocked_ips'] >= 1
    
    def test_get_recent_events(self):
        """Test getting recent events."""
        monitor = SecurityMonitor()
        
        # Add events with different types and threat levels
        monitor.record_event('event1', ThreatLevel.LOW, 'Low threat', '192.168.1.100')
        monitor.record_event('event2', ThreatLevel.HIGH, 'High threat', '192.168.1.101')
        monitor.record_event('event1', ThreatLevel.MEDIUM, 'Medium threat', '192.168.1.102')
        
        # Get all recent events
        all_events = monitor.get_recent_events(limit=10)
        assert len(all_events) == 3
        
        # Filter by threat level
        high_events = monitor.get_recent_events(limit=10, threat_level=ThreatLevel.HIGH)
        assert len(high_events) == 1
        assert high_events[0].threat_level == ThreatLevel.HIGH
        
        # Filter by event type
        event1_events = monitor.get_recent_events(limit=10, event_type='event1')
        assert len(event1_events) == 2
    
    def test_generate_security_report(self):
        """Test generating security report."""
        monitor = SecurityMonitor()
        
        # Add some test events
        monitor.record_event('login_failed', ThreatLevel.MEDIUM, 'Failed login', '192.168.1.100')
        monitor.record_event('sql_injection', ThreatLevel.HIGH, 'SQL injection', '192.168.1.100')
        monitor.record_event('login_failed', ThreatLevel.MEDIUM, 'Failed login', '192.168.1.101')
        
        report = monitor.generate_security_report(days=1)
        
        assert 'report_period' in report
        assert 'total_events' in report
        assert 'event_type_breakdown' in report
        assert 'threat_level_breakdown' in report
        assert 'top_event_types' in report
        assert 'top_threat_ips' in report
        assert 'generated_at' in report
        
        assert report['total_events'] >= 3
        assert 'login_failed' in report['event_type_breakdown']
        assert report['event_type_breakdown']['login_failed'] >= 2


class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_scanner_validator_integration(self):
        """Test scanner and validator working together."""
        scanner = SecurityScanner()
        validator = SecurityValidator()
        
        # Create a file with potential security issues
        content = '''
user_input = request.get('data')
query = "SELECT * FROM users WHERE id = " + user_input
password = "hardcoded_password"
'''
        
        # Scan the content
        with patch('builtins.open', mock_open(read_data=content)):
            issues = scanner.scan_file('test.py')
        
        # Should find SQL injection and hardcoded secret
        assert len(issues) >= 2
        
        # Validate the problematic input
        sql_injection_input = "1; DROP TABLE users;"
        validation_result = validator.validate_input(sql_injection_input, 'user_id')
        assert validation_result.is_valid is False
    
    def test_validator_monitor_integration(self):
        """Test validator and monitor integration."""
        validator = SecurityValidator()
        monitor = SecurityMonitor()
        
        # Simulate validation that triggers monitoring
        malicious_input = "<script>alert('xss')</script>"
        validation_result = validator.validate_input(malicious_input, 'comment')
        
        if not validation_result.is_valid:
            # Record security event based on validation failure
            monitor.record_event(
                event_type='xss_attempt',
                threat_level=ThreatLevel.HIGH,
                message='XSS attempt detected in input validation',
                source_ip='192.168.1.100',
                details={'input': malicious_input[:50]}  # Truncated for safety
            )
        
        # Should have recorded the event
        xss_events = [e for e in monitor.events if e.event_type == 'xss_attempt']
        assert len(xss_events) == 1
    
    def test_full_security_pipeline(self):
        """Test complete security pipeline."""
        scanner = SecurityScanner()
        validator = SecurityValidator()
        monitor = SecurityMonitor()
        
        # Step 1: Scan code for vulnerabilities
        vulnerable_code = '''
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)
'''
        
        with patch('builtins.open', mock_open(read_data=vulnerable_code)):
            scan_issues = scanner.scan_file('login.py')
        
        assert len(scan_issues) > 0
        
        # Step 2: Validate incoming request
        malicious_username = "admin'; DROP TABLE users; --"
        validation_result = validator.validate_input(malicious_username, 'username')
        assert validation_result.is_valid is False
        
        # Step 3: Monitor and record security event
        monitor.record_event(
            event_type='sql_injection',
            threat_level=ThreatLevel.CRITICAL,
            message='SQL injection attempt in login form',
            source_ip='192.168.1.100',
            user_id='attacker'
        )
        
        # Step 4: Check that monitoring detected the attack
        injection_events = [e for e in monitor.events if e.event_type == 'sql_injection']
        assert len(injection_events) == 1
        
        # Step 5: Generate comprehensive security report
        report = monitor.generate_security_report(days=1)
        assert report['total_events'] >= 1
        assert 'sql_injection' in report['event_type_breakdown']


class TestSecurityErrors:
    """Test error handling in security components."""
    
    def test_scanner_file_errors(self):
        """Test scanner error handling for file operations."""
        scanner = SecurityScanner()
        
        # Non-existent file
        with patch('builtins.open', side_effect=FileNotFoundError):
            issues = scanner.scan_file('nonexistent.py')
            assert issues == []  # Should handle gracefully
    
    def test_validator_edge_cases(self):
        """Test validator with edge cases."""
        validator = SecurityValidator()
        
        # Empty input
        result = validator.validate_input('', 'username')
        assert result.is_valid is False  # Empty username not valid
        
        # Very long input
        long_input = 'a' * 10000
        result = validator.validate_input(long_input, 'comment')
        assert result.is_valid is False  # Too long
        
        # None input
        result = validator.validate_input(None, 'username')
        assert result.is_valid is False
    
    def test_monitor_resource_limits(self):
        """Test monitor with resource constraints."""
        # Very small event limit
        monitor = SecurityMonitor(max_events=2)
        
        # Add more events than the limit
        monitor.record_event('event1', ThreatLevel.LOW, 'Message 1')
        monitor.record_event('event2', ThreatLevel.LOW, 'Message 2')
        monitor.record_event('event3', ThreatLevel.LOW, 'Message 3')
        
        # Should only keep the most recent events
        assert len(monitor.events) <= 2
        
        # Should still function normally
        metrics = monitor.get_security_metrics()
        assert isinstance(metrics, dict)