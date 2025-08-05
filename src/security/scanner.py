"""Security scanner for code and configuration analysis."""

import ast
import os
import re
import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Security issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    rule_id: str
    severity: Severity
    message: str
    file_path: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class SecurityScanner:
    """Scanner for detecting security vulnerabilities in code and configuration.
    
    Performs static analysis to identify potential security issues including:
    - Hardcoded secrets and credentials
    - Unsafe function usage
    - Path traversal vulnerabilities
    - Code injection risks
    - Insecure configurations
    
    Examples
    --------
    >>> scanner = SecurityScanner()
    >>> issues = scanner.scan_directory("src/")
    >>> critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
    """
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.excluded_patterns = {
            '*.pyc', '__pycache__', '.git', '.pytest_cache',
            'node_modules', '.venv', 'venv'
        }
    
    def _initialize_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize security scanning rules."""
        return {
            # Hardcoded secrets
            'hardcoded-password': {
                'pattern': r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']([^"\']{8,})["\']',
                'severity': Severity.HIGH,
                'message': 'Hardcoded password detected'
            },
            'hardcoded-api-key': {
                'pattern': r'(?i)(api[_-]?key|apikey|access[_-]?token)\s*[=:]\s*["\']([A-Za-z0-9_-]{20,})["\']',
                'severity': Severity.HIGH,
                'message': 'Hardcoded API key detected'
            },
            'hardcoded-secret': {
                'pattern': r'(?i)(secret|token)\s*[=:]\s*["\']([A-Za-z0-9_-]{16,})["\']',
                'severity': Severity.MEDIUM,
                'message': 'Hardcoded secret detected'
            },
            
            # Unsafe functions
            'eval-usage': {
                'pattern': r'\beval\s*\(',
                'severity': Severity.CRITICAL,
                'message': 'Use of eval() function detected - code injection risk'
            },
            'exec-usage': {
                'pattern': r'\bexec\s*\(',
                'severity': Severity.CRITICAL,
                'message': 'Use of exec() function detected - code injection risk'
            },
            'subprocess-shell': {
                'pattern': r'subprocess\.(call|run|Popen).*shell\s*=\s*True',
                'severity': Severity.HIGH,
                'message': 'subprocess with shell=True detected - command injection risk'
            },
            
            # Path traversal
            'path-traversal': {
                'pattern': r'\.\.[\\/]',
                'severity': Severity.MEDIUM,
                'message': 'Potential path traversal pattern detected'
            },
            
            # Unsafe deserialization
            'pickle-usage': {
                'pattern': r'\bpickle\.(loads?|load)\s*\(',
                'severity': Severity.HIGH,
                'message': 'Unsafe pickle deserialization detected'
            },
            
            # Debug/development patterns
            'debug-print': {
                'pattern': r'print\s*\([^)]*(?:password|secret|key|token)[^)]*\)',
                'severity': Severity.MEDIUM,
                'message': 'Sensitive information in print statement'
            },
            'debug-enabled': {
                'pattern': r'DEBUG\s*=\s*True',
                'severity': Severity.LOW,
                'message': 'Debug mode enabled in production code'
            },
            
            # Weak cryptography
            'weak-hash': {
                'pattern': r'hashlib\.(md5|sha1)\s*\(',
                'severity': Severity.MEDIUM,
                'message': 'Weak cryptographic hash function usage'
            },
            
            # SQL injection patterns
            'sql-injection': {
                'pattern': r'(execute|query)\s*\([^)]*%[sf]',
                'severity': Severity.HIGH,
                'message': 'Potential SQL injection vulnerability'
            },
            
            # Insecure random
            'weak-random': {
                'pattern': r'random\.(random|randint|choice)',
                'severity': Severity.LOW,
                'message': 'Use of non-cryptographically secure random function'
            }
        }
    
    def scan_directory(self, directory: str) -> List[SecurityIssue]:
        """Scan entire directory for security issues.
        
        Parameters
        ----------
        directory : str
            Directory path to scan
            
        Returns
        -------
        List[SecurityIssue]
            List of detected security issues
        """
        issues = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory}")
            return issues
        
        logger.info(f"Scanning directory: {directory}")
        
        for file_path in self._get_scannable_files(directory_path):
            try:
                file_issues = self.scan_file(str(file_path))
                issues.extend(file_issues)
            except Exception as e:
                logger.warning(f"Failed to scan {file_path}: {e}")
        
        logger.info(f"Scan complete: {len(issues)} issues found")
        return issues
    
    def scan_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan individual file for security issues.
        
        Parameters
        ----------
        file_path : str
            Path to file to scan
            
        Returns
        -------
        List[SecurityIssue]
            List of detected security issues in the file
        """
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Regex-based scanning
            issues.extend(self._scan_with_regex(file_path, content, lines))
            
            # AST-based scanning for Python files
            if file_path.endswith('.py'):
                issues.extend(self._scan_python_ast(file_path, content))
            
        except Exception as e:
            logger.warning(f"Error scanning file {file_path}: {e}")
        
        return issues
    
    def _get_scannable_files(self, directory: Path) -> List[Path]:
        """Get list of files to scan, excluding unwanted patterns."""
        scannable_files = []
        
        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not self._is_excluded(d)]
            
            for file in files:
                file_path = Path(root) / file
                
                if not self._is_excluded(file) and self._is_scannable_file(file_path):
                    scannable_files.append(file_path)
        
        return scannable_files
    
    def _is_excluded(self, name: str) -> bool:
        """Check if file/directory should be excluded."""
        for pattern in self.excluded_patterns:
            if pattern.startswith('*'):
                if name.endswith(pattern[1:]):
                    return True
            elif name == pattern:
                return True
        return False
    
    def _is_scannable_file(self, file_path: Path) -> bool:
        """Check if file should be scanned."""
        # Scan text files
        text_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
            '.yaml', '.yml', '.json', '.xml', '.ini', '.cfg',
            '.sh', '.bash', '.ps1', '.sql', '.md', '.txt'
        }
        
        return file_path.suffix.lower() in text_extensions
    
    def _scan_with_regex(self, file_path: str, content: str, lines: List[str]) -> List[SecurityIssue]:
        """Scan file content using regex patterns."""
        issues = []
        
        for rule_id, rule in self.rules.items():
            pattern = rule['pattern']
            
            for line_num, line in enumerate(lines, 1):
                matches = re.finditer(pattern, line)
                
                for match in matches:
                    # Skip if this looks like a test or example
                    if self._is_likely_test_code(file_path, line):
                        continue
                    
                    issue = SecurityIssue(
                        rule_id=rule_id,
                        severity=rule['severity'],
                        message=rule['message'],
                        file_path=file_path,
                        line_number=line_num,
                        column=match.start(),
                        details={
                            'matched_text': match.group(0),
                            'line_content': line.strip()
                        }
                    )
                    issues.append(issue)
        
        return issues
    
    def _scan_python_ast(self, file_path: str, content: str) -> List[SecurityIssue]:
        """Scan Python file using AST analysis."""
        issues = []
        
        try:
            tree = ast.parse(content)
            visitor = SecurityASTVisitor(file_path)
            visitor.visit(tree)
            issues.extend(visitor.issues)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"AST parsing error in {file_path}: {e}")
        
        return issues
    
    def _is_likely_test_code(self, file_path: str, line: str) -> bool:
        """Check if code appears to be test/example code."""
        # Check file path
        test_indicators = ['test', 'tests', 'example', 'demo', 'sample']
        file_lower = file_path.lower()
        
        if any(indicator in file_lower for indicator in test_indicators):
            return True
        
        # Check line content
        line_lower = line.lower()
        test_line_indicators = ['# test', '# example', '# demo', 'test_', 'example_']
        
        return any(indicator in line_lower for indicator in test_line_indicators)
    
    def generate_report(self, issues: List[SecurityIssue]) -> Dict[str, Any]:
        """Generate security scan report.
        
        Parameters
        ----------
        issues : List[SecurityIssue]
            List of detected security issues
            
        Returns
        -------
        Dict[str, Any]
            Security scan report
        """
        # Count by severity
        severity_counts = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 0,
            Severity.MEDIUM: 0,
            Severity.LOW: 0
        }
        
        for issue in issues:
            severity_counts[issue.severity] += 1
        
        # Count by rule
        rule_counts = {}
        for issue in issues:
            rule_counts[issue.rule_id] = rule_counts.get(issue.rule_id, 0) + 1
        
        # Count by file
        file_counts = {}
        for issue in issues:
            file_counts[issue.file_path] = file_counts.get(issue.file_path, 0) + 1
        
        # Calculate risk score (weighted by severity)
        risk_score = (
            severity_counts[Severity.CRITICAL] * 10 +
            severity_counts[Severity.HIGH] * 7 +
            severity_counts[Severity.MEDIUM] * 4 +
            severity_counts[Severity.LOW] * 1
        )
        
        report = {
            'total_issues': len(issues),
            'risk_score': risk_score,
            'severity_breakdown': {s.value: count for s, count in severity_counts.items()},
            'rule_breakdown': rule_counts,
            'file_breakdown': file_counts,
            'most_common_issues': sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'most_problematic_files': sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'issues': [self._issue_to_dict(issue) for issue in issues]
        }
        
        return report
    
    def _issue_to_dict(self, issue: SecurityIssue) -> Dict[str, Any]:
        """Convert SecurityIssue to dictionary."""
        return {
            'rule_id': issue.rule_id,
            'severity': issue.severity.value,
            'message': issue.message,
            'file_path': issue.file_path,
            'line_number': issue.line_number,
            'column': issue.column,
            'details': issue.details or {}
        }
    
    def add_custom_rule(
        self,
        rule_id: str,
        pattern: str,
        severity: Severity,
        message: str
    ) -> None:
        """Add custom security rule.
        
        Parameters
        ----------
        rule_id : str
            Unique identifier for the rule
        pattern : str
            Regex pattern to match
        severity : Severity
            Severity level of issues found
        message : str
            Description of the security issue
        """
        self.rules[rule_id] = {
            'pattern': pattern,
            'severity': severity,
            'message': message
        }
        
        logger.info(f"Added custom security rule: {rule_id}")


class SecurityASTVisitor(ast.NodeVisitor):
    """AST visitor for detecting security issues in Python code."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.issues = []
    
    def visit_Call(self, node):
        """Visit function calls."""
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            if func_name in ['eval', 'exec']:
                self.issues.append(SecurityIssue(
                    rule_id=f'{func_name}-usage',
                    severity=Severity.CRITICAL,
                    message=f'Use of {func_name}() function detected - code injection risk',
                    file_path=self.file_path,
                    line_number=getattr(node, 'lineno', None)
                ))
        
        elif isinstance(node.func, ast.Attribute):
            # Check for module.function patterns
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                func_name = node.func.attr
                
                # Check for pickle.loads/load
                if module_name == 'pickle' and func_name in ['loads', 'load']:
                    self.issues.append(SecurityIssue(
                        rule_id='pickle-usage',
                        severity=Severity.HIGH,
                        message='Unsafe pickle deserialization detected',
                        file_path=self.file_path,
                        line_number=getattr(node, 'lineno', None)
                    ))
                
                # Check for subprocess with shell=True
                if module_name == 'subprocess' and func_name in ['call', 'run', 'Popen']:
                    for keyword in node.keywords:
                        if (keyword.arg == 'shell' and 
                            isinstance(keyword.value, ast.Constant) and
                            keyword.value.value is True):
                            
                            self.issues.append(SecurityIssue(
                                rule_id='subprocess-shell',
                                severity=Severity.HIGH,
                                message='subprocess with shell=True detected - command injection risk',
                                file_path=self.file_path,
                                line_number=getattr(node, 'lineno', None)
                            ))
        
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Visit assignments for hardcoded secrets."""
        # Look for assignments to sensitive variable names
        sensitive_names = {'password', 'secret', 'api_key', 'token', 'private_key'}
        
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.lower() in sensitive_names:
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    if len(node.value.value) > 8:  # Skip short/placeholder values
                        self.issues.append(SecurityIssue(
                            rule_id='hardcoded-secret-assignment',
                            severity=Severity.HIGH,
                            message=f'Hardcoded secret in variable assignment: {target.id}',
                            file_path=self.file_path,
                            line_number=getattr(node, 'lineno', None)
                        ))
        
        self.generic_visit(node)