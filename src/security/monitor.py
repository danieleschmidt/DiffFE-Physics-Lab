"""Security monitoring and alerting system."""

import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Represents a security event."""
    event_type: str
    threat_level: ThreatLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    max_requests: int
    time_window: int  # seconds
    block_duration: int = 300  # seconds
    threat_level: ThreatLevel = ThreatLevel.MEDIUM


class SecurityMonitor:
    """Real-time security monitoring and threat detection system.
    
    Provides comprehensive security monitoring including:
    - Rate limiting and DDoS protection
    - Anomaly detection
    - Attack pattern recognition
    - Real-time alerting
    - Security metrics collection
    
    Examples
    --------
    >>> monitor = SecurityMonitor()
    >>> monitor.add_rate_limit_rule("api", max_requests=100, time_window=60)
    >>> 
    >>> # Check request
    >>> if monitor.is_request_allowed("api", client_ip="192.168.1.1"):
    ...     process_request()
    >>> else:
    ...     return rate_limit_error()
    """
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events = deque(maxlen=max_events)
        self.rate_limits = {}
        self.blocked_ips = {}
        self.alert_handlers = []
        self.metrics = defaultdict(int)
        self.anomaly_detectors = {}
        self._lock = threading.RLock()
        
        # Start background cleanup task
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_blocks, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("Security monitor initialized")
    
    def record_event(
        self,
        event_type: str,
        threat_level: ThreatLevel,
        message: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        **details
    ) -> None:
        """Record a security event.
        
        Parameters
        ----------
        event_type : str
            Type of security event
        threat_level : ThreatLevel
            Severity level of the event
        message : str
            Description of the event
        source_ip : str, optional
            Source IP address
        user_id : str, optional
            Associated user ID
        endpoint : str, optional
            API endpoint involved
        **details
            Additional event details
        """
        event = SecurityEvent(
            event_type=event_type,
            threat_level=threat_level,
            message=message,
            source_ip=source_ip,
            user_id=user_id,
            endpoint=endpoint,
            details=details
        )
        
        with self._lock:
            self.events.append(event)
            self.metrics[f'events_{event_type}'] += 1
            self.metrics[f'threat_{threat_level.value}'] += 1
        
        # Trigger alerts for high-severity events
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._trigger_alerts(event)
        
        # Check for attack patterns
        self._analyze_event_patterns(event)
        
        logger.info(f"Security event recorded: {event_type} - {message}")
    
    def add_rate_limit_rule(
        self,
        rule_name: str,
        max_requests: int,
        time_window: int,
        block_duration: int = 300,
        threat_level: ThreatLevel = ThreatLevel.MEDIUM
    ) -> None:
        """Add rate limiting rule.
        
        Parameters
        ----------
        rule_name : str
            Name of the rate limiting rule
        max_requests : int
            Maximum requests allowed
        time_window : int
            Time window in seconds
        block_duration : int, optional
            Block duration in seconds, by default 300
        threat_level : ThreatLevel, optional
            Threat level for violations, by default MEDIUM
        """
        rule = RateLimitRule(
            max_requests=max_requests,
            time_window=time_window,
            block_duration=block_duration,
            threat_level=threat_level
        )
        
        with self._lock:
            self.rate_limits[rule_name] = {
                'rule': rule,
                'requests': defaultdict(deque),  # IP -> deque of timestamps
                'violations': defaultdict(int)   # IP -> violation count
            }
        
        logger.info(f"Added rate limit rule: {rule_name} - {max_requests}/{time_window}s")
    
    def is_request_allowed(
        self,
        rule_name: str,
        client_ip: str,
        endpoint: Optional[str] = None
    ) -> bool:
        """Check if request is allowed under rate limiting rules.
        
        Parameters
        ----------
        rule_name : str
            Name of rate limiting rule to check
        client_ip : str
            Client IP address
        endpoint : str, optional
            API endpoint being accessed
            
        Returns
        -------
        bool
            True if request is allowed, False if rate limited
        """
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip):
            self.record_event(
                'blocked_request',
                ThreatLevel.MEDIUM,
                f'Request from blocked IP: {client_ip}',
                source_ip=client_ip,
                endpoint=endpoint
            )
            return False
        
        if rule_name not in self.rate_limits:
            return True
        
        with self._lock:
            rule_data = self.rate_limits[rule_name]
            rule = rule_data['rule']
            requests = rule_data['requests'][client_ip]
            
            current_time = time.time()
            
            # Remove old requests outside time window
            while requests and requests[0] < current_time - rule.time_window:
                requests.popleft()
            
            # Check if limit exceeded
            if len(requests) >= rule.max_requests:
                # Rate limit exceeded
                rule_data['violations'][client_ip] += 1
                
                # Block IP if multiple violations
                if rule_data['violations'][client_ip] >= 3:
                    self._block_ip(client_ip, rule.block_duration)
                
                self.record_event(
                    'rate_limit_exceeded',
                    rule.threat_level,
                    f'Rate limit exceeded for {client_ip}: {len(requests)}/{rule.max_requests} in {rule.time_window}s',
                    source_ip=client_ip,
                    endpoint=endpoint,
                    rule_name=rule_name,
                    request_count=len(requests)
                )
                
                return False
            
            # Add current request
            requests.append(current_time)
            return True
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked."""
        if ip not in self.blocked_ips:
            return False
        
        block_until = self.blocked_ips[ip]
        if time.time() > block_until:
            # Block expired
            del self.blocked_ips[ip]
            return False
        
        return True
    
    def _block_ip(self, ip: str, duration: int) -> None:
        """Block IP address for specified duration."""
        block_until = time.time() + duration
        self.blocked_ips[ip] = block_until
        
        self.record_event(
            'ip_blocked',
            ThreatLevel.HIGH,
            f'IP blocked for {duration}s: {ip}',
            source_ip=ip,
            block_duration=duration,
            block_until=block_until
        )
        
        logger.warning(f"Blocked IP {ip} for {duration} seconds")
    
    def add_alert_handler(self, handler: Callable[[SecurityEvent], None]) -> None:
        """Add alert handler for security events.
        
        Parameters
        ----------
        handler : Callable[[SecurityEvent], None]
            Function to handle security alerts
        """
        self.alert_handlers.append(handler)
        logger.info("Added security alert handler")
    
    def _trigger_alerts(self, event: SecurityEvent) -> None:
        """Trigger alerts for security event."""
        for handler in self.alert_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def _analyze_event_patterns(self, event: SecurityEvent) -> None:
        """Analyze event patterns for attack detection."""
        with self._lock:
            recent_events = [e for e in self.events 
                           if e.timestamp > datetime.now() - timedelta(minutes=5)]
        
        # Check for brute force attacks
        if event.event_type in ['login_failed', 'authentication_failed']:
            self._check_brute_force_attack(event, recent_events)
        
        # Check for scan attempts
        if event.event_type in ['404_error', 'forbidden_access']:
            self._check_scan_attempts(event, recent_events)
        
        # Check for injection attempts
        if event.event_type in ['sql_injection', 'xss_attempt', 'command_injection']:
            self._check_injection_attacks(event, recent_events)
    
    def _check_brute_force_attack(self, event: SecurityEvent, recent_events: List[SecurityEvent]) -> None:
        """Check for brute force attack patterns."""
        if not event.source_ip:
            return
        
        # Count recent failed attempts from same IP
        failed_attempts = [
            e for e in recent_events
            if (e.source_ip == event.source_ip and 
                e.event_type in ['login_failed', 'authentication_failed'])
        ]
        
        if len(failed_attempts) >= 10:  # 10 failed attempts in 5 minutes
            self.record_event(
                'brute_force_attack',
                ThreatLevel.HIGH,
                f'Brute force attack detected from {event.source_ip}: {len(failed_attempts)} failed attempts',
                source_ip=event.source_ip,
                failed_attempts=len(failed_attempts)
            )
            
            # Automatically block IP
            self._block_ip(event.source_ip, 3600)  # Block for 1 hour
    
    def _check_scan_attempts(self, event: SecurityEvent, recent_events: List[SecurityEvent]) -> None:
        """Check for scanning/reconnaissance attempts."""
        if not event.source_ip:
            return
        
        # Count recent 404/403 errors from same IP
        scan_events = [
            e for e in recent_events
            if (e.source_ip == event.source_ip and 
                e.event_type in ['404_error', 'forbidden_access'])
        ]
        
        if len(scan_events) >= 20:  # 20 scan attempts in 5 minutes
            self.record_event(
                'scan_attack',
                ThreatLevel.MEDIUM,
                f'Scanning attack detected from {event.source_ip}: {len(scan_events)} scan attempts',
                source_ip=event.source_ip,
                scan_attempts=len(scan_events)
            )
    
    def _check_injection_attacks(self, event: SecurityEvent, recent_events: List[SecurityEvent]) -> None:
        """Check for injection attack patterns."""
        if not event.source_ip:
            return
        
        # Count recent injection attempts from same IP
        injection_events = [
            e for e in recent_events
            if (e.source_ip == event.source_ip and 
                e.event_type in ['sql_injection', 'xss_attempt', 'command_injection'])
        ]
        
        if len(injection_events) >= 3:  # 3 injection attempts in 5 minutes
            self.record_event(
                'injection_attack',
                ThreatLevel.CRITICAL,
                f'Injection attack detected from {event.source_ip}: {len(injection_events)} attempts',
                source_ip=event.source_ip,
                injection_attempts=len(injection_events)
            )
            
            # Immediately block IP for injection attacks
            self._block_ip(event.source_ip, 7200)  # Block for 2 hours
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of security metrics
        """
        with self._lock:
            current_time = time.time()
            
            # Count events by time periods
            last_hour = datetime.now() - timedelta(hours=1)
            last_day = datetime.now() - timedelta(days=1)
            
            events_last_hour = [e for e in self.events if e.timestamp > last_hour]
            events_last_day = [e for e in self.events if e.timestamp > last_day]
            
            # Count by threat level
            threat_counts = defaultdict(int)
            for event in events_last_day:
                threat_counts[event.threat_level.value] += 1
            
            # Top attacking IPs
            ip_counts = defaultdict(int)
            for event in events_last_day:
                if event.source_ip:
                    ip_counts[event.source_ip] += 1
            
            top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'total_events': len(self.events),
                'events_last_hour': len(events_last_hour),
                'events_last_day': len(events_last_day),
                'threat_level_counts': dict(threat_counts),
                'blocked_ips': len(self.blocked_ips),
                'active_blocks': len([ip for ip, block_time in self.blocked_ips.items() 
                                    if current_time < block_time]),
                'top_attacking_ips': top_ips,
                'rate_limit_rules': len(self.rate_limits),
                'alert_handlers': len(self.alert_handlers)
            }
    
    def get_recent_events(
        self,
        limit: int = 100,
        threat_level: Optional[ThreatLevel] = None,
        event_type: Optional[str] = None
    ) -> List[SecurityEvent]:
        """Get recent security events.
        
        Parameters
        ----------
        limit : int, optional
            Maximum number of events to return, by default 100
        threat_level : ThreatLevel, optional
            Filter by threat level
        event_type : str, optional
            Filter by event type
            
        Returns
        -------
        List[SecurityEvent]
            List of recent security events
        """
        with self._lock:
            events = list(self.events)
        
        # Apply filters
        if threat_level:
            events = [e for e in events if e.threat_level == threat_level]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Sort by timestamp (most recent first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def clear_events(self, older_than_days: int = 30) -> int:
        """Clear old security events.
        
        Parameters
        ----------
        older_than_days : int, optional
            Clear events older than this many days, by default 30
            
        Returns
        -------
        int
            Number of events cleared
        """
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        
        with self._lock:
            original_count = len(self.events)
            
            # Keep only recent events
            recent_events = [e for e in self.events if e.timestamp > cutoff_time]
            self.events.clear()
            self.events.extend(recent_events)
            
            cleared_count = original_count - len(self.events)
        
        logger.info(f"Cleared {cleared_count} old security events")
        return cleared_count
    
    def _cleanup_expired_blocks(self) -> None:
        """Background task to clean up expired IP blocks."""
        while True:
            try:
                current_time = time.time()
                expired_ips = []
                
                with self._lock:
                    for ip, block_until in list(self.blocked_ips.items()):
                        if current_time > block_until:
                            expired_ips.append(ip)
                    
                    for ip in expired_ips:
                        del self.blocked_ips[ip]
                
                if expired_ips:
                    logger.info(f"Unblocked {len(expired_ips)} expired IPs")
                
                # Sleep for 1 minute before next cleanup
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in cleanup thread: {e}")
                time.sleep(60)
    
    def generate_security_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive security report.
        
        Parameters
        ----------
        days : int, optional
            Number of days to include in report, by default 7
            
        Returns
        -------
        Dict[str, Any]
            Security report
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        with self._lock:
            period_events = [e for e in self.events if e.timestamp > cutoff_time]
        
        # Analyze events
        event_types = defaultdict(int)
        threat_levels = defaultdict(int)
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        ip_stats = defaultdict(lambda: {'events': 0, 'threat_score': 0})
        
        for event in period_events:
            event_types[event.event_type] += 1
            threat_levels[event.threat_level.value] += 1
            
            # Hourly distribution
            hour_key = event.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_counts[hour_key] += 1
            
            # Daily distribution  
            day_key = event.timestamp.strftime('%Y-%m-%d')
            daily_counts[day_key] += 1
            
            # IP statistics
            if event.source_ip:
                ip_stats[event.source_ip]['events'] += 1
                threat_score = {
                    ThreatLevel.INFO: 1,
                    ThreatLevel.LOW: 2,
                    ThreatLevel.MEDIUM: 5,
                    ThreatLevel.HIGH: 10,
                    ThreatLevel.CRITICAL: 20
                }.get(event.threat_level, 1)
                ip_stats[event.source_ip]['threat_score'] += threat_score
        
        # Top threats
        top_event_types = sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:10]
        top_threat_ips = sorted(ip_stats.items(), key=lambda x: x[1]['threat_score'], reverse=True)[:10]
        
        return {
            'report_period': f'{days} days',
            'total_events': len(period_events),
            'event_type_breakdown': dict(event_types),
            'threat_level_breakdown': dict(threat_levels),
            'top_event_types': top_event_types,
            'top_threat_ips': [(ip, stats) for ip, stats in top_threat_ips],
            'daily_event_counts': dict(daily_counts),
            'hourly_event_counts': dict(hourly_counts),
            'unique_ips': len(ip_stats),
            'blocked_ips_count': len(self.blocked_ips),
            'generated_at': datetime.now().isoformat()
        }