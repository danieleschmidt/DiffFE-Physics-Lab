"""Global compliance and regulatory support."""

import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"           # General Data Protection Regulation (EU)
    CCPA = "ccpa"           # California Consumer Privacy Act (US)
    PDPA = "pdpa"           # Personal Data Protection Act (Singapore)
    SOX = "sox"             # Sarbanes-Oxley Act (US)
    ISO27001 = "iso27001"   # Information Security Management
    HIPAA = "hipaa"         # Health Insurance Portability (US)


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    activity_id: str
    purpose: str
    legal_basis: str
    data_categories: List[str]
    recipients: List[str]
    retention_period: Optional[int]  # Days
    cross_border_transfers: bool
    timestamp: datetime
    user_consent: Optional[bool] = None


@dataclass  
class ComplianceConfig:
    """Configuration for compliance requirements."""
    enabled_frameworks: Set[ComplianceFramework]
    data_retention_days: int
    require_explicit_consent: bool
    log_data_processing: bool
    anonymize_logs: bool
    encryption_required: bool
    audit_trail_enabled: bool
    cross_border_restrictions: Dict[str, List[str]]  # Framework -> restricted countries


class ComplianceManager:
    """Manages global compliance and regulatory requirements."""
    
    def __init__(self, config: Optional[ComplianceConfig] = None):
        self.config = config or self._default_config()
        self.processing_records = []
        self.consent_records = {}
        self.audit_trail = []
        self.data_inventory = {}
        
    def _default_config(self) -> ComplianceConfig:
        """Create default compliance configuration."""
        return ComplianceConfig(
            enabled_frameworks={ComplianceFramework.GDPR, ComplianceFramework.CCPA},
            data_retention_days=365,
            require_explicit_consent=True,
            log_data_processing=True,
            anonymize_logs=True,
            encryption_required=True,
            audit_trail_enabled=True,
            cross_border_restrictions={
                ComplianceFramework.GDPR.value: ["US", "CN"],  # Example restrictions
                ComplianceFramework.CCPA.value: ["CN"]
            }
        )
    
    def record_data_processing(
        self,
        activity_id: str,
        purpose: str,
        legal_basis: str,
        data_categories: List[str],
        recipients: List[str] = None,
        retention_days: Optional[int] = None,
        cross_border: bool = False,
        user_consent: Optional[bool] = None
    ):
        """Record a data processing activity."""
        record = DataProcessingRecord(
            activity_id=activity_id,
            purpose=purpose,
            legal_basis=legal_basis,
            data_categories=data_categories,
            recipients=recipients or [],
            retention_period=retention_days or self.config.data_retention_days,
            cross_border_transfers=cross_border,
            timestamp=datetime.utcnow(),
            user_consent=user_consent
        )
        
        self.processing_records.append(record)
        
        # Validate against enabled frameworks
        for framework in self.config.enabled_frameworks:
            self._validate_processing(record, framework)
        
        if self.config.log_data_processing:
            self._log_processing(record)
    
    def _validate_processing(self, record: DataProcessingRecord, framework: ComplianceFramework):
        """Validate processing record against compliance framework."""
        violations = []
        
        if framework == ComplianceFramework.GDPR:
            violations.extend(self._validate_gdpr(record))
        elif framework == ComplianceFramework.CCPA:
            violations.extend(self._validate_ccpa(record))
        elif framework == ComplianceFramework.PDPA:
            violations.extend(self._validate_pdpa(record))
        
        if violations:
            self._record_compliance_violation(record.activity_id, framework, violations)
    
    def _validate_gdpr(self, record: DataProcessingRecord) -> List[str]:
        """Validate against GDPR requirements."""
        violations = []
        
        # Check consent requirement
        if self.config.require_explicit_consent and record.user_consent is not True:
            if record.legal_basis not in ["contract", "legal_obligation", "vital_interests", "public_task", "legitimate_interests"]:
                violations.append("GDPR: Explicit consent required but not obtained")
        
        # Check retention period
        if record.retention_period and record.retention_period > 2555:  # 7 years max for most data
            violations.append("GDPR: Retention period exceeds reasonable limits")
        
        # Check cross-border transfers
        if record.cross_border_transfers:
            violations.append("GDPR: Cross-border transfer requires adequacy decision or appropriate safeguards")
        
        # Check data minimization
        sensitive_categories = ["biometric", "genetic", "health", "racial", "political", "religious"]
        if any(cat in record.data_categories for cat in sensitive_categories):
            if record.legal_basis not in ["explicit_consent", "vital_interests", "legal_claim"]:
                violations.append("GDPR: Special category data requires specific legal basis")
        
        return violations
    
    def _validate_ccpa(self, record: DataProcessingRecord) -> List[str]:
        """Validate against CCPA requirements."""
        violations = []
        
        # Check sale of personal information
        if "sale" in record.purpose.lower() and record.user_consent is not True:
            violations.append("CCPA: Sale of personal information requires opt-in consent")
        
        # Check retention period
        if record.retention_period and record.retention_period > 365:  # 1 year default
            violations.append("CCPA: Retention period should be reasonably necessary")
        
        return violations
    
    def _validate_pdpa(self, record: DataProcessingRecord) -> List[str]:
        """Validate against PDPA requirements.""" 
        violations = []
        
        # Check consent requirement
        if record.user_consent is not True and record.legal_basis not in ["contract", "legal_obligation"]:
            violations.append("PDPA: Valid consent or legal basis required")
        
        return violations
    
    def _record_compliance_violation(self, activity_id: str, framework: ComplianceFramework, violations: List[str]):
        """Record a compliance violation."""
        violation_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'activity_id': activity_id,
            'framework': framework.value,
            'violations': violations,
            'severity': 'high' if 'consent' in str(violations).lower() else 'medium'
        }
        
        self.audit_trail.append(violation_record)
        
        # Log violation
        for violation in violations:
            logger.warning(f"Compliance violation ({framework.value}): {violation} in activity {activity_id}")
    
    def _log_processing(self, record: DataProcessingRecord):
        """Log data processing activity."""
        log_data = {
            'activity_id': record.activity_id,
            'purpose': record.purpose,
            'timestamp': record.timestamp.isoformat(),
            'data_categories_count': len(record.data_categories)
        }
        
        # Anonymize sensitive data if required
        if self.config.anonymize_logs:
            log_data['activity_id'] = hashlib.sha256(record.activity_id.encode()).hexdigest()[:16]
            log_data['data_categories'] = ["anonymized"] * len(record.data_categories)
        
        logger.info(f"Data processing activity: {json.dumps(log_data)}")
    
    def record_user_consent(
        self,
        user_id: str,
        purpose: str,
        consent_given: bool,
        consent_method: str = "explicit",
        expiry_date: Optional[datetime] = None
    ):
        """Record user consent."""
        consent_record = {
            'user_id': user_id,
            'purpose': purpose,
            'consent_given': consent_given,
            'consent_method': consent_method,
            'timestamp': datetime.utcnow(),
            'expiry_date': expiry_date or datetime.utcnow() + timedelta(days=365),
            'withdrawn': False
        }
        
        consent_key = f"{user_id}:{purpose}"
        self.consent_records[consent_key] = consent_record
        
        logger.info(f"Consent recorded: user={user_id}, purpose={purpose}, given={consent_given}")
    
    def check_user_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given consent for a purpose."""
        consent_key = f"{user_id}:{purpose}"
        
        if consent_key in self.consent_records:
            record = self.consent_records[consent_key]
            
            # Check if consent is still valid
            if not record['withdrawn'] and datetime.utcnow() < record['expiry_date']:
                return record['consent_given']
        
        return False
    
    def withdraw_consent(self, user_id: str, purpose: str):
        """Withdraw user consent."""
        consent_key = f"{user_id}:{purpose}"
        
        if consent_key in self.consent_records:
            self.consent_records[consent_key]['withdrawn'] = True
            self.consent_records[consent_key]['withdrawal_date'] = datetime.utcnow()
            logger.info(f"Consent withdrawn: user={user_id}, purpose={purpose}")
    
    def cleanup_expired_data(self) -> int:
        """Clean up expired data based on retention policies."""
        cleaned_count = 0
        current_time = datetime.utcnow()
        
        # Clean up processing records
        self.processing_records = [
            record for record in self.processing_records
            if current_time - record.timestamp < timedelta(days=record.retention_period or self.config.data_retention_days)
        ]
        
        # Clean up expired consents
        expired_consents = []
        for key, record in self.consent_records.items():
            if current_time > record['expiry_date']:
                expired_consents.append(key)
                cleaned_count += 1
        
        for key in expired_consents:
            del self.consent_records[key]
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired data records")
        
        return cleaned_count
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        current_time = datetime.utcnow()
        
        # Processing activity statistics
        recent_activities = [
            r for r in self.processing_records
            if current_time - r.timestamp < timedelta(days=30)
        ]
        
        # Consent statistics
        total_consents = len(self.consent_records)
        active_consents = len([
            r for r in self.consent_records.values()
            if r['consent_given'] and not r['withdrawn'] and current_time < r['expiry_date']
        ])
        
        # Violation statistics
        recent_violations = [
            v for v in self.audit_trail
            if datetime.fromisoformat(v['timestamp']) > current_time - timedelta(days=30)
        ]
        
        report = {
            'report_timestamp': current_time.isoformat(),
            'enabled_frameworks': [f.value for f in self.config.enabled_frameworks],
            'processing_activities': {
                'total': len(self.processing_records),
                'recent_30_days': len(recent_activities),
                'categories': list(set(cat for record in recent_activities for cat in record.data_categories))
            },
            'consent_management': {
                'total_consents': total_consents,
                'active_consents': active_consents,
                'consent_rate': active_consents / max(total_consents, 1)
            },
            'compliance_violations': {
                'total': len(self.audit_trail),
                'recent_30_days': len(recent_violations),
                'by_framework': {}
            },
            'data_retention': {
                'default_period_days': self.config.data_retention_days,
                'cleanup_enabled': True,
                'last_cleanup': 'auto'
            }
        }
        
        # Group violations by framework
        for violation in recent_violations:
            framework = violation['framework']
            if framework not in report['compliance_violations']['by_framework']:
                report['compliance_violations']['by_framework'][framework] = 0
            report['compliance_violations']['by_framework'][framework] += len(violation['violations'])
        
        return report
    
    def export_data_for_user(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user (GDPR Article 20 - Right to data portability)."""
        user_data = {
            'user_id': user_id,
            'export_timestamp': datetime.utcnow().isoformat(),
            'processing_activities': [],
            'consent_records': [],
            'retention_info': {}
        }
        
        # Find processing activities involving the user
        for record in self.processing_records:
            if user_id in record.recipients or user_id in record.activity_id:
                user_data['processing_activities'].append({
                    'activity_id': record.activity_id,
                    'purpose': record.purpose,
                    'legal_basis': record.legal_basis,
                    'data_categories': record.data_categories,
                    'timestamp': record.timestamp.isoformat(),
                    'retention_period_days': record.retention_period
                })
        
        # Find consent records
        for key, record in self.consent_records.items():
            if record['user_id'] == user_id:
                user_data['consent_records'].append({
                    'purpose': record['purpose'],
                    'consent_given': record['consent_given'],
                    'timestamp': record['timestamp'].isoformat(),
                    'expiry_date': record['expiry_date'].isoformat(),
                    'withdrawn': record['withdrawn']
                })
        
        logger.info(f"Data export generated for user {user_id}")
        return user_data


# Global compliance manager instance
_global_compliance_manager = None


def get_compliance_manager() -> ComplianceManager:
    """Get the global compliance manager instance."""
    global _global_compliance_manager
    if _global_compliance_manager is None:
        _global_compliance_manager = ComplianceManager()
    return _global_compliance_manager


def record_data_processing(**kwargs):
    """Global function to record data processing."""
    return get_compliance_manager().record_data_processing(**kwargs)


def check_user_consent(user_id: str, purpose: str) -> bool:
    """Global function to check user consent."""
    return get_compliance_manager().check_user_consent(user_id, purpose)