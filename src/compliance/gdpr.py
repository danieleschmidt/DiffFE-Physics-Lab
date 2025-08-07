"""GDPR compliance utilities for sentiment analysis framework."""

import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import uuid


class ConsentType(Enum):
    """Types of consent for data processing."""
    
    PERFORMANCE = "performance"  # Performance analytics
    FUNCTIONALITY = "functionality"  # Essential functionality
    ANALYTICS = "analytics"  # Usage analytics
    MARKETING = "marketing"  # Marketing communications
    PERSONALIZATION = "personalization"  # Personalized content
    

class LegalBasis(Enum):
    """Legal basis for data processing under GDPR."""
    
    CONSENT = "consent"  # Article 6(1)(a)
    CONTRACT = "contract"  # Article 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation"  # Article 6(1)(c)
    VITAL_INTERESTS = "vital_interests"  # Article 6(1)(d)
    PUBLIC_TASK = "public_task"  # Article 6(1)(e)
    LEGITIMATE_INTERESTS = "legitimate_interests"  # Article 6(1)(f)


@dataclass
class DataSubject:
    """Data subject information for GDPR compliance."""
    
    subject_id: str
    email: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: float = None
    consent_status: Dict[str, bool] = None
    legal_basis: Dict[str, str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.consent_status is None:
            self.consent_status = {}
        if self.legal_basis is None:
            self.legal_basis = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProcessingActivity:
    """Data processing activity record."""
    
    activity_id: str
    subject_id: str
    activity_type: str
    data_categories: List[str]
    processing_purpose: str
    legal_basis: str
    data_retention_period: int  # in days
    timestamp: float
    data_hash: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ConsentManager:
    """Manager for handling GDPR consent and data subject rights."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize consent manager.
        
        Parameters
        ----------
        storage_path : Path, optional
            Path to store consent records
        """
        self.storage_path = storage_path or Path.home() / '.sentiment_gdpr'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_activities: List[ProcessingActivity] = []
        self._lock = threading.RLock()
        
        # Data categories handled by the sentiment analysis system
        self.data_categories = {
            'text_content': {
                'description': 'Text content submitted for sentiment analysis',
                'sensitivity': 'medium',
                'retention_period': 30,  # days
                'legal_basis': [LegalBasis.CONSENT, LegalBasis.CONTRACT]
            },
            'analysis_results': {
                'description': 'Sentiment analysis results and scores',
                'sensitivity': 'low',
                'retention_period': 90,
                'legal_basis': [LegalBasis.CONSENT, LegalBasis.LEGITIMATE_INTERESTS]
            },
            'usage_analytics': {
                'description': 'API usage statistics and performance metrics',
                'sensitivity': 'low',
                'retention_period': 365,
                'legal_basis': [LegalBasis.LEGITIMATE_INTERESTS]
            },
            'technical_logs': {
                'description': 'System logs and error information',
                'sensitivity': 'low',
                'retention_period': 30,
                'legal_basis': [LegalBasis.LEGITIMATE_INTERESTS]
            },
            'personal_identifiers': {
                'description': 'IP addresses, user agents, session IDs',
                'sensitivity': 'high',
                'retention_period': 30,
                'legal_basis': [LegalBasis.CONSENT, LegalBasis.LEGITIMATE_INTERESTS]
            }
        }
        
        self._load_data()
        
    def _load_data(self):
        """Load existing consent and processing records."""
        try:
            # Load data subjects
            subjects_file = self.storage_path / 'data_subjects.json'
            if subjects_file.exists():
                with open(subjects_file, 'r') as f:
                    subjects_data = json.load(f)
                    for subject_id, data in subjects_data.items():
                        self.data_subjects[subject_id] = DataSubject(**data)
                        
            # Load processing activities
            activities_file = self.storage_path / 'processing_activities.json'
            if activities_file.exists():
                with open(activities_file, 'r') as f:
                    activities_data = json.load(f)
                    self.processing_activities = [
                        ProcessingActivity(**activity) for activity in activities_data
                    ]
                    
        except Exception as e:
            print(f"Error loading GDPR data: {e}")
            
    def _save_data(self):
        """Save consent and processing records."""
        try:
            # Save data subjects
            subjects_file = self.storage_path / 'data_subjects.json'
            subjects_data = {
                subject_id: subject.to_dict() 
                for subject_id, subject in self.data_subjects.items()
            }
            with open(subjects_file, 'w') as f:
                json.dump(subjects_data, f, indent=2)
                
            # Save processing activities
            activities_file = self.storage_path / 'processing_activities.json'
            activities_data = [activity.to_dict() for activity in self.processing_activities]
            with open(activities_file, 'w') as f:
                json.dump(activities_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving GDPR data: {e}")
            
    def create_data_subject(
        self,
        subject_id: Optional[str] = None,
        email: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Create or update a data subject record.
        
        Parameters
        ----------
        subject_id : str, optional
            Unique identifier for the data subject
        email : str, optional
            Email address
        ip_address : str, optional
            IP address
        user_agent : str, optional
            User agent string
            
        Returns
        -------
        str
            Data subject ID
        """
        with self._lock:
            if subject_id is None:
                subject_id = str(uuid.uuid4())
                
            # Hash sensitive identifiers
            hashed_email = hashlib.sha256(email.encode()).hexdigest() if email else None
            hashed_ip = hashlib.sha256(ip_address.encode()).hexdigest() if ip_address else None
            
            subject = DataSubject(
                subject_id=subject_id,
                email=hashed_email,
                ip_address=hashed_ip,
                user_agent=user_agent
            )
            
            self.data_subjects[subject_id] = subject
            self._save_data()
            
            return subject_id
            
    def record_consent(
        self,
        subject_id: str,
        consent_type: Union[ConsentType, str],
        granted: bool,
        legal_basis: Union[LegalBasis, str] = LegalBasis.CONSENT
    ):
        """Record consent for data processing.
        
        Parameters
        ----------
        subject_id : str
            Data subject identifier
        consent_type : ConsentType or str
            Type of consent
        granted : bool
            Whether consent is granted
        legal_basis : LegalBasis or str, optional
            Legal basis for processing
        """
        with self._lock:
            if subject_id not in self.data_subjects:
                raise ValueError(f"Data subject {subject_id} not found")
                
            consent_key = consent_type.value if isinstance(consent_type, ConsentType) else consent_type
            legal_basis_key = legal_basis.value if isinstance(legal_basis, LegalBasis) else legal_basis
            
            self.data_subjects[subject_id].consent_status[consent_key] = granted
            self.data_subjects[subject_id].legal_basis[consent_key] = legal_basis_key
            
            # Record the consent change as a processing activity
            self.record_processing_activity(
                subject_id=subject_id,
                activity_type='consent_change',
                data_categories=['consent_records'],
                processing_purpose=f'Recording consent for {consent_key}',
                legal_basis=LegalBasis.LEGAL_OBLIGATION,
                retention_period=2555  # 7 years as required by GDPR
            )
            
            self._save_data()
            
    def check_consent(
        self,
        subject_id: str,
        consent_type: Union[ConsentType, str]
    ) -> bool:
        """Check if consent has been granted.
        
        Parameters
        ----------
        subject_id : str
            Data subject identifier
        consent_type : ConsentType or str
            Type of consent to check
            
        Returns
        -------
        bool
            True if consent granted
        """
        with self._lock:
            if subject_id not in self.data_subjects:
                return False
                
            consent_key = consent_type.value if isinstance(consent_type, ConsentType) else consent_type
            return self.data_subjects[subject_id].consent_status.get(consent_key, False)
            
    def record_processing_activity(
        self,
        subject_id: str,
        activity_type: str,
        data_categories: List[str],
        processing_purpose: str,
        legal_basis: Union[LegalBasis, str],
        retention_period: Optional[int] = None,
        data_content: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a data processing activity.
        
        Parameters
        ----------
        subject_id : str
            Data subject identifier
        activity_type : str
            Type of processing activity
        data_categories : List[str]
            Categories of data processed
        processing_purpose : str
            Purpose of processing
        legal_basis : LegalBasis or str
            Legal basis for processing
        retention_period : int, optional
            Data retention period in days
        data_content : Dict[str, Any], optional
            Actual data being processed (for hashing)
            
        Returns
        -------
        str
            Activity ID
        """
        with self._lock:
            activity_id = str(uuid.uuid4())
            legal_basis_str = legal_basis.value if isinstance(legal_basis, LegalBasis) else legal_basis
            
            # Determine retention period
            if retention_period is None:
                # Use maximum retention period for involved data categories
                retention_period = max(
                    self.data_categories.get(cat, {}).get('retention_period', 30)
                    for cat in data_categories
                )
            
            # Hash data content if provided (for integrity verification)
            data_hash = None
            if data_content:
                content_str = json.dumps(data_content, sort_keys=True)
                data_hash = hashlib.sha256(content_str.encode()).hexdigest()
                
            activity = ProcessingActivity(
                activity_id=activity_id,
                subject_id=subject_id,
                activity_type=activity_type,
                data_categories=data_categories,
                processing_purpose=processing_purpose,
                legal_basis=legal_basis_str,
                data_retention_period=retention_period,
                timestamp=time.time(),
                data_hash=data_hash
            )
            
            self.processing_activities.append(activity)
            self._save_data()
            
            return activity_id
            
    def handle_data_subject_request(
        self,
        subject_id: str,
        request_type: str
    ) -> Dict[str, Any]:
        """Handle data subject rights requests (Article 15-22).
        
        Parameters
        ----------
        subject_id : str
            Data subject identifier
        request_type : str
            Type of request ('access', 'rectification', 'erasure', 'portability')
            
        Returns
        -------
        Dict[str, Any]
            Response to the request
        """
        with self._lock:
            if subject_id not in self.data_subjects:
                return {'error': 'Data subject not found', 'status': 'failed'}
                
            if request_type == 'access':
                # Right of access (Article 15)
                return self._handle_access_request(subject_id)
            elif request_type == 'rectification':
                # Right to rectification (Article 16)
                return self._handle_rectification_request(subject_id)
            elif request_type == 'erasure':
                # Right to erasure (Article 17)
                return self._handle_erasure_request(subject_id)
            elif request_type == 'portability':
                # Right to data portability (Article 20)
                return self._handle_portability_request(subject_id)
            else:
                return {'error': f'Unknown request type: {request_type}', 'status': 'failed'}
                
    def _handle_access_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle subject access request."""
        subject = self.data_subjects[subject_id]
        activities = [
            activity for activity in self.processing_activities
            if activity.subject_id == subject_id
        ]
        
        return {
            'status': 'completed',
            'subject_data': subject.to_dict(),
            'processing_activities': [activity.to_dict() for activity in activities],
            'data_categories': list(self.data_categories.keys()),
            'legal_bases': list(set(activity.legal_basis for activity in activities)),
            'retention_information': {
                category: info['retention_period']
                for category, info in self.data_categories.items()
            }
        }
        
    def _handle_rectification_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle rectification request."""
        # In practice, this would involve updating incorrect data
        # For this implementation, we'll mark the request as received
        self.record_processing_activity(
            subject_id=subject_id,
            activity_type='rectification_request',
            data_categories=['personal_identifiers'],
            processing_purpose='Data rectification request processing',
            legal_basis=LegalBasis.LEGAL_OBLIGATION
        )
        
        return {
            'status': 'received',
            'message': 'Rectification request recorded and will be processed within 30 days',
            'next_steps': [
                'Verify identity of data subject',
                'Identify data to be corrected',
                'Update records and notify third parties if applicable'
            ]
        }
        
    def _handle_erasure_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle right to be forgotten request."""
        # Record the erasure request
        self.record_processing_activity(
            subject_id=subject_id,
            activity_type='erasure_request',
            data_categories=['all'],
            processing_purpose='Data erasure request processing',
            legal_basis=LegalBasis.LEGAL_OBLIGATION
        )
        
        # In practice, this would trigger actual data deletion
        # For now, we'll mark for deletion but keep audit trail
        subject = self.data_subjects[subject_id]
        subject.metadata = subject.metadata or {}
        subject.metadata['erasure_requested'] = time.time()
        subject.metadata['status'] = 'pending_erasure'
        
        return {
            'status': 'received',
            'message': 'Erasure request recorded and will be processed within 30 days',
            'retention_exemptions': [
                'Legal compliance requirements',
                'Freedom of expression and information',
                'Public interest in the area of public health'
            ]
        }
        
    def _handle_portability_request(self, subject_id: str) -> Dict[str, Any]:
        """Handle data portability request."""
        # Get all data provided by the subject
        activities = [
            activity for activity in self.processing_activities
            if activity.subject_id == subject_id and 
               activity.legal_basis in ['consent', 'contract']
        ]
        
        portable_data = {
            'subject_id': subject_id,
            'data_export_timestamp': time.time(),
            'activities': [
                {
                    'activity_type': activity.activity_type,
                    'timestamp': activity.timestamp,
                    'data_categories': activity.data_categories,
                    'processing_purpose': activity.processing_purpose
                }
                for activity in activities
            ]
        }
        
        return {
            'status': 'completed',
            'export_format': 'JSON',
            'data': portable_data
        }
        
    def cleanup_expired_data(self) -> Dict[str, Any]:
        """Clean up data that has exceeded retention periods.
        
        Returns
        -------
        Dict[str, Any]
            Cleanup report
        """
        with self._lock:
            current_time = time.time()
            cleanup_count = 0
            
            # Find expired activities
            expired_activities = []
            remaining_activities = []
            
            for activity in self.processing_activities:
                retention_seconds = activity.data_retention_period * 24 * 3600
                if current_time - activity.timestamp > retention_seconds:
                    expired_activities.append(activity)
                    cleanup_count += 1
                else:
                    remaining_activities.append(activity)
                    
            # Update activities list
            self.processing_activities = remaining_activities
            
            # Clean up data subjects with no recent activities
            active_subjects = set(activity.subject_id for activity in remaining_activities)
            subjects_to_remove = []
            
            for subject_id, subject in self.data_subjects.items():
                if (subject_id not in active_subjects and 
                    current_time - subject.created_at > 365 * 24 * 3600):  # 1 year
                    subjects_to_remove.append(subject_id)
                    
            for subject_id in subjects_to_remove:
                del self.data_subjects[subject_id]
                
            self._save_data()
            
            return {
                'cleanup_timestamp': current_time,
                'expired_activities': len(expired_activities),
                'removed_subjects': len(subjects_to_remove),
                'remaining_activities': len(remaining_activities),
                'remaining_subjects': len(self.data_subjects)
            }
            
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate GDPR compliance report.
        
        Returns
        -------
        Dict[str, Any]
            Compliance report
        """
        with self._lock:
            current_time = time.time()
            
            # Analyze consent patterns
            consent_stats = {}
            for consent_type in ConsentType:
                granted_count = sum(
                    1 for subject in self.data_subjects.values()
                    if subject.consent_status.get(consent_type.value, False)
                )
                consent_stats[consent_type.value] = {
                    'granted': granted_count,
                    'total': len(self.data_subjects)
                }
                
            # Analyze processing activities
            activity_stats = {}
            for activity in self.processing_activities:
                if activity.activity_type not in activity_stats:
                    activity_stats[activity.activity_type] = 0
                activity_stats[activity.activity_type] += 1
                
            # Check for data retention compliance
            retention_issues = []
            for activity in self.processing_activities:
                retention_seconds = activity.data_retention_period * 24 * 3600
                if current_time - activity.timestamp > retention_seconds:
                    retention_issues.append({
                        'activity_id': activity.activity_id,
                        'age_days': (current_time - activity.timestamp) / (24 * 3600),
                        'retention_limit_days': activity.data_retention_period
                    })
                    
            return {
                'report_timestamp': current_time,
                'total_data_subjects': len(self.data_subjects),
                'total_processing_activities': len(self.processing_activities),
                'consent_statistics': consent_stats,
                'activity_statistics': activity_stats,
                'retention_compliance': {
                    'total_activities': len(self.processing_activities),
                    'overdue_for_deletion': len(retention_issues),
                    'compliance_rate': (len(self.processing_activities) - len(retention_issues)) / max(1, len(self.processing_activities))
                },
                'data_categories': list(self.data_categories.keys()),
                'legal_bases_used': list(set(
                    basis for subject in self.data_subjects.values()
                    for basis in subject.legal_basis.values()
                ))
            }


# Global consent manager
_global_consent_manager = None


def get_consent_manager() -> ConsentManager:
    """Get global consent manager instance."""
    global _global_consent_manager
    
    if _global_consent_manager is None:
        _global_consent_manager = ConsentManager()
        
    return _global_consent_manager


def gdpr_compliant_processing(
    subject_id: str,
    activity_type: str,
    data_categories: List[str],
    processing_purpose: str,
    legal_basis: LegalBasis = LegalBasis.LEGITIMATE_INTERESTS
):
    """Decorator for GDPR-compliant data processing.
    
    Parameters
    ----------
    subject_id : str
        Data subject identifier
    activity_type : str
        Type of processing activity
    data_categories : List[str]
        Categories of data being processed
    processing_purpose : str
        Purpose of processing
    legal_basis : LegalBasis, optional
        Legal basis for processing
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            consent_manager = get_consent_manager()
            
            # Record processing activity
            activity_id = consent_manager.record_processing_activity(
                subject_id=subject_id,
                activity_type=activity_type,
                data_categories=data_categories,
                processing_purpose=processing_purpose,
                legal_basis=legal_basis
            )
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Log error in processing activity
                print(f"Error in GDPR-compliant processing {activity_id}: {e}")
                raise
                
        return wrapper
    return decorator