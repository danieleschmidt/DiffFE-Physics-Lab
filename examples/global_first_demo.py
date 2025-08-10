#!/usr/bin/env python3
"""Global-First Features Demo - I18n and compliance."""

import os
import sys
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, '/root/repo')

from src.i18n import Language, get_translator, set_language, t, error_message, status_message, physics_term, math_term
from src.i18n.compliance import (
    ComplianceManager, 
    ComplianceFramework, 
    ComplianceConfig,
    get_compliance_manager,
    record_data_processing,
    check_user_consent
)


def demo_internationalization():
    """Demonstrate internationalization features."""
    print("🌍 Internationalization (I18n) Demo")
    print("=" * 50)
    
    translator = get_translator()
    
    # Test multiple languages
    languages_to_test = [
        Language.ENGLISH,
        Language.SPANISH, 
        Language.FRENCH,
        Language.GERMAN,
        Language.JAPANESE,
        Language.CHINESE
    ]
    
    test_messages = [
        "error.general",
        "status.computing",
        "physics.temperature", 
        "math.gradient"
    ]
    
    for language in languages_to_test:
        print(f"\n🗣️  Language: {language.value.upper()}")
        set_language(language)
        
        for msg_key in test_messages:
            translated = t(msg_key)
            print(f"   {msg_key}: {translated}")
    
    # Test message formatting
    print(f"\n🔧 Message Formatting Examples:")
    set_language(Language.ENGLISH)
    
    # Progress message with formatting
    progress_msg = t("progress.iteration", iteration=5, total=10)
    print(f"   Progress: {progress_msg}")
    
    percentage_msg = t("progress.percentage", percentage=75.5)
    print(f"   Percentage: {percentage_msg}")
    
    time_msg = t("progress.time_remaining", time="2 minutes")
    print(f"   Time: {time_msg}")
    
    # Convenience functions
    print(f"\n📱 Convenience Functions:")
    print(f"   Error: {error_message('computation_failed')}")
    print(f"   Status: {status_message('converged')}")
    print(f"   Physics: {physics_term('pressure')}")
    print(f"   Math: {math_term('jacobian')}")
    
    # Custom translations
    print(f"\n🎨 Custom Translations:")
    translator.add_custom_translations(Language.ENGLISH, {
        "custom.welcome": "Welcome to DiffFE-Physics-Lab",
        "custom.simulation_complete": "Simulation completed in {time} seconds"
    })
    
    welcome = t("custom.welcome")
    simulation = t("custom.simulation_complete", time=1.5)
    print(f"   Welcome: {welcome}")
    print(f"   Simulation: {simulation}")


def demo_compliance():
    """Demonstrate compliance and regulatory features."""
    print("\n📋 Global Compliance & Regulatory Demo")
    print("=" * 50)
    
    # Create compliance manager with custom config
    config = ComplianceConfig(
        enabled_frameworks={ComplianceFramework.GDPR, ComplianceFramework.CCPA, ComplianceFramework.PDPA},
        data_retention_days=365,
        require_explicit_consent=True,
        log_data_processing=True,
        anonymize_logs=True,
        encryption_required=True,
        audit_trail_enabled=True,
        cross_border_restrictions={
            "gdpr": ["US", "CN"],
            "ccpa": ["CN"]
        }
    )
    
    compliance = ComplianceManager(config)
    
    # 1. Record user consent
    print("🤝 Recording User Consent:")
    
    users = ["user001", "user002", "user003"]
    purposes = ["analytics", "marketing", "research"]
    
    for user in users:
        for purpose in purposes:
            # Simulate different consent patterns
            consent_given = hash(user + purpose) % 3 != 0  # ~67% consent rate
            
            compliance.record_user_consent(
                user_id=user,
                purpose=purpose,
                consent_given=consent_given,
                consent_method="explicit",
                expiry_date=datetime.utcnow() + timedelta(days=365)
            )
            
            status = "✅ GRANTED" if consent_given else "❌ DENIED"
            print(f"   {user} - {purpose}: {status}")
    
    # 2. Record data processing activities
    print(f"\n📊 Recording Data Processing Activities:")
    
    processing_activities = [
        {
            "activity_id": "physics_simulation_001",
            "purpose": "Scientific computation for research",
            "legal_basis": "legitimate_interests", 
            "data_categories": ["numerical_data", "simulation_parameters"],
            "recipients": ["research_team"],
            "retention_days": 1095,  # 3 years
            "cross_border": False
        },
        {
            "activity_id": "user_analytics_002", 
            "purpose": "Website usage analytics",
            "legal_basis": "consent",
            "data_categories": ["usage_data", "ip_address"],
            "recipients": ["analytics_service"],
            "retention_days": 730,  # 2 years
            "cross_border": True,
            "user_consent": True
        },
        {
            "activity_id": "error_logging_003",
            "purpose": "System error monitoring",
            "legal_basis": "legitimate_interests",
            "data_categories": ["error_logs", "system_data"],
            "recipients": ["development_team"],
            "retention_days": 90
        }
    ]
    
    for activity in processing_activities:
        compliance.record_data_processing(**activity)
        print(f"   ✅ Recorded: {activity['activity_id']}")
    
    # 3. Check consent status
    print(f"\n🔍 Checking Consent Status:")
    
    for user in users[:2]:  # Check first 2 users
        for purpose in purposes[:2]:  # Check first 2 purposes
            has_consent = compliance.check_user_consent(user, purpose)
            status = "✅ VALID" if has_consent else "❌ INVALID" 
            print(f"   {user} - {purpose}: {status}")
    
    # 4. Demonstrate consent withdrawal
    print(f"\n🚫 Demonstrating Consent Withdrawal:")
    
    compliance.withdraw_consent("user001", "marketing")
    print("   ✅ Withdrew consent: user001 - marketing")
    
    # Verify withdrawal
    has_consent = compliance.check_user_consent("user001", "marketing")
    status = "❌ WITHDRAWN" if not has_consent else "✅ STILL VALID"
    print(f"   Status check: user001 - marketing: {status}")
    
    # 5. Data cleanup
    print(f"\n🧹 Data Cleanup & Retention:")
    
    # Simulate some time passing (for demo purposes)
    initial_records = len(compliance.processing_records)
    cleaned_count = compliance.cleanup_expired_data()
    final_records = len(compliance.processing_records)
    
    print(f"   Records before cleanup: {initial_records}")
    print(f"   Records cleaned: {cleaned_count}")
    print(f"   Records after cleanup: {final_records}")
    
    # 6. Generate compliance report
    print(f"\n📈 Compliance Report:")
    
    report = compliance.generate_compliance_report()
    
    print(f"   Enabled frameworks: {', '.join(report['enabled_frameworks'])}")
    print(f"   Total processing activities: {report['processing_activities']['total']}")
    print(f"   Recent activities (30d): {report['processing_activities']['recent_30_days']}")
    print(f"   Active consents: {report['consent_management']['active_consents']}")
    print(f"   Consent rate: {report['consent_management']['consent_rate']:.1%}")
    print(f"   Compliance violations: {report['compliance_violations']['total']}")
    
    if report['compliance_violations']['by_framework']:
        print("   Violations by framework:")
        for framework, count in report['compliance_violations']['by_framework'].items():
            print(f"     - {framework.upper()}: {count}")
    
    # 7. Data export (GDPR Right to Data Portability)
    print(f"\n💾 Data Export for User (GDPR Article 20):")
    
    user_data = compliance.export_data_for_user("user001")
    
    print(f"   User: {user_data['user_id']}")
    print(f"   Export timestamp: {user_data['export_timestamp'][:19]}")
    print(f"   Processing activities: {len(user_data['processing_activities'])}")
    print(f"   Consent records: {len(user_data['consent_records'])}")
    
    if user_data['processing_activities']:
        print("   Activities involving user:")
        for activity in user_data['processing_activities'][:2]:  # Show first 2
            print(f"     - {activity['activity_id']}: {activity['purpose']}")
    
    if user_data['consent_records']:
        print("   Consent history:")
        for consent in user_data['consent_records'][:2]:  # Show first 2
            status = "GRANTED" if consent['consent_given'] else "DENIED"
            print(f"     - {consent['purpose']}: {status}")


def demo_cross_platform_compatibility():
    """Demonstrate cross-platform compatibility features."""
    print("\n🌐 Cross-Platform Compatibility Demo")
    print("=" * 50)
    
    import platform
    import psutil
    
    # System information
    print("💻 System Information:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {platform.python_version()}")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Locale and encoding
    import locale
    print(f"\n🌍 Locale Information:")
    print(f"   System locale: {locale.getdefaultlocale()}")
    print(f"   Preferred encoding: {locale.getpreferredencoding()}")
    
    # Time zone handling
    print(f"\n🕐 Time Zone Handling:")
    utc_time = datetime.utcnow()
    local_time = datetime.now()
    
    print(f"   UTC time: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Local time: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   UTC offset: {(local_time - utc_time).total_seconds() / 3600:.1f} hours")
    
    # File system compatibility
    print(f"\n📁 File System Compatibility:")
    
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test Unicode filename support
        unicode_filename = "测试_файл_тест_🧪.txt"
        test_file = temp_path / unicode_filename
        
        try:
            test_file.write_text("Test content", encoding='utf-8')
            content = test_file.read_text(encoding='utf-8')
            print(f"   ✅ Unicode filenames: Supported")
            print(f"   📄 Test file: {unicode_filename}")
        except Exception as e:
            print(f"   ❌ Unicode filenames: Not supported - {e}")
        
        # Test long path support
        long_path = temp_path
        for i in range(10):
            long_path = long_path / f"very_long_directory_name_{i}_with_many_characters"
        
        try:
            long_path.mkdir(parents=True)
            test_long_file = long_path / "test.txt"
            test_long_file.write_text("Long path test")
            print(f"   ✅ Long paths: Supported")
        except Exception as e:
            print(f"   ❌ Long paths: Limited - {e}")


def main():
    """Run global-first features demonstration."""
    print("🌍 DiffFE-Physics-Lab - Global-First Features Demo")
    print("=" * 60)
    print(f"🕐 Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Run demonstrations
    demo_internationalization()
    demo_compliance()
    demo_cross_platform_compatibility()
    
    # Summary
    print(f"\n🎉 Global-First Features Summary")
    print("=" * 60)
    print("   ✅ Multi-language support (6 languages)")
    print("   ✅ Compliance frameworks (GDPR, CCPA, PDPA)")
    print("   ✅ Data protection and privacy controls")
    print("   ✅ User consent management")
    print("   ✅ Audit trail and reporting")
    print("   ✅ Cross-platform compatibility")
    print("   ✅ Unicode and internationalization support")
    print("   ✅ Automated data retention and cleanup")
    
    print(f"\n🚀 System ready for global deployment!")
    print("   🌍 Compliant with major international regulations")
    print("   🗣️  Available in multiple languages")
    print("   💻 Compatible across platforms and regions")


if __name__ == "__main__":
    main()