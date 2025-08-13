"""Comprehensive configuration management with validation and environment support."""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from .exceptions import ConfigurationError, ErrorCode, ValidationError
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "diffhe"
    username: str = "diffhe_user"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    ssl_mode: str = "prefer"


@dataclass
class CacheConfig:
    """Cache configuration."""

    type: str = "memory"  # memory, redis, memcached
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    default_timeout: int = 300
    max_entries: int = 10000


@dataclass
class SecurityConfig:
    """Security configuration."""

    secret_key: str = ""
    jwt_secret_key: str = ""
    jwt_expiration_hours: int = 24
    bcrypt_rounds: int = 12
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600  # seconds
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size_mb: int = 100
    enable_csrf_protection: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "./logs"
    max_file_size_mb: int = 50
    backup_count: int = 5
    enable_structured_logging: bool = True
    enable_performance_logging: bool = True
    security_log_level: str = "WARNING"


@dataclass
class ComputeConfig:
    """Compute configuration."""

    backend: str = "jax"
    fallback_backends: List[str] = field(default_factory=lambda: ["torch", "numpy"])
    enable_gpu: bool = True
    memory_limit_mb: Optional[int] = None
    thread_count: Optional[int] = None
    enable_jit: bool = True
    precision: str = "float64"


@dataclass
class SolverConfig:
    """Solver configuration."""

    default_method: str = "newton"
    max_iterations: int = 100
    tolerance: float = 1e-8
    linear_solver: str = "lu"
    preconditioner: str = "ilu"
    enable_monitoring: bool = True
    memory_efficient: bool = False
    timeout_minutes: Optional[int] = None


@dataclass
class OptimizationConfig:
    """Optimization configuration."""

    default_method: str = "L-BFGS-B"
    max_iterations: int = 1000
    tolerance: float = 1e-8
    gradient_tolerance: float = 1e-6
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 10
    checkpoint_dir: str = "./checkpoints"
    timeout_hours: Optional[int] = None


@dataclass
class APIConfig:
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    workers: int = 4
    max_content_length: int = 100 * 1024 * 1024  # 100MB
    request_timeout: int = 300  # seconds
    enable_docs: bool = True
    docs_url: str = "/docs"
    health_check_url: str = "/health"


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    tracing_endpoint: str = ""
    alert_email: str = ""
    alert_webhooks: List[str] = field(default_factory=list)
    performance_threshold_ms: float = 1000.0
    memory_threshold_mb: float = 1000.0


@dataclass
class DiffHEConfig:
    """Main DiffFE-Physics-Lab configuration."""

    # Core settings
    environment: str = "development"
    debug: bool = False
    version: str = "1.0.0"

    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)


class ConfigValidator:
    """Validator for configuration values."""

    def __init__(self):
        self.validators = {
            "environment": self._validate_environment,
            "port": self._validate_port,
            "email": self._validate_email,
            "url": self._validate_url,
            "path": self._validate_path,
            "positive_int": self._validate_positive_int,
            "positive_float": self._validate_positive_float,
            "boolean": self._validate_boolean,
            "secret_key": self._validate_secret_key,
        }

    def validate_config(self, config: DiffHEConfig) -> List[str]:
        """Validate entire configuration.

        Parameters
        ----------
        config : DiffHEConfig
            Configuration to validate

        Returns
        -------
        List[str]
            List of validation errors
        """
        errors = []

        # Validate environment
        if not self._validate_environment(config.environment):
            errors.append(f"Invalid environment: {config.environment}")

        # Validate database config
        errors.extend(self._validate_database_config(config.database))

        # Validate security config
        errors.extend(self._validate_security_config(config.security))

        # Validate API config
        errors.extend(self._validate_api_config(config.api))

        # Validate solver config
        errors.extend(self._validate_solver_config(config.solver))

        # Validate optimization config
        errors.extend(self._validate_optimization_config(config.optimization))

        # Validate compute config
        errors.extend(self._validate_compute_config(config.compute))

        return errors

    def _validate_environment(self, env: str) -> bool:
        """Validate environment setting."""
        valid_envs = {"development", "testing", "staging", "production"}
        return env.lower() in valid_envs

    def _validate_port(self, port: int) -> bool:
        """Validate port number."""
        return 1 <= port <= 65535

    def _validate_email(self, email: str) -> bool:
        """Validate email address."""
        if not email:
            return True  # Optional field
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    def _validate_url(self, url: str) -> bool:
        """Validate URL format."""
        if not url:
            return True  # Optional field
        pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        return bool(re.match(pattern, url))

    def _validate_path(self, path: str) -> bool:
        """Validate file path."""
        try:
            Path(path)
            return True
        except Exception:
            return False

    def _validate_positive_int(self, value: int) -> bool:
        """Validate positive integer."""
        return isinstance(value, int) and value > 0

    def _validate_positive_float(self, value: float) -> bool:
        """Validate positive float."""
        return isinstance(value, (int, float)) and value > 0

    def _validate_boolean(self, value: bool) -> bool:
        """Validate boolean value."""
        return isinstance(value, bool)

    def _validate_secret_key(self, key: str) -> bool:
        """Validate secret key."""
        return len(key) >= 32 if key else True

    def _validate_database_config(self, db_config: DatabaseConfig) -> List[str]:
        """Validate database configuration."""
        errors = []

        if not self._validate_port(db_config.port):
            errors.append(f"Invalid database port: {db_config.port}")

        if not db_config.host:
            errors.append("Database host cannot be empty")

        if not db_config.database:
            errors.append("Database name cannot be empty")

        if db_config.pool_size < 1:
            errors.append(f"Invalid pool size: {db_config.pool_size}")

        return errors

    def _validate_security_config(self, security_config: SecurityConfig) -> List[str]:
        """Validate security configuration."""
        errors = []

        if not self._validate_secret_key(security_config.secret_key):
            errors.append("Secret key must be at least 32 characters long")

        if not self._validate_secret_key(security_config.jwt_secret_key):
            errors.append("JWT secret key must be at least 32 characters long")

        if security_config.jwt_expiration_hours < 1:
            errors.append(
                f"Invalid JWT expiration: {security_config.jwt_expiration_hours}"
            )

        if not (4 <= security_config.bcrypt_rounds <= 20):
            errors.append(
                f"Bcrypt rounds should be 4-20: {security_config.bcrypt_rounds}"
            )

        return errors

    def _validate_api_config(self, api_config: APIConfig) -> List[str]:
        """Validate API configuration."""
        errors = []

        if not self._validate_port(api_config.port):
            errors.append(f"Invalid API port: {api_config.port}")

        if api_config.workers < 1:
            errors.append(f"Invalid worker count: {api_config.workers}")

        if api_config.max_content_length < 1024:  # At least 1KB
            errors.append(
                f"Max content length too small: {api_config.max_content_length}"
            )

        return errors

    def _validate_solver_config(self, solver_config: SolverConfig) -> List[str]:
        """Validate solver configuration."""
        errors = []

        valid_methods = {"newton", "gmres", "cg", "direct"}
        if solver_config.default_method not in valid_methods:
            errors.append(f"Invalid solver method: {solver_config.default_method}")

        if solver_config.tolerance <= 0 or solver_config.tolerance >= 1:
            errors.append(f"Invalid tolerance: {solver_config.tolerance}")

        if solver_config.max_iterations < 1:
            errors.append(f"Invalid max iterations: {solver_config.max_iterations}")

        return errors

    def _validate_optimization_config(
        self, opt_config: OptimizationConfig
    ) -> List[str]:
        """Validate optimization configuration."""
        errors = []

        valid_methods = {"L-BFGS-B", "BFGS", "CG", "Newton-CG", "SLSQP"}
        if opt_config.default_method not in valid_methods:
            errors.append(f"Invalid optimization method: {opt_config.default_method}")

        if opt_config.tolerance <= 0:
            errors.append(f"Invalid tolerance: {opt_config.tolerance}")

        if opt_config.checkpoint_frequency < 1:
            errors.append(
                f"Invalid checkpoint frequency: {opt_config.checkpoint_frequency}"
            )

        if not self._validate_path(opt_config.checkpoint_dir):
            errors.append(f"Invalid checkpoint directory: {opt_config.checkpoint_dir}")

        return errors

    def _validate_compute_config(self, compute_config: ComputeConfig) -> List[str]:
        """Validate compute configuration."""
        errors = []

        valid_backends = {"jax", "torch", "numpy"}
        if compute_config.backend not in valid_backends:
            errors.append(f"Invalid backend: {compute_config.backend}")

        for backend in compute_config.fallback_backends:
            if backend not in valid_backends:
                errors.append(f"Invalid fallback backend: {backend}")

        valid_precisions = {"float32", "float64"}
        if compute_config.precision not in valid_precisions:
            errors.append(f"Invalid precision: {compute_config.precision}")

        return errors


class ConfigManager:
    """Comprehensive configuration manager with environment support.

    Features:
    - Environment variable override
    - Multiple configuration formats (JSON, YAML)
    - Configuration validation
    - Schema enforcement
    - Default value management
    - Hot reloading support
    """

    def __init__(self, config_paths: Optional[List[str]] = None):
        self.config_paths = config_paths or [
            "./config.json",
            "./config.yaml",
            "./config.yml",
            os.path.expanduser("~/.diffhe/config.json"),
            "/etc/diffhe/config.json",
        ]
        self.validator = ConfigValidator()
        self._config = None
        self._config_file_path = None

        # Environment variable mappings
        self.env_mappings = {
            "DIFFHE_ENVIRONMENT": "environment",
            "DIFFHE_DEBUG": "debug",
            "DIFFHE_SECRET_KEY": "security.secret_key",
            "DIFFHE_JWT_SECRET": "security.jwt_secret_key",
            "DIFFHE_DATABASE_URL": "database.host",
            "DIFFHE_DATABASE_PORT": "database.port",
            "DIFFHE_DATABASE_NAME": "database.database",
            "DIFFHE_DATABASE_USER": "database.username",
            "DIFFHE_DATABASE_PASSWORD": "database.password",
            "DIFFHE_REDIS_URL": "cache.host",
            "DIFFHE_REDIS_PORT": "cache.port",
            "DIFFHE_API_HOST": "api.host",
            "DIFFHE_API_PORT": "api.port",
            "DIFFHE_LOG_LEVEL": "logging.level",
            "DIFFHE_LOG_DIR": "logging.log_dir",
            "DIFFHE_BACKEND": "compute.backend",
            "DIFFHE_ENABLE_GPU": "compute.enable_gpu",
            "DIFFHE_MEMORY_LIMIT": "compute.memory_limit_mb",
            "DIFFHE_CHECKPOINT_DIR": "optimization.checkpoint_dir",
        }

    def load_config(self, config_path: Optional[str] = None) -> DiffHEConfig:
        """Load configuration from file and environment.

        Parameters
        ----------
        config_path : str, optional
            Specific config file path

        Returns
        -------
        DiffHEConfig
            Loaded and validated configuration
        """
        # Start with default configuration
        config_data = asdict(DiffHEConfig())

        # Load from file if exists
        file_config = self._load_from_file(config_path)
        if file_config:
            config_data = self._deep_merge(config_data, file_config)

        # Override with environment variables
        env_config = self._load_from_environment()
        if env_config:
            config_data = self._deep_merge(config_data, env_config)

        # Create configuration object
        config = self._dict_to_dataclass(config_data, DiffHEConfig)

        # Validate configuration
        errors = self.validator.validate_config(config)
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ConfigurationError(
                error_msg,
                error_code=ErrorCode.CONFIGURATION_ERROR,
                context={"validation_errors": errors},
            )

        self._config = config
        logger.info(
            f"Configuration loaded successfully from {self._config_file_path or 'defaults and environment'}"
        )

        return config

    def _load_from_file(
        self, config_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        paths_to_try = [config_path] if config_path else self.config_paths

        for path in paths_to_try:
            if not path:
                continue

            config_file = Path(path)
            if not config_file.exists():
                continue

            try:
                logger.debug(f"Loading configuration from {path}")

                with open(config_file, "r") as f:
                    if config_file.suffix.lower() == ".json":
                        config_data = json.load(f)
                    elif config_file.suffix.lower() in [".yaml", ".yml"]:
                        if not HAS_YAML:
                            logger.warning(f"PyYAML not available, skipping {path}")
                            continue
                        config_data = yaml.safe_load(f)
                    else:
                        logger.warning(f"Unknown config file format: {path}")
                        continue

                self._config_file_path = str(config_file)
                logger.info(f"Loaded configuration from {path}")
                return config_data

            except Exception as e:
                logger.error(f"Error loading config from {path}: {e}")
                continue

        logger.info("No configuration file found, using defaults")
        return None

    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        for env_var, config_path in self.env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(env_value)

                # Set nested configuration value
                self._set_nested_value(env_config, config_path, converted_value)

                logger.debug(
                    f"Set config {config_path} from environment variable {env_var}"
                )

        return env_config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Integer values
        try:
            return int(value)
        except ValueError:
            pass

        # Float values
        try:
            return float(value)
        except ValueError:
            pass

        # String value
        return value

    def _set_nested_value(self, config_dict: Dict[str, Any], path: str, value: Any):
        """Set nested dictionary value using dot notation."""
        keys = path.split(".")
        current = config_dict

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _dict_to_dataclass(self, data: Dict[str, Any], dataclass_type: Type) -> Any:
        """Convert dictionary to dataclass instance."""
        if not is_dataclass(dataclass_type):
            return data

        # Get field types
        field_types = get_type_hints(dataclass_type)
        kwargs = {}

        for field in fields(dataclass_type):
            field_name = field.name
            field_type = field_types.get(field_name, field.type)

            if field_name in data:
                field_value = data[field_name]

                # Handle nested dataclasses
                if is_dataclass(field_type) and isinstance(field_value, dict):
                    kwargs[field_name] = self._dict_to_dataclass(
                        field_value, field_type
                    )
                else:
                    kwargs[field_name] = field_value

        return dataclass_type(**kwargs)

    def save_config(self, config: DiffHEConfig, output_path: str, format: str = "json"):
        """Save configuration to file.

        Parameters
        ----------
        config : DiffHEConfig
            Configuration to save
        output_path : str
            Output file path
        format : str, optional
            Output format ('json' or 'yaml')
        """
        config_dict = asdict(config)

        try:
            with open(output_path, "w") as f:
                if format.lower() == "json":
                    json.dump(config_dict, f, indent=2, default=str)
                elif format.lower() in ["yaml", "yml"]:
                    if not HAS_YAML:
                        raise ImportError("PyYAML required for YAML output")
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Configuration saved to {output_path}")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {e}",
                error_code=ErrorCode.CONFIGURATION_ERROR,
                context={"output_path": output_path, "format": format},
                cause=e,
            )

    def get_config(self) -> Optional[DiffHEConfig]:
        """Get current configuration."""
        return self._config

    def validate_config_file(self, config_path: str) -> List[str]:
        """Validate a configuration file.

        Parameters
        ----------
        config_path : str
            Path to configuration file

        Returns
        -------
        List[str]
            List of validation errors
        """
        try:
            # Load configuration
            temp_manager = ConfigManager([config_path])
            config = temp_manager.load_config()

            # Validate
            return self.validator.validate_config(config)

        except Exception as e:
            return [f"Error loading configuration: {e}"]

    def create_sample_config(self, output_path: str, format: str = "json"):
        """Create a sample configuration file.

        Parameters
        ----------
        output_path : str
            Output file path
        format : str, optional
            Output format
        """
        sample_config = DiffHEConfig()

        # Set some example values
        sample_config.environment = "development"
        sample_config.debug = True
        sample_config.security.secret_key = "your-secret-key-here-min-32-characters"
        sample_config.security.jwt_secret_key = (
            "your-jwt-secret-key-here-min-32-characters"
        )
        sample_config.database.password = "your-database-password"

        self.save_config(sample_config, output_path, format)
        logger.info(f"Sample configuration created at {output_path}")

    def get_environment_variables_help(self) -> Dict[str, str]:
        """Get help for available environment variables.

        Returns
        -------
        Dict[str, str]
            Mapping of environment variables to their descriptions
        """
        descriptions = {
            "DIFFHE_ENVIRONMENT": "Application environment (development, testing, staging, production)",
            "DIFFHE_DEBUG": "Enable debug mode (true/false)",
            "DIFFHE_SECRET_KEY": "Application secret key (min 32 characters)",
            "DIFFHE_JWT_SECRET": "JWT signing secret key (min 32 characters)",
            "DIFFHE_DATABASE_URL": "Database host",
            "DIFFHE_DATABASE_PORT": "Database port",
            "DIFFHE_DATABASE_NAME": "Database name",
            "DIFFHE_DATABASE_USER": "Database username",
            "DIFFHE_DATABASE_PASSWORD": "Database password",
            "DIFFHE_REDIS_URL": "Redis host for caching",
            "DIFFHE_REDIS_PORT": "Redis port",
            "DIFFHE_API_HOST": "API server host",
            "DIFFHE_API_PORT": "API server port",
            "DIFFHE_LOG_LEVEL": "Logging level (DEBUG, INFO, WARNING, ERROR)",
            "DIFFHE_LOG_DIR": "Directory for log files",
            "DIFFHE_BACKEND": "Automatic differentiation backend (jax, torch, numpy)",
            "DIFFHE_ENABLE_GPU": "Enable GPU support (true/false)",
            "DIFFHE_MEMORY_LIMIT": "Memory limit in MB",
            "DIFFHE_CHECKPOINT_DIR": "Directory for optimization checkpoints",
        }

        return descriptions


# Global configuration manager instance
_global_config_manager = None
_global_config = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def load_global_config(config_path: Optional[str] = None) -> DiffHEConfig:
    """Load global configuration.

    Parameters
    ----------
    config_path : str, optional
        Specific config file path

    Returns
    -------
    DiffHEConfig
        Global configuration
    """
    global _global_config
    manager = get_config_manager()
    _global_config = manager.load_config(config_path)
    return _global_config


def get_global_config() -> Optional[DiffHEConfig]:
    """Get global configuration.

    Returns
    -------
    Optional[DiffHEConfig]
        Global configuration or None if not loaded
    """
    return _global_config


def reload_global_config(config_path: Optional[str] = None) -> DiffHEConfig:
    """Reload global configuration.

    Parameters
    ----------
    config_path : str, optional
        Specific config file path

    Returns
    -------
    DiffHEConfig
        Reloaded global configuration
    """
    global _global_config_manager
    _global_config_manager = None  # Force recreation
    return load_global_config(config_path)
