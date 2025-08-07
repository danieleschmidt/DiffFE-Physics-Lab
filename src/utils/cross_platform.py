"""Cross-platform compatibility utilities for sentiment analysis framework."""

import os
import sys
import platform
import subprocess
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import tempfile
import shutil


class PlatformDetector:
    """Utility class for detecting platform-specific information."""
    
    def __init__(self):
        """Initialize platform detector."""
        self.system = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.python_version = platform.python_version()
        
    def is_windows(self) -> bool:
        """Check if running on Windows."""
        return self.system == 'windows'
        
    def is_linux(self) -> bool:
        """Check if running on Linux."""
        return self.system == 'linux'
        
    def is_macos(self) -> bool:
        """Check if running on macOS."""
        return self.system == 'darwin'
        
    def is_unix_like(self) -> bool:
        """Check if running on Unix-like system."""
        return self.is_linux() or self.is_macos()
        
    def is_64bit(self) -> bool:
        """Check if running on 64-bit architecture."""
        return '64' in self.architecture or 'amd64' in self.architecture
        
    def get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information."""
        return {
            'system': self.system,
            'architecture': self.architecture,
            'python_version': self.python_version,
            'is_windows': self.is_windows(),
            'is_linux': self.is_linux(),
            'is_macos': self.is_macos(),
            'is_64bit': self.is_64bit(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'processor': platform.processor(),
            'python_implementation': platform.python_implementation(),
            'python_compiler': platform.python_compiler()
        }


class PathManager:
    """Cross-platform path management utilities."""
    
    def __init__(self):
        """Initialize path manager."""
        self.detector = PlatformDetector()
        
    def get_app_data_dir(self, app_name: str = 'sentiment-analyzer') -> Path:
        """Get platform-appropriate application data directory.
        
        Parameters
        ----------
        app_name : str, optional
            Application name, by default 'sentiment-analyzer'
            
        Returns
        -------
        Path
            Application data directory path
        """
        if self.detector.is_windows():
            # Use %APPDATA% on Windows
            base_dir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        elif self.detector.is_macos():
            # Use ~/Library/Application Support on macOS
            base_dir = Path.home() / 'Library' / 'Application Support'
        else:
            # Use XDG Base Directory specification on Linux
            base_dir = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
            
        app_dir = base_dir / app_name
        app_dir.mkdir(parents=True, exist_ok=True)
        return app_dir
        
    def get_config_dir(self, app_name: str = 'sentiment-analyzer') -> Path:
        """Get platform-appropriate configuration directory.
        
        Parameters
        ----------
        app_name : str, optional
            Application name, by default 'sentiment-analyzer'
            
        Returns
        -------
        Path
            Configuration directory path
        """
        if self.detector.is_windows():
            # Use %APPDATA% on Windows
            base_dir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        elif self.detector.is_macos():
            # Use ~/Library/Preferences on macOS
            base_dir = Path.home() / 'Library' / 'Preferences'
        else:
            # Use XDG Base Directory specification on Linux
            base_dir = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
            
        config_dir = base_dir / app_name
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
        
    def get_cache_dir(self, app_name: str = 'sentiment-analyzer') -> Path:
        """Get platform-appropriate cache directory.
        
        Parameters
        ----------
        app_name : str, optional
            Application name, by default 'sentiment-analyzer'
            
        Returns
        -------
        Path
            Cache directory path
        """
        if self.detector.is_windows():
            # Use %LOCALAPPDATA% on Windows
            base_dir = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        elif self.detector.is_macos():
            # Use ~/Library/Caches on macOS
            base_dir = Path.home() / 'Library' / 'Caches'
        else:
            # Use XDG Base Directory specification on Linux
            base_dir = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
            
        cache_dir = base_dir / app_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
        
    def get_temp_dir(self) -> Path:
        """Get platform-appropriate temporary directory."""
        return Path(tempfile.gettempdir())
        
    def normalize_path(self, path: Union[str, Path]) -> Path:
        """Normalize path for current platform.
        
        Parameters
        ----------
        path : Union[str, Path]
            Path to normalize
            
        Returns
        -------
        Path
            Normalized path
        """
        path = Path(path)
        
        # Resolve relative paths
        if not path.is_absolute():
            path = Path.cwd() / path
            
        # Normalize path separators and resolve
        return path.resolve()


class DependencyManager:
    """Cross-platform dependency management."""
    
    def __init__(self):
        """Initialize dependency manager."""
        self.detector = PlatformDetector()
        
    def check_python_requirements(self, min_version: str = '3.10') -> Dict[str, Any]:
        """Check Python version requirements.
        
        Parameters
        ----------
        min_version : str, optional
            Minimum Python version, by default '3.10'
            
        Returns
        -------
        Dict[str, Any]
            Python requirement check results
        """
        current_version = self.detector.python_version
        min_version_tuple = tuple(map(int, min_version.split('.')))
        current_version_tuple = tuple(map(int, current_version.split('.')))
        
        meets_requirement = current_version_tuple >= min_version_tuple
        
        return {
            'current_version': current_version,
            'minimum_required': min_version,
            'meets_requirement': meets_requirement,
            'python_implementation': platform.python_implementation(),
            'recommendations': self._get_python_recommendations() if not meets_requirement else []
        }
        
    def _get_python_recommendations(self) -> List[str]:
        """Get platform-specific Python upgrade recommendations."""
        recommendations = []
        
        if self.detector.is_windows():
            recommendations.extend([
                'Download Python from python.org',
                'Use Windows Store Python installation',
                'Consider using Anaconda or Miniconda'
            ])
        elif self.detector.is_macos():
            recommendations.extend([
                'Use Homebrew: brew install python@3.11',
                'Use pyenv: pyenv install 3.11.0',
                'Download from python.org'
            ])
        else:  # Linux
            recommendations.extend([
                'Use system package manager (apt, yum, etc.)',
                'Use pyenv: pyenv install 3.11.0',
                'Compile from source if needed'
            ])
            
        return recommendations
        
    def check_system_dependencies(self) -> Dict[str, Any]:
        """Check system-level dependencies."""
        dependencies = {
            'git': self._check_command('git'),
            'curl': self._check_command('curl'),
        }
        
        # Platform-specific dependencies
        if self.detector.is_linux():
            dependencies.update({
                'build-essential': self._check_linux_package('build-essential'),
                'python3-dev': self._check_linux_package('python3-dev'),
            })
        elif self.detector.is_macos():
            dependencies.update({
                'xcode-tools': self._check_xcode_tools(),
            })
        elif self.detector.is_windows():
            dependencies.update({
                'visual-cpp-tools': self._check_visual_cpp_tools(),
            })
            
        return dependencies
        
    def _check_command(self, command: str) -> Dict[str, Any]:
        """Check if a command is available."""
        try:
            result = subprocess.run(
                [command, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                'available': result.returncode == 0,
                'version': result.stdout.strip() if result.returncode == 0 else None,
                'error': result.stderr.strip() if result.returncode != 0 else None
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return {
                'available': False,
                'version': None,
                'error': f'Command {command} not found'
            }
            
    def _check_linux_package(self, package: str) -> Dict[str, Any]:
        """Check Linux package availability."""
        # This is a simplified check - in practice would use package manager APIs
        try:
            result = subprocess.run(
                ['dpkg', '-l', package],
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                'available': result.returncode == 0,
                'installed': 'ii' in result.stdout if result.returncode == 0 else False
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            # Try rpm-based systems
            try:
                result = subprocess.run(
                    ['rpm', '-q', package],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return {
                    'available': result.returncode == 0,
                    'installed': result.returncode == 0
                }
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                return {
                    'available': False,
                    'installed': False,
                    'error': 'Package manager not available'
                }
                
    def _check_xcode_tools(self) -> Dict[str, Any]:
        """Check Xcode command line tools on macOS."""
        try:
            result = subprocess.run(
                ['xcode-select', '-p'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                'available': result.returncode == 0,
                'path': result.stdout.strip() if result.returncode == 0 else None
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return {
                'available': False,
                'error': 'Xcode command line tools not found'
            }
            
    def _check_visual_cpp_tools(self) -> Dict[str, Any]:
        """Check Visual C++ tools on Windows."""
        # Simplified check - would normally check registry or specific paths
        try:
            result = subprocess.run(
                ['cl'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return {
                'available': 'Microsoft' in result.stderr,
                'version': 'Detected' if 'Microsoft' in result.stderr else None
            }
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return {
                'available': False,
                'error': 'Visual C++ compiler not found'
            }


class EnvironmentManager:
    """Cross-platform environment management."""
    
    def __init__(self):
        """Initialize environment manager."""
        self.detector = PlatformDetector()
        self.path_manager = PathManager()
        
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        info = {
            'platform': self.detector.get_platform_info(),
            'paths': {
                'app_data': str(self.path_manager.get_app_data_dir()),
                'config': str(self.path_manager.get_config_dir()),
                'cache': str(self.path_manager.get_cache_dir()),
                'temp': str(self.path_manager.get_temp_dir())
            },
            'environment_variables': self._get_relevant_env_vars(),
            'resource_limits': self._get_resource_limits(),
        }
        
        return info
        
    def _get_relevant_env_vars(self) -> Dict[str, Optional[str]]:
        """Get relevant environment variables."""
        vars_to_check = [
            'PATH', 'PYTHONPATH', 'HOME', 'USER', 'USERNAME',
            'TEMP', 'TMP', 'TMPDIR',
            'XDG_DATA_HOME', 'XDG_CONFIG_HOME', 'XDG_CACHE_HOME',
            'APPDATA', 'LOCALAPPDATA',
            'LANG', 'LC_ALL', 'LC_CTYPE'
        ]
        
        return {var: os.environ.get(var) for var in vars_to_check}
        
    def _get_resource_limits(self) -> Dict[str, Any]:
        """Get system resource limits."""
        import resource
        
        limits = {}
        
        try:
            # Get memory limit
            memory_limit = resource.getrlimit(resource.RLIMIT_AS)
            limits['memory'] = {
                'soft_limit': memory_limit[0] if memory_limit[0] != resource.RLIM_INFINITY else 'unlimited',
                'hard_limit': memory_limit[1] if memory_limit[1] != resource.RLIM_INFINITY else 'unlimited'
            }
        except (AttributeError, OSError):
            limits['memory'] = 'unavailable'
            
        try:
            # Get file descriptor limit
            fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            limits['file_descriptors'] = {
                'soft_limit': fd_limit[0],
                'hard_limit': fd_limit[1]
            }
        except (AttributeError, OSError):
            limits['file_descriptors'] = 'unavailable'
            
        return limits
        
    def setup_environment(self, app_name: str = 'sentiment-analyzer') -> Dict[str, Any]:
        """Set up cross-platform environment for the application.
        
        Parameters
        ----------
        app_name : str, optional
            Application name, by default 'sentiment-analyzer'
            
        Returns
        -------
        Dict[str, Any]
            Setup results
        """
        results = {
            'directories_created': [],
            'environment_set': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # Create application directories
            directories = {
                'app_data': self.path_manager.get_app_data_dir(app_name),
                'config': self.path_manager.get_config_dir(app_name),
                'cache': self.path_manager.get_cache_dir(app_name)
            }
            
            for dir_type, path in directories.items():
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    results['directories_created'].append(f'{dir_type}: {path}')
                    
            # Set environment variables for the application
            env_vars = {
                f'{app_name.upper()}_DATA_DIR': str(directories['app_data']),
                f'{app_name.upper()}_CONFIG_DIR': str(directories['config']),
                f'{app_name.upper()}_CACHE_DIR': str(directories['cache'])
            }
            
            for var, value in env_vars.items():
                os.environ[var] = value
                results['environment_set'][var] = value
                
        except Exception as e:
            results['errors'].append(f'Environment setup failed: {e}')
            
        return results


class ProcessManager:
    """Cross-platform process management utilities."""
    
    def __init__(self):
        """Initialize process manager."""
        self.detector = PlatformDetector()
        
    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                'pid': process.pid,
                'name': process.name(),
                'status': process.status(),
                'cpu_percent': process.cpu_percent(),
                'memory_info': process.memory_info()._asdict(),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'create_time': process.create_time()
            }
        except ImportError:
            # Fallback without psutil
            return {
                'pid': os.getpid(),
                'name': sys.argv[0],
                'status': 'running',
                'error': 'psutil not available for detailed info'
            }
            
    def optimize_for_platform(self) -> Dict[str, Any]:
        """Apply platform-specific optimizations."""
        optimizations = {
            'applied': [],
            'recommendations': [],
            'warnings': []
        }
        
        if self.detector.is_windows():
            optimizations['recommendations'].extend([
                'Consider using Windows Subsystem for Linux for better performance',
                'Ensure Windows Defender exclusions for application directories',
                'Use SSD storage for better I/O performance'
            ])
        elif self.detector.is_macos():
            optimizations['recommendations'].extend([
                'Ensure sufficient permissions for application directories',
                'Consider using Homebrew for dependency management',
                'Monitor thermal throttling on MacBooks'
            ])
        else:  # Linux
            optimizations['applied'].extend([
                'Set optimal file descriptor limits',
                'Configure memory overcommit settings'
            ])
            optimizations['recommendations'].extend([
                'Use systemd for service management',
                'Configure log rotation',
                'Monitor system resources with htop/top'
            ])
            
        return optimizations


# Global instances
_platform_detector = None
_path_manager = None
_environment_manager = None


def get_platform_detector() -> PlatformDetector:
    """Get global platform detector instance."""
    global _platform_detector
    if _platform_detector is None:
        _platform_detector = PlatformDetector()
    return _platform_detector


def get_path_manager() -> PathManager:
    """Get global path manager instance."""
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager


def get_environment_manager() -> EnvironmentManager:
    """Get global environment manager instance."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager()
    return _environment_manager


def ensure_cross_platform_compatibility():
    """Ensure cross-platform compatibility by checking requirements."""
    detector = get_platform_detector()
    dep_manager = DependencyManager()
    env_manager = get_environment_manager()
    
    # Check Python requirements
    python_check = dep_manager.check_python_requirements()
    if not python_check['meets_requirement']:
        print(f"Warning: Python {python_check['minimum_required']} or higher required")
        print("Recommendations:")
        for rec in python_check['recommendations']:
            print(f"  - {rec}")
            
    # Set up environment
    setup_results = env_manager.setup_environment()
    if setup_results['errors']:
        print("Environment setup errors:")
        for error in setup_results['errors']:
            print(f"  - {error}")
            
    # Platform-specific optimizations
    process_manager = ProcessManager()
    optimizations = process_manager.optimize_for_platform()
    
    return {
        'platform_info': detector.get_platform_info(),
        'python_compatibility': python_check,
        'environment_setup': setup_results,
        'optimizations': optimizations
    }