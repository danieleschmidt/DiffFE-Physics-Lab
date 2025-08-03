"""Database connection management."""

import os
import sqlite3
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from contextlib import contextmanager

try:
    import psycopg2
    import psycopg2.extras
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection and transaction manager.
    
    Supports SQLite for lightweight storage and PostgreSQL for production.
    Also provides HDF5 integration for large scientific datasets.
    
    Parameters
    ----------
    database_url : str, optional
        Database connection URL, by default uses SQLite
    pool_size : int, optional
        Connection pool size for PostgreSQL, by default 5
    """
    
    def __init__(self, database_url: str = None, pool_size: int = 5):
        self.database_url = database_url or self._default_url()
        self.pool_size = pool_size
        self._connection = None
        self._pool = None
        
        # Parse database URL
        self.db_type = self._parse_db_type(self.database_url)
        
        # Initialize based on database type
        if self.db_type == 'postgresql' and not HAS_POSTGRES:
            raise ImportError("PostgreSQL support requires psycopg2: pip install psycopg2-binary")
    
    def _default_url(self) -> str:
        """Get default database URL from environment or use SQLite."""
        db_url = os.getenv('DIFFHE_DATABASE_URL')
        if db_url:
            return db_url
        
        # Default to SQLite in cache directory
        cache_dir = Path(os.getenv('DIFFHE_CACHE_PATH', '.cache'))
        cache_dir.mkdir(exist_ok=True)
        return f"sqlite:///{cache_dir}/diffhe.db"
    
    def _parse_db_type(self, url: str) -> str:
        """Parse database type from URL."""
        if url.startswith('sqlite://'):
            return 'sqlite'
        elif url.startswith('postgresql://'):
            return 'postgresql'
        else:
            raise ValueError(f"Unsupported database URL: {url}")
    
    def connect(self) -> None:
        """Establish database connection."""
        if self._connection is not None:
            return
        
        if self.db_type == 'sqlite':
            db_path = self.database_url.replace('sqlite:///', '')
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            self._connection = sqlite3.connect(
                db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._connection.row_factory = sqlite3.Row
            
            # Enable foreign keys
            self._connection.execute('PRAGMA foreign_keys = ON')
            
        elif self.db_type == 'postgresql':
            self._connection = psycopg2.connect(
                self.database_url,
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            self._connection.autocommit = False
        
        logger.info(f"Connected to {self.db_type} database")
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    @contextmanager
    def transaction(self):
        """Database transaction context manager.
        
        Yields
        ------
        connection
            Database connection with active transaction
        """
        if not self._connection:
            self.connect()
        
        try:
            if self.db_type == 'sqlite':
                yield self._connection
                self._connection.commit()
            elif self.db_type == 'postgresql':
                with self._connection:
                    yield self._connection
        except Exception as e:
            if self._connection:
                self._connection.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
    
    def execute(self, query: str, params: tuple = None) -> Any:
        """Execute a database query.
        
        Parameters
        ----------
        query : str
            SQL query
        params : tuple, optional
            Query parameters
            
        Returns
        -------
        Any
            Query result
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                return cursor.rowcount
    
    def execute_many(self, query: str, params_list: list) -> int:
        """Execute query with multiple parameter sets.
        
        Parameters
        ----------
        query : str
            SQL query
        params_list : list
            List of parameter tuples
            
        Returns
        -------
        int
            Total number of affected rows
        """
        with self.transaction() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount
    
    def create_hdf5_store(self, file_path: str) -> 'HDF5Store':
        """Create HDF5 data store for large datasets.
        
        Parameters
        ----------
        file_path : str
            Path to HDF5 file
            
        Returns
        -------
        HDF5Store
            HDF5 data store instance
        """
        if not HAS_HDF5:
            raise ImportError("HDF5 support requires h5py: pip install h5py")
        
        return HDF5Store(file_path)
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class HDF5Store:
    """HDF5 data store for large scientific datasets.
    
    Provides high-performance storage for meshes, solutions, and results.
    
    Parameters
    ----------
    file_path : str
        Path to HDF5 file
    mode : str, optional
        File access mode, by default 'a' (append)
    """
    
    def __init__(self, file_path: str, mode: str = 'a'):
        if not HAS_HDF5:
            raise ImportError("HDF5 support requires h5py")
        
        self.file_path = Path(file_path)
        self.mode = mode
        self._file = None
        
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def open(self) -> h5py.File:
        """Open HDF5 file.
        
        Returns
        -------
        h5py.File
            HDF5 file handle
        """
        if self._file is None:
            self._file = h5py.File(self.file_path, self.mode)
        return self._file
    
    def close(self) -> None:
        """Close HDF5 file."""
        if self._file:
            self._file.close()
            self._file = None
    
    def store_array(self, name: str, data: Any, **kwargs) -> None:
        """Store array data.
        
        Parameters
        ----------
        name : str
            Dataset name
        data : array-like
            Data to store
        **kwargs
            Additional HDF5 dataset options
        """
        with self.open() as f:
            if name in f:
                del f[name]
            f.create_dataset(name, data=data, **kwargs)
    
    def load_array(self, name: str) -> Any:
        """Load array data.
        
        Parameters
        ----------
        name : str
            Dataset name
            
        Returns
        -------
        Any
            Loaded data
        """
        with self.open() as f:
            return f[name][:]
    
    def store_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        """Store metadata attributes.
        
        Parameters
        ----------
        name : str
            Dataset or group name
        metadata : Dict[str, Any]
            Metadata dictionary
        """
        with self.open() as f:
            if name not in f:
                f.create_group(name)
            
            for key, value in metadata.items():
                f[name].attrs[key] = value
    
    def load_metadata(self, name: str) -> Dict[str, Any]:
        """Load metadata attributes.
        
        Parameters
        ----------
        name : str
            Dataset or group name
            
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary
        """
        with self.open() as f:
            return dict(f[name].attrs)
    
    def list_datasets(self, group: str = '/') -> list:
        """List datasets in group.
        
        Parameters
        ----------
        group : str, optional
            Group path, by default '/'
            
        Returns
        -------
        list
            List of dataset names
        """
        with self.open() as f:
            def visit_func(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets.append(name)
            
            datasets = []
            f[group].visititems(visit_func)
            return datasets
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_connection() -> DatabaseManager:
    """Get global database manager instance.
    
    Returns
    -------
    DatabaseManager
        Database manager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def set_database_url(url: str) -> None:
    """Set database URL and reset connection.
    
    Parameters
    ----------
    url : str
        Database connection URL
    """
    global _db_manager
    if _db_manager:
        _db_manager.disconnect()
    _db_manager = DatabaseManager(url)
