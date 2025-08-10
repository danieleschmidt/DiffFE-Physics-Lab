"""Assembly engine for finite element operators."""

from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
import logging
from abc import ABC, abstractmethod

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from ..backends import get_backend
from ..operators.base import BaseOperator, get_operator

logger = logging.getLogger(__name__)


class AssemblyStrategy(ABC):
    """Abstract base class for assembly strategies."""
    
    @abstractmethod
    def assemble_matrix(self, bilinear_form: Any) -> Any:
        """Assemble bilinear form into matrix."""
        pass
    
    @abstractmethod
    def assemble_vector(self, linear_form: Any) -> Any:
        """Assemble linear form into vector."""
        pass


class StandardAssembly(AssemblyStrategy):
    """Standard Firedrake assembly strategy."""
    
    def assemble_matrix(self, bilinear_form: Any) -> Any:
        """Assemble bilinear form using Firedrake."""
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        return fd.assemble(bilinear_form)
    
    def assemble_vector(self, linear_form: Any) -> Any:
        """Assemble linear form using Firedrake."""
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        return fd.assemble(linear_form)


class AssemblyService:
    """Alternative name for AssemblyEngine for backward compatibility."""
    def __init__(self, *args, **kwargs):
        self._engine = AssemblyEngine(*args, **kwargs)
        
    def __getattr__(self, name):
        return getattr(self._engine, name)


class AssemblyEngine:
    """High-performance assembly engine for finite element operations."""
    
    def __init__(self, strategy: str = 'standard', cache_enabled: bool = True, backend: str = 'jax'):
        # Setup assembly strategy
        if strategy == 'standard':
            self.strategy = StandardAssembly()
        else:
            raise ValueError(f"Unknown assembly strategy: {strategy}")
        
        self.cache_enabled = cache_enabled
        self.backend = get_backend(backend)
        
        # Assembly cache
        self._matrix_cache = {}
        self._vector_cache = {}
        
        # Performance metrics
        self.assembly_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"AssemblyEngine initialized with {strategy} strategy")
    
    def assemble_operator(self, operator: Union[BaseOperator, str], function_space: Any, 
                         parameters: Dict[str, Any] = None, use_cache: bool = None) -> Any:
        """Assemble operator into matrix form."""
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        
        # Get operator instance
        if isinstance(operator, str):
            operator = get_operator(operator)()
        
        # Setup parameters
        params = parameters or {}
        
        # Simple assembly without caching for basic functionality
        import time
        start_time = time.time()
        
        logger.debug(f"Assembling matrix for {operator.__class__.__name__}")
        
        # Create trial and test functions
        trial = fd.TrialFunction(function_space)
        test = fd.TestFunction(function_space)
        
        # Get bilinear form
        bilinear_form = operator.forward_assembly(trial, test, params)
        
        # Assemble
        matrix = self.strategy.assemble_matrix(bilinear_form)
        
        end_time = time.time()
        assembly_time = end_time - start_time
        
        logger.debug(f"Matrix assembly completed in {assembly_time:.4f}s")
        
        return matrix
    
    def assemble_rhs(self, source_term: Union[Callable, Any], function_space: Any,
                     parameters: Dict[str, Any] = None, use_cache: bool = None) -> Any:
        """Assemble right-hand side vector."""
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        
        params = parameters or {}
        
        logger.debug("Assembling RHS vector")
        
        # Create test function
        test = fd.TestFunction(function_space)
        
        # Convert source term
        if callable(source_term):
            source_func = fd.Function(function_space)
            source_func.interpolate(source_term)
            source_term = source_func
        elif not isinstance(source_term, (fd.Function, fd.Constant)):
            source_term = fd.Constant(float(source_term))
        
        # Create linear form
        linear_form = source_term * test * fd.dx
        
        # Assemble
        vector = self.strategy.assemble_vector(linear_form)
        
        return vector
    
    def clear_cache(self) -> None:
        """Clear all assembly caches."""
        self._matrix_cache.clear()
        self._vector_cache.clear()
        logger.info("Assembly caches cleared")
    
    def __repr__(self) -> str:
        return f"AssemblyEngine(strategy={self.strategy.__class__.__name__}, backend={self.backend.name})"