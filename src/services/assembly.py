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


class GPUAssembly(AssemblyStrategy):
    """GPU-accelerated assembly strategy."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        logger.info(f"GPU assembly initialized on device {device_id}")
    
    def assemble_matrix(self, bilinear_form: Any) -> Any:
        """Assemble bilinear form on GPU."""
        # Would implement GPU-specific assembly here
        logger.warning("GPU assembly not yet implemented, falling back to standard")
        return StandardAssembly().assemble_matrix(bilinear_form)
    
    def assemble_vector(self, linear_form: Any) -> Any:
        """Assemble linear form on GPU."""
        logger.warning("GPU assembly not yet implemented, falling back to standard")
        return StandardAssembly().assemble_vector(linear_form)


class AssemblyEngine:
    """High-performance assembly engine for finite element operations.
    
    Provides unified interface for assembling finite element operators with
    support for different assembly strategies, caching, and performance optimization.
    
    Parameters
    ----------
    strategy : str, optional
        Assembly strategy ('standard', 'gpu'), by default 'standard'
    cache_enabled : bool, optional
        Enable assembly caching, by default True
    backend : str, optional
        AD backend for differentiation, by default 'jax'
        
    Examples
    --------
    >>> engine = AssemblyEngine()
    >>> matrix = engine.assemble_operator(laplacian_op, function_space)
    >>> vector = engine.assemble_rhs(source_term, function_space)
    """
    
    def __init__(
        self,
        strategy: str = 'standard',
        cache_enabled: bool = True,
        backend: str = 'jax'
    ):
        # Setup assembly strategy
        if strategy == 'standard':
            self.strategy = StandardAssembly()
        elif strategy == 'gpu':
            self.strategy = GPUAssembly()
        else:
            raise ValueError(f"Unknown assembly strategy: {strategy}")
        
        self.cache_enabled = cache_enabled
        self.backend = get_backend(backend)
        
        # Assembly cache
        self._matrix_cache = {}
        self._vector_cache = {}
        self._form_cache = {}
        
        # Performance metrics
        self.assembly_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"AssemblyEngine initialized with {strategy} strategy")
    
    def assemble_operator(
        self,
        operator: Union[BaseOperator, str],
        function_space: Any,
        parameters: Dict[str, Any] = None,
        use_cache: bool = None
    ) -> Any:
        """Assemble operator into matrix form.
        
        Parameters
        ----------
        operator : BaseOperator or str
            Operator to assemble
        function_space : firedrake.FunctionSpace
            Function space
        parameters : Dict[str, Any], optional
            Operator parameters
        use_cache : bool, optional
            Whether to use cache (overrides default)
            
        Returns
        -------
        Any
            Assembled matrix
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        
        # Get operator instance
        if isinstance(operator, str):
            operator = get_operator(operator)()
        
        # Setup parameters
        params = parameters or {}
        
        # Check cache
        cache_key = self._get_cache_key('matrix', operator, function_space, params)
        use_cache = use_cache if use_cache is not None else self.cache_enabled
        
        if use_cache and cache_key in self._matrix_cache:
            self.cache_hits += 1
            logger.debug(f"Matrix cache hit for {operator.__class__.__name__}")
            return self._matrix_cache[cache_key]
        
        self.cache_misses += 1
        
        # Assemble matrix
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
        
        # Store timing
        op_name = operator.__class__.__name__
        if op_name not in self.assembly_times:
            self.assembly_times[op_name] = []
        self.assembly_times[op_name].append(assembly_time)
        
        logger.debug(f"Matrix assembly completed in {assembly_time:.4f}s")
        
        # Cache result
        if use_cache:
            self._matrix_cache[cache_key] = matrix
        
        return matrix
    
    def assemble_rhs(
        self,
        source_term: Union[Callable, Any],
        function_space: Any,
        parameters: Dict[str, Any] = None,
        use_cache: bool = None
    ) -> Any:
        """Assemble right-hand side vector.
        
        Parameters
        ----------
        source_term : Callable or firedrake.Function/Constant
            Source term for RHS
        function_space : firedrake.FunctionSpace
            Function space
        parameters : Dict[str, Any], optional
            Additional parameters
        use_cache : bool, optional
            Whether to use cache
            
        Returns
        -------
        Any
            Assembled RHS vector
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        
        params = parameters or {}
        
        # Check cache
        cache_key = self._get_cache_key('vector', source_term, function_space, params)
        use_cache = use_cache if use_cache is not None else self.cache_enabled
        
        if use_cache and cache_key in self._vector_cache:
            self.cache_hits += 1
            logger.debug("RHS vector cache hit")
            return self._vector_cache[cache_key]
        
        self.cache_misses += 1
        
        # Assemble vector
        import time
        start_time = time.time()
        
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
        
        end_time = time.time()
        assembly_time = end_time - start_time
        
        logger.debug(f"RHS assembly completed in {assembly_time:.4f}s")
        
        # Cache result
        if use_cache:
            self._vector_cache[cache_key] = vector
        
        return vector
    
    def assemble_system(
        self,
        operators: List[Union[BaseOperator, str]],
        function_space: Any,
        parameters: Dict[str, Any] = None,
        boundary_conditions: List = None
    ) -> tuple:
        """Assemble complete system matrix and RHS vector.
        
        Parameters
        ----------
        operators : List[BaseOperator or str]
            List of operators to assemble
        function_space : firedrake.FunctionSpace
            Function space
        parameters : Dict[str, Any], optional
            System parameters
        boundary_conditions : List, optional
            Boundary conditions to apply
            
        Returns
        -------
        tuple
            (system_matrix, rhs_vector, boundary_conditions)
        """
        if not HAS_FIREDRAKE:
            raise ImportError("Firedrake required for assembly")
        
        params = parameters or {}
        
        logger.info(f"Assembling system with {len(operators)} operators")
        
        # Initialize system matrix and RHS
        system_matrix = None
        rhs_vector = None
        
        # Assemble operators
        for i, operator in enumerate(operators):
            logger.debug(f"Processing operator {i+1}/{len(operators)}")
            
            # Get operator instance
            if isinstance(operator, str):
                op_instance = get_operator(operator)()\n            else:\n                op_instance = operator\n            \n            # Assemble operator matrix\n            op_matrix = self.assemble_operator(op_instance, function_space, params)\n            \n            if system_matrix is None:\n                system_matrix = op_matrix\n            else:\n                # Add to system matrix\n                system_matrix += op_matrix\n            \n            # Assemble RHS if source term exists\n            if 'source' in params:\n                source = params['source']\n                if callable(source) or isinstance(source, (fd.Function, fd.Constant)):\n                    op_rhs = self.assemble_rhs(source, function_space, params)\n                    \n                    if rhs_vector is None:\n                        rhs_vector = op_rhs\n                    else:\n                        rhs_vector += op_rhs\n        \n        # Apply boundary conditions if provided\n        if boundary_conditions:\n            for bc in boundary_conditions:\n                if hasattr(bc, 'apply'):\n                    bc.apply(system_matrix)\n                    if rhs_vector is not None:\n                        bc.apply(rhs_vector)\n        \n        logger.info("System assembly completed")\n        \n        return system_matrix, rhs_vector, boundary_conditions\n    \n    def compute_element_matrices(\n        self,\n        operator: BaseOperator,\n        mesh: Any,\n        parameters: Dict[str, Any] = None\n    ) -> np.ndarray:\n        """Compute element-wise matrices for operator.\n        \n        Parameters\n        ----------\n        operator : BaseOperator\n            Operator to compute element matrices for\n        mesh : firedrake.Mesh\n            Computational mesh\n        parameters : Dict[str, Any], optional\n            Operator parameters\n            \n        Returns\n        -------\n        np.ndarray\n            Element matrices\n        """\n        # This would implement element-wise assembly\n        # Useful for GPU kernels and custom assembly\n        logger.warning("Element matrix computation not yet implemented")\n        return np.array([])\n    \n    def differentiate_assembly(\n        self,\n        operator: BaseOperator,\n        function_space: Any,\n        parameters: Dict[str, Any],\n        parameter_name: str\n    ) -> Any:\n        """Compute derivative of assembled operator with respect to parameter.\n        \n        Parameters\n        ----------\n        operator : BaseOperator\n            Operator to differentiate\n        function_space : firedrake.FunctionSpace\n            Function space\n        parameters : Dict[str, Any]\n            Parameters\n        parameter_name : str\n            Parameter to differentiate with respect to\n            \n        Returns\n        -------\n        Any\n            Derivative matrix\n        """\n        logger.debug(f"Computing assembly derivative w.r.t. {parameter_name}")\n        \n        # This would use automatic differentiation to compute\n        # derivatives of the assembly process\n        def assembly_func(param_value):\n            params_copy = parameters.copy()\n            params_copy[parameter_name] = param_value\n            return self.assemble_operator(operator, function_space, params_copy, use_cache=False)\n        \n        # Use backend to compute derivative\n        param_value = parameters[parameter_name]\n        \n        try:\n            grad_func = self.backend.grad(assembly_func)\n            derivative = grad_func(param_value)\n            return derivative\n        except Exception as e:\n            logger.warning(f"Automatic differentiation failed: {e}")\n            # Fall back to finite differences\n            eps = 1e-8\n            param_plus = param_value + eps\n            param_minus = param_value - eps\n            \n            params_plus = parameters.copy()\n            params_minus = parameters.copy()\n            params_plus[parameter_name] = param_plus\n            params_minus[parameter_name] = param_minus\n            \n            matrix_plus = self.assemble_operator(operator, function_space, params_plus, use_cache=False)\n            matrix_minus = self.assemble_operator(operator, function_space, params_minus, use_cache=False)\n            \n            derivative = (matrix_plus - matrix_minus) / (2 * eps)\n            return derivative\n    \n    def _get_cache_key(\n        self,\n        cache_type: str,\n        *args\n    ) -> str:\n        """Generate cache key for assembly results."""\n        # Simple cache key generation - could be more sophisticated\n        key_parts = [cache_type]\n        \n        for arg in args:\n            if hasattr(arg, '__name__'):\n                key_parts.append(arg.__name__)\n            elif hasattr(arg, '__class__'):\n                key_parts.append(arg.__class__.__name__)\n            else:\n                key_parts.append(str(hash(str(arg))))\n        \n        return "_".join(key_parts)\n    \n    def clear_cache(self) -> None:\n        """Clear all assembly caches."""\n        self._matrix_cache.clear()\n        self._vector_cache.clear()\n        self._form_cache.clear()\n        \n        logger.info("Assembly caches cleared")\n    \n    def get_cache_stats(self) -> Dict[str, Any]:\n        """Get cache performance statistics.\n        \n        Returns\n        -------\n        Dict[str, Any]\n            Cache statistics\n        """\n        total_requests = self.cache_hits + self.cache_misses\n        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0\n        \n        return {\n            'cache_hits': self.cache_hits,\n            'cache_misses': self.cache_misses,\n            'hit_rate': hit_rate,\n            'matrix_cache_size': len(self._matrix_cache),\n            'vector_cache_size': len(self._vector_cache),\n            'form_cache_size': len(self._form_cache)\n        }\n    \n    def get_performance_stats(self) -> Dict[str, Any]:\n        """Get assembly performance statistics.\n        \n        Returns\n        -------\n        Dict[str, Any]\n            Performance statistics\n        """\n        stats = {}\n        \n        for op_name, times in self.assembly_times.items():\n            stats[op_name] = {\n                'count': len(times),\n                'total_time': sum(times),\n                'mean_time': np.mean(times),\n                'std_time': np.std(times),\n                'min_time': min(times),\n                'max_time': max(times)\n            }\n        \n        return stats\n    \n    def benchmark_assembly(\n        self,\n        operator: BaseOperator,\n        function_space: Any,\n        parameters: Dict[str, Any] = None,\n        num_runs: int = 10\n    ) -> Dict[str, float]:\n        """Benchmark assembly performance.\n        \n        Parameters\n        ----------\n        operator : BaseOperator\n            Operator to benchmark\n        function_space : firedrake.FunctionSpace\n            Function space\n        parameters : Dict[str, Any], optional\n            Parameters\n        num_runs : int, optional\n            Number of benchmark runs, by default 10\n            \n        Returns\n        -------\n        Dict[str, float]\n            Benchmark results\n        """\n        import time\n        \n        logger.info(f"Benchmarking assembly for {operator.__class__.__name__}")\n        \n        times = []\n        \n        # Warm-up run\n        self.assemble_operator(operator, function_space, parameters, use_cache=False)\n        \n        # Benchmark runs\n        for run in range(num_runs):\n            start_time = time.time()\n            self.assemble_operator(operator, function_space, parameters, use_cache=False)\n            end_time = time.time()\n            \n            times.append(end_time - start_time)\n        \n        results = {\n            'num_runs': num_runs,\n            'mean_time': np.mean(times),\n            'std_time': np.std(times),\n            'min_time': min(times),\n            'max_time': max(times),\n            'dofs': function_space.dim() if hasattr(function_space, 'dim') else 0\n        }\n        \n        logger.info(f"Benchmark complete: {results['mean_time']:.4f}±{results['std_time']:.4f}s")\n        \n        return results\n    \n    def __repr__(self) -> str:\n        return (f"AssemblyEngine("\n                f"strategy={self.strategy.__class__.__name__}, "\n                f"cache_enabled={self.cache_enabled}, "\n                f"backend={self.backend.name}"\n                f")")