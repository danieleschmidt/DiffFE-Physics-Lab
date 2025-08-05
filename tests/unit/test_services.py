"""Unit tests for services module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.services.solver import FEBMLSolver, SolverConfig, SolverMethod
from src.services.optimization import OptimizationService, OptimizationMethod, OptimizationResult
from src.services.assembly import AssemblyEngine, AssemblyConfig, AssemblyStrategy
from src.models.problem import Problem
from src.utils.validation import ValidationError


class TestFEBMLSolver:
    """Test cases for FEBMLSolver."""
    
    def test_solver_init(self):
        """Test solver initialization."""
        mock_problem = Mock()
        
        # Test with default config
        solver = FEBMLSolver(mock_problem)
        assert solver.problem == mock_problem
        assert isinstance(solver.config, SolverConfig)
        assert solver.config.method == SolverMethod.DIRECT
        
        # Test with custom config
        custom_config = SolverConfig(
            method=SolverMethod.ITERATIVE,
            tolerance=1e-8,
            max_iterations=500
        )
        solver_custom = FEBMLSolver(mock_problem, custom_config)
        assert solver_custom.config.tolerance == 1e-8
        assert solver_custom.config.max_iterations == 500
    
    @patch('src.services.solver.HAS_FIREDRAKE', True)
    @patch('src.services.solver.fd')
    def test_solve_direct_method(self, mock_fd):
        """Test solving with direct method."""
        # Mock problem
        mock_problem = Mock()
        mock_problem.function_space = Mock()
        mock_problem.equations = [{'equation': lambda u, v, p: u + v, 'active': True}]
        mock_problem.parameters = {'param1': 1.0}
        mock_problem._assemble_boundary_conditions.return_value = []
        
        # Mock Firedrake objects
        mock_function = Mock()
        mock_test_function = Mock()
        mock_fd.Function.return_value = mock_function
        mock_fd.TestFunction.return_value = mock_test_function
        mock_fd.solve = Mock()
        
        config = SolverConfig(method=SolverMethod.DIRECT)
        solver = FEBMLSolver(mock_problem, config)
        
        result = solver.solve()
        
        assert result == mock_function
        mock_fd.solve.assert_called_once()
    
    @patch('src.services.solver.HAS_FIREDRAKE', True)
    @patch('src.services.solver.fd')
    def test_solve_iterative_method(self, mock_fd):
        """Test solving with iterative method."""
        mock_problem = Mock()
        mock_problem.function_space = Mock()
        mock_problem.equations = [{'equation': lambda u, v, p: u + v, 'active': True}]
        mock_problem.parameters = {}
        mock_problem._assemble_boundary_conditions.return_value = []
        
        # Mock Firedrake objects
        mock_function = Mock()
        mock_test_function = Mock()
        mock_fd.Function.return_value = mock_function
        mock_fd.TestFunction.return_value = mock_test_function
        
        # Mock linear variational solver
        mock_solver = Mock()
        mock_fd.LinearVariationalSolver.return_value = mock_solver
        
        config = SolverConfig(
            method=SolverMethod.ITERATIVE,
            solver_params={'ksp_type': 'cg'}
        )
        solver = FEBMLSolver(mock_problem, config)
        
        result = solver.solve()
        
        assert result == mock_function
        mock_solver.solve.assert_called_once()
    
    @patch('src.services.solver.HAS_FIREDRAKE', False)
    def test_solve_no_firedrake(self):
        """Test solving without Firedrake."""
        mock_problem = Mock()
        solver = FEBMLSolver(mock_problem)
        
        with pytest.raises(ImportError, match="Firedrake is required"):
            solver.solve()
    
    def test_solve_no_equations(self):
        """Test solving with no equations."""
        mock_problem = Mock()
        mock_problem.equations = []
        
        solver = FEBMLSolver(mock_problem)
        
        with pytest.raises(ValueError, match="No active equations"):
            solver.solve()
    
    @patch('src.services.solver.HAS_FIREDRAKE', True)
    @patch('src.services.solver.fd')
    def test_solve_with_nonlinear(self, mock_fd):
        """Test solving with nonlinear solver."""
        mock_problem = Mock()
        mock_problem.function_space = Mock()
        mock_problem.equations = [{'equation': lambda u, v, p: u**2 + v, 'active': True}]
        mock_problem.parameters = {}
        mock_problem._assemble_boundary_conditions.return_value = []
        
        mock_function = Mock()
        mock_test_function = Mock()
        mock_fd.Function.return_value = mock_function
        mock_fd.TestFunction.return_value = mock_test_function
        
        # Mock nonlinear solver
        mock_solver = Mock()
        mock_fd.NonlinearVariationalSolver.return_value = mock_solver
        
        config = SolverConfig(method=SolverMethod.NONLINEAR)
        solver = FEBMLSolver(mock_problem, config)
        
        result = solver.solve()
        
        assert result == mock_function
        mock_solver.solve.assert_called_once()
    
    def test_validate_parameters(self):
        """Test parameter validation."""
        mock_problem = Mock()
        solver = FEBMLSolver(mock_problem)
        
        # Valid parameters
        valid_params = {'diffusion_coeff': 1.0, 'source': 2.0}
        solver._validate_parameters(valid_params)  # Should not raise
        
        # Invalid parameters
        invalid_params = {'diffusion_coeff': -1.0}
        with pytest.raises(ValidationError):
            solver._validate_parameters(invalid_params)
    
    @patch('src.services.solver.HAS_FIREDRAKE', True)
    def test_save_solution(self):
        """Test solution saving."""
        with patch('src.services.solver.fd') as mock_fd:
            mock_problem = Mock()
            mock_solution = Mock()
            mock_file = Mock()
            mock_fd.File.return_value = mock_file
            
            solver = FEBMLSolver(mock_problem)
            solver.save_solution("output.pvd", mock_solution)
            
            mock_fd.File.assert_called_once_with("output.pvd")
            mock_file.write.assert_called_once_with(mock_solution)


class TestOptimizationService:
    """Test cases for OptimizationService."""
    
    def test_optimization_service_init(self):
        """Test optimization service initialization."""
        mock_problem = Mock()
        
        service = OptimizationService(mock_problem)
        assert service.problem == mock_problem
    
    @patch('scipy.optimize.minimize')
    def test_minimize_scalar(self, mock_minimize):
        """Test scalar optimization."""
        # Mock scipy result
        mock_result = Mock()
        mock_result.success = True
        mock_result.x = np.array([2.5])
        mock_result.fun = 0.001
        mock_result.nit = 15
        mock_result.nfev = 45
        mock_result.message = "Optimization terminated successfully"
        mock_minimize.return_value = mock_result
        
        mock_problem = Mock()
        service = OptimizationService(mock_problem)
        
        def objective(x):
            return (x - 2.5)**2
        
        result = service.minimize_scalar(
            objective=objective,
            bounds=(0.0, 5.0),
            method=OptimizationMethod.BFGS
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert result.optimal_value == 2.5
        assert result.objective_value == 0.001
        assert result.iterations == 15
        assert result.function_evaluations == 45
    
    @patch('scipy.optimize.minimize')
    def test_minimize_vector(self, mock_minimize):
        """Test vector optimization."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.x = np.array([1.0, 2.0])
        mock_result.fun = 0.5
        mock_result.nit = 20
        mock_result.nfev = 60
        mock_minimize.return_value = mock_result
        
        mock_problem = Mock()
        service = OptimizationService(mock_problem)
        
        def objective(x):
            return (x[0] - 1.0)**2 + (x[1] - 2.0)**2
        
        result = service.minimize_vector(
            objective=objective,
            initial_guess=np.array([0.0, 0.0]),
            bounds=[(0.0, 2.0), (0.0, 4.0)]
        )
        
        assert result.success is True
        assert len(result.optimal_parameters) == 2
        assert result.optimal_parameters['param_0'] == 1.0
        assert result.optimal_parameters['param_1'] == 2.0
    
    def test_parameter_sweep(self):
        """Test parameter sweep optimization."""
        mock_problem = Mock()
        service = OptimizationService(mock_problem)
        
        def objective(params):
            return (params['x'] - 2.0)**2 + (params['y'] - 3.0)**2
        
        param_ranges = {
            'x': np.linspace(0, 4, 5),
            'y': np.linspace(1, 5, 5)
        }
        
        result = service.parameter_sweep(objective, param_ranges)
        
        assert 'parameters' in result
        assert 'objective_values' in result
        assert 'best_parameters' in result
        assert 'best_objective' in result
        
        # Best should be close to (2, 3)
        best_params = result['best_parameters']
        assert abs(best_params['x'] - 2.0) <= 1.0  # Within grid resolution
        assert abs(best_params['y'] - 3.0) <= 1.0
    
    def test_bayesian_optimization(self):
        """Test Bayesian optimization."""
        mock_problem = Mock()
        service = OptimizationService(mock_problem)
        
        def objective(params):
            x = params['x']
            return -(x * np.sin(x))  # Maximize this function
        
        bounds = {'x': (0.0, 10.0)}
        
        # Mock skopt if available
        try:
            with patch('skopt.gp_minimize') as mock_gp_minimize:
                mock_result = Mock()
                mock_result.x = [7.5]
                mock_result.fun = -5.0
                mock_result.func_vals = [-1.0, -3.0, -5.0]
                mock_gp_minimize.return_value = mock_result
                
                result = service.bayesian_optimization(
                    objective, bounds, n_calls=10
                )
                
                assert 'best_parameters' in result
                assert 'best_objective' in result
                assert result['best_parameters']['x'] == 7.5
                
        except ImportError:
            # Skip if skopt not available
            pytest.skip("scikit-optimize not available")
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        mock_problem = Mock()
        service = OptimizationService(mock_problem)
        
        def objective(params):
            return params['a']**2 + 2*params['b']**2 + params['a']*params['b']
        
        base_params = {'a': 1.0, 'b': 2.0}
        perturbation = 0.1
        
        result = service.sensitivity_analysis(objective, base_params, perturbation)
        
        assert 'sensitivities' in result
        assert 'base_objective' in result
        assert 'perturbed_objectives' in result
        
        # Should have sensitivities for both parameters
        assert 'a' in result['sensitivities']
        assert 'b' in result['sensitivities']


class TestAssemblyEngine:
    """Test cases for AssemblyEngine."""
    
    def test_assembly_engine_init(self):
        """Test assembly engine initialization."""
        config = AssemblyConfig(
            strategy=AssemblyStrategy.STANDARD,
            cache_assembly=True
        )
        
        engine = AssemblyEngine(config)
        assert engine.config.strategy == AssemblyStrategy.STANDARD
        assert engine.config.cache_assembly is True
    
    @patch('src.services.assembly.HAS_FIREDRAKE', True)
    @patch('src.services.assembly.fd')
    def test_assemble_matrix(self, mock_fd):
        """Test matrix assembly."""
        mock_form = Mock()
        mock_matrix = Mock()
        mock_fd.assemble.return_value = mock_matrix
        
        config = AssemblyConfig(strategy=AssemblyStrategy.STANDARD)
        engine = AssemblyEngine(config)
        
        result = engine.assemble_matrix(mock_form)
        
        assert result == mock_matrix
        mock_fd.assemble.assert_called_once_with(mock_form)
    
    @patch('src.services.assembly.HAS_FIREDRAKE', True)
    @patch('src.services.assembly.fd')
    def test_assemble_vector(self, mock_fd):
        """Test vector assembly."""
        mock_form = Mock()
        mock_vector = Mock()
        mock_fd.assemble.return_value = mock_vector
        
        engine = AssemblyEngine()
        result = engine.assemble_vector(mock_form)
        
        assert result == mock_vector
        mock_fd.assemble.assert_called_once_with(mock_form)
    
    def test_assembly_caching(self):
        """Test assembly result caching."""
        with patch('src.services.assembly.HAS_FIREDRAKE', True):
            with patch('src.services.assembly.fd') as mock_fd:
                mock_form = Mock()
                mock_matrix = Mock()
                mock_fd.assemble.return_value = mock_matrix
                
                config = AssemblyConfig(cache_assembly=True)
                engine = AssemblyEngine(config)
                
                # First call should assemble
                result1 = engine.assemble_matrix(mock_form)
                assert result1 == mock_matrix
                assert mock_fd.assemble.call_count == 1
                
                # Second call should use cache
                result2 = engine.assemble_matrix(mock_form)
                assert result2 == mock_matrix
                assert mock_fd.assemble.call_count == 1  # No additional calls
    
    @patch('src.services.assembly.HAS_FIREDRAKE', False)
    def test_assemble_no_firedrake(self):
        """Test assembly without Firedrake."""
        engine = AssemblyEngine()
        mock_form = Mock()
        
        with pytest.raises(ImportError, match="Firedrake is required"):
            engine.assemble_matrix(mock_form)
    
    @patch('src.services.assembly.HAS_FIREDRAKE', True)
    def test_gpu_assembly_strategy(self):
        """Test GPU assembly strategy."""
        with patch('src.services.assembly.fd') as mock_fd:
            mock_form = Mock()
            mock_matrix = Mock()
            mock_fd.assemble.return_value = mock_matrix
            
            config = AssemblyConfig(strategy=AssemblyStrategy.GPU)
            engine = AssemblyEngine(config)
            
            result = engine.assemble_matrix(mock_form)
            
            # Should still work but may log warnings about GPU availability
            assert result == mock_matrix
    
    def test_clear_cache(self):
        """Test cache clearing."""
        with patch('src.services.assembly.HAS_FIREDRAKE', True):
            with patch('src.services.assembly.fd') as mock_fd:
                config = AssemblyConfig(cache_assembly=True)
                engine = AssemblyEngine(config)
                
                # Add some cache entries
                engine._assembly_cache['key1'] = 'value1'
                engine._assembly_cache['key2'] = 'value2'
                
                assert len(engine._assembly_cache) == 2
                
                engine.clear_cache()
                
                assert len(engine._assembly_cache) == 0
    
    def test_get_cache_stats(self):
        """Test cache statistics."""
        with patch('src.services.assembly.HAS_FIREDRAKE', True):
            config = AssemblyConfig(cache_assembly=True)
            engine = AssemblyEngine(config)
            
            # Simulate cache usage
            engine._cache_hits = 15
            engine._cache_misses = 5
            engine._assembly_cache = {'key1': 'val1', 'key2': 'val2'}
            
            stats = engine.get_cache_stats()
            
            assert stats['hits'] == 15
            assert stats['misses'] == 5
            assert stats['hit_rate'] == 0.75  # 15/(15+5)
            assert stats['size'] == 2


class TestServiceIntegration:
    """Integration tests for services working together."""
    
    @patch('src.services.solver.HAS_FIREDRAKE', True)
    @patch('src.services.solver.fd')
    def test_solver_optimization_integration(self, mock_fd):
        """Test solver and optimization service integration."""
        # Mock problem and solver components
        mock_problem = Mock()
        mock_problem.function_space = Mock()
        mock_problem.equations = [{'equation': lambda u, v, p: u + v, 'active': True}]
        mock_problem.parameters = {}
        mock_problem._assemble_boundary_conditions.return_value = []
        mock_problem.set_parameter = Mock()
        
        mock_solution = Mock()
        mock_solution.dat.data = np.array([1.0, 2.0, 3.0])
        
        mock_fd.Function.return_value = mock_solution
        mock_fd.TestFunction.return_value = Mock()
        mock_fd.solve = Mock()
        
        # Create solver and optimization service
        solver = FEBMLSolver(mock_problem)
        opt_service = OptimizationService(mock_problem)
        
        # Define objective function that uses solver
        def objective(params_dict):
            # Set parameters and solve
            for key, value in params_dict.items():
                mock_problem.set_parameter(key, value)
            
            solution = solver.solve()
            # Simple objective based on solution
            return np.sum(solution.dat.data**2)
        
        # Run optimization
        with patch('scipy.optimize.minimize') as mock_minimize:
            mock_result = Mock()
            mock_result.success = True
            mock_result.x = np.array([2.0])
            mock_result.fun = 14.0  # 1^2 + 2^2 + 3^2 = 14
            mock_result.nit = 10
            mock_result.nfev = 30
            mock_minimize.return_value = mock_result
            
            result = opt_service.minimize_vector(
                objective=objective,
                initial_guess=np.array([1.0])
            )
            
            assert result.success is True
            assert result.objective_value == 14.0
    
    def test_assembly_solver_integration(self):
        """Test assembly engine and solver integration."""
        with patch('src.services.assembly.HAS_FIREDRAKE', True):
            with patch('src.services.solver.HAS_FIREDRAKE', True):
                with patch('src.services.assembly.fd') as mock_fd_assembly:
                    with patch('src.services.solver.fd') as mock_fd_solver:
                        # Mock assembly components
                        mock_matrix = Mock()
                        mock_fd_assembly.assemble.return_value = mock_matrix
                        
                        # Mock solver components  
                        mock_problem = Mock()
                        mock_problem.function_space = Mock()
                        mock_problem.equations = [{'equation': lambda u, v, p: u + v, 'active': True}]
                        mock_problem.parameters = {}
                        mock_problem._assemble_boundary_conditions.return_value = []
                        
                        mock_solution = Mock()
                        mock_fd_solver.Function.return_value = mock_solution
                        mock_fd_solver.TestFunction.return_value = Mock()
                        mock_fd_solver.solve = Mock()
                        
                        # Create services
                        assembly_config = AssemblyConfig(cache_assembly=True)
                        assembly_engine = AssemblyEngine(assembly_config)
                        
                        solver_config = SolverConfig(method=SolverMethod.DIRECT)
                        solver = FEBMLSolver(mock_problem, solver_config)
                        
                        # Set assembly engine on solver (if supported)
                        if hasattr(solver, 'assembly_engine'):
                            solver.assembly_engine = assembly_engine
                        
                        # Solve problem
                        result = solver.solve()
                        
                        assert result == mock_solution
                        mock_fd_solver.solve.assert_called_once()


class TestServiceErrors:
    """Test error handling in services."""
    
    def test_solver_config_validation(self):
        """Test solver configuration validation."""
        # Invalid tolerance
        with pytest.raises(ValueError):
            SolverConfig(tolerance=-1.0)
        
        # Invalid max iterations
        with pytest.raises(ValueError):
            SolverConfig(max_iterations=0)
    
    def test_optimization_service_errors(self):
        """Test optimization service error handling."""
        mock_problem = Mock()
        service = OptimizationService(mock_problem)
        
        # Invalid bounds for scalar optimization
        with pytest.raises(ValueError):
            service.minimize_scalar(
                objective=lambda x: x**2,
                bounds=(5.0, 1.0)  # Invalid: max < min
            )
        
        # Invalid initial guess for vector optimization
        with pytest.raises(ValueError):
            service.minimize_vector(
                objective=lambda x: np.sum(x**2),
                initial_guess=np.array([]),  # Empty array
                bounds=[(0, 1), (0, 1)]
            )
    
    def test_assembly_engine_errors(self):
        """Test assembly engine error handling."""
        engine = AssemblyEngine()
        
        # None form
        with pytest.raises(ValueError):
            engine.assemble_matrix(None)
        
        with pytest.raises(ValueError):
            engine.assemble_vector(None)