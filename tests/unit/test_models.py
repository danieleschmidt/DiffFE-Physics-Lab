"""Unit tests for problem models."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.models.problem import Problem, FEBMLProblem
from src.utils.validation import ValidationError


class TestProblem:
    """Test cases for the Problem class."""
    
    def test_problem_init_no_firedrake(self):
        """Test Problem initialization without Firedrake."""
        with patch('src.models.problem.HAS_FIREDRAKE', False):
            with pytest.raises(ImportError, match="Firedrake is required"):
                Problem()
    
    def test_problem_init_basic(self):
        """Test basic Problem initialization."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend') as mock_get_backend:
                mock_backend = Mock()
                mock_get_backend.return_value = mock_backend
                
                problem = Problem(backend='jax')
                
                assert problem.backend_name == 'jax'
                assert problem.backend == mock_backend
                assert problem.equations == []
                assert problem.boundary_conditions == {}
                assert problem.parameters == {}
                assert problem.solution is None
    
    def test_add_equation(self):
        """Test adding equations to problem."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = Problem()
                
                def test_equation(u, v, params):
                    return u + v
                
                result = problem.add_equation(test_equation, name="test_eq")
                
                assert result == problem  # Method chaining
                assert len(problem.equations) == 1
                assert problem.equations[0]['name'] == 'test_eq'
                assert problem.equations[0]['equation'] == test_equation
                assert problem.equations[0]['active'] is True
    
    def test_add_equation_auto_name(self):
        """Test adding equation with automatic naming."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = Problem()
                
                def eq1(u, v, params): return u
                def eq2(u, v, params): return v
                
                problem.add_equation(eq1)
                problem.add_equation(eq2)
                
                assert problem.equations[0]['name'] == 'equation_0'
                assert problem.equations[1]['name'] == 'equation_1'
    
    def test_add_boundary_condition(self):
        """Test adding boundary conditions."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                with patch('src.models.problem.validate_boundary_conditions'):
                    problem = Problem()
                    
                    result = problem.add_boundary_condition(
                        bc_type='dirichlet',
                        boundary_id=1,
                        value=0.0,
                        name='test_bc'
                    )
                    
                    assert result == problem
                    assert 'test_bc' in problem.boundary_conditions
                    bc = problem.boundary_conditions['test_bc']
                    assert bc['type'] == 'dirichlet'
                    assert bc['boundary'] == 1
                    assert bc['value'] == 0.0
    
    def test_add_boundary_condition_auto_name(self):
        """Test adding BC with automatic naming."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                with patch('src.models.problem.validate_boundary_conditions'):
                    problem = Problem()
                    
                    problem.add_boundary_condition('neumann', 'top', 1.0)
                    
                    assert 'neumann_top' in problem.boundary_conditions
    
    def test_set_parameter(self):
        """Test setting parameters."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = Problem()
                
                result = problem.set_parameter('diffusion_coeff', 2.5)
                
                assert result == problem
                assert problem.parameters['diffusion_coeff'] == 2.5
    
    @patch('src.models.problem.HAS_FIREDRAKE', True)
    @patch('src.models.problem.fd')
    def test_solve_no_function_space(self, mock_fd):
        """Test solve with no function space."""
        with patch('src.models.problem.get_backend'):
            problem = Problem()
            
            with pytest.raises(ValueError, match="Function space must be defined"):
                problem.solve()
    
    @patch('src.models.problem.HAS_FIREDRAKE', True)
    @patch('src.models.problem.fd')
    def test_solve_basic(self, mock_fd):
        """Test basic solve."""
        # Mock Firedrake components
        mock_function_space = Mock()
        mock_function = Mock()
        mock_test_function = Mock()
        
        mock_fd.Function.return_value = mock_function
        mock_fd.TestFunction.return_value = mock_test_function
        mock_fd.solve = Mock()
        
        with patch('src.models.problem.get_backend'):
            problem = Problem(function_space=mock_function_space)
            
            def test_equation(u, v, params):
                return u + v
            
            problem.add_equation(test_equation)
            
            # Mock boundary condition assembly
            with patch.object(problem, '_assemble_boundary_conditions', return_value=[]):
                result = problem.solve()
                
                assert result == mock_function
                assert problem.solution == mock_function
                mock_fd.solve.assert_called_once()
    
    def test_solve_with_parameters(self):
        """Test solve with runtime parameters."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.fd') as mock_fd:
                mock_function_space = Mock()
                mock_function = Mock()
                
                mock_fd.Function.return_value = mock_function
                mock_fd.TestFunction.return_value = Mock()
                mock_fd.solve = Mock()
                
                with patch('src.models.problem.get_backend'):
                    problem = Problem(function_space=mock_function_space)
                    problem.set_parameter('base_param', 1.0)
                    
                    def test_equation(u, v, params):
                        # Verify parameters are merged
                        assert params['base_param'] == 1.0
                        assert params['runtime_param'] == 2.0
                        return u + v
                    
                    problem.add_equation(test_equation)
                    
                    with patch.object(problem, '_assemble_boundary_conditions', return_value=[]):
                        problem.solve({'runtime_param': 2.0})
    
    @patch('src.models.problem.HAS_FIREDRAKE', True) 
    @patch('src.models.problem.fd')
    def test_assemble_boundary_conditions(self, mock_fd):
        """Test boundary condition assembly."""
        mock_function_space = Mock()
        mock_constant = Mock()
        mock_bc = Mock()
        
        mock_fd.Constant.return_value = mock_constant
        mock_fd.DirichletBC.return_value = mock_bc
        
        with patch('src.models.problem.get_backend'):
            problem = Problem(function_space=mock_function_space)
            
            # Add Dirichlet BC
            with patch('src.models.problem.validate_boundary_conditions'):
                problem.add_boundary_condition('dirichlet', 1, 5.0)
            
            bcs = problem._assemble_boundary_conditions()
            
            assert len(bcs) == 1
            assert bcs[0] == mock_bc
            mock_fd.Constant.assert_called_with(5.0)
            mock_fd.DirichletBC.assert_called_with(mock_function_space, mock_constant, 1)
    
    @patch('src.models.problem.HAS_FIREDRAKE', True)
    @patch('src.models.problem.fd')
    def test_assemble_boundary_conditions_function(self, mock_fd):
        """Test BC assembly with function value."""
        mock_function_space = Mock()
        mock_function = Mock()
        mock_bc = Mock()
        
        mock_fd.Function.return_value = mock_function
        mock_fd.DirichletBC.return_value = mock_bc
        
        def bc_function(x):
            return x[0]**2
        
        with patch('src.models.problem.get_backend'):
            problem = Problem(function_space=mock_function_space)
            
            with patch('src.models.problem.validate_boundary_conditions'):
                problem.add_boundary_condition('dirichlet', 1, bc_function)
            
            bcs = problem._assemble_boundary_conditions()
            
            mock_fd.Function.assert_called_with(mock_function_space)
            mock_function.interpolate.assert_called_with(bc_function)
    
    def test_differentiable_decorator_jax(self):
        """Test differentiable decorator with JAX."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                with patch('src.models.problem.HAS_JAX', True):
                    with patch('src.models.problem.jax') as mock_jax:
                        mock_jax.jit = Mock(side_effect=lambda f: f)
                        
                        problem = Problem(backend='jax')
                        
                        def test_func(x):
                            return x**2
                        
                        decorated = problem.differentiable(test_func)
                        mock_jax.jit.assert_called_once_with(test_func)
    
    def test_differentiable_decorator_fallback(self):
        """Test differentiable decorator fallback."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                with patch('src.models.problem.HAS_JAX', False):
                    problem = Problem(backend='jax')
                    
                    def test_func(x):
                        return x**2
                    
                    decorated = problem.differentiable(test_func)
                    assert decorated == test_func
    
    def test_compute_gradient_jax(self):
        """Test gradient computation with JAX."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                with patch('src.models.problem.HAS_JAX', True):
                    with patch('src.models.problem.jax') as mock_jax:
                        mock_grad_func = Mock()
                        mock_jax.grad.return_value = mock_grad_func
                        
                        problem = Problem(backend='jax')
                        
                        def objective(params):
                            return params**2
                        
                        grad_func = problem.compute_gradient(objective, ['param1'])
                        
                        assert grad_func == mock_grad_func
                        mock_jax.grad.assert_called_once_with(objective)
    
    def test_compute_gradient_unsupported_backend(self):
        """Test gradient computation with unsupported backend."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = Problem(backend='custom')
                
                def objective(params):
                    return params**2
                
                with pytest.raises(NotImplementedError):
                    problem.compute_gradient(objective, ['param1'])
    
    @patch('scipy.optimize.minimize')
    def test_optimize(self, mock_minimize):
        """Test optimization."""
        # Mock scipy minimize result
        mock_result = Mock()
        mock_result.success = True
        mock_result.x = np.array([1.23, 4.56])
        mock_result.fun = 0.001
        mock_result.nit = 25
        mock_result.message = "Optimization terminated successfully"
        mock_minimize.return_value = mock_result
        
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = Problem()
                
                def objective(params):
                    return (params['x'] - 1)**2 + (params['y'] - 2)**2
                
                initial_guess = {'x': 0.0, 'y': 0.0}
                
                result = problem.optimize(objective, initial_guess)
                
                assert result['success'] is True
                assert result['parameters']['x'] == 1.23
                assert result['parameters']['y'] == 4.56
                assert result['objective_value'] == 0.001
                assert result['iterations'] == 25
    
    def test_generate_observations(self):
        """Test observation generation."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                mock_mesh = Mock()
                mock_mesh.geometric_dimension.return_value = 2
                
                mock_solution = Mock()
                
                problem = Problem(mesh=mock_mesh)
                problem.solution = mock_solution
                
                with patch('numpy.random.seed'):
                    with patch('numpy.random.uniform') as mock_uniform:
                        with patch('numpy.random.normal') as mock_normal:
                            mock_uniform.return_value = np.array([[0.5, 0.5]])
                            mock_normal.return_value = np.array([1.0, 2.0, 3.0])
                            
                            obs = problem.generate_observations(num_points=3, noise_level=0.1, seed=42)
                            
                            assert len(obs) == 3
    
    def test_generate_observations_no_solution(self):
        """Test observation generation without solution."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = Problem()
                
                with pytest.raises(ValueError, match="Must solve problem before generating observations"):
                    problem.generate_observations(10)
    
    def test_repr(self):
        """Test problem string representation."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = Problem(backend='jax')
                
                problem.add_equation(lambda u, v, p: u + v)
                problem.add_boundary_condition('dirichlet', 1, 0.0)
                
                repr_str = repr(problem)
                
                assert 'Problem(' in repr_str
                assert 'backend=jax' in repr_str
                assert 'equations=1' in repr_str
                assert 'bcs=1' in repr_str


class TestFEBMLProblem:
    """Test cases for the FEBMLProblem class."""
    
    def test_febml_problem_init(self):
        """Test FEBMLProblem initialization."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = FEBMLProblem(experiment_name="test_experiment")
                
                assert problem.experiment_name == "test_experiment"
                assert problem.metrics == {}
                assert problem.checkpoints == {}
    
    def test_log_metric(self):
        """Test metric logging."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = FEBMLProblem()
                
                problem.log_metric('loss', 0.1)
                problem.log_metric('loss', 0.05)
                problem.log_metric('accuracy', 0.95)
                
                assert problem.metrics['loss'] == [0.1, 0.05]
                assert problem.metrics['accuracy'] == [0.95]
    
    def test_checkpoint(self):
        """Test solution checkpointing."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = FEBMLProblem()
                problem.set_parameter('test_param', 123)
                
                mock_solution = Mock()
                
                with patch('numpy.datetime64') as mock_datetime:
                    mock_datetime.return_value = 'mock_timestamp'
                    
                    problem.checkpoint(mock_solution, 'test_checkpoint')
                    
                    assert 'test_checkpoint' in problem.checkpoints
                    checkpoint = problem.checkpoints['test_checkpoint']
                    assert checkpoint['solution'] == mock_solution
                    assert checkpoint['parameters']['test_param'] == 123
                    assert checkpoint['timestamp'] == 'mock_timestamp'
    
    def test_parameterize_field(self):
        """Test field parameterization (placeholder)."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = FEBMLProblem()
                
                params = np.array([1, 2, 3])
                result = problem.parameterize_field(params)
                
                # Currently just returns input
                np.testing.assert_array_equal(result, params)
    
    def test_observe(self):
        """Test observation extraction (placeholder)."""
        with patch('src.models.problem.HAS_FIREDRAKE', True):
            with patch('src.models.problem.get_backend'):
                problem = FEBMLProblem()
                
                # Test scalar solution
                scalar_result = problem.observe(5.0)
                np.testing.assert_array_equal(scalar_result, [5.0])
                
                # Test array solution
                array_solution = np.array([1, 2, 3])
                array_result = problem.observe(array_solution)
                np.testing.assert_array_equal(array_result, array_solution)