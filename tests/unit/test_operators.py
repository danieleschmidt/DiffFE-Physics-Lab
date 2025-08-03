"""Unit tests for physics operators."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

try:
    import firedrake as fd
    HAS_FIREDRAKE = True
except ImportError:
    HAS_FIREDRAKE = False

from src.operators import (
    LaplacianOperator, 
    ElasticityOperator,
    laplacian,
    elasticity,
    register_operator,
    get_operator
)
from src.operators.base import BaseOperator


class TestBaseOperator:
    """Test base operator functionality."""
    
    def test_operator_registry(self):
        """Test operator registration system."""
        
        @register_operator("test_op")
        class TestOperator(BaseOperator):
            def forward_assembly(self, trial, test, params):
                return trial * test
            
            def adjoint_assembly(self, grad_output, trial, test, params):
                return trial * test
        
        # Test registration
        op_class = get_operator("test_op")
        assert op_class == TestOperator
        
        # Test instantiation
        op = TestOperator()
        assert isinstance(op, BaseOperator)
    
    def test_operator_validation(self):
        """Test input validation."""
        
        class MockOperator(BaseOperator):
            def forward_assembly(self, trial, test, params):
                return trial * test
            
            def adjoint_assembly(self, grad_output, trial, test, params):
                return trial * test
        
        op = MockOperator()
        
        # Test validation with None inputs
        with pytest.raises(ValueError, match="Trial function cannot be None"):
            op.validate_inputs(None, Mock(), {})
        
        with pytest.raises(ValueError, match="Test function cannot be None"):
            op.validate_inputs(Mock(), None, {})
    
    def test_operator_compilation(self):
        """Test operator compilation."""
        
        class MockOperator(BaseOperator):
            def forward_assembly(self, trial, test, params):
                return trial * test
            
            def adjoint_assembly(self, grad_output, trial, test, params):
                return trial * test
        
        op = MockOperator(backend="jax")
        compiled_op = op.compile()
        
        assert compiled_op is op  # Should return self


class TestLaplacianOperator:
    """Test Laplacian operator."""
    
    def test_initialization(self):
        """Test operator initialization."""
        op = LaplacianOperator(diffusion_coeff=2.0)
        
        assert op.diffusion_coeff == 2.0
        assert op.is_linear
        assert op.is_symmetric
    
    def test_convenience_function(self):
        """Test laplacian convenience function."""
        if not HAS_FIREDRAKE:
            pytest.skip("Firedrake not available")
        
        mesh = fd.UnitSquareMesh(4, 4)
        V = fd.FunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        
        # Test basic usage
        form = laplacian(u, v, diffusion_coeff=1.0)
        assert form is not None
        
        # Test with source term
        form_with_source = laplacian(u, v, source=1.0)
        assert form_with_source is not None
    
    @pytest.mark.firedrake
    def test_manufactured_solution(self, mesh_2d, function_space_scalar):
        """Test manufactured solution generation."""
        op = LaplacianOperator()
        
        # Generate manufactured solution
        ms = op.manufactured_solution(frequency=1.0, dimension=2)
        
        assert "solution" in ms
        assert "source" in ms
        assert callable(ms["solution"])
        assert callable(ms["source"])
        
        # Test evaluation
        x = [0.5, 0.5]
        u_val = ms["solution"](x)
        f_val = ms["source"](x)
        
        assert isinstance(u_val, (int, float))
        assert isinstance(f_val, (int, float))
        
        # Test different dimensions
        ms_1d = op.manufactured_solution(dimension=1)
        ms_3d = op.manufactured_solution(dimension=3)
        
        assert callable(ms_1d["solution"])
        assert callable(ms_3d["solution"])
    
    @pytest.mark.firedrake
    def test_forward_assembly(self, mesh_2d, function_space_scalar):
        """Test forward assembly."""
        op = LaplacianOperator(diffusion_coeff=2.0)
        
        u = fd.TrialFunction(function_space_scalar)
        v = fd.TestFunction(function_space_scalar)
        
        # Test assembly without source
        form = op.forward_assembly(u, v, {})
        assert form is not None
        
        # Test assembly with source
        params = {"source": 1.0}
        form_with_source = op.forward_assembly(u, v, params)
        assert form_with_source is not None
        
        # Test with function-based coefficients
        params_func = {
            "diffusion_coeff": lambda x: 2.0 * x[0],
            "source": lambda x: x[0] + x[1]
        }
        form_func = op.forward_assembly(u, v, params_func)
        assert form_func is not None
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test valid parameters
        op1 = LaplacianOperator(diffusion_coeff=1.0)
        assert op1.diffusion_coeff == 1.0
        
        op2 = LaplacianOperator(diffusion_coeff=lambda x: x[0])
        assert callable(op2.diffusion_coeff)


class TestElasticityOperator:
    """Test elasticity operator."""
    
    def test_initialization(self):
        """Test operator initialization."""
        op = ElasticityOperator(E=200e9, nu=0.3)
        
        assert op.E == 200e9
        assert op.nu == 0.3
        assert op.is_linear
        assert op.is_symmetric
    
    def test_parameter_validation(self):
        """Test material parameter validation."""
        # Valid parameters
        op = ElasticityOperator(E=1.0, nu=0.3)
        assert op.E == 1.0
        assert op.nu == 0.3
        
        # Invalid Poisson's ratio
        with pytest.raises(ValueError, match="Poisson's ratio must be in"):
            ElasticityOperator(E=1.0, nu=0.6)
        
        with pytest.raises(ValueError, match="Poisson's ratio must be in"):
            ElasticityOperator(E=1.0, nu=-0.1)
        
        # Invalid Young's modulus
        with pytest.raises(ValueError, match="Young's modulus must be positive"):
            ElasticityOperator(E=-1.0, nu=0.3)
    
    def test_lame_parameters(self):
        """Test Lamé parameter computation."""
        # Plane strain
        op_strain = ElasticityOperator(E=1.0, nu=0.3, plane_stress=False)
        lmbda, mu = op_strain.lame_parameters()
        
        expected_lmbda = 1.0 * 0.3 / ((1 + 0.3) * (1 - 2 * 0.3))
        expected_mu = 1.0 / (2 * (1 + 0.3))
        
        assert abs(lmbda - expected_lmbda) < 1e-10
        assert abs(mu - expected_mu) < 1e-10
        
        # Plane stress
        op_stress = ElasticityOperator(E=1.0, nu=0.3, plane_stress=True)
        lmbda_stress, mu_stress = op_stress.lame_parameters()
        
        expected_lmbda_stress = 1.0 * 0.3 / (1 - 0.3**2)
        
        assert abs(lmbda_stress - expected_lmbda_stress) < 1e-10
        assert abs(mu_stress - expected_mu) < 1e-10
    
    @pytest.mark.firedrake
    def test_strain_tensor(self, mesh_2d, function_space_vector):
        """Test strain tensor computation."""
        op = ElasticityOperator()
        
        u = fd.Function(function_space_vector)
        
        # Set a simple displacement field
        u.project(fd.as_vector([fd.SpatialCoordinate(mesh_2d)[0], 0]))
        
        strain = op.strain_tensor(u)
        assert strain is not None
    
    @pytest.mark.firedrake  
    def test_stress_tensor(self, mesh_2d, function_space_vector):
        """Test stress tensor computation."""
        op = ElasticityOperator(E=1.0, nu=0.3)
        
        u = fd.Function(function_space_vector)
        u.project(fd.as_vector([fd.SpatialCoordinate(mesh_2d)[0], 0]))
        
        stress = op.stress_tensor(u)
        assert stress is not None
        
        # Test with different material parameters
        params = {"E": 2.0, "nu": 0.25}
        stress_modified = op.stress_tensor(u, params)
        assert stress_modified is not None
    
    @pytest.mark.firedrake
    def test_forward_assembly(self, mesh_2d, function_space_vector):
        """Test forward assembly."""
        op = ElasticityOperator(E=1.0, nu=0.3)
        
        u = fd.TrialFunction(function_space_vector)
        v = fd.TestFunction(function_space_vector)
        
        # Test assembly without body force
        form = op.forward_assembly(u, v, {})
        assert form is not None
        
        # Test assembly with body force
        params = {"body_force": [0.0, -1.0]}
        form_with_body = op.forward_assembly(u, v, params)
        assert form_with_body is not None
        
        # Test with traction boundary condition
        params_traction = {"traction": {1: [1000.0, 0.0]}}
        form_with_traction = op.forward_assembly(u, v, params_traction)
        assert form_with_traction is not None
    
    def test_manufactured_solution(self):
        """Test manufactured solution generation."""
        op = ElasticityOperator(E=1.0, nu=0.3)
        
        # 2D manufactured solution
        ms_2d = op.manufactured_solution(dimension=2, frequency=1.0)
        
        assert "displacement" in ms_2d
        assert "body_force" in ms_2d
        assert callable(ms_2d["displacement"])
        assert callable(ms_2d["body_force"])
        
        # Test evaluation
        x = [0.5, 0.5]
        u_val = ms_2d["displacement"](x)
        f_val = ms_2d["body_force"](x)
        
        assert len(u_val) == 2  # 2D displacement
        assert len(f_val) == 2  # 2D body force
        
        # 3D manufactured solution
        ms_3d = op.manufactured_solution(dimension=3, frequency=1.0)
        
        x_3d = [0.5, 0.5, 0.5]
        u_val_3d = ms_3d["displacement"](x_3d)
        f_val_3d = ms_3d["body_force"](x_3d)
        
        assert len(u_val_3d) == 3  # 3D displacement
        assert len(f_val_3d) == 3  # 3D body force
    
    def test_convenience_function(self):
        """Test elasticity convenience function."""
        if not HAS_FIREDRAKE:
            pytest.skip("Firedrake not available")
        
        mesh = fd.UnitSquareMesh(4, 4)
        V = fd.VectorFunctionSpace(mesh, "CG", 1)
        u = fd.TrialFunction(V)
        v = fd.TestFunction(V)
        
        # Test basic usage
        form = elasticity(u, v, E=1.0, nu=0.3)
        assert form is not None
        
        # Test with body force
        form_with_body = elasticity(u, v, body_force=[0, -1])
        assert form_with_body is not None
        
        # Test with traction
        form_with_traction = elasticity(u, v, traction={1: [100, 0]})
        assert form_with_traction is not None


class TestOperatorErrors:
    """Test error handling in operators."""
    
    def test_unknown_operator(self):
        """Test getting unknown operator."""
        with pytest.raises(KeyError, match="Operator 'unknown' not found"):
            get_operator("unknown")
    
    def test_missing_firedrake(self):
        """Test behavior when Firedrake is missing."""
        with patch('src.operators.laplacian.HAS_FIREDRAKE', False):
            op = LaplacianOperator()
            
            with pytest.raises(ImportError, match="Firedrake required"):
                op.forward_assembly(Mock(), Mock(), {})
    
    def test_invalid_manufactured_solution_dimension(self):
        """Test invalid dimension for manufactured solution."""
        op = LaplacianOperator()
        
        with pytest.raises(ValueError, match="Unsupported dimension"):
            op.manufactured_solution(dimension=4)


class TestOperatorPerformance:
    """Test operator performance characteristics."""
    
    @pytest.mark.slow
    @pytest.mark.firedrake
    def test_assembly_performance(self, mesh_2d, function_space_scalar, performance_thresholds):
        """Test assembly performance."""
        import time
        
        op = LaplacianOperator()
        u = fd.TrialFunction(function_space_scalar)
        v = fd.TestFunction(function_space_scalar)
        
        # Time assembly
        start_time = time.time()
        form = op.forward_assembly(u, v, {"source": 1.0})
        assembly_time = time.time() - start_time
        
        # Should be fast for small problems
        assert assembly_time < 0.1, f"Assembly took {assembly_time}s, expected < 0.1s"
    
    @pytest.mark.slow
    def test_operator_compilation(self):
        """Test operator compilation performance."""
        import time
        
        class MockOperator(BaseOperator):
            def forward_assembly(self, trial, test, params):
                # Simulate expensive computation
                return sum(range(1000))
            
            def adjoint_assembly(self, grad_output, trial, test, params):
                return sum(range(1000))
        
        op = MockOperator(backend="jax")
        
        # Time compilation
        start_time = time.time()
        compiled_op = op.compile()
        compile_time = time.time() - start_time
        
        # Compilation should be fast
        assert compile_time < 1.0, f"Compilation took {compile_time}s"
