"""Test basic imports and functionality without external dependencies."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import operators
        print("✓ operators module imported")
        
        import backends
        print("✓ backends module imported")
        
        import services
        print("✓ services module imported")
        
        import utils
        print("✓ utils module imported")
        
        import models
        print("✓ models module imported")
        
        # Test specific classes
        from operators.base import BaseOperator, LinearOperator, register_operator
        print("✓ Base operator classes imported")
        
        from operators.laplacian import LaplacianOperator, laplacian
        print("✓ Laplacian operator imported")
        
        from backends.base import ADBackend, get_backend, set_default_backend
        print("✓ Backend base classes imported")
        
        from utils.manufactured_solutions import SolutionType, generate_manufactured_solution
        print("✓ Manufactured solution utilities imported")
        
        from models.problem import Problem, FEBMLProblem
        print("✓ Problem classes imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_manufactured_solutions():
    """Test manufactured solution generation."""
    print("\nTesting manufactured solution generation...")
    
    try:
        from utils.manufactured_solutions import SolutionType, generate_manufactured_solution
        
        # Test trigonometric solution
        mms = generate_manufactured_solution(
            solution_type=SolutionType.TRIGONOMETRIC,
            dimension=2,
            parameters={'frequency': 1.0, 'amplitude': 1.0}
        )
        
        # Test solution evaluation (using built-in math)
        import math
        test_point = [0.5, 0.5]
        
        # Mock numpy functionality with built-in math
        class MockNumpy:
            def sin(self, x):
                if isinstance(x, (list, tuple)):
                    return [math.sin(xi) for xi in x]
                return math.sin(x)
            
            def cos(self, x):
                if isinstance(x, (list, tuple)):
                    return [math.cos(xi) for xi in x]
                return math.cos(x)
                
            def pi(self):
                return math.pi
            
            def array(self, x):
                return x
        
        # Temporarily replace numpy for testing
        import utils.manufactured_solutions as mms_module
        original_np = getattr(mms_module, 'np', None)
        
        # Create mock with pi as property
        class MockNP:
            pi = math.pi
            
            def sin(self, x):
                return math.sin(x)
                
            def cos(self, x):
                return math.cos(x)
                
            def array(self, x):
                return x
        
        mms_module.np = MockNP()
        
        try:
            # Re-generate with mock numpy
            mms = generate_manufactured_solution(
                solution_type=SolutionType.TRIGONOMETRIC,
                dimension=2,
                parameters={'frequency': 1.0, 'amplitude': 1.0}
            )
            
            u_val = mms['solution'](test_point)
            f_val = mms['source'](test_point)
            
            print(f"✓ Solution at (0.5, 0.5): u = {u_val:.4f}, f = {f_val:.4f}")
            
            # Restore original
            if original_np:
                mms_module.np = original_np
            
            return True
            
        except Exception as e:
            print(f"✗ Manufactured solution evaluation failed: {e}")
            # Restore original
            if original_np:
                mms_module.np = original_np
            return False
            
    except Exception as e:
        print(f"✗ Manufactured solution test failed: {e}")
        return False

def test_operators():
    """Test operator creation and basic functionality."""
    print("\nTesting operator creation...")
    
    try:
        from operators.laplacian import LaplacianOperator
        from operators.elasticity import ElasticityOperator
        from operators.fluid import NavierStokesOperator
        from operators.nonlinear import HyperelasticOperator
        from operators.electromagnetic import MaxwellOperator
        from operators.transport import AdvectionOperator
        
        # Test operator creation
        lap_op = LaplacianOperator(diffusion_coeff=1.0)
        print(f"✓ Laplacian operator: {lap_op}")
        
        elastic_op = ElasticityOperator(E=200e9, nu=0.3)
        print(f"✓ Elasticity operator: {elastic_op}")
        
        ns_op = NavierStokesOperator(nu=0.01)
        print(f"✓ Navier-Stokes operator: {ns_op}")
        
        hyper_op = HyperelasticOperator(material_model='neo_hookean')
        print(f"✓ Hyperelastic operator: {hyper_op}")
        
        maxwell_op = MaxwellOperator(formulation='electrostatic')
        print(f"✓ Maxwell operator: {maxwell_op}")
        
        advection_op = AdvectionOperator(velocity=[1.0, 0.0])
        print(f"✓ Advection operator: {advection_op}")
        
        return True
        
    except Exception as e:
        print(f"✗ Operator creation failed: {e}")
        return False

def test_backends():
    """Test backend functionality."""
    print("\nTesting backends...")
    
    try:
        from backends.base import get_backend, list_backends
        
        # List available backends
        backends = list_backends()
        print(f"Available backends: {backends}")
        
        # Try to get backends (will fail gracefully without dependencies)
        try:
            jax_backend = get_backend('jax')
            print(f"✓ JAX backend loaded: {jax_backend}")
        except ImportError as e:
            print(f"○ JAX backend not available (expected): {e}")
        
        try:
            torch_backend = get_backend('torch')
            print(f"✓ PyTorch backend loaded: {torch_backend}")
        except ImportError as e:
            print(f"○ PyTorch backend not available (expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backend test failed: {e}")
        return False

def test_services():
    """Test service classes."""
    print("\nTesting services...")
    
    try:
        from services.solver import FEBMLSolver
        from services.optimization import OptimizationService
        from services.assembly import AssemblyEngine
        
        # Test service creation (without actual problem)
        solver = FEBMLSolver()
        print(f"✓ FEBMLSolver created: {solver}")
        
        opt_service = OptimizationService()
        print(f"✓ OptimizationService created: {opt_service}")
        
        assembly = AssemblyEngine()
        print(f"✓ AssemblyEngine created: {assembly}")
        
        return True
        
    except Exception as e:
        print(f"✗ Service test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("DiffFE-Physics-Lab Basic Functionality Test")
    print("=" * 50)
    
    results = []
    
    results.append(test_imports())
    results.append(test_manufactured_solutions())
    results.append(test_operators())
    results.append(test_backends())
    results.append(test_services())
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed!")
        print("\nThe framework is ready for use. To run full examples:")
        print("1. Install dependencies: pip install numpy jax firedrake")
        print("2. Run: python examples/basic_diffusion.py")
    else:
        print("✗ Some tests failed. Check the errors above.")
        
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)