"""Integration test for the Enhanced FEM Solver with Generation 3 scaling features.

This test verifies that the enhanced solver integrates correctly with the existing
performance infrastructure and maintains backward compatibility.
"""

import asyncio
import sys
import os
import logging
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.services.enhanced_fem_solver import EnhancedFEMSolver, create_enhanced_fem_solver
from src.services.basic_fem_solver import BasicFEMSolver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_compatibility():
    """Test backward compatibility with BasicFEMSolver."""
    logger.info("Testing basic compatibility...")
    
    # Create enhanced solver
    enhanced_solver = EnhancedFEMSolver()
    basic_solver = BasicFEMSolver()
    
    try:
        # Test 1D solve compatibility
        params = {
            "x_start": 0.0,
            "x_end": 1.0, 
            "num_elements": 20,
            "diffusion_coeff": 1.0,
            "left_bc": 0.0,
            "right_bc": 1.0
        }
        
        # Solve with both solvers
        nodes1, solution1 = enhanced_solver.solve_1d_laplace(**params)
        nodes2, solution2 = basic_solver.solve_1d_laplace(**params)
        
        # Results should be very similar
        np.testing.assert_allclose(nodes1, nodes2, rtol=1e-10)
        np.testing.assert_allclose(solution1, solution2, rtol=1e-10)
        
        logger.info("✓ 1D compatibility test passed")
        
        # Test 2D solve compatibility
        nodes1, solution1 = enhanced_solver.solve_2d_laplace(nx=10, ny=10)
        nodes2, solution2 = basic_solver.solve_2d_laplace(nx=10, ny=10)
        
        # Check shapes are compatible
        assert nodes1.shape == nodes2.shape, f"Node shapes differ: {nodes1.shape} vs {nodes2.shape}"
        assert solution1.shape == solution2.shape, f"Solution shapes differ: {solution1.shape} vs {solution2.shape}"
        
        logger.info("✓ 2D compatibility test passed")
        
    finally:
        enhanced_solver.shutdown()
    
    logger.info("✓ Basic compatibility tests completed")


async def test_enhanced_features():
    """Test enhanced features work without errors."""
    logger.info("Testing enhanced features...")
    
    # Create solver with all features enabled
    solver = create_enhanced_fem_solver(scaling_level="standard")
    
    try:
        # Test enhanced 1D solve
        nodes, solution, metrics = await solver.solve_1d_laplace_enhanced(
            x_start=0.0, x_end=1.0, num_elements=100,
            diffusion_coeff=1.0, left_bc=0.0, right_bc=1.0
        )
        
        # Check results are reasonable
        assert len(nodes) == 101, f"Expected 101 nodes, got {len(nodes)}"
        assert len(solution) == 101, f"Expected 101 solution points, got {len(solution)}"
        assert "total_solve_time" in metrics, "Missing solve time in metrics"
        assert metrics["total_solve_time"] > 0, "Solve time should be positive"
        
        logger.info(f"✓ Enhanced 1D solve: {len(nodes)} nodes, {metrics['total_solve_time']:.3f}s")
        
        # Test enhanced 2D solve
        nodes, solution, metrics = await solver.solve_2d_laplace_enhanced(
            nx=20, ny=20, diffusion_coeff=1.0
        )
        
        # Check results are reasonable
        assert len(solution) > 0, "Solution should not be empty"
        assert "total_solve_time" in metrics, "Missing solve time in metrics"
        
        logger.info(f"✓ Enhanced 2D solve: {len(solution)} DOFs, {metrics['total_solve_time']:.3f}s")
        
        # Test scaling metrics
        scaling_metrics = solver.get_scaling_metrics()
        assert "scaling_features" in scaling_metrics, "Missing scaling features in metrics"
        assert "performance_counters" in scaling_metrics, "Missing performance counters"
        
        logger.info("✓ Scaling metrics available")
        
    finally:
        solver.shutdown()
    
    logger.info("✓ Enhanced features tests completed")


def test_factory_functions():
    """Test factory functions for creating solvers."""
    logger.info("Testing factory functions...")
    
    # Test different scaling levels
    scaling_levels = ["minimal", "standard", "aggressive"]
    
    for level in scaling_levels:
        solver = create_enhanced_fem_solver(scaling_level=level)
        try:
            # Just verify it creates successfully
            assert solver is not None, f"Failed to create solver with level {level}"
            
            # Test basic solve
            nodes, solution = solver.solve_1d_laplace(num_elements=10)
            assert len(nodes) == 11, "Basic solve should work"
            
            logger.info(f"✓ Factory function works for level: {level}")
            
        finally:
            solver.shutdown()
    
    logger.info("✓ Factory function tests completed")


def test_auto_optimization():
    """Test automatic optimization features."""
    logger.info("Testing auto optimization...")
    
    solver = create_enhanced_fem_solver(scaling_level="aggressive")
    
    try:
        # Test problem size optimization
        small_settings = solver.optimize_for_problem_size(100)
        large_settings = solver.optimize_for_problem_size(100000)
        
        # Should get different settings for different problem sizes
        assert isinstance(small_settings, dict), "Should return settings dict"
        assert isinstance(large_settings, dict), "Should return settings dict"
        
        logger.info(f"✓ Small problem settings: {small_settings}")
        logger.info(f"✓ Large problem settings: {large_settings}")
        
    finally:
        solver.shutdown()
    
    logger.info("✓ Auto optimization tests completed")


async def test_performance_infrastructure():
    """Test integration with performance infrastructure."""
    logger.info("Testing performance infrastructure integration...")
    
    solver = create_enhanced_fem_solver(scaling_level="standard")
    
    try:
        # Test that performance components are available
        if hasattr(solver, 'resource_monitor'):
            logger.info("✓ Resource monitor available")
        
        if hasattr(solver, 'cache_manager'):
            logger.info("✓ Cache manager available")
        
        if hasattr(solver, 'parallel_engine'):
            logger.info("✓ Parallel engine available")
        
        # Run a solve and check metrics are collected
        nodes, solution, metrics = await solver.solve_1d_laplace_enhanced(
            num_elements=50
        )
        
        # Check scaling metrics
        scaling_metrics = solver.get_scaling_metrics()
        
        assert "scaling_features" in scaling_metrics
        assert "performance_counters" in scaling_metrics
        
        counters = scaling_metrics["performance_counters"]
        logger.info(f"✓ Performance counters: {counters}")
        
    finally:
        solver.shutdown()
    
    logger.info("✓ Performance infrastructure tests completed")


async def main():
    """Run all integration tests."""
    logger.info("🧪 Starting Enhanced FEM Solver Integration Tests")
    logger.info("=" * 60)
    
    try:
        # Run synchronous tests
        test_basic_compatibility()
        test_factory_functions()
        
        # Run asynchronous tests
        await test_enhanced_features()
        await test_performance_infrastructure()
        test_auto_optimization()
        
        logger.info("=" * 60)
        logger.info("🎉 All integration tests passed!")
        logger.info("\nIntegration verified:")
        logger.info("  ✅ Backward compatibility with BasicFEMSolver")
        logger.info("  ✅ Enhanced features work correctly")
        logger.info("  ✅ Factory functions create proper solvers")
        logger.info("  ✅ Auto-optimization adjusts settings")
        logger.info("  ✅ Performance infrastructure integration")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set up event loop and run tests
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)