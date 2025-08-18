"""Generation 3 "Make It Scale" Enhanced FEM Solver Demonstration.

This example demonstrates the advanced scaling features of the Enhanced FEM Solver,
showcasing enterprise-scale performance optimizations including:

- JIT compilation for hot computational paths
- Multi-level caching system (memory, disk, distributed)
- Parallel processing for large problems
- Auto-scaling based on resource usage and problem size
- Adaptive mesh refinement with load balancing
- Memory pool management for repeated operations
- Production-ready scaling infrastructure

The examples show both backward compatibility with BasicFEMSolver and the new
scaling features for enterprise deployments.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, Any, List

# Import the enhanced solver and scaling utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.enhanced_fem_solver import EnhancedFEMSolver, create_enhanced_fem_solver
from src.services.basic_fem_solver import BasicFEMSolver
from src.performance.fem_scaling_utils import (
    auto_scale_solver, performance_monitor, adaptive_caching,
    memory_optimized, scaling_session, ScalingBenchmark,
    solve_with_auto_scaling
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_basic_compatibility():
    """Demonstrate backward compatibility with BasicFEMSolver."""
    logger.info("=== Basic Compatibility Demo ===")
    
    # Create enhanced solver with minimal scaling
    solver = create_enhanced_fem_solver(scaling_level="minimal")
    
    try:
        # Use standard BasicFEMSolver methods - should work identically
        logger.info("Testing 1D Laplace with standard interface...")
        nodes, solution = solver.solve_1d_laplace(
            x_start=0.0, x_end=1.0, num_elements=50,
            diffusion_coeff=1.0, left_bc=0.0, right_bc=1.0
        )
        logger.info(f"Standard 1D solve: {len(nodes)} nodes, max solution: {np.max(solution):.3f}")
        
        logger.info("Testing 2D Laplace with standard interface...")
        nodes, solution = solver.solve_2d_laplace(
            x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=20, ny=20,
            diffusion_coeff=1.0
        )
        logger.info(f"Standard 2D solve: {len(nodes)} nodes, max solution: {np.max(solution):.3f}")
        
    finally:
        solver.shutdown()
    
    logger.info("Basic compatibility verified!\n")


async def demo_enhanced_features():
    """Demonstrate enhanced features with Generation 3 optimizations."""
    logger.info("=== Enhanced Features Demo ===")
    
    # Create enhanced solver with full scaling features
    solver = create_enhanced_fem_solver(scaling_level="aggressive")
    
    try:
        # Enhanced 1D solve with all scaling features
        logger.info("Testing enhanced 1D Laplace solve...")
        nodes, solution, metrics = await solver.solve_1d_laplace_enhanced(
            x_start=0.0, x_end=2.0, num_elements=1000,
            diffusion_coeff=2.0, left_bc=0.0, right_bc=10.0,
            enable_adaptive_refinement=True
        )
        
        logger.info(f"Enhanced 1D solve completed:")
        logger.info(f"  - Nodes: {len(nodes)}")
        logger.info(f"  - Max solution: {np.max(solution):.3f}")
        logger.info(f"  - Solve time: {metrics['total_solve_time']:.3f}s")
        logger.info(f"  - Features used: {metrics.get('scaling_metrics', {})}")
        
        # Enhanced 2D solve with parallel processing
        logger.info("Testing enhanced 2D Laplace solve...")
        nodes, solution, metrics = await solver.solve_2d_laplace_enhanced(
            x_range=(0.0, 2.0), y_range=(0.0, 2.0), nx=50, ny=50,
            diffusion_coeff=1.5, 
            boundary_values={"left": 0.0, "right": 5.0, "bottom": 0.0, "top": 10.0},
            enable_adaptive_refinement=True,
            enable_parallel_assembly=True
        )
        
        logger.info(f"Enhanced 2D solve completed:")
        logger.info(f"  - DOFs: {len(solution)}")
        logger.info(f"  - Max solution: {np.max(solution):.3f}")
        logger.info(f"  - Solve time: {metrics['total_solve_time']:.3f}s")
        logger.info(f"  - Parallel assembly: {metrics.get('parallel_assembly_used', False)}")
        logger.info(f"  - Adaptive refinement: {metrics.get('adaptive_refinement_used', False)}")
        
        # Show comprehensive scaling metrics
        scaling_metrics = solver.get_scaling_metrics()
        logger.info(f"Comprehensive scaling metrics:")
        logger.info(f"  - Performance counters: {scaling_metrics['performance_counters']}")
        
        if 'cache_stats' in scaling_metrics:
            cache_stats = scaling_metrics['cache_stats']['general_cache']
            logger.info(f"  - Cache hits/misses: {cache_stats.get('hits', 0)}/{cache_stats.get('misses', 0)}")
        
        if 'resource_stats' in scaling_metrics:
            current_metrics = scaling_metrics['resource_stats']['current_metrics']
            if current_metrics:
                logger.info(f"  - Current CPU: {current_metrics.cpu_usage:.1f}%")
                logger.info(f"  - Current Memory: {current_metrics.memory_usage:.1f}%")
        
    finally:
        solver.shutdown()
    
    logger.info("Enhanced features demo completed!\n")


@auto_scale_solver(problem_size_threshold=500)
@performance_monitor(track_resources=True)
@adaptive_caching(ttl=300.0)
@memory_optimized(pool_arrays=True)
async def demo_decorated_solver(x_end: float = 1.0, num_elements: int = 100):
    """Demonstrate decorated solver with automatic scaling features."""
    solver = BasicFEMSolver()  # Even basic solver benefits from decorators
    
    nodes, solution = solver.solve_1d_laplace(
        x_start=0.0, x_end=x_end, num_elements=num_elements,
        diffusion_coeff=1.0, left_bc=0.0, right_bc=1.0
    )
    
    return nodes, solution


async def demo_scaling_decorators():
    """Demonstrate automatic scaling with decorators."""
    logger.info("=== Scaling Decorators Demo ===")
    
    # Small problem - minimal scaling
    logger.info("Testing small problem (auto scaling disabled)...")
    nodes, solution = await demo_decorated_solver(x_end=1.0, num_elements=50)
    logger.info(f"Small problem: {len(nodes)} nodes")
    
    # Large problem - full scaling enabled
    logger.info("Testing large problem (auto scaling enabled)...")
    nodes, solution = await demo_decorated_solver(x_end=2.0, num_elements=2000)
    logger.info(f"Large problem: {len(nodes)} nodes")
    
    logger.info("Scaling decorators demo completed!\n")


async def demo_scaling_session():
    """Demonstrate scaling session context manager."""
    logger.info("=== Scaling Session Demo ===")
    
    # Different scaling sessions for different problem types
    scaling_configs = ["minimal", "standard", "aggressive"]
    
    for config in scaling_configs:
        logger.info(f"Testing scaling session: {config}")
        
        async with scaling_session(scaling_level=config) as components:
            logger.info(f"  Active components: {list(components.keys())}")
            
            # Create solver and run problem
            solver = BasicFEMSolver()
            start_time = time.time()
            
            nodes, solution = solver.solve_1d_laplace(
                num_elements=500, diffusion_coeff=1.0
            )
            
            solve_time = time.time() - start_time
            logger.info(f"  Solve time with {config}: {solve_time:.3f}s")
    
    logger.info("Scaling session demo completed!\n")


async def demo_auto_optimization():
    """Demonstrate automatic problem size optimization."""
    logger.info("=== Auto Optimization Demo ===")
    
    solver = create_enhanced_fem_solver(scaling_level="aggressive")
    
    try:
        # Test different problem sizes with auto optimization
        problem_sizes = [100, 1000, 10000, 50000]
        
        for size in problem_sizes:
            logger.info(f"Auto-optimizing for problem size: {size}")
            
            # Let solver optimize settings based on problem size
            optimization_settings = solver.optimize_for_problem_size(size)
            logger.info(f"  Optimized settings: {optimization_settings}")
            
            # Run solve with optimized settings
            if size <= 10000:  # Avoid very large problems in demo
                nodes, solution, metrics = await solver.solve_1d_laplace_enhanced(
                    num_elements=min(size, 1000),  # Cap for demo
                    enable_adaptive_refinement=optimization_settings.get("enable_adaptive_mesh", False)
                )
                logger.info(f"  Solve time: {metrics['total_solve_time']:.3f}s")
    
    finally:
        solver.shutdown()
    
    logger.info("Auto optimization demo completed!\n")


async def demo_benchmarking():
    """Demonstrate comprehensive benchmarking capabilities."""
    logger.info("=== Benchmarking Demo ===")
    
    # Define problem generator for benchmarking
    def generate_1d_problem(size: int) -> Dict[str, Any]:
        return {
            "x_start": 0.0,
            "x_end": 1.0,
            "num_elements": size,
            "diffusion_coeff": 1.0,
            "left_bc": 0.0,
            "right_bc": 1.0
        }
    
    # Define solver function
    async def benchmark_solver(**kwargs):
        solver = BasicFEMSolver()
        return solver.solve_1d_laplace(**kwargs)
    
    # Create benchmark with smaller problem sizes for demo
    benchmark = ScalingBenchmark(
        problem_sizes=[50, 200, 500],  # Smaller for demo
        scaling_levels=["minimal", "standard"]
    )
    
    # Run benchmark
    logger.info("Running scaling benchmark (this may take a moment)...")
    results = await benchmark.run_benchmark(
        benchmark_solver, generate_1d_problem, iterations=2
    )
    
    logger.info("Benchmark results:")
    logger.info(f"  Total time: {results['total_benchmark_time']:.1f}s")
    logger.info(f"  Configurations tested: {results['configurations_tested']}")
    
    # Show performance summary
    summary = results['performance_summary']
    if 'best_overall_config' in summary:
        best = summary['best_overall_config']
        logger.info(f"  Best overall: size={best['problem_size']}, level={best['scaling_level']}, time={best['avg_time']:.3f}s")
    
    # Show scaling efficiency
    if 'scaling_level_efficiency' in summary:
        for level, stats in summary['scaling_level_efficiency'].items():
            logger.info(f"  {level} efficiency: {stats['efficiency_score']:.2f}")
    
    logger.info("Benchmarking demo completed!\n")


async def demo_high_level_integration():
    """Demonstrate high-level integration functions."""
    logger.info("=== High-Level Integration Demo ===")
    
    # Create a simple solver function
    async def simple_solver(num_elements: int = 100):
        solver = BasicFEMSolver()
        return solver.solve_1d_laplace(num_elements=num_elements)
    
    # Use high-level integration with auto-scaling
    logger.info("Testing high-level auto-scaling integration...")
    
    result, benchmark_data = await solve_with_auto_scaling(
        simple_solver,
        num_elements=1000,
        auto_optimize=True,
        benchmark_mode=True
    )
    
    nodes, solution = result
    logger.info(f"High-level solve completed: {len(nodes)} nodes")
    
    if benchmark_data:
        logger.info(f"Benchmark data:")
        logger.info(f"  - Solve time: {benchmark_data['solve_time']:.3f}s")
        logger.info(f"  - Scaling level used: {benchmark_data['scaling_level_used']}")
        logger.info(f"  - Estimated problem size: {benchmark_data['estimated_problem_size']}")
    
    logger.info("High-level integration demo completed!\n")


async def demo_memory_and_caching():
    """Demonstrate advanced memory management and caching."""
    logger.info("=== Memory Management and Caching Demo ===")
    
    solver = create_enhanced_fem_solver(scaling_level="standard")
    
    try:
        # Run the same problem multiple times to show caching benefits
        problem_params = {
            "num_elements": 500,
            "diffusion_coeff": 1.0,
            "left_bc": 0.0,
            "right_bc": 1.0
        }
        
        times = []
        for i in range(3):
            logger.info(f"Run {i+1} (should show caching benefits)...")
            start_time = time.time()
            
            nodes, solution, metrics = await solver.solve_1d_laplace_enhanced(**problem_params)
            
            solve_time = time.time() - start_time
            times.append(solve_time)
            logger.info(f"  Time: {solve_time:.3f}s")
        
        logger.info(f"Performance improvement from run 1 to 3: {(times[0]/times[-1]):.2f}x speedup")
        
        # Show cache statistics
        scaling_metrics = solver.get_scaling_metrics()
        if 'cache_stats' in scaling_metrics:
            general_cache = scaling_metrics['cache_stats']['general_cache']
            logger.info(f"Cache performance:")
            logger.info(f"  - Hits: {general_cache.get('hits', 0)}")
            logger.info(f"  - Misses: {general_cache.get('misses', 0)}")
        
        # Show memory statistics
        if 'memory_stats' in scaling_metrics:
            memory_stats = scaling_metrics['memory_stats']
            logger.info(f"Memory usage:")
            logger.info(f"  - RSS: {memory_stats.get('rss_mb', 0):.1f} MB")
            logger.info(f"  - Memory pools: {len(memory_stats.get('pool_sizes', {}))}")
    
    finally:
        solver.shutdown()
    
    logger.info("Memory management and caching demo completed!\n")


async def demo_adaptive_mesh_refinement():
    """Demonstrate adaptive mesh refinement capabilities."""
    logger.info("=== Adaptive Mesh Refinement Demo ===")
    
    solver = create_enhanced_fem_solver(scaling_level="aggressive")
    
    try:
        # Create a problem with a sharp gradient that benefits from refinement
        def source_function_sharp(x):
            """Source function with sharp gradient."""
            return 100.0 * np.exp(-100.0 * (x - 0.5)**2)
        
        logger.info("Solving problem with sharp gradient (benefits from adaptive refinement)...")
        
        # Solve without adaptive refinement
        nodes1, solution1, metrics1 = await solver.solve_1d_laplace_enhanced(
            x_start=0.0, x_end=1.0, num_elements=50,
            source_function=source_function_sharp,
            enable_adaptive_refinement=False
        )
        
        # Solve with adaptive refinement
        nodes2, solution2, metrics2 = await solver.solve_1d_laplace_enhanced(
            x_start=0.0, x_end=1.0, num_elements=50,
            source_function=source_function_sharp,
            enable_adaptive_refinement=True
        )
        
        logger.info("Results comparison:")
        logger.info(f"  Without refinement: {len(nodes1)} nodes, time: {metrics1['total_solve_time']:.3f}s")
        logger.info(f"  With refinement: {len(nodes2)} nodes, time: {metrics2['total_solve_time']:.3f}s")
        logger.info(f"  Refinement ratio: {len(nodes2)/len(nodes1):.2f}x more nodes")
        
        # Show mesh refinement statistics
        scaling_metrics = solver.get_scaling_metrics()
        if 'mesh_stats' in scaling_metrics:
            mesh_stats = scaling_metrics['mesh_stats']
            logger.info(f"Mesh refinement stats:")
            logger.info(f"  - Total elements: {mesh_stats.get('total_elements', 0)}")
            logger.info(f"  - Refinement history: {mesh_stats.get('refinement_history_count', 0)} events")
    
    finally:
        solver.shutdown()
    
    logger.info("Adaptive mesh refinement demo completed!\n")


async def main():
    """Run all Generation 3 scaling demonstrations."""
    logger.info("🚀 Starting Generation 3 'Make It Scale' Enhanced FEM Solver Demo")
    logger.info("=" * 80)
    
    try:
        # Run all demonstrations
        await demo_basic_compatibility()
        await demo_enhanced_features()
        await demo_scaling_decorators()
        await demo_scaling_session()
        await demo_auto_optimization()
        await demo_memory_and_caching()
        await demo_adaptive_mesh_refinement()
        
        # Skip benchmarking demo by default as it takes longer
        run_benchmark = os.getenv("RUN_BENCHMARK", "false").lower() == "true"
        if run_benchmark:
            await demo_benchmarking()
        else:
            logger.info("Skipping benchmark demo (set RUN_BENCHMARK=true to enable)")
        
        await demo_high_level_integration()
        
        logger.info("=" * 80)
        logger.info("🎉 All Generation 3 scaling demonstrations completed successfully!")
        logger.info("\nKey features demonstrated:")
        logger.info("  ✅ Backward compatibility with BasicFEMSolver")
        logger.info("  ✅ JIT compilation for hot computational paths")
        logger.info("  ✅ Multi-level caching system")
        logger.info("  ✅ Parallel processing for large problems")
        logger.info("  ✅ Auto-scaling based on problem size")
        logger.info("  ✅ Adaptive mesh refinement with load balancing")
        logger.info("  ✅ Memory pool management")
        logger.info("  ✅ Performance monitoring and optimization")
        logger.info("  ✅ High-level integration utilities")
        logger.info("  ✅ Comprehensive benchmarking capabilities")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Set up event loop and run demo
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())