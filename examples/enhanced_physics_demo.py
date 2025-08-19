"""Enhanced physics demonstration showcasing advanced FEM capabilities.

Generation 1 enhanced physics operators demonstration:
- Advection-diffusion with SUPG stabilization
- Linear elasticity solver
- Time-dependent problems  
- Manufactured solutions and verification
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.enhanced_fem_solver import EnhancedFEMSolver
from robust.logging_system import get_logger

logger = get_logger(__name__)


def demo_advection_diffusion():
    """Demonstrate advection-diffusion solver with SUPG stabilization."""
    print("\n🌊 ADVECTION-DIFFUSION SOLVER DEMONSTRATION")
    print("=" * 60)
    
    # Initialize enhanced solver
    solver = EnhancedFEMSolver(backend="numpy", enable_monitoring=True)
    
    # Problem parameters
    velocity = 5.0      # Strong advection
    diffusion = 0.1     # Weak diffusion -> high Péclet number
    num_elements = 100
    
    # Manufactured solution
    manufactured = solver.manufactured_solution_advection_diffusion(velocity, diffusion)
    
    print(f"Problem: Advection-Diffusion with v={velocity}, κ={diffusion}")
    print(f"Péclet number: {velocity / diffusion:.1f} (>1 requires stabilization)")
    
    # Solve with SUPG stabilization
    print("\n1. Solving with SUPG stabilization...")
    start_time = time.time()
    x_coords, solution_supg = solver.solve_advection_diffusion(
        x_range=(0.0, 1.0),
        num_elements=num_elements,
        velocity=velocity,
        diffusion_coeff=diffusion,
        source_function=manufactured["source"],
        boundary_conditions={"left": 0.0, "right": 0.0},
        peclet_stabilization=True
    )
    solve_time = time.time() - start_time
    
    # Solve without stabilization for comparison
    print("2. Solving without stabilization (for comparison)...")
    x_coords_unstab, solution_unstab = solver.solve_advection_diffusion(
        x_range=(0.0, 1.0),
        num_elements=num_elements,
        velocity=velocity,
        diffusion_coeff=diffusion,
        source_function=manufactured["source"],
        boundary_conditions={"left": 0.0, "right": 0.0},
        peclet_stabilization=False
    )
    
    # Compute exact solution for comparison
    exact_solution = np.array([manufactured["solution"](x) for x in x_coords])
    
    # Compute errors
    error_supg = np.sqrt(np.mean((solution_supg - exact_solution)**2))
    error_unstab = np.sqrt(np.mean((solution_unstab - exact_solution)**2))
    
    print(f"\nResults:")
    print(f"  Solve time: {solve_time:.3f}s")
    print(f"  DOFs: {len(solution_supg)}")
    print(f"  L2 error (SUPG):     {error_supg:.2e}")
    print(f"  L2 error (unstab):   {error_unstab:.2e}")
    print(f"  Stabilization improvement: {error_unstab/error_supg:.1f}x")
    
    # Visualization
    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(x_coords, exact_solution, 'k-', linewidth=2, label='Exact')
        plt.plot(x_coords, solution_supg, 'b--', linewidth=2, label='SUPG')
        plt.plot(x_coords, solution_unstab, 'r:', linewidth=2, label='Unstabilized')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Advection-Diffusion Solutions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(x_coords, np.abs(solution_supg - exact_solution), 'b-', label='SUPG Error')
        plt.semilogy(x_coords, np.abs(solution_unstab - exact_solution), 'r-', label='Unstab Error')
        plt.xlabel('x')
        plt.ylabel('|Error|')
        plt.title('Solution Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("  (Matplotlib not available for visualization)")
    
    return {"error_supg": error_supg, "error_unstab": error_unstab, "solve_time": solve_time}


def demo_elasticity():
    """Demonstrate 2D linear elasticity solver."""
    print("\n🏗️  LINEAR ELASTICITY SOLVER DEMONSTRATION")
    print("=" * 60)
    
    # Initialize enhanced solver
    solver = EnhancedFEMSolver(backend="numpy")
    
    # Material properties (steel-like)
    E = 200e9  # Young's modulus (Pa)
    nu = 0.3   # Poisson's ratio
    
    # Geometry
    domain_size = (1.0, 0.5)  # 1m x 0.5m beam
    mesh_size = (20, 10)      # 20x10 elements
    
    print(f"Problem: 2D Elasticity")
    print(f"  Domain: {domain_size[0]}m × {domain_size[1]}m")
    print(f"  Material: E={E/1e9:.0f} GPa, ν={nu}")
    print(f"  Mesh: {mesh_size[0]}×{mesh_size[1]} elements")
    
    # Boundary conditions: cantilever beam
    # Fixed left edge, distributed load on right edge
    boundary_conditions = {
        "left_edge": {"type": "displacement", "values": [0.0, 0.0]},  # Fixed
        "right_edge": {"type": "traction", "values": [0.0, -1e6]}     # Downward load (Pa)
    }
    
    print("\n1. Solving elasticity problem...")
    start_time = time.time()
    nodes, displacements = solver.solve_elasticity(
        domain_size=domain_size,
        mesh_size=mesh_size,
        youngs_modulus=E,
        poissons_ratio=nu,
        boundary_conditions=boundary_conditions,
        plane_stress=True
    )
    solve_time = time.time() - start_time
    
    # Compute displacement magnitudes
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)
    max_displacement = np.max(displacement_magnitudes)
    max_displacement_mm = max_displacement * 1000  # Convert to mm
    
    print(f"\nResults:")
    print(f"  Solve time: {solve_time:.3f}s")
    print(f"  Nodes: {nodes.shape[0]}")
    print(f"  DOFs: {displacements.size}")
    print(f"  Max displacement: {max_displacement_mm:.3f} mm")
    print(f"  Max stress (approx): {E * max_displacement / domain_size[0] / 1e6:.1f} MPa")
    
    # Visualization
    try:
        plt.figure(figsize=(15, 5))
        
        # Original mesh
        plt.subplot(1, 3, 1)
        plt.scatter(nodes[:, 0], nodes[:, 1], c='blue', s=10, alpha=0.6)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 0.6)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Original Mesh')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Deformed mesh (scaled for visibility)
        scale = 1000  # Scale displacements for visualization
        deformed_nodes = nodes + scale * displacements
        plt.subplot(1, 3, 2)
        plt.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], c='red', s=10, alpha=0.6)
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.3, 0.6)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f'Deformed Mesh (scale {scale}x)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Displacement magnitude contours
        plt.subplot(1, 3, 3)
        scatter = plt.scatter(nodes[:, 0], nodes[:, 1], 
                            c=displacement_magnitudes*1000, 
                            cmap='viridis', s=15)
        plt.colorbar(scatter, label='Displacement (mm)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Displacement Magnitude')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("  (Matplotlib not available for visualization)")
    
    return {"max_displacement": max_displacement, "solve_time": solve_time}


def demo_time_dependent():
    """Demonstrate time-dependent diffusion solver."""
    print("\n⏰ TIME-DEPENDENT DIFFUSION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize enhanced solver
    solver = EnhancedFEMSolver(backend="numpy")
    
    # Problem parameters
    diffusion_coeff = 0.1
    domain = (0.0, 1.0)
    num_elements = 50
    time_range = (0.0, 1.0)
    num_time_steps = 100
    
    print(f"Problem: Time-dependent diffusion")
    print(f"  Spatial domain: [{domain[0]}, {domain[1]}]")
    print(f"  Time domain: [{time_range[0]}, {time_range[1]}]")
    print(f"  Spatial elements: {num_elements}")
    print(f"  Time steps: {num_time_steps}")
    print(f"  Diffusion coefficient: {diffusion_coeff}")
    
    # Initial condition: Gaussian pulse
    def initial_condition(x):
        return np.exp(-50 * (x - 0.5)**2)
    
    # Source term: heating at x=0.2
    def source_function(x, t):
        return 10 * np.exp(-100 * (x - 0.2)**2) * np.exp(-2*t)
    
    # Solve with backward Euler
    print("\n1. Solving with Backward Euler...")
    start_time = time.time()
    times_be, x_coords, solutions_be = solver.solve_time_dependent(
        initial_condition=initial_condition,
        time_range=time_range,
        num_time_steps=num_time_steps,
        spatial_domain=domain,
        num_elements=num_elements,
        diffusion_coeff=diffusion_coeff,
        source_function=source_function,
        time_scheme="backward_euler"
    )
    solve_time_be = time.time() - start_time
    
    # Solve with Crank-Nicolson for comparison
    print("2. Solving with Crank-Nicolson...")
    times_cn, x_coords_cn, solutions_cn = solver.solve_time_dependent(
        initial_condition=initial_condition,
        time_range=time_range,
        num_time_steps=num_time_steps,
        spatial_domain=domain,
        num_elements=num_elements,
        diffusion_coeff=diffusion_coeff,
        source_function=source_function,
        time_scheme="crank_nicolson"
    )
    
    # Analysis
    final_energy_be = np.trapz(solutions_be[-1, :]**2, x_coords)
    final_energy_cn = np.trapz(solutions_cn[-1, :]**2, x_coords)
    initial_energy = np.trapz(solutions_be[0, :]**2, x_coords)
    
    print(f"\nResults:")
    print(f"  Backward Euler solve time: {solve_time_be:.3f}s")
    print(f"  Initial energy: {initial_energy:.3e}")
    print(f"  Final energy (BE): {final_energy_be:.3e}")
    print(f"  Final energy (CN): {final_energy_cn:.3e}")
    print(f"  Energy ratio (BE): {final_energy_be/initial_energy:.3f}")
    print(f"  Energy ratio (CN): {final_energy_cn/initial_energy:.3f}")
    
    # Visualization
    try:
        plt.figure(figsize=(15, 5))
        
        # Time evolution snapshots
        plt.subplot(1, 3, 1)
        time_indices = [0, num_time_steps//4, num_time_steps//2, 3*num_time_steps//4, -1]
        for i in time_indices:
            plt.plot(x_coords, solutions_be[i, :], 
                    label=f't = {times_be[i]:.2f}', alpha=0.8)
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title('Time Evolution (Backward Euler)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Energy evolution
        plt.subplot(1, 3, 2)
        energy_be = [np.trapz(sol**2, x_coords) for sol in solutions_be]
        energy_cn = [np.trapz(sol**2, x_coords) for sol in solutions_cn]
        plt.plot(times_be, energy_be, 'b-', label='Backward Euler', linewidth=2)
        plt.plot(times_cn, energy_cn, 'r--', label='Crank-Nicolson', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Solution comparison at final time
        plt.subplot(1, 3, 3)
        plt.plot(x_coords, solutions_be[-1, :], 'b-', linewidth=2, label='Backward Euler')
        plt.plot(x_coords_cn, solutions_cn[-1, :], 'r--', linewidth=2, label='Crank-Nicolson')
        plt.xlabel('x')
        plt.ylabel('u(x, T)')
        plt.title('Final Solution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("  (Matplotlib not available for visualization)")
    
    return {
        "solve_time": solve_time_be, 
        "final_energy_be": final_energy_be, 
        "final_energy_cn": final_energy_cn
    }


def main():
    """Run all enhanced physics demonstrations."""
    print("🚀 ENHANCED PHYSICS FEM SOLVER DEMONSTRATIONS")
    print("============================================")
    print("Generation 1 Implementation: Advanced Physics Operators")
    print()
    
    results = {}
    
    try:
        # Advection-diffusion with stabilization
        results["advection_diffusion"] = demo_advection_diffusion()
        
        # Linear elasticity
        results["elasticity"] = demo_elasticity()
        
        # Time-dependent problems
        results["time_dependent"] = demo_time_dependent()
        
        # Summary
        print("\n📊 DEMONSTRATION SUMMARY")
        print("=" * 60)
        print("✅ Advection-Diffusion with SUPG stabilization")
        print(f"   SUPG error: {results['advection_diffusion']['error_supg']:.2e}")
        print(f"   Stabilization factor: {results['advection_diffusion']['error_unstab']/results['advection_diffusion']['error_supg']:.1f}x")
        
        print("✅ Linear Elasticity (2D)")
        print(f"   Max displacement: {results['elasticity']['max_displacement']*1000:.3f} mm")
        print(f"   Solve time: {results['elasticity']['solve_time']:.3f}s")
        
        print("✅ Time-dependent Diffusion")
        print(f"   Energy conservation: {results['time_dependent']['final_energy_be']/results['time_dependent']['final_energy_cn']:.3f}")
        print(f"   Solve time: {results['time_dependent']['solve_time']:.3f}s")
        
        print("\n🎯 Generation 1 Enhanced Capabilities Demonstrated Successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logger.error(f"Enhanced physics demo error: {e}", exc_info=True)
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)