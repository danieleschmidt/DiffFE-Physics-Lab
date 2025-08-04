# DiffFE-Physics-Lab

A differentiable finite element playground integrating FEBML operators with Firedrake for physics-informed machine learning, following the July 2025 arXiv reproducibility standards.

## Overview

DiffFE-Physics-Lab provides a unified framework for combining finite element methods with automatic differentiation, enabling gradient-based optimization of physical systems. The library seamlessly integrates machine learning models into PDE solvers, supporting inverse problems, optimal control, and physics-informed neural networks.

## Key Features

- **Differentiable FEM**: Full automatic differentiation through finite element assembly and solve
- **FEBML Operators**: Pre-built operators for common physics (elasticity, fluid dynamics, heat transfer)
- **Firedrake Integration**: Leverages Firedrake's code generation for performance
- **Multi-Physics Coupling**: Compose complex multi-physics simulations with AD support
- **GPU Acceleration**: CUDA kernels for assembly and linear algebra operations
- **Reproducibility Tools**: Automated experiment tracking and result verification

## Core Capabilities

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDE Problem   │────▶│  FEBML Ops   │────▶│  ML Model   │
│  Specification  │     │  (Diffable)  │     │ Integration │
└─────────────────┘     └──────────────┘     └─────────────┘
         │                      │                     │
         ▼                      ▼                     ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│ Mesh Generation │     │   Assembly   │     │  Optimizer  │
│  & Adaptation   │     │   Engine     │     │  (JAX/Torch)│
└─────────────────┘     └──────────────┘     └─────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- PETSc 3.19+ (with CUDA support optional)
- Firedrake (latest)
- JAX 0.4.25+ or PyTorch 2.4+
- CUDA 12.0+ (for GPU support)

### Standard Installation

```bash
# Install Firedrake first
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install

# Activate Firedrake environment
source firedrake/bin/activate

# Install DiffFE-Physics-Lab
git clone https://github.com/yourusername/DiffFE-Physics-Lab
cd DiffFE-Physics-Lab
pip install -e .
```

### Docker Installation

```bash
docker pull ghcr.io/yourusername/diffhe-physics:latest
docker run --gpus all -it diffhe-physics:latest
```

## Quick Start

### Basic Example: Inverse Heat Conduction

```python
from diffhe import FEBMLProblem, operators as ops
import firedrake as fd
import jax.numpy as jnp
from jax import grad, jit

# Define mesh and function spaces
mesh = fd.UnitSquareMesh(50, 50)
V = fd.FunctionSpace(mesh, "CG", 1)

# Set up differentiable FEM problem
problem = FEBMLProblem(V)

# Define heat equation with unknown conductivity
@problem.differentiable
def heat_equation(u, k):
    return ops.laplacian(u, k) + problem.source_term

# Observation data
observed = problem.generate_observations(num_points=100)

# Loss function
def loss_fn(k_params):
    k = problem.parameterize_field(k_params)
    u = problem.solve(heat_equation, k)
    return jnp.mean((problem.observe(u) - observed)**2)

# Optimize conductivity field
k_opt = problem.optimize(loss_fn, initial_guess=jnp.ones(100))
```

### Advanced Example: Fluid-Structure Interaction

```python
from diffhe import MultiPhysics, operators as ops
from diffhe.domains import FluidDomain, SolidDomain

# Create coupled system
fsi = MultiPhysics()

# Add fluid domain (Navier-Stokes)
fluid = FluidDomain(mesh_resolution=100)
fsi.add_domain("fluid", fluid, equations=[
    ops.navier_stokes(Re=100),
    ops.incompressibility()
])

# Add solid domain (Hyperelastic)
solid = SolidDomain(mesh_resolution=50)
fsi.add_domain("solid", solid, equations=[
    ops.hyperelastic(material="neo_hookean", E=1e6, nu=0.3)
])

# Define coupling conditions
fsi.add_interface("fluid_solid_interface", 
    fluid_condition=ops.no_slip(),
    solid_condition=ops.traction_continuity()
)

# Solve with shape optimization
@fsi.differentiable
def drag_coefficient(shape_params):
    fsi.update_shape(shape_params)
    solution = fsi.solve()
    return ops.compute_drag(solution.fluid.u, solution.fluid.p)

optimal_shape = fsi.optimize_shape(
    objective=drag_coefficient,
    constraints=[ops.volume_constraint(V0=1.0)]
)
```

## FEBML Operators

### Available Operators

| Operator | Physics | Differentiable | GPU Support |
|----------|---------|----------------|-------------|
| `laplacian` | Diffusion | ✓ | ✓ |
| `advection` | Transport | ✓ | ✓ |
| `elasticity` | Linear elasticity | ✓ | ✓ |
| `hyperelastic` | Nonlinear elasticity | ✓ | ✓ |
| `navier_stokes` | Fluid dynamics | ✓ | ✓ |
| `maxwell` | Electromagnetics | ✓ | ✗ |
| `schrodinger` | Quantum mechanics | ✓ | ✓ |

### Custom Operator Definition

```python
from diffhe import Operator, register_operator

@register_operator("my_custom_op")
class CustomOperator(Operator):
    def forward(self, u, params):
        # Define forward operation
        return self.assemble(u, params)
    
    def backward(self, grad_output):
        # Define adjoint operation
        return self.adjoint_assemble(grad_output)
    
    @property
    def is_linear(self):
        return False
```

## Physics-Informed Neural Networks

### PINN Integration

```python
from diffhe.ml import PINN
import jax.nn as jnn

# Define neural network
def mlp(params, x):
    for w, b in params[:-1]:
        x = jnn.relu(jnp.dot(x, w) + b)
    w, b = params[-1]
    return jnp.dot(x, w) + b

# Create PINN solver
pinn = PINN(
    network=mlp,
    pde=ops.helmholtz(k=1.0),
    boundary_conditions={
        "dirichlet": lambda x: jnp.sin(jnp.pi * x[0]),
        "neumann": lambda x: 0.0
    }
)

# Train
params = pinn.train(
    num_epochs=10000,
    batch_size=1000,
    learning_rate=1e-3
)
```

### Hybrid FEM-NN Solver

```python
from diffhe.hybrid import HybridSolver

# Combine FEM accuracy with NN flexibility
hybrid = HybridSolver(
    fem_solver=problem,
    neural_correction=mlp,
    coupling_strength=0.1
)

solution = hybrid.solve(
    pde=ops.reaction_diffusion(),
    use_fem_for=["diffusion"],
    use_nn_for=["reaction"]
)
```

## Reproducibility Features

### Experiment Tracking

```python
from diffhe.reproducibility import Experiment

with Experiment("inverse_problem_v1") as exp:
    # All computations are logged
    exp.log_params({"mesh_size": 100, "Re": 1000})
    
    result = problem.solve()
    exp.log_metrics({"error": compute_error(result)})
    
    # Automatic checkpointing
    exp.checkpoint(result, "final_solution")
```

### Verification Suite

```bash
# Run reproducibility checks
python -m diffhe.verify --experiment inverse_problem_v1

# Generate LaTeX table for paper
python -m diffhe.reports --format latex --output results.tex
```

### Benchmarks

```bash
# Run standard benchmark suite
python benchmarks/run_all.py --gpu --precision float64

# Results are compared against reference solutions
```

## Advanced Features

### Adaptive Mesh Refinement

```python
from diffhe.adaptivity import AdaptiveSolver

solver = AdaptiveSolver(
    error_estimator="dual_weighted_residual",
    target_error=1e-6
)

# Mesh adapts during optimization
solution = solver.solve_adaptive(
    problem,
    max_iterations=10,
    refinement_fraction=0.3
)
```

### Uncertainty Quantification

```python
from diffhe.uq import MCMCSampler

# Bayesian inverse problem
sampler = MCMCSampler(
    likelihood=problem.likelihood,
    prior=ops.gaussian_random_field(length_scale=0.1)
)

samples = sampler.sample(
    num_samples=10000,
    num_chains=4,
    target_accept=0.8
)
```

### Parallel Solvers

```python
from diffhe.parallel import DistributedSolver
import mpi4py.MPI as MPI

# Domain decomposition
solver = DistributedSolver(
    problem,
    decomposition="metis",
    overlap=2
)

# Scales to 1000+ cores
solution = solver.solve_parallel(
    linear_solver="multigrid",
    preconditioner="ilu"
)
```

## Performance Optimization

### GPU Acceleration

```python
# Enable GPU assembly
problem.enable_gpu(device=0)

# Custom CUDA kernels
from diffhe.cuda import custom_kernel

@custom_kernel
def fast_assembly(elements, quadrature):
    # CUDA kernel code
    pass
```

### JIT Compilation

```python
# Firedrake + JAX JIT
@jit
def optimized_solve(params):
    return problem.solve_with_params(params)

# 10-100x speedup for repeated solves
```

## Examples Gallery

The `examples/` directory contains:

1. **Topology Optimization**: Minimize compliance subject to volume constraints
2. **Inverse Scattering**: Recover material properties from wave measurements  
3. **Optimal Control**: Control heat source to achieve target temperature
4. **Neural Operators**: Learn solution operators for parametric PDEs
5. **Multi-Scale Modeling**: Couple atomistic and continuum models

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires GPU)
pytest tests/integration/ -v --gpu

# Convergence tests
python tests/convergence/run_convergence_studies.py
```

### Code Style

```bash
# Format code
black diffhe/ tests/
isort diffhe/ tests/

# Type checking
mypy diffhe/
```

## Citation

```bibtex
@article{diffhe-physics-2025,
  title={Differentiable Finite Elements for Physics-Informed Machine Learning},
  author={Daniel Schmidt},
  journal={arXiv preprint arXiv:2507.XXXXX},
  year={2025}
}
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Firedrake project for the FEM infrastructure
- JAX team for automatic differentiation tools
- Authors of the FEBML framework
