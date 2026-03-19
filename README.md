# DiffFE-Physics-Lab

Differentiable finite elements with ML integration.  Gradient-based
optimisation of physical systems through differentiable PDE solvers.

## What this is

The finite element method (FEM) is the workhorse of computational physics —
it turns a PDE like the Poisson equation into a sparse linear system and
solves it exactly.  This library wires FEM into PyTorch so the solution is
**differentiable**: you can backpropagate through the solve to optimise
material parameters, source terms, or boundary conditions.

Alongside the classical solver, a small neural network (`NeuralPDE`) can
learn the same solution from the PDE residual — giving a mesh-free,
continuous approximation that blends physics and ML.

---

## Core classes

| Class | What it does |
|---|---|
| `FEMesh` | 1D or 2D finite element mesh (nodes, elements, Dirichlet BCs) |
| `DifferentiableFESolver` | Assembles the FEM stiffness matrix and solves via `torch.linalg.solve` — fully differentiable |
| `PhysicsLoss` | Loss measuring FEM residual mismatch (train NN to satisfy the PDE) |
| `NeuralPDE` | Small MLP with boundary-zero mask; trained by `PhysicsLoss` |

---

## Quick start

```python
from diffhe import FEMesh, DifferentiableFESolver, NeuralPDE
import torch

# Solve  -u'' = 1  on [0,1]  with  u(0)=u(1)=0
mesh   = FEMesh.line(n_elements=20)
solver = DifferentiableFESolver(mesh)
x      = mesh.nodes.squeeze(1)
u_fem  = solver(torch.ones_like(x))          # exact: x(1-x)/2

# Train a neural network to learn the same solution
model  = NeuralPDE(mesh, hidden_dim=64, n_layers=3)
losses = model.train_pde(
    forcing_fn=lambda x: torch.ones_like(x),
    n_epochs=3000,
)
u_nn = model()
```

---

## The physics

### 1D Poisson / heat equation

$$-\frac{d}{dx}\!\left(\kappa\, \frac{du}{dx}\right) = f(x), \quad x \in (0,1), \quad u(0)=u(1)=0$$

The weak form (multiply by a test function $v$ and integrate by parts):

$$\int_0^1 \kappa\, u' v'\, dx = \int_0^1 f\, v\, dx$$

FEM discretises this with piecewise-linear hat functions, giving
$\mathbf{K}\mathbf{u} = \mathbf{F}$.  For $f=1$ the exact solution is
$u(x) = x(1-x)/2$; P1 FEM reproduces this exactly at the nodes.

### Differentiable solve

`torch.linalg.solve` is differentiable — so gradients flow back through
$\mathbf{u} = \mathbf{K}^{-1}\mathbf{F}$ to any upstream parameter (e.g. $\kappa$):

```python
kappa = torch.tensor(1.0, requires_grad=True)
solver = DifferentiableFESolver(mesh, kappa=kappa)
u = solver(f)
loss = some_objective(u)
loss.backward()           # ∂loss/∂κ is computed automatically
```

### Neural PDE (Physics-Informed NN)

`NeuralPDE` enforces Dirichlet BCs via the lifting trick:

$$\hat{u}(x) = \varphi(x)\cdot\text{net}(x)$$

where $\varphi(x) = (x-a)(b-x)$ vanishes on $\partial\Omega$.  The
`PhysicsLoss` then minimises $\|\hat{u} - u_{\text{FEM}}\|^2$ — training
the network to agree with the classical solver.

---

## Demo

```bash
~/anaconda3/bin/python3 examples/poisson_1d_demo.py
```

Expected output:

```
[FEM]  max error vs exact: 1.11e-16  (should be ~machine precision)
[Neural PDE]  Epoch  3000  loss = 2.07e-10
[Neural PDE]  max error vs exact: 2.75e-05
[Gradient check]  True kappa = 2.000,  recovered kappa = 2.0000
```

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

18 tests covering:
- Mesh construction (1D/2D, BCs, free nodes)
- FEM exactness for polynomial and sinusoidal forcing
- Second-order convergence under mesh refinement
- Gradient flow through the FEM solve
- Neural PDE: BC enforcement, loss decrease, convergence to FEM

---

## Requirements

- Python 3.9+
- PyTorch ≥ 2.0
- NumPy ≥ 1.24

```bash
pip install -r requirements.txt
```

---

## Roadmap

- [ ] Sparse matrix assembly (scipy / torch.sparse)
- [ ] 2D convergence tests
- [ ] Topology optimisation demo (minimise compliance)
- [ ] Time-dependent problems (heat equation)
- [ ] P2 elements

---

## License

MIT
