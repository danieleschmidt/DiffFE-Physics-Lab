"""Demo: 1D Poisson equation — FEM vs Neural PDE.

Solves  -d²u/dx² = 1  on [0,1]  with  u(0)=u(1)=0.
Exact solution: u(x) = x(1-x)/2.

Compares:
  1. FEM solution (exact at nodes for this problem)
  2. Neural PDE solution trained to match FEM
  3. Analytical exact solution

Run with:
    ~/anaconda3/bin/python3 examples/poisson_1d_demo.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import math
import torch

from diffhe.mesh import FEMesh
from diffhe.solver import DifferentiableFESolver
from diffhe.neural import NeuralPDE

torch.manual_seed(42)

# -----------------------------------------------------------------------
# Problem setup
# -----------------------------------------------------------------------

print("=" * 60)
print("1D Poisson: -u'' = 1,  u(0)=u(1)=0")
print("Exact: u(x) = x(1-x)/2")
print("=" * 60)

n_elements = 20
mesh = FEMesh.line(n_elements=n_elements)
x = mesh.nodes.squeeze(1)

# Exact solution
def exact(x):
    return x * (1.0 - x) / 2.0

# -----------------------------------------------------------------------
# FEM solution
# -----------------------------------------------------------------------

solver = DifferentiableFESolver(mesh)
f = torch.ones_like(x, dtype=torch.float64)
u_fem = solver(f).detach()
u_exact = exact(x)

fem_error = (u_fem - u_exact).abs().max()
print(f"\n[FEM]  max error vs exact: {fem_error:.2e}  (should be ~machine precision)")

# -----------------------------------------------------------------------
# Neural PDE solution
# -----------------------------------------------------------------------

print("\n[Neural PDE]  Training ...")
model = NeuralPDE(mesh, hidden_dim=64, n_layers=3)
losses = model.train_pde(
    forcing_fn=lambda x: torch.ones_like(x),
    n_epochs=3000,
    lr=1e-3,
    mode="fem_match",
    verbose=True,
    log_every=500,
)
u_nn = model().detach()
nn_error = (u_nn - u_exact).abs().max()
print(f"\n[Neural PDE]  max error vs exact: {nn_error:.2e}")

# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------

print("\n--- Solution comparison (selected nodes) ---")
print(f"{'x':>8}  {'exact':>10}  {'FEM':>10}  {'Neural':>10}")
step = max(1, n_elements // 8)
for i in range(0, len(x), step):
    xi = float(x[i])
    print(
        f"{xi:8.3f}  {float(u_exact[i]):10.6f}  "
        f"{float(u_fem[i]):10.6f}  {float(u_nn[i]):10.6f}"
    )

print("\n[Gradient check]  kappa optimisation demo ...")
# Can we recover kappa=2 by minimising FEM solution mismatch?
kappa_true = 2.0
mesh_ref = FEMesh.line(n_elements=30)
x_ref = mesh_ref.nodes.squeeze(1)
f_ref = torch.ones(mesh_ref.n_nodes, dtype=torch.float64)

# True data: -κ u'' = 1  → u = x(1-x)/(2κ)
kappa_true_t = torch.tensor(kappa_true, dtype=torch.float64)
solver_ref = DifferentiableFESolver(mesh_ref, kappa=kappa_true_t)
with torch.no_grad():
    u_data = solver_ref(f_ref)

# Learn kappa from data
kappa_est = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
optim = torch.optim.Adam([kappa_est], lr=0.1)
for step in range(200):
    optim.zero_grad()
    solver_est = DifferentiableFESolver(mesh_ref, kappa=kappa_est.abs())
    u_est = solver_est(f_ref)
    loss = ((u_est - u_data) ** 2).mean()
    loss.backward()
    optim.step()

print(f"  True kappa = {kappa_true:.3f},  recovered kappa = {float(kappa_est.abs()):.4f}")
print("\nDone.")
