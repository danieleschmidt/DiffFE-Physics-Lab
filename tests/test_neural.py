"""Neural PDE solver tests.

Verify that NeuralPDE can learn to approximate the FEM solution for
the 1D Poisson equation.
"""

import math
import pytest
import torch

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from diffhe.mesh import FEMesh
from diffhe.solver import DifferentiableFESolver
from diffhe.neural import NeuralPDE
from diffhe.loss import PhysicsLoss


class TestNeuralPDE:
    def test_bc_enforced_before_training(self):
        """Boundary mask should give zero on Dirichlet nodes at init."""
        mesh = FEMesh.line(n_elements=10)
        model = NeuralPDE(mesh)
        u = model().detach()
        assert abs(float(u[0])) < 1e-10, "BC left not zero"
        assert abs(float(u[-1])) < 1e-10, "BC right not zero"

    def test_bc_enforced_after_training(self):
        """BCs must hold after training."""
        mesh = FEMesh.line(n_elements=10)
        model = NeuralPDE(mesh, hidden_dim=16, n_layers=2)
        f = lambda x: torch.ones_like(x)
        model.train_pde(f, n_epochs=100, verbose=False)
        u = model()
        assert abs(float(u[0])) < 1e-10
        assert abs(float(u[-1])) < 1e-10

    def test_loss_decreases(self):
        """Training should reduce the physics loss."""
        mesh = FEMesh.line(n_elements=20)
        model = NeuralPDE(mesh, hidden_dim=32, n_layers=3)
        f = lambda x: torch.ones_like(x)
        losses = model.train_pde(f, n_epochs=300, verbose=False)
        # Loss should decrease from start to finish
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.3e} → {losses[-1]:.3e}"
        )

    def test_converges_to_fem(self):
        """After training, NN solution should approximate FEM."""
        torch.manual_seed(42)
        mesh = FEMesh.line(n_elements=20)
        model = NeuralPDE(mesh, hidden_dim=64, n_layers=3)
        f_fn = lambda x: torch.ones_like(x)

        # Get FEM reference
        solver = DifferentiableFESolver(mesh)
        x = mesh.nodes.squeeze(1)
        u_fem = solver(f_fn(x)).detach()

        losses = model.train_pde(f_fn, n_epochs=3000, lr=1e-3, verbose=False)

        u_nn = model().detach()
        # mask out boundaries (trivially zero)
        free = mesh.free_nodes()
        err = (u_nn[free] - u_fem[free]).abs().max()
        # Should be within 5% of the max FEM value
        rel = err / u_fem[free].abs().max()
        assert float(rel) < 0.05, (
            f"NN too far from FEM: relative error = {float(rel):.3f}"
        )

    def test_physics_loss_fem_match(self):
        """PhysicsLoss(mode='fem_match') decreases during training."""
        torch.manual_seed(0)
        mesh = FEMesh.line(n_elements=10)
        model = NeuralPDE(mesh, hidden_dim=32, n_layers=2)
        f_fn = lambda x: torch.ones_like(x)
        loss_fn = PhysicsLoss(mesh, f_fn, mode="fem_match")

        with torch.no_grad():
            loss_init = float(loss_fn(model()))

        losses = model.train_pde(f_fn, n_epochs=500, verbose=False)

        with torch.no_grad():
            loss_final = float(loss_fn(model()))

        assert loss_final < loss_init
