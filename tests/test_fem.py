"""FEM correctness tests.

Verify that DifferentiableFESolver solves known problems exactly (up to
machine precision / mesh discretisation error).

1D Poisson test:
    -d²u/dx² = 1   on (0,1),   u(0)=u(1)=0
    exact: u(x) = x(1-x)/2

The FEM solution with linear elements is exact at nodes for this
polynomial source, so we expect very tight agreement.
"""

import math
import pytest
import torch
import numpy as np

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from diffhe.mesh import FEMesh
from diffhe.solver import DifferentiableFESolver


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def line_mesh_10():
    return FEMesh.line(n_elements=10)

@pytest.fixture
def line_mesh_100():
    return FEMesh.line(n_elements=100)


# -----------------------------------------------------------------------
# Tests — mesh
# -----------------------------------------------------------------------

class TestFEMesh:
    def test_1d_shape(self, line_mesh_10):
        m = line_mesh_10
        assert m.n_nodes == 11
        assert m.n_elements == 10
        assert m.dim == 1

    def test_1d_bc(self, line_mesh_10):
        m = line_mesh_10
        assert 0 in m.dirichlet_nodes
        assert 10 in m.dirichlet_nodes
        assert m.dirichlet_nodes[0] == 0.0
        assert m.dirichlet_nodes[10] == 0.0

    def test_free_nodes(self, line_mesh_10):
        free = line_mesh_10.free_nodes()
        assert len(free) == 9
        assert 0 not in free
        assert 10 not in free

    def test_2d_shape(self):
        m = FEMesh.rectangle(nx=4, ny=4)
        assert m.n_nodes == 25
        assert m.n_elements == 32
        assert m.dim == 2

    def test_2d_boundary_nodes(self):
        m = FEMesh.rectangle(nx=4, ny=4)
        # corner + edges: 4*(4+1) - 4 = 16 boundary nodes
        assert len(m.dirichlet_nodes) == 16


# -----------------------------------------------------------------------
# Tests — 1D solver exactness
# -----------------------------------------------------------------------

class TestDifferentiableFESolver1D:
    """For -u'' = 1 on [0,1] with u(0)=u(1)=0, exact = x(1-x)/2."""

    def _exact(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 - x) / 2.0

    def test_coarse_exact(self, line_mesh_10):
        mesh = line_mesh_10
        solver = DifferentiableFESolver(mesh)
        x = mesh.nodes.squeeze(1)
        f = torch.ones_like(x)
        u = solver(f)
        u_exact = self._exact(x)
        # P1 FEM is exact for linear forcing on uniform mesh
        assert torch.allclose(u, u_exact, atol=1e-10), (
            f"max error = {(u - u_exact).abs().max():.2e}"
        )

    def test_fine_exact(self, line_mesh_100):
        mesh = line_mesh_100
        solver = DifferentiableFESolver(mesh)
        x = mesh.nodes.squeeze(1)
        f = torch.ones_like(x)
        u = solver(f)
        u_exact = self._exact(x)
        assert torch.allclose(u, u_exact, atol=1e-9)

    def test_bc_satisfied(self, line_mesh_10):
        mesh = line_mesh_10
        solver = DifferentiableFESolver(mesh)
        x = mesh.nodes.squeeze(1)
        u = solver(torch.ones_like(x))
        assert abs(float(u[0])) < 1e-12
        assert abs(float(u[-1])) < 1e-12

    def test_sinusoidal_forcing(self):
        """For -u'' = π²sin(πx), exact = sin(πx).  Check convergence."""
        errors = []
        for n in [10, 20, 40, 80]:
            mesh = FEMesh.line(n_elements=n)
            solver = DifferentiableFESolver(mesh)
            x = mesh.nodes.squeeze(1)
            f = (math.pi ** 2) * torch.sin(math.pi * x)
            u = solver(f)
            u_exact = torch.sin(math.pi * x)
            err = float((u - u_exact).abs().max())
            errors.append(err)

        # Check second-order convergence (h² → error halves every doubling)
        for i in range(1, len(errors)):
            ratio = errors[i - 1] / (errors[i] + 1e-15)
            assert ratio > 3.0, (
                f"Expected ~4x reduction, got {ratio:.2f} ({errors[i-1]:.2e} → {errors[i]:.2e})"
            )

    def test_nonzero_dirichlet_bc(self):
        """u'' = 0 with u(0)=1, u(1)=2 → exact = 1 + x."""
        mesh = FEMesh.line(n_elements=10, bc_left=1.0, bc_right=2.0)
        solver = DifferentiableFESolver(mesh)
        x = mesh.nodes.squeeze(1)
        f = torch.zeros_like(x)
        u = solver(f)
        u_exact = 1.0 + x
        assert torch.allclose(u, u_exact, atol=1e-10)

    def test_gradient_flows(self):
        """Gradient should flow back through the FEM solve (for kappa)."""
        kappa = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        mesh = FEMesh.line(n_elements=5)
        solver = DifferentiableFESolver(mesh, kappa=kappa)
        x = mesh.nodes.squeeze(1)
        f = torch.ones_like(x)
        u = solver(f)
        loss = u.sum()
        loss.backward()
        assert kappa.grad is not None
        assert kappa.grad.abs() > 1e-10


# -----------------------------------------------------------------------
# Tests — 2D solver (qualitative)
# -----------------------------------------------------------------------

class TestDifferentiableFESolver2D:
    def test_zero_forcing_gives_zero(self):
        """With f=0 and zero Dirichlet BCs, solution should be zero."""
        mesh = FEMesh.rectangle(nx=4, ny=4)
        solver = DifferentiableFESolver(mesh)
        f = torch.zeros(mesh.n_nodes, dtype=torch.float64)
        u = solver(f)
        assert u.abs().max() < 1e-10

    def test_solution_in_range(self):
        """With f>0 and zero BCs, interior solution should be positive."""
        mesh = FEMesh.rectangle(nx=8, ny=8)
        solver = DifferentiableFESolver(mesh)
        f = torch.ones(mesh.n_nodes, dtype=torch.float64)
        u = solver(f)
        free = mesh.free_nodes()
        u_free = u[free]
        assert float(u_free.min()) > 0.0
