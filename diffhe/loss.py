"""Physics-informed loss functions.

PhysicsLoss measures how well a predicted solution u(x) satisfies
the PDE -d²u/dx² = f(x) using the FEM residual: the mismatch between
the FEM solution (the "correct" answer for this f) and the prediction.

Alternatively it can measure the variational (energy) residual directly
from the neural network output using automatic differentiation.
"""

from __future__ import annotations
from typing import Callable

import torch
import torch.nn as nn

from .mesh import FEMesh
from .solver import DifferentiableFESolver


class PhysicsLoss(nn.Module):
    """FEM-residual physics loss.

    Two modes:

    1. **FEM-match** (default): compare NN prediction to the FEM solution.
       Loss = MSE(u_nn, u_fem).  Trains the NN to replicate FEM.

    2. **Variational**: penalise the strong-form residual at collocation
       points.  Loss = mean( (Δu_nn + f)² ).  Mesh-free, gradient-based.

    Parameters
    ----------
    mesh : FEMesh
        Discretisation domain.
    forcing_fn : callable
        f(x) — right-hand side of -d²u/dx² = f.
    mode : str
        "fem_match" or "variational".
    solver : DifferentiableFESolver, optional
        Pre-built solver.  Created from *mesh* if not provided.
    """

    def __init__(
        self,
        mesh: FEMesh,
        forcing_fn: Callable[[torch.Tensor], torch.Tensor],
        mode: str = "fem_match",
        solver: DifferentiableFESolver | None = None,
    ):
        super().__init__()
        if mode not in ("fem_match", "variational"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mesh = mesh
        self.forcing_fn = forcing_fn
        self.mode = mode
        self.solver = solver or DifferentiableFESolver(mesh)

    # ------------------------------------------------------------------

    def forward(self, u_pred: torch.Tensor) -> torch.Tensor:
        """Compute physics loss.

        Parameters
        ----------
        u_pred : torch.Tensor  shape (n_nodes,)
            Predicted solution at mesh nodes.

        Returns
        -------
        loss : torch.Tensor  scalar
        """
        if self.mode == "fem_match":
            return self._fem_match_loss(u_pred)
        else:
            return self._variational_loss(u_pred)

    def _fem_match_loss(self, u_pred: torch.Tensor) -> torch.Tensor:
        x = self.mesh.nodes.squeeze(1)  # (n_nodes,)  for 1D
        f = self.forcing_fn(x)
        with torch.no_grad():
            u_fem = self.solver(f)
        return nn.functional.mse_loss(u_pred.double(), u_fem.double())

    def _variational_loss(self, u_pred: torch.Tensor) -> torch.Tensor:
        """Residual at interior nodes using finite-difference Laplacian."""
        x = self.mesh.nodes.squeeze(1)
        f = self.forcing_fn(x)
        free = self.mesh.free_nodes()
        x_free = x[free]
        u_free = u_pred[free].double()

        # finite-difference second derivative at interior points
        h = float(x_free[1] - x_free[0]) if len(x_free) > 1 else 1.0
        if len(x_free) >= 3:
            # interior of free nodes
            u2 = u_free[2:]
            u1 = u_free[1:-1]
            u0 = u_free[:-2]
            lap = (u0 - 2 * u1 + u2) / (h ** 2)
            residual = lap + f[free][1:-1].double()
        else:
            residual = torch.zeros(1, dtype=torch.float64)

        return (residual ** 2).mean()
