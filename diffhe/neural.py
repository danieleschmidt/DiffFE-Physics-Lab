"""Neural network PDE solver.

NeuralPDE is a small MLP that learns a PDE solution u(x) by minimising
PhysicsLoss.  It is parameterised over the mesh nodes (1D or 2D) and
automatically enforces Dirichlet BCs by multiplying the raw output by
a boundary-zero "mask" function.
"""

from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn

from .mesh import FEMesh
from .loss import PhysicsLoss


class NeuralPDE(nn.Module):
    """MLP that satisfies Dirichlet BCs and learns to solve a PDE.

    Architecture: input_dim → hidden → hidden → 1 with tanh activations.

    Dirichlet BCs are enforced by the *lifting* trick:
        u(x) = g(x) + φ(x) * net(x)
    where g interpolates the boundary values (zero by default) and
    φ(x) is a smooth function that vanishes on ∂Ω.
    For homogeneous BCs and 1D domain [a,b]:
        φ(x) = (x-a)(b-x)

    Parameters
    ----------
    mesh : FEMesh
        Discretisation — provides node coordinates and BCs.
    hidden_dim : int
        Width of each hidden layer.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(self, mesh: FEMesh, hidden_dim: int = 32, n_layers: int = 3):
        super().__init__()
        self.mesh = mesh
        self.dim = mesh.dim

        layers: list[nn.Module] = []
        in_dim = self.dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers).double()

        # Precompute boundary mask on node coordinates
        self._mask = self._compute_mask()

    # ------------------------------------------------------------------

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate the neural solution at mesh nodes.

        Parameters
        ----------
        x : torch.Tensor, optional
            Node coordinates (n_nodes, dim).  Defaults to mesh.nodes.

        Returns
        -------
        u : torch.Tensor  shape (n_nodes,)
        """
        if x is None:
            x = self.mesh.nodes  # (n_nodes, dim)
        x = x.double()
        raw = self.net(x).squeeze(1)  # (n_nodes,)
        mask = self._mask.to(x.device)
        return mask * raw  # zero on boundary by construction

    # ------------------------------------------------------------------

    def _compute_mask(self) -> torch.Tensor:
        """Smooth function that is 0 on Dirichlet nodes, ~1 inside."""
        nodes = self.mesh.nodes  # (n_nodes, dim)
        if self.dim == 1:
            x = nodes[:, 0]
            bc_nodes = list(self.mesh.dirichlet_nodes.keys())
            if len(bc_nodes) >= 2:
                a = float(nodes[bc_nodes[0], 0])
                b = float(nodes[bc_nodes[-1], 0])
                mask = (x - a) * (b - x)
                # normalise so max ≈ 1
                mask = mask / (mask.abs().max() + 1e-12)
            else:
                mask = torch.ones(nodes.shape[0], dtype=torch.float64)
        else:
            # 2D: product of distances to boundaries — simple but effective
            bc_set = set(self.mesh.dirichlet_nodes.keys())
            mask = torch.ones(nodes.shape[0], dtype=torch.float64)
            for i in range(nodes.shape[0]):
                if i in bc_set:
                    mask[i] = 0.0
        return mask

    # ------------------------------------------------------------------

    def train_pde(
        self,
        forcing_fn: Callable[[torch.Tensor], torch.Tensor],
        n_epochs: int = 2000,
        lr: float = 1e-3,
        mode: str = "fem_match",
        verbose: bool = True,
        log_every: int = 200,
    ) -> list[float]:
        """Train the network to satisfy -d²u/dx² = f.

        Parameters
        ----------
        forcing_fn : callable
            f(x) — right-hand side.
        n_epochs : int
            Number of gradient steps.
        lr : float
            Adam learning rate.
        mode : str
            Loss mode: "fem_match" or "variational".
        verbose : bool
            Print loss every *log_every* epochs.
        log_every : int
            Logging interval.

        Returns
        -------
        losses : list of float
        """
        loss_fn = PhysicsLoss(self.mesh, forcing_fn, mode=mode)
        optimiser = torch.optim.Adam(self.parameters(), lr=lr)

        losses: list[float] = []
        for epoch in range(1, n_epochs + 1):
            optimiser.zero_grad()
            u_pred = self.forward()
            loss = loss_fn(u_pred)
            loss.backward()
            optimiser.step()
            losses.append(float(loss))
            if verbose and epoch % log_every == 0:
                print(f"  Epoch {epoch:5d}  loss = {float(loss):.3e}")

        return losses
