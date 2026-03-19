"""Differentiable FEM solver for linear PDEs.

Solves -∇·(κ ∇u) = f  with Dirichlet boundary conditions using
linear finite elements.  The stiffness matrix assembly and linear
solve are all expressed through PyTorch operations so that gradients
can flow through the solution (via torch.linalg.solve).

1D specialisation: -d/dx(κ du/dx) = f
2D specialisation: -(κ ∆u) = f  (triangular P1 elements)
"""

from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn

from .mesh import FEMesh


class DifferentiableFESolver(nn.Module):
    """Assemble and solve the FEM system for a given mesh and forcing.

    Parameters
    ----------
    mesh : FEMesh
        The discretisation (nodes, elements, BCs).
    kappa : float or torch.Tensor
        Diffusion coefficient (scalar or per-element).  Default 1.0.
    """

    def __init__(self, mesh: FEMesh, kappa: float = 1.0):
        super().__init__()
        self.mesh = mesh
        # kappa may be a learnable parameter — wrap as tensor
        if isinstance(kappa, (int, float)):
            self._kappa = torch.tensor(kappa, dtype=torch.float64)
        else:
            self._kappa = kappa.to(dtype=torch.float64)

    @property
    def kappa(self) -> torch.Tensor:
        return self._kappa

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Solve -∇·(κ ∇u) = f.

        Parameters
        ----------
        f : torch.Tensor  shape (n_nodes,)
            Forcing evaluated at the mesh nodes.

        Returns
        -------
        u : torch.Tensor  shape (n_nodes,)
            FEM solution satisfying Dirichlet BCs.
        """
        if self.mesh.dim == 1:
            return self._solve_1d(f)
        elif self.mesh.dim == 2:
            return self._solve_2d(f)
        else:
            raise NotImplementedError("Only 1D and 2D supported")

    # ------------------------------------------------------------------
    # 1D solver
    # ------------------------------------------------------------------

    def _solve_1d(self, f: torch.Tensor) -> torch.Tensor:
        mesh = self.mesh
        n = mesh.n_nodes
        kappa = self.kappa

        # Assemble stiffness K and load F
        K = torch.zeros((n, n), dtype=torch.float64)
        F = torch.zeros(n, dtype=torch.float64)

        for elem in mesh.elements:
            i, j = int(elem[0]), int(elem[1])
            xi, xj = mesh.nodes[i, 0], mesh.nodes[j, 0]
            h_e = xj - xi  # element length (positive)

            # Stiffness for -d/dx(κ du/dx): k_local = κ/h * [[1,-1],[-1,1]]
            k_e = kappa / h_e
            K[i, i] = K[i, i] + k_e
            K[i, j] = K[i, j] - k_e
            K[j, i] = K[j, i] - k_e
            K[j, j] = K[j, j] + k_e

            # Load: midpoint quadrature  F_local = h/2 * [f_i, f_j]
            F[i] = F[i] + h_e / 2.0 * f[i]
            F[j] = F[j] + h_e / 2.0 * f[j]

        return self._apply_bc_and_solve(K, F)

    # ------------------------------------------------------------------
    # 2D solver  (P1 triangles)
    # ------------------------------------------------------------------

    def _solve_2d(self, f: torch.Tensor) -> torch.Tensor:
        mesh = self.mesh
        n = mesh.n_nodes
        kappa = self.kappa

        K = torch.zeros((n, n), dtype=torch.float64)
        F = torch.zeros(n, dtype=torch.float64)

        for elem in mesh.elements:
            i, j, k = int(elem[0]), int(elem[1]), int(elem[2])
            xi, yi = mesh.nodes[i, 0], mesh.nodes[i, 1]
            xj, yj = mesh.nodes[j, 0], mesh.nodes[j, 1]
            xk, yk = mesh.nodes[k, 0], mesh.nodes[k, 1]

            # Area of triangle (signed → take abs)
            area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            if area < 1e-15:
                continue

            # Gradients of P1 basis functions
            # φ_i = (a_i + b_i*x + c_i*y) / (2*area)
            b = torch.stack([
                (yj - yk).detach().clone().to(dtype=torch.float64),
                (yk - yi).detach().clone().to(dtype=torch.float64),
                (yi - yj).detach().clone().to(dtype=torch.float64),
            ])
            c = torch.stack([
                (xk - xj).detach().clone().to(dtype=torch.float64),
                (xi - xk).detach().clone().to(dtype=torch.float64),
                (xj - xi).detach().clone().to(dtype=torch.float64),
            ])

            # k_local[p,q] = kappa * (b_p*b_q + c_p*c_q) / (4*area)
            for p_idx, p in enumerate([i, j, k]):
                for q_idx, q in enumerate([i, j, k]):
                    k_pq = kappa * (b[p_idx] * b[q_idx] + c[p_idx] * c[q_idx]) / (4.0 * area)
                    K[p, q] = K[p, q] + k_pq

            # Load: centroid quadrature  F_p += area/3 * f at centroid
            f_centroid = (f[i] + f[j] + f[k]) / 3.0
            for p in [i, j, k]:
                F[p] = F[p] + area / 3.0 * f_centroid

        return self._apply_bc_and_solve(K, F)

    # ------------------------------------------------------------------
    # BC application + solve (differentiable)
    # ------------------------------------------------------------------

    def _apply_bc_and_solve(
        self, K: torch.Tensor, F: torch.Tensor
    ) -> torch.Tensor:
        """Apply Dirichlet BCs by elimination and solve via torch.linalg.solve.

        The solve is differentiable — gradients propagate through the solution.
        """
        mesh = self.mesh
        n = mesh.n_nodes
        free = mesh.free_nodes()

        # Subtract BC contributions from RHS
        F_free = F[free].clone()
        for node_idx, val in mesh.dirichlet_nodes.items():
            g = torch.tensor(val, dtype=torch.float64)
            for fi, f_node in enumerate(free):
                F_free[fi] = F_free[fi] - K[f_node, node_idx] * g

        K_free = K[torch.tensor(free)][:, torch.tensor(free)]

        # Differentiable linear solve
        u_free = torch.linalg.solve(K_free, F_free)

        # Scatter back into full solution vector
        u = torch.zeros(n, dtype=torch.float64)
        for node_idx, val in mesh.dirichlet_nodes.items():
            u[node_idx] = val
        for fi, f_node in enumerate(free):
            u[f_node] = u_free[fi]

        return u
