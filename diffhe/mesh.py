"""Finite element mesh for 1D and 2D problems.

Supports linear elements (intervals in 1D, triangles in 2D) with
Dirichlet boundary conditions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


@dataclass
class FEMesh:
    """A finite element mesh with nodes, elements, and boundary conditions.

    1D example — N+1 nodes, N intervals::

        mesh = FEMesh.line(n_elements=10, x_left=0.0, x_right=1.0)

    2D example — uniform grid of right triangles::

        mesh = FEMesh.rectangle(nx=8, ny=8)

    Attributes
    ----------
    nodes : torch.Tensor  shape (n_nodes, dim)
        Physical coordinates of every node.
    elements : torch.Tensor  shape (n_elements, nodes_per_element)
        Node index connectivity per element.
    dirichlet_nodes : Dict[int, float]
        Mapping from node index → prescribed value.
    dim : int
        Spatial dimension (1 or 2).
    """

    nodes: torch.Tensor          # (n_nodes, dim)
    elements: torch.Tensor       # (n_elements, n_per_elem)
    dirichlet_nodes: Dict[int, float] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def dim(self) -> int:
        return self.nodes.shape[1]

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def line(
        cls,
        n_elements: int = 10,
        x_left: float = 0.0,
        x_right: float = 1.0,
        bc_left: Optional[float] = 0.0,
        bc_right: Optional[float] = 0.0,
    ) -> "FEMesh":
        """Uniform 1D mesh on [x_left, x_right] with Dirichlet BCs."""
        x = torch.linspace(x_left, x_right, n_elements + 1, dtype=torch.float64)
        nodes = x.unsqueeze(1)  # (N+1, 1)
        idx = torch.arange(n_elements, dtype=torch.long)
        elements = torch.stack([idx, idx + 1], dim=1)  # (N, 2)
        bc: Dict[int, float] = {}
        if bc_left is not None:
            bc[0] = bc_left
        if bc_right is not None:
            bc[n_elements] = bc_right
        return cls(nodes=nodes, elements=elements, dirichlet_nodes=bc)

    @classmethod
    def rectangle(
        cls,
        nx: int = 4,
        ny: int = 4,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        bc_value: float = 0.0,
    ) -> "FEMesh":
        """Uniform 2D grid with Dirichlet BCs on all boundary nodes.

        Each quad is split into 2 triangles (lower-left diagonal split).
        """
        xs = np.linspace(x_range[0], x_range[1], nx + 1)
        ys = np.linspace(y_range[0], y_range[1], ny + 1)
        xx, yy = np.meshgrid(xs, ys)  # (ny+1, nx+1)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # row-major

        def nid(i, j):  # i=row (y), j=col (x)
            return i * (nx + 1) + j

        tris = []
        for i in range(ny):
            for j in range(nx):
                a, b, c, d = nid(i, j), nid(i, j + 1), nid(i + 1, j + 1), nid(i + 1, j)
                tris.append([a, b, d])  # lower-left triangle
                tris.append([b, c, d])  # upper-right triangle

        nodes = torch.tensor(coords, dtype=torch.float64)
        elements = torch.tensor(tris, dtype=torch.long)

        n_nodes = (nx + 1) * (ny + 1)
        bc: Dict[int, float] = {}
        for k in range(n_nodes):
            x, y = coords[k]
            if (
                np.isclose(x, x_range[0])
                or np.isclose(x, x_range[1])
                or np.isclose(y, y_range[0])
                or np.isclose(y, y_range[1])
            ):
                bc[k] = bc_value
        return cls(nodes=nodes, elements=elements, dirichlet_nodes=bc)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def free_nodes(self) -> List[int]:
        """Node indices not constrained by Dirichlet BCs."""
        return [i for i in range(self.n_nodes) if i not in self.dirichlet_nodes]

    def h(self) -> float:
        """Characteristic element size (minimum edge length)."""
        if self.dim == 1:
            diffs = self.nodes[self.elements[:, 1]] - self.nodes[self.elements[:, 0]]
            return float(diffs.abs().min())
        raise NotImplementedError("h() not implemented for dim>1 yet")

    def __repr__(self) -> str:
        return (
            f"FEMesh(dim={self.dim}, n_nodes={self.n_nodes}, "
            f"n_elements={self.n_elements}, "
            f"n_dirichlet={len(self.dirichlet_nodes)})"
        )
