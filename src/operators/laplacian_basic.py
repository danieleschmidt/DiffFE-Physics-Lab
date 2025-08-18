"""Basic numpy-based Laplacian operator implementation."""

from typing import Any, Callable, Dict, Tuple
import numpy as np
from scipy.sparse import csr_matrix

from .base import BaseOperator, LinearOperator, register_operator
from ..utils.mesh import SimpleMesh, SimpleFunctionSpace
from ..utils.fem_assembly import FEMAssembler


@register_operator("laplacian_basic")
class BasicLaplacianOperator(LinearOperator):
    """Basic Laplacian operator using numpy/scipy (no Firedrake dependency).

    Implements the weak form: ∫ κ ∇u · ∇v dx = ∫ f v dx
    where κ is the diffusion coefficient and f is the source term.

    Parameters
    ----------
    diffusion_coeff : float or Callable, optional
        Diffusion coefficient κ, by default 1.0
    **kwargs
        Additional parameters passed to BaseOperator

    Examples
    --------
    >>> op = BasicLaplacianOperator(diffusion_coeff=2.0)
    >>> mesh = create_1d_mesh(0, 1, 10)
    >>> V = SimpleFunctionSpace(mesh, "P1") 
    >>> K, b = op.assemble_system(V, {'source': lambda x: 1.0})
    """

    _is_linear = True
    _is_symmetric = True

    def __init__(self, diffusion_coeff=1.0, **kwargs):
        super().__init__(backend="numpy", **kwargs)
        self.diffusion_coeff = diffusion_coeff

    def forward_assembly(self, trial: Any, test: Any, params: Dict[str, Any]) -> Any:
        """Assemble Laplacian weak form using basic FEM.

        Note: This method signature is kept for compatibility, but the actual
        assembly is done through assemble_system method for basic FEM.

        Parameters
        ----------
        trial : Any
            Trial function (not used in basic implementation)
        test : Any
            Test function (not used in basic implementation)
        params : Dict[str, Any]
            Parameters including function space and source term

        Returns
        -------
        Any
            Placeholder return for compatibility
        """
        raise NotImplementedError(
            "Use assemble_system method for basic FEM assembly"
        )

    def adjoint_assembly(
        self, grad_output: Any, trial: Any, test: Any, params: Dict[str, Any]
    ) -> Any:
        """Assemble adjoint Laplacian operator.

        For the symmetric Laplacian, the adjoint is the same as the forward operator.
        """
        return self.forward_assembly(trial, test, params)

    def assemble_system(
        self, function_space: SimpleFunctionSpace, params: Dict[str, Any] = None
    ) -> Tuple[csr_matrix, np.ndarray]:
        """Assemble Laplacian system matrices.

        Parameters
        ----------
        function_space : SimpleFunctionSpace
            Function space for assembly
        params : Dict[str, Any], optional
            Parameters including source term, by default None

        Returns
        -------
        Tuple[csr_matrix, np.ndarray]
            Stiffness matrix and load vector
        """
        if params is None:
            params = {}

        # Create assembler
        assembler = FEMAssembler(function_space)

        # Get diffusion coefficient
        kappa = params.get("diffusion_coeff", self.diffusion_coeff)

        # Assemble stiffness matrix
        K = assembler.assemble_stiffness_matrix(kappa)

        # Assemble load vector
        source = params.get("source", None)
        source_values = params.get("source_values", None)
        b = assembler.assemble_load_vector(source, source_values)

        return K, b

    def apply_boundary_conditions(
        self,
        K: csr_matrix,
        b: np.ndarray,
        function_space: SimpleFunctionSpace,
        boundary_conditions: Dict[str, Dict[str, Any]],
    ) -> Tuple[csr_matrix, np.ndarray]:
        """Apply boundary conditions to the system.

        Parameters
        ----------
        K : csr_matrix
            Stiffness matrix
        b : np.ndarray
            Load vector
        function_space : SimpleFunctionSpace
            Function space
        boundary_conditions : Dict[str, Dict[str, Any]]
            Boundary conditions

        Returns
        -------
        Tuple[csr_matrix, np.ndarray]
            Modified system matrix and load vector
        """
        assembler = FEMAssembler(function_space)
        return assembler.apply_dirichlet_bcs(K, b, boundary_conditions)

    def solve_1d(
        self,
        x_start: float = 0.0,
        x_end: float = 1.0,
        num_elements: int = 10,
        source_function: Callable = None,
        left_bc: float = 0.0,
        right_bc: float = 1.0,
        **params
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve 1D Laplace equation.

        Parameters
        ----------
        x_start : float, optional
            Domain start, by default 0.0
        x_end : float, optional
            Domain end, by default 1.0
        num_elements : int, optional
            Number of elements, by default 10
        source_function : Callable, optional
            Source function f(x), by default None
        left_bc : float, optional
            Left boundary value, by default 0.0
        right_bc : float, optional
            Right boundary value, by default 1.0
        **params
            Additional parameters

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Node coordinates and solution values
        """
        from ..utils.mesh import create_1d_mesh, SimpleFunctionSpace
        from scipy.sparse.linalg import spsolve

        # Create mesh and function space
        mesh = create_1d_mesh(x_start, x_end, num_elements)
        V = SimpleFunctionSpace(mesh, "P1")

        # Assemble system
        solve_params = {"source": source_function, **params}
        K, b = self.assemble_system(V, solve_params)

        # Apply boundary conditions
        bcs = {
            "left": {"type": "dirichlet", "value": left_bc},
            "right": {"type": "dirichlet", "value": right_bc}
        }
        K_bc, b_bc = self.apply_boundary_conditions(K, b, V, bcs)

        # Solve
        solution = spsolve(K_bc, b_bc)

        return mesh.nodes[:, 0], solution

    def solve_2d(
        self,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        nx: int = 10,
        ny: int = 10,
        source_function: Callable = None,
        boundary_values: Dict[str, float] = None,
        **params
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve 2D Laplace equation.

        Parameters
        ----------
        x_range : Tuple[float, float], optional
            X domain range, by default (0.0, 1.0)
        y_range : Tuple[float, float], optional
            Y domain range, by default (0.0, 1.0)
        nx : int, optional
            Number of x divisions, by default 10
        ny : int, optional
            Number of y divisions, by default 10
        source_function : Callable, optional
            Source function f(x,y), by default None
        boundary_values : Dict[str, float], optional
            Boundary values by name, by default None
        **params
            Additional parameters

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Node coordinates and solution values
        """
        from ..utils.mesh import create_2d_rectangle_mesh, SimpleFunctionSpace
        from scipy.sparse.linalg import spsolve

        # Default boundary values
        if boundary_values is None:
            boundary_values = {"left": 0.0, "right": 1.0, "bottom": 0.0, "top": 0.0}

        # Create mesh and function space
        mesh = create_2d_rectangle_mesh(x_range, y_range, nx, ny)
        V = SimpleFunctionSpace(mesh, "P1")

        # Assemble system
        solve_params = {"source": source_function, **params}
        K, b = self.assemble_system(V, solve_params)

        # Apply boundary conditions
        bcs = {}
        for name, value in boundary_values.items():
            bcs[name] = {"type": "dirichlet", "value": value}
        K_bc, b_bc = self.apply_boundary_conditions(K, b, V, bcs)

        # Solve
        solution = spsolve(K_bc, b_bc)

        return mesh.nodes, solution

    def manufactured_solution(self, **kwargs) -> Dict[str, Callable]:
        """Generate manufactured solution for Laplacian.

        Parameters
        ----------
        **kwargs
            Additional parameters (e.g., 'frequency', 'dimension')

        Returns
        -------
        Dict[str, Callable]
            Dictionary with 'solution' and 'source' functions
        """
        frequency = kwargs.get("frequency", 1.0)
        dimension = kwargs.get("dimension", 2)
        kappa = kwargs.get("diffusion_coeff", self.diffusion_coeff)

        if dimension == 1:

            def solution(x):
                if isinstance(x, np.ndarray) and x.ndim > 1:
                    return np.sin(frequency * np.pi * x[:, 0])
                else:
                    return np.sin(frequency * np.pi * x)

            def source(x):
                if isinstance(x, np.ndarray) and x.ndim > 1:
                    return (kappa * (frequency * np.pi) ** 2 * 
                           np.sin(frequency * np.pi * x[:, 0]))
                else:
                    return (kappa * (frequency * np.pi) ** 2 * 
                           np.sin(frequency * np.pi * x))

        elif dimension == 2:

            def solution(x):
                if isinstance(x, np.ndarray) and x.ndim > 1:
                    return (np.sin(frequency * np.pi * x[:, 0]) * 
                           np.sin(frequency * np.pi * x[:, 1]))
                else:
                    return (np.sin(frequency * np.pi * x[0]) * 
                           np.sin(frequency * np.pi * x[1]))

            def source(x):
                factor = kappa * 2 * (frequency * np.pi) ** 2
                if isinstance(x, np.ndarray) and x.ndim > 1:
                    return (factor * np.sin(frequency * np.pi * x[:, 0]) * 
                           np.sin(frequency * np.pi * x[:, 1]))
                else:
                    return (factor * np.sin(frequency * np.pi * x[0]) * 
                           np.sin(frequency * np.pi * x[1]))

        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

        return {"solution": solution, "source": source}

    def compute_error(
        self, computed_solution: np.ndarray, exact_solution: Callable, 
        mesh: SimpleMesh, norm_type: str = "L2"
    ) -> float:
        """Compute error for Laplacian problems.

        Parameters
        ----------
        computed_solution : np.ndarray
            Computed solution values
        exact_solution : Callable
            Exact solution function
        mesh : SimpleMesh
            Computational mesh
        norm_type : str, optional
            Error norm type ('L2', 'max'), by default 'L2'

        Returns
        -------
        float
            Error value
        """
        # Evaluate exact solution at mesh nodes
        if mesh.dimension == 1:
            exact_values = np.array([exact_solution(node) for node in mesh.nodes[:, 0]])
        else:
            exact_values = np.array([exact_solution(node) for node in mesh.nodes])

        # Compute error
        error = computed_solution - exact_values

        if norm_type.upper() == "L2":
            return np.sqrt(np.mean(error**2))
        elif norm_type.upper() == "MAX" or norm_type.upper() == "LINF":
            return np.max(np.abs(error))
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")


def basic_laplacian(function_space: SimpleFunctionSpace, 
                   diffusion_coeff=1.0, source=None, **kwargs) -> Tuple[csr_matrix, np.ndarray]:
    """Convenience function for basic Laplacian operator.

    Parameters
    ----------
    function_space : SimpleFunctionSpace
        Function space for assembly
    diffusion_coeff : float or Callable, optional
        Diffusion coefficient, by default 1.0
    source : float or Callable, optional
        Source term, by default None
    **kwargs
        Additional parameters

    Returns
    -------
    Tuple[csr_matrix, np.ndarray]
        Stiffness matrix and load vector

    Examples
    --------
    >>> from src.utils.mesh import create_1d_mesh, SimpleFunctionSpace
    >>> mesh = create_1d_mesh(0, 1, 10)
    >>> V = SimpleFunctionSpace(mesh, "P1")
    >>> K, b = basic_laplacian(V, diffusion_coeff=2.0, source=lambda x: x**2)
    """
    params = {"diffusion_coeff": diffusion_coeff, **kwargs}
    if source is not None:
        params["source"] = source

    op = BasicLaplacianOperator(**kwargs)
    return op.assemble_system(function_space, params)