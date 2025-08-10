"""Field definitions for DiffFE problems."""

from typing import Callable, Any, Optional, Dict, Union
import numpy as np


class Field:
    """Represents a field defined over a domain.
    
    Parameters
    ----------
    name : str
        Field identifier
    values : np.ndarray or callable
        Field values or function defining the field
    domain : Any
        Domain where field is defined
    """
    
    def __init__(self, name: str, values: Union[np.ndarray, Callable] = None, domain: Any = None):
        self.name = name
        self.values = values
        self.domain = domain
        self.is_parametric = False
        
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate field at given points."""
        if callable(self.values):
            return self.values(points)
        elif isinstance(self.values, np.ndarray):
            # Simple interpolation for now
            return np.full(points.shape[0], self.values[0] if self.values.size > 0 else 0.0)
        else:
            return np.zeros(points.shape[0])
    
    def norm(self, norm_type: str = 'L2') -> float:
        """Compute field norm."""
        if isinstance(self.values, np.ndarray):
            if norm_type == 'L2':
                return np.linalg.norm(self.values)
            elif norm_type == 'max':
                return np.max(np.abs(self.values))
        return 0.0


class ParametricField(Field):
    """Parametric field that depends on optimization parameters.
    
    Used for inverse problems where field properties are unknown
    and need to be determined through optimization.
    
    Parameters
    ----------
    name : str
        Field identifier
    parameter_function : callable
        Function mapping parameters to field values
    num_parameters : int
        Number of optimization parameters
    """
    
    def __init__(self, name: str, parameter_function: Callable, num_parameters: int):
        super().__init__(name)
        self.parameter_function = parameter_function
        self.num_parameters = num_parameters
        self.is_parametric = True
        self.current_parameters = np.zeros(num_parameters)
    
    def update_parameters(self, parameters: np.ndarray):
        """Update field with new parameter values."""
        if parameters.size != self.num_parameters:
            raise ValueError(f"Expected {self.num_parameters} parameters, got {parameters.size}")
        self.current_parameters = parameters
        self.values = self.parameter_function(parameters)
    
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate parametric field at given points."""
        if self.values is None:
            # Initialize with current parameters
            self.values = self.parameter_function(self.current_parameters)
        return super().evaluate(points)


class BoundaryCondition:
    """Boundary condition specification.
    
    Parameters
    ----------
    bc_type : str
        Type of boundary condition ('dirichlet', 'neumann', 'robin')
    value : float, callable, or np.ndarray
        Boundary condition value(s)
    boundary_id : int or str
        Boundary identifier
    """
    
    def __init__(self, bc_type: str, value: Union[float, Callable, np.ndarray], boundary_id: Union[int, str]):
        self.bc_type = bc_type.lower()
        self.value = value
        self.boundary_id = boundary_id
        self.is_time_dependent = False
        
        if self.bc_type not in ['dirichlet', 'neumann', 'robin', 'periodic']:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")
    
    def evaluate(self, points: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Evaluate boundary condition at given points."""
        if callable(self.value):
            if self.is_time_dependent:
                return self.value(points, time)
            else:
                return self.value(points)
        elif isinstance(self.value, (int, float)):
            return np.full(points.shape[0], self.value)
        elif isinstance(self.value, np.ndarray):
            if self.value.size == 1:
                return np.full(points.shape[0], self.value[0])
            else:
                return self.value
        else:
            return np.zeros(points.shape[0])
    
    def set_time_dependent(self, is_time_dependent: bool = True):
        """Mark boundary condition as time-dependent."""
        self.is_time_dependent = is_time_dependent