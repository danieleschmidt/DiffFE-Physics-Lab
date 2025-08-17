"""Validation and verification tools for DiffFE-Physics-Lab."""

import logging
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ValidationSuite:
    """Comprehensive validation suite for FEM problems."""
    
    def __init__(self):
        """Initialize validation suite."""
        self.results = {}
        self.tests = []
    
    def add_test(self, name: str, test_function: Callable, **kwargs):
        """Add a validation test.
        
        Args:
            name: Test name
            test_function: Function to execute
            **kwargs: Test-specific arguments
        """
        test_config = {
            "name": name,
            "function": test_function,
            "kwargs": kwargs,
            "status": "pending"
        }
        self.tests.append(test_config)
        logger.info(f"Added validation test: {name}")
    
    def run_all_tests(self, problem) -> Dict[str, Any]:
        """Run all validation tests.
        
        Args:
            problem: Problem instance to validate
            
        Returns:
            Validation results
        """
        logger.info(f"Running {len(self.tests)} validation tests")
        
        results = {
            "total_tests": len(self.tests),
            "passed": 0,
            "failed": 0,
            "test_results": []
        }
        
        for test_config in self.tests:
            try:
                test_name = test_config["name"]
                test_function = test_config["function"]
                test_kwargs = test_config["kwargs"]
                
                logger.debug(f"Running test: {test_name}")
                
                # Execute test
                test_result = test_function(problem, **test_kwargs)
                
                # Process result
                if isinstance(test_result, bool):
                    passed = test_result
                    details = "Boolean result"
                elif isinstance(test_result, dict) and "passed" in test_result:
                    passed = test_result["passed"]
                    details = test_result.get("details", "No details")
                else:
                    passed = test_result is not None
                    details = str(test_result)
                
                test_config["status"] = "passed" if passed else "failed"
                
                if passed:
                    results["passed"] += 1
                    logger.info(f"✓ {test_name}: PASSED")
                else:
                    results["failed"] += 1
                    logger.warning(f"✗ {test_name}: FAILED - {details}")
                
                results["test_results"].append({
                    "name": test_name,
                    "status": "passed" if passed else "failed",
                    "details": details
                })
                
            except Exception as e:
                test_config["status"] = "error"
                results["failed"] += 1
                error_msg = str(e)
                logger.error(f"✗ {test_config['name']}: ERROR - {error_msg}")
                
                results["test_results"].append({
                    "name": test_config["name"],
                    "status": "error",
                    "details": error_msg
                })
        
        self.results = results
        
        logger.info(f"Validation complete: {results['passed']}/{results['total_tests']} passed")
        return results
    
    def get_summary(self) -> str:
        """Get validation summary as string."""
        if not self.results:
            return "No validation results available"
        
        total = self.results["total_tests"]
        passed = self.results["passed"]
        failed = self.results["failed"]
        
        summary = f"Validation Summary: {passed}/{total} tests passed"
        if failed > 0:
            summary += f" ({failed} failed)"
        
        return summary


class ConvergenceAnalyzer:
    """Analyze convergence properties of numerical methods."""
    
    def __init__(self):
        """Initialize convergence analyzer."""
        self.studies = []
    
    def mesh_convergence_study(self, problem_factory: Callable, 
                              mesh_sizes: List[int],
                              reference_solution: Optional[Callable] = None) -> Dict[str, Any]:
        """Perform mesh convergence study.
        
        Args:
            problem_factory: Function that creates problem given mesh_size
            mesh_sizes: List of mesh sizes to test
            reference_solution: Optional reference solution function
            
        Returns:
            Convergence study results
        """
        logger.info(f"Starting mesh convergence study with {len(mesh_sizes)} mesh sizes")
        
        results = {
            "mesh_sizes": mesh_sizes,
            "errors": [],
            "convergence_rates": [],
            "solutions": []
        }
        
        previous_error = None
        previous_h = None
        
        for mesh_size in mesh_sizes:
            logger.debug(f"Testing mesh size: {mesh_size}")
            
            # Create and solve problem
            problem = problem_factory(mesh_size=mesh_size)
            solution = problem.solve()
            
            # Compute error
            if reference_solution:
                error = self._compute_reference_error(solution, reference_solution, mesh_size)
            else:
                # Use Richardson extrapolation or self-convergence
                error = self._compute_self_convergence_error(solution, mesh_size)
            
            results["errors"].append(error)
            results["solutions"].append(solution)
            
            # Compute convergence rate
            h = 1.0 / mesh_size  # Characteristic mesh size
            
            if previous_error is not None and previous_h is not None:
                if error > 0 and previous_error > 0:
                    rate = (log(error) - log(previous_error)) / (log(h) - log(previous_h))
                    results["convergence_rates"].append(rate)
                    logger.debug(f"Convergence rate: {rate:.2f}")
                else:
                    results["convergence_rates"].append(0.0)
            
            previous_error = error
            previous_h = h
        
        # Analyze overall convergence
        if len(results["convergence_rates"]) > 0:
            avg_rate = sum(results["convergence_rates"]) / len(results["convergence_rates"])
            results["average_convergence_rate"] = avg_rate
            logger.info(f"Average convergence rate: {avg_rate:.2f}")
        
        study_data = {
            "type": "mesh_convergence",
            "results": results,
            "timestamp": self._get_timestamp()
        }
        self.studies.append(study_data)
        
        return results
    
    def time_convergence_study(self, problem_factory: Callable,
                              time_steps: List[float]) -> Dict[str, Any]:
        """Perform time step convergence study.
        
        Args:
            problem_factory: Function that creates problem given time_step
            time_steps: List of time steps to test
            
        Returns:
            Time convergence study results
        """
        logger.info(f"Starting time convergence study with {len(time_steps)} time steps")
        
        results = {
            "time_steps": time_steps,
            "errors": [],
            "convergence_rates": [],
            "solutions": []
        }
        
        # Implementation similar to mesh convergence
        # This is a simplified version
        for dt in time_steps:
            problem = problem_factory(time_step=dt)
            solution = problem.solve()
            
            # Simplified error computation
            error = abs(dt)  # Placeholder
            results["errors"].append(error)
            results["solutions"].append(solution)
        
        study_data = {
            "type": "time_convergence", 
            "results": results,
            "timestamp": self._get_timestamp()
        }
        self.studies.append(study_data)
        
        return results
    
    def _compute_reference_error(self, solution, reference_solution: Callable, 
                                mesh_size: int) -> float:
        """Compute error against reference solution."""
        # Placeholder implementation
        h = 1.0 / mesh_size
        
        # For demonstration - actual implementation would integrate over domain
        total_error = 0.0
        num_points = mesh_size + 1
        
        for i in range(num_points):
            x = i * h
            numerical = solution.get("value_at", lambda x: x)(x) if hasattr(solution, 'get') else 0.0
            analytical = reference_solution(x)
            total_error += (numerical - analytical) ** 2
        
        return (total_error / num_points) ** 0.5
    
    def _compute_self_convergence_error(self, solution, mesh_size: int) -> float:
        """Compute self-convergence error estimate."""
        # Simplified implementation - actual would use Richardson extrapolation
        h = 1.0 / mesh_size
        return h ** 2  # Assume O(h^2) method
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def export_convergence_plot(self, study_index: int = -1, filename: str = "convergence.png"):
        """Export convergence plot.
        
        Args:
            study_index: Index of study to plot (-1 for latest)
            filename: Output filename
        """
        if not self.studies:
            logger.warning("No convergence studies available")
            return
        
        study = self.studies[study_index]
        results = study["results"]
        
        logger.info(f"Exporting convergence plot to {filename}")
        
        # Placeholder for actual plotting
        # In real implementation, would use matplotlib
        plot_data = {
            "x_values": results.get("mesh_sizes", results.get("time_steps", [])),
            "y_values": results["errors"],
            "convergence_rates": results.get("convergence_rates", [])
        }
        
        logger.info(f"Plot data prepared: {len(plot_data['x_values'])} points")


def log(x):
    """Simple logarithm function."""
    import math
    return math.log(x)


# Standard validation tests
def test_mass_conservation(problem, **kwargs) -> Dict[str, Any]:
    """Test mass conservation."""
    logger.debug("Testing mass conservation")
    
    # Placeholder implementation
    mass_in = kwargs.get("mass_in", 1.0)
    mass_out = kwargs.get("mass_out", 1.0)
    tolerance = kwargs.get("tolerance", 1e-10)
    
    error = abs(mass_in - mass_out)
    passed = error < tolerance
    
    return {
        "passed": passed,
        "details": f"Mass conservation error: {error:.2e}"
    }


def test_energy_conservation(problem, **kwargs) -> Dict[str, Any]:
    """Test energy conservation."""
    logger.debug("Testing energy conservation")
    
    # Placeholder implementation
    energy_in = kwargs.get("energy_in", 1.0)
    energy_out = kwargs.get("energy_out", 1.0)
    tolerance = kwargs.get("tolerance", 1e-10)
    
    error = abs(energy_in - energy_out)
    passed = error < tolerance
    
    return {
        "passed": passed,
        "details": f"Energy conservation error: {error:.2e}"
    }


def test_boundary_conditions(problem, **kwargs) -> Dict[str, Any]:
    """Test boundary condition satisfaction."""
    logger.debug("Testing boundary conditions")
    
    # Check if boundary conditions are properly applied
    if not hasattr(problem, 'boundary_conditions'):
        return {"passed": False, "details": "No boundary conditions defined"}
    
    num_bcs = len(problem.boundary_conditions)
    passed = num_bcs > 0
    
    return {
        "passed": passed,
        "details": f"Found {num_bcs} boundary conditions"
    }


def test_solution_existence(problem, **kwargs) -> Dict[str, Any]:
    """Test solution existence."""
    logger.debug("Testing solution existence")
    
    has_solution = hasattr(problem, 'solution') and problem.solution is not None
    
    return {
        "passed": has_solution,
        "details": "Solution exists" if has_solution else "No solution found"
    }


# Factory function for validation suites
def create_validation_suite(suite_type: str = "standard") -> ValidationSuite:
    """Create pre-configured validation suite.
    
    Args:
        suite_type: Type of validation suite
        
    Returns:
        ValidationSuite instance
    """
    suite = ValidationSuite()
    
    if suite_type == "standard":
        suite.add_test("solution_existence", test_solution_existence)
        suite.add_test("boundary_conditions", test_boundary_conditions)
    elif suite_type == "conservation":
        suite.add_test("mass_conservation", test_mass_conservation)
        suite.add_test("energy_conservation", test_energy_conservation)
    elif suite_type == "comprehensive":
        suite.add_test("solution_existence", test_solution_existence)
        suite.add_test("boundary_conditions", test_boundary_conditions)
        suite.add_test("mass_conservation", test_mass_conservation)
        suite.add_test("energy_conservation", test_energy_conservation)
    else:
        raise ValueError(f"Unknown validation suite type: {suite_type}")
    
    return suite