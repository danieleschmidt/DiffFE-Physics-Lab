# Contributing to DiffFE-Physics-Lab

We welcome contributions to DiffFE-Physics-Lab! This document provides guidelines for contributing to the project.

## Quick Start for Contributors

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Make your changes
5. Run tests to ensure everything works
6. Submit a pull request

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- Git
- PETSc 3.19+ (with CUDA support optional)
- Firedrake (latest)

### Installation

```bash
# Install Firedrake first
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install

# Activate Firedrake environment
source firedrake/bin/activate

# Clone your fork
git clone https://github.com/YOUR_USERNAME/DiffFE-Physics-Lab.git
cd DiffFE-Physics-Lab

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

## Types of Contributions

### Bug Reports

When filing a bug report, please include:
- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Environment details (Python version, OS, etc.)
- Minimal code example that demonstrates the issue
- Any relevant error messages or logs

### Feature Requests

For new features, please provide:
- A clear description of the proposed feature
- Use cases and motivation
- Examples of how the feature would be used
- Any relevant academic references or papers

### Code Contributions

#### Adding New Physics Operators

1. Create operator class inheriting from `BaseOperator`
2. Implement `forward_assembly` and `adjoint_assembly` methods
3. Add comprehensive unit tests
4. Include convergence studies
5. Update documentation with examples
6. Add entry to operator registry

```python
from diffhe.operators import BaseOperator, register_operator

@register_operator("my_operator")
class MyOperator(BaseOperator):
    def forward_assembly(self, trial, test, params):
        # Implementation here
        pass
    
    def adjoint_assembly(self, grad_output):
        # Adjoint implementation here
        pass
```

#### Adding Backend Support

1. Implement backend-specific operations in appropriate module
2. Ensure feature parity with existing backends
3. Add backend-specific tests
4. Update backend abstraction layer
5. Benchmark performance against other backends

## Code Standards

### Python Code Style

- Follow PEP 8 with 88-character line limit
- Use type hints for all public functions
- Write docstrings in NumPy format
- Use meaningful variable and function names

```python
def solve_linear_system(
    matrix: Union[jax.Array, torch.Tensor],
    rhs: Union[jax.Array, torch.Tensor],
    solver_type: str = "direct"
) -> Union[jax.Array, torch.Tensor]:
    """Solve linear system Ax = b.
    
    Parameters
    ----------
    matrix : jax.Array or torch.Tensor
        Coefficient matrix A
    rhs : jax.Array or torch.Tensor
        Right-hand side vector b
    solver_type : str, optional
        Type of solver to use, by default "direct"
    
    Returns
    -------
    Union[jax.Array, torch.Tensor]
        Solution vector x
    """
    # Implementation here
    pass
```

### Testing Requirements

- All new code must have tests with >90% coverage
- Include unit tests for individual functions
- Add integration tests for complete workflows
- Include convergence studies for numerical methods
- Test both JAX and PyTorch backends where applicable

```python
import pytest
import numpy as np
from diffhe.operators import laplacian_operator

def test_laplacian_convergence():
    """Test convergence rate of Laplacian operator."""
    errors = []
    h_values = []
    
    for refinement in range(3, 7):
        mesh_size = 2**refinement
        error = compute_error_with_mesh_size(mesh_size)
        errors.append(error)
        h_values.append(1.0 / mesh_size)
    
    # Check second-order convergence
    rates = np.polyfit(np.log(h_values), np.log(errors), 1)[0]
    assert rates > 1.8, f"Convergence rate {rates} too low"
```

### Documentation

- All public APIs must have docstrings
- Include examples in docstrings
- Update user guide for new features
- Add relevant references to academic literature
- Maintain changelog for releases

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/convergence/

# Run with coverage
pytest tests/ --cov=diffhe --cov-report=html

# Run performance benchmarks
python benchmarks/run_benchmarks.py
```

### Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test complete workflows
- **Convergence tests**: Verify numerical accuracy
- **Performance tests**: Benchmark computational performance
- **GPU tests**: Test CUDA kernel implementations

## Pull Request Process

1. **Create feature branch**: `git checkout -b feature/my-new-feature`
2. **Make changes**: Follow coding standards and add tests
3. **Run tests**: Ensure all tests pass locally
4. **Update documentation**: Update relevant docs and examples
5. **Commit changes**: Use conventional commit messages
6. **Push branch**: `git push origin feature/my-new-feature`
7. **Create PR**: Submit pull request with detailed description

### PR Requirements

- [ ] All tests pass
- [ ] Code coverage maintained or increased
- [ ] Documentation updated
- [ ] Changelog entry added
- [ ] No merge conflicts
- [ ] PR description explains changes and motivation

### Commit Message Format

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`

Examples:
- `feat(operators): add hyperelastic material operator`
- `fix(backends): resolve JAX gradient computation bug`
- `docs(examples): add topology optimization tutorial`

## Performance Considerations

### Benchmarking

- All new operators should include performance benchmarks
- Compare against analytical solutions where possible
- Test scaling behavior for different problem sizes
- Measure memory usage and GPU utilization

### Optimization Guidelines

- Use JAX transformations (`jit`, `vmap`) effectively
- Minimize data transfers between CPU and GPU
- Leverage sparse matrix operations where appropriate
- Profile code to identify bottlenecks

## Scientific Validation

### Method Verification

- Implement method of manufactured solutions (MMS)
- Compare against analytical benchmarks
- Validate against published results
- Include convergence studies

### Documentation Requirements

- Include mathematical formulation
- Reference relevant papers and books
- Provide implementation details
- Document assumptions and limitations

## Community Guidelines

### Communication

- Be respectful and constructive in discussions
- Follow the Code of Conduct at all times
- Use GitHub issues for bug reports and feature requests
- Use discussions for questions and general conversations

### Getting Help

- Check existing documentation and examples first
- Search GitHub issues for similar problems
- Ask questions in GitHub discussions
- Join community chat (details in README)

## Recognition

We value all contributions to DiffFE-Physics-Lab:

- Contributors are acknowledged in release notes
- Significant contributions may lead to co-authorship opportunities
- Outstanding contributors may be invited to join the core team
- We maintain a contributors page recognizing all contributions

## License

By contributing to DiffFE-Physics-Lab, you agree that your contributions will be licensed under the same license as the project (BSD 3-Clause).

---

Thank you for contributing to DiffFE-Physics-Lab! Your efforts help advance the intersection of computational physics and machine learning.
