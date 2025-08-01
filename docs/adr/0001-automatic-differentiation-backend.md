# ADR-0001: Automatic Differentiation Backend Selection

## Status
Accepted

## Context
DiffFE-Physics-Lab requires automatic differentiation (AD) capabilities to enable gradient-based optimization through finite element operations. The choice of AD backend significantly impacts:
- Performance characteristics
- Memory usage patterns
- Integration complexity with existing FEM infrastructure
- Developer experience and ecosystem compatibility
- Long-term maintainability

## Decision
We will use **JAX as the primary AD backend** with **PyTorch as a secondary supported backend**.

## Rationale

### JAX Selection Rationale
1. **Functional Programming Model**: JAX's functional approach aligns well with mathematical operations in FEM
2. **XLA Compilation**: Provides aggressive optimization for numerical computations
3. **Flexible Transformations**: `jit`, `grad`, `vmap`, `pmap` provide powerful abstractions
4. **NumPy Compatibility**: Seamless integration with existing numerical code
5. **Research Adoption**: Strong adoption in computational physics research community

### PyTorch as Secondary Backend
1. **Deep Learning Integration**: Easier integration with existing ML workflows
2. **Industry Adoption**: Broader industrial usage and community support
3. **Dynamic Computation**: Better support for dynamic computational graphs
4. **Ecosystem**: Rich ecosystem of tools and extensions

### Alternatives Considered
- **TensorFlow**: Rejected due to complexity and declining research adoption
- **Custom C++ AD**: Rejected due to development overhead and maintenance burden
- **Enzyme/LLVM**: Considered but deemed too experimental for production use

## Consequences

### Positive
- High-performance gradient computation through XLA
- Clean integration with numerical computing workflows
- Strong research community support and examples
- Flexible backend selection based on use case requirements
- Future-proof architecture with emerging JAX ecosystem

### Negative
- Additional complexity from supporting multiple backends
- JAX learning curve for developers familiar with PyTorch
- Potential performance differences between backends
- Need to maintain parallel implementations for some features

### Neutral
- Backend abstraction layer required for clean API
- Testing complexity increases with multiple backends
- Documentation must cover both backends

## Implementation

### Timeline
- **Phase 1** (Weeks 1-2): JAX backend with core operators
- **Phase 2** (Weeks 3-4): PyTorch backend implementation
- **Phase 3** (Weeks 5-6): Backend abstraction layer and unified API
- **Phase 4** (Weeks 7-8): Performance optimization and benchmarking

### Architecture
```python
class ADBackend:
    def __init__(self, backend_type: str):
        self.backend = self._load_backend(backend_type)
    
    def grad(self, func, has_aux=False):
        return self.backend.grad(func, has_aux=has_aux)
    
    def jacobian(self, func):
        return self.backend.jacobian(func)
```

### Backend Detection
- Automatic backend detection based on input tensor types
- Explicit backend selection through environment variables
- Runtime backend switching for performance comparisons

## Compliance
- All new operators must implement both JAX and PyTorch backends
- Performance benchmarks must be run against both backends
- Documentation examples should demonstrate both backends
- CI/CD pipeline tests both backends in parallel

## Notes
- JAX version requirements: ≥0.4.25 for stable GPU support
- PyTorch version requirements: ≥2.4 for consistent autograd behavior
- Consider Enzyme integration in future ADR for native Firedrake AD support

---

*Last updated: 2025-08-01*
*Next review: 2025-11-01*