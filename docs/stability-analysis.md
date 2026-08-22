# Linear Stability Analysis

`solitonkit` can compute the lowest eigenmodes of the energy Hessian around a
scalar, XY, O(3), or 3D O(3) field. A negative eigenvalue identifies a direction
that lowers the energy to quadratic order.

```python
import solitonkit as sk

field = sk.ScalarField2D(32, 32, value=0.0)
model = sk.Phi4Model(lambda_=1.0, vacuum=1.0)

stability = sk.stability_analysis(
    field,
    model,
    modes=6,
    tolerance=1e-7,
)

print(stability.eigenvalues)
print(stability.negative_mode_count)
print(stability.stationary, stability.stable)
sk.plot_eigenmode(stability, index=0)
```

The unstable homogeneous phi4 state has a lowest eigenvalue of `-1` for the
default parameters. A relaxed local minimum should instead have no eigenvalues
below `-eigenvalue_tolerance`.

## Result diagnostics

`StabilityResult` contains:

- `eigenvalues`: lowest Ritz eigenvalues in ascending order;
- `modes`: NumPy arrays with the same spatial shape as the analyzed field;
- `residual_norms`: `||H v - lambda v||` for each returned mode;
- `gradient_norm`: norm of the active, projected model gradient;
- `stationary`: whether that norm is below `stationarity_tolerance`;
- `stable`: whether the solve converged, the point is stationary, and no
  resolved negative mode is present;
- `converged`, `iterations`, and `degrees_of_freedom`.

An eigenvalue calculation can describe local curvature at any field, but its
usual stability interpretation is valid only at a stationary solution. Always
inspect both `stationary` and `converged`.

## Matrix-free implementation

The Hessian is never assembled. Its action is evaluated as the negative
centered directional derivative of `negative_gradient()`, and the lowest modes
are found with a restarted block Davidson iteration. The projected matrices are
symmetrized before the Rayleigh-Ritz solve.

For O(3) fields, each active spin contributes two tangent degrees of freedom.
Perturbations use the sphere exponential map, gradients are parallel-transported
back to the original spin, and the reported vector modes are tangent to the
field. Pinned `fixed` and `dirichlet` samples contribute no degrees of freedom.

Important controls are:

```python
stability = sk.stability_analysis(
    field,
    model,
    modes=20,
    max_iterations=80,
    subspace_dimension=96,
    tolerance=1e-7,
    finite_difference_step=1e-5,
    stationarity_tolerance=1e-6,
    eigenvalue_tolerance=1e-8,
    seed=12345,
)
```

Increasing `subspace_dimension` usually improves clustered low modes at the
cost of memory and additional Hessian applications. If residuals stagnate,
compare nearby finite-difference steps such as `3e-6`, `1e-5`, and `3e-5`.

## C++ API

The header-only API exposes the same analysis and a reusable matrix-free
Hessian-vector product:

```cpp
#include <solitonkit/solitonkit.hpp>

using namespace solitonkit;

StabilityOptions options;
options.modes = 8;
options.tolerance = 1e-7;

const auto result = stability_analysis(field, model, options);
const auto hessian_direction = hessian_vector_product(
    field,
    model,
    direction,
    1e-5
);
```

The same Hessian-vector API drives the implemented damped Newton-Krylov solver
and the augmented pseudo-arclength corrector. See
[Stationary Solvers And Research Workflows](research-workflows.md).
