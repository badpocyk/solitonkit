# Models, Boundaries, and Solvers

## Boundary conditions

The 2D and 3D lattices support `periodic`, `fixed`, `neumann`, and
`dirichlet` conditions independently on every axis. `fixed` preserves the
initial boundary samples; `dirichlet` pins them to the field's configured
boundary value; `neumann` uses a reflected zero-normal-derivative stencil.

In Python, O(3) fields with mixed boundaries are created with:

```python
field = sk.O3Field.with_boundaries(
    128,
    96,
    dx=0.25,
    dy=0.25,
    boundary_x="periodic",
    boundary_y="dirichlet",
)
```

Scalar and XY fields accept `boundary_x` and `boundary_y` directly. C++ users
can pass `BoundaryConditions2D` or `BoundaryConditions3D` to a lattice.

## Differential operators

The shared operator layer provides first and second derivatives, gradient,
Laplacian, divergence, and curl. It honors the lattice spacing and the boundary
mode of each axis:

```python
dx_phi = sk.derivative_x(field, i, j)
lap_phi = sk.laplacian(field, i, j)
curl_m = sk.curl(magnetization, i, j)
```

C++ users include `solitonkit/operators/DifferentialOperators.hpp` and call the
same operations in the `solitonkit::differential` namespace.

## Common model interface

Every new model exposes `name`, `dimensions`, `field_kind`, and `energy(field)`.
Differentiable C++ models additionally implement `negative_gradient(field)`.
The concrete field/model pairs are:

| Field | Models |
| --- | --- |
| `ScalarField2D` | `Phi4Model`, `SineGordonModel` |
| `XYField` | `XYModel` |
| `O3Field` | `O3SigmaModel`, `BabySkyrmeModel`, `MicromagneticModel` |
| `O3Field3D` | `HopfionModel` |

`minimize()` uses tangent-space updates for unit-vector fields and a
backtracking line search that rejects energy-increasing steps. `solve()` uses a
fixed explicit time step for the corresponding dissipative field equation.

```python
field = sk.ScalarField2D(
    96,
    64,
    dx=0.2,
    dy=0.2,
    value=1.0,
    dirichlet_value=1.0,
    boundary_x="dirichlet",
    boundary_y="dirichlet",
)
field.set(48, 32, 0.0)

model = sk.Phi4Model(lambda_=1.0, vacuum=1.0)
relaxed, records = sk.minimize(
    field,
    model,
    max_steps=500,
    step_size=0.05,
    tolerance=1e-8,
    record_every=10,
)
```

Each `SolverRecord` contains `step`, `time`, `energy`, `gradient_norm`, and
`converged`. `minimize_inplace()` and `solve_inplace()` avoid copying.

Once a stationary state has been found, `stability_analysis()` computes its
lowest Hessian eigenpairs without assembling the Hessian. See
[Linear Stability Analysis](stability-analysis.md) for convergence diagnostics,
O(3) tangent modes, and plotting.

The equivalent C++ entry point is header-only:

```cpp
#include <solitonkit/solitonkit.hpp>

using namespace solitonkit;

Lattice2D lattice{64, 64, 0.25, 0.25, BoundaryCondition::Dirichlet};
ScalarField2D field{lattice, 1.0, 1.0};
Phi4Model model{1.0, 1.0};

MinimizeOptions options;
options.max_steps = 500;
options.step_size = 0.05;

const auto records = minimize(field, model, options);
```

## XY, phi4, and Sine-Gordon

`XYModel` uses the compact nearest-neighbor Hamiltonian
`J (1 - cos(theta_j - theta_i))` and an optional aligning field. This avoids a
branch cut when angles cross `2 pi`.

`Phi4Model` implements
`1/2 |grad phi|^2 + lambda/4 (phi^2 - v^2)^2`.
`SineGordonModel` implements
`1/2 |grad phi|^2 + m^2 (1 - cos(beta phi))/beta^2`.

Explicit `solve()` steps must obey the usual diffusion stability restriction;
when in doubt, start below `min(dx, dy)^2 / 4`. `minimize()` is safer for static
states because it backtracks automatically.

## Micromagnetics and LLG

`MicromagneticModel` combines exchange, bulk or interfacial DMI, uniaxial
anisotropy, and a Zeeman field:

```python
model = sk.MicromagneticModel(
    exchange=1.0,
    dmi=0.2,
    anisotropy=0.1,
    applied_field=sk.Vec3(0.0, 0.0, 0.05),
    easy_axis=sk.Vec3(0.0, 0.0, 1.0),
    dmi_type=sk.DMIType.Interfacial,
)

evolved, records = sk.run_llg(
    field,
    model,
    time_step=1e-3,
    damping=0.2,
    gyromagnetic_ratio=1.0,
    steps=1000,
    record_every=20,
)
```

The LLG integrator uses a normalized Heun predictor-corrector step. It preserves
unit magnetization to floating-point accuracy. Small time steps are still
required when exchange or DMI is stiff.
