# Stationary Solvers And Research Workflows

This layer turns the field models into a small reproducible research platform:
matrix-free stationary solves, continuation, topology diagnostics, parallel
sweeps, checkpointing, and benchmarks all use the same public field objects.

## Newton-Krylov and GMRES

`solve_stationary()` solves the Euler-Lagrange equation rather than integrating
a long artificial-time trajectory. It applies the Hessian matrix-free, solves
each Newton system with restarted right-preconditioned GMRES, limits large
steps with a trust radius, and backtracks on the residual norm.

```python
import solitonkit as sk

field = sk.ScalarField2D(64, 64, value=0.8)
model = sk.Phi4Model(lambda_=1.0, vacuum=1.0)

stationary, history = sk.solve_stationary(
    field,
    model,
    tolerance=1e-9,
    trust_radius=10.0,
    preconditioner_probes=4,
    gmres_restart=30,
    gmres_max_iterations=200,
    gmres_tolerance=1e-9,
)
```

The diagonal preconditioner is estimated with randomized Hutchinson probes,
so neither a sparse Hessian nor model-specific matrix assembly is required.
Every `StationaryRecord` reports nonlinear and linear residuals, damping, GMRES
iterations, and convergence. `solve_stationary_inplace()` mutates its field.

The standalone `gmres(apply, b, ...)` accepts a Python matrix-vector callback:

```python
linear = sk.gmres(lambda x: A @ x, b, tolerance=1e-10)
```

For O(3) fields the Newton coordinates are tangent vectors, fixed and
Dirichlet sites are excluded, and accepted steps use the sphere exponential
map.

## Pseudo-arclength continuation

```python
branch = sk.continue_solution(
    stationary,
    lambda vacuum: sk.Phi4Model(lambda_=1.0, vacuum=vacuum),
    start=1.0,
    stop=1.5,
    step=0.05,
    parameter_name="vacuum",
    analyze_stability=True,
)
```

The first two points are bootstrapped with stationary solves. Later points use
a secant predictor and an augmented field-plus-parameter Newton corrector.
Arc steps grow after easy corrections and shrink on failed ones. The lowest
Hessian eigenvalue is tracked by default; a sign change or near-zero value
marks adjacent `BranchPoint` objects as `bifurcation_candidate`.

```python
branch.plot(y="energy")
branch.plot(y="lowest_eigenvalue")
print(branch.reached_stop, branch.converged)
```

`model_factory(parameter)` must return the same supported concrete model type
at every call. `stop` specifies the desired direction and stopping parameter;
a fold can temporarily move the branch parameter away from it.

## Topological observables

For 2D O(3) fields, `degree(field)` uses the geometric solid-angle lattice
formula. For XY fields, circulation is accumulated around every plaquette:

```python
charge = sk.degree(o3_field)
defects = sk.detect_defects(xy_field)
net_vorticity = sk.vortex_number(xy_field)
```

Each vortex defect contains its physical `x`, `y`, integer `charge`, and
plaquette indices `i`, `j`. On a periodic torus the net charge is normally
zero even when vortex-antivortex pairs are present, so inspect
`detect_defects()` as well as `vortex_number()`.

The numerical Hopf invariant reconstructs a Coulomb-gauge vector potential by
solving a periodic Poisson equation and integrates its helicity:

```python
diagnostics = sk.hopf_charge(
    hopfion,
    tolerance=1e-8,
    return_diagnostics=True,
)
print(diagnostics.charge)
print(diagnostics.poisson_residual, diagnostics.divergence_norm)
```

Finite volume, boundary mismatch, and lattice spacing make the result only
approximately integer. Convergence should be checked by enlarging the box,
refining the grid, and inspecting both diagnostics.

## Parallel sweeps and phase diagrams

```python
diagram = sk.phase_diagram(
    initial_field,
    lambda exchange, dmi: sk.MicromagneticModel(
        exchange=exchange,
        dmi=dmi,
    ),
    {
        "exchange": [0.8, 1.0, 1.2],
        "dmi": [0.0, 0.1, 0.2],
    },
    solver="stationary",
    solver_kwargs={"tolerance": 1e-8},
    workers=4,
    keep_fields=True,
)

diagram.plot(observable="phase")
diagram.plot(observable="energy")
```

`parameter_sweep()` accepts any number of named axes; `phase_diagram()` requires
exactly two. Grid points are independent and can run in a thread pool because
the C++ stationary solve releases the Python GIL. Supply a custom
`classifier(field)` when the built-in uniform/modulated/topological labels are
not appropriate for the model.

## HDF5 runs and checkpoints

```python
sk.save_run(branch, "branch.h5", metadata={"experiment": "vacuum scan"})
stored = sk.load_run("branch.h5")
last_field = stored.latest_field

sk.save_checkpoint(
    last_field,
    "checkpoint.h5",
    step=250,
    parameters={"vacuum": 1.5},
)
checkpoint = sk.resume("checkpoint.h5")
field = checkpoint.latest_field
```

The versioned HDF5 schema stores arrays, per-axis spacing and boundaries,
solver/branch/sweep records, and JSON metadata. `h5py` is a standard package
dependency. NPZ remains the lightweight single-field interchange format.

## Benchmarks

```python
result = sk.run_benchmarks(sizes=(32, 64, 128), repeats=5)
print(result.format_table())
```

The default suite measures energy, topological charge, and one gradient-flow
step. Add `include_stability=True` for the lowest-mode solver. The equivalent
CLI is:

```powershell
solitonkit benchmark --sizes 32 64 128 --repeats 5 --stability
```

Report platform, compiler, build type, OpenMP status, sizes, and repeat count
with benchmark results. Timings are medians and are intended for regression
tracking, not cross-machine claims without that context.

## C++ headers

The public umbrella header includes all research components:

```cpp
#include <solitonkit/solitonkit.hpp>

const auto history = solve_stationary(field, model, stationary_options);
const auto branch = continue_solution(field, model_factory, continuation_options);
const double q = topology::degree(field);
const auto hopf = topology::hopf_charge(field3d);
```

Direct headers are `analysis/LinearStability.hpp`,
`analysis/Continuation.hpp`, `solvers/StationarySolvers.hpp`, and
`topology/Topology.hpp`.
