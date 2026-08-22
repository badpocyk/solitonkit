# Python API Overview

Import the public API as:

```python
import solitonkit as sk
```

## Common model and solver API

All new concrete models derive from `Model` and expose `name`, `dimensions`,
`field_kind`, and `energy(field)`. Supported field/model combinations are:

```python
scalar = sk.ScalarField2D(64, 64)
phi4 = sk.Phi4Model(lambda_=1.0, vacuum=1.0)
sine_gordon = sk.SineGordonModel(mass=1.0, beta=1.0)

angles = sk.XYField(64, 64)
xy = sk.XYModel(coupling=1.0, field_strength=0.0)

spins = sk.O3Field(64, 64)
sigma = sk.O3SigmaModel(coupling=1.0)
baby = sk.BabySkyrmeModel(kappa=1.0, mass=1.0, dmi=0.1)
```

Use the same solver calls for each compatible pair:

```python
relaxed, records = sk.minimize(
    scalar,
    phi4,
    max_steps=500,
    step_size=0.05,
    tolerance=1e-8,
    record_every=10,
)

evolved, records = sk.solve(
    angles,
    xy,
    steps=500,
    time_step=1e-3,
    record_every=10,
)
```

The `_inplace` variants mutate their input and return only records.

For a root of the Euler-Lagrange equation, use the matrix-free Newton-Krylov
solver instead of long dissipative evolution:

```python
stationary, records = sk.solve_stationary(
    scalar,
    phi4,
    tolerance=1e-9,
    gmres_tolerance=1e-9,
)
```

`sk.gmres(apply, b)` is also exposed as a standalone restarted linear solver.

## Linear stability

```python
stability = sk.stability_analysis(
    relaxed,
    model,
    modes=12,
    tolerance=1e-7,
)

print(stability.eigenvalues)
print(stability.residual_norms)
print(stability.stationary, stability.stable)
sk.plot_eigenmode(stability, index=0)
```

`StabilityResult.modes` contains scalar `(ny, nx)`, vector `(ny, nx, 3)`, or
3D vector `(nz, ny, nx, 3)` arrays. O(3) modes are tangent perturbations. See
[Linear Stability Analysis](stability-analysis.md) for all solver controls and
interpretation notes.

## Boundary conditions and operators

`ScalarField2D` and `XYField` accept independent `boundary_x` and `boundary_y`
strings. Use `O3Field.with_boundaries()` for a mixed-boundary vector field.

```python
value = sk.derivative_x(field, i, j)
value = sk.derivative_y(field, i, j)
value = sk.derivative_z(field3d, i, j, k)
value = sk.laplacian(field, i, j)
value = sk.gradient(scalar, i, j)
value = sk.curl(spins, i, j)
```

## Continuation and topology

```python
branch = sk.continue_solution(
    stationary,
    lambda vacuum: sk.Phi4Model(lambda_=1.0, vacuum=vacuum),
    start=1.0,
    stop=1.5,
    step=0.05,
    parameter_name="vacuum",
)

print(branch.energies)
print(branch.lowest_eigenvalues)
print(sk.degree(o3_field))
print(sk.detect_defects(xy_field))
print(sk.hopf_charge(hopfion))
```

Continuation uses an augmented pseudo-arclength corrector and can traverse
folds. See [Stationary Solvers And Research Workflows](research-workflows.md)
for controls and numerical diagnostics.

## Field Creation

```python
field = sk.make_skyrmion_field(
    nx=128,
    ny=128,
    spacing=0.25,
    radius=5.0,
    charge=1,
    boundary="dirichlet",
)

multi = sk.make_multi_skyrmion_field(
    128,
    128,
    spacing=0.25,
    specs=[
        sk.SkyrmionSpec(x=-4.0, y=0.0, charge=1),
        sk.SkyrmionSpec(x=4.0, y=0.0, charge=-1),
    ],
)
```

## Observables

```python
energy = sk.total_energy(field)
charge = sk.topological_charge(field)
e_density = sk.energy_density(field)
q_density = sk.topological_density(field)
```

For Baby Skyrme terms:

```python
terms = sk.baby_skyrme_energy_terms(field, kappa=1.0, mass=1.0, dmi=0.1)
```

## Relaxation

```python
relaxed, records = sk.run_baby_skyrme_gradient_flow(field)
relaxed, records = sk.run_baby_skyrme_riemannian_gradient_descent(field)
relaxed, records = sk.run_baby_skyrme_barzilai_borwein(field)
relaxed, records = sk.run_baby_skyrme_lbfgs(field)
relaxed, records = sk.run_baby_skyrme_semi_implicit_flow(field)
```

Each record exposes:

```python
record.step
record.energy
record.topological_charge
```

## Dynamics

```python
evolved, records = sk.run_landau_lifshitz(
    field,
    kappa=1.0,
    mass=1.0,
    dmi=0.1,
    dt=1e-4,
    steps=100,
    damping=0.3,
)
```

For a full micromagnetic energy and Gilbert dynamics:

```python
model = sk.MicromagneticModel(
    exchange=1.0,
    dmi=0.2,
    anisotropy=0.1,
    applied_field=sk.Vec3(0.0, 0.0, 0.05),
    dmi_type=sk.DMIType.Interfacial,
)

evolved, records = sk.run_llg(
    field,
    model,
    time_step=1e-3,
    damping=0.2,
    steps=1000,
)
```

## 3D fields and Hopfions

```python
hopfion = sk.make_hopfion_field(
    65,
    65,
    65,
    spacing=0.25,
    scale=3.0,
    winding_p=1,
    winding_q=1,
)

model = sk.HopfionModel(coupling=1.0, kappa=0.25, mass=0.1)
print(model.energy_terms(hopfion))
relaxed, records = sk.minimize(hopfion, model, step_size=1e-3)
```

`O3Field3D.to_numpy()` returns `(nz, ny, nx, 3)`.

## I/O

```python
sk.save_field_npz(field, "field.npz", metadata={"model": "baby-skyrme"})
loaded, metadata = sk.load_field_npz("field.npz", return_metadata=True)
```

Format version 3 accepts both `(ny, nx, 3)` and `(nz, ny, nx, 3)` fields and
preserves each axis boundary separately. Versions 1 and 2 remain readable.

Use the versioned HDF5 container for a full solver, branch, or sweep result:

```python
sk.save_run(branch, "branch.h5", metadata={"model": "phi4"})
stored = sk.load_run("branch.h5")
restart_field = stored.latest_field

sk.save_checkpoint(restart_field, "checkpoint.h5", step=100)
restart = sk.resume("checkpoint.h5")
```

## Sweeps and benchmarks

```python
diagram = sk.phase_diagram(
    initial,
    lambda lambda_, vacuum: sk.Phi4Model(
        lambda_=lambda_, vacuum=vacuum
    ),
    {"lambda_": [0.8, 1.0, 1.2], "vacuum": [0.8, 1.0, 1.2]},
    workers=4,
)
diagram.plot()

benchmarks = sk.run_benchmarks(sizes=(32, 64, 128), repeats=5)
print(benchmarks.format_table())
```

## Visualization

```python
sk.save_skyrmion_plot(field, "field.png")
sk.save_skyrmion_diagnostics(field, "diagnostics.png", spacing=0.25)
```

For animations:

```python
relaxed, snapshots = sk.run_baby_skyrme_gradient_flow_snapshots(field)
sk.save_flow_animation(snapshots, "flow.gif")
```
