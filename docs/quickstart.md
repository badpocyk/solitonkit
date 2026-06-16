# Quickstart

This guide assumes the Python package is installed in editable mode:

```powershell
python -m pip install -e .
```

## Generate A Skyrmion

```python
import solitonkit as sk

field = sk.make_skyrmion_field(
    128,
    128,
    spacing=0.25,
    radius=5.0,
    charge=1,
    boundary="dirichlet",
)

print(sk.topological_charge(field))
```

`boundary="dirichlet"` pins the edge to the vacuum `n=(0,0,1)`, which is the
recommended starting point for isolated finite-grid Skyrmions.

## Inspect Energy Terms

```python
terms = sk.baby_skyrme_energy_terms(
    field,
    kappa=1.0,
    mass=1.0,
    dmi=0.1,
)

print(terms)
```

The returned dictionary contains `sigma`, `skyrme`, `potential`, `dmi`, and
`total`.

## Relax The Field

```python
relaxed, records = sk.run_baby_skyrme_lbfgs(
    field,
    kappa=1.0,
    mass=1.0,
    dmi=0.1,
    steps=100,
    record_every=10,
)

for record in records:
    print(record.step, record.energy, record.topological_charge)
```

For a conservative baseline, use `run_baby_skyrme_gradient_flow`. For faster
experiments, try `run_baby_skyrme_lbfgs` or
`run_baby_skyrme_barzilai_borwein`. For topological sectors, track both energy
and `topological_charge`; overly aggressive settings can exploit finite-lattice
topology changes.

## Save And Plot

```python
sk.save_field_npz(relaxed, "relaxed.npz")
sk.save_skyrmion_diagnostics(relaxed, "relaxed.png", spacing=0.25)
```

## Animate A Relaxation

```python
relaxed, snapshots = sk.run_baby_skyrme_gradient_flow_snapshots(
    field,
    kappa=1.0,
    mass=1.0,
    dmi=0.1,
    step_size=1e-4,
    steps=200,
    frame_every=10,
)

sk.save_flow_animation(snapshots, "gradient_flow.gif", fps=8)
```
