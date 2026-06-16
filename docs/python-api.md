# Python API Overview

Import the public API as:

```python
import solitonkit as sk
```

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

## I/O

```python
sk.save_field_npz(field, "field.npz", metadata={"model": "baby-skyrme"})
loaded, metadata = sk.load_field_npz("field.npz", return_metadata=True)
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
