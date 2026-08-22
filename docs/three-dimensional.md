# 3D Fields and Hopfions

`Lattice3D` and `O3Field3D` extend the unit-vector field API to a regular
three-dimensional grid. Python arrays use the shape `(nz, ny, nx, 3)`.

## Rational-map initial condition

`make_hopfion_field()` compactifies physical space to `S3` and applies a
rational Hopf map. Positive windings `(p, q)` give nominal Hopf charge `p*q`:

```python
field = sk.make_hopfion_field(
    65,
    65,
    65,
    spacing=0.25,
    scale=3.0,
    winding_p=1,
    winding_q=1,
    boundary="dirichlet",
)

values = field.to_numpy()  # (65, 65, 65, 3)
sk.save_field_npz(field, "hopfion.npz")
loaded = sk.load_field_npz("hopfion.npz")
```

Dirichlet boundaries pin the field to `(0, 0, 1)`, the compactification vacuum.
Choose a box several times larger than `scale` to keep the texture away from
the pinned boundary.

## Faddeev-Skyrme energy

`HopfionModel` contains the three-dimensional sigma term, the Faddeev-Skyrme
quartic term, and an optional vacuum potential:

```python
model = sk.HopfionModel(coupling=1.0, kappa=0.25, mass=0.1)
print(model.energy_terms(field))

relaxed, records = sk.minimize(
    field,
    model,
    max_steps=100,
    step_size=1e-3,
    record_every=5,
)
```

3D minimization is memory- and compute-intensive: a `129^3` vector field alone
contains more than six million doubles. Start with `33^3` or `65^3`, verify
spacing/box-size convergence, and only then increase resolution.

The `(p, q)` value labels the analytic ansatz. For an arbitrary field, compute
the nonlocal numerical invariant and its Poisson diagnostics with:

```python
hopf = sk.hopf_charge(
    relaxed,
    tolerance=1e-8,
    return_diagnostics=True,
)
print(hopf.charge, hopf.poisson_residual, hopf.divergence_norm)
```

This reconstructs a Coulomb-gauge vector potential on the periodic numerical
box. Finite volume, a Dirichlet/periodic mismatch at the Poisson stage, and
coarse derivatives can shift the value away from an integer. Verify the charge
under both grid refinement and box enlargement.
