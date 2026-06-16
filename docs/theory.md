# Theory Notes

## O(3) Fields

An O(3) field assigns a unit vector

```text
n(x, y) = (n_x, n_y, n_z),     |n| = 1
```

to every lattice site. `solitonkit` stores this as an `O3Field` over a
`Lattice2D`.

## Topological Charge

The continuum topological charge is

```text
Q = 1 / (4 pi) integral n . (partial_x n cross partial_y n) dx dy
```

For smooth isolated Skyrmions, `Q` should be close to an integer. The library
provides both density and total-charge observables.

## Baby Skyrme Energy

The implemented Baby Skyrme energy is decomposed into:

- `sigma`: gradient energy,
- `skyrme`: stabilizing quartic term,
- `potential`: vacuum-selecting mass term,
- `dmi`: bulk Dzyaloshinskii-Moriya interaction,
- `total`: signed sum of all active terms.

The component API is useful for understanding which physical term dominates a
run:

```python
sk.baby_skyrme_energy_terms(field, kappa=1.0, mass=1.0, dmi=0.2)
```

## Boundary Conditions

The boundary condition changes how the finite lattice approximates the intended
physics:

- `periodic`: wraps edges, useful for periodic media.
- `fixed`: keeps edge sites at their initial values.
- `neumann`: clamps neighbor access at the edge.
- `dirichlet`: pins edge sites to `n=(0,0,1)`.

For a single Skyrmion on a finite grid, Dirichlet boundary conditions are often
closer to the infinite-plane picture because the far-away field should approach
the vacuum.

## Relaxation Versus Dynamics

Gradient-based relaxation decreases energy and is used to find stable or
metastable configurations. Landau-Lifshitz evolution is closer to physical time
evolution because it precesses around the effective field and can include
damping.

Use relaxation when you want a low-energy static field. Use dynamics when you
want motion, interaction, precession, or damping-driven evolution.
