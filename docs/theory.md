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

The condition can differ by axis. This makes cylindrical domains possible, for
example periodic `x` with Dirichlet `y`. Neumann first derivatives vanish at the
normal boundary; second derivatives use a reflected ghost cell.

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

## Linear stability

At a stationary field `phi_0`, the second variation defines the Hessian

```text
H v = delta^2 E[phi_0] v.
```

Positive eigenvalues are restoring directions, negative eigenvalues are
energy-lowering instabilities, and near-zero modes usually reflect symmetries
or very soft deformations. Numerical eigenvalues should be interpreted only
after checking both the projected gradient norm and the eigenpair residuals.

For O(3) fields the physical variations live in the tangent plane of `S2`.
Using three unconstrained components would introduce a spurious radial mode;
the stability implementation therefore uses two tangent coordinates per active
spin and a Riemannian finite-difference Hessian.

## Scalar and XY models

The phi4 and Sine-Gordon models use the continuum energies

```text
E_phi4 = integral [1/2 |grad phi|^2 + lambda/4 (phi^2 - v^2)^2] d^2x
E_SG   = integral [1/2 |grad phi|^2 + m^2(1-cos(beta phi))/beta^2] d^2x
```

The XY model instead uses compact nearest-neighbor angle differences,
`J[1-cos(theta_j-theta_i)]`, so it remains smooth across the `2 pi` branch cut.

## Micromagnetics

The micromagnetic energy contains exchange, uniaxial anisotropy, Zeeman, and
either bulk or interfacial DMI. The implemented DMI densities are

```text
bulk:         D m . curl(m)
interfacial:  D [m_z div(m) - (m . grad) m_z]
```

`LLGDynamics` evolves the Landau-Lifshitz-Gilbert equation with a normalized
Heun predictor-corrector method. Damping aligns the field with the effective
field while the conservative part produces precession.

## Hopfions

A Hopfion is a map from compactified three-dimensional space `S3` to `S2`.
`HopfionAnsatz` constructs rational-map data with windings `(p,q)` and nominal
Hopf charge `p q`. `HopfionModel` uses the Faddeev-Skyrme energy: a quadratic
sigma term, quartic cross-product terms for each derivative pair, and an
optional vacuum potential.

The library labels the analytic initial condition but does not yet expose a
nonlocal numerical Hopf-invariant estimator for arbitrary evolved fields.
