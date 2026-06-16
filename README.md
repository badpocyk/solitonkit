# solitonkit
C++/Python toolkit for simulating, analyzing, and visualizing topological solitons, O(3) sigma models, and baby Skyrmions.

Baby Skyrme simulations support sigma, Skyrme, potential, and bulk
Dzyaloshinskii-Moriya interaction terms:

```python
terms = sk.baby_skyrme_energy_terms(field, kappa=1.0, mass=1.0, dmi=0.2)
relaxed, records = sk.run_baby_skyrme_gradient_flow(field, dmi=0.2)
evolved, records = sk.run_landau_lifshitz(field, damping=0.5, dmi=0.2)
```

Boundary conditions can be selected with `boundary="periodic"`, `"fixed"`,
`"neumann"`, or `"dirichlet"`. Dirichlet pins the edge to the vacuum
`n=(0,0,1)`, which is useful for isolated Skyrmions on a finite grid:

```python
field = sk.make_skyrmion_field(128, 128, spacing=0.25, boundary="dirichlet")
```

Baby Skyrme relaxation also includes several optimizer variants:

```python
relaxed, records = sk.run_baby_skyrme_riemannian_gradient_descent(field)
relaxed, records = sk.run_baby_skyrme_barzilai_borwein(field)
relaxed, records = sk.run_baby_skyrme_lbfgs(field)
relaxed, records = sk.run_baby_skyrme_semi_implicit_flow(field)
```

From the CLI:

```powershell
solitonkit generate --nx 128 --ny 128 --boundary dirichlet --output field.npz
solitonkit relax --input field.npz --output relaxed.npz --optimizer lbfgs
```
