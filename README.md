# solitonkit
C++/Python toolkit for simulating, analyzing, and visualizing topological solitons, O(3) sigma models, and baby Skyrmions.

Baby Skyrme simulations support sigma, Skyrme, potential, and bulk
Dzyaloshinskii-Moriya interaction terms:

```python
terms = sk.baby_skyrme_energy_terms(field, kappa=1.0, mass=1.0, dmi=0.2)
relaxed, records = sk.run_baby_skyrme_gradient_flow(field, dmi=0.2)
evolved, records = sk.run_landau_lifshitz(field, damping=0.5, dmi=0.2)
```
