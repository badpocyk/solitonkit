# solitonkit

`solitonkit` is a C++/Python toolkit for simulating, relaxing, analyzing, and
visualizing two-dimensional topological solitons: O(3) sigma-model fields,
baby Skyrmions, Dzyaloshinskii-Moriya interaction terms, gradient flows, and
Landau-Lifshitz-type dynamics.


![Skyrmion diagnostics](https://raw.githubusercontent.com/badpocyk/solitonkit/main/docs/assets/skyrmion_diagnostics.png)

## Highlights

- C++ core for `Vec3`, `Lattice2D`, `O3Field`, observables, flows, and models.
- Python API for field generation, relaxation, dynamics, visualization, I/O,
  and dataset generation.
- Baby Skyrme energy decomposed into `sigma`, `skyrme`, `potential`, `dmi`,
  and `total` terms.
- Boundary conditions: `periodic`, `fixed`, `neumann`, and `dirichlet`.
- Baby Skyrme optimizers:
  - gradient flow,
  - Riemannian gradient descent with the exponential map,
  - Barzilai-Borwein gradient method,
  - L-BFGS,
  - semi-implicit flow.
- Landau-Lifshitz evolution with optional damping.
- Export to PNG, GIF, MP4, CSV, and NPZ.

![Gradient-flow animation](https://raw.githubusercontent.com/badpocyk/solitonkit/main/docs/assets/gradient_flow.gif)

## Installation

The package builds a pybind11 extension from the C++ headers.

After the first PyPI release:

```powershell
python -m pip install solitonkit
```

For an editable source installation:

```powershell
python -m pip install -U pip
python -m pip install -e .
```

For a C++-only build and tests:

```powershell
cmake -S . -B build -DSOLITONKIT_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

For Python bindings through CMake:

```powershell
python -m pip install pybind11 numpy matplotlib pillow
cmake -S . -B build -DSOLITONKIT_BUILD_PYTHON=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Quickstart

```python
import solitonkit as sk

field = sk.make_skyrmion_field(
    128,
    128,
    spacing=0.25,
    radius=5.0,
    boundary="dirichlet",
)

print("Q =", sk.topological_charge(field))
print("E =", sk.baby_skyrme_energy(field, kappa=1.0, mass=1.0, dmi=0.1))

relaxed, records = sk.run_baby_skyrme_lbfgs(
    field,
    kappa=1.0,
    mass=1.0,
    dmi=0.1,
    steps=100,
    record_every=10,
)

sk.save_field_npz(relaxed, "relaxed.npz")
sk.save_skyrmion_diagnostics(relaxed, "relaxed.png", spacing=0.25)
```

## Boundary Conditions

`solitonkit` supports four boundary modes:

| Boundary | Meaning | Typical use |
| --- | --- | --- |
| `periodic` | Left/right and top/bottom edges wrap around. | Periodic media, lattices, bulk phases. |
| `fixed` | Boundary sites keep their initial values. | Hand-crafted constraints. |
| `neumann` | Boundary accesses are clamped to edge values. | Approximate zero-normal-gradient edges. |
| `dirichlet` | Boundary sites are pinned to `n=(0,0,1)`. | Isolated Skyrmions on a finite grid. |

![Boundary conditions](https://raw.githubusercontent.com/badpocyk/solitonkit/main/docs/assets/boundary_conditions.png)

For a single Skyrmion, `dirichlet` is usually the physically cleaner finite-grid
choice because the boundary represents the vacuum at infinity.

## Baby Skyrme Model

The Baby Skyrme model includes the sigma-model term, Skyrme stabilizing term,
potential term, and an optional bulk Dzyaloshinskii-Moriya interaction:

```python
terms = sk.baby_skyrme_energy_terms(
    field,
    kappa=1.0,
    mass=1.0,
    dmi=0.2,
)

print(terms)
# {'sigma': ..., 'skyrme': ..., 'potential': ..., 'dmi': ..., 'total': ...}
```

![Energy terms](https://raw.githubusercontent.com/badpocyk/solitonkit/main/docs/assets/energy_terms.png)

## Optimizers

All Baby Skyrme optimizers return `(relaxed_field, records)`, where each record
contains `step`, `energy`, and `topological_charge`.

```python
relaxed, records = sk.run_baby_skyrme_gradient_flow(field)
relaxed, records = sk.run_baby_skyrme_riemannian_gradient_descent(field)
relaxed, records = sk.run_baby_skyrme_barzilai_borwein(field)
relaxed, records = sk.run_baby_skyrme_lbfgs(field)
relaxed, records = sk.run_baby_skyrme_semi_implicit_flow(field)
```

![Optimizer energy comparison](https://raw.githubusercontent.com/badpocyk/solitonkit/main/docs/assets/optimizer_energy.png)

The comparison above uses a perturbed vacuum field as a quick convergence
sanity check. For topological sectors, monitor both energy and `Q`; very
aggressive settings can relax through topology-changing lattice artifacts.

Use `gradient_flow` as the most conservative baseline, `lbfgs` for faster
relaxation experiments, and `semi_implicit_flow` when sigma-model stiffness
limits explicit step sizes.

## CLI

The package exposes a `solitonkit` command:

```powershell
solitonkit generate --nx 128 --ny 128 --spacing 0.25 --boundary dirichlet --output field.npz
solitonkit relax --input field.npz --output relaxed.npz --optimizer lbfgs --steps 200
solitonkit evolve --input relaxed.npz --output evolved.npz --steps 100 --damping 0.3
solitonkit plot --input relaxed.npz --output relaxed.png
```

Available Baby Skyrme relaxers:

```powershell
solitonkit relax --optimizer gradient
solitonkit relax --optimizer riemannian
solitonkit relax --optimizer barzilai-borwein
solitonkit relax --optimizer lbfgs
solitonkit relax --optimizer semi-implicit
```

## Tutorials And Docs

- [Documentation index](docs/index.md)
- [Quickstart guide](docs/quickstart.md)
- [Theory notes](docs/theory.md)
- [Python API overview](docs/python-api.md)
- [CLI guide](docs/cli.md)
- [Publishing to PyPI](docs/pypi-release.md)
- [Demonstration notebook](notebooks/01_solitonkit_demo.ipynb)

## Development

Run the C++ tests:

```powershell
cmake -S . -B build/validation -DSOLITONKIT_BUILD_PYTHON=OFF
cmake --build build/validation
ctest --test-dir build/validation --output-on-failure
```

Run the Python integration tests after building the extension:

```powershell
python tests/test_python_io_animation.py -v
```

Regenerate documentation screenshots, GIFs, and the demo notebook:

```powershell
python scripts/generate_docs_demo.py
```

Generated research outputs are ignored by default, but curated documentation
media under `docs/assets/` is tracked intentionally.
