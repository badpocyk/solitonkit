# CLI Guide

The Python package installs a `solitonkit` command.

## Generate

```powershell
solitonkit generate `
  --nx 128 `
  --ny 128 `
  --spacing 0.25 `
  --radius 5.0 `
  --charge 1 `
  --boundary dirichlet `
  --output field.npz
```

## Relax

```powershell
solitonkit relax `
  --input field.npz `
  --output relaxed.npz `
  --optimizer lbfgs `
  --kappa 1.0 `
  --mass 1.0 `
  --dmi 0.1 `
  --steps 200 `
  --record-every 10 `
  --records relaxation.csv
```

Available `--optimizer` values:

- `gradient`
- `riemannian`
- `barzilai-borwein`
- `lbfgs`
- `semi-implicit`

## Evolve

```powershell
solitonkit evolve `
  --input relaxed.npz `
  --output evolved.npz `
  --time-step 1e-4 `
  --steps 100 `
  --damping 0.3
```

Use the full micromagnetic energy and LLG integrator with:

```powershell
solitonkit evolve `
  --model micromagnetic `
  --input relaxed.npz `
  --output evolved.npz `
  --exchange 1.0 `
  --dmi 0.2 `
  --dmi-type interfacial `
  --anisotropy 0.1 `
  --field-z 0.05 `
  --time-step 1e-3 `
  --damping 0.2 `
  --steps 1000
```

## Plot

```powershell
solitonkit plot --input relaxed.npz --output relaxed.png
```

The CLI stores metadata in `.npz` files, so downstream commands preserve model
parameters and boundary information where possible.

## Benchmark

Run the repeatable core-kernel benchmark suite with:

```powershell
solitonkit benchmark --sizes 32 64 128 --repeats 5
```

Add `--stability` to include the matrix-free lowest-Hessian-mode solver. The
table reports median wall time and processed lattice sites per second. Include
compiler, build type, OpenMP status, and hardware when publishing results.
