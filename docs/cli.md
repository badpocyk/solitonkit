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
  --dt 1e-4 `
  --steps 100 `
  --damping 0.3
```

## Plot

```powershell
solitonkit plot --input relaxed.npz --output relaxed.png
```

The CLI stores metadata in `.npz` files, so downstream commands preserve model
parameters and boundary information where possible.
