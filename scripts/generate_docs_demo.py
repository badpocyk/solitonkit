from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import solitonkit as sk


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
NOTEBOOKS = ROOT / "notebooks"


def save_energy_terms(path: Path) -> None:
    field = sk.make_skyrmion_field(
        64,
        64,
        spacing=0.3,
        radius=4.0,
        boundary="dirichlet",
    )
    terms = sk.baby_skyrme_energy_terms(
        field,
        kappa=1.0,
        mass=0.8,
        dmi=0.12,
    )

    names = ["sigma", "skyrme", "potential", "dmi", "total"]
    values = [float(terms[name]) for name in names]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(names, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("energy contribution")
    ax.set_title("Baby Skyrme energy components")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_boundary_conditions(path: Path) -> None:
    periodic = sk.make_uniform_field(
        40,
        40,
        x=1.0,
        y=0.0,
        z=0.0,
        boundary="periodic",
    )
    dirichlet = sk.make_uniform_field(
        40,
        40,
        x=1.0,
        y=0.0,
        z=0.0,
        boundary="dirichlet",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    sk.plot_rgb(periodic, ax=axes[0], title="periodic: edge wraps")
    sk.plot_rgb(dirichlet, ax=axes[1], title="dirichlet: edge is vacuum")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def optimizer_run(method, field, **kwargs):
    _, records = method(
        field,
        kappa=0.6,
        mass=0.6,
        dmi=0.0,
        steps=35,
        record_every=1,
        **kwargs,
    )
    return records


def save_optimizer_energy(path: Path) -> None:
    def make_field() -> sk.O3Field:
        field = sk.make_uniform_field(
            48,
            48,
            spacing=0.35,
            boundary="dirichlet",
        )
        field.set(24, 24, sk.Vec3(1.0, 0.0, 0.0))
        field.set(23, 24, sk.Vec3(0.8, 0.2, 0.5))
        field.set(24, 23, sk.Vec3(0.6, -0.4, 0.6))
        field.set(25, 24, sk.Vec3(-0.2, 0.8, 0.5))
        return field

    methods = [
        (
            "gradient",
            sk.run_baby_skyrme_gradient_flow,
            {"step_size": 8e-5},
        ),
        (
            "riemannian",
            sk.run_baby_skyrme_riemannian_gradient_descent,
            {"step_size": 8e-5},
        ),
        (
            "barzilai-borwein",
            sk.run_baby_skyrme_barzilai_borwein,
            {"initial_step_size": 8e-5, "max_step_size": 2e-3},
        ),
        (
            "lbfgs",
            sk.run_baby_skyrme_lbfgs,
            {"initial_step_size": 0.25, "memory": 5},
        ),
        (
            "semi-implicit",
            sk.run_baby_skyrme_semi_implicit_flow,
            {"step_size": 8e-4, "implicit_iterations": 12},
        ),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for name, method, kwargs in methods:
        records = optimizer_run(method, make_field(), **kwargs)
        steps = [record.step for record in records]
        energies = [record.energy for record in records]
        ax.plot(steps, energies, marker="o", markersize=2.5, linewidth=1.4, label=name)

    ax.set_xlabel("step")
    ax.set_ylabel("Baby Skyrme energy")
    ax.set_title("Perturbed-vacuum relaxation by optimizer")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_diagnostics_and_animation() -> None:
    field = sk.make_skyrmion_field(
        64,
        64,
        spacing=0.3,
        radius=4.0,
        boundary="dirichlet",
    )

    relaxed, snapshots = sk.run_baby_skyrme_gradient_flow_snapshots(
        field,
        kappa=0.8,
        mass=0.8,
        dmi=0.08,
        step_size=8e-5,
        steps=90,
        frame_every=6,
    )

    sk.save_skyrmion_diagnostics(
        relaxed,
        ASSETS / "skyrmion_diagnostics.png",
        spacing=0.3,
        quiver_step=6,
        dpi=160,
    )
    sk.save_flow_animation(
        snapshots,
        ASSETS / "gradient_flow.gif",
        fps=8,
        dpi=90,
    )


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def save_notebook(path: Path) -> None:
    notebook = {
        "cells": [
            markdown_cell(
                """
                # solitonkit demo

                This notebook demonstrates the core workflow: generate an
                isolated Skyrmion, inspect observables, decompose Baby Skyrme
                energy terms, relax the field, save/load it, and export an
                animation.
                """
            ),
            markdown_cell(
                """
                ## Visual preview

                The static images and GIF below are generated by
                `scripts/generate_docs_demo.py`.

                ![Skyrmion diagnostics](../docs/assets/skyrmion_diagnostics.png)

                ![Gradient flow](../docs/assets/gradient_flow.gif)
                """
            ),
            code_cell(
                """
                import numpy as np
                import matplotlib.pyplot as plt
                import solitonkit as sk
                """
            ),
            markdown_cell(
                """
                ## 1. Generate a Dirichlet Skyrmion

                Dirichlet boundary conditions pin the edge to `n=(0,0,1)`,
                which approximates the vacuum at infinity for an isolated
                Skyrmion.
                """
            ),
            code_cell(
                """
                field = sk.make_skyrmion_field(
                    96,
                    96,
                    spacing=0.25,
                    radius=5.0,
                    charge=1,
                    boundary="dirichlet",
                )

                print("boundary:", field.boundary)
                print("topological charge:", sk.topological_charge(field))
                print("Baby Skyrme energy:", sk.baby_skyrme_energy(field))
                """
            ),
            code_cell(
                """
                fig = sk.show_skyrmion_diagnostics(
                    field,
                    spacing=0.25,
                    quiver_step=8,
                )
                """
            ),
            markdown_cell(
                """
                ## 2. Decompose the Baby Skyrme energy
                """
            ),
            code_cell(
                """
                terms = sk.baby_skyrme_energy_terms(
                    field,
                    kappa=1.0,
                    mass=1.0,
                    dmi=0.1,
                )
                terms
                """
            ),
            markdown_cell(
                """
                ![Energy terms](../docs/assets/energy_terms.png)
                """
            ),
            markdown_cell(
                """
                ## 3. Compare relaxation optimizers
                """
            ),
            code_cell(
                """
                optimizers = [
                    ("gradient", sk.run_baby_skyrme_gradient_flow, {"step_size": 1e-4}),
                    ("riemannian", sk.run_baby_skyrme_riemannian_gradient_descent, {"step_size": 1e-4}),
                    ("barzilai-borwein", sk.run_baby_skyrme_barzilai_borwein, {"initial_step_size": 1e-4}),
                    ("lbfgs", sk.run_baby_skyrme_lbfgs, {"initial_step_size": 0.25}),
                    ("semi-implicit", sk.run_baby_skyrme_semi_implicit_flow, {"step_size": 1e-3}),
                ]

                for name, method, kwargs in optimizers:
                    relaxed, records = method(
                        field,
                        kappa=0.8,
                        mass=0.8,
                        steps=25,
                        record_every=5,
                        **kwargs,
                    )
                    print(name, "E0 =", records[0].energy, "Efinal =", records[-1].energy)
                """
            ),
            markdown_cell(
                """
                ![Optimizer comparison](../docs/assets/optimizer_energy.png)
                """
            ),
            markdown_cell(
                """
                ## 4. Save, load, and animate
                """
            ),
            code_cell(
                """
                relaxed, snapshots = sk.run_baby_skyrme_gradient_flow_snapshots(
                    field,
                    kappa=0.8,
                    mass=0.8,
                    dmi=0.08,
                    step_size=8e-5,
                    steps=60,
                    frame_every=6,
                )

                sk.save_field_npz(relaxed, "demo_relaxed.npz")
                loaded = sk.load_field_npz("demo_relaxed.npz")

                print("loaded boundary:", loaded.boundary)
                print("loaded charge:", sk.topological_charge(loaded))
                """
            ),
            code_cell(
                """
                sk.save_flow_animation(
                    snapshots,
                    "demo_gradient_flow.gif",
                    fps=8,
                )
                """
            ),
            markdown_cell(
                """
                ## 5. CLI equivalent

                ```powershell
                solitonkit generate --nx 96 --ny 96 --spacing 0.25 --boundary dirichlet --output field.npz
                solitonkit relax --input field.npz --output relaxed.npz --optimizer lbfgs --steps 100
                solitonkit plot --input relaxed.npz --output relaxed.png
                ```
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)

    save_diagnostics_and_animation()
    save_energy_terms(ASSETS / "energy_terms.png")
    save_boundary_conditions(ASSETS / "boundary_conditions.png")
    save_optimizer_energy(ASSETS / "optimizer_energy.png")
    save_notebook(NOTEBOOKS / "01_solitonkit_demo.ipynb")

    print(f"wrote assets to {ASSETS}")
    print(f"wrote notebook to {NOTEBOOKS / '01_solitonkit_demo.ipynb'}")


if __name__ == "__main__":
    main()
