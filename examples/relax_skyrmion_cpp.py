# examples/relax_skyrmion_cpp.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


import solitonkit as sk  # noqa: E402


def save_flow_plot(
    records,
    output_path: Path,
    *,
    value_name: str,
    title: str,
    ylabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = [int(record.step) for record in records]
    values = [float(getattr(record, value_name)) for record in records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("gradient flow step")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_density_plot(
    density,
    output_path: Path,
    *,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ax = sk.plot_density(
        density,
        title=title,
    )

    sk.save_figure(
        output_path,
        fig=ax.figure,
        close=True,
    )


def save_relaxation_images(result, output_dir: Path) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    sk.save_skyrmion_plot(
        result.initial_field,
        image_dir / "initial_field_rgb.png",
        quiver_step=6,
        title="Initial skyrmion field",
    )

    sk.save_skyrmion_plot(
        result.relaxed_field,
        image_dir / "relaxed_field_rgb.png",
        quiver_step=6,
        title="Relaxed skyrmion field",
    )

    save_density_plot(
        sk.energy_density(result.initial_field),
        image_dir / "initial_energy_density.png",
        title="Initial energy density",
    )

    save_density_plot(
        sk.energy_density(result.relaxed_field),
        image_dir / "relaxed_energy_density.png",
        title="Relaxed energy density",
    )

    save_density_plot(
        sk.topological_density(result.initial_field),
        image_dir / "initial_topological_density.png",
        title="Initial topological density",
    )

    save_density_plot(
        sk.topological_density(result.relaxed_field),
        image_dir / "relaxed_topological_density.png",
        title="Relaxed topological density",
    )

    save_flow_plot(
        result.records,
        image_dir / "flow_energy.png",
        value_name="energy",
        title="Gradient flow energy",
        ylabel="energy",
    )

    save_flow_plot(
        result.records,
        image_dir / "flow_topological_charge.png",
        value_name="topological_charge",
        title="Gradient flow topological charge",
        ylabel="topological charge",
    )


def print_summary(result, output_dir: Path) -> None:
    print("solitonkit C++ gradient flow example")
    print()
    print(f"HAS_CPP_CORE: {sk.HAS_CPP_CORE}")
    print(f"output_dir: {output_dir}")
    print()
    print(f"initial_field: {result.initial_field}")
    print(f"relaxed_field: {result.relaxed_field}")
    print()
    print(f"initial_energy: {sk.total_energy(result.initial_field)}")
    print(f"final_energy:   {sk.total_energy(result.relaxed_field)}")
    print()
    print(f"initial_topological_charge: {sk.topological_charge(result.initial_field)}")
    print(f"final_topological_charge:   {sk.topological_charge(result.relaxed_field)}")
    print()
    print("records:")
    sk.print_records(result.records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run skyrmion relaxation using solitonkit C++ backend.",
    )

    parser.add_argument(
        "--nx",
        type=int,
        default=128,
        help="Number of lattice points in x direction.",
    )

    parser.add_argument(
        "--ny",
        type=int,
        default=128,
        help="Number of lattice points in y direction.",
    )

    parser.add_argument(
        "--spacing",
        type=float,
        default=0.2,
        help="Uniform lattice spacing.",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=4.0,
        help="Skyrmion radius parameter.",
    )

    parser.add_argument(
        "--charge",
        type=int,
        default=1,
        help="Skyrmion charge / winding number.",
    )

    parser.add_argument(
        "--step-size",
        type=float,
        default=0.001,
        help="Gradient flow step size.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of gradient flow steps.",
    )

    parser.add_argument(
        "--record-every",
        type=int,
        default=10,
        help="Record energy and topological charge every N steps.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "relax_skyrmion_cpp",
        help="Directory for simulation outputs.",
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not save CSV files.",
    )

    parser.add_argument(
        "--no-npy",
        action="store_true",
        help="Do not save NumPy .npy files.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = sk.SimulationConfig(
        nx=args.nx,
        ny=args.ny,
        spacing=args.spacing,
        radius=args.radius,
        charge=args.charge,
        step_size=args.step_size,
        steps=args.steps,
        record_every=args.record_every,
    )

    result = sk.run_skyrmion_relaxation(config)

    sk.save_simulation_result(
        result,
        output_dir=args.output_dir,
        save_csv=not args.no_csv,
        save_npy=not args.no_npy,
    )

    save_relaxation_images(
        result,
        output_dir=args.output_dir,
    )

    print_summary(
        result,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
