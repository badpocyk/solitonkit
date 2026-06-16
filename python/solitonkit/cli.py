from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence

from .core import (
    baby_skyrme_energy,
    make_skyrmion_field,
    run_baby_skyrme_gradient_flow,
    run_gradient_flow,
    run_landau_lifshitz,
    topological_charge,
    total_energy,
)
from .io import load_field_npz, save_field_npz
from .runner import save_dynamics_records_csv, save_records_csv
from .viz import save_skyrmion_diagnostics, save_skyrmion_plot


def _add_grid_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--spacing", type=float, default=0.2)
    parser.add_argument("--dx", type=float)
    parser.add_argument("--dy", type=float)


def _command_generate(args: Namespace) -> int:
    field = make_skyrmion_field(
        args.nx,
        args.ny,
        spacing=args.spacing,
        dx=args.dx,
        dy=args.dy,
        radius=args.radius,
        charge=args.charge,
        boundary=args.boundary,
    )

    output = save_field_npz(
        field,
        args.output,
        metadata={
            "generator": "skyrmion",
            "radius": args.radius,
            "charge": args.charge,
            "boundary": args.boundary,
        },
    )

    print(f"saved field: {output}")
    print(f"topological charge: {topological_charge(field):.12g}")

    return 0


def _command_relax(args: Namespace) -> int:
    field, input_metadata = load_field_npz(
        args.input,
        return_metadata=True,
    )

    if args.model == "baby-skyrme":
        relaxed, records = run_baby_skyrme_gradient_flow(
            field,
            kappa=args.kappa,
            mass=args.mass,
            step_size=args.step_size,
            steps=args.steps,
            record_every=args.record_every,
            dmi=args.dmi,
        )
        final_energy = baby_skyrme_energy(
            relaxed,
            kappa=args.kappa,
            mass=args.mass,
            dmi=args.dmi,
        )
    else:
        relaxed, records = run_gradient_flow(
            field,
            step_size=args.step_size,
            steps=args.steps,
            record_every=args.record_every,
        )
        final_energy = total_energy(relaxed)

    metadata = dict(input_metadata)
    metadata["relaxation"] = {
        "model": args.model,
        "kappa": args.kappa if args.model == "baby-skyrme" else None,
        "mass": args.mass if args.model == "baby-skyrme" else None,
        "dmi": args.dmi if args.model == "baby-skyrme" else None,
        "step_size": args.step_size,
        "steps": args.steps,
        "record_every": args.record_every,
    }

    output = save_field_npz(relaxed, args.output, metadata=metadata)

    if args.records is not None:
        save_records_csv(records, args.records)

    print(f"saved relaxed field: {output}")
    print(f"final energy: {final_energy:.12g}")
    print(f"final topological charge: {topological_charge(relaxed):.12g}")

    return 0


def _command_plot(args: Namespace) -> int:
    field = load_field_npz(args.input)

    if args.kind == "diagnostics":
        output = save_skyrmion_diagnostics(
            field,
            args.output,
            quiver_step=args.quiver_step,
            show_quiver=not args.no_quiver,
            dpi=args.dpi,
        )
    else:
        output = save_skyrmion_plot(
            field,
            args.output,
            quiver_step=args.quiver_step,
            show_quiver=not args.no_quiver,
            dpi=args.dpi,
        )

    print(f"saved plot: {output}")

    return 0


def _command_evolve(args: Namespace) -> int:
    field, input_metadata = load_field_npz(
        args.input,
        return_metadata=True,
    )

    evolved, records = run_landau_lifshitz(
        field,
        kappa=args.kappa,
        mass=args.mass,
        time_step=args.time_step,
        damping=args.damping,
        steps=args.steps,
        record_every=args.record_every,
        dmi=args.dmi,
    )

    metadata = dict(input_metadata)
    metadata["dynamics"] = {
        "model": "landau-lifshitz",
        "kappa": args.kappa,
        "mass": args.mass,
        "dmi": args.dmi,
        "time_step": args.time_step,
        "damping": args.damping,
        "steps": args.steps,
        "record_every": args.record_every,
    }

    output = save_field_npz(evolved, args.output, metadata=metadata)

    if args.records is not None:
        save_dynamics_records_csv(records, args.records)

    print(f"saved evolved field: {output}")
    print(f"final time: {records[-1].time:.12g}")
    print(f"final energy: {records[-1].energy:.12g}")
    print(f"final topological charge: {records[-1].topological_charge:.12g}")

    return 0


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="solitonkit",
        description="Generate, relax, evolve, and visualize O(3) soliton fields.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a skyrmion field")
    _add_grid_arguments(generate)
    generate.add_argument("--radius", type=float, default=4.0)
    generate.add_argument("--charge", type=int, default=1)
    generate.add_argument(
        "--boundary",
        choices=("periodic", "fixed", "neumann"),
        default="periodic",
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.set_defaults(handler=_command_generate)

    relax = subparsers.add_parser("relax", help="relax a saved field")
    relax.add_argument("--input", type=Path, required=True)
    relax.add_argument("--output", type=Path, required=True)
    relax.add_argument("--records", type=Path)
    relax.add_argument(
        "--model",
        choices=("baby-skyrme", "o3"),
        default="baby-skyrme",
    )
    relax.add_argument("--kappa", type=float, default=1.0)
    relax.add_argument("--mass", type=float, default=1.0)
    relax.add_argument("--dmi", type=float, default=0.0)
    relax.add_argument("--step-size", type=float, default=1e-4)
    relax.add_argument("--steps", type=int, default=1000)
    relax.add_argument("--record-every", type=int, default=10)
    relax.set_defaults(handler=_command_relax)

    evolve = subparsers.add_parser(
        "evolve",
        help="evolve a saved field with Landau-Lifshitz dynamics",
    )
    evolve.add_argument("--input", type=Path, required=True)
    evolve.add_argument("--output", type=Path, required=True)
    evolve.add_argument("--records", type=Path)
    evolve.add_argument("--kappa", type=float, default=1.0)
    evolve.add_argument("--mass", type=float, default=1.0)
    evolve.add_argument("--dmi", type=float, default=0.0)
    evolve.add_argument("--time-step", type=float, default=1e-5)
    evolve.add_argument("--damping", type=float, default=0.0)
    evolve.add_argument("--steps", type=int, default=1000)
    evolve.add_argument("--record-every", type=int, default=10)
    evolve.set_defaults(handler=_command_evolve)

    plot = subparsers.add_parser("plot", help="plot a saved field")
    plot.add_argument("--input", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument(
        "--kind",
        choices=("field", "diagnostics"),
        default="field",
    )
    plot.add_argument("--quiver-step", type=int, default=8)
    plot.add_argument("--no-quiver", action="store_true")
    plot.add_argument("--dpi", type=int, default=160)
    plot.set_defaults(handler=_command_plot)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    return int(args.handler(args))


__all__ = [
    "build_parser",
    "main",
]
