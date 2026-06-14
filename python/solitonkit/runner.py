# python/solitonkit/runner.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import csv
import json

import numpy as np

from .core import (
    O3Field,
    FlowRecord,
    DynamicsRecord,
    make_skyrmion_field,
    field_to_numpy,
    energy_density,
    total_energy,
    topological_density,
    topological_charge,
    run_gradient_flow,
)


@dataclass
class SimulationConfig:
    """
    Configuration for a simple skyrmion relaxation run.
    """

    nx: int = 128
    ny: int = 128

    spacing: float = 0.2
    dx: Optional[float] = None
    dy: Optional[float] = None

    radius: float = 4.0
    charge: int = 1
    boundary: str = "periodic"

    step_size: float = 0.001
    steps: int = 100
    record_every: int = 10

    def effective_dx(self) -> float:
        if self.dx is not None:
            return self.dx
        return self.spacing

    def effective_dy(self) -> float:
        if self.dy is not None:
            return self.dy
        return self.spacing


@dataclass
class SimulationResult:
    """
    Result of a skyrmion relaxation run.
    """

    initial_field: O3Field
    relaxed_field: O3Field
    records: list[FlowRecord]
    config: SimulationConfig

    def initial_numpy(self) -> np.ndarray:
        return field_to_numpy(self.initial_field)

    def relaxed_numpy(self) -> np.ndarray:
        return field_to_numpy(self.relaxed_field)

    def final_energy(self) -> float:
        return total_energy(self.relaxed_field)

    def final_topological_charge(self) -> float:
        return topological_charge(self.relaxed_field)


def flow_record_to_dict(record: FlowRecord) -> dict[str, float | int]:
    return {
        "step": int(record.step),
        "energy": float(record.energy),
        "topological_charge": float(record.topological_charge),
    }


def flow_records_to_dicts(records: Iterable[FlowRecord]) -> list[dict[str, float | int]]:
    return [flow_record_to_dict(record) for record in records]


def flow_records_to_numpy(records: Iterable[FlowRecord]) -> np.ndarray:
    rows = [
        [
            int(record.step),
            float(record.energy),
            float(record.topological_charge),
        ]
        for record in records
    ]

    return np.asarray(rows, dtype=float)


def dynamics_record_to_dict(
    record: DynamicsRecord,
) -> dict[str, float | int]:
    return {
        "step": int(record.step),
        "time": float(record.time),
        "energy": float(record.energy),
        "topological_charge": float(record.topological_charge),
    }


def save_dynamics_records_csv(
    records: Iterable[DynamicsRecord],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "time",
                "energy",
                "topological_charge",
            ],
        )
        writer.writeheader()

        for record in records:
            writer.writerow(dynamics_record_to_dict(record))

    return path


def create_initial_skyrmion(config: SimulationConfig) -> O3Field:
    return make_skyrmion_field(
        config.nx,
        config.ny,
        spacing=config.spacing,
        radius=config.radius,
        charge=config.charge,
        dx=config.dx,
        dy=config.dy,
        boundary=config.boundary,
    )


def run_skyrmion_relaxation(config: SimulationConfig) -> SimulationResult:
    """
    Create a skyrmion field and relax it with C++ GradientFlow.
    """

    initial_field = create_initial_skyrmion(config)

    relaxed_field, records = run_gradient_flow(
        initial_field,
        step_size=config.step_size,
        steps=config.steps,
        record_every=config.record_every,
    )

    return SimulationResult(
        initial_field=initial_field,
        relaxed_field=relaxed_field,
        records=records,
        config=config,
    )


def save_records_csv(records: Iterable[FlowRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "energy",
                "topological_charge",
            ],
        )

        writer.writeheader()

        for record in records:
            writer.writerow(flow_record_to_dict(record))

    return path


def save_array_npy(array: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(path, array)

    return path


def save_array_csv(array: np.ndarray, path: str | Path) -> Path:
    """
    Save 2D array to CSV.

    For vector fields with shape (height, width, 3), use save_field_csv.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim != 2:
        raise ValueError("save_array_csv expects a 2D array")

    np.savetxt(path, array, delimiter=",")

    return path


def save_field_csv(field: O3Field, path: str | Path) -> Path:
    """
    Save vector field to CSV with columns:
    i, j, nx, ny, nz
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array = field_to_numpy(field)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["i", "j", "nx", "ny", "nz"])

        height, width, _ = array.shape

        for j in range(height):
            for i in range(width):
                writer.writerow(
                    [
                        i,
                        j,
                        float(array[j, i, 0]),
                        float(array[j, i, 1]),
                        float(array[j, i, 2]),
                    ]
                )

    return path


def save_config_json(config: SimulationConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "nx": config.nx,
        "ny": config.ny,
        "spacing": config.spacing,
        "dx": config.dx,
        "dy": config.dy,
        "radius": config.radius,
        "charge": config.charge,
        "boundary": config.boundary,
        "step_size": config.step_size,
        "steps": config.steps,
        "record_every": config.record_every,
    }

    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    return path


def save_summary_txt(result: SimulationResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    initial_energy = total_energy(result.initial_field)
    initial_charge = topological_charge(result.initial_field)

    final_energy = total_energy(result.relaxed_field)
    final_charge = topological_charge(result.relaxed_field)

    text = "\n".join(
        [
            "solitonkit simulation summary",
            "",
            f"nx = {result.config.nx}",
            f"ny = {result.config.ny}",
            f"dx = {result.config.effective_dx()}",
            f"dy = {result.config.effective_dy()}",
            f"radius = {result.config.radius}",
            f"charge = {result.config.charge}",
            f"boundary = {result.config.boundary}",
            f"step_size = {result.config.step_size}",
            f"steps = {result.config.steps}",
            f"record_every = {result.config.record_every}",
            "",
            f"initial_energy = {initial_energy}",
            f"initial_topological_charge = {initial_charge}",
            f"final_energy = {final_energy}",
            f"final_topological_charge = {final_charge}",
        ]
    )

    path.write_text(text, encoding="utf-8")

    return path


def save_simulation_result(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    save_csv: bool = True,
    save_npy: bool = True,
) -> dict[str, Path]:
    """
    Save simulation result to output_dir.

    Created files:
    - config.json
    - summary.txt
    - flow_records.csv
    - initial_field.npy
    - relaxed_field.npy
    - initial_energy_density.npy
    - relaxed_energy_density.npy
    - initial_topological_density.npy
    - relaxed_topological_density.npy

    If save_csv=True, also creates CSV files for fields and 2D densities.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    paths["config"] = save_config_json(
        result.config,
        output_dir / "config.json",
    )

    paths["summary"] = save_summary_txt(
        result,
        output_dir / "summary.txt",
    )

    paths["records_csv"] = save_records_csv(
        result.records,
        output_dir / "flow_records.csv",
    )

    initial_field_array = field_to_numpy(result.initial_field)
    relaxed_field_array = field_to_numpy(result.relaxed_field)

    initial_energy_density = energy_density(result.initial_field)
    relaxed_energy_density = energy_density(result.relaxed_field)

    initial_topological_density = topological_density(result.initial_field)
    relaxed_topological_density = topological_density(result.relaxed_field)

    if save_npy:
        paths["initial_field_npy"] = save_array_npy(
            initial_field_array,
            output_dir / "initial_field.npy",
        )

        paths["relaxed_field_npy"] = save_array_npy(
            relaxed_field_array,
            output_dir / "relaxed_field.npy",
        )

        paths["initial_energy_density_npy"] = save_array_npy(
            initial_energy_density,
            output_dir / "initial_energy_density.npy",
        )

        paths["relaxed_energy_density_npy"] = save_array_npy(
            relaxed_energy_density,
            output_dir / "relaxed_energy_density.npy",
        )

        paths["initial_topological_density_npy"] = save_array_npy(
            initial_topological_density,
            output_dir / "initial_topological_density.npy",
        )

        paths["relaxed_topological_density_npy"] = save_array_npy(
            relaxed_topological_density,
            output_dir / "relaxed_topological_density.npy",
        )

    if save_csv:
        paths["initial_field_csv"] = save_field_csv(
            result.initial_field,
            output_dir / "initial_field.csv",
        )

        paths["relaxed_field_csv"] = save_field_csv(
            result.relaxed_field,
            output_dir / "relaxed_field.csv",
        )

        paths["initial_energy_density_csv"] = save_array_csv(
            initial_energy_density,
            output_dir / "initial_energy_density.csv",
        )

        paths["relaxed_energy_density_csv"] = save_array_csv(
            relaxed_energy_density,
            output_dir / "relaxed_energy_density.csv",
        )

        paths["initial_topological_density_csv"] = save_array_csv(
            initial_topological_density,
            output_dir / "initial_topological_density.csv",
        )

        paths["relaxed_topological_density_csv"] = save_array_csv(
            relaxed_topological_density,
            output_dir / "relaxed_topological_density.csv",
        )

    return paths


def run_and_save_skyrmion_relaxation(
    output_dir: str | Path,
    config: Optional[SimulationConfig] = None,
    *,
    save_csv: bool = True,
    save_npy: bool = True,
) -> SimulationResult:
    """
    Convenience function:
    create skyrmion -> run gradient flow -> save outputs.
    """

    if config is None:
        config = SimulationConfig()

    result = run_skyrmion_relaxation(config)

    save_simulation_result(
        result,
        output_dir=output_dir,
        save_csv=save_csv,
        save_npy=save_npy,
    )

    return result


def print_records(records: Iterable[FlowRecord]) -> None:
    print("step, energy, topological_charge")

    for record in records:
        print(
            f"{int(record.step)}, "
            f"{float(record.energy):.12g}, "
            f"{float(record.topological_charge):.12g}"
        )


__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "flow_record_to_dict",
    "flow_records_to_dicts",
    "flow_records_to_numpy",
    "dynamics_record_to_dict",
    "save_dynamics_records_csv",
    "create_initial_skyrmion",
    "run_skyrmion_relaxation",
    "save_records_csv",
    "save_array_npy",
    "save_array_csv",
    "save_field_csv",
    "save_config_json",
    "save_summary_txt",
    "save_simulation_result",
    "run_and_save_skyrmion_relaxation",
    "print_records",
]
