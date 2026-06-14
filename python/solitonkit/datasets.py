# python/solitonkit/datasets.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import json
import math
import random

import numpy as np

from .core import (
    make_multi_skyrmion_field,
    to_numpy,
    total_energy,
    topological_charge,
)

try:
    from .core import topological_charge_geometric
except ImportError:
    topological_charge_geometric = None


def _sample_center(
    rng: random.Random,
    extent_x: float,
    extent_y: float,
    margin_fraction: float = 0.25,
) -> tuple[float, float]:
    margin_x = margin_fraction * extent_x
    margin_y = margin_fraction * extent_y

    x = rng.uniform(-0.5 * extent_x + margin_x, 0.5 * extent_x - margin_x)
    y = rng.uniform(-0.5 * extent_y + margin_y, 0.5 * extent_y - margin_y)

    return x, y


def _sample_configuration(
    rng: random.Random,
    *,
    nx: int,
    ny: int,
    spacing: float,
    max_skyrmions: int,
    allow_antiskyrmions: bool,
    scale_range: tuple[float, float],
) -> dict[str, Any]:
    extent_x = nx * spacing
    extent_y = ny * spacing

    n_objects = rng.randint(1, max_skyrmions)

    centers = []
    charges = []
    scales = []
    phases = []

    for _ in range(n_objects):
        centers.append(
            _sample_center(
                rng,
                extent_x=extent_x,
                extent_y=extent_y,
            )
        )

        if allow_antiskyrmions:
            charges.append(rng.choice([-1, 1]))
        else:
            charges.append(1)

        scales.append(rng.uniform(scale_range[0], scale_range[1]))
        phases.append(rng.uniform(0.0, 2.0 * math.pi))

    return {
        "centers": centers,
        "charges": charges,
        "scales": scales,
        "phases": phases,
    }


def generate_skyrmion_dataset(
    output_path: str | Path,
    *,
    n_samples: int = 100,
    nx: int = 128,
    ny: int = 128,
    spacing: float = 0.2,
    max_skyrmions: int = 3,
    allow_antiskyrmions: bool = True,
    scale_range: tuple[float, float] = (2.0, 5.0),
    seed: int = 12345,
    dtype: str = "float32",
) -> dict[str, Any]:
    """
    Generate a dataset of O(3) fields and save it as a compressed .npz file.

    Parameters
    ----------
    output_path:
        Path to .npz file.

    n_samples:
        Number of generated field configurations.

    nx, ny:
        Lattice size.

    spacing:
        Lattice spacing.

    max_skyrmions:
        Maximum number of skyrmions / anti-skyrmions in one field.

    allow_antiskyrmions:
        If True, charges can be +1 or -1.
        If False, only +1 objects are generated.

    scale_range:
        Min and max scale of individual objects.

    seed:
        Random seed.

    dtype:
        NumPy dtype for saved field arrays.

    Returns
    -------
    dict
        Metadata summary.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    fields = []
    energies = []
    charges = []
    geometric_charges = []
    labels = []
    configs = []

    for sample_id in range(n_samples):
        config = _sample_configuration(
            rng,
            nx=nx,
            ny=ny,
            spacing=spacing,
            max_skyrmions=max_skyrmions,
            allow_antiskyrmions=allow_antiskyrmions,
            scale_range=scale_range,
        )

        field = make_multi_skyrmion_field(
            nx,
            ny,
            spacing=spacing,
            centers=config["centers"],
            charges=config["charges"],
            scales=config["scales"],
            phases=config["phases"],
        )

        array = to_numpy(field).astype(dtype, copy=False)

        energy = float(total_energy(field))
        charge = float(topological_charge(field))

        if topological_charge_geometric is not None:
            geometric_charge = float(topological_charge_geometric(field))
        else:
            geometric_charge = float("nan")

        integer_charge = int(round(geometric_charge if not math.isnan(geometric_charge) else charge))

        fields.append(array)
        energies.append(energy)
        charges.append(charge)
        geometric_charges.append(geometric_charge)
        labels.append(integer_charge)

        config_for_json = {
            "sample_id": sample_id,
            "centers": [[float(x), float(y)] for x, y in config["centers"]],
            "charges": [int(q) for q in config["charges"]],
            "scales": [float(s) for s in config["scales"]],
            "phases": [float(p) for p in config["phases"]],
            "energy": energy,
            "charge": charge,
            "geometric_charge": geometric_charge,
            "label": integer_charge,
        }

        configs.append(config_for_json)

    fields_array = np.stack(fields, axis=0)

    energies_array = np.asarray(energies, dtype=np.float64)
    charges_array = np.asarray(charges, dtype=np.float64)
    geometric_charges_array = np.asarray(geometric_charges, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)

    np.savez_compressed(
        output_path,
        fields=fields_array,
        energies=energies_array,
        charges=charges_array,
        geometric_charges=geometric_charges_array,
        labels=labels_array,
    )

    metadata = {
        "n_samples": int(n_samples),
        "nx": int(nx),
        "ny": int(ny),
        "spacing": float(spacing),
        "max_skyrmions": int(max_skyrmions),
        "allow_antiskyrmions": bool(allow_antiskyrmions),
        "scale_range": [float(scale_range[0]), float(scale_range[1])],
        "seed": int(seed),
        "dtype": dtype,
        "output_path": str(output_path),
        "fields_shape": list(fields_array.shape),
    }

    metadata_path = output_path.with_suffix(".json")

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "samples": configs,
            },
            f,
            indent=2,
        )

    return metadata


def load_skyrmion_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """
    Load a dataset generated by generate_skyrmion_dataset.
    """

    path = Path(path)

    with np.load(path) as data:
        return {
            "fields": data["fields"],
            "energies": data["energies"],
            "charges": data["charges"],
            "geometric_charges": data["geometric_charges"],
            "labels": data["labels"],
        }