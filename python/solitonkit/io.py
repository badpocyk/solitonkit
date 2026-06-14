from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, overload

import json

import numpy as np

from .core import O3Field, field_from_numpy


FIELD_FORMAT_VERSION = 2
SUPPORTED_FIELD_FORMAT_VERSIONS = {1, FIELD_FORMAT_VERSION}
BOUNDARY_CONDITIONS = {"periodic", "fixed", "neumann"}


def _field_spacing(
    field: Any,
    dx: Optional[float],
    dy: Optional[float],
) -> tuple[float, float]:
    if dx is None:
        dx = float(getattr(field, "dx", getattr(field, "spacing", 1.0)))

    if dy is None:
        dy = float(getattr(field, "dy", getattr(field, "spacing", dx)))

    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")

    return dx, dy


def _field_array(field: Any) -> np.ndarray:
    if hasattr(field, "to_numpy"):
        field = field.to_numpy()

    array = np.asarray(field, dtype=float)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("field must have shape (height, width, 3)")

    return array


def save_field_npz(
    field: Any,
    path: str | Path,
    *,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    compressed: bool = True,
    boundary: Optional[str] = None,
) -> Path:
    """
    Save an O(3) field, lattice spacing, and JSON metadata to an NPZ file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array = _field_array(field)
    dx, dy = _field_spacing(field, dx, dy)

    if boundary is None:
        boundary = str(getattr(field, "boundary", "periodic"))

    if boundary not in BOUNDARY_CONDITIONS:
        raise ValueError("boundary must be 'periodic', 'fixed', or 'neumann'")

    if not np.all(np.isfinite(array)):
        raise ValueError("field contains non-finite values")

    metadata_json = json.dumps(
        dict(metadata or {}),
        ensure_ascii=True,
        sort_keys=True,
    )

    save = np.savez_compressed if compressed else np.savez
    save(
        path,
        field=array,
        dx=np.asarray(dx),
        dy=np.asarray(dy),
        boundary=np.asarray(boundary),
        format_version=np.asarray(FIELD_FORMAT_VERSION),
        metadata_json=np.asarray(metadata_json),
    )

    return path


@overload
def load_field_npz(
    path: str | Path,
    *,
    return_metadata: bool = False,
) -> O3Field:
    ...


@overload
def load_field_npz(
    path: str | Path,
    *,
    return_metadata: bool,
) -> tuple[O3Field, dict[str, Any]]:
    ...


def load_field_npz(
    path: str | Path,
    *,
    return_metadata: bool = False,
) -> O3Field | tuple[O3Field, dict[str, Any]]:
    """
    Load an O(3) field saved by save_field_npz.
    """

    path = Path(path)

    with np.load(path, allow_pickle=False) as data:
        required = {"field", "dx", "dy", "format_version", "metadata_json"}
        missing = required.difference(data.files)

        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"invalid solitonkit field file; missing: {names}")

        version = int(data["format_version"].item())

        if version not in SUPPORTED_FIELD_FORMAT_VERSIONS:
            raise ValueError(f"unsupported field format version: {version}")

        array = np.asarray(data["field"], dtype=float)
        dx = float(data["dx"].item())
        dy = float(data["dy"].item())
        boundary = (
            str(data["boundary"].item())
            if "boundary" in data.files
            else "periodic"
        )
        metadata_json = str(data["metadata_json"].item())

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("saved field must have shape (height, width, 3)")

    if not np.all(np.isfinite(array)):
        raise ValueError("saved field contains non-finite values")

    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("saved dx and dy must be positive")

    metadata = json.loads(metadata_json)

    if not isinstance(metadata, dict):
        raise ValueError("saved metadata must be a JSON object")

    field = field_from_numpy(array, dx=dx, dy=dy, boundary=boundary)

    if return_metadata:
        return field, metadata

    return field


__all__ = [
    "FIELD_FORMAT_VERSION",
    "SUPPORTED_FIELD_FORMAT_VERSIONS",
    "BOUNDARY_CONDITIONS",
    "save_field_npz",
    "load_field_npz",
]
