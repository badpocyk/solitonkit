from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, overload

import json

import numpy as np

from .core import O3Field, O3Field3D, field3d_from_numpy, field_from_numpy


FIELD_FORMAT_VERSION = 3
SUPPORTED_FIELD_FORMAT_VERSIONS = {1, 2, FIELD_FORMAT_VERSION}
BOUNDARY_CONDITIONS = {"periodic", "fixed", "neumann", "dirichlet"}


def _field_spacing(
    field: Any,
    dx: Optional[float],
    dy: Optional[float],
    dz: Optional[float],
    dimensions: int,
) -> tuple[float, float, float]:
    if dx is None:
        dx = float(getattr(field, "dx", getattr(field, "spacing", 1.0)))

    if dy is None:
        dy = float(getattr(field, "dy", getattr(field, "spacing", dx)))

    if dz is None:
        dz = float(getattr(field, "dz", getattr(field, "spacing", dx)))

    if dx <= 0.0 or dy <= 0.0 or (dimensions == 3 and dz <= 0.0):
        raise ValueError("field spacings must be positive")

    return dx, dy, dz


def _field_array(field: Any) -> np.ndarray:
    if hasattr(field, "to_numpy"):
        field = field.to_numpy()

    array = np.asarray(field, dtype=float)

    if not (
        (array.ndim == 3 and array.shape[2] == 3)
        or (array.ndim == 4 and array.shape[3] == 3)
    ):
        raise ValueError(
            "field must have shape (ny, nx, 3) or (nz, ny, nx, 3)"
        )

    return array


def save_field_npz(
    field: Any,
    path: str | Path,
    *,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    dz: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    compressed: bool = True,
    boundary: Optional[str] = None,
    boundary_x: Optional[str] = None,
    boundary_y: Optional[str] = None,
    boundary_z: Optional[str] = None,
) -> Path:
    """
    Save a 2D or 3D O(3) field, spacings, boundaries, and metadata.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array = _field_array(field)
    dimensions = array.ndim - 1
    dx, dy, dz = _field_spacing(field, dx, dy, dz, dimensions)

    boundary_override = boundary is not None
    if boundary is None:
        boundary = str(getattr(field, "boundary", "periodic"))

    if boundary_x is None:
        boundary_x = (
            boundary if boundary_override
            else getattr(field, "boundary_x", boundary)
        )
    if boundary_y is None:
        boundary_y = (
            boundary if boundary_override
            else getattr(field, "boundary_y", boundary)
        )
    if boundary_z is None:
        boundary_z = (
            boundary if boundary_override
            else getattr(field, "boundary_z", boundary)
        )
    boundary_x = str(boundary_x)
    boundary_y = str(boundary_y)
    boundary_z = str(boundary_z)

    boundaries = (boundary_x, boundary_y)
    if dimensions == 3:
        boundaries += (boundary_z,)

    if any(item not in BOUNDARY_CONDITIONS for item in boundaries):
        raise ValueError(
            "boundaries must be periodic, fixed, neumann, or dirichlet"
        )

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
        dz=np.asarray(dz),
        dimensions=np.asarray(dimensions),
        boundary=np.asarray(boundary_x),
        boundary_x=np.asarray(boundary_x),
        boundary_y=np.asarray(boundary_y),
        boundary_z=np.asarray(boundary_z),
        format_version=np.asarray(FIELD_FORMAT_VERSION),
        metadata_json=np.asarray(metadata_json),
    )

    return path


@overload
def load_field_npz(
    path: str | Path,
    *,
    return_metadata: bool = False,
) -> O3Field | O3Field3D:
    ...


@overload
def load_field_npz(
    path: str | Path,
    *,
    return_metadata: bool,
) -> tuple[O3Field | O3Field3D, dict[str, Any]]:
    ...


def load_field_npz(
    path: str | Path,
    *,
    return_metadata: bool = False,
) -> O3Field | O3Field3D | tuple[O3Field | O3Field3D, dict[str, Any]]:
    """
    Load a 2D or 3D O(3) field saved by :func:`save_field_npz`.
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
        dz = float(data["dz"].item()) if "dz" in data.files else 1.0
        legacy_boundary = (
            str(data["boundary"].item())
            if "boundary" in data.files
            else "periodic"
        )
        boundary_x = (
            str(data["boundary_x"].item())
            if "boundary_x" in data.files else legacy_boundary
        )
        boundary_y = (
            str(data["boundary_y"].item())
            if "boundary_y" in data.files else legacy_boundary
        )
        boundary_z = (
            str(data["boundary_z"].item())
            if "boundary_z" in data.files else legacy_boundary
        )
        dimensions = (
            int(data["dimensions"].item())
            if "dimensions" in data.files else array.ndim - 1
        )
        metadata_json = str(data["metadata_json"].item())

    valid_2d = dimensions == 2 and array.ndim == 3 and array.shape[2] == 3
    valid_3d = dimensions == 3 and array.ndim == 4 and array.shape[3] == 3
    if not (valid_2d or valid_3d):
        raise ValueError("saved field dimensions and array shape are inconsistent")

    if not np.all(np.isfinite(array)):
        raise ValueError("saved field contains non-finite values")

    if dx <= 0.0 or dy <= 0.0 or (dimensions == 3 and dz <= 0.0):
        raise ValueError("saved field spacings must be positive")

    boundaries = (boundary_x, boundary_y)
    if dimensions == 3:
        boundaries += (boundary_z,)
    if any(item not in BOUNDARY_CONDITIONS for item in boundaries):
        raise ValueError("saved field contains an unknown boundary condition")

    metadata = json.loads(metadata_json)

    if not isinstance(metadata, dict):
        raise ValueError("saved metadata must be a JSON object")

    if dimensions == 3:
        field = field3d_from_numpy(
            array,
            dx=dx,
            dy=dy,
            dz=dz,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
            boundary_z=boundary_z,
        )
    else:
        field = field_from_numpy(
            array,
            dx=dx,
            dy=dy,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )

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
