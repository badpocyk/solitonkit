"""Versioned HDF5 run storage and resumable checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .core import (
    BranchResult,
    O3Field,
    O3Field3D,
    ScalarField2D,
    XYField,
    Vec3,
    field3d_from_numpy,
    field_from_numpy,
)


HDF5_FORMAT_VERSION = 1


def _h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - installation diagnostic
        raise ImportError(
            "HDF5 support requires h5py; install solitonkit with its "
            "standard dependencies"
        ) from error
    return h5py


@dataclass(frozen=True)
class StoredRun:
    """In-memory description of a run loaded from an HDF5 container."""

    path: Path
    kind: str
    metadata: dict[str, Any]
    records: dict[str, np.ndarray]
    field_values: tuple[np.ndarray, ...]
    field_specs: tuple[dict[str, Any], ...]

    def field(self, index: int = -1) -> Any:
        """Reconstruct one stored field as a native solitonkit object."""

        if not self.field_values:
            raise ValueError("this stored run does not contain fields")
        return _field_from_array(self.field_values[index], self.field_specs[index])

    @property
    def latest_field(self) -> Any:
        return self.field(-1)


def _field_spec(field: Any) -> dict[str, Any]:
    if isinstance(field, XYField):
        field_type = "xy2d"
    elif isinstance(field, ScalarField2D):
        field_type = "scalar2d"
    elif isinstance(field, O3Field3D):
        field_type = "o3_3d"
    elif isinstance(field, O3Field):
        field_type = "o3_2d"
    else:
        raise TypeError(f"unsupported field type: {type(field).__name__}")
    spec = {
        "field_type": field_type,
        "dx": float(field.dx),
        "dy": float(field.dy),
        "boundary_x": str(field.boundary_x),
        "boundary_y": str(field.boundary_y),
    }
    if field_type == "o3_3d":
        spec.update({
            "dz": float(field.dz),
            "boundary_z": str(field.boundary_z),
        })
    if hasattr(field, "dirichlet_value"):
        value = field.dirichlet_value
        if hasattr(value, "x"):
            spec["dirichlet_value"] = [
                float(value.x), float(value.y), float(value.z)
            ]
        else:
            spec["dirichlet_value"] = float(value)
    return spec


def _field_from_array(values: np.ndarray, spec: Mapping[str, Any]) -> Any:
    field_type = spec["field_type"]
    if field_type == "o3_2d":
        field = field_from_numpy(
            values,
            dx=float(spec["dx"]),
            dy=float(spec["dy"]),
            boundary_x=str(spec["boundary_x"]),
            boundary_y=str(spec["boundary_y"]),
        )
        if "dirichlet_value" in spec:
            field.dirichlet_value = Vec3(*spec["dirichlet_value"])
        return field
    if field_type == "o3_3d":
        field = field3d_from_numpy(
            values,
            dx=float(spec["dx"]),
            dy=float(spec["dy"]),
            dz=float(spec["dz"]),
            boundary_x=str(spec["boundary_x"]),
            boundary_y=str(spec["boundary_y"]),
            boundary_z=str(spec["boundary_z"]),
        )
        if "dirichlet_value" in spec:
            field.dirichlet_value = Vec3(*spec["dirichlet_value"])
        return field
    height, width = values.shape
    field_class = XYField if field_type == "xy2d" else ScalarField2D
    field = field_class(
        width,
        height,
        dx=float(spec["dx"]),
        dy=float(spec["dy"]),
        dirichlet_value=float(spec.get("dirichlet_value", 0.0)),
        boundary_x=str(spec["boundary_x"]),
        boundary_y=str(spec["boundary_y"]),
    )
    for j in range(height):
        for i in range(width):
            field.set(i, j, float(values[j, i]))
    return field


def _record_columns(records: Sequence[Any]) -> dict[str, np.ndarray]:
    if not records:
        return {}
    names = (
        "step",
        "time",
        "energy",
        "gradient_norm",
        "residual_norm",
        "damping",
        "linear_iterations",
        "linear_residual",
        "linear_converged",
        "converged",
    )
    return {
        name: np.asarray([getattr(record, name) for record in records])
        for name in names
        if hasattr(records[0], name)
    }


def _unpack_result(result: Any) -> tuple[str, list[Any], dict[str, np.ndarray]]:
    from .sweeps import SweepResult

    if isinstance(result, BranchResult):
        fields = [point.field for point in result.points]
        records = {
            "parameter": np.asarray([point.parameter for point in result.points]),
            "energy": np.asarray([point.energy for point in result.points]),
            "residual_norm": np.asarray([
                point.residual_norm for point in result.points
            ]),
            "lowest_eigenvalue": np.asarray([
                point.lowest_eigenvalue for point in result.points
            ]),
            "corrector_steps": np.asarray([
                point.corrector_steps for point in result.points
            ]),
            "converged": np.asarray([
                point.converged for point in result.points
            ]),
            "stable": np.asarray([point.stable for point in result.points]),
            "bifurcation_candidate": np.asarray([
                point.bifurcation_candidate for point in result.points
            ]),
        }
        return "branch", fields, records
    if isinstance(result, SweepResult):
        fields = [point.field for point in result.points if point.field is not None]
        records = {
            name: np.asarray([
                point.parameters[name] for point in result.points
            ]) for name in result.parameter_names
        }
        records.update({
            "energy": np.asarray([point.energy for point in result.points]),
            "residual_norm": np.asarray([
                point.residual_norm for point in result.points
            ]),
            "converged": np.asarray([
                point.converged for point in result.points
            ]),
            "phase": np.asarray([point.phase for point in result.points]),
            "topological_value": np.asarray([
                point.topological_value for point in result.points
            ]),
        })
        return "sweep", fields, records
    if isinstance(result, tuple) and len(result) == 2:
        field, history = result
        return "solver", [field], _record_columns(history)
    if hasattr(result, "to_numpy"):
        return "field", [result], {}
    raise TypeError("expected a field, (field, history), branch, or sweep result")


def save_run(
    result: Any,
    path: Any,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Store fields, solver diagnostics, and metadata in one HDF5 file."""

    h5py = _h5py()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    kind, fields, records = _unpack_result(result)
    with h5py.File(target, "w") as handle:
        handle.attrs["format"] = "solitonkit-run"
        handle.attrs["format_version"] = HDF5_FORMAT_VERSION
        handle.attrs["kind"] = kind
        handle.attrs["metadata_json"] = json.dumps(
            dict(metadata or {}), sort_keys=True, default=str
        )
        if fields:
            group = handle.create_group("fields")
            arrays = [np.asarray(field.to_numpy(), dtype=float) for field in fields]
            if any(array.shape != arrays[0].shape for array in arrays):
                raise ValueError("all stored fields must have the same shape")
            group.create_dataset(
                "values", data=np.stack(arrays), compression="gzip", shuffle=True
            )
            group.attrs["specs_json"] = json.dumps([
                _field_spec(field) for field in fields
            ], sort_keys=True)
        record_group = handle.create_group("records")
        for name, values in records.items():
            array = np.asarray(values)
            if array.dtype.kind in {"U", "O"}:
                dtype = h5py.string_dtype(encoding="utf-8")
                record_group.create_dataset(name, data=array.astype(dtype), dtype=dtype)
            else:
                record_group.create_dataset(name, data=array)
    return target


def load_run(path: Any) -> StoredRun:
    """Read a versioned HDF5 run and validate its schema version."""

    h5py = _h5py()
    source = Path(path)
    with h5py.File(source, "r") as handle:
        if handle.attrs.get("format") != "solitonkit-run":
            raise ValueError("not a solitonkit HDF5 run")
        version = int(handle.attrs.get("format_version", -1))
        if version != HDF5_FORMAT_VERSION:
            raise ValueError(f"unsupported HDF5 run version: {version}")
        metadata = json.loads(str(handle.attrs.get("metadata_json", "{}")))
        records = {}
        for name, dataset in handle["records"].items():
            values = np.asarray(dataset)
            if values.dtype.kind == "S":
                values = values.astype(str)
            records[name] = values
        field_values: tuple[np.ndarray, ...] = ()
        field_specs: tuple[dict[str, Any], ...] = ()
        if "fields" in handle:
            values = np.asarray(handle["fields/values"])
            field_values = tuple(np.asarray(value) for value in values)
            field_specs = tuple(json.loads(handle["fields"].attrs["specs_json"]))
        kind = str(handle.attrs["kind"])
    return StoredRun(
        path=source,
        kind=kind,
        metadata=metadata,
        records=records,
        field_values=field_values,
        field_specs=field_specs,
    )


def save_checkpoint(
    field: Any,
    path: Any,
    *,
    step: int = 0,
    parameters: Optional[Mapping[str, Any]] = None,
    history: Sequence[Any] = (),
) -> Path:
    """Write a restartable field checkpoint with optional solver history."""

    metadata = {"checkpoint_step": int(step), "parameters": dict(parameters or {})}
    return save_run((field, history), path, metadata=metadata)


def resume(path: Any) -> StoredRun:
    """Load a checkpoint or stored run; use ``latest_field`` to continue it."""

    return load_run(path)


__all__ = [
    "HDF5_FORMAT_VERSION",
    "StoredRun",
    "save_run",
    "load_run",
    "save_checkpoint",
    "resume",
]
