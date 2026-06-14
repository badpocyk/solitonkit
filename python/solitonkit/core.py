# python/solitonkit/core.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import importlib

import numpy as np

_cpp = importlib.import_module("solitonkit._core")
from ._core import SkyrmionSpec
from ._core import make_multi_skyrmion_field as _make_multi_skyrmion_field

# ---------------------------------------------------------------------
# Backend flags
# ---------------------------------------------------------------------

HAS_CPP_CORE = True
HAS_CPP_BACKEND = True
CPP_BACKEND_AVAILABLE = True
CPP_CORE_AVAILABLE = True
BACKEND = "cpp"


# ---------------------------------------------------------------------
# Direct C++ classes
# ---------------------------------------------------------------------

Vec3 = _cpp.Vec3
O3Field = _cpp.O3Field
FlowRecord = _cpp.FlowRecord
GradientFlow = _cpp.GradientFlow
BoundaryCondition = _cpp.BoundaryCondition


# ---------------------------------------------------------------------
# Old Python API compatibility
# ---------------------------------------------------------------------

@dataclass
class SkyrmionConfig:
    """
    Configuration for creating a skyrmion.

    This class keeps compatibility with the older Python API,
    where make_skyrmion_from_config(config) was used.
    """

    width: int = 128
    height: int = 128
    radius: float = 20.0
    center_x: Optional[float] = None
    center_y: Optional[float] = None

    # New C++ backend parameters
    spacing: float = 1.0
    dx: Optional[float] = None
    dy: Optional[float] = None
    charge: int = 1
    boundary: str = "periodic"

    @property
    def nx(self) -> int:
        return self.width

    @property
    def ny(self) -> int:
        return self.height

    @property
    def effective_dx(self) -> float:
        if self.dx is not None:
            return self.dx
        return self.spacing

    @property
    def effective_dy(self) -> float:
        if self.dy is not None:
            return self.dy
        return self.spacing

    @property
    def effective_center_x(self) -> float:
        if self.center_x is not None:
            return self.center_x
        return 0.5 * float(self.width - 1)

    @property
    def effective_center_y(self) -> float:
        if self.center_y is not None:
            return self.center_y
        return 0.5 * float(self.height - 1)


class Field2D:
    def __init__(
        self,
        width: int,
        height: int,
        spacing: float = 1.0,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        boundary: str = "periodic",
    ) -> None:
        if dx is None:
            dx = spacing

        if dy is None:
            dy = spacing

        self._field = _cpp.O3Field(width, height, dx, dy, boundary)

    @classmethod
    def from_cpp(cls, field: O3Field) -> "Field2D":
        obj = cls.__new__(cls)
        obj._field = field
        return obj

    @property
    def cpp(self) -> O3Field:
        return self._field

    @property
    def width(self) -> int:
        return int(self._field.nx)

    @property
    def height(self) -> int:
        return int(self._field.ny)

    @property
    def nx(self) -> int:
        return int(self._field.nx)

    @property
    def ny(self) -> int:
        return int(self._field.ny)

    @property
    def dx(self) -> float:
        return float(self._field.dx)

    @property
    def dy(self) -> float:
        return float(self._field.dy)

    @property
    def spacing(self) -> float:
        return float(self._field.spacing)

    @property
    def boundary(self) -> str:
        return str(self._field.boundary)

    def get(self, x: int, y: int) -> Vec3:
        return self._field.get(x, y)

    def set(self, x: int, y: int, value: Vec3) -> None:
        self._field.set(x, y, value)

    def to_numpy(self) -> np.ndarray:
        return self._field.to_numpy()

    def __repr__(self) -> str:
        return (
            "Field2D("
            f"width={self.width}, "
            f"height={self.height}, "
            f"dx={self.dx}, "
            f"dy={self.dy}, "
            f"boundary={self.boundary!r}"
            ")"
        )


def _unwrap_field(field: Any) -> O3Field:
    if isinstance(field, Field2D):
        return field.cpp

    return field


# ---------------------------------------------------------------------
# Backend info
# ---------------------------------------------------------------------

def require_cpp_core() -> Any:
    """
    Return the C++ backend module.

    Kept for compatibility with the older API.
    """

    return _cpp


def core_info() -> dict[str, Any]:
    """
    Return basic information about the active backend.
    """

    return {
        "backend": "cpp",
        "has_cpp_core": True,
        "module": "solitonkit._core",
        "classes": [
            "Vec3",
            "O3Field",
            "FlowRecord",
            "GradientFlow",
            "BoundaryCondition",
        ],
    }


def backend_name() -> str:
    return "cpp"


def gradient_flow_description() -> str:
    """
    Compatibility helper for the old placeholder GradientFlow API.
    """

    return (
        "GradientFlow is provided by the C++ backend. "
        "Use run_gradient_flow(field, step_size, steps, record_every)."
    )


# ---------------------------------------------------------------------
# Field creation
# ---------------------------------------------------------------------

def make_field2d(
    width: int,
    height: int,
    spacing: float = 1.0,
    *,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    boundary: str = "periodic",
) -> Field2D:
    return Field2D(
        width=width,
        height=height,
        spacing=spacing,
        dx=dx,
        dy=dy,
        boundary=boundary,
    )


def make_uniform_field(
    nx: int,
    ny: int,
    spacing: float = 1.0,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 1.0,
    *,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    boundary: str = "periodic",
) -> O3Field:
    if dx is None:
        dx = spacing

    if dy is None:
        dy = spacing

    return _cpp.make_uniform_field(nx, ny, dx, dy, x, y, z, boundary)


def make_skyrmion_field(
    nx: int,
    ny: int,
    spacing: float = 1.0,
    radius: float = 20.0,
    charge: int = 1,
    *,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    boundary: str = "periodic",
) -> O3Field:
    """
    Create a skyrmion as a real C++ O3Field.

    boundary may be "periodic", "fixed", or "neumann". Fixed boundaries
    retain their initial edge values during gradient flow.
    """

    if dx is None:
        dx = spacing

    if dy is None:
        dy = spacing

    return _cpp.make_skyrmion_field(
        nx,
        ny,
        dx,
        dy,
        radius,
        charge,
        boundary,
    )


def make_skyrmion_field_xy(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    radius: float = 20.0,
    charge: int = 1,
    boundary: str = "periodic",
) -> O3Field:
    return _cpp.make_skyrmion_field(
        nx,
        ny,
        dx,
        dy,
        radius,
        charge,
        boundary,
    )


def make_skyrmion(
    width: int,
    height: int,
    radius: float = 20.0,
) -> np.ndarray:
    """
    Old API: return NumPy array with shape (height, width, 3).
    """

    return _cpp.make_skyrmion(width, height, radius)


def make_skyrmion_at(
    width: int,
    height: int,
    radius: float,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    """
    Old API: return NumPy array with explicit skyrmion center.
    """

    return _cpp.make_skyrmion_at(
        width,
        height,
        radius,
        center_x,
        center_y,
    )


def make_skyrmion_default(
    width: int,
    height: int,
    radius: float = 20.0,
) -> np.ndarray:
    return make_skyrmion(width, height, radius)


def make_skyrmion_from_config(
    config: SkyrmionConfig,
    *,
    as_field: bool = False,
) -> np.ndarray | O3Field:
    """
    Create a skyrmion from SkyrmionConfig.

    By default returns NumPy array for compatibility with the old API.
    Use as_field=True to get a C++ O3Field.
    """

    if as_field:
        return make_skyrmion_field(
            config.width,
            config.height,
            spacing=config.spacing,
            radius=config.radius,
            charge=config.charge,
            dx=config.dx,
            dy=config.dy,
            boundary=config.boundary,
        )

    if config.center_x is None and config.center_y is None:
        return make_skyrmion(
            config.width,
            config.height,
            config.radius,
        )

    return make_skyrmion_at(
        config.width,
        config.height,
        config.radius,
        config.effective_center_x,
        config.effective_center_y,
    )


# ---------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------

def field_from_numpy(
    array: np.ndarray,
    spacing: float = 1.0,
    *,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    boundary: str = "periodic",
) -> O3Field:
    if dx is None:
        dx = spacing

    if dy is None:
        dy = spacing

    return _cpp.field_from_numpy(array, dx, dy, boundary)


def field_to_numpy(field: Any) -> np.ndarray:
    return _cpp.field_to_numpy(_unwrap_field(field))


def field2d_to_numpy(field: Field2D) -> np.ndarray:
    return field.to_numpy()


def to_numpy(field: Any) -> np.ndarray:
    return field_to_numpy(field)


# ---------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------

def energy_density(field: Any) -> np.ndarray:
    return _cpp.energy_density(_unwrap_field(field))


def total_energy(field: Any) -> float:
    return float(_cpp.total_energy(_unwrap_field(field)))


def topological_density(field: Any) -> np.ndarray:
    return _cpp.topological_density(_unwrap_field(field))


def topological_charge(field: Any) -> float:
    return float(_cpp.topological_charge(_unwrap_field(field)))

def topological_charge_geometric(field: Any) -> float:
    """
    Compute the topological charge using a geometric lattice formula.

    This method is usually more stable on discrete fields than
    derivative-based topological charge.
    """

    return float(_cpp.topological_charge_geometric(_unwrap_field(field)))


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------

def run_gradient_flow_inplace(
    field: Any,
    step_size: float,
    steps: int,
    record_every: int = 10,
) -> list[FlowRecord]:
    return _cpp.run_gradient_flow_inplace(
        _unwrap_field(field),
        step_size,
        steps,
        record_every,
    )


def run_gradient_flow(
    field: Any,
    step_size: float,
    steps: int,
    record_every: int = 10,
) -> Tuple[O3Field, list[FlowRecord]]:
    return _cpp.run_gradient_flow(
        _unwrap_field(field),
        step_size,
        steps,
        record_every,
    )


def gradient_flow(
    field: Any,
    step_size: float,
    steps: int,
    record_every: int = 10,
) -> Tuple[O3Field, list[FlowRecord]]:
    return run_gradient_flow(
        field,
        step_size,
        steps,
        record_every,
    )


# ---------------------------------------------------------------------
# Extra aliases
# ---------------------------------------------------------------------

def create_skyrmion(
    width: int,
    height: int,
    radius: float = 20.0,
) -> np.ndarray:
    return make_skyrmion(width, height, radius)


def generate_skyrmion(
    width: int,
    height: int,
    radius: float = 20.0,
) -> np.ndarray:
    return make_skyrmion(width, height, radius)

def make_multi_skyrmion_field(
    nx,
    ny,
    spacing=0.2,
    centers=None,
    charges=None,
    scales=None,
    phases=None,
    boundary="periodic",
):
    """
    Create an O(3) field containing multiple skyrmions / anti-skyrmions.

    Parameters
    ----------
    nx, ny:
        Grid size.

    spacing:
        Lattice spacing. Same value is used for dx and dy.

    centers:
        List of physical coordinates [(x0, y0), ...].

    charges:
        List of topological charges.
        Use +1 for skyrmion, -1 for anti-skyrmion.

    scales:
        List of skyrmion sizes.

    phases:
        List of internal phases / rotations.

    boundary:
        "periodic", "fixed", or "neumann".
    """

    if centers is None:
        centers = [(0.0, 0.0)]

    n = len(centers)

    if n == 0:
        raise ValueError("centers must contain at least one point")

    if charges is None:
        charges = [1] * n

    if scales is None:
        scales = [2.0] * n

    if phases is None:
        phases = [0.0] * n

    if not (len(charges) == len(scales) == len(phases) == n):
        raise ValueError(
            "centers, charges, scales and phases must have the same length"
        )

    specs = []

    for center, charge, scale, phase in zip(
        centers,
        charges,
        scales,
        phases,
    ):
        if len(center) != 2:
            raise ValueError("each center must be a pair (x0, y0)")

        x0, y0 = center

        specs.append(
            SkyrmionSpec(
                float(x0),
                float(y0),
                int(charge),
                float(scale),
                float(phase),
            )
        )

    return _make_multi_skyrmion_field(
        int(nx),
        int(ny),
        specs,
        dx=float(spacing),
        dy=float(spacing),
        boundary=str(boundary),
    )

def baby_skyrme_energy(field, kappa: float = 1.0, mass: float = 1.0) -> float:
    """
    Compute the Baby Skyrme model energy.

    Parameters
    ----------
    field:
        O(3) field.
    kappa:
        Strength of the Skyrme stabilizing term.
    mass:
        Strength of the potential term.

    Returns
    -------
    float
        Baby Skyrme energy.
    """

    return float(
        _cpp.baby_skyrme_energy(
            _unwrap_field(field),
            float(kappa),
            float(mass),
        )
    )

def baby_skyrme_energy_terms(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
) -> dict[str, float]:
    """
    Return the sigma, Skyrme, potential, and total energy contributions.
    """

    terms = _cpp.baby_skyrme_energy_terms(
        _unwrap_field(field),
        float(kappa),
        float(mass),
    )

    return {name: float(value) for name, value in terms.items()}


def run_baby_skyrme_gradient_flow_inplace(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    step_size: float = 1e-4,
    steps: int = 1000,
    record_every: int = 10,
) -> list[FlowRecord]:
    """
    Relax a field in place using the full Baby Skyrme energy.
    """

    return _cpp.run_baby_skyrme_gradient_flow_inplace(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(step_size),
        int(steps),
        int(record_every),
    )


def run_baby_skyrme_gradient_flow(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    step_size: float = 1e-4,
    steps: int = 1000,
    record_every: int = 10,
) -> Tuple[O3Field, list[FlowRecord]]:
    """
    Return a relaxed copy of a field and Baby Skyrme flow records.
    """

    return _cpp.run_baby_skyrme_gradient_flow(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(step_size),
        int(steps),
        int(record_every),
    )

def openmp_enabled() -> bool:
    """
    Return True if the C++ backend was built with OpenMP.
    """

    if hasattr(_cpp, "openmp_enabled"):
        return bool(_cpp.openmp_enabled())

    return False


def openmp_max_threads() -> int:
    """
    Return maximum number of OpenMP threads.
    """

    if hasattr(_cpp, "openmp_max_threads"):
        return int(_cpp.openmp_max_threads())

    return 1

__all__ = [
    "HAS_CPP_CORE",
    "HAS_CPP_BACKEND",
    "CPP_BACKEND_AVAILABLE",
    "CPP_CORE_AVAILABLE",
    "BACKEND",
    "SkyrmionConfig",
    "Vec3",
    "O3Field",
    "Field2D",
    "FlowRecord",
    "GradientFlow",
    "BoundaryCondition",
    "require_cpp_core",
    "core_info",
    "backend_name",
    "gradient_flow_description",
    "make_field2d",
    "make_uniform_field",
    "make_skyrmion_field",
    "make_skyrmion_field_xy",
    "make_skyrmion",
    "make_skyrmion_at",
    "make_skyrmion_default",
    "make_skyrmion_from_config",
    "field_from_numpy",
    "field_to_numpy",
    "field2d_to_numpy",
    "to_numpy",
    "energy_density",
    "total_energy",
    "topological_density",
    "topological_charge",
    "topological_charge_geometric",
    "run_gradient_flow_inplace",
    "run_gradient_flow",
    "gradient_flow",
    "create_skyrmion",
    "generate_skyrmion",
    "baby_skyrme_energy",
    "baby_skyrme_energy_terms",
    "run_baby_skyrme_gradient_flow",
    "run_baby_skyrme_gradient_flow_inplace",
    "make_multi_skyrmion_field",
    "openmp_enabled",
    "openmp_max_threads",
]
