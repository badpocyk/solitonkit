# python/solitonkit/core.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

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
Vec2 = _cpp.Vec2
O3Field = _cpp.O3Field
O3Field3D = _cpp.O3Field3D
ScalarField2D = _cpp.ScalarField2D
XYField = _cpp.XYField
FlowRecord = _cpp.FlowRecord
DynamicsRecord = _cpp.DynamicsRecord
SolverRecord = _cpp.SolverRecord
GradientFlow = _cpp.GradientFlow
BoundaryCondition = _cpp.BoundaryCondition
BoundaryConditions2D = _cpp.BoundaryConditions2D
BoundaryConditions3D = _cpp.BoundaryConditions3D
FieldKind = _cpp.FieldKind
Model = _cpp.Model
MinimizeOptions = _cpp.MinimizeOptions
SolveOptions = _cpp.SolveOptions
StabilityOptions = _cpp.StabilityOptions
GMRESOptions = _cpp.GMRESOptions
StationaryOptions = _cpp.StationaryOptions
StationaryRecord = _cpp.StationaryRecord
ContinuationOptions = _cpp.ContinuationOptions
VortexDefect = _cpp.VortexDefect
HopfChargeOptions = _cpp.HopfChargeOptions
HopfChargeResult = _cpp.HopfChargeResult
Phi4Model = _cpp.Phi4Model
SineGordonModel = _cpp.SineGordonModel
XYModel = _cpp.XYModel
O3SigmaModel = _cpp.O3SigmaModel
BabySkyrmeModel = _cpp.BabySkyrmeModel
DMIType = _cpp.DMIType
MicromagneticModel = _cpp.MicromagneticModel
HopfionModel = _cpp.HopfionModel
HopfionSpec = _cpp.HopfionSpec
LLGDynamics = _cpp.LLGDynamics


@dataclass(frozen=True)
class StabilityResult:
    """Lowest Hessian eigenpairs and diagnostics for a field state."""

    eigenvalues: np.ndarray
    residual_norms: np.ndarray
    modes: tuple[np.ndarray, ...]
    gradient_norm: float
    iterations: int
    degrees_of_freedom: int
    converged: bool
    stationary: bool
    stable: bool
    eigenvalue_tolerance: float

    @property
    def negative_mode_count(self) -> int:
        return int(np.count_nonzero(
            self.eigenvalues < -self.eigenvalue_tolerance
        ))

    @property
    def soft_mode_count(self) -> int:
        return int(np.count_nonzero(
            np.abs(self.eigenvalues) <= self.eigenvalue_tolerance
        ))


@dataclass(frozen=True)
class LinearSolveResult:
    """Result of a matrix-free GMRES solve."""

    solution: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class BranchPoint:
    """One corrected point on a pseudo-arclength continuation branch."""

    parameter: float
    energy: float
    residual_norm: float
    lowest_eigenvalue: float
    corrector_steps: int
    converged: bool
    stable: bool
    bifurcation_candidate: bool
    field: Any


@dataclass(frozen=True)
class BranchResult:
    """A parameterized branch of stationary field configurations."""

    points: tuple[BranchPoint, ...]
    reached_stop: bool
    converged: bool
    parameter_name: str = "parameter"

    @property
    def parameters(self) -> np.ndarray:
        return np.asarray([point.parameter for point in self.points])

    @property
    def energies(self) -> np.ndarray:
        return np.asarray([point.energy for point in self.points])

    @property
    def lowest_eigenvalues(self) -> np.ndarray:
        return np.asarray([
            point.lowest_eigenvalue for point in self.points
        ])

    @property
    def bifurcation_candidates(self) -> tuple[BranchPoint, ...]:
        return tuple(
            point for point in self.points if point.bifurcation_candidate
        )

    def plot(self, y: str = "energy", *, ax: Any = None) -> Any:
        """Plot energy or the lowest Hessian eigenvalue along the branch."""

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        choices = {
            "energy": (self.energies, "Energy"),
            "lowest_eigenvalue": (
                self.lowest_eigenvalues,
                "Lowest Hessian eigenvalue",
            ),
        }
        if y not in choices:
            raise ValueError(
                "y must be 'energy' or 'lowest_eigenvalue'"
            )
        values, label = choices[y]
        ax.plot(self.parameters, values, marker="o")
        ax.set_xlabel(self.parameter_name)
        ax.set_ylabel(label)
        return ax


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


def _unwrap_field(field: Any) -> Any:
    if isinstance(field, Field2D):
        return field.cpp

    return field


def minimize(
    field: Any,
    model: Model,
    *,
    max_steps: int = 1000,
    step_size: float = 1e-2,
    tolerance: float = 1e-8,
    record_every: int = 10,
    line_search: bool = True,
    min_step_size: float = 1e-12,
):
    """Minimize a model energy and return ``(field_copy, records)``."""

    options = MinimizeOptions()
    options.max_steps = int(max_steps)
    options.step_size = float(step_size)
    options.tolerance = float(tolerance)
    options.record_every = int(record_every)
    options.line_search = bool(line_search)
    options.min_step_size = float(min_step_size)
    return _cpp.minimize(_unwrap_field(field), model, options)


def minimize_inplace(
    field: Any,
    model: Model,
    *,
    max_steps: int = 1000,
    step_size: float = 1e-2,
    tolerance: float = 1e-8,
    record_every: int = 10,
    line_search: bool = True,
    min_step_size: float = 1e-12,
) -> list[SolverRecord]:
    """Minimize a model energy in place."""

    options = MinimizeOptions()
    options.max_steps = int(max_steps)
    options.step_size = float(step_size)
    options.tolerance = float(tolerance)
    options.record_every = int(record_every)
    options.line_search = bool(line_search)
    options.min_step_size = float(min_step_size)
    return _cpp.minimize_inplace(_unwrap_field(field), model, options)


def solve(
    field: Any,
    model: Model,
    *,
    steps: int = 1000,
    time_step: float = 1e-3,
    record_every: int = 10,
    tolerance: float = 0.0,
):
    """Integrate the model's dissipative field equation on a copy."""

    options = SolveOptions()
    options.steps = int(steps)
    options.time_step = float(time_step)
    options.record_every = int(record_every)
    options.tolerance = float(tolerance)
    return _cpp.solve(_unwrap_field(field), model, options)


def solve_inplace(
    field: Any,
    model: Model,
    *,
    steps: int = 1000,
    time_step: float = 1e-3,
    record_every: int = 10,
    tolerance: float = 0.0,
) -> list[SolverRecord]:
    """Integrate the model's dissipative field equation in place."""

    options = SolveOptions()
    options.steps = int(steps)
    options.time_step = float(time_step)
    options.record_every = int(record_every)
    options.tolerance = float(tolerance)
    return _cpp.solve_inplace(_unwrap_field(field), model, options)


def stability_analysis(
    field: Any,
    model: Model,
    *,
    modes: int = 6,
    max_iterations: int = 80,
    subspace_dimension: Optional[int] = None,
    tolerance: float = 1e-7,
    finite_difference_step: float = 1e-5,
    stationarity_tolerance: float = 1e-6,
    eigenvalue_tolerance: float = 1e-8,
    seed: int = 12345,
) -> StabilityResult:
    """
    Compute the lowest matrix-free Hessian eigenmodes around ``field``.

    Scalar and XY fields use their active lattice values directly. O(3) modes
    are computed in the tangent bundle, with fixed boundary values excluded.
    ``stationary`` reports whether the projected gradient is small enough for
    the eigenvalues to have their usual linear-stability interpretation.
    """

    options = StabilityOptions()
    options.modes = int(modes)
    options.max_iterations = int(max_iterations)
    options.subspace_dimension = (
        0 if subspace_dimension is None else int(subspace_dimension)
    )
    options.tolerance = float(tolerance)
    options.finite_difference_step = float(finite_difference_step)
    options.stationarity_tolerance = float(stationarity_tolerance)
    options.eigenvalue_tolerance = float(eigenvalue_tolerance)
    options.seed = int(seed)

    raw = _cpp.stability_analysis(_unwrap_field(field), model, options)
    return StabilityResult(
        eigenvalues=np.asarray(raw["eigenvalues"], dtype=float),
        residual_norms=np.asarray(raw["residual_norms"], dtype=float),
        modes=tuple(np.asarray(mode, dtype=float) for mode in raw["modes"]),
        gradient_norm=float(raw["gradient_norm"]),
        iterations=int(raw["iterations"]),
        degrees_of_freedom=int(raw["degrees_of_freedom"]),
        converged=bool(raw["converged"]),
        stationary=bool(raw["stationary"]),
        stable=bool(raw["stable"]),
        eigenvalue_tolerance=float(eigenvalue_tolerance),
    )


def gmres(
    apply: Callable[[np.ndarray], np.ndarray],
    right_hand_side: Any,
    *,
    restart: int = 30,
    max_iterations: int = 200,
    tolerance: float = 1e-5,
    inverse_diagonal: Optional[Any] = None,
) -> LinearSolveResult:
    """Solve ``A x = b`` with restarted matrix-free right-preconditioned GMRES."""

    options = GMRESOptions()
    options.restart = int(restart)
    options.max_iterations = int(max_iterations)
    options.tolerance = float(tolerance)
    rhs = np.asarray(right_hand_side, dtype=float).reshape(-1)
    preconditioner = [] if inverse_diagonal is None else np.asarray(
        inverse_diagonal, dtype=float
    ).reshape(-1).tolist()

    def wrapped(values: list[float]) -> list[float]:
        output = np.asarray(apply(np.asarray(values, dtype=float)), dtype=float)
        return output.reshape(-1).tolist()

    raw = _cpp.gmres(wrapped, rhs.tolist(), options, preconditioner)
    return LinearSolveResult(
        solution=np.asarray(raw["solution"], dtype=float),
        residual_norm=float(raw["residual_norm"]),
        iterations=int(raw["iterations"]),
        converged=bool(raw["converged"]),
    )


def _stationary_options(
    *,
    max_steps: int,
    tolerance: float,
    finite_difference_step: float,
    initial_damping: float,
    minimum_damping: float,
    trust_radius: float,
    line_search: bool,
    preconditioner_probes: int,
    preconditioner_floor: float,
    seed: int,
    gmres_restart: int,
    gmres_max_iterations: int,
    gmres_tolerance: float,
) -> StationaryOptions:
    options = StationaryOptions()
    options.max_steps = int(max_steps)
    options.tolerance = float(tolerance)
    options.finite_difference_step = float(finite_difference_step)
    options.initial_damping = float(initial_damping)
    options.minimum_damping = float(minimum_damping)
    options.trust_radius = float(trust_radius)
    options.line_search = bool(line_search)
    options.preconditioner_probes = int(preconditioner_probes)
    options.preconditioner_floor = float(preconditioner_floor)
    options.seed = int(seed)
    options.gmres.restart = int(gmres_restart)
    options.gmres.max_iterations = int(gmres_max_iterations)
    options.gmres.tolerance = float(gmres_tolerance)
    return options


def solve_stationary(
    field: Any,
    model: Model,
    *,
    max_steps: int = 40,
    tolerance: float = 1e-8,
    finite_difference_step: float = 1e-5,
    initial_damping: float = 1.0,
    minimum_damping: float = 1e-6,
    trust_radius: float = 10.0,
    line_search: bool = True,
    preconditioner_probes: int = 4,
    preconditioner_floor: float = 1e-6,
    seed: int = 12345,
    gmres_restart: int = 30,
    gmres_max_iterations: int = 200,
    gmres_tolerance: float = 1e-9,
):
    """Find a stationary state with damped matrix-free Newton--Krylov."""

    options = _stationary_options(
        max_steps=max_steps,
        tolerance=tolerance,
        finite_difference_step=finite_difference_step,
        initial_damping=initial_damping,
        minimum_damping=minimum_damping,
        trust_radius=trust_radius,
        line_search=line_search,
        preconditioner_probes=preconditioner_probes,
        preconditioner_floor=preconditioner_floor,
        seed=seed,
        gmres_restart=gmres_restart,
        gmres_max_iterations=gmres_max_iterations,
        gmres_tolerance=gmres_tolerance,
    )
    return _cpp.solve_stationary(_unwrap_field(field), model, options)


def solve_stationary_inplace(
    field: Any,
    model: Model,
    **kwargs: Any,
) -> list[StationaryRecord]:
    """Newton--Krylov solve that updates ``field`` in place."""

    defaults = {
        "max_steps": 40,
        "tolerance": 1e-8,
        "finite_difference_step": 1e-5,
        "initial_damping": 1.0,
        "minimum_damping": 1e-6,
        "trust_radius": 10.0,
        "line_search": True,
        "preconditioner_probes": 4,
        "preconditioner_floor": 1e-6,
        "seed": 12345,
        "gmres_restart": 30,
        "gmres_max_iterations": 200,
        "gmres_tolerance": 1e-9,
    }
    unknown = set(kwargs) - set(defaults)
    if unknown:
        raise TypeError(f"unexpected option(s): {', '.join(sorted(unknown))}")
    defaults.update(kwargs)
    options = _stationary_options(**defaults)
    return _cpp.solve_stationary_inplace(
        _unwrap_field(field), model, options
    )


def continue_solution(
    field: Any,
    model_factory: Callable[[float], Model],
    *,
    start: float,
    stop: float,
    step: float,
    parameter_name: str = "parameter",
    minimum_step: float = 1e-4,
    maximum_step: float = 0.25,
    max_points: int = 100,
    corrector_tolerance: float = 1e-7,
    max_corrector_steps: int = 12,
    analyze_stability: bool = True,
    bifurcation_tolerance: float = 1e-5,
    stationary_tolerance: float = 1e-8,
    stability_modes: int = 1,
    seed: int = 12345,
) -> BranchResult:
    """Trace stationary solutions with adaptive pseudo-arclength continuation."""

    options = ContinuationOptions()
    options.start = float(start)
    options.stop = float(stop)
    options.step = float(step)
    options.minimum_step = float(minimum_step)
    options.maximum_step = float(maximum_step)
    options.max_points = int(max_points)
    options.corrector_tolerance = float(corrector_tolerance)
    options.max_corrector_steps = int(max_corrector_steps)
    options.analyze_stability = bool(analyze_stability)
    options.bifurcation_tolerance = float(bifurcation_tolerance)
    options.stationary.tolerance = float(stationary_tolerance)
    options.stationary.gmres.tolerance = min(
        1e-8, max(1e-12, 0.1 * float(stationary_tolerance))
    )
    options.gmres.tolerance = min(
        1e-7, max(1e-12, 0.1 * float(corrector_tolerance))
    )
    options.stability.modes = int(stability_modes)
    options.seed = int(seed)

    prototype = model_factory(float(start))
    backends = (
        (Phi4Model, _cpp._continue_phi4),
        (SineGordonModel, _cpp._continue_sine_gordon),
        (XYModel, _cpp._continue_xy),
        (O3SigmaModel, _cpp._continue_o3_sigma),
        (BabySkyrmeModel, _cpp._continue_baby_skyrme),
        (MicromagneticModel, _cpp._continue_micromagnetic),
        (HopfionModel, _cpp._continue_hopfion),
    )
    backend = next(
        (function for model_type, function in backends
         if isinstance(prototype, model_type)),
        None,
    )
    if backend is None:
        raise TypeError("model_factory returned an unsupported model type")
    raw = backend(_unwrap_field(field), model_factory, options)
    points = tuple(BranchPoint(
        parameter=float(item["parameter"]),
        energy=float(item["energy"]),
        residual_norm=float(item["residual_norm"]),
        lowest_eigenvalue=float(item["lowest_eigenvalue"]),
        corrector_steps=int(item["corrector_steps"]),
        converged=bool(item["converged"]),
        stable=bool(item["stable"]),
        bifurcation_candidate=bool(item["bifurcation_candidate"]),
        field=item["field"],
    ) for item in raw["points"])
    return BranchResult(
        points=points,
        reached_stop=bool(raw["reached_stop"]),
        converged=bool(raw["converged"]),
        parameter_name=str(parameter_name),
    )


def degree(field: Any) -> float:
    """Geometric Brouwer degree of a two-dimensional O(3) field."""

    return float(_cpp.degree(_unwrap_field(field)))


def detect_defects(field: XYField, *, threshold: float = 0.5):
    """Locate integer plaquette vortices in an XY field."""

    return _cpp.detect_defects(field, float(threshold))


def vortex_number(field: XYField) -> int:
    """Return the net integer XY vortex number."""

    return int(_cpp.vortex_number(field))


def winding_number(field: XYField) -> int:
    """Alias for :func:`vortex_number`."""

    return int(_cpp.winding_number(field))


def hopf_charge(
    field: O3Field3D,
    *,
    max_iterations: int = 2000,
    tolerance: float = 1e-8,
    return_diagnostics: bool = False,
):
    """Compute the numerical Hopf invariant via a Coulomb-gauge Poisson solve."""

    options = HopfChargeOptions()
    options.max_iterations = int(max_iterations)
    options.tolerance = float(tolerance)
    result = _cpp.hopf_charge(field, options)
    return result if return_diagnostics else float(result.charge)


derivative_x = _cpp.derivative_x
derivative_y = _cpp.derivative_y
derivative_z = _cpp.derivative_z
laplacian = _cpp.laplacian
gradient = _cpp.gradient
curl = _cpp.curl
make_hopfion_field = _cpp.make_hopfion_field
run_llg = _cpp.run_llg


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
            "Vec2",
            "Vec3",
            "O3Field",
            "O3Field3D",
            "ScalarField2D",
            "XYField",
            "Model",
            "Phi4Model",
            "SineGordonModel",
            "XYModel",
            "MicromagneticModel",
            "HopfionModel",
            "FlowRecord",
            "DynamicsRecord",
            "SolverRecord",
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

    boundary may be "periodic", "fixed", "neumann", or "dirichlet".
    Fixed boundaries retain their initial edge values during relaxation;
    Dirichlet boundaries pin the edge to the vacuum n=(0, 0, 1).
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
    boundary_x: Optional[str] = None,
    boundary_y: Optional[str] = None,
) -> O3Field:
    if dx is None:
        dx = spacing

    if dy is None:
        dy = spacing

    if boundary_x is None:
        boundary_x = boundary

    if boundary_y is None:
        boundary_y = boundary

    return _cpp.field_from_numpy_with_boundaries(
        array,
        dx,
        dy,
        boundary_x,
        boundary_y,
    )


def field3d_from_numpy(
    array: np.ndarray,
    spacing: float = 1.0,
    *,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    dz: Optional[float] = None,
    boundary: str = "periodic",
    boundary_x: Optional[str] = None,
    boundary_y: Optional[str] = None,
    boundary_z: Optional[str] = None,
) -> O3Field3D:
    """Create a normalized 3D O(3) field from ``(nz, ny, nx, 3)`` data."""

    dx = spacing if dx is None else dx
    dy = spacing if dy is None else dy
    dz = spacing if dz is None else dz
    boundary_x = boundary if boundary_x is None else boundary_x
    boundary_y = boundary if boundary_y is None else boundary_y
    boundary_z = boundary if boundary_z is None else boundary_z
    return _cpp.field3d_from_numpy(
        array,
        dx,
        dy,
        dz,
        boundary_x,
        boundary_y,
        boundary_z,
    )


def field_to_numpy(field: Any) -> np.ndarray:
    field = _unwrap_field(field)

    if isinstance(field, O3Field):
        return _cpp.field_to_numpy(field)

    if hasattr(field, "to_numpy"):
        return np.asarray(field.to_numpy(), dtype=float)

    raise TypeError("field does not provide a NumPy conversion")


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
        "periodic", "fixed", "neumann", or "dirichlet".
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

def baby_skyrme_energy(
    field,
    kappa: float = 1.0,
    mass: float = 1.0,
    dmi: float = 0.0,
) -> float:
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
    dmi:
        Strength of the bulk Dzyaloshinskii-Moriya interaction.

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
            float(dmi),
        )
    )

def baby_skyrme_energy_terms(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    dmi: float = 0.0,
) -> dict[str, float]:
    """
    Return the sigma, Skyrme, potential, DMI, and total contributions.
    """

    terms = _cpp.baby_skyrme_energy_terms(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(dmi),
    )

    return {name: float(value) for name, value in terms.items()}


def run_baby_skyrme_gradient_flow_inplace(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    step_size: float = 1e-4,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
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
        float(dmi),
    )


def run_baby_skyrme_gradient_flow(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    step_size: float = 1e-4,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
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
        float(dmi),
    )

def run_baby_skyrme_riemannian_gradient_descent_inplace(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    step_size: float = 1e-4,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
) -> list[FlowRecord]:
    """
    Relax a field in place using Riemannian gradient descent.

    The update is applied with the exponential map on S^2 instead of
    Euler plus renormalization.
    """

    return _cpp.run_baby_skyrme_riemannian_gradient_descent_inplace(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(step_size),
        int(steps),
        int(record_every),
        float(dmi),
    )


def run_baby_skyrme_riemannian_gradient_descent(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    step_size: float = 1e-4,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
) -> Tuple[O3Field, list[FlowRecord]]:
    """
    Return a relaxed copy using Riemannian gradient descent.
    """

    return _cpp.run_baby_skyrme_riemannian_gradient_descent(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(step_size),
        int(steps),
        int(record_every),
        float(dmi),
    )


def run_baby_skyrme_barzilai_borwein_inplace(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    initial_step_size: float = 1e-4,
    min_step_size: float = 1e-8,
    max_step_size: float = 1e-2,
    max_line_search_steps: int = 12,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
) -> list[FlowRecord]:
    """
    Relax a field in place with a Barzilai-Borwein gradient step.
    """

    return _cpp.run_baby_skyrme_barzilai_borwein_inplace(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(initial_step_size),
        float(min_step_size),
        float(max_step_size),
        int(max_line_search_steps),
        int(steps),
        int(record_every),
        float(dmi),
    )


def run_baby_skyrme_barzilai_borwein(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    initial_step_size: float = 1e-4,
    min_step_size: float = 1e-8,
    max_step_size: float = 1e-2,
    max_line_search_steps: int = 12,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
) -> Tuple[O3Field, list[FlowRecord]]:
    """
    Return a relaxed copy using Barzilai-Borwein gradient steps.
    """

    return _cpp.run_baby_skyrme_barzilai_borwein(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(initial_step_size),
        float(min_step_size),
        float(max_step_size),
        int(max_line_search_steps),
        int(steps),
        int(record_every),
        float(dmi),
    )


def run_baby_skyrme_lbfgs_inplace(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    initial_step_size: float = 1.0,
    memory: int = 5,
    max_line_search_steps: int = 12,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
) -> list[FlowRecord]:
    """
    Relax a field in place using a limited-memory BFGS approximation.
    """

    return _cpp.run_baby_skyrme_lbfgs_inplace(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(initial_step_size),
        int(memory),
        int(max_line_search_steps),
        int(steps),
        int(record_every),
        float(dmi),
    )


def run_baby_skyrme_lbfgs(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    initial_step_size: float = 1.0,
    memory: int = 5,
    max_line_search_steps: int = 12,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
) -> Tuple[O3Field, list[FlowRecord]]:
    """
    Return a relaxed copy using limited-memory BFGS.
    """

    return _cpp.run_baby_skyrme_lbfgs(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(initial_step_size),
        int(memory),
        int(max_line_search_steps),
        int(steps),
        int(record_every),
        float(dmi),
    )


def run_baby_skyrme_semi_implicit_flow_inplace(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    step_size: float = 1e-3,
    implicit_iterations: int = 20,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
) -> list[FlowRecord]:
    """
    Relax a field in place with an implicit sigma-model smoothing step.
    """

    return _cpp.run_baby_skyrme_semi_implicit_flow_inplace(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(step_size),
        int(implicit_iterations),
        int(steps),
        int(record_every),
        float(dmi),
    )


def run_baby_skyrme_semi_implicit_flow(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    step_size: float = 1e-3,
    implicit_iterations: int = 20,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
) -> Tuple[O3Field, list[FlowRecord]]:
    """
    Return a relaxed copy with semi-implicit flow.
    """

    return _cpp.run_baby_skyrme_semi_implicit_flow(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(step_size),
        int(implicit_iterations),
        int(steps),
        int(record_every),
        float(dmi),
    )


def run_landau_lifshitz_inplace(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    time_step: float = 1e-5,
    damping: float = 0.0,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
) -> list[DynamicsRecord]:
    """
    Evolve a field in place with damped Landau-Lifshitz dynamics.

    The C++ integrator uses a forward Euler step followed by normalization.
    """

    _validate_dynamics_parameters(
        kappa,
        mass,
        time_step,
        damping,
        steps,
        record_every,
    )

    return _cpp.run_landau_lifshitz_inplace(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(time_step),
        float(damping),
        int(steps),
        int(record_every),
        float(dmi),
    )


def run_landau_lifshitz(
    field: Any,
    kappa: float = 1.0,
    mass: float = 1.0,
    time_step: float = 1e-5,
    damping: float = 0.0,
    steps: int = 1000,
    record_every: int = 10,
    dmi: float = 0.0,
) -> Tuple[O3Field, list[DynamicsRecord]]:
    """
    Return an evolved copy of a field and Landau-Lifshitz records.

    Each record includes step, physical time, energy, and topological charge.
    """

    _validate_dynamics_parameters(
        kappa,
        mass,
        time_step,
        damping,
        steps,
        record_every,
    )

    return _cpp.run_landau_lifshitz(
        _unwrap_field(field),
        float(kappa),
        float(mass),
        float(time_step),
        float(damping),
        int(steps),
        int(record_every),
        float(dmi),
    )


def _validate_dynamics_parameters(
    kappa: float,
    mass: float,
    time_step: float,
    damping: float,
    steps: int,
    record_every: int,
) -> None:
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")

    if mass < 0.0:
        raise ValueError("mass must be non-negative")

    if time_step <= 0.0:
        raise ValueError("time_step must be positive")

    if damping < 0.0:
        raise ValueError("damping must be non-negative")

    if steps < 0:
        raise ValueError("steps must be non-negative")

    if record_every <= 0:
        raise ValueError("record_every must be positive")


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
    "Vec2",
    "Vec3",
    "O3Field",
    "O3Field3D",
    "ScalarField2D",
    "XYField",
    "Field2D",
    "FlowRecord",
    "DynamicsRecord",
    "SolverRecord",
    "GradientFlow",
    "BoundaryCondition",
    "BoundaryConditions2D",
    "BoundaryConditions3D",
    "FieldKind",
    "Model",
    "MinimizeOptions",
    "SolveOptions",
    "StabilityOptions",
    "StabilityResult",
    "GMRESOptions",
    "StationaryOptions",
    "StationaryRecord",
    "ContinuationOptions",
    "LinearSolveResult",
    "BranchPoint",
    "BranchResult",
    "VortexDefect",
    "HopfChargeOptions",
    "HopfChargeResult",
    "Phi4Model",
    "SineGordonModel",
    "XYModel",
    "O3SigmaModel",
    "BabySkyrmeModel",
    "DMIType",
    "MicromagneticModel",
    "HopfionModel",
    "HopfionSpec",
    "LLGDynamics",
    "require_cpp_core",
    "core_info",
    "backend_name",
    "gradient_flow_description",
    "minimize",
    "minimize_inplace",
    "solve",
    "solve_inplace",
    "stability_analysis",
    "gmres",
    "solve_stationary",
    "solve_stationary_inplace",
    "continue_solution",
    "degree",
    "detect_defects",
    "vortex_number",
    "winding_number",
    "hopf_charge",
    "derivative_x",
    "derivative_y",
    "derivative_z",
    "laplacian",
    "gradient",
    "curl",
    "make_hopfion_field",
    "run_llg",
    "make_field2d",
    "make_uniform_field",
    "make_skyrmion_field",
    "make_skyrmion_field_xy",
    "make_skyrmion",
    "make_skyrmion_at",
    "make_skyrmion_default",
    "make_skyrmion_from_config",
    "field_from_numpy",
    "field3d_from_numpy",
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
    "run_baby_skyrme_riemannian_gradient_descent",
    "run_baby_skyrme_riemannian_gradient_descent_inplace",
    "run_baby_skyrme_barzilai_borwein",
    "run_baby_skyrme_barzilai_borwein_inplace",
    "run_baby_skyrme_lbfgs",
    "run_baby_skyrme_lbfgs_inplace",
    "run_baby_skyrme_semi_implicit_flow",
    "run_baby_skyrme_semi_implicit_flow_inplace",
    "run_landau_lifshitz",
    "run_landau_lifshitz_inplace",
    "make_multi_skyrmion_field",
    "openmp_enabled",
    "openmp_max_threads",
]
