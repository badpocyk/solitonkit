"""Parallel parameter sweeps and compact phase-diagram helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from os import cpu_count
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .core import (
    O3Field,
    O3Field3D,
    ScalarField2D,
    XYField,
    degree,
    detect_defects,
    minimize,
    solve_stationary,
    vortex_number,
)


@dataclass(frozen=True)
class SweepPoint:
    parameters: dict[str, float]
    energy: float
    residual_norm: float
    converged: bool
    phase: str
    topological_value: float
    field: Any = None


@dataclass(frozen=True)
class SweepResult:
    parameter_names: tuple[str, ...]
    points: tuple[SweepPoint, ...]

    def values(self, name: str) -> np.ndarray:
        """Return a numeric column by parameter or observable name."""

        if name in self.parameter_names:
            return np.asarray([point.parameters[name] for point in self.points])
        if name not in {"energy", "residual_norm", "topological_value"}:
            raise KeyError(name)
        return np.asarray([getattr(point, name) for point in self.points])

    def phases(self) -> np.ndarray:
        return np.asarray([point.phase for point in self.points], dtype=object)


@dataclass(frozen=True)
class PhaseDiagramResult(SweepResult):
    axes: tuple[np.ndarray, np.ndarray] = (
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
    )

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.axes[0]), len(self.axes[1]))

    @property
    def phase_grid(self) -> np.ndarray:
        return self.phases().reshape(self.shape)

    @property
    def energy_grid(self) -> np.ndarray:
        return self.values("energy").reshape(self.shape)

    def plot(self, *, observable: str = "phase", ax: Any = None) -> Any:
        """Plot the categorical phase map or a numeric sweep observable."""

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if observable == "phase":
            labels = self.phase_grid
            categories = sorted(set(labels.ravel().tolist()))
            lookup = {label: index for index, label in enumerate(categories)}
            values = np.vectorize(lookup.__getitem__)(labels)
            image = ax.imshow(values, origin="lower", aspect="auto")
            colorbar = ax.figure.colorbar(image, ax=ax, ticks=range(len(categories)))
            colorbar.ax.set_yticklabels(categories)
        else:
            values = self.values(observable).reshape(self.shape)
            image = ax.imshow(values, origin="lower", aspect="auto")
            ax.figure.colorbar(image, ax=ax, label=observable)
        ax.set_yticks(range(len(self.axes[0])), self.axes[0])
        ax.set_xticks(range(len(self.axes[1])), self.axes[1])
        ax.set_ylabel(self.parameter_names[0])
        ax.set_xlabel(self.parameter_names[1])
        return ax


def classify_phase(field: Any) -> str:
    """Apply a deterministic, model-agnostic texture classifier."""

    values = np.asarray(field.to_numpy(), dtype=float)
    if isinstance(field, XYField):
        order = abs(np.mean(np.exp(1j * values)))
        defects = 0 if values.size == 0 else len(detect_defects(field))
        if defects:
            return "vortex"
        return "uniform" if order > 0.95 else "modulated"
    if isinstance(field, O3Field3D):
        mean_norm = np.linalg.norm(np.mean(values, axis=(0, 1, 2)))
        return "uniform" if mean_norm > 0.95 else "textured"
    if isinstance(field, O3Field):
        charge = abs(degree(field))
        if charge > 0.5:
            return "topological"
        mean_norm = np.linalg.norm(np.mean(values, axis=(0, 1)))
        return "uniform" if mean_norm > 0.95 else "modulated"
    if isinstance(field, ScalarField2D):
        return "uniform" if float(np.std(values)) < 1e-3 else "structured"
    return "unknown"


def _topological_value(field: Any) -> float:
    if isinstance(field, XYField):
        return float(vortex_number(field))
    if isinstance(field, O3Field) and not isinstance(field, O3Field3D):
        return float(degree(field))
    return float("nan")


def parameter_sweep(
    initial_field: Any,
    model_factory: Callable[..., Any],
    parameters: Mapping[str, Sequence[float]],
    *,
    solver: str = "stationary",
    solver_kwargs: Optional[Mapping[str, Any]] = None,
    classifier: Optional[Callable[[Any], str]] = None,
    workers: Optional[int] = None,
    keep_fields: bool = False,
) -> SweepResult:
    """Evaluate an independent Cartesian parameter grid in parallel."""

    if not parameters:
        raise ValueError("parameters must not be empty")
    names = tuple(parameters)
    axes = tuple(tuple(float(value) for value in parameters[name]) for name in names)
    if any(not axis for axis in axes):
        raise ValueError("each parameter axis must contain at least one value")
    if solver not in {"stationary", "minimize"}:
        raise ValueError("solver must be 'stationary' or 'minimize'")
    kwargs = dict(solver_kwargs or {})
    classify = classifier or classify_phase
    combinations = [dict(zip(names, values)) for values in product(*axes)]

    def evaluate(values: dict[str, float]) -> SweepPoint:
        model = model_factory(**values)
        field = initial_field(**values) if callable(initial_field) else initial_field
        if solver == "stationary":
            solved, records = solve_stationary(field, model, **kwargs)
            final = records[-1]
            residual = float(final.residual_norm)
            converged = bool(final.converged)
        else:
            solved, records = minimize(field, model, **kwargs)
            final = records[-1]
            residual = float(final.gradient_norm)
            converged = bool(final.converged)
        return SweepPoint(
            parameters=dict(values),
            energy=float(model.energy(solved)),
            residual_norm=residual,
            converged=converged,
            phase=str(classify(solved)),
            topological_value=_topological_value(solved),
            field=solved if keep_fields else None,
        )

    worker_count = workers if workers is not None else min(
        len(combinations), cpu_count() or 1
    )
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    if worker_count == 1:
        points = tuple(evaluate(values) for values in combinations)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            points = tuple(executor.map(evaluate, combinations))
    return SweepResult(parameter_names=names, points=points)


def phase_diagram(
    initial_field: Any,
    model_factory: Callable[..., Any],
    parameters: Mapping[str, Sequence[float]],
    **kwargs: Any,
) -> PhaseDiagramResult:
    """Run a two-parameter sweep and arrange it as a phase diagram."""

    if len(parameters) != 2:
        raise ValueError("phase_diagram requires exactly two parameters")
    result = parameter_sweep(
        initial_field, model_factory, parameters, **kwargs
    )
    axes = tuple(
        np.asarray(tuple(float(value) for value in parameters[name]))
        for name in result.parameter_names
    )
    return PhaseDiagramResult(
        parameter_names=result.parameter_names,
        points=result.points,
        axes=(axes[0], axes[1]),
    )


__all__ = [
    "SweepPoint",
    "SweepResult",
    "PhaseDiagramResult",
    "classify_phase",
    "parameter_sweep",
    "phase_diagram",
]
