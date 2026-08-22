"""Repeatable micro-benchmarks for the numerical core."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Callable, Sequence

from .core import (
    O3SigmaModel,
    make_skyrmion_field,
    run_gradient_flow,
    stability_analysis,
    topological_charge,
    total_energy,
)


@dataclass(frozen=True)
class BenchmarkEntry:
    operation: str
    size: int
    seconds: float
    sites_per_second: float


@dataclass(frozen=True)
class BenchmarkResult:
    entries: tuple[BenchmarkEntry, ...]
    repeats: int

    def format_table(self) -> str:
        header = "operation                 size    seconds      sites/s"
        rows = [header, "-" * len(header)]
        for entry in self.entries:
            rows.append(
                f"{entry.operation:<25} {entry.size:>5} "
                f"{entry.seconds:>10.6f} {entry.sites_per_second:>12.3g}"
            )
        return "\n".join(rows)


def _measure(call: Callable[[], object], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        samples.append(perf_counter() - start)
    return float(median(samples))


def run_benchmarks(
    *,
    sizes: Sequence[int] = (32, 64, 128),
    repeats: int = 3,
    include_stability: bool = False,
) -> BenchmarkResult:
    """Benchmark energy, topology, flow, and optionally Hessian modes."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    entries: list[BenchmarkEntry] = []
    for raw_size in sizes:
        size = int(raw_size)
        if size < 3:
            raise ValueError("benchmark sizes must be at least 3")
        field = make_skyrmion_field(
            size, size, spacing=0.25, radius=max(1.0, size / 10.0)
        )
        operations: list[tuple[str, Callable[[], object]]] = [
            ("energy", lambda: total_energy(field)),
            ("topological_charge", lambda: topological_charge(field)),
            (
                "gradient_flow_step",
                lambda: run_gradient_flow(
                    field, step_size=1e-4, steps=1, record_every=1
                ),
            ),
        ]
        if include_stability:
            model = O3SigmaModel()
            operations.append((
                "lowest_hessian_mode",
                lambda: stability_analysis(
                    field,
                    model,
                    modes=1,
                    max_iterations=20,
                    subspace_dimension=min(20, 2 * size * size),
                    tolerance=1e-5,
                ),
            ))
        for operation, call in operations:
            seconds = _measure(call, repeats)
            entries.append(BenchmarkEntry(
                operation=operation,
                size=size,
                seconds=seconds,
                sites_per_second=(size * size) / max(seconds, 1e-15),
            ))
    return BenchmarkResult(entries=tuple(entries), repeats=int(repeats))


__all__ = ["BenchmarkEntry", "BenchmarkResult", "run_benchmarks"]
