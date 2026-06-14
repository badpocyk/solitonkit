# python/solitonkit/adaptive.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .core import (
    O3Field,
    run_gradient_flow,
    total_energy,
    topological_charge,
)


@dataclass
class AdaptiveFlowRecord:
    step: int
    energy: float
    topological_charge: float
    step_size: float
    accepted: bool


def run_adaptive_gradient_flow(
    field: Any,
    *,
    initial_step_size: float = 1e-3,
    min_step_size: float = 1e-8,
    max_step_size: float = 1e-2,
    steps: int = 1000,
    chunk_size: int = 1,
    record_every: int = 10,
    growth_factor: float = 1.05,
    shrink_factor: float = 0.5,
    tolerance: float = 1e-12,
) -> tuple[O3Field, list[AdaptiveFlowRecord]]:
    """
    Run gradient flow with adaptive step size.

    The algorithm accepts an update if energy does not increase.
    If energy increases, the update is rejected and step size is reduced.
    """

    if initial_step_size <= 0.0:
        raise ValueError("initial_step_size must be positive")

    if min_step_size <= 0.0:
        raise ValueError("min_step_size must be positive")

    if max_step_size <= 0.0:
        raise ValueError("max_step_size must be positive")

    if not (min_step_size <= initial_step_size <= max_step_size):
        raise ValueError(
            "initial_step_size must be between min_step_size and max_step_size"
        )

    if steps <= 0:
        raise ValueError("steps must be positive")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if record_every <= 0:
        raise ValueError("record_every must be positive")

    current = field
    step_size = float(initial_step_size)

    current_energy = float(total_energy(current))
    current_charge = float(topological_charge(current))

    records: list[AdaptiveFlowRecord] = [
        AdaptiveFlowRecord(
            step=0,
            energy=current_energy,
            topological_charge=current_charge,
            step_size=step_size,
            accepted=True,
        )
    ]

    completed_steps = 0

    while completed_steps < steps:
        actual_chunk = min(chunk_size, steps - completed_steps)

        candidate, _ = run_gradient_flow(
            current,
            step_size=step_size,
            steps=actual_chunk,
            record_every=actual_chunk,
        )

        candidate_energy = float(total_energy(candidate))

        accepted = candidate_energy <= current_energy + tolerance

        if accepted:
            current = candidate
            current_energy = candidate_energy
            current_charge = float(topological_charge(current))

            completed_steps += actual_chunk

            step_size = min(
                max_step_size,
                step_size * growth_factor,
            )
        else:
            step_size = max(
                min_step_size,
                step_size * shrink_factor,
            )

            if step_size <= min_step_size:
                break

        if completed_steps % record_every == 0 or not accepted:
            records.append(
                AdaptiveFlowRecord(
                    step=completed_steps,
                    energy=current_energy,
                    topological_charge=current_charge,
                    step_size=step_size,
                    accepted=accepted,
                )
            )

    if records[-1].step != completed_steps:
        records.append(
            AdaptiveFlowRecord(
                step=completed_steps,
                energy=current_energy,
                topological_charge=current_charge,
                step_size=step_size,
                accepted=True,
            )
        )

    return current, records