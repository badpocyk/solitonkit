from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
import numpy as np

from .core import (
    O3Field,
    baby_skyrme_energy,
    field_from_numpy,
    run_baby_skyrme_gradient_flow_inplace,
    topological_charge,
)
from .viz import ComponentName, component_index, ensure_field, field_to_rgb


AnimationMode = Literal["rgb", "component"]


@dataclass
class FlowSnapshot:
    step: int
    field: O3Field
    energy: float
    topological_charge: float


def _copy_field(field: Any) -> O3Field:
    dx = float(getattr(field, "dx", getattr(field, "spacing", 1.0)))
    dy = float(getattr(field, "dy", getattr(field, "spacing", dx)))
    boundary = str(getattr(field, "boundary", "periodic"))

    return field_from_numpy(
        ensure_field(field),
        dx=dx,
        dy=dy,
        boundary=boundary,
    )


def run_baby_skyrme_gradient_flow_snapshots(
    field: Any,
    *,
    kappa: float = 1.0,
    mass: float = 1.0,
    step_size: float = 1e-4,
    steps: int = 1000,
    frame_every: int = 10,
) -> tuple[O3Field, list[FlowSnapshot]]:
    """
    Relax a copy of a field and capture field snapshots for animation.
    """

    if steps < 0:
        raise ValueError("steps must be non-negative")

    if frame_every <= 0:
        raise ValueError("frame_every must be positive")

    relaxed = _copy_field(field)

    snapshots = [
        FlowSnapshot(
            step=0,
            field=_copy_field(relaxed),
            energy=baby_skyrme_energy(relaxed, kappa=kappa, mass=mass),
            topological_charge=topological_charge(relaxed),
        )
    ]

    completed = 0

    while completed < steps:
        chunk = min(frame_every, steps - completed)

        records = run_baby_skyrme_gradient_flow_inplace(
            relaxed,
            kappa=kappa,
            mass=mass,
            step_size=step_size,
            steps=chunk,
            record_every=chunk,
        )

        completed += chunk
        final_record = records[-1]

        snapshots.append(
            FlowSnapshot(
                step=completed,
                field=_copy_field(relaxed),
                energy=float(final_record.energy),
                topological_charge=float(final_record.topological_charge),
            )
        )

    return relaxed, snapshots


def save_field_animation(
    fields: Iterable[Any],
    path: str | Path,
    *,
    fps: int = 10,
    mode: AnimationMode = "rgb",
    component: ComponentName = "z",
    titles: Optional[Sequence[str]] = None,
    dpi: int = 120,
    repeat: bool = True,
) -> Path:
    """
    Save a sequence of fields as a GIF or MP4 animation.
    """

    if fps <= 0:
        raise ValueError("fps must be positive")

    arrays = [ensure_field(field) for field in fields]

    if not arrays:
        raise ValueError("fields must contain at least one field")

    shape = arrays[0].shape

    if any(array.shape != shape for array in arrays):
        raise ValueError("all fields must have the same shape")

    if titles is not None and len(titles) != len(arrays):
        raise ValueError("titles and fields must have the same length")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "rgb":
        images = [field_to_rgb(array) for array in arrays]
        imshow_kwargs: dict[str, Any] = {}
    elif mode == "component":
        index = component_index(component)
        images = [array[:, :, index] for array in arrays]
        imshow_kwargs = {
            "cmap": "coolwarm",
            "vmin": -1.0,
            "vmax": 1.0,
        }
    else:
        raise ValueError("mode must be 'rgb' or 'component'")

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(images[0], origin="lower", **imshow_kwargs)
    title = ax.set_title(titles[0] if titles is not None else "frame 0")
    ax.set_xlabel("i")
    ax.set_ylabel("j")

    def update(frame: int) -> tuple[Any, ...]:
        image.set_data(images[frame])
        title.set_text(titles[frame] if titles is not None else f"frame {frame}")
        return image, title

    animation = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=len(images),
        interval=1000.0 / fps,
        blit=False,
        repeat=repeat,
    )

    suffix = path.suffix.lower()

    try:
        if suffix == ".gif":
            writer = mpl_animation.PillowWriter(fps=fps)
        elif suffix == ".mp4":
            if not mpl_animation.FFMpegWriter.isAvailable():
                raise RuntimeError("MP4 export requires ffmpeg")

            writer = mpl_animation.FFMpegWriter(fps=fps)
        else:
            raise ValueError("animation path must end with .gif or .mp4")

        animation.save(path, writer=writer, dpi=dpi)
    finally:
        plt.close(fig)

    return path


def save_flow_animation(
    snapshots: Sequence[FlowSnapshot],
    path: str | Path,
    *,
    fps: int = 10,
    mode: AnimationMode = "rgb",
    component: ComponentName = "z",
    dpi: int = 120,
    repeat: bool = True,
) -> Path:
    """
    Save FlowSnapshot objects with step, energy, and charge in frame titles.
    """

    titles = [
        (
            f"step {snapshot.step} | "
            f"E = {snapshot.energy:.6g} | "
            f"Q = {snapshot.topological_charge:.6g}"
        )
        for snapshot in snapshots
    ]

    return save_field_animation(
        [snapshot.field for snapshot in snapshots],
        path,
        fps=fps,
        mode=mode,
        component=component,
        titles=titles,
        dpi=dpi,
        repeat=repeat,
    )


__all__ = [
    "AnimationMode",
    "FlowSnapshot",
    "run_baby_skyrme_gradient_flow_snapshots",
    "save_field_animation",
    "save_flow_animation",
]
