# python/solitonkit/viz.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt


ComponentName = Literal["x", "y", "z", 0, 1, 2]


def ensure_field(field: Any) -> np.ndarray:
    """
    Convert supported field-like objects to a NumPy array.

    Supported inputs:
    - NumPy array with shape (height, width, 3)
    - C++ O3Field with .to_numpy()
    - Field2D compatibility wrapper with .to_numpy()
    """

    if hasattr(field, "to_numpy"):
        field = field.to_numpy()

    array = np.asarray(field, dtype=float)

    if array.ndim != 3:
        raise ValueError(
            "field must have shape (height, width, 3)"
        )

    if array.shape[2] != 3:
        raise ValueError(
            "field last dimension must be 3"
        )

    return array


def normalize_field(field: Any, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each vector in a field to unit length.
    """

    array = ensure_field(field)

    norms = np.linalg.norm(
        array,
        axis=2,
        keepdims=True,
    )

    return array / np.maximum(norms, eps)


def component_index(component: ComponentName) -> int:
    if component == "x":
        return 0

    if component == "y":
        return 1

    if component == "z":
        return 2

    if component in (0, 1, 2):
        return int(component)

    raise ValueError(
        "component must be one of 'x', 'y', 'z', 0, 1, 2"
    )


def field_to_rgb(field: Any) -> np.ndarray:
    """
    Convert O(3) vector field to RGB image.

    Input values are expected to be approximately in [-1, 1].
    Output values are clipped to [0, 1].
    """

    array = normalize_field(field)

    rgb = 0.5 * (array + 1.0)

    return np.clip(rgb, 0.0, 1.0)


def finite_difference_x(field: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    return (
        np.roll(field, shift=-1, axis=1) -
        np.roll(field, shift=1, axis=1)
    ) / (2.0 * spacing)


def finite_difference_y(field: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    return (
        np.roll(field, shift=-1, axis=0) -
        np.roll(field, shift=1, axis=0)
    ) / (2.0 * spacing)


def energy_density(
    field: Any,
    spacing: float = 1.0,
) -> np.ndarray:
    """
    Compute approximate O(3) sigma-model energy density from a NumPy field.

    For precise C++ backend values, use solitonkit.core.energy_density
    on O3Field.
    """

    array = normalize_field(field)

    dx = finite_difference_x(array, spacing=spacing)
    dy = finite_difference_y(array, spacing=spacing)

    return 0.5 * (
        np.sum(dx * dx, axis=2) +
        np.sum(dy * dy, axis=2)
    )


def topological_density(
    field: Any,
    spacing: float = 1.0,
) -> np.ndarray:
    """
    Compute approximate topological density from a NumPy field.

    For precise C++ backend values, use solitonkit.core.topological_density
    on O3Field.
    """

    array = normalize_field(field)

    dx = finite_difference_x(array, spacing=spacing)
    dy = finite_difference_y(array, spacing=spacing)

    cross = np.cross(dx, dy)

    return np.sum(array * cross, axis=2) / (4.0 * np.pi)


def topological_charge(
    field: Any,
    spacing: float = 1.0,
) -> float:
    """
    Compute approximate total topological charge from a NumPy field.

    For precise C++ backend values, use solitonkit.core.topological_charge
    on O3Field.
    """

    density = topological_density(
        field,
        spacing=spacing,
    )

    return float(np.sum(density) * spacing * spacing)


def plot_component(
    field: Any,
    component: ComponentName = "z",
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    colorbar: bool = True,
    origin: str = "lower",
    **imshow_kwargs: Any,
) -> plt.Axes:
    """
    Plot one component of the vector field.
    """

    array = ensure_field(field)

    index = component_index(component)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        array[:, :, index],
        origin=origin,
        **imshow_kwargs,
    )

    if title is None:
        title = f"field component {component}"

    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")

    if colorbar:
        ax.figure.colorbar(image, ax=ax)

    return ax


def plot_rgb(
    field: Any,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "O(3) field RGB",
    origin: str = "lower",
) -> plt.Axes:
    """
    Plot vector field as RGB image.
    """

    rgb = field_to_rgb(field)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(
        rgb,
        origin=origin,
    )

    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")

    return ax


def plot_quiver(
    field: Any,
    *,
    step: int = 6,
    quiver_step: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "in-plane field",
    origin: str = "lower",
    scale: Optional[float] = None,
) -> plt.Axes:
    """
    Plot in-plane components (nx, ny) as a quiver plot.
    """

    array = ensure_field(field)

    if quiver_step is not None:
        step = quiver_step

    if step <= 0:
        raise ValueError("step must be positive")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    height, width, _ = array.shape

    y = np.arange(0, height, step)
    x = np.arange(0, width, step)

    xx, yy = np.meshgrid(x, y)

    u = array[::step, ::step, 0]
    v = array[::step, ::step, 1]

    if origin == "lower":
        ax.quiver(xx, yy, u, v, scale=scale)
    else:
        ax.quiver(xx, height - 1 - yy, u, -v, scale=scale)

    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_xlim(0, width - 1)
    ax.set_ylim(0, height - 1)

    return ax


def plot_density(
    density: Any,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "density",
    colorbar: bool = True,
    origin: str = "lower",
    **imshow_kwargs: Any,
) -> plt.Axes:
    """
    Plot scalar density.
    """

    array = np.asarray(density, dtype=float)

    if array.ndim != 2:
        raise ValueError("density must be a 2D array")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(
        array,
        origin=origin,
        **imshow_kwargs,
    )

    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")

    if colorbar:
        ax.figure.colorbar(image, ax=ax)

    return ax


def plot_skyrmion(
    field: Any,
    *,
    quiver_step: int = 6,
    show_quiver: bool = True,
    title: str = "skyrmion field",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot skyrmion field as RGB image with optional in-plane quiver overlay.
    """

    array = ensure_field(field)
    rgb = field_to_rgb(array)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(
        rgb,
        origin="lower",
    )

    if show_quiver:
        plot_quiver(
            array,
            step=quiver_step,
            ax=ax,
            title=title,
        )
    else:
        ax.set_title(title)
        ax.set_xlabel("i")
        ax.set_ylabel("j")

    return fig, ax


def plot_surface(
    field: Any,
    component: ComponentName = "z",
    *,
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one component as a 3D surface.
    """

    array = ensure_field(field)
    index = component_index(component)

    height, width, _ = array.shape

    x = np.arange(width)
    y = np.arange(height)

    xx, yy = np.meshgrid(x, y)
    zz = array[:, :, index]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        xx,
        yy,
        zz,
        linewidth=0,
        antialiased=True,
    )

    if title is None:
        title = f"field component {component}"

    ax.set_title(title)
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel(str(component))

    return fig, ax


def save_figure(
    path: str | Path,
    *,
    fig: Optional[plt.Figure] = None,
    dpi: int = 160,
    close: bool = False,
) -> Path:
    """
    Save current or provided matplotlib figure.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fig is None:
        fig = plt.gcf()

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    if close:
        plt.close(fig)

    return path


def save_skyrmion_plot(
    field: Any,
    path: str | Path,
    *,
    quiver_step: int = 6,
    show_quiver: bool = True,
    title: str = "skyrmion field",
    dpi: int = 160,
) -> Path:
    """
    Plot and save skyrmion field to an image file.
    """

    fig, _ = plot_skyrmion(
        field,
        quiver_step=quiver_step,
        show_quiver=show_quiver,
        title=title,
    )

    return save_figure(
        path,
        fig=fig,
        dpi=dpi,
        close=True,
    )


def show() -> None:
    plt.show()

def _try_core_energy_density(field: Any, spacing: float = 1.0) -> np.ndarray:
    """
    Use C++ backend energy_density for O3Field when possible.
    Fall back to NumPy approximation otherwise.
    """

    try:
        from .core import energy_density as core_energy_density

        return np.asarray(
            core_energy_density(field),
            dtype=float,
        )
    except Exception:
        return energy_density(
            field,
            spacing=spacing,
        )


def _try_core_topological_density(field: Any, spacing: float = 1.0) -> np.ndarray:
    """
    Use C++ backend topological_density for O3Field when possible.
    Fall back to NumPy approximation otherwise.
    """

    try:
        from .core import topological_density as core_topological_density

        return np.asarray(
            core_topological_density(field),
            dtype=float,
        )
    except Exception:
        return topological_density(
            field,
            spacing=spacing,
        )


def _try_core_total_energy(field: Any, spacing: float = 1.0) -> float:
    """
    Use C++ backend total_energy for O3Field when possible.
    Fall back to NumPy approximation otherwise.
    """

    try:
        from .core import total_energy as core_total_energy

        return float(core_total_energy(field))
    except Exception:
        density = energy_density(
            field,
            spacing=spacing,
        )

        return float(np.sum(density) * spacing * spacing)


def _try_core_topological_charge(field: Any, spacing: float = 1.0) -> float:
    """
    Use C++ backend topological_charge for O3Field when possible.
    Fall back to NumPy approximation otherwise.
    """

    try:
        from .core import topological_charge as core_topological_charge

        return float(core_topological_charge(field))
    except Exception:
        return topological_charge(
            field,
            spacing=spacing,
        )


def show_skyrmion_plot(
    field: Any,
    *,
    quiver_step: int = 6,
    show_quiver: bool = True,
    title: str = "skyrmion field",
) -> plt.Figure:
    """
    Plot skyrmion field and immediately show it.
    Useful for Jupyter notebooks.
    """

    fig, _ = plot_skyrmion(
        field,
        quiver_step=quiver_step,
        show_quiver=show_quiver,
        title=title,
    )

    plt.show()

    return fig


def show_field_components(
    field: Any,
) -> plt.Figure:
    """
    Show n_x, n_y, n_z components side by side.
    """

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    plot_component(
        field,
        "x",
        ax=axes[0],
        title="n_x",
        vmin=-1.0,
        vmax=1.0,
    )

    plot_component(
        field,
        "y",
        ax=axes[1],
        title="n_y",
        vmin=-1.0,
        vmax=1.0,
    )

    plot_component(
        field,
        "z",
        ax=axes[2],
        title="n_z",
        vmin=-1.0,
        vmax=1.0,
    )

    fig.tight_layout()
    plt.show()

    return fig


def show_energy_density(
    field: Any,
    *,
    spacing: float = 1.0,
) -> plt.Figure:
    """
    Show energy density.
    """

    density = _try_core_energy_density(
        field,
        spacing=spacing,
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    plot_density(
        density,
        ax=ax,
        title="energy density",
    )

    fig.tight_layout()
    plt.show()

    return fig


def show_topological_density(
    field: Any,
    *,
    spacing: float = 1.0,
) -> plt.Figure:
    """
    Show topological charge density.
    """

    density = _try_core_topological_density(
        field,
        spacing=spacing,
    )

    vmax = float(np.max(np.abs(density)))

    if vmax <= 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6, 5))

    plot_density(
        density,
        ax=ax,
        title="topological charge density",
        vmin=-vmax,
        vmax=vmax,
    )

    fig.tight_layout()
    plt.show()

    return fig


def show_skyrmion_diagnostics(
    field: Any,
    *,
    spacing: float = 1.0,
    quiver_step: int = 8,
    show_quiver: bool = True,
) -> plt.Figure:
    """
    Show field, energy density and topological charge density side by side.
    """

    e_density = _try_core_energy_density(
        field,
        spacing=spacing,
    )

    q_density = _try_core_topological_density(
        field,
        spacing=spacing,
    )

    energy = _try_core_total_energy(
        field,
        spacing=spacing,
    )

    charge = _try_core_topological_charge(
        field,
        spacing=spacing,
    )

    q_vmax = float(np.max(np.abs(q_density)))

    if q_vmax <= 0.0:
        q_vmax = 1.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
    )

    plot_rgb(
        field,
        ax=axes[0],
        title="O(3) field RGB",
    )

    if show_quiver:
        plot_quiver(
            field,
            step=quiver_step,
            ax=axes[0],
            title="O(3) field RGB",
        )

    plot_density(
        e_density,
        ax=axes[1],
        title=f"energy density\nE = {energy:.6g}",
    )

    plot_density(
        q_density,
        ax=axes[2],
        title=f"topological density\nQ = {charge:.6g}",
        vmin=-q_vmax,
        vmax=q_vmax,
    )

    fig.tight_layout()
    plt.show()

    return fig


def save_skyrmion_diagnostics(
    field: Any,
    path: str | Path,
    *,
    spacing: float = 1.0,
    quiver_step: int = 8,
    show_quiver: bool = True,
    dpi: int = 160,
) -> Path:
    """
    Save field, energy density and topological density diagnostics.
    """

    e_density = _try_core_energy_density(
        field,
        spacing=spacing,
    )

    q_density = _try_core_topological_density(
        field,
        spacing=spacing,
    )

    energy = _try_core_total_energy(
        field,
        spacing=spacing,
    )

    charge = _try_core_topological_charge(
        field,
        spacing=spacing,
    )

    q_vmax = float(np.max(np.abs(q_density)))

    if q_vmax <= 0.0:
        q_vmax = 1.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
    )

    plot_rgb(
        field,
        ax=axes[0],
        title="O(3) field RGB",
    )

    if show_quiver:
        plot_quiver(
            field,
            step=quiver_step,
            ax=axes[0],
            title="O(3) field RGB",
        )

    plot_density(
        e_density,
        ax=axes[1],
        title=f"energy density\nE = {energy:.6g}",
    )

    plot_density(
        q_density,
        ax=axes[2],
        title=f"topological density\nQ = {charge:.6g}",
        vmin=-q_vmax,
        vmax=q_vmax,
    )

    fig.tight_layout()

    return save_figure(
        path,
        fig=fig,
        dpi=dpi,
        close=True,
    )

__all__ = [
    "ensure_field",
    "normalize_field",
    "component_index",
    "field_to_rgb",
    "finite_difference_x",
    "finite_difference_y",
    "energy_density",
    "topological_density",
    "topological_charge",
    "plot_component",
    "plot_rgb",
    "plot_quiver",
    "plot_density",
    "plot_skyrmion",
    "plot_surface",
    "save_figure",
    "save_skyrmion_plot",
    "show",
    "show_skyrmion_plot",
    "show_field_components",
    "show_energy_density",
    "show_topological_density",
    "show_skyrmion_diagnostics",
    "save_skyrmion_diagnostics",
]
