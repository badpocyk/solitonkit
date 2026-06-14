import numpy as np

from solitonkit.viz import plot_skyrmion, plot_surface, show


def make_test_skyrmion(size: int = 101, radius: float = 12.0):
    y, x = np.mgrid[0:size, 0:size]

    cx = 0.5 * (size - 1)
    cy = 0.5 * (size - 1)

    X = x - cx
    Y = y - cy

    r = np.sqrt(X * X + Y * Y)
    phi = np.arctan2(Y, X)

    f = 2.0 * np.arctan2(radius, r + 1e-12)

    nx = np.sin(f) * np.cos(phi)
    ny = np.sin(f) * np.sin(phi)
    nz = np.cos(f)

    return np.stack([nx, ny, nz], axis=-1)


field = make_test_skyrmion()

plot_skyrmion(field, quiver_step=6)
plot_surface(field, component="z")

show()