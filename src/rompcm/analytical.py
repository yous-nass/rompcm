"""Analytical self-similar PCM front, shared by tests and scripts.

Convention: snapshots have shape (Nt, Nx), or (Nt, Nx, Ny) in 2D.
"""
import numpy as np

EPS = 0.01

def st(t):
    """Interface position scale s(t)."""
    return 0.001330 + 0.039757 * np.sqrt(t)


def front(mu, x, t):
    """Temperature field, shape (Nt, Nx)."""
    x, t = np.asarray(x, float), np.atleast_1d(np.asarray(t, float))
    return np.maximum(mu - x[None, :] / st(t)[:, None], -EPS)


def front_snapshots(mu=1.0, nx=121):
    x = np.linspace(0.0, 1.0, nx)
    t = np.arange(2.1, 51.7, 0.5, dtype=float)
    return x, t, front(mu, x, t)


def front_snapshots_2d(mu=1.0, nx=121, ny=41):
    """Same 1D front copied along y in [0, 1]. Returns x, y, t, snaps (Nt, Nx, Ny)."""
    x, t, snaps = front_snapshots(mu, nx)
    y = np.linspace(0.0, 1.0, ny)
    return x, y, t, np.repeat(snaps[:, :, None], ny, axis=2)