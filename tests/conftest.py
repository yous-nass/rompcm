import pytest

from rompcm.analytical import front_snapshots


@pytest.fixture(scope="session")
def grid():
    x, t, _ = front_snapshots()
    return x, t


@pytest.fixture(scope="session")
def snapshots():
    return front_snapshots()[2]            # (Nt, Nx)

"""
from rompcm.
EPS = 0.01

def st(t):
    return 0.001330 + 0.039757 * np.sqrt(t)


def f(mu, x, t):
    return np.maximum(mu - x / st(t), -EPS)


@pytest.fixture(scope="session")
def grid():
    x = np.linspace(0.0, 1.0, 121)
    t = np.arange(2.1, 51.7, 0.5, dtype=float)
    return x, t


@pytest.fixture(scope="session")
def snapshots(grid):
    ###Physical snapshots, shape (Nx, Nt), for mu = 1.
    x, t = grid
    return np.stack([f(1.0, x, tt) for tt in t], axis=0)

"""