import numpy as np
import pytest

from rompcm.analytical import EPS
from rompcm.sst_snap   import SST


@pytest.fixture(scope="module")
def fitted(grid, snapshots):
    x, t = grid
    sst = SST(eps=EPS)
    xf = sst.fit(snapshots, x, t)
    zeta = x / xf[-1]
    return sst, xf, zeta


def test_fit_returns_x_fit(fitted):
    sst, xf, _ = fitted
    np.testing.assert_array_equal(xf, sst.x_fit)
    assert np.all(np.diff(xf) > 0)               # interface moves forward


def test_explicit_x_fit_equals_stored(grid, snapshots, fitted):
    x, _ = grid
    sst, xf, zeta = fitted
    Xb = sst.backward(snapshots, x, zeta)
    np.testing.assert_allclose(sst.backward(snapshots, x, zeta, x_fit=xf), Xb)
    np.testing.assert_allclose(sst.forward(Xb, zeta, x, x_fit=xf),
                               sst.forward(Xb, zeta, x))


def test_unfitted_instance_works_with_explicit_x_fit(grid, snapshots, fitted):
    x, _ = grid
    sst, xf, zeta = fitted
    fresh = SST(eps=EPS)                         # never fitted
    Xb = fresh.backward(snapshots, x, zeta, x_fit=xf)
    np.testing.assert_allclose(Xb, sst.backward(snapshots, x, zeta))


def test_unfitted_instance_without_x_fit_raises(grid, snapshots):
    x, _ = grid
    with pytest.raises(ValueError):
        SST(eps=EPS).backward(snapshots, x, x)


def test_round_trip(grid, snapshots, fitted):
    x, _ = grid
    sst, xf, zeta = fitted
    Xf = sst.forward(sst.backward(snapshots, x, zeta, x_fit=xf), zeta, x, x_fit=xf)
    assert np.max(abs(snapshots - Xf)) / np.max(abs(snapshots)) < 5e-3   # as in your test


def test_perturbed_x_fit_changes_prediction(grid, snapshots, fitted):
    # explicit x_fit really is used (not silently ignored)
    x, _ = grid
    sst, xf, zeta = fitted
    Xb = sst.backward(snapshots, x, zeta, x_fit=xf)
    a = sst.forward(Xb, zeta, x, x_fit=xf)
    b = sst.forward(Xb, zeta, x, x_fit=1.05 * xf)
    assert not np.allclose(a, b)