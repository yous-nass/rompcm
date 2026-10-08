import numpy as np
import pytest

from rompcm.analytical import front_snapshots, front_snapshots_2d
from rompcm.pod_basis import POD
# from rompcm.sst_snap import SST


def flat(S):
    return S.reshape(S.shape[0], -1)          # (Nt, Nx*Ny), y varies fastest


def phi_of(X):
    pod = POD(eps=1.e-12, center=False)
    pod.fit(X[None])                          # (ns=1, nt, ndof)
    return pod.phi


def rel_err(X, Xr):
    return np.linalg.norm(X - Xr) / np.linalg.norm(X)


def pod_error(X, r):
    P = phi_of(X)[:, :r]
    return rel_err(X, (X @ P) @ P.T)


def energy(X):
    s = np.linalg.svd(X, compute_uv=False)
    return np.cumsum(s**2) / np.sum(s**2)


@pytest.fixture(scope="module")
def d2():
    x, y, t, S = front_snapshots_2d(1.0, nx=121, ny=41)
    return x, y, t, S


@pytest.fixture(scope="module")
def d1():
    return front_snapshots(1.0, nx=121)       # x, t, snaps (Nt, Nx)


def test_snapshots_are_invariant_along_y(d2):
    _, _, _, S = d2
    assert S.shape == (100, 121, 41)
    np.testing.assert_array_equal(S, np.repeat(S[:, :, :1], S.shape[2], axis=2))


def test_pod_2d_basis_is_orthonormal(d2):
    P = phi_of(flat(d2[3]))[:, :10]
    np.testing.assert_allclose(P.T @ P, np.eye(10), atol=1e-8)


def test_energy_2d_equals_energy_1d(d1, d2):
    np.testing.assert_allclose(energy(flat(d2[3]))[:15], energy(d1[2])[:15], atol=1e-10)


@pytest.mark.parametrize("r", [1, 5, 10])
def test_pod_error_2d_equals_1d(d1, d2, r):
    np.testing.assert_allclose(pod_error(flat(d2[3]), r), pod_error(d1[2], r), rtol=1e-6)


def test_pod_modes_are_constant_along_y(d2):
    x, y, _, S = d2
    P = phi_of(flat(S))[:, :3].reshape(len(x), len(y), 3)
    assert np.ptp(P, axis=1).max() < 1e-8



# --- GPOD 2D --------------------------------------------------------------
# ADAPTER: connect to your 2D self-similar transform (see questions below).
def to_selfsimilar_2d(x, y, t, S):
    """Return Xb (Nt, Nzeta*Ny) and a function to map back to physical space."""
    raise NotImplementedError("adapt to the SST 2D interface")


@pytest.mark.skip(reason="adapter to be written")
def test_gpod_2d_one_mode_beats_pod(d2):
    x, y, t, S = d2
    Xb, back = to_selfsimilar_2d(x, y, t, S)
    P = phi_of(Xb)[:, :1]
    e_gpod = rel_err(S, back((Xb @ P) @ P.T))
    assert e_gpod < 1e-2                                  # 1D reference: 2.8e-3
    assert e_gpod < pod_error(flat(S), 1) / 20            # 1D reference: POD 2.1e-1