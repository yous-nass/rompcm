# tests/test_pod_vs_gpod.py
import numpy as np
import pytest

from rompcm.pod_basis import POD
from rompcm.sst_snap import SST

RMAX = 20


def energy(X):
    """Cumulative energy of the singular values of X, shape (Nt, ndof)."""
    s = np.linalg.svd(X, compute_uv=False)
    return np.cumsum(s**2) / np.sum(s**2)


def phi_of(X):
    pod = POD(eps=1.e-12, center=False)
    pod.fit(X[None])                 # (ns=1, nt, ndof)
    return pod.phi                   # (ndof, Nr)


def rel_err(X, Xr):
    return np.linalg.norm(X - Xr) / np.linalg.norm(X)


def first_r(errors, tol):
    """Smallest reduced dimension r (1-based) such that error < tol."""
    below = np.flatnonzero(np.asarray(errors) < tol)
    return int(below[0]) + 1 if below.size else None


@pytest.fixture(scope="module")
def ss(grid, snapshots):
    x, t = grid
    sst = SST(eps=0.01)
    sst.fit(snapshots, x, t)
    zeta = x / sst.x_fit[-1]
    Xb = sst.backward(snapshots, x, zeta)        # self-similar snapshots
    return sst, zeta, Xb


@pytest.fixture(scope="module")
def curves(grid, snapshots, ss):
    """Errors and energies for r = 1..R, POD vs GPOD, in physical space."""
    x, _ = grid
    sst, zeta, Xb = ss
    phi_p = phi_of(snapshots)
    phi_g = phi_of(Xb)
    R = min(RMAX, phi_p.shape[1], phi_g.shape[1])

    e_pod, e_gpod = [], []
    for r in range(1, R + 1):
        P = phi_p[:, :r]
        e_pod.append(rel_err(snapshots, (snapshots @ P) @ P.T))

        G = phi_g[:, :r]
        Xb_r = (Xb @ G) @ G.T                    # reconstruction in xi
        Xf_r = sst.forward(Xb_r, zeta, x)        # back to physical space
        e_gpod.append(rel_err(snapshots, Xf_r))

    return {
        "r": np.arange(1, R + 1),
        "err_pod": np.array(e_pod),
        "err_gpod": np.array(e_gpod),
        "E_pod": energy(snapshots)[:R],
        "E_gpod": energy(Xb)[:R],
    }


def test_print_table(curves):
    """Run with `pytest -s` to see the numbers used to tune the thresholds."""
    c = curves
    print("\n r | err POD   err GPOD | 1-E POD   1-E GPOD")
    for i, r in enumerate(c["r"]):
        print(f"{r:2d} | {c['err_pod'][i]:.2e} {c['err_gpod'][i]:.2e} |"
              f" {1 - c['E_pod'][i]:.2e} {1 - c['E_gpod'][i]:.2e}")



def test_energy_is_cumulative(curves):
    for key in ("E_pod", "E_gpod"):
        E = curves[key]
        assert np.all(np.diff(E) >= -1e-12)
        assert 0 < E[0] <= 1 and E[-1] <= 1 + 1e-12


def test_pod_error_matches_discarded_energy(curves):
    # Eckart-Young (no centering): relative error^2 == 1 - cumulative energy
    np.testing.assert_allclose(
        curves["err_pod"] ** 2, 1 - curves["E_pod"], rtol=1e-3
    )


def test_gpod_energy_concentrated_in_first_mode(curves):
    # measured: 1-E = 1.4e-5 (GPOD) vs 4.3e-2 (POD)
    assert (1 - curves["E_gpod"][0]) < 1e-4
    assert (1 - curves["E_pod"][0]) > 1e-2


def test_errors_decrease_with_r(curves):
    assert np.all(np.diff(curves["err_pod"]) < 0)


def test_gpod_beats_pod_at_equal_r(curves):
    for i in range(5):                            # r = 1..5
        assert curves["err_gpod"][i] < curves["err_pod"][i]


def test_gpod_needs_smaller_reduced_dimension(curves):
    tol = 5e-3
    r_pod = first_r(curves["err_pod"], tol)
    r_gpod = first_r(curves["err_gpod"], tol)
    assert r_gpod is not None and r_gpod <= 3
    assert r_pod is None or r_pod >= 3 * r_gpod