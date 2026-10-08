"""Compare POD and GPOD (self-similar POD) on the analytical PCM front.

Panels: cumulative energy, reconstruction error vs r, first modes of each basis.

Usage:
    python scripts/compare_pod_gpod.py            # saves figures/compare_pod_gpod.png
    python scripts/compare_pod_gpod.py --show     # also opens the window
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rompcm.pod_basis import POD
from rompcm.sst_snap import SST

from rompcm.analytical import EPS, front_snapshots


# x, t, snaps = front_snapshots()

TOL = 5e-3

"""
def st(t):
    return 0.001330 + 0.039757 * np.sqrt(t)


def f(mu, x, t):
    return np.maximum(mu - x / st(t), -EPS)


def build_snapshots(mu=1.0):
    x = np.linspace(0.0, 1.0, 121)
    t = np.arange(2.1, 51.7, 0.5, dtype=float)
    snaps = np.stack([f(mu, x, tt) for tt in t])          # (Nt, Nx)
    return x, t, snaps
"""

def phi_of(X):
    pod = POD(eps=1.e-12, center=False)
    pod.fit(X[None])                                      # (ns=1, nt, ndof)
    return pod.phi                                        # (ndof, Nr)


def cumulative_energy(X):
    s = np.linalg.svd(X, compute_uv=False)
    return np.cumsum(s**2) / np.sum(s**2)


def rel_err(X, Xr):
    return np.linalg.norm(X - Xr) / np.linalg.norm(X)


def first_r(errors, tol):
    below = np.flatnonzero(errors < tol)
    return int(below[0]) + 1 if below.size else None


def oriented(phi, k):
    """Mode k with a sign convention, so figures do not flip between runs."""
    mode = phi[:, k]
    return mode * np.sign(mode[np.argmax(np.abs(mode))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmax", type=int, default=17)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", default="figures/compare_pod_gpod.png")
    args = parser.parse_args()
    if not args.show:
        plt.switch_backend("Agg")

    x, t, snaps = front_snapshots()

    # --- GPOD: map snapshots to the self-similar coordinate xi -------------
    sst = SST(eps=EPS)
    _   = sst.fit(snaps, x, t)
    zeta = x / sst.x_fit[-1]
    Xb = sst.backward(snaps, x, zeta)

    phi_p, phi_g = phi_of(snaps), phi_of(Xb)
    R = min(args.rmax, phi_p.shape[1], phi_g.shape[1])
    r = np.arange(1, R + 1)

    err_pod, err_gpod = [], []
    for k in r:
        P = phi_p[:, :k]
        err_pod.append(rel_err(snaps, (snaps @ P) @ P.T))
        G = phi_g[:, :k]
        err_gpod.append(rel_err(snaps, sst.forward((Xb @ G) @ G.T, zeta, x)))
    err_pod, err_gpod = np.array(err_pod), np.array(err_gpod)
    E_pod, E_gpod = cumulative_energy(snaps)[:R], cumulative_energy(Xb)[:R]

    # --- figure -------------------------------------------------------------
    fig, ax = plt.subplot_mosaic(
        [["energy", "error", "pod"], ["energy", "error", "gpod"]],
        figsize=(15, 7), constrained_layout=True,
    )

    ax["energy"].semilogy(r, 1 - E_pod, "o-", label="POD")
    ax["energy"].semilogy(r, 1 - E_gpod, "s-", label="GPOD")
    ax["energy"].set(xlabel="reduced dimension $r$", ylabel="$1 - E(r)$",
                     title="Discarded cumulative energy")

    ax["error"].semilogy(r, err_pod, "o-", label="POD")
    ax["error"].semilogy(r, err_gpod, "s-", label="GPOD")
    ax["error"].axhline(TOL, color="gray", ls="--", lw=1, label=f"tol = {TOL:g}")
    ax["error"].set(xlabel="reduced dimension $r$",
                    ylabel=r"$\|X - X_r\| / \|X\|$",
                    title="Reconstruction error (physical space)")

    for k in range(2):
        ax["pod"].plot(x, oriented(phi_p, k), label=f"mode {k + 1}")
        ax["gpod"].plot(zeta, oriented(phi_g, k), label=f"mode {k + 1}")
    ax["pod"].set(xlabel="$x$", title="POD modes (physical coordinate)")
    ax["gpod"].set(xlabel=r"$\xi = x / s(t)$",
                   title="GPOD modes (self-similar coordinate)")

    for a in ax.values():
        a.grid(alpha=0.2)
        a.legend(frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved {out}")

    print(f"r for error < {TOL:g}:  POD = {first_r(err_pod, TOL)}   "
          f"GPOD = {first_r(err_gpod, TOL)}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()