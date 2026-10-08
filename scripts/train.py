"""Train POD-RBF or GPOD-RBF surrogates over mu on the analytical front.

Usage:
    python scripts/train.py --method pod  --nr 10
    python scripts/train.py --method gpod --nr 3
"""
import argparse
from pathlib import Path

import numpy as np

from rompcm.analytical import EPS, front_snapshots
from rompcm.pod_basis import POD
from rompcm.sst_snap import SST

MUS = np.array([1.00, 1.02, 1.04, 1.06, 1.08, 1.10])


def rbf_inputs(mus, t):
    """Rows (mu, t), mu outer loop, shape (ns*nt, 2)."""
    return np.column_stack([np.repeat(mus, len(t)), np.tile(t, len(mus))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["pod", "gpod"], required=True)
    p.add_argument("--nr", type=int, default=10)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = Path(args.out or f"models/{args.method}_rbf.npz")

    data = [front_snapshots(mu) for mu in MUS]
    x, t = data[0][0], data[0][1]
    snaps = np.stack([d[2] for d in data])                 # (ns, nt, nx)

    extra = {}
    if args.method == "gpod":
        xfit = np.empty((len(MUS), len(t)))
        for i in range(len(MUS)):
            sst = SST(eps=EPS)
            sst.fit(snaps[i], x, t)
            xfit[i] = sst.x_fit                            # ASSUMPTION: (nt,) array
        zeta = x / xfit[:, -1].max()                       # common xi grid
        snaps = np.stack([
            _backward(snaps[i], x, t, xfit[i], zeta) for i in range(len(MUS))
        ])
        extra = {"zeta": zeta, "xfit": xfit.reshape(-1)}

    pod = POD(eps=1.e-12, center=False)
    pod.fit(snaps)                                         # (ns, nt, ndof)
    phi = pod.phi[:, :args.nr]
    theta = pod.theta[:, :, :args.nr]                      # (ns, nt, nr)
    #theta = pod.theta[..., :args.nr].reshape(-1, args.nr)  # (ns*nt, nr)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, method=args.method, x=x, t=t, mus=MUS, phi=phi,
             theta=theta, **extra)
    print(f"saved {out}  (phi {phi.shape}, theta {theta.shape})")


def _backward(snap, x, t, xfit, zeta):
    sst = SST(eps=EPS)
    sst.fit(snap, x, t)
    sst.x_fit = xfit                                       # ASSUMPTION: settable
    return sst.backward(snap, x, zeta)


if __name__ == "__main__":
    main()