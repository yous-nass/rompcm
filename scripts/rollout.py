"""Predict mu = 1.05 with a trained POD-RBF / GPOD-RBF model and compare to truth.

Usage:
    python scripts/rollout.py --model models/pod_rbf.npz
    python scripts/rollout.py --model models/gpod_rbf.npz --show
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator


from rompcm.analytical import EPS, front_snapshots
from rompcm.pod_rbf import RBF_Explicit
from rompcm.sst_snap import SST

MU_TEST = 1.05


def rbf_inputs(mus, t):
    return np.column_stack([np.repeat(mus, len(t)), np.tile(t, len(mus))])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--mu", type=float, default=MU_TEST)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if not args.show:
        plt.switch_backend("Agg")

    m = np.load(args.model, allow_pickle=False)
    method, x, t, mus, phi = str(m["method"]), m["x"], m["t"], m["mus"], m["phi"]

    pts = rbf_inputs(mus, t)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    norm = lambda a: (a - lo) / (hi - lo)
    query = norm(rbf_inputs(np.array([args.mu]), t))
    '''
    theta_pred = RBFInterpolator(norm(pts), m["theta"])(query)   # (nt, nr)
    '''
    kernels = { "kernel":"cubic","smooth":0.00000 , "degre":None, "epsilon":None}

    rbf = RBF_Explicit(kernels=kernels)
    rbf.fit(theta=m["theta"], samples_pars=mus, samples_t=t)
    theta_pred, _ = rbf.predict_theta(MU_TEST, len(t))
    print(theta_pred.shape)
    Xr = theta_pred @ phi.T                                      # (nt, ndof)

    if method == "gpod":
        xfit_pred = RBFInterpolator(norm(pts), m["xfit"])(query)
        sst = SST(eps=EPS)
        #sst.x_fit = xfit_pred                                    # ASSUMPTION
        Xr = sst.forward(Xr, m["zeta"], x, xfit_pred)

    _, _, truth = front_snapshots(args.mu)
    err_t = np.linalg.norm(truth - Xr, axis=1) / np.linalg.norm(truth, axis=1)
    print(f"{method}: mu = {args.mu}, nr = {phi.shape[1]}, "
          f"relative error = {np.linalg.norm(truth - Xr) / np.linalg.norm(truth):.2e}, "
          f"max over t = {err_t.max():.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for k in (0, len(t) // 2, -1):
        ax[0].plot(x, truth[k], "-", lw=1.5, label=f"truth t={t[k]:.1f}")
        ax[0].plot(x, Xr[k], "--", lw=1.5, label=f"{method} t={t[k]:.1f}")
    ax[0].set(xlabel="$x$", title=f"Prediction at $\\mu={args.mu}$")
    ax[1].semilogy(t, err_t)
    ax[1].set(xlabel="$t$", ylabel="relative error", title="Error vs time")
    for a in ax:
        a.grid(alpha=0.2)
    ax[0].legend(frameon=False, fontsize=8)

    out = Path(f"figures/rollout_{method}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()