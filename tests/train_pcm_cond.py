import pickle
import numpy as np
import time 

from rompcm.paths import DATADIR, TESTSDIR, MODELS_DIR
from rompcm.pcmcond.pod_rbf_cond import POD_RBF_Trainer


def training_mp(
    MP: bool = True,
    method: str = "POD",
):

    dt = 0.5

    t = np.linspace(2.1, 51.6, 100)

    Th = np.array([
        1.,
        1.015,
        1.03,
        1.045,
        1.06,
        1.07,
        1.08,
        1.09,
        1.1
    ])

    # =========================================================
    # Load lifting function
    # =========================================================

    g1 = np.loadtxt(TESTSDIR / "g1.txt")
    g1 = g1[None, :]

    # =========================================================
    # Load snapshots
    # =========================================================

    data = np.empty((len(Th), len(t), 14641))

    for i in range(len(Th)):

        for j in range(len(t)):

            file_path = (
                DATADIR
                / "PCMnoconv"
                / f"Th_{Th[i]}"
                / f"T.t.{t[j]}.txt"
            )

            data[i, j, :] = np.loadtxt(file_path)

        # lifting
        data[i, :, :] -= Th[i] * g1

    print(
        f"samples: {len(Th)}, "
        f"time steps: {len(t)}, "
        f"features: {data.shape[-1]}"
    )

    # =========================================================
    # ROM parameters
    # =========================================================

    TOL = 0.01
    atol = 1.e-7

    Nt = len(t)

    eps = 1.e-10
    eps_project = 1.e-7

    train_indices = [0, len(Th)-1]

    kernels = {
        "kernel": "cubic",
        "smooth": 0.0,
        "degre": None,
        "epsilon": None,
    }

    # =========================================================
    # Train ROM
    # =========================================================

    trainer = POD_RBF_Trainer(
        data[:, :Nt],
        g1,
        Th[:],
        t[:Nt],
        method,
        eps,
        eps_project,
        atol,
        kernels
    )

    if MP:

        _ = trainer.greedy_openmp(
            train_indices,
            Nt,
            dt,
            TOL,
            plot_modes=False
        )

    else:

        _ = trainer.greedy(
            train_indices,
            Nt,
            dt,
            TOL,
            plot_modes=False
        )

    # =========================================================
    # Save models
    # =========================================================

    MODELS_DIR.mkdir(exist_ok=True)

    pod_path = (
        MODELS_DIR
        / f"pod_cond_method_{method}_TOL_{TOL}.pkl"
    )

    rbf_path = (
        MODELS_DIR
        / f"rbf_cond_method_{method}_TOL_{TOL}.pkl"
    )

    metadata = {
        "method": method,
        "TOL": TOL,
        "dt": dt,
        "g1": g1
    }

    with open(pod_path, "wb") as f:

        pickle.dump(
            {
                "pod": trainer.pod,
                "metadata": metadata,
            },
            f
        )

    metadata = {
                "train_parameters": Th.tolist(),
                "train_times": t.tolist(),
                "kernel": kernels
                }

    with open(rbf_path, "wb") as f:

        pickle.dump(
            {
                "rbf": trainer.rbf,
                "metadata": metadata,
            },
            f
        )

    print(f"POD model saved in {pod_path}")
    print(f"RBF model saved in {rbf_path}")

    return trainer


if __name__ == "__main__":
    t0 = time.perf_counter()
    #test_mpi()
    training_mp(MP=True, method="POD")
    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0:.2f} s")