# Two options are avaible to run the script:
# -- python file.py for multiprocessing 
# -- mpiexec -n 8  python file.py for MPI



import numpy as np
from rompcm.pcmcond.pod_rbf_cond import POD_RBF_Trainer
from rompcm.paths import DATADIR, TESTSDIR
import time
# from mpi4py import MPI



def training_mp(OPENMP:str=True, method:str="POD"):
    dt = 0.5
    t = np.linspace(2.1, 51.6, 100)

    Th = np.array([1., 1.015,  1.03, 1.045,  1.06,  1.07, 1.08, 1.09, 1.1])

    g1 = np.empty(14641)
    g1 = np.loadtxt(TESTSDIR/"g1.txt")
    g1 = g1[None,:]

    data = np.empty((len(Th), len(t), 14641))
    for i in range(len(Th)):
        for j in range(len(t)):
            file_path = (DATADIR / "PCMnoconv" / f"Th_{Th[i]}"/ f"T.t.{t[j]}.txt")
            data[i, j, :] = np.loadtxt(file_path, unpack=True)
        data[i,:,:] -= Th[i]*g1  # lifting on left wall to get homogeounous Direchlet BC

    print(f"samples: {len(Th)}, time steps: {len(t)}, features: {data.shape[-1]}")
    TOL=.01; atol=1.e-7; Nt=len(t); eps=1.e-10; eps_project=1.e-7; 
    train_indices = [0, len(Th)-1]
    kernels = { "kernel":"cubic","smooth":0.00000 , "degre":None, "epsilon":None}
    
    trainer = POD_RBF_Trainer(data[:, :Nt], g1, Th[:], t[:Nt], method, eps, eps_project, atol, kernels)

    if OPENMP:
        _ = trainer.greedy_openmp(train_indices, Nt, dt, TOL, plot_modes=False)
    else:
        _ = trainer.greedy(train_indices, Nt, dt, TOL, plot_modes=False)


"""
def test_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #start_time = MPI.Wtime()

    if rank==0:
        dt = 0.5
        t = np.linspace(2.1, 51.6, 100)

        #Th = np.array([1., 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05, 1.055,1.06, 1.065, 1.07, 1.075, 1.08, 1.085, 1.09, 1.095, 1.1])
        Th = np.array([1., 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1])
        #Th = np.array([1., 1.02,  1.04,  1.06,  1.08, 1.1])

        data = np.empty((len(Th), len(t), 14641))
        for i in range(len(Th)):
            for j in range(len(t)):
                data[i,j,:]= np.loadtxt("../data/PCMnoconv/Th_{}/T.t.{:1}.txt".format(Th[i], t[j]), unpack=True)

        print(f"samples: {len(Th)}, time steps: {len(t)}, features: {data.shape[-1]}")
        TOL=.01; atol=1.e-7; Nt=len(t); eps=1.e-10; eps_project=1.e-7; train_indices = [0, len(Th)-1]
        kernels = { "kernel":"cubic","smooth":0.00000 , "degre":None, "epsilon":None}
        method = "POD"
    else:
        dt = None; t=None; Th=None; data=None; Nt=None; eps=None; eps_project=None
        TOL=None; atol=None; train_indices=None; method=None; kernels=None


    dt = comm.bcast(dt, root=0)
    t  = comm.bcast(t,  root=0)
    Th = comm.bcast(Th, root=0)
    data = comm.bcast(data, root=0)
    eps = comm.bcast(eps, root=0)
    eps_project = comm.bcast(eps_project, root=0)
    Nt = comm.bcast(Nt, root=0)
    TOL = comm.bcast(TOL, root=0)
    atol = comm.bcast(atol, root=0)
    train_indices = comm.bcast(train_indices, root=0)
    method = comm.bcast(method, root=0)
    kernels = comm.bcast(kernels, root=0)

    trainer = POD_RBF_Trainer(data[:, :Nt], Th[:], t[:Nt], method, eps, eps_project, atol, kernels)

    _ = trainer.greedy_mpi(train_indices, Nt, dt, TOL, plot_modes=False)

    #end_time = MPI.Wtime()
    #elapsed = end_time - start_time
    #print(f"Rank {rank} execution time: {elapsed:.4f} s")
    #max_time = comm.reduce(elapsed, op=MPI.MAX, root=0)
    #if rank == 0:
    #    print(f"Total execution time (max across ranks): {max_time:.4f} s")
"""

if __name__ == "__main__":
    t0 = time.perf_counter()
    #test_mpi()
    training_mp(MP=True, method="POD")
    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0:.2f} s")

   


