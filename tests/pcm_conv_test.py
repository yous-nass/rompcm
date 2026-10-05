import numpy as np
import time

from rompcm.pcmconv.pod_rbf_conv import POD_RBF_Trainerfull
from rompcm.paths import TESTSDIR, DATADIR


def test_mp(MP:str=True):
	dt=0.5; 
	t = np.arange(2.1, 51.7, dt, dtype=float)
	Nt = 50
	Th = np.array([1., 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1])
	TOL = .1; atol=1.e-7; eps=1.e-9; eps_project=1.e-7
	train_indices = [0, len(Th)-1] 

	g1 = np.empty(14641)
	g1 = np.loadtxt(TESTSDIR / "g1.txt")
	g1 = g1[None,:]

	data = np.empty((3, len(Th), Nt, 14641))
	for i in range(len(Th)):
		for j in range(Nt):
			file_path = (DATADIR / "PCMconv" / f"Th_{Th[i]}"/ f"UVT.t.{t[j]}.txt")
			data[:, i, j, :] = np.loadtxt(file_path, unpack=True)
		data[-1,i,:,:] -= Th[i]*g1 
	
	kernels = { "kernel":"thin_plate_spline","smooth":0.00000 , "degre":None, "epsilon":None}
	method = "POD"
	print(f"samples: {len(Th)}, time steps: {Nt}, features: {data.shape[-1]}, eps {eps}, method {str(method)}")
	trainer = POD_RBF_Trainerfull((data[:, :, :Nt], 0), g1, Th[:], t[:Nt], method, 1, 1, 1, eps, eps_project, atol, kernels)

	if MP:
		_ = trainer.greedy_openmp(train_indices, Nt, dt, TOL, plot_modes=False)
	else:
		_ = trainer.greedy(train_indices, Nt, dt, TOL, plot_modes=False)
		


if __name__ == "__main__":
    t0 = time.perf_counter()
    #test_mpi()
    test_mp(MP=False)
    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0:.2f} s")