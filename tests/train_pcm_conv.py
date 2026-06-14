import numpy as np
import time, pickle


from rompcm.pcmconv.pod_rbf_conv import POD_RBF_Trainerfull
from rompcm.paths import DATADIR, TESTSDIR, MODELS_DIR

def test_mp(MP:str=True, method:str="POD"):
	dt=0.5; #t = np.linspace(2.1, 51.6, 100); Nt=100
	t = np.arange(2.1, 51.7, dt, dtype=float)
	#print(t)
	Nt = 50
	Th = np.array([1., 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1])
	#Th = np.array([1., 1.05, 1.1])
	TOL = .1; atol=1.e-7; eps=1.e-9; eps_project=1.e-7
	train_indices = [0, len(Th)-1] # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #

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
	
	print(f"samples: {len(Th)}, time steps: {Nt}, features: {data.shape[-1]}, eps {eps}, method {str(method)}")
	trainer = POD_RBF_Trainerfull((data[:, :, :Nt], 0), g1, Th[:], t[:Nt], method, 1, 1, 1, eps, eps_project, atol, kernels)

	if MP:
		_ = trainer.greedy_openmp(train_indices, Nt, dt, TOL, plot_modes=False)
	else:
		_ = trainer.greedy(train_indices, Nt, dt, TOL, plot_modes=False)
		

	# =========================================================
	# Save models
	# =========================================================
	MODELS_DIR.mkdir(exist_ok=True)
	pod_path = (
		MODELS_DIR
		/ f"pod_conv_method_{method}_TOL_{TOL}.pkl"
	)
	rbf_path = (
		MODELS_DIR
		/ f"rbf_conv_method_{method}_TOL_{TOL}.pkl"
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
    test_mp(MP=True, method="POD")
    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0:.2f} s")