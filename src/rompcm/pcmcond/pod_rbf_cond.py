import numpy as np
import sys

import os


from multiprocessing import Pool

sys.path.append("..") 

from rompcm.pod_basis import POD, IncrementalPOD
from rompcm.pod_rbf import RBF_Explicit
from rompcm.tools import estimator_cond, init_worker, greedy_chunk	


class POD_RBF_Trainer:

	def __init__(self, samples, g1, samples_pars, samples_t, method:str="POD",
				 eps_init:float=1.e-10, eps_project=1.e-8, atol:float=1.e-8,
				 kernels:dict=None, has_normalization:bool=False):
		
		self.samples = samples
		self.g1 = g1
		self.samples_pars = samples_pars
		self.samples_t = samples_t
		self.method = method
		self.kernels = kernels or {"kernel":"thin_plate_spline","smooth":0.0,"degre":1,"epsilon":1.0}

		if method == "POD":
			self.pod = POD('svd', eps_init, atol)
		elif method=="PIPOD":
			self.pod = IncrementalPOD(eps_init, eps_project, atol)
		else:
			raise ValueError(f"Method must be POD or PIPOD, got: {method}")

		self.rbf = RBF_Explicit(self.kernels, has_normalization)


	def init_POD(self, train_indices, plot_modes):
	
		if self.method == "POD":
			self.pod.fit(self.samples[train_indices], plot_modes)
		else:
			self.pod.initialize(self.samples[train_indices], plot_modes)
			
		self.rbf.fit(self.pod.theta, self.samples_pars[train_indices], self.samples_t)
		

	# ----------------- POD + RBF training -----------------
	def udate_POD(self, idx, train_indices, plot_modes):
		
		if self.method == "POD":
			self.pod.fit(self.samples[train_indices], plot_modes)
		else:
			self.pod.update(train_indices.index(idx), self.samples[train_indices], plot_modes)

		self.rbf.fit(self.pod.theta, self.samples_pars[train_indices], self.samples_t)
	

	# ----------------- serial training without greedy -------
	def pod_rbf_fit(self, train_indices, Nt_pred, dt, plot_modes=False):
		
		self.init_POD(train_indices, plot_modes)
		
		l2_err = np.zeros((len(train_indices), 3))
		
		# Parallel prediction + error computation
		for idx, i in enumerate(train_indices):
			V_r = []
			theta = (self.samples[i] - self.pod.samples_mean) @ self.pod.phi
			theta_r, l2_err[idx,-1]= self.rbf.predict_theta(self.samples_pars[i], Nt_pred, theta)
			V_r = theta_r @ self.pod.phi.T + self.pod.samples_mean
			V_r += self.samples_pars[i]*self.g1
			l2_err[idx,:-1] = estimator_cond( V_r, self.samples[i] + self.samples_pars[i]*self.g1, self.samples_pars[i], dt)
			print(f"parameters µ {self.samples_pars[i]}, Errors:: {l2_err[idx]}") 


																	
	# ----------------- Greedy iteration ----------------------
	def greedy(self, train_indices, Nt_pred, dt, tol=1., plot_modes=False):
		indices = [i for i in range(len(self.samples_pars))]
		self.init_POD(train_indices, plot_modes)
		errors = []
		cunt = 1
		while len(train_indices) <= len(self.samples_pars) :

			candidate_indices = [i for i in indices if i not in train_indices]
			l2_err = np.zeros((len(candidate_indices), 3))
			res_ = -1
			# Parallel prediction + error computation
			for idx, i in enumerate(candidate_indices):
				V_r = []
				theta = (self.samples[i] - self.pod.samples_mean) @ self.pod.phi
				theta_r, l2_err[idx,-1]= self.rbf.predict_theta(self.samples_pars[i], Nt_pred, theta)
				V_r = theta_r @ self.pod.phi.T + self.pod.samples_mean
				V_r += self.samples_pars[i]*self.g1
				l2_err[idx,:-1] = estimator_cond( V_r, self.samples[i] + self.samples_pars[i]*self.g1, self.samples_pars[i], dt)
				if res_ < l2_err[idx, 1]:
					res_ = l2_err[idx, 1]
					new_index = i
			
			print(l2_err)
			errors.append(np.max(l2_err, 0, keepdims=True))
			print(f"Greedy iter {cunt}, Modes: {self.pod.phi.shape[1]}, Errors:: {errors[-1][0,:]}") 
			if  res_ < tol:
				print("TOL error reached. Stopping.")
				break		
				
			cunt += 1
			train_indices.append(new_index)
			train_indices.sort()
			self.udate_POD(new_index, train_indices, plot_modes)
			


	# ----------------- multiprocessing-parallel Greedy iteration -----------------------
	def greedy_openmp(self, train_indices, Nt_pred, dt, tol=1., plot_modes=False, nproc=8, blas_threads=12):
		# Force deterministic BLAS per process
		os.environ["OMP_NUM_THREADS"] = str(blas_threads)
		os.environ["MKL_NUM_THREADS"] = str(blas_threads)
		os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)

		# Initialize candidate indices, POD and RBF on main process
		candidates = [i for i in range(len(self.samples_pars)) if i not in train_indices]
		self.init_POD(train_indices, plot_modes)

		iteration = 1
		while candidates:
			print(candidates)
			# Chunk candidates for workers
			chunks = np.array_split(candidates, min(nproc, len(candidates)))

			# Launch worker pool
			with Pool(processes=min(nproc, len(candidates)),
					initializer=init_worker,
					initargs=(self.samples, self.g1, self.samples_pars, self.pod, self.rbf, Nt_pred, dt)
					) as pool:

				results = pool.map(greedy_chunk, chunks)

			# Reduce errors and pick worst
			errors = np.vstack(results)
			worst_row = np.argmax(errors[:, 2])

			print(f"Greedy iter {iteration}, Modes: {self.pod.phi.shape[1]}, Errors:: {errors[worst_row, 1:]}")
			if errors[worst_row, 2] < tol:
				print("Stopping criterion reached.")
				print("Unseen parameters :: ", self.samples_pars[candidates])
				break

			iteration += 1
			# Update train indices, POD/RBF on main process only
			new_index = int(errors[worst_row, 0])
			train_indices.append(new_index)
			train_indices.sort()
			candidates = [i for i in candidates if i != new_index]
			self.udate_POD(new_index, train_indices, plot_modes)

			if len(candidates)==0:
				print("all parameters are used")
				candidates = train_indices