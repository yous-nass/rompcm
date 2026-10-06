import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

from scipy.interpolate import RBFInterpolator

import os

from mpi4py import MPI
from multiprocessing import Pool

sys.path.append("..") 

from rompcm.pod_basis import POD, IncrementalPOD
from rompcm.pod_rbf   import RBF_Steady, RBF_Explicit
from ..tools          import estimator_conv_temperature, init_worker, greedy_chunk_full


class POD_RBF_Trainertemperature:

	def __init__(self, samples, samples_pars, samples_t, method:str,
				 eps_init:1.e-10, eps_project=1.e-8, atol:float=1.e-8,
				 kernels:dict=None, has_normalization:bool=False):
		
		self.samples = samples
		self.samples_pars = samples_pars
		self.samples_t = samples_t
		self.method = method
		self.kernels = kernels or {"kernel":"thin_plate_spline","smooth":0.0,"degre":None,"epsilon":None}

		if method == "POD":
			self.pod = POD('svd', eps_init, atol)
		else:
			self.pod = IncrementalPOD(eps_init, eps_project, atol)

		self.model = RBF_Steady(self.kernels, has_normalization)


	def init_POD(self, train_indices, plot_modes):
	
		if self.method == "POD":
			self.pod.fit(self.samples[2, train_indices], plot_modes)
		else:
			self.pod.initialize(self.samples[2, train_indices], plot_modes)
			
		self.model.fit(self.pod.theta, self.samples_pars[train_indices], self.samples_t)
		

	def udate_POD(self, ids, train_indices, plot_modes):
		
		if self.method == "POD":
			self.pod.fit(self.samples[2, train_indices], plot_modes)
		else:
			self.pod.update(train_indices.index(ids), self.samples[2, train_indices], plot_modes)
			
		self.model.fit(self.pod.theta, self.samples_pars[train_indices], self.samples_t)
		
		

	# ----------------- MPI-parallel Greedy selection -----------------
	def mpi_greedy(self, train_indices, Nt_pred, dt, tol=1., plot_modes=True):
	   
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		size = comm.Get_size()

		# Split candidates
		candidate_indices = [i for i in range(len(self.samples_pars))] 
		indices_per_rank = candidate_indices[rank::size]
		l2_err_partial = np.zeros((len(indices_per_rank), 3))
		
		# initialize basis
		if rank == 0:
			cunt = 1
			greedy_errors = []
			ordered_errors = np.zeros((len(self.samples_pars), 3))
			self.init_POD(train_indices, plot_modes)

		while len(train_indices) <= len(self.samples_pars) :

			# Broadcast POD and model data
			pod_data = comm.bcast( {"S":self.pod.S, "phi": self.pod.phi, "samples_mean": self.pod.samples_mean} 
								if rank==0 else None, root=0
								)

			model_data = comm.bcast({"a": self.model.a, "b": self.model.b, "mu_t": self.model.mu_t, "theta": self.model.theta,
									"nr": self.model.nr, "t": self.model.t, "kernels": self.model.kernels,
									"has_normalization": self.model.has_normalization}
									if rank==0 else None, root=0
									)

			# Rebuild POD and RBF objects
			self.pod.S = pod_data["S"]
			self.pod.phi = pod_data["phi"]
			self.pod.samples_mean = pod_data["samples_mean"]
	
			md = model_data
			m = RBF_Steady(md["kernels"], has_normalization=md["has_normalization"])
			m.a, m.b, m.mu_t, m.theta = md["a"], md["b"], md["mu_t"], md["theta"]
			m.nr, m.t = md["nr"], md["t"]
			m.rbf = [
				RBFInterpolator(y=m.mu_t, d=m.theta[:, r],
								kernel=md["kernels"]["kernel"],
								smoothing=md["kernels"].get("smooth",0.0),
								degree=md["kernels"].get("degre",None),
								epsilon=md["kernels"].get("epsilon",None))
				for r in range(m.nr)
			]
			self.model = m

			# Parallel prediction + error computation
			for idx, i in enumerate(indices_per_rank):
				V_r = []
				theta = (self.samples[2, i] - self.pod.samples_mean) @ self.pod.phi
				theta_r, l2_err_partial[idx,-1]= self.model.predict_theta(self.samples_pars[i], Nt_pred, theta)
				l2_err_partial[idx,-1] /= np.sqrt(np.mean(theta**2))
				V_r = theta_r @ self.pod.phi.T + self.pod.samples_mean

				l2_err_partial[idx,:-1] = estimator_conv_temperature(
																	V_r, self.samples[2,i], self.samples[0,i],
																	self.samples[1,i], self.samples_pars[i], dt
																	)

			# Gather errors
			all_indices = comm.gather(indices_per_rank, root=0)
			all_errors = comm.gather(l2_err_partial[:], root=0)
			
			new_index = None
			if rank==0:
				global_indices = np.concatenate(all_indices)      
				global_errors  = np.vstack(all_errors)            
				# Fill in the correct mapping
				for idx, err in zip(global_indices, global_errors):
					ordered_errors[idx] = err.copy()
				new_index = np.argmax(ordered_errors[:,1])  
				
				# plot history
				if False:
					plt.figure()
					plt.semilogy(self.samples_pars, ordered_errors[:,0], '*-', label="$\mathcal{E}_2$")
					plt.semilogy(self.samples_pars, ordered_errors[:,1], 'o-', label="$\mathcal{R}_2$")
					plt.semilogy(self.samples_pars, ordered_errors[:,2], 'v-', label="$\mathcal{I}_2$ ") 
					plt.legend()
					plt.grid(True)
					plt.xlabel("Parameter μ")
					plt.ylabel("Error")
					plt.show(block=False)
					plt.pause(10)
					plt.close()
					
				greedy_errors.append(np.max(ordered_errors, 0, keepdims=True))
				print(f"\nGreedy iter {cunt}, method: {self.method}, modes: {self.model.nr}, errors={greedy_errors[-1]}")

				# Stop criterion
				if ordered_errors[new_index,1] < tol:
					print("Stopping criterion reached.")
					savename = "pod_conv_temp_method_{}_TOL_{}.pkl".format(self.method, tol)
					with open(savename, "wb") as f:
						pickle.dump(self.pod, f)
					print('POD module is saved in file ', savename)
					savename = "rbf_conv_temp_method_{}_TOL_{}.pkl".format(self.method, tol)
					with open(savename, "wb") as f:
						pickle.dump(self.model, f)
					print('RBF module is saved in file ', savename)
					break
				
				cunt += 1

				# Broadcast selected parameter
				train_indices.append(new_index)
				train_indices.sort()
				#train_indices = np.sort([*train_indices, new_index])
				self.udate_POD(new_index, train_indices, plot_modes)

		if rank==0:
			iters = range(1, len(greedy_errors)+1)
			greedy_errors = np.squeeze(np.array(greedy_errors))
			
			plt.figure()
			plt.semilogy(iters, greedy_errors[:,0], marker='o', label="$\mathcal{E}_2$")
			plt.semilogy(iters, greedy_errors[:,1], marker='*', label="$\mathcal{R}_2$")
			plt.semilogy(iters, greedy_errors[:,2], marker='x', label="$\mathcal{I}_2$")
			plt.xlabel("Greedy Iteration")
			plt.ylabel("Error")
			plt.title("Greedy Error Decay")
			plt.legend()
			plt.grid(True)
			plt.show()
			return train_indices
		else:
			return None




class POD_RBF_Trainerfull:
	def __init__(self, samples, g1, samples_pars, samples_t,  method,
				 Uscal=1, Vscal=1, Pscal=1,
				 eps_init=1.e-10, eps_project=1.e-8, atol=1.e-8,
				 kernels=None, has_normalization=False):
	
		self.samples = samples[0]
		self.g1 = g1
		self.psamples = samples[1]
		self.samples_pars = samples_pars
		self.samples_mean = []
		self.samples_t = samples_t
		self.var = 3
		self.kernels = kernels or {"kernel":"thin_plate_spline","smooth":0.0,"degre":None,"epsilon":None}
		self.Uscal = Uscal
		self.Vscal = Vscal
		self.Pscal = Pscal
		self.method = method

		if method == "POD":
			self.pod = [POD('svd', eps_init, atol) for _ in range(self.var)]
		else:
			self.pod = [IncrementalPOD(eps_init, eps_project, atol) for _ in range(self.var)]

		self.rbf = [RBF_Explicit(self.kernels, has_normalization) for _ in range(self.var)]
	

	def init_POD_RBF(self, train_indices, plot_modes):
	
		if self.method == "POD":
			for i in range(self.var):
				self.pod[i].fit(self.samples[i, train_indices], plot_modes)
				self.rbf[i].fit(self.pod[i].theta, self.samples_pars[train_indices], self.samples_t)
		else:
			for i in range(self.var):
				self.pod[i].initialize(self.samples[i, train_indices], plot_modes)
				self.rbf[i].fit(self.pod[i].theta, self.samples_pars[train_indices], self.samples_t)


	def udate_POD_RBF(self, ids, train_indices, plot_modes):
		
		if self.method == "POD":
			for i in range(self.var):
				self.pod[i].fit(self.samples[i, train_indices], plot_modes)
				self.rbf[i].fit(self.pod[i].theta, self.samples_pars[train_indices], self.samples_t)
		else:
			for i in range(self.var):
				self.pod[i].update(train_indices.index(ids), self.samples[i, train_indices], plot_modes)
				self.rbf[i].fit(self.pod[i].theta, self.samples_pars[train_indices], self.samples_t)



	def pod_rbf_fit(self, train_indices, Nt_pred, dt, plot_modes=True):
	
		self.init_POD_RBF(train_indices, plot_modes)
		l2_err = np.zeros((len(train_indices), 3))
		for idx, i in enumerate(train_indices):
			V_r = []
			for j in range(self.var):
				theta = (self.samples[j, i] - self.pod[j].samples_mean) @ self.pod[j].phi
				theta_r, l2_err[idx, -1]= self.rbf[j].predict_theta(self.samples_pars[i], Nt_pred, theta)
				V_r.append(theta_r @ self.pod[j].phi.T + self.pod[j].samples_mean)
			V_r[-1] += self.samples_pars[i]*self.g1

			l2_err[idx, :-1] = estimator_conv_temperature(
														V_r[2], self.samples[2,i] + self.samples_pars[i]*self.g1, 
														V_r[0]*self.Uscal, V_r[1]*self.Vscal, self.samples_pars[i], dt
														)
															
			print(f"Modes: {self.pod[0].phi.shape[1], self.pod[1].phi.shape[1], self.pod[2].phi.shape[1]}, Errors:: {l2_err[idx]}") 
			
	# ----------------- Greedy selection -----------------
	def greedy(self, train_indices, Nt_pred, dt, tol=1., plot_modes=True):
		indices = [i for i in range(len(self.samples_pars))]
		errors = []
		cunt = 1
		candidates = [i for i in indices if i not in train_indices]
		self.init_POD_RBF(train_indices, False)
		
		while candidates:
			
			l2_err = np.zeros((len(candidates), 3))
			res_ = -1

			for idx, i in enumerate(candidates):
				V_r = []
				for j in range(self.var):
					theta = (self.samples[j, i] - self.pod[j].samples_mean) @ self.pod[j].phi
					theta_r, l2_err[idx, -1]= self.rbf[j].predict_theta(self.samples_pars[i], Nt_pred, theta)
					V_r.append(theta_r @ self.pod[j].phi.T + self.pod[j].samples_mean)
				V_r[-1] += self.samples_pars[i]*self.g1

				l2_err[idx, :-1] = estimator_conv_temperature(
															V_r[2], self.samples[2,i]+ self.samples_pars[i]*self.g1,
															V_r[0]*self.Uscal, V_r[1]*self.Vscal, self.samples_pars[i], dt
															)
				if res_ < l2_err[idx, 1]:
					res_ = l2_err[idx, 1]
					new_index = i
			
			errors.append(np.max(l2_err, 0, keepdims=True))
			print(f"Greedy iter {cunt}, Modes: {self.pod[0].phi.shape[1], self.pod[1].phi.shape[1], self.pod[2].phi.shape[1]}, Errors:: {errors[-1][0,:]}") 
			if  res_ < tol:
				print("TOL error reached. Stopping.")
				break		
				
			cunt += 1
			train_indices.append(new_index)
			train_indices.sort()
			candidates = [i for i in candidates if i != new_index]
			self.udate_POD_RBF(new_index, train_indices, plot_modes)
			if len(candidates)==0:
				print("all parameters are used")
				candidates = train_indices

	# ----------------- multiprocessing-parallel Greedy iteration -----------------------
	def greedy_openmp(self, train_indices, Nt_pred, dt, tol=1., plot_modes=False, nproc=8, blas_threads=12):
		# Force deterministic BLAS per process
		os.environ["OMP_NUM_THREADS"] = str(blas_threads)
		os.environ["MKL_NUM_THREADS"] = str(blas_threads)
		os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)

		# Initialize candidate indices, POD and RBF on main process
		candidates = [i for i in range(len(self.samples_pars)) if i not in train_indices]
		self.init_POD_RBF(train_indices, plot_modes)

		iteration = 1
		while candidates:
			# Chunk candidates for workers
			chunks = np.array_split(candidates, min(nproc, len(candidates)))

			# Launch worker pool
			with Pool(processes=min(nproc, len(candidates)),
					initializer=init_worker,
					initargs=(self.samples, self.g1, self.samples_pars, self.pod, self.rbf, Nt_pred, dt)
					) as pool:

				results = pool.map(greedy_chunk_full, chunks)

			# Reduce errors and pick worst
			errors = np.vstack(results)
			worst_row = np.argmax(errors[:, 2])

			print(f"Greedy iter {iteration}, Modes: {self.pod[0].phi.shape[1], self.pod[1].phi.shape[1], self.pod[2].phi.shape[1]}, Errors:: {errors[worst_row, 1:]}") 
			if errors[worst_row, 2] < tol:
				print("Stopping criterion reached.")
				break

			iteration += 1
			# Update train indices, POD/RBF on main process only
			new_index = int(errors[worst_row, 0])
			train_indices.append(new_index)
			train_indices.sort()
			candidates = [i for i in candidates if i != new_index]
			self.udate_POD_RBF(new_index, train_indices, plot_modes)

			if len(candidates)==0:
				print("all parameters are used")
				candidates = train_indices

"""
	def sync_POD_RBF(self, comm, rank):
		
		# Broadcast POD and model data
		pod_data = comm.bcast([{"S":self.pod[j].S, "phi": self.pod[j].phi, "samples_mean": self.pod[j].samples_mean}
								for j in range(self.var)
								] if rank==0 else None, root=0)
	
		model_data = comm.bcast([{"a": self.model[j].a, "b": self.model[j].b, "mu_t": self.model[j].mu_t, "theta": self.model[j].theta,
								"nr": self.model[j].nr, "t": self.model[j].t, "kernels": self.model[j].kernels,
								"has_normalization": self.model[j].has_normalization}
								for j in range(self.var)
								] if rank==0 else None, root=0)

		# Rebuild POD and RBF objects
		for j in range(self.var):
			self.pod[j].S = pod_data[j]["S"]
			self.pod[j].phi = pod_data[j]["phi"]
			self.pod[j].samples_mean = pod_data[j]["samples_mean"]
		
			md = model_data[j]
			m = RBF_Steady(md["kernels"], has_normalization=md["has_normalization"])
			m.a, m.b, m.mu_t, m.theta = md["a"], md["b"], md["mu_t"], md["theta"]
			m.nr, m.t = md["nr"], md["t"]
			m.rbf = [RBFInterpolator(y=m.mu_t, d=m.theta[:, r],
									kernel=md["kernels"]["kernel"],
									smoothing=md["kernels"].get("smooth",0.0),
									degree=md["kernels"].get("degre",None),
									epsilon=md["kernels"].get("epsilon", None)
									) for r in range(m.nr)
					]
			self.model[j] = m


	# ----------------- MPI-parallel Greedy selection -----------------
	def mpi_greedy(self, train_indices, Nt_pred, dt, tol=1., plot_modes=True):
		
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		size = comm.Get_size()

		# Split candidates
		candidate_indices = [i for i in range(len(self.samples_pars))] 
		indices_per_rank = candidate_indices[rank::size]
		l2_err_partial = np.zeros((len(indices_per_rank), 3))

		if rank==0:
			cunt = 1
			#train_ind = train_indices
			greedy_errors = []
			ordered_errors = np.zeros((len(self.samples_pars), 3)) 
			self.init_POD_RBF(train_indices, False)
	
		
		while len(train_indices) <= len(self.samples_pars):
			#train_ind = comm.bcast(train_ind, root=0)
			#candidate_indices = [i for i in range(len(self.samples_pars)) if i not in train_ind]
			
			#indices_per_rank = candidate_indices[rank::size]
			#l2_err_partial = np.zeros((len(indices_per_rank), 3))
			self.sync_POD_RBF(comm, rank)

			# Parallel prediction + error computation
			err = 0
			for idx, i in enumerate(indices_per_rank):
				V_r = []
				for j in range(self.var):
					theta = (self.samples[j, i] - self.pod[j].samples_mean) @ self.pod[j].phi
					theta_r, err= self.model[j].predict_theta(self.samples_pars[i], Nt_pred, theta)
					V_r.append(theta_r @ self.pod[j].phi.T + self.pod[j].samples_mean)
				
				l2_err_partial[idx, -1] = err/np.sqrt(np.mean(theta**2))
				l2_err_partial[idx, :-1] = estimator_conv_temperature(
																	V_r[2], self.samples[2,i], V_r[0]*self.Uscal, 
																	V_r[1]*self.Vscal, self.samples_pars[i], dt
																	)

			# Gather errors
			all_indices = comm.gather(indices_per_rank, root=0)
			all_errors  = comm.gather(l2_err_partial, root=0)
			

			if rank==0:
				global_indices = np.concatenate(all_indices)     
				global_errors  = np.vstack(all_errors)    
			  
				# Fill in the correct mapping
				for i, err in zip(global_indices, global_errors):
					ordered_errors[i] = err.copy()
				
				new_index = np.argmax(ordered_errors[:,1])  
				#greedy_indicator = np.max(ordered_errors, axis=1)
				# Enforce exclusion of training points
				#greedy_indicator[train_indices] = -np.inf
				#new_index = np.argmax(greedy_indicator)

				# plot history
				if False:
					fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
					variable_names = [r"$\mathcal{E}_2$", r"$\mathcal{R}_2$", r"$\mathcal{I}_2$"]
					for i in range(3):
						ax = axes[i]
						ax.semilogy(self.samples_pars, ordered_errors[:, 0, i], '*-', label="u")
						ax.semilogy(self.samples_pars, ordered_errors[:, 1, i], 'o-', label="v")
						ax.semilogy(self.samples_pars, ordered_errors[:, 2, i], 'v-', label="T")
						
						ax.set_title(f"{variable_names[i]}")
						ax.set_xlabel(r"Parameter $\mu$")
						ax.grid(True)
						if i == 0:
							ax.set_ylabel("Error")
						ax.legend()
					plt.tight_layout()
					plt.show()
				
				greedy_errors.append(np.max(ordered_errors, 0, keepdims=True))
				print(f"\nGreedy iter {cunt}, modes: {self.model[0].nr}, {self.model[1].nr}, {self.model[-1].nr}, errors={greedy_errors[-1][0]}")

				# Stop criterion
				if ordered_errors[new_index, 1] <= tol:
					print(f"Stopping criterion reached: {ordered_errors[new_index,1]}")
					savename = "pod_conv_method_{}_TOL_{}.pkl".format(self.method, tol)
					with open(savename, "wb") as f:
						pickle.dump(self.pod, f)
					print('POD module is saved in file ', savename)
					savename = "rbf_conv_method_{}_TOL_{}.pkl".format(self.method, tol)
					with open(savename, "wb") as f:
						pickle.dump(self.model, f)
					print('RBF module is saved in file ', savename)
					break
				
				cunt += 1
				train_indices.append(new_index)
				train_indices.sort()
				self.udate_POD_RBF(new_index, train_indices, plot_modes)

			

		if rank==0:
			
			iters = range(1, len(greedy_errors)+1)
			greedy_errors = np.squeeze(np.array(greedy_errors))
			fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
			fig.suptitle("POD-RBF Greedy Error Convergence", fontsize=16)
			variable_names = ["$\mathcal{E}_2$", "$\mathcal{R}_2$", "$\mathcal{I}_2$"]
			for i in range(3):
				ax = axes[i]
				ax.semilogy(iters, greedy_errors[:, 0, i], '*-', label="u")
				ax.semilogy(iters, greedy_errors[:, 1, i], 'o-', label="v")
				ax.semilogy(iters, greedy_errors[:, 2, i], 'v-', label="T")
				ax.semilogy(iters, greedy_errors[:, 3, i], 'x-', label="P")
				
				ax.set_title(f"{variable_names[i]}")
				plt.xlabel("Greedy iteration")
				ax.grid(True)
				#if i == 0:
				ax.set_ylabel("Error")
				ax.legend()

			plt.tight_layout()
			plt.show()
			
			return train_indices
		else:
			return None
"""

	

	
		
		 
