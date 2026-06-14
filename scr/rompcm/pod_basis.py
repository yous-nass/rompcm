import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class POD:

	def __init__(self, method:str="svd", eps:float=1.e-7, atol:float=1.e-8):
		
		self.method = method
		self.eps = eps
		self.atol = atol

	def fit(self, samples:np.ndarray, plot_modes:bool=False):
		assert len(samples.shape) ==3, "the size of samples must be at least 3: (parameter, time, space)"
		ns, nt, nh = samples.shape
		
		samples_ = samples.reshape(ns*nt, nh)
		self.samples_mean = np.mean(samples_, 0, keepdims=True) 
		samples_ = samples_ - self.samples_mean

		if self.method=="svd":
			v, S, _ = LA.svd(samples_ , full_matrices=False) 
			S = S**2
		else:
			C = samples_ @ samples_.T
			S, v = LA.eigh(C)
			idx = np.argsort(S)[::-1]
			S = abs(S[idx])
			v = v[:, idx]
		
		# Fix reduced dimension using energy ratio
		cum_energy = np.cumsum(S) /  np.sum(S)
		Nr = np.searchsorted(cum_energy, 1 - self.eps) + 1
		if plot_modes:
			plt.semilogy(S**.5, '*', label="Singular values")
			#plt.semilogy(cum_energy, 'v', label="energy ratio")
			plt.legend()
			plt.grid()
			plt.show()
			plt.pause(10)
			plt.close()
		
		# Reduced basis and modes
		self.phi = samples_.T @ v[:, :Nr] / S[None,:Nr]**.5 
		self.S = S**0.5
		err = np.linalg.norm(self.phi.T @ self.phi - np.eye(Nr))
		assert err <= self.atol, f"orthogonality error = {err}"
		# POD coefficient of shape (ns, nt, nr)	
		self.theta = samples_ @ self.phi
		self.theta = self.theta.reshape(ns, nt, Nr)
		#self.nr = Nr
		return self
	


class IncrementalPOD:
	
	def __init__(self, eps:1.e-10, eps_project=1.e-8, atol:float=1.e-8):
		self.eps = eps
		self.eps_proj = eps_project
		self.atol = atol
	

	def initialize(self, snapshots:np.ndarray, plot_modes=False):
		assert len(snapshots.shape) ==3, "the size of snapshots must be at least 3: (parameter, time, space)"
		ns, nt, nh = snapshots.shape
		X = snapshots.reshape(ns*nt, nh)
		self.samples_mean = np.mean(X, 0, keepdims=True)
		X -= self.samples_mean

		# compute SVD
		v, S, _ = np.linalg.svd(X, full_matrices=False) 
		cum_energy = np.cumsum(S**2) /  np.sum(S**2)
		Nr = np.searchsorted(cum_energy, 1 - self.eps) + 1
		if plot_modes:
			plt.semilogy(S, '*', label="Singular values")
			plt.legend()
			plt.grid()
			plt.show()

		self.phi = X.T @ v[:, :Nr] / S[None,:Nr]
		self.S = S[:Nr]
		self.theta = (X @ self.phi).reshape(ns, nt, Nr)
		return self


	def project(self, snapshot:np.ndarray, plot_modes=False):
		p = self.phi.T @ snapshot
		q = snapshot - self.phi @ p
		q_norm = np.linalg.norm(q)
		r = self.phi.shape[1]
		snap_norm = np.linalg.norm(snapshot)
		if q_norm < self.eps_proj * snap_norm:
			return  self 
		
		# ---- Build small K matrix ----
		K = np.zeros((r+1, r+1))
		K[:r, :r] = np.diag(self.S)
		K[:r, r] = p
		K[r, r] = q_norm

		# ---- SVD of (r+1)x(r+1) ----
		Ut, St, _ = np.linalg.svd(K, full_matrices=False)

		if plot_modes:
			plt.semilogy(St, '*', label="Singular values")
			plt.legend()
			plt.grid()
			plt.show()

		# ---- Truncate small singular values ----
		cum_energy = np.cumsum(St**2) / np.sum(St**2)
		r = np.searchsorted(cum_energy, 1 - self.eps) + 1

		# ---- Update basis ----
		q /= q_norm
		phi_aug = np.column_stack([self.phi, q])
		self.phi = phi_aug @ Ut[:, :r]
		self.S = St[:r]

		# Enforce orthogonality
		self.phi, _ = np.linalg.qr(self.phi)
		err = np.linalg.norm(self.phi.T @ self.phi - np.eye(r))
		assert err <= self.atol, f"orthogonality error = {err}"
		return self
	
	def update(self, ids:int, snapshots:np.ndarray, plot_modes=False):
		#assert len(snapshots.shape) ==3, "the size of snapshots must be at least 3: (parameter, time, space)"
		ns, nt, nh = snapshots.shape
		for t in range(nt):
			snap = snapshots[ids, t] - self.samples_mean[0]
			self.project(snap, plot_modes)
		
		X = snapshots.reshape(ns * nt, nh) - self.samples_mean
		self.theta = (X @ self.phi).reshape(ns, nt, self.phi.shape[1])
		return self



class ReducedPOD:
	def __init__(self, eps:1.e-10, eps_project=1.e-8, atol:float=1.e-8):
		self.eps = eps
		self.eps_proj = eps_project
		self.atol = atol
	

	def initialize(self, snapshots:np.ndarray, plot_modes=False):
		assert len(snapshots.shape) ==3, "the size of snapshots must be at least 3: (parameter, time, space)"
		ns, nt, nh = snapshots.shape
		X = snapshots.reshape(ns*nt, nh)
		self.samples_mean = np.mean(X, 0, keepdims=True)
		X -= self.samples_mean

		# compute SVD
		v, S, _ = np.linalg.svd(X, full_matrices=False) 
		cum_energy = np.cumsum(S**2) /  np.sum(S**2)
		Nr = np.searchsorted(cum_energy, 1 - self.eps) + 1
		if plot_modes:
			plt.semilogy(S, '*', label="Singular values")
			plt.legend()
			plt.grid()
			plt.show()

		self.phi = X.T @ v[:, :Nr] / S[None,:Nr]
		self.S = S[:Nr]
		self.theta = (X @ self.phi).reshape(ns, nt, Nr)
		return self


	def project(self, snapshot:np.ndarray, plot_modes=False):
		q = self.phi.T @ snapshot
		q = snapshot - self.phi @ q
		q_norm = np.linalg.norm(q)
		r = self.phi.shape[1]
		snap_norm = np.linalg.norm(snapshot)
		if q_norm < self.eps_proj * snap_norm:
			return  self 
		
		# ---- Build small K matrix and did SVD----
		K = np.column_stack([self.phi, snapshot])
		Ut, St, _ = np.linalg.svd(K, full_matrices=False)

		if plot_modes:
			plt.semilogy(St, '*', label="Singular values")
			plt.legend()
			plt.grid()
			plt.show()

		# ---- Truncate small singular values ----
		cum_energy = np.cumsum(St**2) / np.sum(St**2)
		r = np.searchsorted(cum_energy, 1 - self.eps) + 1

		# ---- Update basis ----
		self.phi = Ut[:, :r]
		self.S = St[:r]

		# Check orthogonality
		err = np.linalg.norm(self.phi.T @ self.phi - np.eye(r))
		assert err <= self.atol, f"orthogonality error = {err}"
		return self
	
	def update(self, ids:int, snapshots:np.ndarray, plot_modes=False):
		#assert len(snapshots.shape) ==3, "the size of snapshots must be at least 3: (parameter, time, space)"
		ns, nt, nh = snapshots.shape
		for t in range(nt):
			snap = snapshots[ids, t] - self.samples_mean[0]
			self.project(snap, plot_modes)
		
		X = snapshots.reshape(ns * nt, nh) - self.samples_mean
		self.theta = (X @ self.phi).reshape(ns, nt, self.phi.shape[1])
		return self
	


class IncrementalHAPOD:
	def __init__(self, eps=1.e-8, atol:float=1.e-8):
		"""
		eps_local  : tolerance for local POD (new parameter)
		eps_global : tolerance for global recompression
		"""
		self.eps = eps
		self.atol = atol
		self.phi = None

	def fit(self, snapshots:np.ndarray, plot_modes):
		ns = snapshots.shape[0]
	
		v, S, _ = LA.svd(snapshots / ns**0.5, full_matrices=False) 
		S = S**2
		#assert np.allclose( v.T @ v, np.identity(ns)), f"orthogonality fails "
		
		cum_energy = np.cumsum(S) /  np.sum(S)
		Nr = np.searchsorted(cum_energy, 1 - self.eps) + 1
		if plot_modes:
			plt.semilogy(S**.5, '*', label="Singular values")
			#plt.semilogy(cum_energy, 'v', label="energy ratio")
			plt.legend()
			plt.grid()
			plt.show()
			#plt.pause(10)
			plt.close()

		# Reduced basis and modes
		self.basis = snapshots.T @ v[:, :Nr] / (S[None,:Nr]*nt)**.5 

		#check orthogonality
		assert np.allclose( self.basis.T @ self.basis, np.identity(Nr), atol=self.atol), f"orthogonality fails Nr = {Nr}, and {(S[Nr-1]*nt)**.5}"
		return self
	

	def initialize(self, snapshots:np.ndarray, plot_modes=False):
		"""
		Perform incremental HAPOD update
		snapshots_new: (n x Nt)
		"""
		# Local POD on new snapshots
		self.fit(snapshots, plot_modes)
		self.phi = self.basis
		return self
	

	def update(self, snapshots:np.ndarray, plot_modes=False):
		"""
		Perform incremental HAPOD update
		snapshots_new: (n x Nt)
		"""
		#self.phi = pod.phi
		# Local POD on new snapshots
		self.fit(snapshots, False)

		# Merge old basis + new local basis
		phi_merged = np.concatenate((self.phi, self.basis), 1).T
		self.fit(phi_merged, plot_modes)
		self.phi = self.basis
		return self