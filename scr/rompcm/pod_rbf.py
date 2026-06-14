import numpy as np
from scipy.interpolate import RBFInterpolator


class RBF_Explicit:
	"""
	Explicit kernel POD–RBF interpolator
	θ_r(mu, t) = Σ_j coeffs[j, r] * φ(||(mu,t)-(mu_j,t_j)||)
	"""

	def __init__(self, kernels:dict, has_normalization=False):
		self.kernels = kernels
		self.has_normalization = has_normalization

	# --------------------------------------------------
	# Kernel definitions
	# --------------------------------------------------
	def _kernel_eval(self, r, eps=1):
		kernel = self.kernels["kernel"]
		if  kernel== "thin_plate_spline":
			out = r * r
			mask = r > 0
			out[mask] *= np.log(r[mask])
			return out
		elif kernel == "cubic":
			return r**3
		elif kernel == "gaussian":
			return np.exp(-(eps * r) ** 2)
		else:
			raise ValueError("Unknown kernel")

	# --------------------------------------------------
	# Fit
	# --------------------------------------------------
	def fit(self, theta, samples_pars, samples_t):
		"""
		theta : (ns, nt, nr)
		samples_pars : (ns,)
		samples_t : (nt,)
		"""

		ns, nt, nr = theta.shape
		self.ns, self.nt, self.nr = ns, nt, nr

		theta_flat = theta.reshape(ns * nt, nr)

		# Normalize parameters if needed
		if self.has_normalization:
			self.a = samples_pars.max() - samples_pars.min()
			self.b = samples_pars.min()
			mu = (samples_pars - self.b) / self.a
		else:
			self.a, self.b = 1.0, 0.0
			mu = samples_pars

		# Build (mu, t) grid
		mu_grid = np.repeat(mu, nt)
		t_grid  = np.tile(samples_t, ns)
		self.mu_t = np.column_stack([mu_grid, t_grid])

		# Build kernel matrix Φ
		diff = self.mu_t[:, None, :] - self.mu_t[None, :, :]
		r = np.linalg.norm(diff, axis=-1)

		K = self._kernel_eval(r)
		K += self.kernels["smooth"] * np.eye(K.shape[0])

		# Solve Φ C = Θ
		self.coeffs = np.linalg.solve(K, theta_flat)

		return self

	# --------------------------------------------------
	# Predict
	# --------------------------------------------------
	def predict_theta(self, target_par, Nt_pred, theta_init=None):
		"""
		target_par : scalar μ
		returns: (Nt_pred, nr), relative L2 error
		"""

		# Normalize
		#mu = (target_par - self.b) / self.a
		mu = target_par
		# Query points
		mu_q = np.full(Nt_pred, mu)
		t_q  = self.mu_t[:Nt_pred, 1]
		mu_t_q = np.column_stack([mu_q, t_q])

		# Kernel eval (OpenMP via NumPy)
		diff = self.mu_t[None, :, :] - mu_t_q[:, None, :]
		r = np.linalg.norm(diff, axis=-1)
		K_q = self._kernel_eval(r)

		theta_pred = K_q @ self.coeffs

		err = 0.0
		if theta_init is not None:
			err += np.sqrt(
				np.mean((theta_init - theta_pred) ** 2) /
				np.mean(theta_init ** 2)
			)
		return theta_pred, err
	

class RBF_Steady:

	def __init__(self, kernels:dict, has_normalization:bool=False):
		self.kernels = kernels
		self.has_normalization = has_normalization

	def fit(self, samples_theta:np.ndarray, samples_pars:np.array, samples_t:np.array):
		ns, nt, nr = samples_theta.shape  # parameter, time, modes
		self.theta = samples_theta.reshape(ns*nt, nr)
	
		if self.has_normalization:
			self.a = samples_pars.max() - samples_pars.min()
			self.b = samples_pars.min()
			µs = (samples_pars - self.b) / self.a
			ts = (samples_t - samples_t.min()) / (samples_t.max() - samples_t.min())
			ts = np.tile(ts, ns)
			µs = np.repeat(µs, nt)
		
		else:	
			self.a = 1.
			self.b = 0.
			ts = np.tile(samples_t, ns)
			µs = np.repeat(samples_pars, nt)
		
		self.mu_t = (µs + ts)[:, None]
		self.rbf = [
					RBFInterpolator(y=self.mu_t, 
									d=self.theta[:, r],
									kernel=self.kernels["kernel"], 
									smoothing=self.kernels["smooth"],
									degree=self.kernels["degre"],
									epsilon=self.kernels["epsilon"]
									)   for r in range(nr)
					]
		self.nr = nr; self.t = ts
		return self
    
	def predict_theta(self, target_par:float=1., Nt_pred:int=1, theta_init:np.ndarray=None):
		target_points =(target_par - self.b) / self.a + self.t[:Nt_pred, None]
		err = 0.				   
		theta_rbf = np.empty((Nt_pred, self.nr)) 
		for r in range(self.nr):
			theta_rbf[:, r] = self.rbf[r](target_points)

		if theta_init is not None:
			err +=  np.sqrt(np.mean((theta_init - theta_rbf)**2)) / np.sqrt(np.mean(theta_init**2))
		return theta_rbf, err
	

	
class RBF_Unsteady:

	def __init__(self, kernels:dict, has_normalization:bool=False):
		self.kernels = kernels
		self.has_normalization = has_normalization

	def fit(self, samples_theta:np.ndarray, samples_pars:np.array, samples_t:np.array):
		ns, nt, nr = samples_theta.shape  # parameter, time, modes
		self.theta = samples_theta.reshape(ns*nt, nr)
	
		if self.has_normalization:
			self.a = samples_pars.max() - samples_pars.min()
			self.b = samples_pars.min()
			µs = (samples_pars - self.b) / self.a
			ts = (samples_t - samples_t.min()) / (samples_t.max() - samples_t.min())
			ts = np.tile(ts, ns)
			µs = np.repeat(µs, nt)
		
		else:	
			self.a = 1.
			self.b = 0.
			ts = np.tile(samples_t, ns)
			µs = np.repeat(samples_pars, nt)

		self.mu_t = np.stack((µs, ts), 1)
		self.rbf = [
					RBFInterpolator(y=self.mu_t, 
									d=self.theta[:, r],
									kernel=self.kernels["kernel"], 
									smoothing=self.kernels["smooth"],
									degree=self.kernels["degre"],
									epsilon=self.kernels["epsilon"]
									)   for r in range(nr)
					]
		self.nr = nr; self.t = ts
		return self
    
	def predict_theta(self, target_par:float=1., Nt_pred:int=1, theta_init:np.ndarray=None):
		target_points = np.stack(((target_par - self.b) * np.ones(Nt_pred) / self.a, 
							       self.t[:Nt_pred]), 1)
		err = 0.				   
		theta_rbf = np.empty((Nt_pred, self.nr)) 
		for r in range(self.nr):
			theta_rbf[:, r] = self.rbf[r](target_points)

		if theta_init is not None:
			err =  np.sqrt(np.mean((theta_init - theta_rbf)**2))
		return theta_rbf, err



class RBF_evol:

	def __init__(self, 
			     theta:np.ndarray, 
			     samples_µ:np.array, 
			     kernels:dict):
		
	
		# perform RBF based on POD coefficients	
		rbf = []
		ns, nt, nr = theta.shape 
		m = ns*(nt-1)
		y_t = np.transpose(theta, (2, 0, 1)) # (nr, ns, nt)
		y_tp = y_t[:,:,1:].reshape(nr, m)
		y_t  = y_t[:,:,:-1].reshape(nr, m)
		
		for r in range(nr):
			Th_t = np.stack((np.repeat(samples_µ, nt-1), y_t[r]), 1)   #
			rbf.append( RBFInterpolator(y=Th_t, d=y_tp[r], 
										kernel=kernels["kernel"],
										smoothing=kernels["smooth"],
										degree=kernels["degre"],
										epsilon=kernels["epsilon"]) )
		self.rbf = rbf

	
	def predict_theta(self, theta0:np.array, Nt_pred:int, mu:float):
		n = len(theta0)
		theta = np.empty((Nt_pred, n))
		
		theta[0] = theta0
		for k in range(1, Nt_pred):
			for r in range(n):
				Xq = np.array([[mu, theta[k-1, r]]])
				theta[k, r] = self.rbf[r](Xq)[-1]
		return theta