"""
sst.py — Self-Similar Transformation (SST).

Align a family of snapshots (e.g. moving interface, self-similar profiles)
onto a common reference zeta-grid where all interface is fixed at zeta=1

Typical usage
-------------
>>> sst = SST(eps="melting temperature")
>>> aligned, params = sst.fit_transform(snapshots, x)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

class SST:
	"""Self-similar transformation for snapshot pre-processing.

	Parameters
	----------
	eps: melting temperature
	"""


	def __init__(self, eps:float=0.009):
		self.eps = eps


	def fit(self, snapshots:np.ndarray, x:np.ndarray, t:np.ndarray) -> "SST":
		"""
		Find sub-grid interface position x0(t) where X(t, x0) = eps,
		for every time step, using linear interpolation.
		Parameters
		----------
		snapshots : 2D array, shape (n_t, n_x)
		x : 1D array, shape (n_x,)
		t : 1D array, shape (n_t,)
		eps : float
		Returns
		-------
		st : 1D array, shape (n_t,)
		Interpolated interface position at each time. NaN if not found.
		"""
		n_t = snapshots.shape[0]
		st = np.full(n_t, np.nan)
		for i in range(n_t):
			row = snapshots[i, :]
			mask = row <= self.eps
			if not np.any(mask):
				continue
			id = np.argmax(mask)   # first grid point where condition holds
			if id == 0:
				st[i] = x[0]        # already below eps at first point
				continue
				
			# linear interpolation between (idx-1, idx) to find where X == eps
			x_left, x_right = x[id - 1], x[id]
			y_left, y_right = row[id - 1], row[id]


			if y_right == y_left:
				st[i] = x_right
			else:
				frac = (self.eps - y_left) / (y_right - y_left)
				st[i] = x_left + frac * (x_right - x_left)
		beta, x0 = np.polyfit(np.sqrt(t), st, 1)
		print(f"Fitted parameters: x_0 = {x0:.6f}, beta = {beta:.6f}")
		self.x_fit = np.asarray(x0 + beta * np.sqrt(t))
		return self

    
	def backward(self, X, x, xi):
		Ts = []
		for k in range(len(self.x_fit)):
			xi_k = x / self.x_fit[k]
			order = np.argsort(xi_k)
			f = interp1d(xi_k[order], X[k, order],
					kind='cubic',
					bounds_error=False,
					fill_value=(X[k,0], X[k,-1]))
			Ts.append(f(xi))
		return np.asarray(Ts)


	def forward(self, X, xi, x):
		Ts = []
		for k in range(len(self.x_fit)):
			xi_k = xi * self.x_fit[k] 
			order = np.argsort(xi_k)
			f = interp1d(xi_k[order], X[k, order],
					kind='cubic',
					bounds_error=False,
					fill_value=(X[k,0], X[k,-1]))
			Ts.append(f(x))
		return np.asarray(Ts)
	

	def backward2D(self, T_flat, x, xi, idx):
		"""
		T_flat: (Ns, N_nodes) physical field on flat mesh nodes
		Returns: (Ns, Ny, len(xi)) self-similar field, full y-dependence retained
		"""
		Ny, _ = idx.shape
		out = np.zeros((len(self.x_fit), Ny, len(xi)))
		for k in range(len(self.x_fit)):
			T_grid = T_flat[k][idx]              # (Ny, Nx)
			xi_k = x / self.x_fit[k]
			order = np.argsort(xi_k)
			f = interp1d(xi_k[order], T_grid[:, order], axis=-1,
					 kind='cubic', bounds_error=False,
					 fill_value=(T_grid[:, order][:, 0], T_grid[:, order][:, -1]))
			out[k] = f(xi)
		return out


	def forward2D(self, Theta, xi, x, idx):
		"""
		Theta: (Ns, Ny, len(xi)) self-similar field
		Returns: (Ns, N_nodes) reconstructed physical field on flat mesh
		"""
		Ny, Nx = idx.shape
		N_nodes = idx.max() + 1
		out = np.zeros((len(self.x_fit), N_nodes))
		for k in range(len(self.x_fit)):
			xi_k = xi * self.x_fit[k]
			order = np.argsort(xi_k)
			f = interp1d(xi_k[order], Theta[k][:, order], axis=-1,
					 kind='cubic', bounds_error=False,
					 fill_value=(Theta[k][:, order][:, 0], Theta[k][:, order][:, -1]))
			T_grid = f(x)                    # (Ny, Nx)
			out[k][idx] = T_grid
		return out