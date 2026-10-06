import numpy as np
import matplotlib.pyplot as plt
import pytest

from rompcm.paths import DATADIR
from rompcm.sst_snap import SST

x, y = np.loadtxt(DATADIR / "femcoordsp2nh60.txt", unpack=True)
nodes = np.stack((x, y), 1)

Ste = 0.045; Pr=56.2; IPr = 1. /Pr; eps = 1.e-2
t = np.arange(2.1, 51.7, 0.5, dtype=float)
Nt = len(t)
Nh = 14641
Th = [1.0]
T = np.empty((Nt, Nh))


for j in range(len(t)):
    file_path = (DATADIR / "PCMnoconv" / f"Th_{Th[0]}"/ f"T.t.{t[j]}.txt")
    T[j, :] = np.loadtxt(file_path, unpack=True)


def build_grid_index(x, y, x_cut, y_cut, tol=1e-8):
	Ny, Nx = len(y_cut), len(x_cut)
	idx = np.empty((Ny, Nx), dtype=int)
	for iy, yl in enumerate(y_cut):
		row_mask = np.isclose(y, yl, atol=tol)
		row_idx  = np.nonzero(row_mask)[0]
		order    = np.argsort(x[row_idx])
		idx[iy]  = row_idx[order]     # idx[iy, ix] -> flat node index
	return idx


y_cut = np.unique(y.copy())
x_cut = np.sort(x[np.isclose(y.copy(), y_cut[0])])
idx   = build_grid_index(x, y.copy(), x_cut, y_cut)   # (Ny, Nx) -> flat node index

idx_y05 = idx[np.argmin(np.abs(y_cut - 0.5))]
X_1d = T[:, idx_y05]

sst = SST()
sst.fit(X_1d, x_cut, t)
print('')


for it in [-1]:
	zeta = x_cut / sst.x_fit[it]
	#zeta = np.linspace(0, 2.5, 5*len(x_cut))
	Xb = sst.backward(X_1d,x_cut, zeta)
	Xf = sst.forward(Xb, zeta, x_cut)
	"""
	for i in [0, 47, 99]:
		print(f't={t[i]} , error = {np.max(abs(X_1d[i] - Xf[i])):.4e}')
	"""
	err = abs(X_1d - Xf)
	print(f'self-similar transformation :: {t[it]}, global error = {err.max():.4e} and mean error = {np.linalg.norm(err) / np.linalg.norm(X_1d)} ')