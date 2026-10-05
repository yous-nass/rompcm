import numpy as np
import matplotlib.pyplot as plt
import pytest

from rompcm.paths import DATADIR
from rompcm.sst_snap import SST
from rompcm.pod_basis import POD

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
	Xb = sst.backward2D(T, x_cut, zeta, idx)


fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
for ax, tk_target in zip(axes, [2.1, 27.1, 51.6]):
	k = np.argmin(np.abs(t - tk_target))
	pcm = ax.pcolormesh(zeta, y_cut, Xb[k], cmap='RdBu_r', vmin=-eps, vmax=1, shading='auto')
	0# pcm = ax.pcolormesh(x_cut, y_cut, T_grid[k], cmap='RdBu_r', vmin=-eps, vmax=1, shading='auto')
	ax.axvline(1.0, color='k', linestyle='--', linewidth=3)
	#ax.axvline(x_fit[0,k], color='k', linestyle='--', linewidth=1.)
	ax.set_title(f"t={t[k]:.1f}")
	ax.set_xlabel(r"$\xi$")
	#ax.set_xlabel(r"$x$")
axes[0].set_ylabel("$y$")
fig.colorbar(pcm, ax=axes, label="self-similar Temperature")
#fig.colorbar(pcm, ax=axes, label="reference Temperature")
plt.show()


Xb_flat = Xb.reshape(Nt, Xb.shape[1]*Xb.shape[2])   # row-major: y slowest, xi fastest
for eps in [1.e-8]:
	pod = POD(eps=eps, center=False)
	pod.fit(samples=T[None])

	gpod = POD(eps=eps, center=False)
	gpod.fit(samples=Xb_flat[None])

	print(f"Energy threshold {eps}, truncated modes::: POD : r={pod.phi.shape[1]} vs GPOD : r={gpod.phi.shape[1]}")

plt.semilogy(1 - np.cumsum(pod.S[:]**2) /  np.sum(pod.S[:]**2), '*')
plt.semilogy(1 - np.cumsum(gpod.S[:]**2) /  np.sum(gpod.S[:]**2), 'o')
#plt.ylim([5.e-10, 0.1])
plt.xlim([-1, 50])
plt.xlabel('r')
plt.ylabel(f'Cummulative energy')
plt.legend(["SVD(T)", "SVD(X)"])
plt.grid()
plt.show()