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
	Xb = sst.backward(X_1d,x_cut, zeta)


#def plot_snap1d(zeta, Xb):
fig, ax = plt.subplots(1, 2, figsize=(14, 5),  constrained_layout=True)
threshold = 0.009
for k in [0, 99]:
	# ----------- Interface position --------------
	id_x = np.argmin(np.abs(X_1d[k] - threshold))
	idy = np.argmin(np.abs(Xb[k] - threshold))
	x_int = x_cut[id_x]; xi_int = zeta[idy]
	
	# -------- Original coordinates ---------------
	ax[0].plot(x_cut, X_1d[k],'o', ms=4.5,label=fr"$t={t[k]:.1f}$")
	ax[0].axvline(x_int,color='k',ls='--',lw=1)
	# -------- Self-similar coordinates ----------
	ax[1].plot(zeta, Xb[ k],'o',ms=4.5,label=fr"$t={t[k]:.1f}$")
	ax[1].axvline(xi_int,color='k',ls='--',lw=1)
	# -------------------------------------------------
ax[0].set_title("Reference temperature")
ax[1].set_title("Self-similar temperature")
ax[0].set_xlabel(r"$x$")
ax[1].set_xlabel(r"$\xi$")

for a in ax:
	a.axhline(0., color='gray', ls=':', lw=1)
	a.grid(alpha=0.15)
	a.legend(frameon=False)
plt.show()



for eps in [1.e-8]:
	pod = POD(eps=eps, center=False)
	pod.fit(samples=X_1d[None])

	gpod = POD(eps=eps, center=False)
	gpod.fit(samples=Xb[None])

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