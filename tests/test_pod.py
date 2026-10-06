import numpy as np


from rompcm.pod_basis import POD
from rompcm.sst_snap import SST




def test_theta_reconstructs_snapshots(snapshots):
	samples = snapshots[None, :, :]
	pod = POD(center=False)                       
	pod.fit(samples)          
	recon = pod.theta @ pod.phi.T
	err = np.linalg.norm(recon[0] - samples[0]) / np.linalg.norm(snapshots)
	assert err < 5e-3


def test_self_similar_snapshots(grid, snapshots):
	x, t = grid
	sst = SST(eps=0.01)
	sst.fit(snapshots, x, t)

	for it in [-1]:
		zeta = x / sst.x_fit[it]
		Xb = sst.backward(snapshots,x,zeta)
		Xf = sst.forward(Xb, zeta, x)
		err = np.max(abs(snapshots - Xf)) / np.max(abs(snapshots))
	assert err <5.e-3