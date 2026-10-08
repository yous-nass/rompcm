import numpy as np
#import matplotlib.pyplot as plt

from rompcm.analytical import EPS, front_snapshots, front_snapshots_2d


def test_front_shape_and_floor():
	x, t, S = front_snapshots()
	assert S.shape == (len(t), len(x)) == (100, 121)
	assert S.min() >= -EPS and S.max() <= 1.0


def test_front_moves_to_the_right():
	x, _, S = front_snapshots()
	front_pos = [x[np.argmax(S[k] <= EPS + 1e-12)] for k in (0, 50, 99)]
	assert front_pos[0] < front_pos[1] < front_pos[2]


def test_2d_is_1d_copied_along_y():
	_, _, S1 = front_snapshots()
	_, y, _, S2 = front_snapshots_2d()
	"""
	fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
	for ax, tk_target in zip(axes, [2.1, 25.1, 50.1]):
		k = np.argmin(np.abs(t - tk_target))
		pcm = ax.pcolormesh(x, y, S2[k].T, cmap='RdBu_r', vmin=-EPS, vmax=1, shading='auto')


	#ax.set_title(f"t={t[k]:.1f}")
	ax.set_xlabel(r"$x$")
	axes[0].set_ylabel("$y$")
	fig.colorbar(pcm, ax=axes, label="reference Temperature")
	plt.show()
	"""
	assert S2.shape == (S1.shape[0], S1.shape[1], len(y))
	np.testing.assert_array_equal(S2[:, :, 7], S1)