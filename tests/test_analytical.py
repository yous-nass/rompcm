import numpy as np

from rompcm.analytical import EPS, front_snapshots


def test_front_shape_and_floor():
    x, t, S = front_snapshots()
    assert S.shape == (len(t), len(x)) == (100, 121)
    assert S.min() >= -EPS and S.max() <= 1.0


def test_front_moves_to_the_right():
    x, _, S = front_snapshots()
    front_pos = [x[np.argmax(S[k] <= EPS + 1e-12)] for k in (0, 50, 99)]
    assert front_pos[0] < front_pos[1] < front_pos[2]
