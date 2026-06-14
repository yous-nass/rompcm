import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri
import pytest


from rompcm.inference import PODRBFInference
from rompcm.paths import DATADIR


# load mesh
x, y = np.loadtxt( DATADIR / "femcoordsp2nh60.txt", unpack=True)
triang = tri.Triangulation(x, y)


# =========================================================
# Fixtures
# =========================================================

@pytest.fixture(scope="module")
def cond_model():
    return PODRBFInference("cond")


@pytest.fixture(scope="module")
def conv_model():
    return PODRBFInference("conv")


# =========================================================
# Basic loading tests
# =========================================================

def test_cond_model_load(cond_model):
    print(cond_model)
    assert cond_model.pod is not None
    assert cond_model.rbf is not None



def test_conv_model_load(conv_model):
    print(conv_model)
    assert conv_model.pod is not None
    assert conv_model.rbf is not None


# =========================================================
# Reconstruction tests
# =========================================================

@pytest.mark.parametrize(
    "mu, Nt",
    [
        (1.01, 10),
    ]
)
def test_predict_shape(cond_model, mu, Nt):

    field = cond_model.predict(mu, Nt)
    assert isinstance(field, np.ndarray)
    assert field.ndim == 2


@pytest.mark.parametrize(
    "mu, t",
    [
        (1.01, 2.5),
        (1.02, 10.0),
        (1.035, 25.0),
    ]
)
def test_predict_field_point(cond_model, mu, t):

    field = cond_model.predict_field_point(mu, t)
    assert isinstance(field, np.ndarray)
    assert field.ndim == 1


# =========================================================
# Numerical sanity checks
# =========================================================

@pytest.mark.parametrize(
    "mu, t",
    [
        (1.01, 2.5),
        (1.02, 10.0),
    ]
)
def test_no_nan(cond_model, mu, t):

    field = cond_model.predict_field_point(mu, t)

    assert np.isfinite(field).all()


def test_nonzero_solution(cond_model):

    field = cond_model.predict_field_point(1.035, 5.0)

    assert np.linalg.norm(field) > 0.


# =========================================================
# test field at given time t0
# =========================================================

def test_predict_(model="conv", mu=1.035, t0=27.6):

    if model not in ("cond", "conv"):
        raise ValueError( "model must be 'cond' or 'conv'")
    

    rom = PODRBFInference(model)

    field = rom.predict_field_point(mu, t0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal")
    contour = ax.tricontourf(triang, field, levels=200, cmap="jet")
    fig.colorbar(contour, ax=ax)
    ax.set_title(f"ROM prediction, at time t = {t0}")
    plt.show()

    assert fig is not None



# =========================================================
# Animation utility
# ========================================================
def animate_prediction(model="cond", mu=1.035, Nt=50):

    # =====================================================
    # Load ROM
    # =====================================================
    if model not in ("cond", "conv"):
        raise ValueError( "model must be 'cond' or 'conv'")
    
    rom = PODRBFInference(model)

    # =====================================================
    # ROM prediction
    # =====================================================
    field = rom.predict(mu, Nt)

    # =====================================================
    # Mesh coordinates
    # =====================================================


    lev = 200

    # =====================================================
    # Figure
    # =====================================================

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.set_aspect("equal")

    contour = ax.tricontourf(triang, field[0], levels=lev, cmap="jet")

    fig.colorbar(contour, ax=ax)

    ax.set_title(f"ROM prediction, t step = 0")

    # =====================================================
    # Animation update
    # =====================================================

    def update(frame):

        ax.clear()

        ax.set_aspect("equal")

        contour = ax.tricontourf(
            triang,
            field[frame],
            levels=lev,
            cmap="jet"
        )

        ax.set_title(
            f"ROM {model} prediction\n"
            f"$\\mu$ = {mu}, step = {frame}"
        )

        return contour.collections

    # =====================================================
    # Animation
    # =====================================================

    ani = FuncAnimation(
        fig,
        update,
        frames=Nt,
        interval=100,
        blit=False,
    )

    plt.show()

    return ani

# =========================================================
# Optional animation test
# =========================================================

@pytest.mark.parametrize(
    "model, mu, Nt", 
    [
     ("cond", 1.085, 100), 
     ("conv", 1.085, 50)
    ]
)
def test_animation(model, mu, Nt):

    ani = animate_prediction(model, mu, Nt)

    assert ani is not None
