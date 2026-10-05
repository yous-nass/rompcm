from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"

podcond = MODELS_DIR / "pod_cond_method_POD_TOL_0.01.pkl"
rbfcond = MODELS_DIR / "rbf_cond_method_POD_TOL_0.01.pkl"
podcondTbasis = MODELS_DIR / "podgalcond_basis"
podconvTbasis = MODELS_DIR / "podgalconv_Tbasis"
podconvUbasis = MODELS_DIR / "podgalconv_Ubasis"
podconvVbasis = MODELS_DIR / "podgalconv_Vbasis"

podconv = MODELS_DIR / "pod_conv_method_POD_TOL_0.1.pkl"
rbfconv = MODELS_DIR / "rbf_conv_method_POD_TOL_0.1.pkl"

DATADIR = PROJECT_ROOT / "data"
SRCDIR = PROJECT_ROOT / "src"
SOLVERDIR = PROJECT_ROOT / "solvers"
TESTSDIR = PROJECT_ROOT / "tests"
FIGURESDIR = PROJECT_ROOT / "figures"
