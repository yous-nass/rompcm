import shutil
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
HAS_FREEFEM = shutil.which("FreeFem++") or shutil.which("FreeFem")

# fichiers qui lisent data/ à l'import : ignorés si data/ est absent
NEEDS_DATA = [
    "sst2d_test.py", "sst_test.py", "sstpod2d_test.py", "sstpod_test.py",
    "pcm_cond_test.py", "pcm_conv_test.py", "pcm_gal_cond_test.py",
    "pcm_gal_conv_temp_test.py", "pcm_gal_conv_test.py",
]
collect_ignore = [] if DATA.exists() else NEEDS_DATA


def pytest_collection_modifyitems(config, items):
    """Marque 'freefem' les tests dont le fichier mentionne pyfreefem/FreeFem."""
    for item in items:
        source = Path(item.fspath).read_text(errors="ignore")
        if "pyfreefem" in source or "FreeFem" in source:
            item.add_marker(pytest.mark.freefem)
            if not HAS_FREEFEM:
                item.add_marker(pytest.mark.skip(reason="FreeFEM non installé"))
