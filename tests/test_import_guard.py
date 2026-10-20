"""Runtime purity guard. The Streamlit deploy runs on requirements.txt's
8 dependencies; gs-quant / scipy / statsmodels are DEV/CI oracles only.
Each core module is imported in a fresh subprocess and the loaded module set
is checked, so a transitive import through any helper is caught too.

The *_core.py modules also declare themselves streamlit-free and
network-free in their headers -- enforce that while we're here."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

BANNED_DEV = ("gs_quant", "scipy", "statsmodels", "sklearn")
BANNED_CORE = BANNED_DEV + ("streamlit", "yfinance", "requests")

# pure math modules: banned from dev deps AND from UI/network libs
CORE_MODULES = ["arb_core", "backtest_core", "cost_model", "crisis_core",
                "dcf_core", "exposure_core", "factor_core", "pit_layer",
                "portfolio_core", "risk_core", "signal_research"]


def _check(module: str, banned: tuple) -> None:
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        f"bad = sorted(b for b in {banned!r} if b in sys.modules)\n"
        f"assert not bad, {module!r} + ' transitively imports ' + repr(bad)\n"
    )
    r = subprocess.run([sys.executable, "-c", code],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_is_pure(module):
    _check(module, BANNED_CORE)

def test_requirements_txt_stays_lean():
    text = (ROOT / "requirements.txt").read_text().lower()
    for dep in ("gs-quant", "gs_quant", "scipy", "statsmodels", "scikit-learn"):
        assert dep not in text, f"{dep} must stay in requirements-dev.txt only"
