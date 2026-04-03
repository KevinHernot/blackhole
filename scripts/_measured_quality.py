from __future__ import annotations

"""Compatibility shim for script-local measured-quality imports."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_core.measured_quality import *  # noqa: F401,F403
from blackhole_core.measured_quality import __all__  # noqa: F401
