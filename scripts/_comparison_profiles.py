from __future__ import annotations

"""Compatibility shim for script-local imports.

The shared implementation now lives in ``blackhole_core.comparison_profiles`` so
the project is a real installable package. This module re-exports that surface
to keep existing script entry points working.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_core.comparison_profiles import *  # noqa: F401,F403
from blackhole_core.comparison_profiles import __all__  # noqa: F401
