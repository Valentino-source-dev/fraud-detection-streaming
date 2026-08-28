"""Shared test fixtures."""

import sys
from pathlib import Path

# Allow imports from service directories
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "consumer"))
sys.path.insert(0, str(ROOT / "generator"))
sys.path.insert(0, str(ROOT / "training"))
