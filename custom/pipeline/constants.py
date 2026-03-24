"""Lightweight constants shared by pipeline modules (no heavy deps)."""

from typing import Tuple

# RapidFuzz thresholds used to show metric sensitivity from lenient to near-exact:
#   0.50 → very lenient
#   0.75 → lenient
#   0.90 → strict
#   0.95 → near-exact
RAPIDFUZZ_THRESHOLDS: Tuple[float, ...] = (0.50, 0.75, 0.90, 0.95)
