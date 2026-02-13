import sys
from pathlib import Path


# Make `custom.*` imports work when running pytest from repo root.
# Without this, Python might only see the tests folder on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
