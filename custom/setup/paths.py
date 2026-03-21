"""
Resolve path to the Milvus Lite DB used by the custom RAG stack.

Priority:
1. CUSTOM_DB_PATH
2. DB_PATH
3. If custom/milvus_verbatim_new.db exists → use it (multi-bank ingest default)
4. Else custom/milvus_verbatim.db
"""

from __future__ import annotations

import os
from pathlib import Path


def custom_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_custom_milvus_db_path() -> str:
    for key in ("CUSTOM_DB_PATH", "DB_PATH"):
        raw = os.getenv(key)
        if raw and raw.strip():
            return os.path.abspath(os.path.expanduser(raw.strip()))

    root = custom_root()
    new_db = root / "milvus_verbatim_new.db"
    old_db = root / "milvus_verbatim.db"
    if new_db.is_file():
        return str(new_db)
    return str(old_db)
