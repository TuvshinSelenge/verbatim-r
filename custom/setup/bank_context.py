"""
Load per-bank labels and span files; match banks to indexed PDF paths.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_CUSTOM_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILES_PATH = _CUSTOM_ROOT / "data" / "bank_profiles.json"


def profiles_path() -> Path:
    return Path(os.environ.get("BANK_PROFILES_PATH", str(DEFAULT_PROFILES_PATH))).expanduser().resolve()


def load_bank_profiles(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or profiles_path()
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    banks = data.get("banks", data) if isinstance(data, dict) else data
    if not isinstance(banks, list):
        return []
    return banks


def get_profile_by_id(bank_id: str, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    bid = (bank_id or "").strip().lower()
    if not bid:
        return None
    for b in load_bank_profiles(path):
        if str(b.get("id", "")).strip().lower() == bid:
            return b
    return None


def _norm_filename_stem(name: str) -> str:
    """Treat Erste_Bank / Erste Bank / Erste-Bank as the same logical file."""
    stem = Path(name).stem
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def _scope_from_source_basenames(profile: Dict[str, Any], scopes: List[Any]) -> Optional[Any]:
    """
    Match by PDF filename (not full path). Handles space vs underscore vs hyphen in the name.
    """
    raw = profile.get("source_basenames") or []
    candidates = [str(x).strip() for x in raw if str(x).strip()]
    if not candidates:
        return None
    lower_full = {c.lower() for c in candidates}
    stem_norm = {_norm_filename_stem(c) for c in candidates}
    for sc in scopes:
        if not sc.meta_value:
            continue
        base = Path(sc.meta_value).name
        bl = base.lower()
        if bl in lower_full:
            return sc
        if _norm_filename_stem(base) in stem_norm:
            return sc
    return None


def _scope_from_exact_paths(profile: Dict[str, Any], scopes: List[Any]) -> Optional[Any]:
    for ep in profile.get("exact_source_paths") or []:
        ep = str(ep).strip()
        if not ep:
            continue
        for sc in scopes:
            if not sc.meta_value:
                continue
            mv = sc.meta_value
            if mv == ep or mv.endswith(ep) or ep in mv:
                return sc
    return None


def resolve_scope_via_dataset_id(db_path: str, wanted: str) -> Optional[Any]:
    """
    Find a Milvus filter scope when chunks tag dataset_id (path may not contain the bank name).
    """
    from pymilvus import MilvusClient

    from custom.setup.report_scope import COLLECTION, ReportScope, _milvus_str, _normalize_metadata

    wanted_clean = (wanted or "").strip()
    if not wanted_clean:
        return None
    wanted_l = wanted_clean.lower()

    client = MilvusClient(db_path)

    for candidate in {wanted_clean, wanted_l, wanted_clean.upper()}:
        filt = f'metadata["dataset_id"] == "{_milvus_str(candidate)}"'
        try:
            rows = client.query(
                collection_name=COLLECTION,
                filter=filt,
                output_fields=["metadata"],
                limit=1,
            )
        except Exception:
            rows = []
        if rows:
            meta = _normalize_metadata(rows[0].get("metadata"))
            val = meta.get("dataset_id")
            canon = str(val).strip() if val is not None else candidate
            return ReportScope(meta_key="dataset_id", meta_value=canon, count=0)

    # Slow path: Milvus JSON filter not supported or different typing — scan once
    it = client.query_iterator(
        collection_name=COLLECTION,
        batch_size=2000,
        output_fields=["metadata"],
        filter="",
    )
    canonical: Optional[str] = None
    count = 0
    try:
        while True:
            batch = it.next()
            if not batch:
                break
            for row in batch:
                meta = _normalize_metadata(row.get("metadata"))
                ds = meta.get("dataset_id")
                if ds is None:
                    continue
                sds = str(ds).strip()
                if sds.lower() == wanted_l:
                    canonical = sds
                    count += 1
    finally:
        it.close()

    if canonical is not None:
        return ReportScope(meta_key="dataset_id", meta_value=canonical, count=count)
    return None


def pick_scope_for_profile(
    profile: Dict[str, Any],
    scopes: List[Any],
    db_path: Optional[str] = None,
) -> Optional[Any]:
    hit = _scope_from_exact_paths(profile, scopes)
    if hit:
        return hit

    hit = _scope_from_source_basenames(profile, scopes)
    if hit:
        return hit

    subs: List[str] = list(profile.get("source_substrings") or [])
    if not subs and profile.get("source_substring"):
        subs = [str(profile["source_substring"])]
    subs = [s.lower() for s in subs if s]

    best = None
    best_score = 0
    if subs:
        for sc in scopes:
            if not sc.meta_value:
                continue
            mv = sc.meta_value.lower()
            base = Path(sc.meta_value).name.lower()
            # Prefer hits on the filename, not a parent folder (e.g. .../erste/.../RBI.pdf).
            score = 0
            for sub in subs:
                if sub in base:
                    score += 10
                elif sub in mv:
                    score += 1
            if score > best_score:
                best_score = score
                best = sc
        if best_score > 0:
            return best

    mid = profile.get("match_dataset_id") or profile.get("dataset_id")
    if mid is not None and str(mid).strip() and db_path:
        return resolve_scope_via_dataset_id(db_path, str(mid).strip())

    return None


def span_path_for_profile(profile: Dict[str, Any]) -> Path:
    name = profile.get("span_file") or "span.json"
    return (_CUSTOM_ROOT / "data" / name).resolve()
