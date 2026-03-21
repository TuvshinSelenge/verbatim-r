"""
Milvus chunk scopes: list indexed reports and build metadata filters.

Shared by interactive demos and the custom RAG service.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

COLLECTION = os.getenv("VERBATIM_COLLECTION_NAME", "verbatim_rag")


def _normalize_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _milvus_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _report_field_and_value(meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Pick one identifying string per chunk for grouping/filtering.

    Order matters: some ingest pipelines only set ``filename`` / ``file_path``,
    not ``source_file`` / ``source`` — those chunks were invisible to bank matching before.
    """
    for key in (
        "source_file",
        "source",
        "filename",
        "file_path",
        "pdf_path",
        "original_filename",
        "filepath",
        "path",
    ):
        v = meta.get(key)
        if v is not None and str(v).strip():
            return key, str(v)
    ds = meta.get("dataset_id")
    if ds is not None and str(ds).strip():
        return "dataset_id", str(ds)
    return "", ""


@dataclass(frozen=True)
class ReportScope:
    meta_key: str
    meta_value: str
    count: int

    def filter_expr(self) -> str:
        if not self.meta_value:
            raise ValueError("Cannot filter empty scope")
        return f'metadata["{self.meta_key}"] == "{_milvus_str(self.meta_value)}"'

    @property
    def label(self) -> str:
        if not self.meta_value:
            return "(unknown)"
        base = Path(self.meta_value.rstrip("/")).name
        if base and base != self.meta_value:
            return f"{base}  ← {self.meta_value}"
        return self.meta_value


def list_report_scopes(db_path: str) -> List[ReportScope]:
    from pymilvus import MilvusClient

    client = MilvusClient(db_path)
    if not client.has_collection(COLLECTION):
        raise FileNotFoundError(f"No collection {COLLECTION!r} in {db_path!r}")

    from collections import Counter

    counts: Counter[Tuple[str, str]] = Counter()
    it = client.query_iterator(
        collection_name=COLLECTION,
        batch_size=2000,
        output_fields=["metadata"],
        filter="",
    )
    try:
        while True:
            batch: Optional[list] = it.next()
            if not batch:
                break
            for row in batch:
                meta = _normalize_metadata(row.get("metadata"))
                k, v = _report_field_and_value(meta)
                if not k:
                    counts[("", "(unknown)")] += 1
                else:
                    counts[(k, v)] += 1
    finally:
        it.close()

    scopes: List[ReportScope] = []
    for (mk, mv), n in counts.items():
        if mk == "":
            scopes.append(ReportScope(meta_key="source", meta_value="", count=n))
        else:
            scopes.append(ReportScope(meta_key=mk, meta_value=mv, count=n))
    return sorted(scopes, key=lambda s: (s.label.lower(), s.meta_value))


def list_indexed_source_records(db_path: str) -> List[Dict[str, Any]]:
    """
    Read Milvus and return every distinct source identifier on chunks (for bank_profiles tuning).

    Each row is what we use for scoping: metadata_field + exact value → copy substrings from ``value``.
    """
    records: List[Dict[str, Any]] = []
    for s in list_report_scopes(db_path):
        if not s.meta_value:
            records.append(
                {
                    "metadata_field": "unknown",
                    "value": "",
                    "chunk_count": s.count,
                    "basename": None,
                }
            )
            continue
        base = Path(s.meta_value.rstrip("/")).name
        records.append(
            {
                "metadata_field": s.meta_key,
                "value": s.meta_value,
                "chunk_count": s.count,
                "basename": base,
            }
        )
    return records
