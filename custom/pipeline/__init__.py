"""Shared pipeline utilities for custom benchmarks and tests."""

from __future__ import annotations

from .retrieval import (
    extract_preds,
    get_text_and_meta,
    merge_and_dedup,
    retrieve_and_rerank,
)
from .io import run_with_timeout

# Heavy deps (bert_score, etc.) load only when these names are accessed.
_METRIC_NAMES = (
    "RAPIDFUZZ_THRESHOLDS",
    "compute_bertscore_best_match_prf",
    "compute_rapidfuzz_threshold_prf",
    "compute_rouge_l_best_match_prf",
    "flatten_extracted_spans",
    "get_bertscore_scorer",
    "normalize_answer",
    "normalize_extraction_text",
)


def __getattr__(name: str):
    if name in _METRIC_NAMES:
        from . import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | set(_METRIC_NAMES)
        | {
            "extract_preds",
            "get_text_and_meta",
            "merge_and_dedup",
            "retrieve_and_rerank",
            "run_with_timeout",
        }
    )
