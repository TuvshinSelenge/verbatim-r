"""Shared pipeline utilities for custom benchmarks and tests."""

from .retrieval import (
    extract_preds,
    get_text_and_meta,
    merge_and_dedup,
    retrieve_and_rerank,
)
from .metrics import (
    compute_bertscore_best_match_prf,
    compute_rouge_l_best_match_prf,
    flatten_extracted_spans,
    get_bertscore_scorer,
    normalize_answer,
    normalize_extraction_text,
)
from .runtime import is_rate_limit_error, run_with_timeout
