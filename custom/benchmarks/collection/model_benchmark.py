"""
Model Benchmark Framework
=========================
Tests different LLM models on retrieval and extraction metrics.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from statistics import mean
from typing import List, Dict, Tuple, Optional

import openai
from openai import OpenAI
from dotenv import load_dotenv

from custom.setup import connect_to_index, BGEReranker, QueryRewriter, QueryGenerator
from custom.setup.bank_context import (
    get_profile_by_id,
    load_bank_profiles,
    pick_scope_for_profile,
    span_path_for_profile,
)
from custom.setup.report_scope import list_report_scopes
from custom.pipeline.retrieval import retrieve_and_rerank
from custom.pipeline.metrics import (
    RAPIDFUZZ_THRESHOLDS,
    compute_bertscore_best_match_prf,
    compute_rapidfuzz_threshold_prf,
    compute_rouge_l_best_match_prf,
    flatten_extracted_spans,
    get_bertscore_scorer,
)
from custom.pipeline.io import (
    append_span_metric_scores,
    append_zero_span_metric_scores,
    build_benchmark_header_and_rows,
    build_table_report_lines,
    init_span_metric_lists,
    make_benchmark_result_row,
    print_and_write_report,
    run_with_timeout,
    summarize_span_metric_lists,
    zero_chunk_metrics,
    zero_span_metrics,
)
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.llm_client import LLMClient

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim_new.db"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS_TO_TEST = [
    "google/gemini-3-flash-preview"
]

TOP_K = 5
PER_SUBQ_K = 20
SKIP_SENTINEL_3000 = True
QUERY_TIMEOUT = 120
MAX_EXTRACTED_SPANS = 5


def _default_chunk_ids_path_for_profile(profile: Dict) -> Path:
    span_name = str(profile.get("span_file", "span.json"))
    if span_name.startswith("span_"):
        candidate = DATA_DIR / span_name.replace("span_", "chunk_ids_", 1)
    else:
        candidate = DATA_DIR / "qa_with_chunk_ids.json"
    return candidate


def _resolve_benchmark_inputs(
    bank_id: str,
    chunk_data_path_arg: Optional[str],
    span_data_path_arg: Optional[str],
) -> Tuple[Dict, Path, Path, Optional[str], str, str]:
    profile = get_profile_by_id(bank_id)
    if not profile:
        known = [str(p.get("id")) for p in load_bank_profiles()]
        raise SystemExit(f"Unknown --bank {bank_id!r}. Known ids: {known}")

    scopes = list_report_scopes(DB_PATH)
    scope = pick_scope_for_profile(profile, scopes, DB_PATH)
    milvus_filter = scope.filter_expr() if scope and scope.meta_value else None

    span_data_path = (
        Path(span_data_path_arg).expanduser().resolve()
        if span_data_path_arg
        else span_path_for_profile(profile)
    )
    chunk_data_path = (
        Path(chunk_data_path_arg).expanduser().resolve()
        if chunk_data_path_arg
        else _default_chunk_ids_path_for_profile(profile)
    )

    if not chunk_data_path.exists():
        raise SystemExit(
            f"Chunk data file not found: {chunk_data_path}\n"
            "Create the bank-specific chunk_ids_*.json file or pass --chunk-data explicitly."
        )
    if not span_data_path.exists():
        raise SystemExit(f"Span data file not found: {span_data_path}")

    legal = str(profile.get("legal_name", ""))
    short = str(profile.get("short_name", ""))
    return profile, chunk_data_path, span_data_path, milvus_filter, legal, short


def run_unified_evaluation(
    chunk_data: List[Dict],
    span_data: List[Dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
    rag: VerbatimRAG,
    bert_scorer,
    metadata_filter: Optional[str],
    bank_name: str,
    bank_short_name: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Phase 1: retrieval metrics, while caching retrieved chunks for reuse.
    retrieval_results = []
    chunk_cache: Dict[str, Tuple[List, str]] = {}
    skipped = 0
    total_chunk_queries = len(chunk_data)

    for i, item in enumerate(chunk_data, 1):
        query = item["query"]
        print(f"[retrieval {i}/{total_chunk_queries}] {query[:80]}")
        expected_idxs = item["expected_chunk_index"]
        if not isinstance(expected_idxs, list):
            expected_idxs = [expected_idxs]
        if SKIP_SENTINEL_3000 and all(idx == 3000 for idx in expected_idxs):
            skipped += 1
            print("  skipped (sentinel 3000)")
            continue
        gold_idxs = set(expected_idxs)

        try:
            # Single retrieval pass: rewrite -> multi-query -> rerank.
            reranked, rewritten, preds = run_with_timeout(
                lambda q=query: retrieve_and_rerank(
                    q,
                    query_rewriter,
                    query_generator,
                    rag_index,
                    reranker,
                    top_k=TOP_K,
                    per_subq_k=PER_SUBQ_K,
                    filter=metadata_filter,
                    bank_name=bank_name,
                    bank_short_name=bank_short_name,
                ),
                timeout_sec=QUERY_TIMEOUT,
            )
            chunk_cache[query] = (reranked, rewritten)
        except Exception:
            print("  error during retrieval/eval, counting as 0")
            retrieval_results.append({"hit@k": 0, "rr": 0.0, "recall@k": 0.0})
            continue

        hit = any(idx in gold_idxs for _, idx in preds)
        rank = None
        for r, (_, idx) in enumerate(preds, 1):
            if idx in gold_idxs:
                rank = r
                break
        rr = 1.0 / rank if rank else 0.0
        retrieved_idxs = {idx for _, idx in preds}
        recall_at_k = len(gold_idxs & retrieved_idxs) / len(gold_idxs) if gold_idxs else 0.0
        retrieval_results.append({"hit@k": 1 if hit else 0, "rr": rr, "recall@k": recall_at_k})

    if retrieval_results:
        hit_rate = mean([r["hit@k"] for r in retrieval_results])
        mrr = mean([r["rr"] for r in retrieval_results])
        recall_at_k = mean([r["recall@k"] for r in retrieval_results])
    else:
        hit_rate = 0.0
        mrr = 0.0
        recall_at_k = 0.0
    chunk_metrics = {"hit_rate": hit_rate, "mrr": mrr, "recall@k": recall_at_k}

    # Phase 2: extraction metrics. Reuse cached chunks when possible.
    span_metric_lists = init_span_metric_lists()
    unanswerable_correct = []
    total_span_queries = len(span_data)
    for i, item in enumerate(span_data, 1):
        query = item["query"]
        print(f"[extraction {i}/{total_span_queries}] {query[:80]}")
        gold_spans = item.get("top_spans", [])
        if isinstance(gold_spans, str):
            gold_spans = [gold_spans] if gold_spans else []
        is_unanswerable = len(gold_spans) == 0
        try:
            cached = chunk_cache.get(query)
            if cached:
                chunks, rewritten = cached
            else:
                # Fallback retrieval when query is missing in cache.
                chunks, rewritten, _ = run_with_timeout(
                    lambda q=query: retrieve_and_rerank(
                        q,
                        query_rewriter,
                        query_generator,
                        rag_index,
                        reranker,
                        top_k=TOP_K,
                        per_subq_k=PER_SUBQ_K,
                        filter=metadata_filter,
                        bank_name=bank_name,
                        bank_short_name=bank_short_name,
                    ),
                    timeout_sec=QUERY_TIMEOUT,
                )

            def do_extract(c=chunks, r=rewritten):
                # Collect non-empty spans returned by the extractor.
                spans_raw = rag.extractor.extract_spans(r, c)
                return flatten_extracted_spans(spans_raw)[:MAX_EXTRACTED_SPANS]

            extracted_spans = run_with_timeout(do_extract, timeout_sec=QUERY_TIMEOUT)
            # ROUGE-L one-to-one best-match PRF (soft LCS similarity).
            rouge_scores = compute_rouge_l_best_match_prf(extracted_spans, gold_spans)
            # BERTScore one-to-one best-match PRF (contextual semantic similarity).
            bert_scores = compute_bertscore_best_match_prf(
                extracted_spans, gold_spans, scorer=bert_scorer
            )
            # RapidFuzz one-to-one best-match PRF at configured thresholds
            # (primary extraction metric; binary hard-match via WRatio similarity).
            rapidfuzz_scores = compute_rapidfuzz_threshold_prf(extracted_spans, gold_spans)
            append_span_metric_scores(
                metric_lists=span_metric_lists,
                rouge_scores=rouge_scores,
                bert_scores=bert_scores,
                rapidfuzz_scores=rapidfuzz_scores,
            )
            if is_unanswerable:
                unanswerable_correct.append(1 if len(extracted_spans) == 0 else 0)
        except Exception:
            print("  error during extraction/eval, counting as 0")
            append_zero_span_metric_scores(span_metric_lists)
            if is_unanswerable:
                unanswerable_correct.append(0)

    span_metrics = summarize_span_metric_lists(span_metric_lists, unanswerable_correct)
    return chunk_metrics, span_metrics


def main():
    print("=== INITIALIZING BENCHMARK FRAMEWORK ===")
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

    parser = argparse.ArgumentParser(description="Bank-scoped model benchmark.")
    parser.add_argument(
        "--bank",
        default=os.getenv("CUSTOM_BANK_ID", "rbi"),
        help="bank id from custom/data/bank_profiles.json (e.g. rbi, bawag, erste, uni)",
    )
    parser.add_argument(
        "--chunk-data",
        default=None,
        help="optional path to chunk ids json (defaults to bank-derived chunk_ids_*.json)",
    )
    parser.add_argument(
        "--span-data",
        default=None,
        help="optional path to span json (defaults to bank profile span_file)",
    )
    args = parser.parse_args()

    profile, chunk_data_path, span_data_path, metadata_filter, bank_name, bank_short_name = _resolve_benchmark_inputs(
        str(args.bank).strip().lower(),
        args.chunk_data,
        args.span_data,
    )
    print(
        f"Active bank: [{profile.get('id')}] {bank_name} ({bank_short_name})\n"
        f"Chunk data:  {chunk_data_path}\n"
        f"Span data:   {span_data_path}\n"
        f"Filter:      {metadata_filter if metadata_filter else 'none (all chunks)'}"
    )

    chunk_data = json.loads(chunk_data_path.read_text())
    span_data = json.loads(span_data_path.read_text())

    # Build shared index + reranker once; swap only LLM-dependent parts per model.
    rag_index, _ = connect_to_index(db_path=DB_PATH, verbose=False)
    reranker = BGEReranker()
    results_table = []
    bert_scorer = get_bertscore_scorer(lang="en")

    for model in MODELS_TO_TEST:
        # Per-model clients/components for fair benchmarking across providers.
        print(f"\n=== Running model: {model} ===")
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, timeout=120.0)
        query_rewriter = QueryRewriter(openai_client=client, model=model)
        query_generator = QueryGenerator(client=client, model=model)
        llm_client = LLMClient(model=model, api_base=OPENROUTER_BASE_URL)
        llm_client.client = client
        llm_client.async_client = openai.AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, timeout=120.0
        )
        rag = VerbatimRAG(rag_index, llm_client=llm_client)
        rag.template_manager.use_contextual_mode(use_per_fact=True)

        try:
            chunk_metrics, span_metrics = run_unified_evaluation(
                chunk_data,
                span_data,
                rag_index,
                reranker,
                query_rewriter,
                query_generator,
                rag,
                bert_scorer,
                metadata_filter=metadata_filter,
                bank_name=bank_name,
                bank_short_name=bank_short_name,
            )
        except Exception:
            chunk_metrics = zero_chunk_metrics()
            span_metrics = zero_span_metrics()

        results_table.append(
            make_benchmark_result_row(
                labels={"Model": model},
                chunk_metrics=chunk_metrics,
                span_metrics=span_metrics,
            )
        )

    header, row_lines = build_benchmark_header_and_rows(
        rows=results_table,
        leading_columns=[("Model", 35)],
    )
    bank_id = str(profile.get("id", "unknown")).strip().lower() or "unknown"
    bank_label = str(profile.get("label", bank_id)).strip() or bank_id
    report_lines = build_table_report_lines(
        title=(
            f"FINAL BENCHMARK RESULTS [{bank_label}] "
            f"(RF thresholds: {'/'.join(str(int(round(t * 100))) for t in RAPIDFUZZ_THRESHOLDS)}%)"
        ),
        width=max(280, len(header) + 4),
        header=header,
        row_lines=row_lines,
        leading_blank_lines=2,
    )
    output_path = RESULTS_DIR / f"benchmark_results_{bank_id}.txt"
    print_and_write_report(report_lines, output_path)


if __name__ == "__main__":
    main()
