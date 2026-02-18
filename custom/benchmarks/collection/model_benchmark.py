"""
Model Benchmark Framework
=========================
Tests different LLM models on retrieval and extraction metrics.
"""

import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import List, Dict, Tuple

import openai
from openai import OpenAI
from dotenv import load_dotenv

from custom.setup import connect_to_index, BGEReranker, QueryRewriter, QueryGenerator
from custom.pipeline.retrieval import retrieve_and_rerank
from custom.pipeline.metrics import (
    compute_bertscore_best_match_prf,
    compute_rouge_l_best_match_prf,
    compute_token_precision_recall_f1,
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

DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS_TO_TEST = [
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash-lite", 
    "moonshotai/kimi-k2-0905",
    "meta-llama/llama-4-scout",
    "openai/gpt-5.1",
    "openai/gpt-4.1-mini"
]

TOP_K = 5
PER_SUBQ_K = 20
SKIP_SENTINEL_1300 = True
QUERY_TIMEOUT = 120
MAX_EXTRACTED_SPANS = 5


def run_unified_evaluation(
    chunk_data: List[Dict],
    span_data: List[Dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
    rag: VerbatimRAG,
    bert_scorer,
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
        if SKIP_SENTINEL_1300 and all(idx == 1300 for idx in expected_idxs):
            skipped += 1
            print("  skipped (sentinel 1300)")
            continue
        gold_idxs = set(expected_idxs)

        try:
            # Single retrieval pass: rewrite -> multi-query -> rerank.
            reranked, rewritten, preds = run_with_timeout(
                lambda q=query: retrieve_and_rerank(
                    q, query_rewriter, query_generator, rag_index, reranker, top_k=TOP_K, per_subq_k=PER_SUBQ_K
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
                        q, query_rewriter, query_generator, rag_index, reranker, top_k=TOP_K, per_subq_k=PER_SUBQ_K
                    ),
                    timeout_sec=QUERY_TIMEOUT,
                )

            def do_extract(c=chunks, r=rewritten):
                # Collect non-empty spans returned by the extractor.
                spans_raw = rag.extractor.extract_spans(r, c)
                return flatten_extracted_spans(spans_raw)[:MAX_EXTRACTED_SPANS]

            extracted_spans = run_with_timeout(do_extract, timeout_sec=QUERY_TIMEOUT)
            # Token-level P/R/F1 with one-to-one span alignment.
            token_metrics = compute_token_precision_recall_f1(extracted_spans, gold_spans)
            # ROUGE-L best-match aggregation.
            rouge_scores = compute_rouge_l_best_match_prf(extracted_spans, gold_spans)

            # BERTScore best-match aggregation.
            bert_scores = compute_bertscore_best_match_prf(
                extracted_spans, gold_spans, scorer=bert_scorer
            )
            append_span_metric_scores(
                metric_lists=span_metric_lists,
                token_metrics=token_metrics,
                rouge_scores=rouge_scores,
                bert_scores=bert_scores,
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

    chunk_data_path = DATA_DIR / "qa_with_chunk_ids.json"
    span_data_path = DATA_DIR / "span.json"
    if not chunk_data_path.exists() or not span_data_path.exists():
        print("ERROR: required data files are missing in custom/data/")
        sys.exit(1)

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
    report_lines = build_table_report_lines(
        title="FINAL BENCHMARK RESULTS",
        width=132,
        header=header,
        row_lines=row_lines,
        leading_blank_lines=2,
    )
    output_path = RESULTS_DIR / "benchmark_results.txt"
    print_and_write_report(report_lines, output_path)


if __name__ == "__main__":
    main()
