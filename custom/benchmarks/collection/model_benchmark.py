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
from custom.pipeline.metrics import evaluate_span_extraction
from custom.pipeline.runtime import run_with_timeout
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
    "moonshotai/kimi-k2-0905",
    "meta-llama/llama-4-scout",
    "openai/gpt-5.1"
]

TOP_K = 5
PER_SUBQ_K = 20
SKIP_SENTINEL_1300 = True
QUERY_TIMEOUT = 120


def run_unified_evaluation(
    chunk_data: List[Dict],
    span_data: List[Dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
    rag: VerbatimRAG,
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
    all_em, all_precision, all_recall, all_f1 = [], [], [], []
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
                extracted = []
                if isinstance(spans_raw, dict):
                    for _, span_list in spans_raw.items():
                        if isinstance(span_list, list):
                            for span in span_list:
                                if span and span.strip():
                                    extracted.append(span.strip())
                return extracted

            extracted_spans = run_with_timeout(do_extract, timeout_sec=QUERY_TIMEOUT)
            metrics = evaluate_span_extraction(extracted_spans, gold_spans)
            all_em.append(metrics["exact_match"])
            all_precision.append(metrics["precision"])
            all_recall.append(metrics["recall"])
            all_f1.append(metrics["f1"])
            if is_unanswerable:
                unanswerable_correct.append(1 if len(extracted_spans) == 0 else 0)
        except Exception:
            print("  error during extraction/eval, counting as 0")
            all_em.append(0.0)
            all_precision.append(0.0)
            all_recall.append(0.0)
            all_f1.append(0.0)
            if is_unanswerable:
                unanswerable_correct.append(0)

    span_metrics = {
        "exact_match": mean(all_em) if all_em else 0.0,
        "precision": mean(all_precision) if all_precision else 0.0,
        "recall": mean(all_recall) if all_recall else 0.0,
        "f1": mean(all_f1) if all_f1 else 0.0,
        "unanswerable_accuracy": mean(unanswerable_correct) if unanswerable_correct else 1.0,
    }
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
                chunk_data, span_data, rag_index, reranker, query_rewriter, query_generator, rag
            )
        except Exception:
            chunk_metrics = {"hit_rate": 0.0, "mrr": 0.0, "recall@k": 0.0}
            span_metrics = {
                "exact_match": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "unanswerable_accuracy": 0.0,
            }

        results_table.append({
            "Model": model,
            "Hit Rate": chunk_metrics.get("hit_rate", 0),
            "Recall@K": chunk_metrics.get("recall@k", 0),
            "MRR": chunk_metrics.get("mrr", 0),
            "EM": span_metrics.get("exact_match", 0),
            "Prec": span_metrics.get("precision", 0),
            "Rec": span_metrics.get("recall", 0),
            "F1": span_metrics.get("f1", 0),
            "Unans.Acc": span_metrics.get("unanswerable_accuracy", 0),
        })

    report_lines = []
    report_lines.append("\n\n" + "=" * 100)
    report_lines.append(f"{'FINAL BENCHMARK RESULTS':^100}")
    report_lines.append("=" * 100)
    header = (
        f"{'Model':<35} | {'Hit Rate':<8} | {'Recall@K':<8} | {'MRR':<6} | "
        f"{'EM':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Unans.Acc':<9}"
    )
    report_lines.append(header)
    report_lines.append("-" * 100)
    for row in results_table:
        line = (
            f"{row['Model']:<35} | {row['Hit Rate']:.3f}    | {row['Recall@K']:.3f}    | {row['MRR']:.3f}  | "
            f"{row['EM']:.3f}  | {row['Prec']:.3f}  | {row['Rec']:.3f}  | {row['F1']:.3f}  | {row['Unans.Acc']:.3f}"
        )
        report_lines.append(line)
    report_lines.append("=" * 100)
    print("\n".join(report_lines))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "benchmark_results.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
