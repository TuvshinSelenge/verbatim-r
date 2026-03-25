"""
Retrieval Strategy Comparison Script
====================================
Compares different retrieval strategies for Hit Rate, Recall@K, and MRR.
"""

import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import List, Tuple, Dict

from custom.setup import connect_to_index, QueryRewriter, QueryGenerator, BGEReranker
from custom.pipeline.io import (
    build_table_report_lines,
    print_and_write_report,
    run_with_timeout,
)
from custom.pipeline.retrieval import extract_preds, merge_and_dedup
from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))
TOP_K = 5
PER_SUBQ_K = 20
SEARCH_K = 50
SKIP_SENTINEL_3000 = True
QUERY_TIMEOUT = 120
MAX_429_RETRIES = 3
RETRY_BASE_DELAY_SEC = 2

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-3-flash-preview"

STRATEGIES_FIRST_RUN = [
    "Baseline (Vector Only)",
    "Baseline + Reranker",
    "Baseline + Rewriting + Reranker",
]

STRATEGIES_SECOND_RUN = [
    "Baseline + Multi-Query + Reranker",
    "Baseline + Rewriting + Multi-Query + Reranker",
]

LLM_CALLS = {
    "Baseline (Vector Only)": "0",
    "Baseline + Reranker": "0",
    "Baseline + Rewriting + Reranker": "1/query",
    "Baseline + Multi-Query + Reranker": "1/query",
    "Baseline + Rewriting + Multi-Query + Reranker": "2/query",
}


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection of provider/API rate-limit errors."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    # Fallback: detect rate-limit phrases in error text.
    text = str(exc).lower()
    return (
        " 429 " in f" {text} "
        or "error code: 429" in text
        or "rate limit" in text
        or "rate-limited" in text
    )


def run_strategy_retrieval(
    strategy_name: str,
    query_text: str,
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter = None,
    query_generator: QueryGenerator = None,
) -> Tuple[List[Tuple], list, str]:
    """
    Return (pred tuples, reranked chunks, rewritten query text).
    """
    if strategy_name == "Baseline (Vector Only)":
        chunks = rag_index.query(query_text, k=TOP_K)
        return extract_preds(chunks), chunks, query_text

    if strategy_name == "Baseline + Reranker":
        hits = rag_index.query(query_text, k=SEARCH_K)
        chunks, _ = reranker.rerank(query_text, hits, top_k=TOP_K)
        return extract_preds(chunks), chunks, query_text
    
    if strategy_name == "Baseline + Rewriting + Reranker":
        rewritten = query_rewriter.rewrite(query_text)
        hits = rag_index.query(rewritten, k=SEARCH_K)
        chunks, _ = reranker.rerank(rewritten, hits, top_k=TOP_K)
        return extract_preds(chunks), chunks, rewritten
  
    if strategy_name == "Baseline + Multi-Query + Reranker":
        subqs = query_generator.generate_queries(query_text)
        merged = merge_and_dedup(subqs, rag_index, PER_SUBQ_K)
        chunks, _ = reranker.rerank(query_text, merged, top_k=TOP_K)
        return extract_preds(chunks), chunks, query_text
   
    if strategy_name == "Baseline + Rewriting + Multi-Query + Reranker":
        rewritten = query_rewriter.rewrite(query_text)
        subqs = query_generator.generate_queries(rewritten)
        merged = merge_and_dedup(subqs, rag_index, PER_SUBQ_K)
        chunks, _ = reranker.rerank(rewritten, merged, top_k=TOP_K)
        return extract_preds(chunks), chunks, rewritten
    raise ValueError(f"Unknown strategy: {strategy_name}")


def evaluate_strategy(
    strategy_name: str,
    gold_data: List[Dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter = None,
    query_generator: QueryGenerator = None,
) -> dict:
    
    per_query = []
    total_queries = len(gold_data)
    skipped = 0

    for i, item in enumerate(gold_data, 1):
        # Read query and normalize gold index shape.
        query = item["query"]
        expected_idxs = item["expected_chunk_index"]
        if not isinstance(expected_idxs, list):
            expected_idxs = [expected_idxs]
        # Skip sentinel-only rows that are not real retrieval targets.
        if SKIP_SENTINEL_3000 and all(idx == 3000 for idx in expected_idxs):
            skipped += 1
            print(f"[{strategy_name}] [{i}/{total_queries}] skipped (sentinel 3000): {query[:80]}")
            continue

        gold_idxs = set(expected_idxs)
        print(f"[{strategy_name}] [{i}/{total_queries}] handling query: {query[:80]}")

        preds = None
        last_error = None
        # Retry only on rate-limit errors with exponential backoff.
        for attempt in range(MAX_429_RETRIES + 1):
            try:
                preds, _, _ = run_with_timeout(
                    lambda q=query: run_strategy_retrieval(
                        strategy_name, q, rag_index, reranker, query_rewriter, query_generator
                    ),
                    timeout_sec=QUERY_TIMEOUT,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                if is_rate_limit_error(e) and attempt < MAX_429_RETRIES:
                    wait_sec = RETRY_BASE_DELAY_SEC * (2 ** attempt)
                    print(f"  RATE LIMITED (attempt {attempt + 1}/{MAX_429_RETRIES + 1}) - retrying in {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
                break

        # If retrieval still failed, count this query as zero score.
        if last_error is not None:
            print(f"  ERROR: {last_error}")
            per_query.append({"hit@k": 0, "rr": 0.0, "recall@k": 0.0})
            time.sleep(2)
            continue

        # Compute retrieval metrics for this query.
        hit = any(idx in gold_idxs for _, idx in preds)
        rank = None
        for r, (_, idx) in enumerate(preds, 1):
            if idx in gold_idxs:
                rank = r
                break
        rr = 1.0 / rank if rank else 0.0
        retrieved_idxs = {idx for _, idx in preds}
        recall_at_k = len(gold_idxs & retrieved_idxs) / len(gold_idxs) if gold_idxs else 0.0
        per_query.append({"hit@k": 1 if hit else 0, "rr": rr, "recall@k": recall_at_k})
        time.sleep(1)

    # Aggregate per-query metrics into final strategy scores.
    if per_query:
        hit_rate = mean([r["hit@k"] for r in per_query])
        mrr = mean([r["rr"] for r in per_query])
        recall_at_k = mean([r["recall@k"] for r in per_query])
    else:
        hit_rate = 0.0
        mrr = 0.0
        recall_at_k = 0.0

    return {
        "strategy": strategy_name,
        "hit_rate": hit_rate,
        "recall@k": recall_at_k,
        "mrr": mrr,
        "num_evaluated": len(per_query),
        "num_skipped": skipped,
    }


def run_strategy_suite(selected_strategies: List[str], title: str, output_filename: str) -> List[Dict]:
    # Print run heading for readability in terminal logs.
    print("=" * 70)
    print(title)
    print("=" * 70)
    if OPENROUTER_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
    else:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    retrieval_data_path = DATA_DIR / "qa_with_chunk_ids.json"
    if not retrieval_data_path.exists():
        print(f"ERROR: Evaluation data missing at {retrieval_data_path}")
        sys.exit(1)
    gold_data = json.loads(retrieval_data_path.read_text())

    # Initialize shared backend components once for fair comparison.
    rag_index, _ = connect_to_index(db_path=DB_PATH, verbose=False)
    reranker = BGEReranker()
    openai_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, timeout=120.0)
    query_rewriter = QueryRewriter(openai_client=openai_client, model=MODEL_NAME)
    query_generator = QueryGenerator(client=openai_client, model=MODEL_NAME)

    strategy_kwargs = {
        "Baseline (Vector Only)": {},
        "Baseline + Reranker": {},
        "Baseline + Rewriting + Reranker": {"query_rewriter": query_rewriter},
        "Baseline + Multi-Query + Reranker": {"query_generator": query_generator},
        "Baseline + Rewriting + Multi-Query + Reranker": {
            "query_rewriter": query_rewriter,
            "query_generator": query_generator,
        },
    }

    all_results = []
    # Evaluate each selected strategy under the same setup.
    for name in selected_strategies:
        kwargs = strategy_kwargs.get(name)
        if kwargs is None:
            raise ValueError(f"Unknown strategy in selected_strategies: {name}")
        all_results.append(
            evaluate_strategy(
                name,
                gold_data,
                rag_index,
                reranker,
                **kwargs,
            )
        )

    # Build final table rows for console + output file.
    header = f"{'Strategy':<30} | {'Hit Rate':<8} | {'Recall@K':<8} | {'MRR':<6} | {'LLM Calls':<10}"
    row_lines = []
    for r in all_results:
        row_lines.append(
            f"{r['strategy']:<30} | {r['hit_rate']:.3f}    | {r['recall@k']:.3f}    | {r['mrr']:.3f}  | "
            f"{LLM_CALLS.get(r['strategy'], '?')}"
        )
    # Reuse shared report helpers for consistent formatting/writing.
    report_lines = build_table_report_lines(
        title="FINAL COMPARISON RESULTS",
        width=132,
        header=header,
        row_lines=row_lines,
        leading_blank_lines=2,
    )
    output_path = RESULTS_DIR / output_filename
    report_with_prefix = [title] + report_lines
    print_and_write_report(report_with_prefix, output_path)
    return all_results


def main():
    run_strategy_suite(
        selected_strategies=STRATEGIES_FIRST_RUN,
        title="RETRIEVAL STRATEGY COMPARISON (PART 1: BASELINE/REWRITING)",
        output_filename="variations_reranker_results_part1.txt",
    )


if __name__ == "__main__":
    main()
