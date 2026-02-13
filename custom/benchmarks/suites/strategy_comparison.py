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
from custom.pipeline.retrieval import extract_preds, merge_and_dedup
from custom.pipeline.runtime import run_with_timeout, is_rate_limit_error
from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))
TOP_K = 5
PER_SUBQ_K = 20
SEARCH_K = 50
SKIP_SENTINEL_1300 = True
QUERY_TIMEOUT = 180
MAX_429_RETRIES = 3
RETRY_BASE_DELAY_SEC = 2

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-3-flash-preview"


def collect_hits_baseline(query_text: str, rag_index, reranker: BGEReranker) -> List[Tuple]:
    # Strategy A: raw vector retrieval only.
    hits = rag_index.query(query_text, k=TOP_K)
    return extract_preds(hits)


def collect_hits_baseline_reranker(query_text: str, rag_index, reranker: BGEReranker) -> List[Tuple]:
    # Strategy B: raw retrieval + reranker.
    hits = rag_index.query(query_text, k=SEARCH_K)
    reranked, _ = reranker.rerank(query_text, hits, top_k=TOP_K)
    return extract_preds(reranked)


def collect_hits_rewriting(
    query_text: str,
    query_rewriter: QueryRewriter,
    rag_index,
    reranker: BGEReranker,
) -> List[Tuple]:
    # Strategy C: rewrite query, then retrieve + rerank.
    rewritten = query_rewriter.rewrite(query_text)
    hits = rag_index.query(rewritten, k=SEARCH_K)
    reranked, _ = reranker.rerank(rewritten, hits, top_k=TOP_K)
    return extract_preds(reranked)


def collect_hits_multiquery(
    query_text: str,
    rag_index,
    reranker: BGEReranker,
    query_generator: QueryGenerator,
) -> List[Tuple]:
    # Strategy D: multi-query generation, merge, rerank.
    subqs = query_generator.generate_queries(query_text)
    merged = merge_and_dedup(subqs, rag_index, PER_SUBQ_K)
    reranked, _ = reranker.rerank(query_text, merged, top_k=TOP_K)
    return extract_preds(reranked)


def collect_hits_full_pipeline(
    query_text: str,
    query_rewriter: QueryRewriter,
    rag_index,
    reranker: BGEReranker,
    query_generator: QueryGenerator,
) -> List[Tuple]:
    # Strategy E: rewrite + multi-query + rerank (full pipeline).
    rewritten = query_rewriter.rewrite(query_text)
    subqs = query_generator.generate_queries(rewritten)
    merged = merge_and_dedup(subqs, rag_index, PER_SUBQ_K)
    reranked, _ = reranker.rerank(rewritten, merged, top_k=TOP_K)
    return extract_preds(reranked)


def evaluate_strategy(
    strategy_name: str,
    gold_data: List[Dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter = None,
    query_generator: QueryGenerator = None,
) -> dict:
    # Evaluate one retrieval strategy over the labeled dataset.
    per_query = []
    total_queries = len(gold_data)
    skipped = 0

    for i, item in enumerate(gold_data, 1):
        query = item["query"]
        expected_idxs = item["expected_chunk_index"]
        if not isinstance(expected_idxs, list):
            expected_idxs = [expected_idxs]
        if SKIP_SENTINEL_1300 and all(idx == 1300 for idx in expected_idxs):
            skipped += 1
            print(f"[{strategy_name}] [{i}/{total_queries}] skipped (sentinel 1300): {query[:80]}")
            continue

        gold_idxs = set(expected_idxs)
        print(f"[{strategy_name}] [{i}/{total_queries}] handling query: {query[:80]}")

        preds = None
        last_error = None
        for attempt in range(MAX_429_RETRIES + 1):
            try:
                if strategy_name == "Baseline (Vector Only)":
                    preds = run_with_timeout(
                        lambda q=query: collect_hits_baseline(q, rag_index, reranker),
                        timeout_sec=QUERY_TIMEOUT,
                    )
                elif strategy_name == "Baseline + Reranker":
                    preds = run_with_timeout(
                        lambda q=query: collect_hits_baseline_reranker(q, rag_index, reranker),
                        timeout_sec=QUERY_TIMEOUT,
                    )
                elif strategy_name == "Baseline + Rewriting + Reranker":
                    preds = run_with_timeout(
                        lambda q=query: collect_hits_rewriting(q, query_rewriter, rag_index, reranker),
                        timeout_sec=QUERY_TIMEOUT,
                    )
                elif strategy_name == "Baseline + Multi-Query + Reranker":
                    preds = run_with_timeout(
                        lambda q=query: collect_hits_multiquery(q, rag_index, reranker, query_generator),
                        timeout_sec=QUERY_TIMEOUT,
                    )
                elif strategy_name == "Baseline + Rewriting + Multi-Query + Reranker":
                    preds = run_with_timeout(
                        lambda q=query: collect_hits_full_pipeline(q, query_rewriter, rag_index, reranker, query_generator),
                        timeout_sec=QUERY_TIMEOUT,
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy_name}")
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

        if last_error is not None:
            print(f"  ERROR: {last_error}")
            per_query.append({"hit@k": 0, "rr": 0.0, "recall@k": 0.0})
            time.sleep(2)
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
        per_query.append({"hit@k": 1 if hit else 0, "rr": rr, "recall@k": recall_at_k})
        time.sleep(1)

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


def main():
    print("=" * 70)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 70)
    if OPENROUTER_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
    else:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    data_path = DATA_DIR / "qa_with_chunk_ids.json"
    if not data_path.exists():
        print(f"ERROR: Evaluation data not found at {data_path}")
        sys.exit(1)
    gold_data = json.loads(data_path.read_text())

    # Shared backend components.
    rag_index, _ = connect_to_index(db_path=DB_PATH, verbose=False)
    reranker = BGEReranker()
    openai_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, timeout=120.0)
    query_rewriter = QueryRewriter(openai_client=openai_client, model=MODEL_NAME)
    query_generator = QueryGenerator(client=openai_client, model=MODEL_NAME)

    # Run each strategy with identical data/config for fair comparison.
    all_results = []
    for name, kwargs in [
        ("Baseline (Vector Only)", {}),
        ("Baseline + Reranker", {}),
        ("Baseline + Rewriting + Reranker", {"query_rewriter": query_rewriter}),
        ("Baseline + Multi-Query + Reranker", {"query_generator": query_generator}),
        ("Baseline + Rewriting + Multi-Query + Reranker", {"query_rewriter": query_rewriter, "query_generator": query_generator}),
    ]:
        all_results.append(evaluate_strategy(name, gold_data, rag_index, reranker, **kwargs))

    llm_calls = {
        "Baseline (Vector Only)": "0",
        "Baseline + Reranker": "0",
        "Baseline + Rewriting + Reranker": "1/query",
        "Baseline + Multi-Query + Reranker": "1/query",
        "Baseline + Rewriting + Multi-Query + Reranker": "2/query",
    }

    print("\n\n" + "=" * 85)
    print(f"{'FINAL COMPARISON RESULTS':^85}")
    print("=" * 85)
    print(f"{'Strategy':<30} | {'Hit Rate':<10} | {'Recall@K':<10} | {'MRR':<10} | {'LLM Calls':<10}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['strategy']:<30} | {r['hit_rate']:.3f}      | {r['recall@k']:.3f}      | {r['mrr']:.3f}      | {llm_calls.get(r['strategy'], '?')}")
    print("=" * 85)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "variations_reranker_results.txt"
    with open(output_path, "w") as f:
        f.write("RETRIEVAL STRATEGY COMPARISON RESULTS\n")
        f.write("=" * 85 + "\n")
        f.write(f"{'Strategy':<30} | {'Hit Rate':<10} | {'Recall@K':<10} | {'MRR':<10} | {'LLM Calls':<10}\n")
        f.write("-" * 85 + "\n")
        for r in all_results:
            f.write(f"{r['strategy']:<30} | {r['hit_rate']:.3f}      | {r['recall@k']:.3f}      | {r['mrr']:.3f}      | {llm_calls.get(r['strategy'], '?')}\n")
        f.write("=" * 85 + "\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
