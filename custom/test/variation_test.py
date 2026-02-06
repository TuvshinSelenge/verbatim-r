"""
Retrieval Strategy Comparison Script
====================================
Compares different retrieval strategies for Hit Rate, Recall@K, and MRR:
1. Raw Query + Reranker (No LLM)
2. Query Rewriting + Reranker
3. Multi-Query Variations + Reranker
4. Rewriting + Multi-Query + Reranker (Full Pipeline)

Uses the evaluation data from custom/test_data/qa_with_chunk_ids.json
"""

import json
import os
import sys
import time
import concurrent.futures
from pathlib import Path
from statistics import mean
from typing import List, Tuple, Dict

# Setup paths
SCRIPT_DIR = Path(__file__).parent          # custom/test/
PROJECT_ROOT = SCRIPT_DIR.parent            # custom/
REPO_ROOT = PROJECT_ROOT.parent             # verbatim-r/
SETUP_DIR = PROJECT_ROOT / "set-up"         # custom/set-up/

# Add paths for imports
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SETUP_DIR))

# Import from set-up modules
from connect_index import connect_to_index
from query_rewriter import QueryRewriter
from query_generator import QueryGenerator
from bge_ranker import BGEReranker
from openai import OpenAI

# Database path
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))

# Configuration
TOP_K = 5
PER_SUBQ_K = 20
SEARCH_K = 50  # For raw query strategy
SKIP_SENTINEL_1300 = True
QUERY_TIMEOUT = 180  # 3 minutes per query

# OpenRouter Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-3-flash-preview"


# =============================================================================
# HELPERS
# =============================================================================

def get_text_and_meta(chunk):
    """Extract text and metadata from chunk (supports dict or object)."""
    if isinstance(chunk, dict):
        return chunk.get("text", ""), chunk.get("metadata", {}) or {}
    return getattr(chunk, "text", ""), getattr(chunk, "metadata", {}) or {}


def extract_preds(chunks) -> List[Tuple]:
    """Extract (source_file, chunk_index) predictions from reranked chunks."""
    preds = []
    for h in chunks:
        _, m = get_text_and_meta(h)
        preds.append((m.get("source_file"), m.get("chunk_index")))
    return preds


def merge_and_dedup(queries: List[str], rag_index, k: int) -> list:
    """Search multiple queries and merge/dedup results."""
    merged, seen = [], set()
    for q in queries:
        hits = rag_index.query(q, k=k)
        for h in hits:
            t, m = get_text_and_meta(h)
            if not t:
                continue
            key = (m.get("source_file"), m.get("chunk_index"), t[:200])
            if key in seen:
                continue
            seen.add(key)
            merged.append(h)
    return merged


def run_with_timeout(func, timeout_sec=QUERY_TIMEOUT):
    """Run a function in a thread with a hard timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            print(f"  TIMEOUT after {timeout_sec}s — skipping")
            raise TimeoutError(f"Timed out after {timeout_sec}s")

# =============================================================================
# STRATEGY 0: Baseline - Raw Query (Vector Search Only)
# =============================================================================

def collect_hits_baseline(query_text: str, rag_index, reranker: BGEReranker) -> List[Tuple]:
    """Direct vector search. No LLM, No Reranker."""
    hits = rag_index.query(query_text, k=TOP_K)
    return extract_preds(hits)

# =============================================================================
# STRATEGY 1: Raw Query + Reranker (No LLM)
# =============================================================================

def collect_hits_raw(query_text: str, rag_index, reranker: BGEReranker) -> List[Tuple]:
    """Direct vector search + reranking. No LLM involved."""
    hits = rag_index.query(query_text, k=SEARCH_K)
    reranked, _ = reranker.rerank(query_text, hits, top_k=TOP_K)
    return extract_preds(reranked)


# =============================================================================
# STRATEGY 2: Query Rewriting + Reranker
# =============================================================================

def collect_hits_rewriting(
    query_text: str,
    query_rewriter: QueryRewriter,
    rag_index,
    reranker: BGEReranker,
) -> List[Tuple]:
    """Rewrite query for better search, then rerank."""
    rewritten = query_rewriter.rewrite(query_text)
    hits = rag_index.query(rewritten, k=SEARCH_K)
    reranked, _ = reranker.rerank(rewritten, hits, top_k=TOP_K)
    return extract_preds(reranked)


# =============================================================================
# STRATEGY 3: Multi-Query Variations + Reranker (No Rewriting)
# =============================================================================

def collect_hits_multiquery(
    query_text: str,
    rag_index,
    reranker: BGEReranker,
    query_generator: QueryGenerator,
) -> List[Tuple]:
    """Generate query variations from raw query, search all, merge, rerank."""
    subqs = query_generator.generate_queries(query_text)
    merged = merge_and_dedup(subqs, rag_index, PER_SUBQ_K)
    reranked, _ = reranker.rerank(query_text, merged, top_k=TOP_K)
    return extract_preds(reranked)


# =============================================================================
# STRATEGY 4: Rewriting + Multi-Query + Reranker (Full Pipeline)
# =============================================================================

def collect_hits_full_pipeline(
    query_text: str,
    query_rewriter: QueryRewriter,
    rag_index,
    reranker: BGEReranker,
    query_generator: QueryGenerator,
) -> List[Tuple]:
    """Rewrite, then generate variations, merge, rerank."""
    rewritten = query_rewriter.rewrite(query_text)
    subqs = query_generator.generate_queries(rewritten)
    merged = merge_and_dedup(subqs, rag_index, PER_SUBQ_K)
    reranked, _ = reranker.rerank(rewritten, merged, top_k=TOP_K)
    return extract_preds(reranked)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_strategy(
    strategy_name: str,
    gold_data: List[Dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter = None,
    query_generator: QueryGenerator = None,
) -> dict:
    """Evaluate a retrieval strategy and return Hit Rate and MRR."""
    per_query = []

    print(f"\n{'='*60}")
    print(f"STRATEGY: {strategy_name}")
    print(f"{'='*60}")
    print(f"Config: TOP_K={TOP_K}, PER_SUBQ_K={PER_SUBQ_K}, SEARCH_K={SEARCH_K}")
    print(f"{'='*60}\n")

    total_queries = len(gold_data)
    skipped = 0

    for i, item in enumerate(gold_data, 1):
        query = item["query"]
        expected_idxs = item["expected_chunk_index"]

        if not isinstance(expected_idxs, list):
            expected_idxs = [expected_idxs]

        # Skip sentinel questions
        if SKIP_SENTINEL_1300 and all(idx == 1300 for idx in expected_idxs):
            skipped += 1
            continue

        gold_idxs = set(expected_idxs)
        print(f"[{i}/{total_queries}] Evaluating: {query[:50]}...")

        try:
            if strategy_name == "Baseline (Vector Only)":
                preds = run_with_timeout(
                    lambda q=query: collect_hits_baseline(q, rag_index, reranker)
                )
            elif strategy_name == "Raw + Reranker":
                preds = run_with_timeout(
                    lambda q=query: collect_hits_raw(q, rag_index, reranker)
                )
            elif strategy_name == "Rewriting + Reranker":
                preds = run_with_timeout(
                    lambda q=query: collect_hits_rewriting(q, query_rewriter, rag_index, reranker)
                )
            elif strategy_name == "Multi-Query + Reranker":
                preds = run_with_timeout(
                    lambda q=query: collect_hits_multiquery(q, rag_index, reranker, query_generator)
                )
            elif strategy_name == "Full Pipeline":
                preds = run_with_timeout(
                    lambda q=query: collect_hits_full_pipeline(q, query_rewriter, rag_index, reranker, query_generator)
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

        except Exception as e:
            print(f"  ERROR: {e}")
            per_query.append({"hit@k": 0, "rr": 0.0})
            time.sleep(2)  # Rate limit recovery
            continue

        # Calculate Hit@K
        hit = any(idx in gold_idxs for _, idx in preds)

        # Calculate Reciprocal Rank
        rank = None
        for r, (_, idx) in enumerate(preds, 1):
            if idx in gold_idxs:
                rank = r
                break
        rr = 1.0 / rank if rank else 0.0

        # Calculate Recall@K: fraction of expected chunks found in top K
        retrieved_idxs = {idx for _, idx in preds}
        recall_at_k = len(gold_idxs & retrieved_idxs) / len(gold_idxs) if gold_idxs else 0.0

        per_query.append({"hit@k": 1 if hit else 0, "rr": rr, "recall@k": recall_at_k})

        status = "HIT" if hit else "MISS"
        print(f"  Result: {status} | RR: {rr:.3f} | Recall@{TOP_K}: {recall_at_k:.3f} ({len(gold_idxs & retrieved_idxs)}/{len(gold_idxs)})")
        time.sleep(1)  # Rate limit buffer

    # Calculate aggregate metrics
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main evaluation function."""
    print("="*70)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("="*70)
    print("Comparing: Baseline, Raw+Rerank, Rewriting, Multi-Query, Full Pipeline")
    print("="*70)

    # Set OPENAI_API_KEY for modules that use it internally
    if OPENROUTER_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
    else:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    # Load gold data
    print("\nLoading evaluation data...")
    data_path = PROJECT_ROOT / "test_data" / "qa_with_chunk_ids.json"
    if not data_path.exists():
        print(f"ERROR: Evaluation data not found at {data_path}")
        sys.exit(1)

    gold_data = json.loads(data_path.read_text())
    print(f"Loaded {len(gold_data)} queries from {data_path}")

    # Connect to index
    print(f"\nConnecting to Milvus index (DB: {DB_PATH})...")
    try:
        rag_index, _ = connect_to_index(db_path=DB_PATH, verbose=False)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Initialize reranker
    print("\nInitializing reranker...")
    reranker = BGEReranker()

    # Initialize OpenAI client
    print("\nInitializing OpenRouter client...")
    openai_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        timeout=120.0
    )

    # Initialize query rewriter and generator
    query_rewriter = QueryRewriter(openai_client=openai_client, model=MODEL_NAME)
    query_generator = QueryGenerator(client=openai_client, model=MODEL_NAME)

    # Run all strategies
    all_results = []

    for name, kwargs in [
        ("Baseline (Vector Only)", {}),
        ("Raw + Reranker", {}),
        ("Rewriting + Reranker", {"query_rewriter": query_rewriter}),
        ("Multi-Query + Reranker", {"query_generator": query_generator}),
        ("Full Pipeline", {"query_rewriter": query_rewriter, "query_generator": query_generator}),
    ]:
        results = evaluate_strategy(name, gold_data, rag_index, reranker, **kwargs)
        all_results.append(results)

    # ===========================================
    # FINAL COMPARISON TABLE
    # ===========================================
    llm_calls = {
        "Baseline (Vector Only)": "0",
        "Raw + Reranker": "0",
        "Rewriting + Reranker": "1/query",
        "Multi-Query + Reranker": "1/query",
        "Full Pipeline": "2/query"
    }

    print("\n\n" + "="*85)
    print(f"{'FINAL COMPARISON RESULTS':^85}")
    print("="*85)
    print(f"{'Strategy':<30} | {'Hit Rate':<10} | {'Recall@K':<10} | {'MRR':<10} | {'LLM Calls':<10}")
    print("-"*85)

    for r in all_results:
        print(f"{r['strategy']:<30} | {r['hit_rate']:.3f}      | {r['recall@k']:.3f}      | {r['mrr']:.3f}      | {llm_calls.get(r['strategy'], '?')}")

    print("="*85)

    best = max(all_results, key=lambda x: x['hit_rate'])
    print(f"\nBest Hit Rate:  {best['strategy']} ({best['hit_rate']:.3f})")

    best_recall = max(all_results, key=lambda x: x['recall@k'])
    print(f"Best Recall@K:  {best_recall['strategy']} ({best_recall['recall@k']:.3f})")

    best_mrr = max(all_results, key=lambda x: x['mrr'])
    print(f"Best MRR:       {best_mrr['strategy']} ({best_mrr['mrr']:.3f})")

    # Save results
    output_path = SCRIPT_DIR / "variations_results.txt"
    with open(output_path, "w") as f:
        f.write("RETRIEVAL STRATEGY COMPARISON RESULTS\n")
        f.write("="*85 + "\n")
        f.write(f"{'Strategy':<30} | {'Hit Rate':<10} | {'Recall@K':<10} | {'MRR':<10} | {'LLM Calls':<10}\n")
        f.write("-"*85 + "\n")
        for r in all_results:
            f.write(f"{r['strategy']:<30} | {r['hit_rate']:.3f}      | {r['recall@k']:.3f}      | {r['mrr']:.3f}      | {llm_calls.get(r['strategy'], '?')}\n")
        f.write("="*85 + "\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
