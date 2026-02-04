"""
Retrieval Strategy Comparison Script
====================================
Compares different retrieval strategies for Hit Rate and MRR:
1. Raw Query + Reranker (No LLM)
2. Query Rewriting + Reranker
3. Multi-Query Variations + Reranker
4. Rewriting + Multi-Query + Reranker (Full Pipeline)

Uses the evaluation data from custom/test_data/qa_with_chunk_ids.json
"""

import json
import os
import sys
from pathlib import Path
from statistics import mean

# Setup paths
SCRIPT_DIR = Path(__file__).parent          # custom/test/
PROJECT_ROOT = SCRIPT_DIR.parent            # custom/
REPO_ROOT = PROJECT_ROOT.parent             # verbatim-r/
SETUP_DIR = PROJECT_ROOT / "set-up"         # custom/set-up/

# Add paths for imports
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SETUP_DIR))

# Import from set-up modules (now accessible via sys.path)
from connect_index import connect_to_index
from query_rewriter import QueryRewriter
from bge_ranker import BGEReranker
from openai import OpenAI

# Database path - adjust this to your actual database location
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))

# Configuration
TOP_K = 5
PER_SUBQ_K = 20
SEARCH_K = 50  # For raw query strategy
SKIP_SENTINEL_1300 = True

# OpenRouter Configuration (for LLM-based strategies)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-3-flash-preview"


def get_text_and_meta(chunk):
    """Extract text and metadata from chunk (supports dict or object)."""
    if isinstance(chunk, dict):
        return chunk.get("text", ""), chunk.get("metadata", {}) or {}
    return getattr(chunk, "text", ""), getattr(chunk, "metadata", {}) or {}


def generate_search_queries(
    question: str,
    client: OpenAI,
    bank_name: str = "Raiffeisen Bank International AG",
    bank_short: str = "RBI",
    model: str = "gpt-5.1"
) -> list[str]:
    """Generate multiple search queries for better retrieval."""
    prompt = f"""
You generate search queries to retrieve relevant chunks from a bank annual report for: {bank_name} ({bank_short}).

Return JSON only with this schema:
{{"queries": ["q1","q2","q3"]}}

<Rules>
- Make 3 queries max
- Do NOT start queries with: Confirm / Please / Advise
- Use short keyword-ish phrases that would appear in annual reports
- Include 1 exact-name query with the bank name or short name when helpful
- Include variants with synonyms / acronyms (e.g. O-SII, G-SIB, systemic risk buffer, credit rating Moody's, etc.)
</Rules>

Question: {question}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    queries = [q.strip() for q in data.get("queries", []) if q and q.strip()]
    
    # Dedupe while preserving order
    out, seen = [], set()
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


# =============================================================================
# STRATEGY 1: Raw Query + Reranker (No LLM)
# =============================================================================
def collect_hits_raw(query_text: str, rag_index, reranker: BGEReranker) -> list[tuple]:
    """
    Strategy 1: Direct vector search + reranking. No LLM involved.
    """
    # Direct search with raw query
    hits = rag_index.query(query_text, k=SEARCH_K)
    
    # Rerank results
    _, reranked = reranker.rerank(query_text, hits, top_k=TOP_K)
    
    # Extract predictions
    preds = []
    for h in reranked:
        _, m = get_text_and_meta(h)
        preds.append((m.get("source_file"), m.get("chunk_index")))
    
    return preds


# =============================================================================
# STRATEGY 2: Query Rewriting + Reranker
# =============================================================================
def collect_hits_rewriting(
    query_text: str,
    query_rewriter: QueryRewriter,
    rag_index,
    reranker: BGEReranker
) -> list[tuple]:
    """
    Strategy 2: Rewrite query for better search, then rerank.
    """
    # Rewrite the query
    rewritten = query_rewriter.rewrite(query_text)
    
    # Search with rewritten query
    hits = rag_index.query(rewritten, k=SEARCH_K)
    
    # Rerank results (against rewritten query)
    _, reranked = reranker.rerank(rewritten, hits, top_k=TOP_K)
    
    # Extract predictions
    preds = []
    for h in reranked:
        _, m = get_text_and_meta(h)
        preds.append((m.get("source_file"), m.get("chunk_index")))
    
    return preds


# =============================================================================
# STRATEGY 3: Multi-Query Variations + Reranker (No Rewriting)
# =============================================================================
def collect_hits_multiquery(
    query_text: str,
    rag_index,
    reranker: BGEReranker,
    openai_client: OpenAI,
    model_name: str
) -> list[tuple]:
    """
    Strategy 3: Generate query variations, search all, merge, then rerank.
    Uses the raw query (no rewriting) for variations.
    """
    # Generate search query variations from raw query
    subqs = generate_search_queries(query_text, openai_client, model=model_name)
    
    # Merge results from all sub-queries
    merged, seen = [], set()
    for q in subqs:
        hits = rag_index.query(q, k=PER_SUBQ_K)
        for h in hits:
            t, m = get_text_and_meta(h)
            if not t:
                continue
            key = (m.get("source_file"), m.get("chunk_index"), t[:200])
            if key in seen:
                continue
            seen.add(key)
            merged.append(h)
    
    # Rerank results (against original query)
    _, reranked = reranker.rerank(query_text, merged, top_k=TOP_K)
    
    # Extract predictions
    preds = []
    for h in reranked:
        _, m = get_text_and_meta(h)
        preds.append((m.get("source_file"), m.get("chunk_index")))
    
    return preds


# =============================================================================
# STRATEGY 4: Rewriting + Multi-Query + Reranker (Full Pipeline)
# =============================================================================
def collect_hits_full_pipeline(
    query_text: str,
    query_rewriter: QueryRewriter,
    rag_index,
    reranker: BGEReranker,
    openai_client: OpenAI,
    model_name: str
) -> list[tuple]:
    """
    Strategy 4: Full pipeline - rewrite, then generate variations, merge, rerank.
    This is the original chunk_test_results.py approach.
    """
    # Rewrite the query for better retrieval
    rewritten = query_rewriter.rewrite(query_text)
    
    # Generate multiple search queries from rewritten query
    subqs = generate_search_queries(rewritten, openai_client, model=model_name)
    
    # Merge results from all sub-queries
    merged, seen = [], set()
    for q in subqs:
        hits = rag_index.query(q, k=PER_SUBQ_K)
        for h in hits:
            t, m = get_text_and_meta(h)
            if not t:
                continue
            key = (m.get("source_file"), m.get("chunk_index"), t[:200])
            if key in seen:
                continue
            seen.add(key)
            merged.append(h)
    
    # Rerank results (against rewritten query)
    _, reranked = reranker.rerank(rewritten, merged, top_k=TOP_K)
    
    # Extract predictions
    preds = []
    for h in reranked:
        _, m = get_text_and_meta(h)
        preds.append((m.get("source_file"), m.get("chunk_index")))
    
    return preds


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================
def evaluate_strategy(
    strategy_name: str,
    gold_data: list[dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter = None,
    openai_client: OpenAI = None,
    model_name: str = "gpt-5.1"
) -> dict:
    """
    Evaluate a retrieval strategy and return Hit Rate and MRR.
    """
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
        
        # Handle single index vs list
        if not isinstance(expected_idxs, list):
            expected_idxs = [expected_idxs]
        
        # Skip sentinel questions (no answer in documents)
        if SKIP_SENTINEL_1300 and all(idx == 1300 for idx in expected_idxs):
            skipped += 1
            continue
        
        gold_idxs = set(expected_idxs)
        
        print(f"[{i}/{total_queries}] Evaluating: {query[:50]}...")
        
        try:
            # Select strategy
            if strategy_name == "Raw + Reranker":
                preds = collect_hits_raw(query, rag_index, reranker)
            elif strategy_name == "Rewriting + Reranker":
                preds = collect_hits_rewriting(query, query_rewriter, rag_index, reranker)
            elif strategy_name == "Multi-Query + Reranker":
                preds = collect_hits_multiquery(query, rag_index, reranker, openai_client, model_name)
            elif strategy_name == "Full Pipeline":
                preds = collect_hits_full_pipeline(query, query_rewriter, rag_index, reranker, openai_client, model_name)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            per_query.append({
                "query": query,
                "hit@k": 0,
                "rr": 0.0,
                "error": str(e)
            })
            continue
        
        # Calculate Hit@K
        hit = any(idx in gold_idxs for _, idx in preds)
        
        # Calculate Reciprocal Rank (for MRR)
        rank = None
        for r, (_, idx) in enumerate(preds, 1):
            if idx in gold_idxs:
                rank = r
                break
        rr = 1.0 / rank if rank else 0.0
        
        per_query.append({
            "query": query,
            "hit@k": 1 if hit else 0,
            "rr": rr,
        })
        
        status = "HIT" if hit else "MISS"
        print(f"  Result: {status} | RR: {rr:.3f}")
    
    # Calculate aggregate metrics
    if per_query:
        hit_rate = mean([r["hit@k"] for r in per_query])
        mrr = mean([r["rr"] for r in per_query])
    else:
        hit_rate = 0.0
        mrr = 0.0
    
    return {
        "strategy": strategy_name,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "num_evaluated": len(per_query),
        "num_skipped": skipped,
    }


def main():
    """Main evaluation function."""
    print("="*70)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("="*70)
    print("Comparing: Raw, Rewriting, Multi-Query, and Full Pipeline")
    print("="*70)
    
    # Load gold data - use test_data folder
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
    
    # Initialize OpenAI client (for LLM-based strategies)
    print("\nInitializing OpenRouter client...")
    openai_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )
    
    # Initialize query rewriter
    query_rewriter = QueryRewriter(openai_client=openai_client, model=MODEL_NAME)
    
    # Store all results
    all_results = []
    
    # ===========================================
    # STRATEGY 1: Raw + Reranker (NO LLM)
    # ===========================================
    results = evaluate_strategy(
        "Raw + Reranker",
        gold_data,
        rag_index,
        reranker
    )
    all_results.append(results)
    
    # ===========================================
    # STRATEGY 2: Rewriting + Reranker
    # ===========================================
    results = evaluate_strategy(
        "Rewriting + Reranker",
        gold_data,
        rag_index,
        reranker,
        query_rewriter=query_rewriter
    )
    all_results.append(results)
    
    # ===========================================
    # STRATEGY 3: Multi-Query + Reranker
    # ===========================================
    results = evaluate_strategy(
        "Multi-Query + Reranker",
        gold_data,
        rag_index,
        reranker,
        openai_client=openai_client,
        model_name=MODEL_NAME
    )
    all_results.append(results)
    
    # ===========================================
    # STRATEGY 4: Full Pipeline
    # ===========================================
    results = evaluate_strategy(
        "Full Pipeline",
        gold_data,
        rag_index,
        reranker,
        query_rewriter=query_rewriter,
        openai_client=openai_client,
        model_name=MODEL_NAME
    )
    all_results.append(results)
    
    # ===========================================
    # FINAL COMPARISON TABLE
    # ===========================================
    print("\n\n" + "="*70)
    print(f"{'FINAL COMPARISON RESULTS':^70}")
    print("="*70)
    print(f"{'Strategy':<30} | {'Hit Rate':<10} | {'MRR':<10} | {'LLM Calls':<10}")
    print("-"*70)
    
    llm_calls = {
        "Raw + Reranker": "0",
        "Rewriting + Reranker": "1/query",
        "Multi-Query + Reranker": "1/query",
        "Full Pipeline": "2/query"
    }
    
    for r in all_results:
        print(f"{r['strategy']:<30} | {r['hit_rate']:.3f}      | {r['mrr']:.3f}      | {llm_calls.get(r['strategy'], '?')}")
    
    print("="*70)
    
    # Find best strategy
    best = max(all_results, key=lambda x: x['hit_rate'])
    print(f"\nBest Hit Rate: {best['strategy']} ({best['hit_rate']:.3f})")
    
    best_mrr = max(all_results, key=lambda x: x['mrr'])
    print(f"Best MRR:      {best_mrr['strategy']} ({best_mrr['mrr']:.3f})")
    
    # Save results to file
    output_path = SCRIPT_DIR / "variations_results.txt"
    with open(output_path, "w") as f:
        f.write("RETRIEVAL STRATEGY COMPARISON RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"{'Strategy':<30} | {'Hit Rate':<10} | {'MRR':<10} | {'LLM Calls':<10}\n")
        f.write("-"*70 + "\n")
        for r in all_results:
            f.write(f"{r['strategy']:<30} | {r['hit_rate']:.3f}      | {r['mrr']:.3f}      | {llm_calls.get(r['strategy'], '?')}\n")
        f.write("="*70 + "\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
