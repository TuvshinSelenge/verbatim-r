"""
Reranker Comparison Script
==========================
Compares retrieval quality across rerankers using:
- Hit Rate
- Recall@K
- MRR

Pipeline per query:
1) Vector search (top SEARCH_K)
2) Rerank candidates (top TOP_K)
3) Evaluate against expected chunk indices

Output:
- Prints final table
- Saves table to custom/test/reranker_comparison_results.txt
"""

import json
import os
import sys
import time
import concurrent.futures
from pathlib import Path
from statistics import mean
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


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


# Database path
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))

# OpenRouter Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-3-flash-preview"

# Evaluation config
TOP_K = 5
SEARCH_K = 50
PER_SUBQ_K = 20
SKIP_SENTINEL_1300 = True
QUERY_TIMEOUT = 120  # seconds


# Rerankers to compare:
RERANKERS_TO_TEST = [
    "BAAI/bge-reranker-v2-m3",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]


def get_text_and_meta(chunk):
    """Extract text and metadata from chunk (supports dict or object)."""
    if isinstance(chunk, dict):
        return chunk.get("text", ""), chunk.get("metadata", {}) or {}
    return getattr(chunk, "text", ""), getattr(chunk, "metadata", {}) or {}


def extract_preds(chunks) -> List[Tuple]:
    """Extract (source_file, chunk_index) from chunks."""
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


class HFReranker:
    """Generic Hugging Face reranker using sequence classification logits."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading reranker: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = self._resolve_max_length()
        print(f"Using max_length={self.max_length} for {model_name}")
        print(f"Ready: {model_name}")

    def _resolve_max_length(self) -> int:
        """Pick a safe max length compatible with tokenizer/model limits."""
        tok_max = getattr(self.tokenizer, "model_max_length", 512)
        cfg_max = getattr(self.model.config, "max_position_embeddings", 512)

        # Some tokenizers expose huge sentinel values for "unknown" max length.
        if tok_max is None or tok_max <= 0 or tok_max > 100000:
            tok_max = 512
        if cfg_max is None or cfg_max <= 0 or cfg_max > 100000:
            cfg_max = 512

        # Keep a practical upper cap while honoring model constraints.
        return int(min(tok_max, cfg_max, 1024))

    @torch.inference_mode()
    def rerank(self, query: str, chunks: list, top_k: int = 5, text_key: str = "text"):
        """Return (top_chunks, ranking) sorted by descending score."""
        if not chunks:
            return [], []

        pairs = []
        for c in chunks:
            if isinstance(c, dict):
                txt = c.get(text_key, "") or c.get("text", "")
            else:
                txt = getattr(c, "text", "")
            pairs.append([query, txt])

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)

        scores = (
            self.model(**inputs, return_dict=True)
            .logits.view(-1,)
            .float()
            .detach()
            .cpu()
            .tolist()
        )

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranking = [{"id": int(i), "score": float(s)} for i, s in ranked]
        top_chunks = [chunks[i] for i, _ in ranked[:top_k]]
        return top_chunks, ranking


def evaluate_reranker(
    reranker_name: str,
    reranker: HFReranker,
    gold_data: List[Dict],
    rag_index,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
) -> dict:
    """Evaluate one reranker on retrieval metrics."""
    per_query = []
    skipped = 0
    total_queries = len(gold_data)

    print(f"\n{'='*70}")
    print(f"RERANKER: {reranker_name}")
    print(f"{'='*70}")
    print(
        f"Config: SEARCH_K={SEARCH_K}, PER_SUBQ_K={PER_SUBQ_K}, "
        f"TOP_K={TOP_K}, timeout={QUERY_TIMEOUT}s"
    )
    print(f"{'='*70}\n")

    for i, item in enumerate(gold_data, 1):
        query = item["query"]
        expected_idxs = item["expected_chunk_index"]
        if not isinstance(expected_idxs, list):
            expected_idxs = [expected_idxs]

        if SKIP_SENTINEL_1300 and all(idx == 1300 for idx in expected_idxs):
            skipped += 1
            continue

        gold_idxs = set(expected_idxs)
        print(f"[{i}/{total_queries}] {query[:60]}...")

        try:
            preds = run_with_timeout(
                lambda q=query: _retrieve_then_rerank_preds(
                    q, rag_index, reranker, query_rewriter, query_generator
                ),
                timeout_sec=QUERY_TIMEOUT,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            per_query.append({"hit@k": 0, "rr": 0.0, "recall@k": 0.0})
            time.sleep(1)
            continue

        # Hit@K
        hit = any(idx in gold_idxs for _, idx in preds)

        # Reciprocal rank
        rank = None
        for r, (_, idx) in enumerate(preds, 1):
            if idx in gold_idxs:
                rank = r
                break
        rr = 1.0 / rank if rank else 0.0

        # Recall@K
        retrieved_idxs = {idx for _, idx in preds}
        recall_at_k = len(gold_idxs & retrieved_idxs) / len(gold_idxs) if gold_idxs else 0.0

        per_query.append({"hit@k": 1 if hit else 0, "rr": rr, "recall@k": recall_at_k})
        status = "HIT" if hit else "MISS"
        print(f"  {status} | RR: {rr:.3f} | Recall@{TOP_K}: {recall_at_k:.3f}")
        time.sleep(0.2)

    hit_rate = mean([r["hit@k"] for r in per_query]) if per_query else 0.0
    mrr = mean([r["rr"] for r in per_query]) if per_query else 0.0
    recall_at_k = mean([r["recall@k"] for r in per_query]) if per_query else 0.0

    return {
        "reranker": reranker_name,
        "hit_rate": hit_rate,
        "recall@k": recall_at_k,
        "mrr": mrr,
        "num_evaluated": len(per_query),
        "num_skipped": skipped,
    }


def _retrieve_then_rerank_preds(
    query_text: str,
    rag_index,
    reranker: HFReranker,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
) -> List[Tuple]:
    """Rewrite + multi-query retrieval, then rerank."""
    rewritten = query_rewriter.rewrite(query_text)
    subqs = query_generator.generate_queries(rewritten)
    merged = merge_and_dedup(subqs, rag_index, PER_SUBQ_K)
    if not merged:
        merged = rag_index.query(rewritten, k=SEARCH_K)
    reranked, _ = reranker.rerank(rewritten, merged, top_k=TOP_K)
    return extract_preds(reranked)


def main():
    print("=" * 80)
    print("RERANKER COMPARISON")
    print("=" * 80)

    # Set OPENAI_API_KEY for modules that use it internally
    if OPENROUTER_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
        os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
    else:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

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

    all_results = []

    print("\nInitializing OpenRouter client...")
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        timeout=120.0,
    )
    query_rewriter = QueryRewriter(openai_client=client, model=MODEL_NAME)
    query_generator = QueryGenerator(client=client, model=MODEL_NAME)

    for model_name in RERANKERS_TO_TEST:
        try:
            reranker = HFReranker(model_name)
        except Exception as e:
            print(f"\nFAILED to initialize reranker {model_name}: {e}")
            all_results.append({
                "reranker": model_name,
                "hit_rate": 0.0,
                "recall@k": 0.0,
                "mrr": 0.0,
                "num_evaluated": 0,
                "num_skipped": 0,
            })
            continue

        results = evaluate_reranker(
            model_name,
            reranker,
            gold_data,
            rag_index,
            query_rewriter,
            query_generator,
        )
        all_results.append(results)

    # Final table
    print("\n\n" + "=" * 95)
    print(f"{'FINAL RERANKER COMPARISON':^95}")
    print("=" * 95)
    print(f"{'Reranker':<35} | {'Hit Rate':<10} | {'Recall@K':<10} | {'MRR':<10} | {'Evaluated':<10}")
    print("-" * 95)
    for r in all_results:
        print(
            f"{r['reranker']:<35} | "
            f"{r['hit_rate']:.3f}      | "
            f"{r['recall@k']:.3f}      | "
            f"{r['mrr']:.3f}      | "
            f"{r['num_evaluated']:<10}"
        )
    print("=" * 95)

    best_hit = max(all_results, key=lambda x: x["hit_rate"])
    best_recall = max(all_results, key=lambda x: x["recall@k"])
    best_mrr = max(all_results, key=lambda x: x["mrr"])
    print(f"\nBest Hit Rate:  {best_hit['reranker']} ({best_hit['hit_rate']:.3f})")
    print(f"Best Recall@K:  {best_recall['reranker']} ({best_recall['recall@k']:.3f})")
    print(f"Best MRR:       {best_mrr['reranker']} ({best_mrr['mrr']:.3f})")

    # Save to txt
    output_path = SCRIPT_DIR / "reranker_comparison_results.txt"
    with open(output_path, "w") as f:
        f.write("RERANKER COMPARISON RESULTS\n")
        f.write("=" * 95 + "\n")
        f.write(f"{'Reranker':<35} | {'Hit Rate':<10} | {'Recall@K':<10} | {'MRR':<10} | {'Evaluated':<10}\n")
        f.write("-" * 95 + "\n")
        for r in all_results:
            f.write(
                f"{r['reranker']:<35} | "
                f"{r['hit_rate']:.3f}      | "
                f"{r['recall@k']:.3f}      | "
                f"{r['mrr']:.3f}      | "
                f"{r['num_evaluated']:<10}\n"
            )
        f.write("=" * 95 + "\n")
        f.write(f"Best Hit Rate:  {best_hit['reranker']} ({best_hit['hit_rate']:.3f})\n")
        f.write(f"Best Recall@K:  {best_recall['reranker']} ({best_recall['recall@k']:.3f})\n")
        f.write(f"Best MRR:       {best_mrr['reranker']} ({best_mrr['mrr']:.3f})\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
