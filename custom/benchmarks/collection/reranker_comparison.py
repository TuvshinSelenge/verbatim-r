"""
Reranker Comparison Script
==========================
Compares retrieval quality across rerankers using Hit Rate, Recall@K, and MRR.
"""

import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from dotenv import load_dotenv

from custom.setup import connect_to_index, QueryRewriter, QueryGenerator
from custom.pipeline.retrieval import extract_preds, merge_and_dedup
from custom.pipeline.io import run_with_timeout

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-3-flash-preview"

TOP_K = 5
SEARCH_K = 50
PER_SUBQ_K = 20
SKIP_SENTINEL_3000 = True
QUERY_TIMEOUT = 120
RERANKERS_TO_TEST = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base"
]


class HFReranker:
    def __init__(self, model_name: str):
        # Keep reranker fully local so model-to-model comparison is consistent.
        self.model_name = model_name
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = self._resolve_max_length()

    def _resolve_max_length(self) -> int:
        tok_max = getattr(self.tokenizer, "model_max_length", 512)
        cfg_max = getattr(self.model.config, "max_position_embeddings", 512)
        if tok_max is None or tok_max <= 0 or tok_max > 100000:
            tok_max = 512
        if cfg_max is None or cfg_max <= 0 or cfg_max > 100000:
            cfg_max = 512
        return int(min(tok_max, cfg_max, 1024))

    @torch.inference_mode()
    def rerank(self, query: str, chunks: list, top_k: int = 5, text_key: str = "text"):
        if not chunks:
            return [], []
        pairs = []
        for c in chunks:
            txt = c.get(text_key, "") or c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
            pairs.append([query, txt])
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
        ).to(self.device)
        scores = self.model(**inputs, return_dict=True).logits.view(-1,).float().detach().cpu().tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        ranking = [{"id": int(i), "score": float(s)} for i, s in ranked]
        top_chunks = [chunks[i] for i, _ in ranked[:top_k]]
        return top_chunks, ranking


def _retrieve_then_rerank_preds(
    query_text: str,
    rag_index,
    reranker: HFReranker,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
) -> List[Tuple]:
    # Shared retrieval pipeline before reranker scoring.
    rewritten = query_rewriter.rewrite(query_text)
    subqs = query_generator.generate_queries(rewritten)
    merged = merge_and_dedup(subqs, rag_index, PER_SUBQ_K)
    if not merged:
        merged = rag_index.query(rewritten, k=SEARCH_K)
    reranked, _ = reranker.rerank(rewritten, merged, top_k=TOP_K)
    return extract_preds(reranked)


def evaluate_reranker(
    reranker_name: str,
    reranker: HFReranker,
    gold_data: List[Dict],
    rag_index,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
) -> dict:
    # Evaluate a single reranker over all labeled queries.
    per_query = []
    skipped = 0
    total_queries = len(gold_data)
    for i, item in enumerate(gold_data, 1):
        query = item["query"]
        print(f"[{reranker_name}] [{i}/{total_queries}] handling query: {query[:80]}")
        expected_idxs = item["expected_chunk_index"]
        if not isinstance(expected_idxs, list):
            expected_idxs = [expected_idxs]
        if SKIP_SENTINEL_3000 and all(idx == 3000 for idx in expected_idxs):
            skipped += 1
            print("  skipped (sentinel 3000)")
            continue
        gold_idxs = set(expected_idxs)
        try:
            preds = run_with_timeout(
                lambda q=query: _retrieve_then_rerank_preds(q, rag_index, reranker, query_rewriter, query_generator),
                timeout_sec=QUERY_TIMEOUT,
            )
        except Exception:
            print("  error during retrieval/rerank eval, counting as 0")
            per_query.append({"hit@k": 0, "rr": 0.0, "recall@k": 0.0})
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


def main():
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

    # Initialize shared dependencies once.
    rag_index, _ = connect_to_index(db_path=DB_PATH, verbose=False)
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, timeout=120.0)
    query_rewriter = QueryRewriter(openai_client=client, model=MODEL_NAME)
    query_generator = QueryGenerator(client=client, model=MODEL_NAME)

    all_results = []
    for model_name in RERANKERS_TO_TEST:
        print(f"\n=== Running reranker: {model_name} ===")
        try:
            # Instantiate candidate reranker and evaluate it end-to-end.
            reranker = HFReranker(model_name)
        except Exception:
            all_results.append({
                "reranker": model_name,
                "hit_rate": 0.0,
                "recall@k": 0.0,
                "mrr": 0.0,
                "num_evaluated": 0,
                "num_skipped": 0,
            })
            continue
        all_results.append(evaluate_reranker(model_name, reranker, gold_data, rag_index, query_rewriter, query_generator))

    print("\n\n" + "=" * 95)
    print(f"{'FINAL RERANKER COMPARISON':^95}")
    print("=" * 95)
    print(f"{'Reranker':<35} | {'Hit Rate':<10} | {'Recall@K':<10} | {'MRR':<10} | {'Evaluated':<10}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['reranker']:<35} | {r['hit_rate']:.3f}      | {r['recall@k']:.3f}      | {r['mrr']:.3f}      | {r['num_evaluated']:<10}")
    print("=" * 95)

    best_hit = max(all_results, key=lambda x: x["hit_rate"])
    best_recall = max(all_results, key=lambda x: x["recall@k"])
    best_mrr = max(all_results, key=lambda x: x["mrr"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "reranker_comparison_results.txt"
    with open(output_path, "w") as f:
        f.write("RERANKER COMPARISON RESULTS\n")
        f.write("=" * 95 + "\n")
        f.write(f"{'Reranker':<35} | {'Hit Rate':<10} | {'Recall@K':<10} | {'MRR':<10} | {'Evaluated':<10}\n")
        f.write("-" * 95 + "\n")
        for r in all_results:
            f.write(f"{r['reranker']:<35} | {r['hit_rate']:.3f}      | {r['recall@k']:.3f}      | {r['mrr']:.3f}      | {r['num_evaluated']:<10}\n")
        f.write("=" * 95 + "\n")
        f.write(f"Best Hit Rate:  {best_hit['reranker']} ({best_hit['hit_rate']:.3f})\n")
        f.write(f"Best Recall@K:  {best_recall['reranker']} ({best_recall['recall@k']:.3f})\n")
        f.write(f"Best MRR:       {best_mrr['reranker']} ({best_mrr['mrr']:.3f})\n")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
