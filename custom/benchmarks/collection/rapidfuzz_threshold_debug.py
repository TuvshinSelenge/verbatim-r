#!/usr/bin/env python3
"""
Debug RapidFuzz threshold matching using the same retrieval+extraction pipeline
as custom/benchmarks/collection/model_benchmark.py.

Usage:
  python custom/benchmarks/collection/rapidfuzz_threshold_debug.py
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
import openai
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from custom.setup import connect_to_index, BGEReranker, QueryRewriter, QueryGenerator
from custom.pipeline.retrieval import retrieve_and_rerank
from custom.pipeline.metrics import flatten_extracted_spans, normalize_answer
from custom.pipeline.io import run_with_timeout
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.llm_client import LLMClient

# --- same-ish constants as model_benchmark.py ---
TOP_K = 5
PER_SUBQ_K = 20
QUERY_TIMEOUT = 120
MAX_EXTRACTED_SPANS = 5
THRESHOLDS = [0.50, 0.75, 0.90, 0.95]  # Lenient→strict sweep matching benchmark thresholds


def normalize_spans(spans: List[str]) -> List[str]:
    out = []
    for s in spans:
        n = normalize_answer(s)
        if n:
            out.append(n)
    return out


def pair_matrix(preds: List[str], golds: List[str]) -> List[List[float]]:
    return [[fuzz.WRatio(p, g) / 100.0 for g in golds] for p in preds]


def match_at_threshold(
    preds: List[str],
    golds: List[str],
    sim: List[List[float]],
    t: float,
) -> Dict:
    pred_n = len(preds)
    gold_n = len(golds)

    if pred_n == 0 and gold_n == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "pairs": [], "unmatched_preds": [], "unmatched_golds": []
        }
    if pred_n == 0 or gold_n == 0:
        return {
            "tp": 0, "fp": pred_n, "fn": gold_n,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "pairs": [], "unmatched_preds": list(range(pred_n)), "unmatched_golds": list(range(gold_n))
        }

    eligible = [[1.0 if s >= t else 0.0 for s in row] for row in sim]
    cost = [[-e for e in row] for row in eligible]
    row_idx, col_idx = linear_sum_assignment(cost)

    pairs = []
    used_preds, used_golds = set(), set()
    tp = 0

    for i, j in zip(row_idx, col_idx):
        used_preds.add(i)
        used_golds.add(j)
        if eligible[i][j] > 0:
            tp += 1
            pairs.append({
                "pred_idx": i,
                "gold_idx": j,
                "score": sim[i][j],
                "pred": preds[i],
                "gold": golds[j],
            })

    fp = pred_n - tp
    fn = gold_n - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    unmatched_preds = [i for i in range(pred_n) if i not in {p["pred_idx"] for p in pairs}]
    unmatched_golds = [j for j in range(gold_n) if j not in {p["gold_idx"] for p in pairs}]

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "pairs": sorted(pairs, key=lambda x: x["score"], reverse=True),
        "unmatched_preds": unmatched_preds,
        "unmatched_golds": unmatched_golds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="How many span queries to inspect")
    parser.add_argument("--model", type=str, default="google/gemini-3-flash-preview", help="OpenRouter model")
    args = parser.parse_args()

    load_dotenv()

    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    db_path = os.getenv("DB_PATH", str(project_root / "milvus_verbatim.db"))
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    openrouter_base_url = "https://openrouter.ai/api/v1"

    if not openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    # same env setup style as benchmark script
    os.environ["OPENAI_API_KEY"] = openrouter_api_key
    os.environ["OPENAI_BASE_URL"] = openrouter_base_url

    span_data = json.loads((data_dir / "span.json").read_text())

    rag_index, _ = connect_to_index(db_path=db_path, verbose=False)
    reranker = BGEReranker()

    client = OpenAI(base_url=openrouter_base_url, api_key=openrouter_api_key, timeout=120.0)
    query_rewriter = QueryRewriter(openai_client=client, model=args.model)
    query_generator = QueryGenerator(client=client, model=args.model)

    llm_client = LLMClient(model=args.model, api_base=openrouter_base_url)
    llm_client.client = client
    llm_client.async_client = openai.AsyncOpenAI(
        base_url=openrouter_base_url,
        api_key=openrouter_api_key,
        timeout=120.0,
    )
    rag = VerbatimRAG(rag_index, llm_client=llm_client)
    rag.template_manager.use_contextual_mode(use_per_fact=True)

    seen = 0
    for item in span_data:
        if seen >= args.limit:
            break

        query = item["query"]
        gold_raw = item.get("top_spans", [])
        if isinstance(gold_raw, str):
            gold_raw = [gold_raw] if gold_raw else []
        if len(gold_raw) == 0:
            continue

        print("\n" + "=" * 120)
        print(f"QUERY: {query}")

        try:
            reranked, rewritten, _ = run_with_timeout(
                lambda q=query: retrieve_and_rerank(
                    q, query_rewriter, query_generator, rag_index, reranker,
                    top_k=TOP_K, per_subq_k=PER_SUBQ_K
                ),
                timeout_sec=QUERY_TIMEOUT,
            )

            def do_extract(c=reranked, r=rewritten):
                spans_raw = rag.extractor.extract_spans(r, c)
                return flatten_extracted_spans(spans_raw)[:MAX_EXTRACTED_SPANS]

            pred_raw = run_with_timeout(do_extract, timeout_sec=QUERY_TIMEOUT)

        except Exception as e:
            print(f"ERROR: {e}")
            continue

        preds = normalize_spans(pred_raw)
        golds = normalize_spans(gold_raw)
        sim = pair_matrix(preds, golds)

        print(f"\nPRED ({len(preds)}):")
        for i, p in enumerate(preds):
            print(f"  [{i}] {p}")

        print(f"\nGOLD ({len(golds)}):")
        for j, g in enumerate(golds):
            print(f"  [{j}] {g}")

        for t in THRESHOLDS:
            result = match_at_threshold(preds, golds, sim, t)
            label = int(t * 100)
            print(f"\n--- Threshold {label}% ---")
            print(
                f"P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f} "
                f"(TP={result['tp']}, FP={result['fp']}, FN={result['fn']})"
            )

            if result["pairs"]:
                print("Matched pairs:")
                for m in result["pairs"]:
                    print(
                        f"  pred[{m['pred_idx']}] <-> gold[{m['gold_idx']}] "
                        f"score={m['score']:.3f}\n"
                        f"    pred: {m['pred']}\n"
                        f"    gold: {m['gold']}"
                    )
            else:
                print("Matched pairs: none")

            if result["unmatched_preds"]:
                print("Unmatched preds:", result["unmatched_preds"])
            if result["unmatched_golds"]:
                print("Unmatched golds:", result["unmatched_golds"])

        seen += 1


if __name__ == "__main__":
    main()