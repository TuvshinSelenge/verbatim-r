"""
Extraction Relevance Benchmark (Full Pipeline + Gemini)
=======================================================

Purpose
-------
Evaluate extracted spans with a set-based best-match method.

For ROUGE-L:
1) Recall: for each gold span, find the best matching predicted span, then average.
2) Precision: for each predicted span, find the best matching gold span, then average.
3) F1: harmonic mean of precision and recall.

Also reports:
- Unanswerable accuracy

Pipeline:
rewrite -> multi-query -> retrieve -> rerank -> extract spans
"""

import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import openai
from dotenv import load_dotenv
from openai import OpenAI

from custom.pipeline.retrieval import retrieve_and_rerank
from custom.pipeline.runtime import run_with_timeout
from custom.setup import BGEReranker, QueryGenerator, QueryRewriter, connect_to_index
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.llm_client import LLMClient

load_dotenv()

# --------------------------------------------------------------------------------------
# Paths and runtime configuration
# --------------------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_NAME = "google/gemini-3-flash-preview"
TOP_K = 5
PER_SUBQ_K = 20
QUERY_TIMEOUT = 120
MAX_EXTRACTED_SPANS = 5


def normalize_ws(text: str) -> str:
    """Trim and collapse whitespace so span matching is less brittle."""
    return " ".join((text or "").split())


def flatten_extracted_spans(spans_raw: dict) -> List[str]:
    """Flatten extractor output dict -> list of non-empty strings."""
    extracted: List[str] = []
    if not isinstance(spans_raw, dict):
        return extracted
    for _, span_list in spans_raw.items():
        if not isinstance(span_list, list):
            continue
        for span in span_list:
            span = normalize_ws(span)
            if span:
                extracted.append(span)
    return extracted


# --------------------------------------------------------------------------------------
# Pairwise ROUGE-L scorer
# --------------------------------------------------------------------------------------

# Compute Longest Common Subsequence (LCS) length between two token lists.
# - Returns 0 immediately if either list is empty.
# - Swaps inputs so tokens_b is the shorter list.
# - Uses DP with two rows:
#     prev[j] = LCS length for previous token_a prefix vs tokens_b[:j]
#     curr[j] = current row being built
# - Transition:
#     if tokens match -> prev[j-1] + 1
#     else            -> max(left, up) = max(curr[-1], prev[j])
# - After processing all tokens, prev[-1] is the final LCS length.

def _lcs_length(tokens_a: List[str], tokens_b: List[str]) -> int:
    """Longest common subsequence length between two token lists."""
    if not tokens_a or not tokens_b:
        return 0
    if len(tokens_b) > len(tokens_a):
        tokens_a, tokens_b = tokens_b, tokens_a

    prev = [0] * (len(tokens_b) + 1)
    for tok_a in tokens_a:
        curr = [0]
        for j, tok_b in enumerate(tokens_b, start=1):
            if tok_a == tok_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def rouge_l_pair_score(candidate: str, reference: str) -> float:
    """
    ROUGE-L F1 for one candidate-reference span pair.

    This gives a soft match score [0, 1] used inside best-match aggregation.
    """
    cand = normalize_ws(candidate)
    ref = normalize_ws(reference)

    if not cand and not ref:
        return 1.0
    if not cand or not ref:
        return 0.0

    cand_tokens = cand.split()
    ref_tokens = ref.split()
    lcs = _lcs_length(cand_tokens, ref_tokens)

    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------------------
# Set-based best-match aggregation
# --------------------------------------------------------------------------------------

def compute_set_best_match_prf(
    predicted_spans: List[str],
    gold_spans: List[str],
) -> Tuple[float, float, float]:
    """
    Compute set-based best-match precision/recall/F1.

    What happens:
    - Recall side: each gold span is "covered" by its best predicted span.
    - Precision side: each predicted span is "justified" by its best gold span.
    - F1 balances both sides.
    """
    preds = [normalize_ws(s) for s in predicted_spans if normalize_ws(s)]
    golds = [normalize_ws(s) for s in gold_spans if normalize_ws(s)]

    if not preds and not golds:
        return 1.0, 1.0, 1.0
    if not preds or not golds:
        return 0.0, 0.0, 0.0

    # Recall: for each gold, keep only the highest matching predicted span.
    recall_scores = [max(rouge_l_pair_score(pred, gold) for pred in preds) for gold in golds]
    # Precision: for each prediction, keep only the highest matching gold span.
    precision_scores = [max(rouge_l_pair_score(pred, gold) for gold in golds) for pred in preds]

    recall = mean(recall_scores) if recall_scores else 0.0
    precision = mean(precision_scores) if precision_scores else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_answer_relevance_benchmark(
    span_data: List[Dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
    rag: VerbatimRAG,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Run full pipeline per query and score answerable queries with:
    - ROUGE-L (set-based best-match P/R/F1)

    Unanswerable queries are evaluated with abstention accuracy.
    """
    rouge_p_all, rouge_r_all, rouge_f1_all = [], [], []
    unanswerable_correct = []
    details = []

    total = len(span_data)
    for i, item in enumerate(span_data, 1):
        query = item["query"]
        gold_spans = item.get("top_spans", [])
        if isinstance(gold_spans, str):
            gold_spans = [gold_spans] if gold_spans else []
        is_unanswerable = len(gold_spans) == 0

        print(f"[extraction-relevance {i}/{total}] {query[:80]}")
        row = {"idx": i, "query": query, "is_unanswerable": is_unanswerable, "gold_spans": gold_spans}

        try:
            chunks, rewritten, _ = run_with_timeout(
                lambda q=query: retrieve_and_rerank(
                    q,
                    query_rewriter,
                    query_generator,
                    rag_index,
                    reranker,
                    top_k=TOP_K,
                    per_subq_k=PER_SUBQ_K,
                ),
                timeout_sec=QUERY_TIMEOUT,
            )

            spans_raw = run_with_timeout(
                lambda q=query, c=chunks: rag.extractor.extract_spans(q, c),
                timeout_sec=QUERY_TIMEOUT,
            )
            extracted_spans = flatten_extracted_spans(spans_raw)
            # Hard cap: keep only the top-N spans returned by the extractor.
            extracted_spans = extracted_spans[:MAX_EXTRACTED_SPANS]

            if not is_unanswerable:
                r_p, r_r, r_f1 = compute_set_best_match_prf(
                    extracted_spans,
                    gold_spans,
                )

                rouge_p_all.append(r_p)
                rouge_r_all.append(r_r)
                rouge_f1_all.append(r_f1)

                row["rouge_l_precision"] = r_p
                row["rouge_l_recall"] = r_r
                row["rouge_l_f1"] = r_f1
            else:
                # For unanswerable queries, success means extracting nothing.
                abstained = len(extracted_spans) == 0
                unanswerable_correct.append(1 if abstained else 0)
                row["abstained_correctly"] = abstained

            row["rewritten_query"] = rewritten
            row["extracted_spans"] = extracted_spans
            row["num_extracted_spans"] = len(extracted_spans)
            row["status"] = "OK"
        except Exception as e:
            row["status"] = "ERROR"
            row["error"] = str(e)
            if not is_unanswerable:
                rouge_p_all.append(0.0)
                rouge_r_all.append(0.0)
                rouge_f1_all.append(0.0)
            else:
                unanswerable_correct.append(0)

        details.append(row)

    aggregate = {
        "rouge_l_precision": mean(rouge_p_all) if rouge_p_all else 0.0,
        "rouge_l_recall": mean(rouge_r_all) if rouge_r_all else 0.0,
        "rouge_l_f1": mean(rouge_f1_all) if rouge_f1_all else 0.0,
        "unanswerable_accuracy": mean(unanswerable_correct) if unanswerable_correct else 1.0,
        "num_answerable": len(rouge_f1_all),
        "num_unanswerable": len(unanswerable_correct),
    }
    return aggregate, details


def main():
    print("=== EXTRACTION RELEVANCE BENCHMARK (SET-BASED ROUGE-L) ===")
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

    span_data_path = DATA_DIR / "span.json"
    if not span_data_path.exists():
        print("ERROR: required data file custom/data/span.json is missing.")
        sys.exit(1)
    span_data = json.loads(span_data_path.read_text())

    rag_index, _ = connect_to_index(db_path=DB_PATH, verbose=False)
    reranker = BGEReranker()
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, timeout=120.0)
    query_rewriter = QueryRewriter(openai_client=client, model=MODEL_NAME)
    query_generator = QueryGenerator(client=client, model=MODEL_NAME)

    llm_client = LLMClient(model=MODEL_NAME, api_base=OPENROUTER_BASE_URL)
    llm_client.client = client
    llm_client.async_client = openai.AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        timeout=120.0,
    )
    rag = VerbatimRAG(rag_index, llm_client=llm_client, extraction_mode="individual")

    aggregate, details = run_answer_relevance_benchmark(
        span_data=span_data,
        rag_index=rag_index,
        reranker=reranker,
        query_rewriter=query_rewriter,
        query_generator=query_generator,
        rag=rag,
    )

    report_lines = []
    report_lines.append("=" * 132)
    report_lines.append(f"{'EXTRACTION RELEVANCE BENCHMARK RESULTS':^132}")
    report_lines.append("=" * 132)
    report_lines.append(
        f"{'Model':<35} | "
        f"{'ROUGE-L(P/R/F1)':<22} | "
        f"{'Unans.Acc':<9}"
    )
    report_lines.append("-" * 132)
    report_lines.append(
        f"{MODEL_NAME:<35} | "
        f"{aggregate['rouge_l_precision']:.3f}/{aggregate['rouge_l_recall']:.3f}/{aggregate['rouge_l_f1']:.3f} | "
        f"{aggregate['unanswerable_accuracy']:.3f}"
    )
    report_lines.append("-" * 132)
    report_lines.append(
        f"Answerable queries scored with set-based ROUGE-L: {aggregate['num_answerable']}, "
        f"Unanswerable queries: {aggregate['num_unanswerable']}"
    )
    report_lines.append("=" * 132)

    print("\n".join(report_lines))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_out = RESULTS_DIR / "answer_relevance_results.txt"
    json_out = RESULTS_DIR / "answer_relevance_details.json"

    with open(txt_out, "w") as f:
        f.write("\n".join(report_lines))
    with open(json_out, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "aggregate": aggregate,
                "per_query": details,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {txt_out}")
    print(f"Per-query details saved to: {json_out}")


if __name__ == "__main__":
    main()
