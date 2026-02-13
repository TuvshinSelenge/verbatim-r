"""
Benchmark Script using Zilliz SemanticHighlightExtractor for Span Extraction.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from statistics import mean

from openai import OpenAI
from dotenv import load_dotenv

from custom.setup import connect_to_index, BGEReranker, QueryRewriter, QueryGenerator
from custom.pipeline.retrieval import retrieve_and_rerank as shared_retrieve_and_rerank
from custom.pipeline.metrics import compute_exact_match, token_metrics
from custom.pipeline.runtime import run_with_timeout
from custom.pipeline.types import SearchResultWrapper

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

try:
    from verbatim_rag.extractors import SemanticHighlightExtractor
except ImportError:
    print("ERROR: SemanticHighlightExtractor not found!")
    print("Please update verbatim-rag: pip install --upgrade verbatim-rag")
    sys.exit(1)

DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRIEVAL_MODELS = ["google/gemini-3-flash-preview"]
TOP_K = 5
PER_SUBQ_K = 20
QUERY_TIMEOUT = 120

# Threshold 0.3: Keeps sentences with >30% probability score.
# Threshold Default: Keeps sentences with >50% probability score.
ZILLIZ_CONFIGS = [
    {"name": "sentences-0.3", "output_mode": "sentences", "threshold": 0.3},
    {"name": "sentences-0.5", "output_mode": "sentences", "threshold": 0.5},
]


def wrap_chunks(chunks: list) -> List[SearchResultWrapper]:
    # The extractor expects objects with a `.text` attribute.
    wrapped = []
    for c in chunks:
        txt = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
        if txt:
            wrapped.append(SearchResultWrapper(text=txt))
    return wrapped


def evaluate_span_extraction(extracted_spans: List[str], gold_spans: List[str]) -> Dict[str, float]:
    is_unanswerable = len(gold_spans) == 0
    if is_unanswerable:
        abstained = len(extracted_spans) == 0
        return {
            "exact_match": 1.0 if abstained else 0.0,
            "precision": 1.0 if abstained else 0.0,
            "recall": 1.0 if abstained else 0.0,
            "f1": 1.0 if abstained else 0.0,
            "is_unanswerable": True,
            "correctly_abstained": abstained,
        }
    if not extracted_spans:
        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "is_unanswerable": False, "correctly_abstained": False}
    for pred in extracted_spans:
        for gold in gold_spans:
            if compute_exact_match(pred, gold):
                return {
                    "exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                    "is_unanswerable": False, "correctly_abstained": False,
                    "matched_pred": pred, "matched_gold": gold, "match_type": "EM",
                }
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_pred = ""
    best_gold = ""
    for pred in extracted_spans:
        for gold in gold_spans:
            p, r, f1 = token_metrics(pred, gold)
            if f1 > best_f1:
                best_precision = p
                best_recall = r
                best_f1 = f1
                best_pred = pred
                best_gold = gold
    return {
        "exact_match": 0.0, "precision": best_precision, "recall": best_recall, "f1": best_f1,
        "is_unanswerable": False, "correctly_abstained": False, "matched_pred": best_pred, "matched_gold": best_gold, "match_type": "F1",
    }


def run_zilliz_extraction_evaluation(
    gold_data: List[Dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
    zilliz_extractor: SemanticHighlightExtractor,
    retrieval_cache: Optional[Dict[str, Tuple[list, str]]] = None,
) -> Tuple[Dict, List[Dict]]:
    # Cache stores retrieval output per query:
    #   query -> (reranked_chunks, rewritten_query)
    # This lets us run extraction repeatedly with different extractor settings
    # without paying retrieval cost again.
    retrieval_cache = retrieval_cache or {}
    all_em, all_precision, all_recall, all_f1, unanswerable_correct = [], [], [], [], []
    per_query_details = []
    total_queries = len(gold_data)

    for i, item in enumerate(gold_data, 1):
        query = item["query"]
        print(f"[span {i}/{total_queries}] handling query: {query[:80]}")
        gold_spans = item.get("top_spans", [])
        if isinstance(gold_spans, str):
            gold_spans = [gold_spans] if gold_spans else []
        is_unanswerable = len(gold_spans) == 0
        detail = {"idx": i, "query": query, "gold_spans": gold_spans, "is_unanswerable": is_unanswerable}

        try:
            # Step 1: Retrieve chunks once (or reuse from cache).
            cached = retrieval_cache.get(query)
            if cached:
                reranked, rewritten = cached
            else:
                reranked, rewritten, _ = run_with_timeout(
                    lambda q=query: shared_retrieve_and_rerank(
                        q, query_rewriter, query_generator, rag_index, reranker, top_k=TOP_K, per_subq_k=PER_SUBQ_K
                    ),
                    timeout_sec=QUERY_TIMEOUT,
                )
                retrieval_cache[query] = (reranked, rewritten)

            # Step 2: Convert retrieval results to extractor input shape.
            wrapped_chunks = wrap_chunks(reranked)
            if not wrapped_chunks:
                extracted_spans = []
            else:
                # Step 3: Run span extraction with the current Zilliz config.
                def do_extract(c=wrapped_chunks, r=rewritten):
                    extraction_result = zilliz_extractor.extract_spans(r, c)
                    spans = []
                    for _, span_list in extraction_result.items():
                        for s in span_list:
                            if s and s.strip():
                                spans.append(s.strip())
                    return spans
                extracted_spans = run_with_timeout(do_extract, timeout_sec=QUERY_TIMEOUT)

            # Step 4: Score extracted spans against gold spans.
            metrics = evaluate_span_extraction(extracted_spans, gold_spans)
            all_em.append(metrics["exact_match"])
            all_precision.append(metrics["precision"])
            all_recall.append(metrics["recall"])
            all_f1.append(metrics["f1"])
            detail["extracted_spans"] = extracted_spans
            detail["metrics"] = metrics
            if is_unanswerable:
                unanswerable_correct.append(1 if metrics["correctly_abstained"] else 0)
        except Exception as e:
            print("  error during extraction eval, counting as 0")
            detail["extracted_spans"] = []
            detail["metrics"] = {"exact_match": 0, "precision": 0, "recall": 0, "f1": 0}
            detail["status"] = "ERROR"
            detail["error"] = str(e)
            all_em.append(0.0)
            all_precision.append(0.0)
            all_recall.append(0.0)
            all_f1.append(0.0)
            if is_unanswerable:
                unanswerable_correct.append(0)
            time.sleep(2)
        per_query_details.append(detail)
        time.sleep(1)

    aggregate = {
        "exact_match": mean(all_em) if all_em else 0.0,
        "precision": mean(all_precision) if all_precision else 0.0,
        "recall": mean(all_recall) if all_recall else 0.0,
        "f1": mean(all_f1) if all_f1 else 0.0,
        "unanswerable_accuracy": mean(unanswerable_correct) if unanswerable_correct else 1.0,
    }
    return aggregate, per_query_details


def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
    span_data_path = DATA_DIR / "span.json"
    if not span_data_path.exists():
        print(f"ERROR: Span data not found at {span_data_path}")
        sys.exit(1)
    span_data = json.loads(span_data_path.read_text())

    # Shared retrieval infrastructure (used by all models/configs).
    rag_index, _ = connect_to_index(db_path=DB_PATH, verbose=False)
    reranker = BGEReranker()
    results_table = []

    # Retrieval is independent from Zilliz thresholds/output mode.
    # Keep one retrieval cache per retrieval model and reuse across configs.
    retrieval_cache_by_model: Dict[str, Dict[str, Tuple[list, str]]] = {}

    for zilliz_config in ZILLIZ_CONFIGS:
        print(f"\n=== Extractor config: {zilliz_config['name']} ===")
        zilliz_extractor = SemanticHighlightExtractor(
            model_name="zilliz/semantic-highlight-bilingual-v1",
            threshold=zilliz_config["threshold"],
            output_mode=zilliz_config["output_mode"],
        )
        for retrieval_model in RETRIEVAL_MODELS:
            print(f"--- Retrieval model: {retrieval_model} ---")
            retrieval_cache = retrieval_cache_by_model.setdefault(retrieval_model, {})
            client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, timeout=120.0)
            query_rewriter = QueryRewriter(openai_client=client, model=retrieval_model)
            query_generator = QueryGenerator(client=client, model=retrieval_model)
            try:
                span_metrics, query_details = run_zilliz_extraction_evaluation(
                    span_data, rag_index, reranker, query_rewriter, query_generator, zilliz_extractor, retrieval_cache
                )
            except Exception:
                span_metrics = {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "unanswerable_accuracy": 0.0}
                query_details = []
            results_table.append({
                "Extractor": zilliz_config["name"],
                "Retrieval": retrieval_model.split("/")[-1],
                "EM": span_metrics.get("exact_match", 0),
                "Prec": span_metrics.get("precision", 0),
                "Rec": span_metrics.get("recall", 0),
                "F1": span_metrics.get("f1", 0),
                "Unans.Acc": span_metrics.get("unanswerable_accuracy", 0),
                "details": query_details,
            })

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append(f"{'ZILLIZ EXTRACTOR BENCHMARK RESULTS':^100}")
    report_lines.append("=" * 100)
    header = f"{'Extractor':<25} | {'Retrieval':<25} | {'EM':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Unans.Acc':<9}"
    report_lines.append(header)
    report_lines.append("-" * 100)
    for row in results_table:
        report_lines.append(
            f"{row['Extractor']:<25} | {row['Retrieval']:<25} | {row['EM']:.3f}  | {row['Prec']:.3f}  | {row['Rec']:.3f}  | {row['F1']:.3f}  | {row['Unans.Acc']:.3f}"
        )
    report_lines.append("=" * 100)
    print("\n".join(report_lines))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "SemanticHighlighter_results.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
