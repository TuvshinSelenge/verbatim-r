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
from custom.pipeline.metrics import (
    compute_bertscore_best_match_prf,
    compute_rapidfuzz_threshold_prf,
    compute_rouge_l_best_match_prf,
    get_bertscore_scorer,
)
from custom.pipeline.retrieval import retrieve_and_rerank as shared_retrieve_and_rerank
from custom.pipeline.io import (
    append_span_metric_scores,
    append_zero_span_metric_scores,
    build_benchmark_header_and_rows,
    build_table_report_lines,
    init_span_metric_lists,
    make_benchmark_result_row,
    print_and_write_report,
    run_with_timeout,
    summarize_span_metric_lists,
    zero_chunk_metrics,
    zero_span_metrics,
)
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
SKIP_SENTINEL_3000 = True
MAX_EXTRACTED_SPANS = 5

# Threshold 0.3: Keeps sentences with >30% probability score.
# Threshold Default: Keeps sentences with >50% probability score.
# Threshold 0.7: Keeps sentences with >70% probability score, resulting in more conservative extraction.
ZILLIZ_CONFIGS = [
    {"name": "sentences-0.3", "output_mode": "sentences", "threshold": 0.3}
]


def wrap_chunks(chunks: list) -> List[SearchResultWrapper]:
    wrapped = []
    for c in chunks:
        txt = c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
        if txt:
            wrapped.append(SearchResultWrapper(text=txt))
    return wrapped


def run_zilliz_extraction_evaluation(
    gold_data: List[Dict],
    retrieval_gold_map: Dict[str, List[int]],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
    zilliz_extractor: SemanticHighlightExtractor,
    bert_scorer,
    retrieval_cache: Optional[Dict[str, Tuple[list, str, List[Tuple]]]] = None,
) -> Tuple[Dict, List[Dict]]:
    # Cache stores retrieval output per query:
    # query -> (reranked_chunks, rewritten_query)
    # This lets us run extraction repeatedly with different extractor settings
    # without paying retrieval cost again.
    retrieval_cache = retrieval_cache or {}
    all_hit, all_rr, all_recall_k = [], [], []
    span_metric_lists = init_span_metric_lists()
    unanswerable_correct = []
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
                reranked, rewritten, preds = cached
            else:
                reranked, rewritten, preds = run_with_timeout(
                    lambda q=query: shared_retrieve_and_rerank(
                        q, query_rewriter, query_generator, rag_index, reranker, top_k=TOP_K, per_subq_k=PER_SUBQ_K
                    ),
                    timeout_sec=QUERY_TIMEOUT,
                )
                retrieval_cache[query] = (reranked, rewritten, preds)

            # Step 1b: Retrieval metrics against qa_with_chunk_ids gold indices.
            expected_idxs = retrieval_gold_map.get(query, [])
            if expected_idxs:
                gold_idxs = set(expected_idxs)
                hit = any(idx in gold_idxs for _, idx in preds)
                rank = None
                for r, (_, idx) in enumerate(preds, 1):
                    if idx in gold_idxs:
                        rank = r
                        break
                rr = 1.0 / rank if rank else 0.0
                retrieved_idxs = {idx for _, idx in preds}
                recall_at_k = len(gold_idxs & retrieved_idxs) / len(gold_idxs) if gold_idxs else 0.0
                all_hit.append(1 if hit else 0)
                all_rr.append(rr)
                all_recall_k.append(recall_at_k)
                detail["retrieval"] = {"hit@k": 1 if hit else 0, "rr": rr, "recall@k": recall_at_k}
            else:
                detail["retrieval"] = {"hit@k": None, "rr": None, "recall@k": None}

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
                    return spans[:MAX_EXTRACTED_SPANS]
                extracted_spans = run_with_timeout(do_extract, timeout_sec=QUERY_TIMEOUT)

            # Step 4: Score extracted spans against gold spans.
            detail["extracted_spans"] = extracted_spans
            if not is_unanswerable:
                rouge_scores = compute_rouge_l_best_match_prf(extracted_spans, gold_spans)
                bert_scores = compute_bertscore_best_match_prf(
                    extracted_spans, gold_spans, scorer=bert_scorer
                )
                rapidfuzz_scores = compute_rapidfuzz_threshold_prf(extracted_spans, gold_spans)
                append_span_metric_scores(
                    metric_lists=span_metric_lists,
                    rouge_scores=rouge_scores,
                    bert_scores=bert_scores,
                    rapidfuzz_scores=rapidfuzz_scores,
                )
                rouge_p, rouge_r, rouge_f1 = rouge_scores
                bert_p, bert_r, bert_f1 = bert_scores
                rf75_p, rf75_r, rf75_f1 = rapidfuzz_scores["75"]
                rf90_p, rf90_r, rf90_f1 = rapidfuzz_scores["90"]
                rf95_p, rf95_r, rf95_f1 = rapidfuzz_scores["95"]
                detail["metrics"] = {
                    "rouge_l_precision": rouge_p,
                    "rouge_l_recall": rouge_r,
                    "rouge_l_f1": rouge_f1,
                    "bertscore_precision": bert_p,
                    "bertscore_recall": bert_r,
                    "bertscore_f1": bert_f1,
                    "rapidfuzz_75_precision": rf75_p,
                    "rapidfuzz_75_recall": rf75_r,
                    "rapidfuzz_75_f1": rf75_f1,
                    "rapidfuzz_90_precision": rf90_p,
                    "rapidfuzz_90_recall": rf90_r,
                    "rapidfuzz_90_f1": rf90_f1,
                    "rapidfuzz_95_precision": rf95_p,
                    "rapidfuzz_95_recall": rf95_r,
                    "rapidfuzz_95_f1": rf95_f1,
                }
            else:
                abstained = len(extracted_spans) == 0
                unanswerable_correct.append(1 if abstained else 0)
                detail["metrics"] = {
                    "rouge_l_precision": None,
                    "rouge_l_recall": None,
                    "rouge_l_f1": None,
                    "bertscore_precision": None,
                    "bertscore_recall": None,
                    "bertscore_f1": None,
                    "rapidfuzz_75_precision": None,
                    "rapidfuzz_75_recall": None,
                    "rapidfuzz_75_f1": None,
                    "rapidfuzz_90_precision": None,
                    "rapidfuzz_90_recall": None,
                    "rapidfuzz_90_f1": None,
                    "rapidfuzz_95_precision": None,
                    "rapidfuzz_95_recall": None,
                    "rapidfuzz_95_f1": None,
                }
                detail["correctly_abstained"] = abstained
        except Exception as e:
            print("  error during extraction eval, counting as 0")
            detail["extracted_spans"] = []
            detail["metrics"] = {
                "rouge_l_precision": 0.0,
                "rouge_l_recall": 0.0,
                "rouge_l_f1": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "rapidfuzz_75_precision": 0.0,
                "rapidfuzz_75_recall": 0.0,
                "rapidfuzz_75_f1": 0.0,
                "rapidfuzz_90_precision": 0.0,
                "rapidfuzz_90_recall": 0.0,
                "rapidfuzz_90_f1": 0.0,
                "rapidfuzz_95_precision": 0.0,
                "rapidfuzz_95_recall": 0.0,
                "rapidfuzz_95_f1": 0.0,
            }
            detail["status"] = "ERROR"
            detail["error"] = str(e)
            if not is_unanswerable:
                append_zero_span_metric_scores(span_metric_lists)
            if is_unanswerable:
                unanswerable_correct.append(0)
            time.sleep(2)
        per_query_details.append(detail)
        time.sleep(1)

    aggregate = {
        "hit_rate": mean(all_hit) if all_hit else 0.0,
        "recall@k": mean(all_recall_k) if all_recall_k else 0.0,
        "mrr": mean(all_rr) if all_rr else 0.0,
        **summarize_span_metric_lists(span_metric_lists, unanswerable_correct),
    }
    return aggregate, per_query_details


def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL
    span_data_path = DATA_DIR / "span.json"
    retrieval_data_path = DATA_DIR / "qa_with_chunk_ids.json"
    if not span_data_path.exists() or not retrieval_data_path.exists():
        print(f"ERROR: Span or retrieval data not found at {span_data_path} / {retrieval_data_path}")
        sys.exit(1)
    span_data = json.loads(span_data_path.read_text())
    retrieval_data = json.loads(retrieval_data_path.read_text())

    retrieval_gold_map: Dict[str, List[int]] = {}
    for item in retrieval_data:
        query = item["query"]
        expected_idxs = item.get("expected_chunk_index", [])
        if not isinstance(expected_idxs, list):
            expected_idxs = [expected_idxs]
        if SKIP_SENTINEL_3000 and all(idx == 3000 for idx in expected_idxs):
            continue
        retrieval_gold_map[query] = expected_idxs

    # Shared retrieval infrastructure (used by all models/configs).
    rag_index, _ = connect_to_index(db_path=DB_PATH, verbose=False)
    reranker = BGEReranker()
    bert_scorer = get_bertscore_scorer(lang="en")
    results_table = []

    # Retrieval is independent from Zilliz thresholds/output mode.
    # Keep one retrieval cache per retrieval model and reuse across configs.
    retrieval_cache_by_model: Dict[str, Dict[str, Tuple[list, str, List[Tuple]]]] = {}

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
                    span_data,
                    retrieval_gold_map,
                    rag_index,
                    reranker,
                    query_rewriter,
                    query_generator,
                    zilliz_extractor,
                    bert_scorer,
                    retrieval_cache,
                )
            except Exception:
                span_metrics = zero_span_metrics()
                chunk_metrics = zero_chunk_metrics()
                query_details = []
            else:
                chunk_metrics = {
                    "hit_rate": span_metrics.get("hit_rate", 0.0),
                    "recall@k": span_metrics.get("recall@k", 0.0),
                    "mrr": span_metrics.get("mrr", 0.0),
                }
            results_table.append(
                make_benchmark_result_row(
                    labels={
                        "Extractor": zilliz_config["name"],
                        "Retrieval": retrieval_model.split("/")[-1],
                    },
                    chunk_metrics=chunk_metrics,
                    span_metrics=span_metrics,
                    extras={"details": query_details},
                )
            )

    header, row_lines = build_benchmark_header_and_rows(
        rows=results_table,
        leading_columns=[("Extractor", 20), ("Retrieval", 22)],
    )
    report_lines = build_table_report_lines(
        title="ZILLIZ EXTRACTOR BENCHMARK RESULTS",
        width=300,
        header=header,
        row_lines=row_lines,
    )
    output_path = RESULTS_DIR / "SemanticHighlighter_results.txt"
    print_and_write_report(report_lines, output_path)


if __name__ == "__main__":
    main()
