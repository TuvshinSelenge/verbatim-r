"""
Benchmark Script using Zilliz SemanticHighlightExtractor for Span Extraction
===========================================================================
- Retrieval: Uses LLMs via OpenRouter for query rewriting + multi-query search
- Extraction: Uses Zilliz semantic-highlight model (NO LLM costs!)

Evaluation metrics: SQuAD-style EM, Precision, Recall, F1
"""

import json
import os
import sys
import re
import time
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from statistics import mean

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
from bge_ranker import BGEReranker
from query_rewriter import QueryRewriter
from query_generator import QueryGenerator

# Import Zilliz extractor
try:
    from verbatim_rag.extractors import SemanticHighlightExtractor
except ImportError:
    print("ERROR: SemanticHighlightExtractor not found!")
    print("Please update verbatim-rag: pip install --upgrade verbatim-rag")
    sys.exit(1)

# Database path
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))

# CONFIGURATION
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to test for RETRIEVAL only (query rewriting, search query generation)
RETRIEVAL_MODELS = [
    "google/gemini-3-flash-preview"
]

# Retrieval config 
TOP_K = 5
PER_SUBQ_K = 20
QUERY_TIMEOUT = 180  # 3 minutes per query

# Zilliz extractor configurations to test
# "sentences" mode — most gold spans are full sentences; low threshold for recall
ZILLIZ_CONFIGS = [
    {
        "name": "zilliz-sentences-0.3",
        "output_mode": "sentences",
        "threshold": 0.3,
        "min_span_tokens": 3,
        "merge_gap": 2,
    }
]


# =============================================================================
# CHUNK WRAPPER (SemanticHighlightExtractor needs .text attribute)
# =============================================================================

@dataclass
class SearchResultWrapper:
    """Wrapper to give chunks a .text attribute for the extractor."""
    text: str


def wrap_chunks(chunks: list) -> List[SearchResultWrapper]:
    """Wrap chunks (dict or object) into SearchResultWrapper for the extractor."""
    wrapped = []
    for c in chunks:
        if isinstance(c, dict):
            txt = c.get("text", "")
        else:
            txt = getattr(c, "text", "")
        if txt:
            wrapped.append(SearchResultWrapper(text=txt))
    return wrapped


# =============================================================================
# HELPERS
# =============================================================================

def get_text_and_meta(chunk) -> Tuple[str, dict]:
    """Extract text and metadata from chunk."""
    if isinstance(chunk, dict):
        return chunk.get("text", ""), chunk.get("metadata", {}) or {}
    return getattr(chunk, "text", ""), getattr(chunk, "metadata", {}) or {}


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
# RETRIEVAL: rewrite → generate → search → rerank (full pipeline)
# =============================================================================

def retrieve_and_rerank(
    query_text: str,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
    rag_index,
    reranker: BGEReranker,
) -> Tuple[List, str]:
    """
    Full retrieval pipeline. Returns:
      - reranked chunks (for extraction)
      - rewritten query (for extraction)
    """
    # Rewrite the query (1 LLM call)
    rewritten = query_rewriter.rewrite(query_text)

    # Generate multiple search queries (1 LLM call)
    subqs = query_generator.generate_queries(rewritten)

    # Merge results from all sub-queries (vector search only, fast)
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

    # Rerank (local model, fast)
    reranked, _ = reranker.rerank(rewritten, merged, top_k=TOP_K)

    return reranked, rewritten


# =============================================================================
# TEXT NORMALIZATION FOR METRICS (SQuAD-style) — self-contained, no external deps
# =============================================================================

def _strip_markdown_tables(text: str) -> str:
    """Strip markdown table formatting."""
    if not text:
        return ""
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        raw = line.strip()
        if re.match(r'^\s*\|?\s*[-:]+(?:\s*\|\s*[-:]+)+\s*\|?\s*$', raw):
            continue
        if "|" in raw:
            cells = [c.strip() for c in raw.strip("|").split("|")]
            row_text = " ".join(c for c in cells if c)
            if row_text:
                out_lines.append(row_text)
        else:
            out_lines.append(raw)
    return " ".join(out_lines)


def normalize_extraction_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.replace("**", " ").replace("__", " ").replace("_", " ").replace("`", " ")
    text = _strip_markdown_tables(text)
    text = " ".join(text.split())
    return text


def tokenize(text: str) -> Set[str]:
    """Tokenize text for F1 calculation."""
    text = normalize_extraction_text(text)
    text = text.lower()
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return set(text.split())


def normalize_answer(text: str) -> str:
    """Normalize answer for exact match."""
    text = normalize_extraction_text(text)
    text = text.lower()
    text = "".join(c if c.isalnum() or c.isspace() else "" for c in text)
    return " ".join(text.split())


# =============================================================================
# EXTRACTION METRICS (SQuAD-style)
# =============================================================================

def compute_exact_match(pred: str, gold: str) -> bool:
    """True EM: strings equal after normalization."""
    return normalize_answer(pred) == normalize_answer(gold)


def token_metrics(pred_span: str, gold_span: str) -> Tuple[float, float, float]:
    """Calculate token-level Precision, Recall, F1."""
    pred_tokens = tokenize(pred_span)
    gold_tokens = tokenize(gold_span)

    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    overlap = pred_tokens & gold_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_span_extraction(
    extracted_spans: List[str],
    gold_spans: List[str]
) -> Dict[str, float]:
    """SQuAD-style span evaluation with unanswerable detection."""
    is_unanswerable = len(gold_spans) == 0

    # Unanswerable: correct if model also extracted nothing
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

    # Answerable but no extractions
    if not extracted_spans:
        return {
            "exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "is_unanswerable": False, "correctly_abstained": False,
        }

    # Check for exact match
    for pred in extracted_spans:
        for gold in gold_spans:
            if compute_exact_match(pred, gold):
                return {
                    "exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                    "is_unanswerable": False, "correctly_abstained": False,
                    "matched_pred": pred, "matched_gold": gold, "match_type": "EM",
                }

    # Find best F1 match
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
        "exact_match": 0.0,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
        "is_unanswerable": False,
        "correctly_abstained": False,
        "matched_pred": best_pred, "matched_gold": best_gold, "match_type": "F1",
    }


# =============================================================================
# ZILLIZ EXTRACTION EVALUATION
# =============================================================================

def run_zilliz_extraction_evaluation(
    gold_data: List[Dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
    zilliz_extractor: SemanticHighlightExtractor,
) -> Tuple[Dict, List[Dict]]:
    """
    Run span extraction evaluation using Zilliz extractor.
    - Retrieval: LLM (rewrite → multi-query → search → rerank)
    - Extraction: Zilliz SemanticHighlightExtractor
    """
    all_em = []
    all_precision = []
    all_recall = []
    all_f1 = []
    unanswerable_correct = []
    per_query_details = []

    print(f"\n{'='*60}")
    print("SPAN EXTRACTION EVALUATION (Zilliz)")
    print(f"{'='*60}")
    print(f"Config: TOP_K={TOP_K}, PER_SUBQ_K={PER_SUBQ_K}")
    print(f"{'='*60}\n")

    total_queries = len(gold_data)

    for i, item in enumerate(gold_data, 1):
        query = item["query"]
        gold_spans = item.get("top_spans", [])
        if isinstance(gold_spans, str):
            gold_spans = [gold_spans] if gold_spans else []
        is_unanswerable = len(gold_spans) == 0

        print(f"[{i}/{total_queries}] {query[:80]}...")

        detail = {
            "idx": i,
            "query": query,
            "gold_spans": gold_spans,
            "is_unanswerable": is_unanswerable,
        }

        try:
            # Step 1: Full pipeline retrieval (LLM calls)
            reranked, rewritten = run_with_timeout(
                lambda q=query: retrieve_and_rerank(
                    q, query_rewriter, query_generator, rag_index, reranker
                ),
                timeout_sec=QUERY_TIMEOUT
            )

            # Step 2: Wrap chunks for the extractor
            wrapped_chunks = wrap_chunks(reranked)
            if not wrapped_chunks:
                print(f"No chunks retrieved, skipping extraction")
                extracted_spans = []
            else:
                # Step 3: Extract spans using Zilliz
                def do_extract(c=wrapped_chunks, r=rewritten):
                    extraction_result = zilliz_extractor.extract_spans(r, c)
                    spans = []
                    for chunk_text, span_list in extraction_result.items():
                        for s in span_list:
                            if s and s.strip():
                                spans.append(s.strip())
                    return spans

                extracted_spans = run_with_timeout(do_extract, timeout_sec=QUERY_TIMEOUT)

            # Step 4: Evaluate
            metrics = evaluate_span_extraction(extracted_spans, gold_spans)

            all_em.append(metrics["exact_match"])
            all_precision.append(metrics["precision"])
            all_recall.append(metrics["recall"])
            all_f1.append(metrics["f1"])

            detail["extracted_spans"] = extracted_spans
            detail["metrics"] = metrics

            if is_unanswerable:
                unanswerable_correct.append(1 if metrics["correctly_abstained"] else 0)
                status = "OK" if metrics["correctly_abstained"] else "FAIL"
                detail["status"] = status
                print(f"Extracted: {len(extracted_spans)} spans | [{status}] Unanswerable")
                if not metrics["correctly_abstained"]:
                    print(f"(Model extracted {len(extracted_spans)} spans but should have abstained)")
            else:
                status = "EM" if metrics["exact_match"] == 1.0 else (
                    "OK" if metrics["f1"] >= 0.5 else "LOW"
                )
                detail["status"] = status
                print(f"Extracted: {len(extracted_spans)} spans | [{status}] P:{metrics['precision']:.2f} R:{metrics['recall']:.2f} F1:{metrics['f1']:.2f}")

                # Print match details
                matched_pred = metrics.get("matched_pred", "")
                matched_gold = metrics.get("matched_gold", "")
                match_type = metrics.get("match_type", "")

                if match_type == "EM" and matched_pred:
                    print(f"EXACT MATCH:")
                    print(f"Predicted: {matched_pred[:120]}{'...' if len(matched_pred) > 120 else ''}")
                    print(f"Gold:      {matched_gold[:120]}{'...' if len(matched_gold) > 120 else ''}")
                elif match_type == "F1" and matched_pred and metrics["f1"] > 0:
                    print(f"BEST MATCH (F1={metrics['f1']:.2f}):")
                    print(f"Predicted: {matched_pred[:120]}{'...' if len(matched_pred) > 120 else ''}")
                    print(f"Gold:      {matched_gold[:120]}{'...' if len(matched_gold) > 120 else ''}")
                elif metrics["f1"] == 0:
                    print(f"NO MATCH extracted spans did not overlap with any gold span")

        except Exception as e:
            print(f"ERROR: {e}")
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
        time.sleep(1)  # Rate limit buffer

    aggregate = {
        "exact_match": mean(all_em) if all_em else 0.0,
        "precision": mean(all_precision) if all_precision else 0.0,
        "recall": mean(all_recall) if all_recall else 0.0,
        "f1": mean(all_f1) if all_f1 else 0.0,
        "unanswerable_accuracy": mean(unanswerable_correct) if unanswerable_correct else 1.0,
    }

    return aggregate, per_query_details


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== ZILLIZ EXTRACTOR BENCHMARK (SPAN EXTRACTION) ===")
    print("Extraction: Zilliz SemanticHighlightExtractor (NO LLM)")
    print("Retrieval:  LLMs via OpenRouter (rewrite + multi-query)")

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set.")
        sys.exit(1)

    # Set env vars for modules that use them internally
    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE_URL

    # Load dataset
    print("\nLoading span evaluation data...")
    span_data_path = PROJECT_ROOT / "test_data" / "span.json"
    if not span_data_path.exists():
        print(f"ERROR: Span data not found at {span_data_path}")
        sys.exit(1)

    span_data = json.loads(span_data_path.read_text())
    print(f"Loaded {len(span_data)} span queries")

    # Connect to index
    print(f"\nConnecting to Milvus (DB: {DB_PATH})...")
    rag_index, _ = connect_to_index(db_path=DB_PATH, verbose=False)

    # Initialize reranker
    print("Initializing Reranker...")
    reranker = BGEReranker()

    results_table = []

    for zilliz_config in ZILLIZ_CONFIGS:
        print(f"\n\n{'#'*80}")
        print(f"ZILLIZ CONFIG: {zilliz_config['name']}")
        print(f"  mode={zilliz_config['output_mode']}, threshold={zilliz_config['threshold']}")
        print(f"{'#'*80}")

        # Initialize Zilliz extractor
        print(f"Loading Zilliz extractor...")
        zilliz_extractor = SemanticHighlightExtractor(
            model_name="zilliz/semantic-highlight-bilingual-v1",
            threshold=zilliz_config["threshold"],
            output_mode=zilliz_config["output_mode"],
            min_span_tokens=zilliz_config.get("min_span_tokens", 3),
            merge_gap=zilliz_config.get("merge_gap", 2),
        )

        # Test with each retrieval model
        for retrieval_model in RETRIEVAL_MODELS:
            print(f"\n>>> Retrieval Model: {retrieval_model}")

            # Initialize OpenRouter client
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
                timeout=120.0
            )

            # Initialize query rewriter and generator
            query_rewriter = QueryRewriter(openai_client=client, model=retrieval_model)
            query_generator = QueryGenerator(client=client, model=retrieval_model)

            # Run extraction evaluation
            print(f"\n>>> Running Zilliz Extraction Test...")
            query_details = []
            try:
                span_metrics, query_details = run_zilliz_extraction_evaluation(
                    span_data, rag_index, reranker,
                    query_rewriter, query_generator,
                    zilliz_extractor,
                )
            except Exception as e:
                print(f"Extraction Test Failed: {e}")
                import traceback
                traceback.print_exc()
                span_metrics = {
                    "exact_match": 0.0, "precision": 0.0,
                    "recall": 0.0, "f1": 0.0,
                    "unanswerable_accuracy": 0.0,
                }

            # Collect results
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

    # =========================================================================
    # FINAL REPORT 
    # =========================================================================
    report_lines = []

    # --- Per-query detail section ---
    for result_row in results_table:
        details = result_row.get("details", [])
        if not details:
            continue

        report_lines.append("")
        report_lines.append("=" * 100)
        report_lines.append(f"PER-QUERY DETAILS: {result_row['Extractor']} | Retrieval: {result_row['Retrieval']}")
        report_lines.append("=" * 100)

        for d in details:
            report_lines.append("")
            report_lines.append(f"[{d['idx']}] {d['query']}")
            report_lines.append(f"    Status: [{d.get('status', '?')}]")

            metrics = d.get("metrics", {})
            if d.get("is_unanswerable"):
                abstained = metrics.get("correctly_abstained", False)
                extracted = d.get("extracted_spans", [])
                report_lines.append(f"Unanswerable: {'CORRECT (abstained)' if abstained else f'WRONG (extracted {len(extracted)} spans)'}")
                if extracted:
                    for j, s in enumerate(extracted, 1):
                        report_lines.append(f"Extracted [{j}]: {s[:150]}{'...' if len(s) > 150 else ''}")
            else:
                report_lines.append(f"P:{metrics.get('precision', 0):.3f}  R:{metrics.get('recall', 0):.3f}  F1:{metrics.get('f1', 0):.3f}  EM:{metrics.get('exact_match', 0):.0f}")

                # Gold spans
                gold_spans = d.get("gold_spans", [])
                report_lines.append(f"Gold spans ({len(gold_spans)}):")
                for j, g in enumerate(gold_spans, 1):
                    report_lines.append(f"{j}] {g[:150]}{'...' if len(g) > 150 else ''}")

                # Extracted spans
                extracted = d.get("extracted_spans", [])
                report_lines.append(f"Extracted spans ({len(extracted)}):")
                for j, s in enumerate(extracted, 1):
                    report_lines.append(f"{j}] {s[:150]}{'...' if len(s) > 150 else ''}")

                # Match info
                matched_pred = metrics.get("matched_pred", "")
                matched_gold = metrics.get("matched_gold", "")
                match_type = metrics.get("match_type", "")

                if match_type == "EM" and matched_pred:
                    report_lines.append(f"EXACT MATCH:")
                    report_lines.append(f"Predicted: {matched_pred[:150]}{'...' if len(matched_pred) > 150 else ''}")
                    report_lines.append(f"Gold:      {matched_gold[:150]}{'...' if len(matched_gold) > 150 else ''}")
                elif match_type == "F1" and matched_pred and metrics.get("f1", 0) > 0:
                    report_lines.append(f"BEST MATCH (F1={metrics['f1']:.3f}):")
                    report_lines.append(f"Predicted: {matched_pred[:150]}{'...' if len(matched_pred) > 150 else ''}")
                    report_lines.append(f"Gold:      {matched_gold[:150]}{'...' if len(matched_gold) > 150 else ''}")
                elif metrics.get("f1", 0) == 0 and extracted:
                    report_lines.append(f"NO MATCH extracted spans did not overlap with any gold span")

            if d.get("error"):
                report_lines.append(f"ERROR: {d['error']}")

    # --- Summary table ---
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append(f"{'ZILLIZ EXTRACTOR BENCHMARK RESULTS':^100}")
    report_lines.append("=" * 100)

    header = (
        f"{'Extractor':<25} | {'Retrieval':<25} | {'EM':<6} | {'Prec':<6} "
        f"| {'Rec':<6} | {'F1':<6} | {'Unans.Acc':<9}"
    )
    report_lines.append(header)
    report_lines.append("-" * 100)

    for row in results_table:
        line = (
            f"{row['Extractor']:<25} | {row['Retrieval']:<25} | {row['EM']:.3f}  "
            f"| {row['Prec']:.3f}  | {row['Rec']:.3f}  | {row['F1']:.3f}  "
            f"| {row['Unans.Acc']:.3f}"
        )
        report_lines.append(line)
    report_lines.append("=" * 100)

    # Print to console
    print("\n".join(report_lines))


    output_path = SCRIPT_DIR / "zilliz_benchmark_results.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
