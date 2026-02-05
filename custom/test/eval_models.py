"""
Model Benchmark Framework
=========================
Tests different LLM models on:
1. Retrieval metrics (Hit Rate, MRR)
2. Extraction metrics (Exact Match, Precision, Recall, F1)

This is a self-contained benchmark script that works with the custom/ folder structure.
"""

import json
import os
import sys
import re
from pathlib import Path
from statistics import mean
from typing import List, Dict

import openai
from openai import OpenAI

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
from bge_ranker import BGEReranker
from query_rewriter import QueryRewriter
from query_generator import QueryGenerator

# Import from verbatim packages
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.llm_client import LLMClient

# Database path - adjust this to your actual database location
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "milvus_verbatim.db"))

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") 
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS_TO_TEST = [
    "google/gemini-3-flash-preview"
]

# Configuration
TOP_K = 5
PER_SUBQ_K = 20
SKIP_SENTINEL_1300 = True


# =============================================================================
# JSON PARSING UTILITIES
# =============================================================================

def safe_parse_json(response: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks and extra text."""
    if not response or not response.strip():
        raise ValueError("Empty response")
    
    content = response.strip()

    # Method 1: Extraction from markdown code blocks 
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Method 2: Find the first { and last }
    brace_match = re.search(r'(\{[\s\S]*\})', content)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Method 3: Direct parse
    return json.loads(content)


# =============================================================================
# TEXT NORMALIZATION FOR METRICS
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


def tokenize(text: str) -> set:
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
# HELPER FUNCTIONS
# =============================================================================

def get_text_and_meta(chunk):
    """Extract text and metadata from chunk."""
    if isinstance(chunk, dict):
        return chunk.get("text", ""), chunk.get("metadata", {}) or {}
    return getattr(chunk, "text", ""), getattr(chunk, "metadata", {}) or {}


def patch_llm_client_for_robust_json(llm_client: LLMClient):
    """Patch the LLMClient to handle various JSON response formats."""
    original_complete = llm_client.complete
    
    def robust_extract_spans(question: str, documents: dict) -> dict:
        prompt = llm_client._build_extraction_prompt(question, documents)
        try:
            response = original_complete(prompt, json_mode=True)
            return safe_parse_json(response)
        except (json.JSONDecodeError, ValueError) as e:
            try:
                response = original_complete(prompt, json_mode=False)
                return safe_parse_json(response)
            except (json.JSONDecodeError, ValueError) as e2:
                print(f"Span extraction failed: {e2}")
                if response:
                    print(f"Raw response preview: {response[:300]}...")
                return {doc_id: [] for doc_id in documents.keys()}
    
    llm_client.extract_spans = robust_extract_spans
    llm_client.extract_relevant_spans_batch = robust_extract_spans
    return llm_client


# =============================================================================
# RETRIEVAL METRICS (Hit Rate, MRR)
# =============================================================================

def calculate_retrieval_metrics(
    gold_data: List[Dict],
    rag_index,
    reranker: BGEReranker,
    query_rewriter: QueryRewriter,
    query_generator: QueryGenerator,
) -> Dict[str, float]:
    """
    Calculate Hit Rate and MRR for retrieval.
    """
    per_query = []
    
    for i, item in enumerate(gold_data, 1):
        query = item["query"]
        expected_idxs = item["expected_chunk_index"]
        
        if not isinstance(expected_idxs, list):
            expected_idxs = [expected_idxs]
        
        # Skip sentinel questions
        if SKIP_SENTINEL_1300 and all(idx == 1300 for idx in expected_idxs):
            continue
        
        gold_idxs = set(expected_idxs)
        
        print(f"  [{i}/{len(gold_data)}] {query[:50]}...")
        
        try:
            # Rewrite query then generate multiple search queries
            rewritten = query_rewriter.rewrite(query)
            subqs = query_generator.generate_queries(rewritten)
            
            # Merge results
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
            
            # Rerank
            _, reranked = reranker.rerank(query, merged, top_k=TOP_K)
            
            # Extract predictions
            preds = []
            for h in reranked:
                _, m = get_text_and_meta(h)
                preds.append((m.get("source_file"), m.get("chunk_index")))
            
            # Calculate metrics
            hit = any(idx in gold_idxs for _, idx in preds)
            
            rank = None
            for r, (_, idx) in enumerate(preds, 1):
                if idx in gold_idxs:
                    rank = r
                    break
            rr = 1.0 / rank if rank else 0.0
            
            per_query.append({"hit@k": 1 if hit else 0, "rr": rr})
            
        except Exception as e:
            print(f"    ERROR: {e}")
            per_query.append({"hit@k": 0, "rr": 0.0})
    
    if per_query:
        hit_rate = mean([r["hit@k"] for r in per_query])
        mrr = mean([r["rr"] for r in per_query])
    else:
        hit_rate = 0.0
        mrr = 0.0
    
    return {"hit_rate": hit_rate, "mrr": mrr}


# =============================================================================
# EXTRACTION METRICS (SQuAD-style)
# =============================================================================

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Compute token-level F1, precision, and recall."""
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)
    
    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    common = pred_tokens & gold_tokens
    num_common = len(common)
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_extraction_metrics(
    span_data: List[Dict],
    rag: VerbatimRAG,
) -> Dict[str, float]:
    """
    Calculate extraction metrics (EM, Precision, Recall, F1).
    """
    all_em = []
    all_precision = []
    all_recall = []
    all_f1 = []
    
    for i, item in enumerate(span_data, 1):
        query = item["query"]
        expected_span = item.get("expected_span", "")
        
        print(f"  [{i}/{len(span_data)}] {query[:50]}...")
        
        try:
            # Use RAG to get answer
            response = rag.query(query, k=TOP_K)
            
            # Extract the answer text
            if hasattr(response, 'answer'):
                predicted = response.answer
            elif isinstance(response, dict):
                predicted = response.get('answer', '')
            else:
                predicted = str(response)
            
            # Calculate metrics
            em = compute_exact_match(predicted, expected_span)
            f1_metrics = compute_f1(predicted, expected_span)
            
            all_em.append(em)
            all_precision.append(f1_metrics["precision"])
            all_recall.append(f1_metrics["recall"])
            all_f1.append(f1_metrics["f1"])
            
        except Exception as e:
            print(f"    ERROR: {e}")
            all_em.append(0.0)
            all_precision.append(0.0)
            all_recall.append(0.0)
            all_f1.append(0.0)
    
    return {
        "exact_match": mean(all_em) if all_em else 0.0,
        "precision": mean(all_precision) if all_precision else 0.0,
        "recall": mean(all_recall) if all_recall else 0.0,
        "f1": mean(all_f1) if all_f1 else 0.0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=== INITIALIZING BENCHMARK FRAMEWORK ===")
    
    if OPENROUTER_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
    
    print("Loading datasets...")
    chunk_data_path = PROJECT_ROOT / "test_data" / "qa_with_chunk_ids.json"
    span_data_path = PROJECT_ROOT / "test_data" / "span.json"
    
    if not chunk_data_path.exists():
        print(f"ERROR: Chunk data not found at {chunk_data_path}")
        sys.exit(1)
    if not span_data_path.exists():
        print(f"ERROR: Span data not found at {span_data_path}")
        sys.exit(1)
    
    chunk_data = json.loads(chunk_data_path.read_text())
    span_data = json.loads(span_data_path.read_text())
    
    print(f"Loaded {len(chunk_data)} retrieval queries")
    print(f"Loaded {len(span_data)} extraction queries")
    
    print(f"Connecting to Milvus (DB: {DB_PATH})...")
    rag_index, _ = connect_to_index(db_path=DB_PATH, verbose=False)
    
    print("Initializing Reranker...")
    reranker = BGEReranker()

    results_table = []

    # ITERATE MODELS
    for model in MODELS_TO_TEST:
        print(f"\n\n{'#'*80}")
        print(f"BENCHMARKING MODEL: {model}")
        print(f"{'#'*80}")

        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY
        )
        
        # Initialize query rewriter and generator
        query_rewriter = QueryRewriter(openai_client=client, model=model)
        query_generator = QueryGenerator(client=client, model=model)

        # --- TEST 1: RETRIEVAL (Hit Rate / MRR) ---
        print(f"\n>>> Running Retrieval Test ({model})...")
        try:
            chunk_metrics = calculate_retrieval_metrics(
                chunk_data, rag_index, reranker, query_rewriter, query_generator
            )
        except Exception as e:
            print(f"Retrieval Test Failed: {e}")
            import traceback
            traceback.print_exc()
            chunk_metrics = {"hit_rate": 0.0, "mrr": 0.0}

        # --- TEST 2: EXTRACTION (SQuAD Metrics) ---
        print(f"\n>>> Running Extraction Test ({model})...")
        try:
            llm_client = LLMClient(
                model=model,
                api_base=OPENROUTER_BASE_URL
            )
            
            # Override both sync and async clients
            llm_client.client = client
            llm_client.async_client = openai.AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY
            )
            
            # Make JSON parsing robust for all models
            patch_llm_client_for_robust_json(llm_client)
            
            rag = VerbatimRAG(rag_index, llm_client=llm_client)
            rag.template_manager.use_contextual_mode(use_per_fact=True)
            
            span_metrics = calculate_extraction_metrics(span_data, rag)
        except Exception as e:
            print(f"Extraction Test Failed: {e}")
            import traceback
            traceback.print_exc()
            span_metrics = {
                "exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0
            }

        # Collect Results
        results_table.append({
            "Model": model,
            "Hit Rate": chunk_metrics.get("hit_rate", 0),
            "MRR": chunk_metrics.get("mrr", 0),
            "EM": span_metrics.get("exact_match", 0),
            "Prec": span_metrics.get("precision", 0),
            "Rec": span_metrics.get("recall", 0),
            "F1": span_metrics.get("f1", 0)
        })
        
    # PRINT AND SAVE RESULTS
    report_lines = []
    report_lines.append("\n\n" + "="*90)
    report_lines.append(f"{'FINAL BENCHMARK RESULTS':^90}")
    report_lines.append("="*90)
    
    header = f"{'Model':<35} | {'Hit Rate':<8} | {'MRR':<6} | {'EM':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}"
    report_lines.append(header)
    report_lines.append("-" * 90)
    
    for row in results_table:
        line = f"{row['Model']:<35} | {row['Hit Rate']:.3f}    | {row['MRR']:.3f}  | {row['EM']:.3f}  | {row['Prec']:.3f}  | {row['Rec']:.3f}  | {row['F1']:.3f}"
        report_lines.append(line)
    report_lines.append("="*90)

    print("\n".join(report_lines))

    output_path = SCRIPT_DIR / "benchmark_results.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
