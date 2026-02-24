import re
from collections.abc import Callable
from typing import Any, Dict, Tuple, List

from bert_score import BERTScorer
from rapidfuzz import fuzz
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment

# RapidFuzz thresholds used to show metric sensitivity from lenient to near-exact:
#   0.50 → very lenient
#   0.75 → lenient
#   0.90 → strict
#   0.95 → near-exact
RAPIDFUZZ_THRESHOLDS: Tuple[float, ...] = (0.50, 0.75, 0.90, 0.95)


def _strip_markdown_tables(text: str) -> str:
    """
    Inputs: raw text that may contain markdown table rows.
    Steps: remove separator rows, flatten table cells, normalize whitespace.
    Output: plain text string with table artifacts removed.
    """
    # empty input should stay empty.
    if not text:
        return ""

    # Process text line by line so we can remove markdown separators
    # and flatten table rows into normal text.
    lines = text.splitlines()
    out_lines = []

    for line in lines:
        # Remove outer spaces to simplify pattern checks.
        raw = line.strip()

        # Skip markdown table separator rows like:
        # | --- | --- |
        if re.match(r"^\s*\|?\s*[-:]+(?:\s*\|\s*[-:]+)+\s*\|?\s*$", raw):
            continue

        # If this line looks like a table row, split cells and keep their content.
        if "|" in raw:
            cells = [c.strip() for c in raw.strip("|").split("|")]
            row_text = " ".join(c for c in cells if c)
            if row_text:
                out_lines.append(row_text)
        else:
            # Normal non-table line: keep as-is (after strip above).
            out_lines.append(raw)

    # Return one clean whitespace-normalized string.
    return " ".join(out_lines)


def normalize_extraction_text(text: str) -> str:
    """
    Inputs: extracted text from model output.
    Steps: remove markdown styling, strip table formatting, collapse spaces.
    Output: normalized extraction text for downstream metrics.
    """
    # Guard: empty input should stay empty.
    if not text:
        return ""

    # Remove common markdown emphasis/code markers.
    text = text.replace("**", " ").replace("__", " ").replace("_", " ").replace("`", " ")

    # Flatten markdown tables into normal plain text.
    text = _strip_markdown_tables(text)

    # Collapse repeated whitespace.
    return " ".join(text.split())


def normalize_answer(text: str) -> str:
    """
    Inputs: answer/span text.
    Steps: normalize extraction text, lowercase, remove punctuation, collapse spaces.
    Output: alphanumeric normalized text for token-level overlap matching.
    """
    # 1) Normalize formatting artifacts.
    text = normalize_extraction_text(text).lower()

    # 2) Keep only alphanumeric chars and spaces.
    text = "".join(c if c.isalnum() or c.isspace() else "" for c in text)

    # 3) Collapse repeated whitespace.
    return " ".join(text.split())


def normalize_ws(text: str) -> str:
    """
    Inputs: any text (including None/empty).
    Steps: split and re-join on whitespace.
    Output: whitespace-normalized text.
    """
    # Minimal whitespace normalization helper used by ROUGE/BERTScore flows.
    return " ".join((text or "").split())


def flatten_extracted_spans(spans_raw: dict) -> List[str]:
    """
    Inputs: extractor output dictionary of span groups.
    Steps: iterate groups, normalize whitespace, remove empty spans.
    Output: flat list of cleaned span strings.
    """
    extracted: List[str] = []

    if not isinstance(spans_raw, dict):
        return extracted
    for _, span_list in spans_raw.items():
        if not isinstance(span_list, list):
            continue
        for span in span_list:
            # Remove extra spaces and ignore empty spans.
            span = normalize_ws(span)
            if span:
                extracted.append(span)
    return extracted


# ---------------------------------------------------------------------------
# Token-overlap span matching (commented out; superseded by RapidFuzz PRF)
# ---------------------------------------------------------------------------
# def _token_overlap_counts(pred_text: str, gold_text: str) -> Tuple[int, int, int, float]:
#     pred_tokens = pred_text.split()
#     gold_tokens = gold_text.split()
#     pred_counter = Counter(pred_tokens)
#     gold_counter = Counter(gold_tokens)
#     overlap = pred_counter & gold_counter
#     tp = sum(overlap.values())
#     fp = len(pred_tokens) - tp
#     fn = len(gold_tokens) - tp
#     denom = 2 * tp + fp + fn
#     f1 = (2 * tp / denom) if denom > 0 else 0.0
#     return tp, fp, fn, f1


def _normalize_and_filter_spans(spans: List[str], normalizer: Callable[[str], str]) -> List[str]:
    """
    Inputs: list of spans plus a normalization function.
    Steps: apply normalizer per span and drop empty results.
    Output: cleaned span list.
    """
    cleaned: List[str] = []
    for span in spans:
        normalized = normalizer(span)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _empty_case_prf(preds: List[str], golds: List[str]) -> Tuple[bool, float, float, float]:
    """
    Inputs: cleaned prediction and gold span lists.
    Steps: check empty-empty and one-empty edge cases.
    Output: (handled, precision, recall, f1) edge-case result.
    """
    # Nothing predicted and nothing expected: perfect score.
    if not preds and not golds:
        return True, 1.0, 1.0, 1.0

    # One side empty and the other non-empty: total miss.
    if not preds or not golds:
        return True, 0.0, 0.0, 0.0

    # Non-empty on both sides: continue normal scoring flow.
    return False, 0.0, 0.0, 0.0


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    """
    Inputs: precision and recall scalars.
    Steps: apply harmonic-mean formula with zero-denominator guard.
    Output: F1 score.
    """
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def _prepare_span_sets(
    predicted_spans: List[str],
    gold_spans: List[str],
    normalizer: Callable[[str], str],
) -> Tuple[List[str], List[str], bool, float, float, float]:
    """
    Inputs: raw predicted spans, raw gold spans, and a normalizer.
    Steps: normalize both span lists and apply shared empty-case handling.
    Output: (preds, golds, handled, precision, recall, f1).
    """
    preds = _normalize_and_filter_spans(predicted_spans, normalizer)
    golds = _normalize_and_filter_spans(gold_spans, normalizer)
    handled, precision, recall, f1 = _empty_case_prf(preds, golds)
    return preds, golds, handled, precision, recall, f1


def _build_pair_score_matrix(
    preds: List[str],
    golds: List[str],
    pair_score_fn: Callable[[str, str], float],
) -> List[List[float]]:
    """
    Inputs: cleaned predictions/golds and a pair-scoring function.
    Steps: score every (pred, gold) combination.
    Output: dense matrix with rows=preds and cols=golds.
    """
    return [[pair_score_fn(pred, gold) for gold in golds] for pred in preds]


def _normalize_for_bert(text: str) -> str:
    """
    Inputs: raw span text.
    Steps: normalize extraction artifacts then normalize whitespace.
    Output: text prepared for BERTScore comparison.
    """
    return normalize_ws(normalize_extraction_text(text))


def _build_bertscore_f1_matrix(preds: List[str], golds: List[str], scorer: Any) -> List[List[float]]:
    """
    Inputs: cleaned prediction/gold spans plus a BERTScore scorer.
    Steps: batch-score all pairs, reconstruct P/R matrices, convert to F1 matrix.
    Output: dense pairwise BERTScore F1 matrix.
    """
    # Build flat pair lists for one batched scorer call.
    pair_cands: List[str] = []
    pair_refs: List[str] = []
    pair_index: List[Tuple[int, int]] = []
    for i, pred in enumerate(preds):
        for j, gold in enumerate(golds):
            pair_cands.append(pred)
            pair_refs.append(gold)
            pair_index.append((i, j))

    # Run BERTScore once for all pairs and map output back to matrix form.
    p_tensor, r_tensor, _ = scorer.score(pair_cands, pair_refs)
    pair_p = [float(v) for v in p_tensor.tolist()]
    pair_r = [float(v) for v in r_tensor.tolist()]

    p_matrix = [[0.0 for _ in golds] for _ in preds]
    r_matrix = [[0.0 for _ in golds] for _ in preds]
    for (i, j), p_val, r_val in zip(pair_index, pair_p, pair_r):
        p_matrix[i][j] = p_val
        r_matrix[i][j] = r_val

    # Convert pairwise P/R into pairwise F1 for symmetric matching.
    return [
        [_f1_from_precision_recall(p_matrix[i][j], r_matrix[i][j]) for j in range(len(golds))]
        for i in range(len(preds))
    ]


def _rapidfuzz_pair_similarity(pred: str, gold: str) -> float:
    """
    Inputs: one normalized predicted span and one normalized gold span.
    Steps: compute RapidFuzz weighted-ratio similarity and scale to [0, 1].
    Output: pairwise RapidFuzz similarity score.
    """
    return float(fuzz.WRatio(pred, gold) / 100.0)


def _compute_rapidfuzz_prf_at_threshold(
    score_matrix: List[List[float]],
    threshold: float,
) -> Tuple[float, float, float]:
    """
    Inputs: pairwise RapidFuzz similarity matrix and a match threshold.
    Steps: gate the matrix at the threshold (keep original score if score >= threshold,
        else 0.0), then delegate to _one_to_one_prf_from_pair_matrix.
        This keeps strict thresholding while preserving above-threshold match quality
        during Hungarian one-to-one assignment and PRF aggregation.
    Output: (precision, recall, f1) at the provided threshold.
    """
    thresholded_weight_matrix = [
        [score if score >= threshold else 0.0 for score in row]
        for row in score_matrix
    ]
    return _one_to_one_prf_from_pair_matrix(thresholded_weight_matrix)


def compute_rapidfuzz_threshold_prf(
    predicted_spans: List[str],
    gold_spans: List[str],
    thresholds: Tuple[float, ...] = RAPIDFUZZ_THRESHOLDS,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Inputs: predicted span list, gold span list, and RapidFuzz thresholds.
    Steps: normalize spans, build pairwise RapidFuzz matrix once, compute PRF per threshold.
    Output: map of threshold key -> (precision, recall, f1).
    """
    preds, golds, handled, precision, recall, f1 = _prepare_span_sets(
        predicted_spans, gold_spans, normalize_answer
    )
    if handled:
        return {
            f"{int(round(threshold * 100))}": (precision, recall, f1)
            for threshold in thresholds
        }

    score_matrix = _build_pair_score_matrix(preds, golds, _rapidfuzz_pair_similarity)
    return {
        f"{int(round(threshold * 100))}": _compute_rapidfuzz_prf_at_threshold(
            score_matrix=score_matrix,
            threshold=threshold,
        )
        for threshold in thresholds
    }


# ---------------------------------------------------------------------------
# Token-level P/R/F1 (commented out; RapidFuzz@threshold is the primary metric)
# ---------------------------------------------------------------------------
# def compute_token_precision_recall_f1(
#     predicted_spans: List[str],
#     gold_spans: List[str],
# ) -> Dict[str, float]:
#     """Token-overlap span matching via SQuAD-style micro-aggregated TP/FP/FN."""
#     preds, golds, handled, precision, recall, f1 = _prepare_span_sets(
#         predicted_spans, gold_spans, normalize_answer
#     )
#     if handled:
#         return {"precision": precision, "recall": recall, "f1": f1}
#     pred_n, gold_n = len(preds), len(golds)
#     pair_f1_matrix = [[0.0] * gold_n for _ in range(pred_n)]
#     tp_matrix = [[0] * gold_n for _ in range(pred_n)]
#     fp_matrix = [[0] * gold_n for _ in range(pred_n)]
#     fn_matrix = [[0] * gold_n for _ in range(pred_n)]
#     for i, pred in enumerate(preds):
#         for j, gold in enumerate(golds):
#             tp, fp, fn, pair_f1 = _token_overlap_counts(pred, gold)
#             pair_f1_matrix[i][j] = pair_f1
#             tp_matrix[i][j] = tp
#             fp_matrix[i][j] = fp
#             fn_matrix[i][j] = fn
#     cost_matrix = [[-max(0.0, s) for s in row] for row in pair_f1_matrix]
#     row_idx, col_idx = linear_sum_assignment(cost_matrix)
#     used_preds, used_golds = set(row_idx), set(col_idx)
#     total_tp = sum(tp_matrix[i][j] for i, j in zip(row_idx, col_idx))
#     total_fp = sum(fp_matrix[i][j] for i, j in zip(row_idx, col_idx))
#     total_fn = sum(fn_matrix[i][j] for i, j in zip(row_idx, col_idx))
#     for i, pred in enumerate(preds):
#         if i not in used_preds:
#             total_fp += len(pred.split())
#     for j, gold in enumerate(golds):
#         if j not in used_golds:
#             total_fn += len(gold.split())
#     precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
#     recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
#     return {"precision": precision, "recall": recall, "f1": _f1_from_precision_recall(precision, recall)}


def get_rouge_l_scorer() -> Any:
    """
    Inputs: none.
    Steps: instantiate rouge-score scorer configured for ROUGE-L.
    Output: reusable ROUGE-L scorer instance.
    """
    # use_stemmer=False keeps exact-token behavior (closer to your prior logic).
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def _one_to_one_prf_from_pair_matrix(score_matrix: List[List[float]]) -> Tuple[float, float, float]:
    """
    Shared PRF aggregator used by ROUGE-L, BERTScore, and RapidFuzz metrics.

    Inputs: pairwise similarity matrix (rows=preds, cols=golds). Scores may be
        continuous in [0, 1] (ROUGE-L, BERTScore), threshold-gated weighted
        RapidFuzz scores, or binary 0/1.
    Steps:
        1. Clip scores to [0, 1] so negative entries do not reduce the sum.
        2. Solve optimal one-to-one assignment (Hungarian algorithm) that
           maximises the total matched-score sum.
        3. Compute set-level P and R with the unified formula:
               P = Σ matched_scores / |preds|
               R = Σ matched_scores / |golds|
           Unmatched predictions contribute 0 to the numerator, penalising
           over-generation; unmatched golds penalise under-generation.
        4. F1 = harmonic mean of P and R.
    Output: (precision, recall, f1).

    This formula is equivalent to the standard binary TP/FP/FN formula when
    scores are 0/1 (TP = Σ scores, FP = |preds| − TP, FN = |golds| − TP),
    and generalises it to soft/graded match quality for continuous metrics.
    """
    pred_n = len(score_matrix)
    gold_n = len(score_matrix[0]) if pred_n > 0 else 0

    if pred_n == 0 and gold_n == 0:
        return 1.0, 1.0, 1.0
    if pred_n == 0 or gold_n == 0:
        return 0.0, 0.0, 0.0

    clipped_scores: List[List[float]] = [
        [max(0.0, score) for score in row] for row in score_matrix
    ]
    cost_matrix = [[-score for score in row] for row in clipped_scores]

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matched_score_sum = sum(clipped_scores[i][j] for i, j in zip(row_idx, col_idx))

    precision = matched_score_sum / pred_n
    recall = matched_score_sum / gold_n
    f1 = _f1_from_precision_recall(precision, recall)
    return precision, recall, f1


def rouge_l_pair_score(candidate: str, reference: str, scorer: Any) -> float:
    """
    Inputs: one candidate span, one reference span, and ROUGE scorer.
    Steps: normalize whitespace, handle edge cases, score with ROUGE-L.
    Output: pairwise ROUGE-L F1.
    """
    # Step 1: normalize only whitespace (ROUGE-L keeps token sequence).
    cand = normalize_ws(candidate)
    ref = normalize_ws(reference)

    # Step 2: handle edge cases explicitly.
    if not cand and not ref:
        return 1.0
    if not cand or not ref:
        return 0.0
        
    # compute ROUGE-L F1 for this pair.
    scores = scorer.score(ref, cand)

    return float(scores["rougeL"].fmeasure)


def compute_rouge_l_best_match_prf(
    predicted_spans: List[str],
    gold_spans: List[str],
) -> Tuple[float, float, float]:
    """
    Inputs: predicted span list and gold span list.
    Steps: normalize, handle empty cases, pairwise ROUGE-L, one-to-one assignment.
    Output: set-level precision, recall, and F1.
    """
    # Step 1: normalize both span lists and apply shared empty-input handling.
    preds, golds, handled, precision, recall, f1 = _prepare_span_sets(
        predicted_spans, gold_spans, normalize_ws
    )
    if handled:
        return precision, recall, f1

    # Step 3: create one reusable ROUGE-L scorer instance.
    rouge_l_scorer = get_rouge_l_scorer()

    # Step 4: build pairwise ROUGE-L F1 matrix.
    score_matrix = _build_pair_score_matrix(
        preds,
        golds,
        pair_score_fn=lambda pred, gold: rouge_l_pair_score(pred, gold, scorer=rouge_l_scorer),
    )

    # Step 5: apply strict one-to-one aggregation.
    return _one_to_one_prf_from_pair_matrix(score_matrix)


def get_bertscore_scorer(lang: str = "en") -> Any:
    """
    Inputs: language code.
    Steps: instantiate BERTScore scorer for the provided language.
    Output: reusable BERTScore scorer instance.
    """
    return BERTScorer(lang=lang)


def compute_bertscore_best_match_prf(
    predicted_spans: List[str],
    gold_spans: List[str],
    scorer: Any,
) -> Tuple[float, float, float]:
    """
    Inputs: predicted span list, gold span list, and BERTScore scorer.
    Steps: normalize, handle empty cases, build pairwise BERT F1 matrix, assign one-to-one.
    Output: set-level precision, recall, and F1.
    """
    # Step 1: normalize both span lists and apply shared empty-input handling.
    preds, golds, handled, precision, recall, f1 = _prepare_span_sets(
        predicted_spans, gold_spans, _normalize_for_bert
    )
    if handled:
        return precision, recall, f1

    # Step 3: build pairwise BERTScore F1 matrix.
    f_matrix = _build_bertscore_f1_matrix(preds, golds, scorer)

    # Step 4: apply strict one-to-one aggregation over the F1 matrix.
    precision, recall, f1 = _one_to_one_prf_from_pair_matrix(f_matrix)
    return precision, recall, f1
