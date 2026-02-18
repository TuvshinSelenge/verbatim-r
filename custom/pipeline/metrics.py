import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple, List

from bert_score import BERTScorer
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TokenPairMatchCandidate:
    """
    Candidate alignment between one predicted span and one gold span.
    """
    pair_f1: float
    tp: int
    pred_idx: int
    gold_idx: int
    fp: int
    fn: int


def _strip_markdown_tables(text: str) -> str:
    # Guard: empty input should stay empty.
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
    # 1) Normalize formatting artifacts.
    text = normalize_extraction_text(text).lower()

    # 2) Keep only alphanumeric chars and spaces.
    text = "".join(c if c.isalnum() or c.isspace() else "" for c in text)

    # 3) Collapse repeated whitespace.
    return " ".join(text.split())


def normalize_ws(text: str) -> str:
    # Minimal whitespace normalization helper used by ROUGE/BERTScore flows.
    return " ".join((text or "").split())


def flatten_extracted_spans(spans_raw: dict) -> List[str]:
    """
    Flatten extractor output dict -> list of non-empty strings.
    """
    extracted: List[str] = []

    # Guard: extractor output is expected to be dict-like.
    if not isinstance(spans_raw, dict):
        return extracted

    # Expected shape:
    # {
    #   "<group_name>": ["span a", "span b", ...],
    #   ...
    # }
    for _, span_list in spans_raw.items():
        if not isinstance(span_list, list):
            continue
        for span in span_list:
            # Remove extra spaces and ignore empty spans.
            span = normalize_ws(span)
            if span:
                extracted.append(span)
    return extracted


def _token_overlap_counts(pred_text: str, gold_text: str) -> Tuple[int, int, int, float]:
    # Split each span into word tokens.
    pred_tokens = pred_text.split()
    gold_tokens = gold_text.split()

    # Use counters so repeated tokens are counted correctly (multiset overlap).
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    # Counter intersection gives per-token minimum counts.
    overlap = pred_counter & gold_counter

    # Token-level confusion counts for this span pair.
    tp = sum(overlap.values())
    fp = len(pred_tokens) - tp
    fn = len(gold_tokens) - tp

    # Pair-level F1 built directly from TP/FP/FN.
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    return tp, fp, fn, f1


def _normalize_and_filter_spans(spans: List[str], normalizer: Callable[[str], str]) -> List[str]:
    """
    Apply a normalizer to each span and drop empty results.
    """
    cleaned: List[str] = []
    for span in spans:
        normalized = normalizer(span)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _empty_case_prf(preds: List[str], golds: List[str]) -> Tuple[bool, float, float, float]:
    """
    Handle common empty-input edge cases for set metrics.

    Returns:
    - handled: whether an edge case was applied
    - precision, recall, f1: valid only when handled=True
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
    Compute F1 from precision and recall.
    """
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def _prepare_span_sets(
    predicted_spans: List[str],
    gold_spans: List[str],
    normalizer: Callable[[str], str],
) -> Tuple[List[str], List[str], bool, float, float, float]:
    """
    Normalize spans on both sides and apply shared empty-input handling.
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
    Build a dense pairwise score matrix where rows=predictions and cols=golds.
    """
    return [[pair_score_fn(pred, gold) for gold in golds] for pred in preds]


def _normalize_for_bert(text: str) -> str:
    """
    BERTScore normalization: extraction cleanup plus whitespace normalization.
    """
    return normalize_ws(normalize_extraction_text(text))


def _build_bertscore_f1_matrix(preds: List[str], golds: List[str], scorer: Any) -> List[List[float]]:
    """
    Compute pairwise BERTScore F1 matrix for all (pred, gold) combinations.
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


def compute_token_precision_recall_f1(
    predicted_spans: List[str],
    gold_spans: List[str],
) -> Dict[str, float]:
    """
    Token-level P/R/F1 with one-to-one greedy span alignment.

    - spans are normalized before scoring
    - per-pair scores are based on token overlap
    - matching is one-to-one over span pairs
    """
    # Step 1: normalize both span lists and apply shared empty-input handling.
    preds, golds, handled, precision, recall, f1 = _prepare_span_sets(
        predicted_spans, gold_spans, normalize_answer
    )
    if handled:
        return {"precision": precision, "recall": recall, "f1": f1}

    # Step 3: build all positive-overlap candidate pairs.
    pair_scores: List[TokenPairMatchCandidate] = []
    for i, pred in enumerate(preds):
        for j, gold in enumerate(golds):
            tp, fp, fn, pair_f1 = _token_overlap_counts(pred, gold)
            if pair_f1 > 0.0:
                pair_scores.append(
                    TokenPairMatchCandidate(
                        pair_f1=pair_f1,
                        tp=tp,
                        pred_idx=i,
                        gold_idx=j,
                        fp=fp,
                        fn=fn,
                    )
                )

    # Step 4: greedy order = highest F1 first; tie-break by larger TP.
    pair_scores.sort(key=lambda x: (x.pair_f1, x.tp), reverse=True)

    # Step 5: greedily select one-to-one matches.
    used_preds: Set[int] = set()
    used_golds: Set[int] = set()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # If both sides are unmatched, accept the pair and accumulate counts.
    for pair in pair_scores:
        if pair.pred_idx in used_preds or pair.gold_idx in used_golds:
            continue
        used_preds.add(pair.pred_idx)
        used_golds.add(pair.gold_idx)
        total_tp += pair.tp
        total_fp += pair.fp
        total_fn += pair.fn

    # Step 6: any unmatched prediction contributes only FP tokens.
    for i, pred in enumerate(preds):
        if i not in used_preds:
            total_fp += len(pred.split())

    # Step 7: any unmatched gold contributes only FN tokens.
    for j, gold in enumerate(golds):
        if j not in used_golds:
            total_fn += len(gold.split())

    # Step 8: compute global precision/recall/F1 from aggregate counts.
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = _f1_from_precision_recall(precision, recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def get_rouge_l_scorer() -> Any:
    """
    Create a rouge-score ROUGE-L scorer instance
    """
    # use_stemmer=False keeps exact-token behavior (closer to your prior logic).
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def _one_to_one_prf_from_pair_matrix(score_matrix: List[List[float]]) -> Tuple[float, float, float]:
    """
    Strict one-to-one alignment over pair scores
    used to maximize total pair score under one-to-one constraints
    """
    # Matrix shape:
    # rows = predictions, columns = gold spans.
    pred_n = len(score_matrix)
    gold_n = len(score_matrix[0]) if pred_n > 0 else 0

    # Edge cases first.
    if pred_n == 0 and gold_n == 0:
        return 1.0, 1.0, 1.0
    if pred_n == 0 or gold_n == 0:
        return 0.0, 0.0, 0.0

    # convert similarity scores to costs.
    # We clip to non-negative to preserve previous behavior where
    # negative pair scores do not reduce matched_score_sum.
    clipped_scores: List[List[float]] = []
    for row in score_matrix:
        clipped_scores.append([max(0.0, score) for score in row])
    cost_matrix = [[-score for score in row] for row in clipped_scores]

    # Solve global one-to-one assignment for min(pred_n, gold_n) pairs.
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    # Sum assigned pair scores.
    matched_score_sum = 0.0
    for i, j in zip(row_idx, col_idx):
        matched_score_sum += clipped_scores[i][j]

    # Convert matched-score sum into set-level precision and recall.
    # - precision averages score over prediction count
    # - recall averages score over gold count
    precision = matched_score_sum / pred_n
    recall = matched_score_sum / gold_n
    f1 = _f1_from_precision_recall(precision, recall)
    return precision, recall, f1


def rouge_l_pair_score(candidate: str, reference: str, scorer: Any) -> float:
    """
    ROUGE-L F1 for one candidate-reference pair.
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
    Set-based best-match aggregation for ROUGE-L.
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
    Create a BERTScore scorer instance.
    """
    return BERTScorer(lang=lang)


def compute_bertscore_best_match_prf(
    predicted_spans: List[str],
    gold_spans: List[str],
    scorer: Any,
) -> Tuple[float, float, float]:
    """
    Set-based best-match aggregation over pairwise BERTScore P/R/F1.
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
