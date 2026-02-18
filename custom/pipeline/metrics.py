import re
from collections import Counter
from typing import Any, Dict, Set, Tuple, List


def _strip_markdown_tables(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    out_lines = []
    for line in lines:
        raw = line.strip()
        if re.match(r"^\s*\|?\s*[-:]+(?:\s*\|\s*[-:]+)+\s*\|?\s*$", raw):
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
    if not text:
        return ""
    text = text.replace("**", " ").replace("__", " ").replace("_", " ").replace("`", " ")
    text = _strip_markdown_tables(text)
    return " ".join(text.split())


def normalize_answer(text: str) -> str:
    text = normalize_extraction_text(text).lower()
    text = "".join(c if c.isalnum() or c.isspace() else "" for c in text)
    return " ".join(text.split())


def normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def flatten_extracted_spans(spans_raw: dict) -> List[str]:
    """
    Flatten extractor output dict -> list of non-empty strings.
    """
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


def _token_overlap_counts(pred_text: str, gold_text: str) -> Tuple[int, int, int, float]:
    pred_tokens = pred_text.split()
    gold_tokens = gold_text.split()
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = pred_counter & gold_counter
    tp = sum(overlap.values())
    fp = len(pred_tokens) - tp
    fn = len(gold_tokens) - tp
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    return tp, fp, fn, f1


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
    preds = [normalize_answer(s) for s in predicted_spans if normalize_answer(s)]
    golds = [normalize_answer(s) for s in gold_spans if normalize_answer(s)]

    if not preds and not golds:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not preds or not golds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pair_scores: List[Tuple[float, int, int, int, int, int]] = []
    for i, pred in enumerate(preds):
        for j, gold in enumerate(golds):
            tp, fp, fn, pair_f1 = _token_overlap_counts(pred, gold)
            if pair_f1 > 0.0:
                pair_scores.append((pair_f1, tp, i, j, fp, fn))

    pair_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    used_preds: Set[int] = set()
    used_golds: Set[int] = set()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for _, tp, pred_idx, gold_idx, fp, fn in pair_scores:
        if pred_idx in used_preds or gold_idx in used_golds:
            continue
        used_preds.add(pred_idx)
        used_golds.add(gold_idx)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    for i, pred in enumerate(preds):
        if i not in used_preds:
            total_fp += len(pred.split())

    for j, gold in enumerate(golds):
        if j not in used_golds:
            total_fn += len(gold.split())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _lcs_length(tokens_a: List[str], tokens_b: List[str]) -> int:
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


def _one_to_one_prf_from_pair_matrix(score_matrix: List[List[float]]) -> Tuple[float, float, float]:
    """
    Strict one-to-one alignment over pair scores.

    Greedy max-score matching is used to enforce unique assignments
    between predicted and gold spans (stricter than many-to-one best-match).
    """
    pred_n = len(score_matrix)
    gold_n = len(score_matrix[0]) if pred_n > 0 else 0
    if pred_n == 0 and gold_n == 0:
        return 1.0, 1.0, 1.0
    if pred_n == 0 or gold_n == 0:
        return 0.0, 0.0, 0.0

    used_preds: Set[int] = set()
    used_golds: Set[int] = set()
    matched_score_sum = 0.0

    while len(used_preds) < pred_n and len(used_golds) < gold_n:
        best_i = -1
        best_j = -1
        best_score = -1.0
        for i in range(pred_n):
            if i in used_preds:
                continue
            for j in range(gold_n):
                if j in used_golds:
                    continue
                score = score_matrix[i][j]
                if score > best_score:
                    best_score = score
                    best_i = i
                    best_j = j
        if best_i == -1 or best_j == -1:
            break
        used_preds.add(best_i)
        used_golds.add(best_j)
        matched_score_sum += max(0.0, best_score)

    precision = matched_score_sum / pred_n
    recall = matched_score_sum / gold_n
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def rouge_l_pair_score(candidate: str, reference: str) -> float:
    """
    ROUGE-L F1 for one candidate-reference pair.
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


def compute_rouge_l_best_match_prf(
    predicted_spans: List[str],
    gold_spans: List[str],
) -> Tuple[float, float, float]:
    """
    Set-based best-match aggregation for ROUGE-L.
    """
    preds = [normalize_ws(s) for s in predicted_spans if normalize_ws(s)]
    golds = [normalize_ws(s) for s in gold_spans if normalize_ws(s)]

    if not preds and not golds:
        return 1.0, 1.0, 1.0
    if not preds or not golds:
        return 0.0, 0.0, 0.0

    score_matrix = [
        [rouge_l_pair_score(pred, gold) for gold in golds]
        for pred in preds
    ]
    return _one_to_one_prf_from_pair_matrix(score_matrix)


def get_bertscore_scorer(lang: str = "en") -> Any:
    """
    Create a BERTScore scorer instance (lazy import).
    """
    try:
        from bert_score import BERTScorer  # type: ignore
    except Exception as exc:
        raise RuntimeError("BERTScore requires `bert-score`. Install with: pip install bert-score") from exc
    return BERTScorer(lang=lang)


def compute_bertscore_best_match_prf(
    predicted_spans: List[str],
    gold_spans: List[str],
    scorer: Any,
) -> Tuple[float, float, float]:
    """
    Set-based best-match aggregation over pairwise BERTScore P/R/F1.
    """
    preds = [normalize_ws(normalize_extraction_text(s)) for s in predicted_spans if normalize_ws(normalize_extraction_text(s))]
    golds = [normalize_ws(normalize_extraction_text(s)) for s in gold_spans if normalize_ws(normalize_extraction_text(s))]

    if not preds and not golds:
        return 1.0, 1.0, 1.0
    if not preds or not golds:
        return 0.0, 0.0, 0.0

    pair_cands: List[str] = []
    pair_refs: List[str] = []
    pair_index: List[Tuple[int, int]] = []
    for i, pred in enumerate(preds):
        for j, gold in enumerate(golds):
            pair_cands.append(pred)
            pair_refs.append(gold)
            pair_index.append((i, j))

    p_tensor, r_tensor, _ = scorer.score(pair_cands, pair_refs)
    pair_p = [float(v) for v in p_tensor.tolist()]
    pair_r = [float(v) for v in r_tensor.tolist()]

    p_matrix = [[0.0 for _ in golds] for _ in preds]
    r_matrix = [[0.0 for _ in golds] for _ in preds]
    for (i, j), p_val, r_val in zip(pair_index, pair_p, pair_r):
        p_matrix[i][j] = p_val
        r_matrix[i][j] = r_val

    # Symmetric pair score for strict one-to-one span assignment.
    f_matrix = [
        [
            (2 * p_matrix[i][j] * r_matrix[i][j] / (p_matrix[i][j] + r_matrix[i][j]))
            if (p_matrix[i][j] + r_matrix[i][j]) > 0
            else 0.0
            for j in range(len(golds))
        ]
        for i in range(len(preds))
    ]
    precision, recall, f1 = _one_to_one_prf_from_pair_matrix(f_matrix)
    return precision, recall, f1
