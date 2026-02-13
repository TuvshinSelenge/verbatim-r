import re
from typing import Dict, Set, Tuple, List


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


def tokenize(text: str) -> Set[str]:
    text = normalize_extraction_text(text).lower()
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return set(text.split())


def normalize_answer(text: str) -> str:
    text = normalize_extraction_text(text).lower()
    text = "".join(c if c.isalnum() or c.isspace() else "" for c in text)
    return " ".join(text.split())


def compute_exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_metrics(pred_span: str, gold_span: str) -> Tuple[float, float, float]:
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


def evaluate_span_extraction(extracted_spans: List[str], gold_spans: List[str]) -> Dict[str, float]:
    """SQuAD-style span evaluation."""
    if not gold_spans:
        if not extracted_spans:
            return {"exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not extracted_spans:
        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for pred in extracted_spans:
        for gold in gold_spans:
            if compute_exact_match(pred, gold):
                return {"exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    for pred in extracted_spans:
        for gold in gold_spans:
            p, r, f1 = token_metrics(pred, gold)
            if f1 > best_f1:
                best_precision = p
                best_recall = r
                best_f1 = f1

    return {
        "exact_match": 0.0,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
    }
