import concurrent.futures
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from custom.pipeline.constants import RAPIDFUZZ_THRESHOLDS


def _threshold_label(threshold: float) -> str:
    return f"{int(round(threshold * 100))}"


def _rapidfuzz_metric_key(threshold_label: str, metric: str) -> str:
    return f"rapidfuzz_{threshold_label}_{metric}"


def _rapidfuzz_threshold_labels(
    thresholds: Sequence[float] = RAPIDFUZZ_THRESHOLDS,
) -> List[str]:
    return [_threshold_label(threshold) for threshold in thresholds]


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report_lines(lines: List[str], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines))


def build_table_report_lines(
    title: str,
    width: int,
    header: str,
    row_lines: List[str],
    leading_blank_lines: int = 0,
) -> List[str]:
    """
    Build a printable/writable fixed-width table report.
    """
    lines: List[str] = []
    if leading_blank_lines > 0:
        lines.append("\n" * leading_blank_lines + "=" * width)
    else:
        lines.append("=" * width)
    lines.append(f"{title:^{width}}")
    lines.append("=" * width)
    lines.append(header)
    lines.append("-" * width)
    lines.extend(row_lines)
    lines.append("=" * width)
    return lines


def print_and_write_report(lines: List[str], output_path: Path) -> None:
    """
    Print report lines and persist them to disk.
    """
    print("\n".join(lines))
    write_report_lines(lines, output_path)
    print(f"\nResults saved to: {output_path}")


def zero_chunk_metrics() -> Dict[str, float]:
    """
    Default retrieval metrics used on evaluation failure.
    """
    return {"hit_rate": 0.0, "mrr": 0.0, "recall@k": 0.0}


def zero_span_metrics() -> Dict[str, float]:
    """
    Default extraction metrics used on evaluation failure.
    """
    metrics = {
        "rouge_l_precision": 0.0,
        "rouge_l_recall": 0.0,
        "rouge_l_f1": 0.0,
        "bertscore_precision": 0.0,
        "bertscore_recall": 0.0,
        "bertscore_f1": 0.0,
        "unanswerable_accuracy": 0.0,
    }
    for threshold_label in _rapidfuzz_threshold_labels():
        metrics[_rapidfuzz_metric_key(threshold_label, "precision")] = 0.0
        metrics[_rapidfuzz_metric_key(threshold_label, "recall")] = 0.0
        metrics[_rapidfuzz_metric_key(threshold_label, "f1")] = 0.0
    return metrics


def make_benchmark_result_row(
    labels: Dict[str, Any],
    chunk_metrics: Dict[str, float],
    span_metrics: Dict[str, float],
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build one standardized benchmark result row.
    """
    row: Dict[str, Any] = {
        **labels,
        "Hit Rate": chunk_metrics.get("hit_rate", 0.0),
        "Recall@K": chunk_metrics.get("recall@k", 0.0),
        "MRR": chunk_metrics.get("mrr", 0.0),
        "Unans.Acc": span_metrics.get("unanswerable_accuracy", 0.0),
    }
    for threshold_label in _rapidfuzz_threshold_labels():
        row[f"RF@{threshold_label}-P"] = span_metrics.get(
            _rapidfuzz_metric_key(threshold_label, "precision"), 0.0
        )
        row[f"RF@{threshold_label}-R"] = span_metrics.get(
            _rapidfuzz_metric_key(threshold_label, "recall"), 0.0
        )
        row[f"RF@{threshold_label}-F1"] = span_metrics.get(
            _rapidfuzz_metric_key(threshold_label, "f1"), 0.0
        )
    row["RougeF1"] = span_metrics.get("rouge_l_f1", 0.0)
    row["BertF1"] = span_metrics.get("bertscore_f1", 0.0)
    if extras:
        row.update(extras)
    return row


def build_benchmark_header_and_rows(
    rows: List[Dict[str, Any]],
    leading_columns: Sequence[Tuple[str, int]],
) -> Tuple[str, List[str]]:
    """
    Build header and row lines for benchmark tables with shared metric columns.
    """
    metric_columns: List[Tuple[str, int, bool]] = [
        ("Hit Rate", 8, True),
        ("Recall@K", 8, True),
        ("MRR", 6, True),
        ("RougeF1", 8, True),
        ("BertF1", 8, True),
        ("Unans.Acc", 9, True),
    ]
    rapidfuzz_columns: List[Tuple[str, int, bool]] = []
    for threshold_label in _rapidfuzz_threshold_labels():
        rapidfuzz_columns.extend(
            [
                (f"RF@{threshold_label}-P", 8, True),
                (f"RF@{threshold_label}-R", 8, True),
                (f"RF@{threshold_label}-F1", 9, True),
            ]
        )
    metric_columns = metric_columns[:3] + rapidfuzz_columns + metric_columns[3:]
    all_columns: List[Tuple[str, int, bool]] = [(name, width, False) for name, width in leading_columns] + metric_columns

    header = " | ".join(f"{name:<{width}}" for name, width, _ in all_columns)
    row_lines: List[str] = []
    for row in rows:
        parts: List[str] = []
        for name, width, is_metric in all_columns:
            value = row.get(name, "")
            if is_metric:
                parts.append(f"{float(value):.3f}".ljust(width))
            else:
                parts.append(f"{str(value):<{width}}")
        row_lines.append(" | ".join(parts))
    return header, row_lines


def init_span_metric_lists() -> Dict[str, List[float]]:
    """
    Create empty accumulators for span-level metric means.
    """
    metric_lists = {
        "rouge_l_precision": [],
        "rouge_l_recall": [],
        "rouge_l_f1": [],
        "bertscore_precision": [],
        "bertscore_recall": [],
        "bertscore_f1": [],
    }
    for threshold_label in _rapidfuzz_threshold_labels():
        metric_lists[_rapidfuzz_metric_key(threshold_label, "precision")] = []
        metric_lists[_rapidfuzz_metric_key(threshold_label, "recall")] = []
        metric_lists[_rapidfuzz_metric_key(threshold_label, "f1")] = []
    return metric_lists


def append_span_metric_scores(
    metric_lists: Dict[str, List[float]],
    rouge_scores: Tuple[float, float, float],
    bert_scores: Tuple[float, float, float],
    rapidfuzz_scores: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> None:
    """
    Append one successful query's span metric scores to accumulators.
    """
    rouge_p, rouge_r, rouge_f1 = rouge_scores
    bert_p, bert_r, bert_f1 = bert_scores
    metric_lists["rouge_l_precision"].append(rouge_p)
    metric_lists["rouge_l_recall"].append(rouge_r)
    metric_lists["rouge_l_f1"].append(rouge_f1)
    metric_lists["bertscore_precision"].append(bert_p)
    metric_lists["bertscore_recall"].append(bert_r)
    metric_lists["bertscore_f1"].append(bert_f1)
    rapidfuzz_scores = rapidfuzz_scores or {}
    for threshold_label in _rapidfuzz_threshold_labels():
        p, r, f1 = rapidfuzz_scores.get(threshold_label, (0.0, 0.0, 0.0))
        metric_lists[_rapidfuzz_metric_key(threshold_label, "precision")].append(p)
        metric_lists[_rapidfuzz_metric_key(threshold_label, "recall")].append(r)
        metric_lists[_rapidfuzz_metric_key(threshold_label, "f1")].append(f1)


def append_zero_span_metric_scores(metric_lists: Dict[str, List[float]]) -> None:
    """
    Append zero scores for one failed query.
    """
    for key in metric_lists:
        metric_lists[key].append(0.0)


def summarize_span_metric_lists(
    metric_lists: Dict[str, List[float]],
    unanswerable_correct: List[int],
) -> Dict[str, float]:
    """
    Convert span metric accumulators into final mean metrics.
    """
    summary = {
        "rouge_l_precision": mean(metric_lists["rouge_l_precision"]) if metric_lists["rouge_l_precision"] else 0.0,
        "rouge_l_recall": mean(metric_lists["rouge_l_recall"]) if metric_lists["rouge_l_recall"] else 0.0,
        "rouge_l_f1": mean(metric_lists["rouge_l_f1"]) if metric_lists["rouge_l_f1"] else 0.0,
        "bertscore_precision": mean(metric_lists["bertscore_precision"]) if metric_lists["bertscore_precision"] else 0.0,
        "bertscore_recall": mean(metric_lists["bertscore_recall"]) if metric_lists["bertscore_recall"] else 0.0,
        "bertscore_f1": mean(metric_lists["bertscore_f1"]) if metric_lists["bertscore_f1"] else 0.0,
        "unanswerable_accuracy": mean(unanswerable_correct) if unanswerable_correct else 1.0,
    }
    for threshold_label in _rapidfuzz_threshold_labels():
        precision_key = _rapidfuzz_metric_key(threshold_label, "precision")
        recall_key = _rapidfuzz_metric_key(threshold_label, "recall")
        f1_key = _rapidfuzz_metric_key(threshold_label, "f1")
        summary[precision_key] = mean(metric_lists[precision_key]) if metric_lists[precision_key] else 0.0
        summary[recall_key] = mean(metric_lists[recall_key]) if metric_lists[recall_key] else 0.0
        summary[f1_key] = mean(metric_lists[f1_key]) if metric_lists[f1_key] else 0.0
    return summary


def run_with_timeout(func, timeout_sec: int):
    """Run a function in a thread with a hard timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            print(f"  TIMEOUT after {timeout_sec}s - skipping")
            raise TimeoutError(f"Timed out after {timeout_sec}s")
