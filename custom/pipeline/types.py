from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SearchResultWrapper:
    text: str


@dataclass
class RetrievalMetrics:
    hit_rate: float
    recall_at_k: float
    mrr: float


@dataclass
class SpanMetrics:
    exact_match: float
    precision: float
    recall: float
    f1: float


@dataclass
class BenchmarkRow:
    model: str
    retrieval: Dict[str, float]
    extraction: Dict[str, float]
    details: List[dict]
