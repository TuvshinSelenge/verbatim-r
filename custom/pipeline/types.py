from dataclasses import dataclass


@dataclass
class SearchResultWrapper:
    text: str


@dataclass(frozen=True)
class TokenPairMatchCandidate:
    """
    One token-overlap match candidate between a prediction and a gold span.
    """
    pair_f1: float
    tp: int
    pred_idx: int
    gold_idx: int
    fp: int
    fn: int
