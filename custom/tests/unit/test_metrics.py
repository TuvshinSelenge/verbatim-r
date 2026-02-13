from custom.pipeline.metrics import (
    compute_exact_match,
    evaluate_span_extraction,
    token_metrics,
)


# Checks that markdown/casing normalization still allows exact matches.
def test_exact_match_normalized_true():
    assert compute_exact_match("**Capital Ratio**", "capital ratio")


# Checks that partial token overlap gives non-zero precision/recall/F1.
def test_token_metrics_partial_overlap():
    p, r, f1 = token_metrics("tier 1 capital ratio", "capital ratio")
    assert p > 0
    assert r > 0
    assert f1 > 0


# Checks expected behavior for unanswerable questions.
def test_unanswerable_scoring():
    assert evaluate_span_extraction([], [])["exact_match"] == 1.0
    assert evaluate_span_extraction(["something"], [])["exact_match"] == 0.0
