from custom.pipeline.metrics import normalize_answer, normalize_extraction_text


# Verifies markdown table cleanup keeps meaningful text.
def test_markdown_normalization():
    text = "| col |\n| --- |\n| **RBI** |"
    normalized = normalize_extraction_text(text)
    assert "RBI" in normalized


# Verifies casing and extra spaces are normalized consistently.
def test_answer_normalization_whitespace_and_case():
    assert normalize_answer("  CET1   Ratio ") == "cet1 ratio"
