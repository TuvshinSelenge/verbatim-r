from custom.pipeline.retrieval import merge_and_dedup


# Tiny fake index so we can unit-test dedup without Milvus.
class DummyIndex:
    def __init__(self, hits_by_query):
        self.hits_by_query = hits_by_query

    def query(self, q, k=10):
        return self.hits_by_query.get(q, [])[:k]


# Duplicate chunks across queries should collapse to one result.
def test_merge_and_dedup_uses_source_chunk_and_text_prefix():
    hit_a = {"text": "same text", "metadata": {"source_file": "a.pdf", "chunk_index": 1}}
    hit_b = {"text": "same text", "metadata": {"source_file": "a.pdf", "chunk_index": 1}}
    hit_c = {"text": "different", "metadata": {"source_file": "a.pdf", "chunk_index": 2}}
    idx = DummyIndex({"q1": [hit_a, hit_c], "q2": [hit_b]})

    merged = merge_and_dedup(["q1", "q2"], idx, k=10)
    assert len(merged) == 2
