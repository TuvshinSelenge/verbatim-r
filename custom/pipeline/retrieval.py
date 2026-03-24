from typing import List, Optional, Tuple


def get_text_and_meta(chunk) -> Tuple[str, dict]:
    """Extract text and metadata from chunk (dict or object)."""
    if isinstance(chunk, dict):
        return chunk.get("text", ""), chunk.get("metadata", {}) or {}
    return getattr(chunk, "text", ""), getattr(chunk, "metadata", {}) or {}


def extract_preds(chunks) -> List[Tuple]:
    """Extract (source_file, chunk_index) tuples from chunks."""
    preds = []
    for h in chunks:
        _, m = get_text_and_meta(h)
        preds.append((m.get("source_file"), m.get("chunk_index")))
    return preds


def merge_and_dedup(
    queries: List[str], rag_index, k: int, filter: Optional[str] = None
) -> list:
    """Search with multiple queries, then merge and deduplicate hits."""
    merged, seen = [], set()
    for q in queries:
        hits = rag_index.query(q, k=k, filter=filter)
        for h in hits:
            t, m = get_text_and_meta(h)
            if not t:
                continue
            key = (m.get("source_file"), m.get("chunk_index"), t[:200])
            if key in seen:
                continue
            seen.add(key)
            merged.append(h)
    return merged


def retrieve_and_rerank(
    query_text: str,
    query_rewriter,
    query_generator,
    rag_index,
    reranker,
    top_k: int,
    per_subq_k: int,
    filter: Optional[str] = None,
    bank_name: Optional[str] = None,
    bank_short_name: Optional[str] = None,
) -> Tuple[list, str, List[Tuple]]:
    """
    Standard retrieval pass: rewrite -> generate -> search -> merge -> rerank.

    Returns:
      - reranked chunks
      - rewritten query
      - prediction tuples (source_file, chunk_index)
    """
    if bank_name is not None or bank_short_name is not None:
        rewritten = query_rewriter.rewrite(
            query_text,
            bank_name if bank_name is not None else query_rewriter.bank_name,
            bank_short_name if bank_short_name is not None else query_rewriter.bank_short_name,
        )
        subqs = query_generator.generate_queries(
            rewritten,
            bank_name if bank_name is not None else query_generator.bank_name,
            bank_short_name if bank_short_name is not None else query_generator.bank_short_name,
        )
    else:
        rewritten = query_rewriter.rewrite(query_text)
        subqs = query_generator.generate_queries(rewritten)
    merged = merge_and_dedup(subqs, rag_index, per_subq_k, filter=filter)
    reranked, _ = reranker.rerank(rewritten, merged, top_k=top_k)
    return reranked, rewritten, extract_preds(reranked)
