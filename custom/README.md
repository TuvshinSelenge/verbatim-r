# Custom Module Layout

## Folders

- `pipeline/` - shared retrieval, metrics, runtime, and IO helpers
- `setup/` - reusable setup components for index/reranker/query modules
- `benchmarks/collection/` - benchmark implementations
- `tests/` - unit tests
- `data/` - benchmark input datasets
- `results/` - benchmark outputs

## Benchmark Commands

Run from repository root:

- `python -m custom.benchmarks.collection.model_benchmark`
- `python -m custom.benchmarks.collection.strategy_comparison` (part 1: baseline, baseline+reranker, baseline+rewriting+reranker)
- `python -m custom.benchmarks.collection.strategy_comparison_part2` (part 2: multi-query strategies)
- `python -m custom.benchmarks.collection.reranker_comparison`
- `python -m custom.benchmarks.collection.semantic_highlighter`
- `python -m custom.benchmarks.collection.answer_relevance_benchmark` (full pipeline with Gemini; ROUGE-L F1)


## Answer Relevance Benchmark Notes

`answer_relevance_benchmark.py` runs the full pipeline:

1. rewrite query
2. generate sub-queries
3. retrieve and rerank chunks
4. extract spans
5. score extracted-text quality

Metrics produced:

- `ROUGE-L F1` 
- `Unanswerable accuracy`

Outputs:

- `custom/results/answer_relevance_results.txt` (summary table)
- `custom/results/answer_relevance_details.json` (per-query details)

Requirements:

- no extra metric dependency (ROUGE-L is implemented directly in the benchmark script)

