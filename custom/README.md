# Custom Module Layout

## Folders

- `pipeline/` - shared retrieval, metrics, runtime, and IO helpers
- `setup/` - reusable setup components for index/reranker/query modules
- `benchmarks/suites/` - benchmark implementations
- `tests/` - unit tests
- `data/` - benchmark input datasets
- `results/` - benchmark outputs

## Benchmark Commands

Run from repository root:

- `python -m custom.benchmarks.suites.model_benchmark`
- `python -m custom.benchmarks.suites.strategy_comparison`
- `python -m custom.benchmarks.suites.reranker_comparison`
- `python -m custom.benchmarks.suites.semantic_highlighter`

