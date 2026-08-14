# Custom Module Layout

## Folders

- `pipeline/` - shared retrieval, metrics, runtime, and IO helpers
- `setup/` - reusable setup components for index/reranker/query modules
- `indexing/` - Docling-based PDF parsing and Milvus index creation
- `benchmarks/collection/` - benchmark implementations
- `tests/` - unit tests
- `data/` - benchmark input datasets (`bank_profiles.json` maps banks → span files + PDF path hints)
- `results/` - benchmark outputs

## Index creation

`indexing/create_index.py` contains the thesis-specific integration of Docling's
`DocumentConverter` and `HybridChunker` with Verbatim RAG and local Milvus. It
enables table-structure recognition, serializes tables as Markdown, merges peer
chunks, and stores source/page/chunk metadata alongside dense MiniLM and sparse
SPLADE vectors.

Build a new index from all PDFs in a folder:

```bash
PYTHONPATH=. python -m custom.indexing.create_index \
  --pdf-folder /path/to/annual-reports \
  --db-path custom/milvus_verbatim_new.db
```

The full build asks before replacing an existing database. Use `--force` or
`--yes` only when replacement is intentional.

Add newly discovered PDFs while retaining existing records:

```bash
PYTHONPATH=. python -m custom.indexing.create_index \
  --pdf-folder /path/to/annual-reports \
  --db-path custom/milvus_verbatim_new.db \
  --incremental --yes
```

Replace selected reports in an existing index:

```bash
PYTHONPATH=. python -m custom.indexing.create_index \
  --pdf-folder /path/to/annual-reports \
  --db-path custom/milvus_verbatim_new.db \
  --incremental --reindex "Report A.pdf" "Report B.pdf" --yes
```

Defaults can be configured with `PDF_SOURCE_FOLDER` and `CUSTOM_DB_PATH`
(`DB_PATH` is also accepted). Use the same embedding model and chunk settings
for incremental updates as for the original index.

## Multi-bank testing

0. **Which `.db` file?** If `custom/milvus_verbatim_new.db` exists, the custom RAG service and `/api/custom/banks` use it automatically (unless you set `CUSTOM_DB_PATH` or `DB_PATH`). Otherwise they use `custom/milvus_verbatim.db`. Set `CUSTOM_DB_PATH` explicitly if you keep multiple DBs and need a specific one.

   **See real paths in Milvus** (stop guessing `source_substrings`): open `GET /api/custom/indexed-sources` or run `PYTHONPATH=. python -m custom.setup.interactive_query_demo --list-sources`. Copy the exact `value` / `basename` into `bank_profiles.json` — spelling must match what is stored (e.g. `Erst_Bank.pdf` vs `Erste_Bank.pdf`).

1. **Profiles** – Edit `data/bank_profiles.json`:
   - **`source_basenames`** – PDF filenames only (e.g. `Erste_Bank.pdf`); matching ignores spaces vs underscores vs hyphens (`Erste Bank.pdf` ≡ `Erste_Bank.pdf`).
   - **`source_substrings`** – substrings of the stored path; matching weights the **PDF basename** higher than parent folders. Chunk metadata may use `source_file`, `source`, `filename`, `file_path`, etc. (see `/api/custom/indexed-sources`).
   - **`match_dataset_id`** – if your ingest sets **`metadata.dataset_id`** (e.g. `erste`, `rbi`), matching uses that even when the file path does not contain the bank name.
   - **`exact_source_paths`** (optional) – one or more path fragments that must match the stored source string.
   If a bank still does not match, the API returns an error instead of searching the whole index.
2. **Default bank (prompts)** – Set `CUSTOM_BANK_ID` (e.g. `rbi`, `bawag`) for the custom RAG service’s default rewriter/generator wording when the API does not pass `bank_id`.
3. **API** – `POST /api/custom/query/stream` accepts `bank_id` and optional `metadata_filter` (Milvus expression; overrides `bank_id` when set). Omit `bank_id` to search all chunks (same as before).
4. **Web UI** – Start API (`PYTHONPATH=. uvicorn api.app:app --reload --port 8000`) and frontend (`npm run dev` in `frontend/`). Use **Bank / report scope** above the question box; it loads options from `GET /api/custom/banks` and sends `bank_id` with `POST /api/custom/query/stream`.
5. **CLI demo** – From repo root:
   - `PYTHONPATH=. python -m custom.setup.interactive_query_demo --list-banks`
   - `PYTHONPATH=. python -m custom.setup.interactive_query_demo --bank bawag`
   - `PYTHONPATH=. python -m custom.setup.interactive_query_demo --bank rbi --run-spans`

## Benchmark Commands

Run from repository root:

- `python -m custom.benchmarks.collection.model_benchmark`
- `python -m custom.benchmarks.collection.strategy_comparison` (part 1: baseline, baseline+reranker, baseline+rewriting+reranker)
- `python -m custom.benchmarks.collection.strategy_comparison_part2` (part 2: multi-query strategies)
- `python -m custom.benchmarks.collection.reranker_comparison`
- `python -m custom.benchmarks.collection.semantic_highlighter`

## Per-bank run commands

Run from repository root (`PYTHONPATH=.` ensures package imports work):

### Interactive query demo (bank-scoped)

- RBI: `PYTHONPATH=. python -m custom.setup.interactive_query_demo --bank rbi`
- BAWAG: `PYTHONPATH=. python -m custom.setup.interactive_query_demo --bank bawag`
- Erste: `PYTHONPATH=. python -m custom.setup.interactive_query_demo --bank erste`
- UniCredit: `PYTHONPATH=. python -m custom.setup.interactive_query_demo --bank uni`

Optional: run the configured span suite for that bank:

- `PYTHONPATH=. python -m custom.setup.interactive_query_demo --bank <bank_id> --run-spans`

### Model benchmark (bank-scoped)

`model_benchmark.py` supports per-bank files and Milvus scope filter:

- RBI: `PYTHONPATH=. python -m custom.benchmarks.collection.model_benchmark --bank rbi`
- BAWAG: `PYTHONPATH=. python -m custom.benchmarks.collection.model_benchmark --bank bawag`
- Erste: `PYTHONPATH=. python -m custom.benchmarks.collection.model_benchmark --bank erste`
- UniCredit: `PYTHONPATH=. python -m custom.benchmarks.collection.model_benchmark --bank uni`

If needed, override data files explicitly:

- `PYTHONPATH=. python -m custom.benchmarks.collection.model_benchmark --bank erste --chunk-data custom/data/chunk_ids_Erste.json --span-data custom/data/span_Erste.json`

Results are saved as bank-specific files, e.g.:

- `custom/results/benchmark_results_rbi.txt`
- `custom/results/benchmark_results_bawag.txt`
- `custom/results/benchmark_results_erste.txt`
- `custom/results/benchmark_results_uni.txt`

