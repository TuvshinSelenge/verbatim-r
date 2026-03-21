# Custom Module Layout

## Folders

- `pipeline/` - shared retrieval, metrics, runtime, and IO helpers
- `setup/` - reusable setup components for index/reranker/query modules
- `benchmarks/collection/` - benchmark implementations
- `tests/` - unit tests
- `data/` - benchmark input datasets (`bank_profiles.json` maps banks → span files + PDF path hints)
- `results/` - benchmark outputs

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

