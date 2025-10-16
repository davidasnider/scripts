# Copilot Instructions for `scripts`

## Core Architecture

- End-to-end pipeline lives in `main.py`, orchestrating extraction → AI analysis
  → vector persistence with three thread pools (extraction, analysis, database)
  fed by shared `queue.Queue` work queues.
- File metadata/state comes from `data/manifest.json`, which stores
  `schema.FileRecord` entries including `analysis_tasks`;
  `save_manifest_periodically` flushes progress every 10s and signal handlers
  persist on shutdown.
- Persistent embeddings and metadata are stored in a Chroma collection at
  `data/chromadb`; `database_manager.add_file_to_db` consolidates
  summary/description/text and writes a 768-d vector (averaged chunks when
  needed).

## Data Intake & Task Assignment

- `src/discover_files.py create_file_manifest` scans a root path, limits to 10
  files per MIME type (default 100 total), and precomputes `analysis_tasks`
  based on MIME; update `_get_analysis_tasks` when adding new analyses so
  downstream workers pick them up automatically.
- `FileRecord.status` drives routing: `pending_extraction` → extraction queue,
  `pending_analysis` → analysis queue, `complete` skipped; ensure custom code
  preserves this contract or queues stall.
- Failures funnel through `log_unprocessed_file`, appending to
  `data/unprocessed_files.csv`; add new failure modes here so operators can
  triage skipped assets.

## AI & External Services

- Model names and embeddings are configured in `config.yaml`; Ollama must be
  running locally (`make setup-ollama` starts the daemon and pulls `llama3`,
  `deepseek-coder-v2`, `llava`, `nomic-embed-text`).
- `ai_analyzer.py` handles text/image/video/financial tasks, chunking long
  inputs via `text_utils.chunk_text` (global HuggingFace tokenizer loads at
  import). Keep outputs strictly JSON when extending prompts so downstream
  parsing stays resilient.
- `NSFWClassifier` initializes a HuggingFace vision pipeline on every
  instantiation; reuse a single instance when adding new image/video flows to
  avoid repeated model loads.

## Developer Workflows

- Environment is managed with `uv`; run `make install` to sync dependencies and
  install Tesseract/Ollama, `uv run python src/discover_files.py <path>` to
  build a manifest, then
  `uv run python main.py [--model text_analysis|image_description|video_summary|all]`
  to process.
- Launch the Streamlit explorer with `uv run streamlit run app.py` (expects
  processed manifest + populated Chroma DB). `make clean` wipes manifest,
  embeddings, and extracted frames for a fresh run.
- Enable git hooks via `uv run pre-commit install`; lint with
  `uv run pre-commit run --all-files`. Ruff/Black settings are in
  `pyproject.toml` (line length 88, double quotes enforced).

## Implementation Patterns & Gotchas

- Video handling writes frames into `data/frames` and runs per-frame image
  description + NSFW checks; guard against `.asx` playlists (already skipped)
  when adding formats.
- Financial analysis only fires when `extracted_text` contains "financial" or
  "account"; consider adjusting this heuristic if expanding coverage.
- `scripts/search_archive.py` currently imports `database_manager` without the
  `src.` prefix—run it with `PYTHONPATH=src` or update imports when modifying
  that script.
- `app.py` depends on `collection.get` including metadata, then fetches richer
  summaries from `data/manifest.json`; keep manifest fields in sync with UI
  expectations (summary/description/is_nsfw/has_financial_red_flags).
- Heavy logging lands in `file_catalog.log`; respect existing logging setup when
  adding threads to keep progress reporting consistent.

Let me know if any part of these instructions needs clarification or coverage of
additional workflows.
