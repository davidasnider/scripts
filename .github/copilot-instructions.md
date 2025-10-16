# Copilot Instructions for `scripts`

## Shared Guidelines

- Treat `../AGENTS.md` as the canonical contributor handbook. It covers project
  layout, commands, style, testing, and PR expectations. If that file changes,
  mirror only the relevant updates here or link back rather than diverging.

## Architectural Highlights

- `main.py` orchestrates extraction → AI analysis → Chroma persistence. It
  coordinates worker threads backed by `queue.Queue` instances; preserve status
  transitions (`pending_extraction` → `pending_analysis` → `complete`) when
  extending the pipeline.
- Manifest management happens in `data/manifest.json` using `schema.FileRecord`.
  `save_manifest_periodically` and signal handlers flush state; keep new fields
  serializable.
- Long-term storage lives in `data/chromadb`. `database_manager.add_file_to_db`
  expects summaries, descriptions, and averaged embeddings; maintain that
  contract when changing analyzers.

## Data Intake & Processing Tips

- `src/discover_files.py` precomputes `analysis_tasks` per MIME type. Extend
  `_get_analysis_tasks` for new analyses so downstream queues pick them up
  automatically.
- Failure paths should funnel through `log_unprocessed_file` to append entries
  to `data/unprocessed_files.csv`; add new error categories there for operator
  visibility.
- Video pipelines emit frames to `data/frames`; clean up temporary artifacts on
  failure to avoid blocking reruns.

## External Services & Configuration

- Model defaults are declared in `config.yaml`. Avoid hard-coding credentials;
  rely on environment variables for overrides.
- Ollama and Tesseract must be available before running pipelines. Use
  `make install` or `make check-ollama` / `make check-tesseract` to validate
  setup.
- `ai_analyzer.py` keeps a global tokenizer and expects JSON-like responses for
  downstream parsing—maintain strict schema when adjusting prompts.

## Developer Workflow Reminders

- Use `uv run streamlit run app.py` for the UI, `uv run python main.py` for
  end-to-end runs, and `uv run pre-commit run --all-files` before committing.
  See `../AGENTS.md` for the full command list.
- Keep new CLI entry points under Typer in `src/file_catalog` and follow the
  `snake_case`/`PascalCase` style described in the shared guide.
