# Repository Guidelines

## Project Structure & Module Organization

Core code sits in `src/`. `main.py` drives the pipeline with `ai_analyzer.py`,
`content_extractor.py`, `database_manager.py`, and the Typer CLI in
`src/file_catalog/__main__.py`. Keep the Streamlit UI in `app.py`, one-off
utilities in `scripts/`, and generated artifacts (manifest, ChromaDB, frames)
under `data/`. Adjust model defaults via `config.yaml`; store credentials in
environment variables, not commits.

## Build, Test, and Development Commands

- `make install` — Syncs dependencies, checks Tesseract/Ollama, and pulls
  required models.
- `make run` — Starts the Streamlit interface (`uv run streamlit run app.py`).
- `uv run python src/discover_files.py /path/to/dir` — Refreshes the file
  manifest.
- `uv run python main.py` — Executes extraction, embedding, and database
  updates.
- `uv run pre-commit run --all-files` — Runs Black and Ruff before committing.
- `make clean` — Clears manifests, ChromaDB indexes, and cached frames.

## Coding Style & Naming Conventions

Target Python 3.11–3.13. Format with Black (`line-length = 88`) and lint with
Ruff (E, F, I; double quotes; spaces). Use `snake_case` for functions,
variables, and files; `PascalCase` for classes; uppercase constants; and type
hints on public APIs. New CLI surfaces should route through Typer and live
beside related modules in `src/`.

## Testing Guidelines

Automated tests are not yet established. Smoke-test changes with
`uv run python main.py` and inspect outputs in `data/`. When adding tests,
mirror the module layout in a `tests/` package, use `pytest`, name files
`test_<feature>.py`, and favor focused fixtures over large binaries. Cover
extraction edge cases and AI fallback paths before merging.

## Commit & Pull Request Guidelines

Follow the conventional commit style in history (`feat:`, `fix:`, `chore:`) and
keep each change focused. Run `uv run pre-commit run --all-files` before
pushing, document config or dependency updates, and include screenshots or logs
for UI or pipeline tweaks. Reference related issues or TODOs and flag any
required data resets in the PR description.

## Configuration & Environment Notes

Model defaults live in `config.yaml`; do not commit secrets. `data/` is
gitignored for local experiments—sanitize artifacts before sharing. Long-running
workflows assume Ollama and Tesseract are available (`make check-ollama`,
`make check-tesseract`) prior to pipeline runs.
