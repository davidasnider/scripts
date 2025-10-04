# File Catalog CLI

Command-line application for cataloging and analyzing files. The tool builds a manifest of local documents, images, and videos, extracts text or embeddings, and enables downstream analysis with local and vector-backed AI tooling.

## Features

- Detects MIME types, extracts text from PDFs and Word documents, and performs OCR on images and scanned pages.
- Generates embeddings and stores them in a local ChromaDB instance for fast semantic queries.
- Integrates with a locally hosted Ollama LLM to power interactive analysis workflows.
- Provides a modular command structure so you can swap out or extend analysis pipelines easily.

## Project layout

```text
├── data/            # Persistent manifest and vector database artifacts
├── scripts/         # Standalone maintenance or automation scripts
└── src/
    ├── content_extractor.py # Shared extraction and OCR helpers
    ├── discover_files.py   # Standalone manifest builder script
    └── file_catalog/
        └── __main__.py  # CLI entry point
```

## Getting started

```bash
uv sync --python 3.13
uv run python -m file_catalog --help
```

The `scan` command takes a directory to catalog and writes a manifest file. The `analyze` command runs follow-up analyses using the manifest. Both commands currently print placeholders; fill in the implementation using the helpers in `src/file_catalog/`.

## Discover files script

Use the `src/discover_files.py` helper when you just need to crawl a directory tree and build a manifest JSON without invoking the full CLI. The script walks every file beneath a root, records metadata (path, name, MIME type, size, and extraction status), and writes the results to a JSON file.

```bash
uv run python src/discover_files.py /path/to/directory --manifest-path data/manifest.json
```

- `root_directory` (positional) is the folder to scan. The default manifest destination is `data/manifest.json`, but you can override it with `--manifest-path`.
- MIME types are resolved with `python-magic` when libmagic is available; otherwise the script falls back to Python's built-in `mimetypes` module.
- The output is prettified JSON, suitable for checking into version control or feeding into downstream analysis.

## Development notes

- Ensure system dependencies for OCR are installed (Tesseract, poppler for `pdf2image`, etc.).
- Torch and transformers may require additional system packages depending on your hardware. Consult their documentation for accelerated backends.
- Add standalone scripts to the `scripts/` directory when workflows need bespoke orchestration outside the core CLI interface.
- Manage Python dependencies with [`uv`](https://docs.astral.sh/uv/): run `uv add <package>` to include new libraries and `uv sync` to refresh the virtual environment.
- Use `uv run pre-commit install` once to activate git hooks, then `uv run pre-commit run --all-files` to verify formatting and linting locally.
