# File Catalog CLI

Command-line application for cataloging and analyzing files. The tool builds a
manifest of local documents, images, and videos, extracts text or embeddings,
and enables downstream analysis with local and vector-backed AI tooling.

## Features

- Detects MIME types, extracts text from PDFs and Word documents, and performs
  OCR on images and scanned pages.
- Generates embeddings and stores them in a local ChromaDB instance for fast
  semantic queries.
- Integrates with a locally hosted Ollama LLM to power interactive analysis
  workflows.
- Provides a modular command structure so you can swap out or extend analysis
  pipelines easily.

## Project layout

```text
├── data/            # Persistent manifest and vector database artifacts
├── scripts/         # Standalone maintenance or automation scripts
└── src/
    ├── ai_analyzer.py          # AI analysis (LLM text/financial/image)
    ├── content_extractor.py    # Shared extraction and OCR helpers
    ├── discover_files.py       # Standalone manifest builder script
    ├── nsfw_classifier.py      # NSFW detection for images
    └── file_catalog/
        └── __main__.py         # CLI entry point
```

## Getting started

```bash
uv sync --python 3.13
uv run python -m file_catalog --help
```

The `scan` command takes a directory to catalog and writes a manifest file. The
`analyze` command runs follow-up analyses using the manifest. Both commands
currently print placeholders; fill in the implementation using the helpers in
`src/file_catalog/`.

## Content extraction utilities

The `src/content_extractor.py` module provides helpers for pulling text from
various file types:

- `preprocess_for_ocr(image)`: Prepares a PIL image for OCR by converting to
  grayscale and applying binary thresholding.
- `extract_content_from_docx(file_path)`: Extracts all text from a .docx file
  using python-docx.
- `extract_content_from_image(file_path)`: Opens an image with Pillow,
  preprocesses it for OCR, and extracts text using pytesseract.
- `extract_content_from_pdf(file_path)`: Extracts text from PDFs using a hybrid
  approach: digital text first, OCR for scanned pages (detected by low text
  length).
- `extract_frames_from_video(file_path, output_dir, interval_sec)`: Extracts
  frames from videos at specified intervals and saves them as JPEG images.

These functions are designed to be called from the main CLI or standalone
scripts for content processing.

## Development notes

- Ensure system dependencies for OCR are installed (Tesseract, poppler for
  `pdf2image`, etc.).
- Torch and transformers may require additional system packages depending on
  your hardware. Consult their documentation for accelerated backends.
- Add standalone scripts to the `scripts/` directory when workflows need bespoke
  orchestration outside the core CLI interface.
- Manage Python dependencies with [`uv`](https://docs.astral.sh/uv/): run
  `uv add <package>` to include new libraries and `uv sync` to refresh the
  virtual environment.
- Use `uv run pre-commit install` once to activate git hooks, then
  `uv run pre-commit run --all-files` to verify formatting and linting locally.
