# `src/` Directory

This directory contains the core source code for the file cataloging and
analysis application.

## Modules

- **`access_analysis.py`**: Provides utilities for ingesting and analyzing
  Microsoft Access databases, including table loading and data extraction.

- **`ai_analyzer.py`**: Handles AI-powered analysis of file content using the
  Ollama library. It includes functions for text summarization, entity
  recognition, and other AI-driven insights.

- **`backup_manager.py`**: Manages the automated backup of the manifest file,
  with a tiered retention policy to preserve historical snapshots.

- **`config_utils.py`**: Contains helper functions for loading and parsing the
  project's `config.yaml` file, providing a centralized way to manage
  configuration.

- **`content_extractor.py`**: A suite of functions for extracting text and other
  content from various file formats, such as PDFs, DOCX files, images (via OCR),
  and videos.

- **`database_manager.py`**: Manages the ChromaDB vector database, including
  initialization, data insertion, and the generation of embeddings for semantic
  search.

- **`discover_files.py`**: The main script for discovering files in a directory,
  calculating their metadata (SHA256 hash, MIME type, etc.), and creating the
  initial manifest.

- **`filters.py`**: Implements the logic for filtering the file manifest based
  on various criteria, such as file type, NSFW content, and analysis status.

- **`logging_utils.py`**: A utility module for configuring the application's
  logging, ensuring consistent log formatting and output.

- **`manifest_utils.py`**: Provides helper functions for managing and updating
  the file manifest, including resetting outdated analysis tasks and preparing
  records for re-scanning.

- **`nsfw_classifier.py`**: A module for detecting Not-Safe-For-Work (NSFW)
  content in images using a pre-trained transformer model.

- **`schema.py`**: Defines the Pydantic data models and enums used throughout
  the application, ensuring data consistency and validation.

- **`task_utils.py`**: Contains logic for determining which analysis tasks
  should be performed on a given file based on its MIME type.

- **`text_utils.py`**: Provides utilities for text processing, such as chunking
  text into smaller segments and counting tokens, which is essential for working
  with LLMs.

## Subdirectories

- **`file_catalog/`**: The entry point for the command-line interface (CLI) of
  the application.
