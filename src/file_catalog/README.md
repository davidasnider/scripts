# `src/file_catalog/` Directory

This directory serves as the entry point for the command-line interface (CLI) of the file catalog application.

## `__main__.py`

The `__main__.py` module is responsible for defining and handling the CLI commands. It uses the `argparse` library to create a user-friendly interface for interacting with the application from the command line.

### Commands

The CLI provides the following commands:

-   **`scan`**: This command scans a specified directory, discovers all the files within it, and creates a manifest file in JSON format. The manifest contains metadata for each file, such as its path, size, MIME type, and SHA256 hash.

-   **`analyze`**: This command runs various analysis tasks on the files listed in the manifest. These tasks can include text extraction, AI-powered summarization, and other forms of content analysis.

### Usage

To use the CLI, you can run the `file_catalog` module as a script:

```bash
uv run python -m src.file_catalog [COMMAND] [OPTIONS]
```

#### Scan a Directory

To scan a directory and generate a manifest, use the `scan` command:

```bash
uv run python -m src.file_catalog scan /path/to/your/files --manifest /path/to/your/manifest.json
```

#### Analyze the Manifest

To run the analysis pipeline on an existing manifest, use the `analyze` command:

```bash
uv run python -m src.file_catalog analyze --manifest /path/to/your/manifest.json
```
