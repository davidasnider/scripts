# Scripts Directory

This directory contains standalone scripts for maintenance, automation, and other tasks that are not part of the main application's core logic.

## `prune_duplicates.py`

This script identifies and removes duplicate files from the manifest based on their SHA256 hash. It provides options for a dry run to preview changes and for creating a backup of the manifest before making any modifications.

### Usage

```bash
uv run python scripts/prune_duplicates.py [OPTIONS]
```

### Command-Line Options

-   `--manifest`: Path to the manifest JSON file. Defaults to `data/manifest.json`.
-   `--dry-run`: Preview the duplicate pruning actions without deleting files or modifying the manifest.
-   `--log-level`: Set the logging level. Defaults to `INFO`.
-   `--backup`: Create a backup of the original manifest before writing changes. Defaults to `True`.

### Example

To perform a dry run of the duplicate pruning process, use the following command:

```bash
uv run python scripts/prune_duplicates.py --dry-run
```

To permanently remove duplicates and update the manifest, run the script without the `--dry-run` flag:

```bash
uv run python scripts/prune_duplicates.py
```

## `search_archive.py`

This script allows you to perform semantic searches on the digital archive using a text query. It generates an embedding for the query and retrieves the most relevant documents from the ChromaDB database.

### Usage

```bash
uv run python scripts/search_archive.py "your search query" [OPTIONS]
```

### Command-Line Options

-   `query`: The search query text.
-   `--num-results`: The number of search results to return. Defaults to `5`.
-   `--db-path`: Path to the ChromaDB database. Defaults to `data`.
-   `--is-nsfw`: Filter for NSFW content.
-   `--has-financial-red-flags`: Filter for content with financial red flags.

### Example

To search for documents related to "project plans," you can use the following command:

```bash
uv run python scripts/search_archive.py "project plans"
```

To search for NSFW content with the query "annual report" and limit the results to 10, you can use:

```bash
uv run python scripts/search_archive.py "annual report" --num-results 10 --is-nsfw
```
