#!/usr/bin/env python3
"""Search the digital archive using semantic queries."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from database_manager import generate_embedding, initialize_db
from logging_utils import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    # Set up logging
    configure_logging(level=logging.INFO, force=True)
    logger = logging.getLogger("file_catalog.search")
    logger.info("Starting search archive script")
    parser = argparse.ArgumentParser(
        description="Search the digital archive with semantic queries.",
    )
    parser.add_argument("query", help="Search query text.")
    parser.add_argument(
        "--num-results",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data"),
        help="Path to the ChromaDB database (default: data).",
    )
    parser.add_argument(
        "--is-nsfw",
        action="store_true",
        help="Filter for NSFW content.",
    )
    parser.add_argument(
        "--has-financial-red-flags",
        action="store_true",
        help="Filter for content with financial red flags.",
    )

    args = parser.parse_args()
    logger.info(
        "Executing search for %r (max_results=%d, nsfw=%s, red_flags=%s, db=%s)",
        args.query,
        args.num_results,
        args.is_nsfw,
        args.has_financial_red_flags,
        args.db_path,
    )

    # Initialize DB
    collection = initialize_db(str(args.db_path))

    # Generate embedding
    query_embedding = generate_embedding(args.query)

    # Build filters
    where = {}
    if args.is_nsfw:
        where["is_nsfw"] = True
    if args.has_financial_red_flags:
        where["has_financial_red_flags"] = True

    # Query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.num_results,
        where=where or None,
    )
    logger.info("Received %d result(s) from ChromaDB", len(results["documents"]))

    # Print results
    for i, doc in enumerate(results["documents"]):
        file_data = json.loads(doc)
        logger.info("Result %d:", i + 1)
        logger.info("  Path: %s", file_data.get("file_path", "N/A"))
        logger.info("  Summary: %s", file_data.get("summary", "N/A"))
        logger.info("  MIME Type: %s", file_data.get("mime_type", "N/A"))
        logger.info("  Size: %s bytes", file_data.get("size_bytes", "N/A"))

    logger.info("Search archive operation finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
