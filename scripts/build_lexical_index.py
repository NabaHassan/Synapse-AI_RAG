#!/usr/bin/env python3
"""Build a SQLite FTS lexical sidecar from an existing Qdrant collection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdrant_client import QdrantClient

from src.retrieval.lexical_search_backend import SQLiteFTSLexicalSearchBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--index-dir", default="./data/lexical_indices", help="Directory for SQLite FTS files")
    parser.add_argument("--batch-size", type=int, default=1000, help="Qdrant scroll batch size")
    parser.add_argument("--limit", type=int, default=None, help="Optional document limit for smoke tests")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    client = QdrantClient(url=args.qdrant_url, timeout=600, prefer_grpc=False)
    backend = SQLiteFTSLexicalSearchBackend(
        collection_name=args.collection,
        qdrant_client=client,
        index_dir=args.index_dir,
    )
    result = backend.build_from_qdrant(batch_size=args.batch_size, limit=args.limit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
