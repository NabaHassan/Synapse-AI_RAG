#!/usr/bin/env python3
"""Backfill evidence metadata overlays into existing Qdrant payloads.

This script never deletes points and never regenerates embeddings. Use
`--dry-run` first for CAFL/Epstein.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdrant_client import QdrantClient

from src.indexing.metadata_overlay import MetadataOverlayExtractor
from src.kb_management.kb_registry import KBRegistry


def resolve_collection(kb_id: str, registry_path: Path) -> str:
    registry = KBRegistry(registry_path=str(registry_path))
    kb = registry.get_kb(kb_id)
    if not kb:
        raise ValueError(f"KB not found in registry: {kb_id}")
    collection = kb.get("collection_name")
    if not collection:
        raise ValueError(f"KB has no collection_name: {kb_id}")
    return str(collection)


def scroll_points(
    client: QdrantClient,
    collection_name: str,
    *,
    batch_size: int,
    limit: Optional[int],
) -> Iterable[Any]:
    offset = None
    seen = 0
    while True:
        remaining = None if limit is None else max(0, limit - seen)
        if remaining == 0:
            break
        current_limit = batch_size if remaining is None else min(batch_size, remaining)
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=current_limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            yield point
            seen += 1
            if limit is not None and seen >= limit:
                break
        if offset is None:
            break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "./data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing evidence_v1_overlay fields")
    parser.add_argument("--sample", type=int, default=5)
    args = parser.parse_args()

    registry_path = Path(args.data_dir) / "kb_registry.json"
    collection_name = resolve_collection(args.kb_id, registry_path)
    client = QdrantClient(url=args.qdrant_url, timeout=600, prefer_grpc=False)
    extractor = MetadataOverlayExtractor()

    started = time.time()
    scanned = 0
    updated = 0
    skipped = 0
    field_counts: Dict[str, int] = {}
    samples: List[Dict[str, Any]] = []

    for point in scroll_points(client, collection_name, batch_size=args.batch_size, limit=args.limit):
        scanned += 1
        payload = dict(point.payload or {})
        if payload.get("payload_version") == "evidence_v1_overlay" and not args.force:
            skipped += 1
            continue

        overlay = extractor.extract(payload, kb_id=args.kb_id).to_payload()
        if not overlay:
            skipped += 1
            continue

        for key, value in overlay.items():
            if value not in (None, "", []):
                field_counts[key] = field_counts.get(key, 0) + 1

        if len(samples) < args.sample:
            samples.append(
                {
                    "point_id": str(point.id),
                    "source": payload.get("source_filename") or payload.get("source"),
                    "overlay": overlay,
                }
            )

        if not args.dry_run:
            client.set_payload(
                collection_name=collection_name,
                payload=overlay,
                points=[point.id],
            )
            updated += 1

    summary = {
        "kb_id": args.kb_id,
        "collection_name": collection_name,
        "dry_run": bool(args.dry_run),
        "scanned": scanned,
        "updated": updated,
        "skipped": skipped,
        "field_counts": field_counts,
        "samples": samples,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

