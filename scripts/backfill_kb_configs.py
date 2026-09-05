#!/usr/bin/env python3
"""Backfill KB config snapshots for existing registry entries."""

import argparse
import os
from pathlib import Path

from src.kb_management import KBRegistry, KBManager
from src.kb_management.kb_manager import KBManagerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill KB config snapshots")
    parser.add_argument("--kb-id", help="Only backfill this KB", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate snapshots even if already present",
    )
    args = parser.parse_args()

    data_dir = os.getenv("DATA_DIR", "./data")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    config_dir = os.getenv("CONFIG_DIR", str(Path(__file__).resolve().parents[1] / "src" / "config"))
    registry_path = os.getenv("KB_REGISTRY_PATH", str(Path(data_dir) / "kb_registry.json"))

    registry = KBRegistry(registry_path=registry_path)
    kb_manager = KBManager(
        registry=registry,
        config=KBManagerConfig(
            qdrant_url=qdrant_url,
            data_dir=data_dir,
            config_dir=config_dir,
        ),
    )

    targets = [args.kb_id] if args.kb_id else [kb["kb_id"] for kb in registry.list_kbs()]
    for kb_id in targets:
        kb = registry.get_kb(kb_id)
        if kb is None:
            print(f"KB not found: {kb_id}")
            continue

        snapshot_path = kb.get("config_path")
        snapshot_exists = bool(snapshot_path and Path(snapshot_path).exists())
        if snapshot_exists and kb.get("profile_template_id") and not args.force:
            print(f"Skipping {kb_id} (snapshot already present)")
            continue

        result = kb_manager.create_or_update_snapshot(kb_id)
        print(f"Updated {kb_id}: {result['snapshot_path']}")


if __name__ == "__main__":
    main()
