#!/usr/bin/env python3
"""Force a KB to use a specific profile template and regenerate its snapshot."""

import argparse
import os
from pathlib import Path

from src.kb_management import KBRegistry, KBManager
from src.kb_management.kb_manager import KBManagerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Set KB profile template")
    parser.add_argument("--kb-id", required=True, help="KB identifier")
    parser.add_argument("--profile-template-id", required=True, help="Profile template id")
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

    result = kb_manager.create_or_update_snapshot(
        kb_id=args.kb_id,
        profile_template_id=args.profile_template_id,
    )
    print(f"Updated {args.kb_id} -> {args.profile_template_id}")
    print(f"Snapshot: {result['snapshot_path']}")


if __name__ == "__main__":
    main()
