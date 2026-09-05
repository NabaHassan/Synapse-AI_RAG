#!/usr/bin/env python3
"""
Bulk session migration: JSON conversation files -> Redis session hot state.

This script is intended for one-time warm migration before enabling
multi-worker mode in Phase 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, ".")

from src.concurrency import RedisConnection, RedisRuntimeConfig, RedisSessionStore  # noqa: E402


def _load_session_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)
    if not isinstance(parsed, dict):
        raise ValueError("session file root is not an object")
    session = parsed.get("session")
    if not isinstance(session, dict):
        raise ValueError("missing 'session' object")
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("missing/invalid session.session_id")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate conversation JSON files to Redis session store")
    parser.add_argument("--conversations-dir", default="./data/conversations", help="Directory containing *.json session files")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0", help="Redis connection URL")
    parser.add_argument("--redis-key-prefix", default="synapse", help="Redis key prefix")
    parser.add_argument("--session-ttl-seconds", type=int, default=7 * 24 * 3600, help="Redis TTL for session state")
    parser.add_argument("--max-sessions", type=int, default=0, help="Optional cap on number of sessions to import")
    parser.add_argument("--dry-run", action="store_true", help="Validate and count only; do not write to Redis")
    args = parser.parse_args()

    conversations_dir = Path(args.conversations_dir)
    if not conversations_dir.exists():
        print(json.dumps({"error": "conversations_dir_not_found", "path": str(conversations_dir)}))
        return 2

    runtime = RedisRuntimeConfig(
        enabled=True,
        url=args.redis_url,
        key_prefix=args.redis_key_prefix,
    )
    connection = RedisConnection(runtime)
    if connection.init_error:
        print(json.dumps({"error": "redis_init_failed", "detail": connection.init_error}))
        return 2
    if not connection.ping():
        print(json.dumps({"error": "redis_ping_failed"}))
        return 2

    store = RedisSessionStore(connection=connection, session_ttl_seconds=args.session_ttl_seconds)

    files = sorted(conversations_dir.glob("*.json"))
    if args.max_sessions > 0:
        files = files[:args.max_sessions]

    imported = 0
    skipped = 0
    failed = 0

    for session_file in files:
        try:
            payload = _load_session_payload(session_file)
            session_obj = payload.get("session", {})
            session_id = session_obj.get("session_id")
            if args.dry_run:
                imported += 1
                continue
            ok = store.set_session_payload(str(session_id), payload)
            if ok:
                imported += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(
                json.dumps(
                    {
                        "error": "session_import_failed",
                        "file": str(session_file),
                        "detail": str(exc),
                    }
                ),
                file=sys.stderr,
            )

    summary = {
        "conversations_dir": str(conversations_dir),
        "redis_url": args.redis_url,
        "redis_key_prefix": args.redis_key_prefix,
        "dry_run": args.dry_run,
        "files_seen": len(files),
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
        "success": failed == 0,
    }
    print(json.dumps(summary, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

