"""KB Registry for multi-KB orchestration.

Stores metadata for each KB in a JSON file with atomic writes.
"""

import json
import logging
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KBMetadata:
    kb_id: str
    collection_name: str
    display_name: str
    created_at: str
    updated_at: str
    status: str = "active"  # active, indexing, error, deleted
    doc_count: int = 0
    description: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    config_path: Optional[str] = None
    profile_template_id: Optional[str] = None
    profile_template_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KBRegistry:
    """Manages KB metadata persistence."""

    def __init__(self, registry_path: str = "./data/kb_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: Dict[str, Dict[str, Any]] = self._load_registry()
        logger.info("KBRegistry initialized (entries: %s, path: %s)", len(self._data), self.registry_path)

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        try:
            with self.registry_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                logger.warning("KB registry malformed (expected dict). Resetting.")
                return {}
        except json.JSONDecodeError:
            logger.warning("KB registry corrupted. Resetting.")
            return {}
        except Exception as exc:
            logger.error("Failed to load KB registry: %s", exc, exc_info=True)
            return {}

    def _atomic_save(self, data: Dict[str, Dict[str, Any]]):
        tmp_path = self.registry_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.registry_path)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def register_kb(
        self,
        kb_id: str,
        display_name: str,
        collection_name: str,
        description: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        config_path: Optional[str] = None,
        profile_template_id: Optional[str] = None,
        profile_template_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if kb_id in self._data:
                raise ValueError(f"KB already exists: {kb_id}")

            now = self._now_iso()
            metadata = KBMetadata(
                kb_id=kb_id,
                collection_name=collection_name,
                display_name=display_name,
                created_at=now,
                updated_at=now,
                status="active",
                doc_count=0,
                description=description,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                config_path=config_path,
                profile_template_id=profile_template_id,
                profile_template_version=profile_template_version,
            )
            self._data[kb_id] = metadata.to_dict()
            self._atomic_save(self._data)
            return self._data[kb_id]

    def get_kb(self, kb_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get(kb_id)

    def list_kbs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._data.values())

    def update_kb(self, kb_id: str, **updates) -> Optional[Dict[str, Any]]:
        with self._lock:
            if kb_id not in self._data:
                return None

            current = self._data[kb_id]
            updates["updated_at"] = self._now_iso()
            current.update({k: v for k, v in updates.items() if v is not None})
            self._data[kb_id] = current
            self._atomic_save(self._data)
            return current

    def delete_kb(self, kb_id: str) -> bool:
        with self._lock:
            if kb_id not in self._data:
                return False
            del self._data[kb_id]
            self._atomic_save(self._data)
            return True

    def exists(self, kb_id: str) -> bool:
        with self._lock:
            return kb_id in self._data
