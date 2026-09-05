"""Durable job store for Ownify AI provisioning and ingestion jobs."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProvisioningJobStore:
    """Persists provisioning job records as one JSON file per job."""

    def __init__(self, jobs_dir: str):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = self._load_jobs()
        logger.info(
            "ProvisioningJobStore initialized (entries=%s, path=%s)",
            len(self._jobs),
            self.jobs_dir,
        )

    def _load_jobs(self) -> Dict[str, Dict[str, Any]]:
        jobs: Dict[str, Dict[str, Any]] = {}
        for path in self.jobs_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    record = json.load(handle)
                job_id = str(record.get("job_id") or path.stem)
                if job_id:
                    jobs[job_id] = record
            except Exception:
                logger.warning("Failed to load provisioning job record: %s", path, exc_info=True)
        return jobs

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _atomic_save(self, record: Dict[str, Any]) -> None:
        job_id = str(record["job_id"])
        path = self._job_path(job_id)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False, sort_keys=True)
        tmp_path.replace(path)

    def create(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            job_id = str(record["job_id"])
            if job_id in self._jobs:
                raise ValueError(f"Provisioning job already exists: {job_id}")
            stored = dict(record)
            self._jobs[job_id] = stored
            self._atomic_save(stored)
            return dict(stored)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record is not None else None

    def update(self, job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return None
            updated = dict(current)
            for key, value in updates.items():
                updated[key] = value
            self._jobs[job_id] = updated
            self._atomic_save(updated)
            return dict(updated)

    def find_by_idempotency_key(
        self,
        *,
        tenant_id: str,
        idempotency_key: Optional[str],
        job_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not idempotency_key:
            return None
        with self._lock:
            matches = [
                job
                for job in self._jobs.values()
                if job.get("tenant_id") == tenant_id
                and job.get("idempotency_key") == idempotency_key
                and (job_type is None or job.get("job_type") == job_type)
            ]
            if not matches:
                return None
            matches.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
            return dict(matches[0])

    def latest_for_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            matches = [job for job in self._jobs.values() if job.get("tenant_id") == tenant_id]
            if not matches:
                return None
            matches.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
            return dict(matches[0])

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(job) for job in self._jobs.values()]

    def terminal_older_than(self, terminal_states: set[str], cutoff: float) -> List[str]:
        with self._lock:
            return [
                job_id
                for job_id, job in self._jobs.items()
                if job.get("job_status") in terminal_states
                and float(job.get("updated_at") or 0) < cutoff
            ]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            existed = self._jobs.pop(job_id, None) is not None
            if existed:
                try:
                    self._job_path(job_id).unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to delete provisioning job record: %s", job_id, exc_info=True)
            return existed

    def delete_for_tenant(self, tenant_id: str, exclude_job_id: Optional[str] = None) -> int:
        with self._lock:
            job_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.get("tenant_id") == tenant_id and job_id != exclude_job_id
            ]
            deleted = 0
            for job_id in job_ids:
                if self.delete(job_id):
                    deleted += 1
            return deleted
