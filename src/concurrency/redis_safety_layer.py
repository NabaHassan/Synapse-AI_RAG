"""
Redis safety-layer primitives for Phase 1 concurrency hardening.

This module centralizes Redis-backed primitives used by the API server and
pipeline:
- Distributed same-session locks
- Shared session hot-state storage
- Shared in-flight counters for per-session/per-user limits
- Shared query-cache storage
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - import failure path
    redis = None


class RedisBackendUnavailable(RuntimeError):
    """Raised when Redis is required but unavailable."""


@dataclass(frozen=True)
class RedisRuntimeConfig:
    """Redis runtime settings shared by safety-layer components."""

    enabled: bool = False
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "synapse"
    socket_timeout_seconds: float = 0.2
    connect_timeout_seconds: float = 0.2

    def key(self, raw_key: str) -> str:
        clean_prefix = self.key_prefix.strip(":")
        return f"{clean_prefix}:{raw_key}" if clean_prefix else raw_key


class RedisConnection:
    """
    Thin Redis connection wrapper with lazy client creation and guarded failures.
    """

    def __init__(self, config: RedisRuntimeConfig):
        self.config = config
        self._client: Any = None
        self._init_error: Optional[str] = None
        if self.config.enabled:
            self._initialize_client()

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def _initialize_client(self) -> None:
        if not self.config.enabled:
            self._client = None
            return
        if redis is None:
            self._init_error = "redis_python_package_missing"
            self._client = None
            return
        try:
            self._client = redis.Redis.from_url(
                self.config.url,
                decode_responses=True,
                socket_timeout=self.config.socket_timeout_seconds,
                socket_connect_timeout=self.config.connect_timeout_seconds,
            )
            # Validate early so readiness reflects misconfiguration quickly.
            self._client.ping()
            self._init_error = None
        except Exception as exc:
            self._client = None
            self._init_error = str(exc)

    def client(self) -> Any:
        if not self.config.enabled:
            raise RedisBackendUnavailable("Redis disabled")
        if self._client is None and self._init_error is None:
            self._initialize_client()
        if self._client is None:
            raise RedisBackendUnavailable(self._init_error or "Redis client unavailable")
        return self._client

    def ping(self) -> bool:
        if not self.config.enabled:
            return False
        try:
            return bool(self.client().ping())
        except Exception:
            return False


class DistributedSessionLockBackend:
    """
    Redis distributed lock keyed by session ID.

    Key format:
      session:{session_id}:lock
    """

    _RELEASE_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("DEL", KEYS[1])
else
  return 0
end
"""

    def __init__(
        self,
        connection: RedisConnection,
        lock_ttl_seconds: float = 300.0,
        retry_sleep_seconds: float = 0.05,
    ):
        self.connection = connection
        self.lock_ttl_seconds = max(1.0, float(lock_ttl_seconds))
        self.retry_sleep_seconds = max(0.01, float(retry_sleep_seconds))

    def _lock_key(self, session_id: str) -> str:
        return self.connection.config.key(f"session:{session_id}:lock")

    def acquire(self, session_id: str, owner_token: str, wait_timeout_seconds: float) -> bool:
        if not session_id:
            raise ValueError("session_id is required")
        if not owner_token:
            raise ValueError("owner_token is required")

        deadline = time.monotonic() + max(0.0, float(wait_timeout_seconds))
        ttl_ms = int(self.lock_ttl_seconds * 1000)
        key = self._lock_key(session_id)

        while True:
            try:
                acquired = bool(self.connection.client().set(key, owner_token, nx=True, px=ttl_ms))
            except Exception as exc:
                raise RedisBackendUnavailable(f"Redis lock acquire failed: {exc}") from exc

            if acquired:
                return True

            if time.monotonic() >= deadline:
                return False

            time.sleep(self.retry_sleep_seconds)

    def release(self, session_id: str, owner_token: str) -> bool:
        if not session_id or not owner_token:
            return False
        key = self._lock_key(session_id)
        try:
            result = self.connection.client().eval(self._RELEASE_SCRIPT, 1, key, owner_token)
            return bool(result)
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis lock release failed: {exc}") from exc


class RedisSessionStore:
    """
    Redis-backed session hot-state store.

    Stored value is a JSON payload compatible with ConversationMemory persistence
    shape: {"version": "...", "saved_at": "...", "config": {...}, "session": {...}}
    """

    def __init__(
        self,
        connection: RedisConnection,
        session_ttl_seconds: int = 7 * 24 * 3600,
        session_namespace: Optional[str] = None,
    ):
        self.connection = connection
        self.session_ttl_seconds = max(0, int(session_ttl_seconds))
        self.session_namespace = self._normalize_namespace(session_namespace)
        self._index_key = self.connection.config.key(self._index_raw_key())

    @staticmethod
    def _normalize_namespace(session_namespace: Optional[str]) -> Optional[str]:
        if session_namespace is None:
            return None
        cleaned = str(session_namespace).strip()
        return cleaned or None

    def _index_raw_key(self) -> str:
        if self.session_namespace:
            return f"sessions:index:{self.session_namespace}"
        return "sessions:index"

    def _session_key(self, session_id: str) -> str:
        if self.session_namespace:
            raw_key = f"session:{self.session_namespace}:{session_id}:state"
        else:
            raw_key = f"session:{session_id}:state"
        return self.connection.config.key(raw_key)

    def get_session_payload(self, session_id: str) -> Optional[Dict[str, Any]]:
        key = self._session_key(session_id)
        try:
            raw = self.connection.client().get(key)
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis session read failed: {exc}") from exc

        if raw is None:
            return None

        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            logger.warning("Invalid Redis session payload for %s; treating as missing", session_id)
            return None

    def set_session_payload(self, session_id: str, payload: Dict[str, Any]) -> bool:
        key = self._session_key(session_id)
        try:
            serialized = json.dumps(payload, ensure_ascii=False)
            client = self.connection.client()
            pipeline = client.pipeline()
            if self.session_ttl_seconds > 0:
                pipeline.set(key, serialized, ex=self.session_ttl_seconds)
            else:
                pipeline.set(key, serialized)
            pipeline.sadd(self._index_key, session_id)
            pipeline.execute()
            return True
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis session write failed: {exc}") from exc

    def delete_session(self, session_id: str) -> bool:
        key = self._session_key(session_id)
        try:
            client = self.connection.client()
            pipeline = client.pipeline()
            pipeline.delete(key)
            pipeline.srem(self._index_key, session_id)
            pipeline.execute()
            return True
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis session delete failed: {exc}") from exc

    def list_session_ids(self) -> List[str]:
        try:
            raw = self.connection.client().smembers(self._index_key)
            if not raw:
                return []
            return sorted(str(item) for item in raw)
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis session list failed: {exc}") from exc


@dataclass
class RateLimitDecision:
    allowed: bool
    reason: Optional[str] = None
    retry_after_seconds: int = 2
    session_inflight: Optional[int] = None
    user_inflight: Optional[int] = None
    backend_available: bool = True


class RedisInFlightRateLimiter:
    """
    Redis-backed in-flight counters for per-session and per-user caps.

    Counter keys:
      limiter:session:{session_id}:inflight
      limiter:user:{user_id}:inflight
    """

    def __init__(
        self,
        connection: RedisConnection,
        counter_ttl_seconds: int = 120,
    ):
        self.connection = connection
        self.counter_ttl_seconds = max(10, int(counter_ttl_seconds))

    def _session_key(self, session_id: str) -> str:
        return self.connection.config.key(f"limiter:session:{session_id}:inflight")

    def _user_key(self, user_id: str) -> str:
        return self.connection.config.key(f"limiter:user:{user_id}:inflight")

    def acquire(
        self,
        *,
        session_id: Optional[str],
        user_id: Optional[str],
        max_inflight_per_session: int,
        max_inflight_per_user: int,
        retry_after_seconds: int,
    ) -> RateLimitDecision:
        if not session_id and not user_id:
            return RateLimitDecision(allowed=True)

        session_key = self._session_key(session_id) if session_id else None
        user_key = self._user_key(user_id) if user_id else None

        try:
            client = self.connection.client()
        except RedisBackendUnavailable:
            # Fail-open for rate limits to preserve availability.
            return RateLimitDecision(allowed=True, backend_available=False)

        session_count: Optional[int] = None
        user_count: Optional[int] = None

        try:
            if session_key and max_inflight_per_session > 0:
                session_count = int(client.incr(session_key))
                client.expire(session_key, self.counter_ttl_seconds)
                if session_count > max_inflight_per_session:
                    client.decr(session_key)
                    return RateLimitDecision(
                        allowed=False,
                        reason="session_rate_limited",
                        retry_after_seconds=max(1, int(retry_after_seconds)),
                        session_inflight=session_count,
                    )

            if user_key and max_inflight_per_user > 0:
                user_count = int(client.incr(user_key))
                client.expire(user_key, self.counter_ttl_seconds)
                if user_count > max_inflight_per_user:
                    client.decr(user_key)
                    if session_key and session_count is not None:
                        client.decr(session_key)
                    return RateLimitDecision(
                        allowed=False,
                        reason="user_rate_limited",
                        retry_after_seconds=max(1, int(retry_after_seconds)),
                        session_inflight=session_count,
                        user_inflight=user_count,
                    )

            return RateLimitDecision(
                allowed=True,
                session_inflight=session_count,
                user_inflight=user_count,
            )
        except Exception:
            # Fail-open on transient backend errors.
            logger.warning("Redis rate limiter failed; allowing request", exc_info=True)
            return RateLimitDecision(allowed=True, backend_available=False)

    def release(self, *, session_id: Optional[str], user_id: Optional[str]) -> None:
        if not session_id and not user_id:
            return

        try:
            client = self.connection.client()
        except RedisBackendUnavailable:
            return

        def _release_key(key: Optional[str]) -> None:
            if not key:
                return
            try:
                value = client.decr(key)
                if int(value) <= 0:
                    client.delete(key)
            except Exception:
                logger.warning("Redis rate limiter key release failed: %s", key, exc_info=True)

        _release_key(self._session_key(session_id) if session_id else None)
        _release_key(self._user_key(user_id) if user_id else None)


class RedisQueryCacheStore:
    """
    Redis-backed storage for query cache entries.

    Uses:
      cache:entry:{normalized_key} -> JSON payload
      cache:index -> ZSET(normalized_key, last_access_epoch)
    """

    def __init__(
        self,
        connection: RedisConnection,
        default_ttl_seconds: int,
        max_entries: int,
    ):
        self.connection = connection
        self.default_ttl_seconds = max(60, int(default_ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._index_key = self.connection.config.key("cache:index")

    def _entry_key(self, normalized_key: str) -> str:
        return self.connection.config.key(f"cache:entry:{normalized_key}")

    def get_entry(self, normalized_key: str) -> Optional[Dict[str, Any]]:
        key = self._entry_key(normalized_key)
        try:
            raw = self.connection.client().get(key)
            if raw is None:
                return None
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis cache read failed: {exc}") from exc

    def set_entry(
        self,
        normalized_key: str,
        payload: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        ttl = self.default_ttl_seconds if ttl_seconds is None else max(60, int(ttl_seconds))
        key = self._entry_key(normalized_key)
        score = float(time.time())
        try:
            client = self.connection.client()
            pipeline = client.pipeline()
            pipeline.set(key, json.dumps(payload, ensure_ascii=False), ex=ttl)
            pipeline.zadd(self._index_key, {normalized_key: score})
            pipeline.execute()
            self._evict_if_needed(client)
            return True
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis cache write failed: {exc}") from exc

    def delete_entry(self, normalized_key: str) -> None:
        key = self._entry_key(normalized_key)
        try:
            client = self.connection.client()
            pipeline = client.pipeline()
            pipeline.delete(key)
            pipeline.zrem(self._index_key, normalized_key)
            pipeline.execute()
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis cache delete failed: {exc}") from exc

    def touch(self, normalized_key: str) -> None:
        try:
            self.connection.client().zadd(self._index_key, {normalized_key: float(time.time())})
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis cache touch failed: {exc}") from exc

    def clear(self) -> None:
        try:
            client = self.connection.client()
            keys = client.zrange(self._index_key, 0, -1)
            pipeline = client.pipeline()
            for normalized_key in keys:
                pipeline.delete(self._entry_key(str(normalized_key)))
            pipeline.delete(self._index_key)
            pipeline.execute()
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis cache clear failed: {exc}") from exc

    def count(self) -> int:
        try:
            return int(self.connection.client().zcard(self._index_key))
        except Exception as exc:
            raise RedisBackendUnavailable(f"Redis cache count failed: {exc}") from exc

    def _evict_if_needed(self, client: Any) -> None:
        try:
            total = int(client.zcard(self._index_key))
            overflow = total - self.max_entries
            if overflow <= 0:
                return
            oldest = client.zrange(self._index_key, 0, overflow - 1)
            if not oldest:
                return
            pipeline = client.pipeline()
            for normalized_key in oldest:
                normalized_key = str(normalized_key)
                pipeline.delete(self._entry_key(normalized_key))
                pipeline.zrem(self._index_key, normalized_key)
            pipeline.execute()
        except Exception:
            logger.warning("Redis cache eviction failed", exc_info=True)
