"""YAML loader helpers for runtime policy, prompts, and per-KB profiles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .schemas import RuntimePolicy, ProfileTemplate


try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


def _require_yaml() -> None:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for config loading. Install with: pip install pyyaml"
        ) from _YAML_IMPORT_ERROR


def load_yaml_file(path: Path) -> Dict[str, Any]:
    _require_yaml()
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return data or {}


def resolve_config_dir(config_dir: Optional[Path] = None) -> Path:
    """Resolve the config directory from explicit path, env, or repository default."""
    if config_dir is not None:
        return Path(config_dir)

    env_value = os.getenv("CONFIG_DIR")
    if env_value:
        return Path(env_value)

    return Path(__file__).resolve().parent


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_int(raw: str) -> Any:
    try:
        return int(raw)
    except ValueError:
        return raw


def _maybe_float(raw: str) -> Any:
    try:
        return float(raw)
    except ValueError:
        return raw


def _set_path_value(data: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cursor = data
    for key in path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path[-1]] = value


def apply_runtime_env_overrides(data: Dict[str, Any], env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    env = env or os.environ

    mapping = {
        "QUERY_EXECUTOR_WORKERS": ("executors", "query_workers", int),
        "INDEX_EXECUTOR_WORKERS": ("executors", "index_workers", int),
        "QUERY_MAX_INFLIGHT": ("admission_control", "query_max_inflight", int),
        "QUERY_ADMISSION_TIMEOUT_SECONDS": ("admission_control", "query_admission_timeout_seconds", float),
        "QUERY_RETRY_AFTER_SECONDS": ("admission_control", "query_retry_after_seconds", int),
        "SESSION_LOCK_TIMEOUT_SECONDS": ("admission_control", "session_lock_timeout_seconds", float),
        "SESSION_LOCK_RETRY_AFTER_SECONDS": ("admission_control", "session_lock_retry_after_seconds", int),
        "QUERY_READ_GUARD_TIMEOUT_SECONDS": ("admission_control", "query_read_guard_timeout_seconds", float),
        "ADMIN_WRITE_TIMEOUT_SECONDS": ("admission_control", "admin_write_timeout_seconds", float),
        "INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS": ("admission_control", "indexing_write_drain_timeout_seconds", float),
        "INDEXING_RETRY_AFTER_SECONDS": ("admission_control", "indexing_retry_after_seconds", int),
        "QUERY_TIMEOUT_SECONDS": ("admission_control", "query_timeout_seconds", float),
        "LLM_MAX_CONCURRENCY": ("llm_concurrency", "max_concurrency", int),
        "LLM_ADMISSION_TIMEOUT_SECONDS": ("llm_concurrency", "admission_timeout_seconds", float),
        "LLM_RETRY_AFTER_SECONDS": ("llm_concurrency", "retry_after_seconds", int),
        "ASYNC_QUERY_JOBS_ENABLED": ("async_jobs", "enabled", _parse_bool),
        "ASYNC_QUERY_JOB_WORKERS": ("async_jobs", "workers", int),
        "ASYNC_QUERY_JOB_QUEUE_MAX_SIZE": ("async_jobs", "queue_max_size", int),
        "ASYNC_QUERY_JOB_DEFAULT_TIMEOUT_SECONDS": ("async_jobs", "default_timeout_seconds", float),
        "ASYNC_QUERY_JOB_MAX_TIMEOUT_SECONDS": ("async_jobs", "max_timeout_seconds", float),
        "ASYNC_QUERY_JOB_RETENTION_SECONDS": ("async_jobs", "retention_seconds", int),
        "ASYNC_QUERY_JOB_CLEANUP_INTERVAL_SECONDS": ("async_jobs", "cleanup_interval_seconds", float),
        "ASYNC_QUERY_JOB_MIN_RETRY_DELAY_SECONDS": ("async_jobs", "min_retry_delay_seconds", float),
        "ASYNC_QUERY_JOB_MAX_RETRY_DELAY_SECONDS": ("async_jobs", "max_retry_delay_seconds", float),
        "OWNIFY_PROVISIONING_JOBS_ENABLED": ("provisioning_jobs", "enabled", _parse_bool),
        "OWNIFY_PROVISIONING_JOB_WORKERS": ("provisioning_jobs", "workers", int),
        "OWNIFY_PROVISIONING_JOB_QUEUE_MAX_SIZE": ("provisioning_jobs", "queue_max_size", int),
        "OWNIFY_PROVISIONING_JOB_DEFAULT_TIMEOUT_SECONDS": ("provisioning_jobs", "default_timeout_seconds", float),
        "OWNIFY_PROVISIONING_JOB_MAX_TIMEOUT_SECONDS": ("provisioning_jobs", "max_timeout_seconds", float),
        "OWNIFY_PROVISIONING_JOB_RETENTION_SECONDS": ("provisioning_jobs", "retention_seconds", int),
        "OWNIFY_PROVISIONING_JOB_CLEANUP_INTERVAL_SECONDS": ("provisioning_jobs", "cleanup_interval_seconds", float),
        "OWNIFY_PROVISIONING_JOB_MIN_RETRY_DELAY_SECONDS": ("provisioning_jobs", "min_retry_delay_seconds", float),
        "OWNIFY_PROVISIONING_JOB_MAX_RETRY_DELAY_SECONDS": ("provisioning_jobs", "max_retry_delay_seconds", float),
        "INDEXING_MODE": ("indexing", "mode", str),
        "INDEXING_DENSE_VISIBILITY_FILTER_ENABLED": ("indexing", "dense_visibility_filter_enabled", _parse_bool),
        "INDEXING_ONLINE_MIN_TEXT_LENGTH": ("indexing", "online_min_text_length", int),
        "INDEXING_ONLINE_MIN_QUALITY_SCORE": ("indexing", "online_min_quality_score", float),
        "INDEXING_ONLINE_REMOVE_URLS": ("indexing", "online_remove_urls", _parse_bool),
        "INDEXING_ONLINE_REMOVE_EMAILS": ("indexing", "online_remove_emails", _parse_bool),
        "INDEXING_ONLINE_USE_UNSTRUCTURED_FALLBACK": ("indexing", "online_use_unstructured_fallback", _parse_bool),
        "BACKFILL_INGEST_STATE_ON_STARTUP": ("indexing", "backfill_ingest_state_on_startup", _parse_bool),
        "INDEXING_BACKFILL_ON_STARTUP": ("indexing", "backfill_ingest_state_on_startup", _parse_bool),
        "REDIS_ENABLED": ("redis", "enabled", _parse_bool),
        "REDIS_URL": ("redis", "url", str),
        "REDIS_KEY_PREFIX": ("redis", "key_prefix", str),
        "REDIS_SOCKET_TIMEOUT_SECONDS": ("redis", "socket_timeout_seconds", float),
        "REDIS_CONNECT_TIMEOUT_SECONDS": ("redis", "connect_timeout_seconds", float),
        "REDIS_LOCK_ENABLED": ("redis", "lock_enabled", _parse_bool),
        "REDIS_LOCK_TTL_SECONDS": ("redis", "lock_ttl_seconds", float),
        "REDIS_LOCK_RETRY_INTERVAL_SECONDS": ("redis", "lock_retry_interval_seconds", float),
        "REDIS_RATE_LIMIT_ENABLED": ("redis", "rate_limit_enabled", _parse_bool),
        "REDIS_RATE_COUNTER_TTL_SECONDS": ("redis", "rate_counter_ttl_seconds", int),
        "RATE_LIMIT_RETRY_AFTER_SECONDS": ("redis", "rate_limit_retry_after_seconds", int),
        "SESSION_MAX_INFLIGHT_REDIS": ("redis", "session_max_inflight", int),
        "USER_MAX_INFLIGHT_REDIS": ("redis", "user_max_inflight", int),
        "REDIS_SESSION_STORE_ENABLED": ("redis", "session_store_enabled", _parse_bool),
        "REDIS_SESSION_TTL_SECONDS": ("redis", "session_ttl_seconds", int),
        "REDIS_SESSION_READ_THROUGH_ENABLED": ("redis", "session_read_through_enabled", _parse_bool),
        "REDIS_QUERY_CACHE_ENABLED": ("redis", "query_cache_enabled", _parse_bool),
        "REDIS_QUERY_CACHE_TTL_SECONDS": ("redis", "query_cache_ttl_seconds", int),
        "LLM_STUCK_WATCHDOG_ENABLED": ("watchdog", "enabled", _parse_bool),
        "LLM_STUCK_THRESHOLD_SECONDS": ("watchdog", "stuck_threshold_seconds", float),
        "LLM_STUCK_CHECK_INTERVAL_SECONDS": ("watchdog", "check_interval_seconds", float),
    }

    updated = dict(data)
    for env_key, (section, field, caster) in mapping.items():
        raw = env.get(env_key)
        if raw is None:
            continue
        value = caster(raw)
        _set_path_value(updated, (section, field), value)

    return updated


def load_runtime_policy(runtime_path: Path, env: Optional[Dict[str, str]] = None) -> RuntimePolicy:
    raw = load_yaml_file(runtime_path)
    raw = apply_runtime_env_overrides(raw, env=env)
    return RuntimePolicy(**raw)


def load_defaults(config_dir: Path) -> Dict[str, Any]:
    defaults_path = config_dir / "defaults.yaml"
    return load_yaml_file(defaults_path)


def load_profile_template(profile_path: Path, defaults: Optional[Dict[str, Any]] = None) -> ProfileTemplate:
    raw = load_yaml_file(profile_path)
    if defaults:
        raw = deep_merge_dicts(defaults, raw)
    return ProfileTemplate(**raw)


def load_prompt_catalog(config_dir: Optional[Path] = None) -> Dict[str, str]:
    resolved_config_dir = resolve_config_dir(config_dir)
    raw = load_yaml_file(resolved_config_dir / "prompt_catalog.yaml")
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def resolve_prompt_path(prompt_ref: str, config_dir: Optional[Path] = None) -> Path:
    resolved_config_dir = resolve_config_dir(config_dir)
    candidate = Path(prompt_ref)
    if candidate.is_absolute():
        return candidate
    return resolved_config_dir / candidate


def load_prompt_text(
    *,
    template: Optional[str] = None,
    template_name: Optional[str] = None,
    config_dir: Optional[Path] = None,
) -> Tuple[str, str]:
    """Load prompt text from config path, named catalog entry, or inline text."""
    resolved_config_dir = resolve_config_dir(config_dir)

    if template:
        candidate = resolve_prompt_path(template, resolved_config_dir)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8"), str(candidate)
        return template, "inline"

    if template_name:
        catalog = load_prompt_catalog(resolved_config_dir)
        prompt_ref = catalog.get(template_name)
        if prompt_ref:
            prompt_path = resolve_prompt_path(prompt_ref, resolved_config_dir)
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8"), str(prompt_path)
        raise ValueError(f"Prompt template '{template_name}' not found in prompt catalog")

    catalog = load_prompt_catalog(resolved_config_dir)
    default_ref = catalog.get("default")
    if not default_ref:
        raise ValueError("Prompt catalog does not define a 'default' prompt")
    default_path = resolve_prompt_path(default_ref, resolved_config_dir)
    return default_path.read_text(encoding="utf-8"), str(default_path)
