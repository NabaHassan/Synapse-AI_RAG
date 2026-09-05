"""Pydantic schemas for runtime and per-KB configuration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# =============================================================================
# Runtime Policy (global)
# =============================================================================


class RuntimeExecutors(BaseModel):
    query_workers: int = 4
    index_workers: int = 2


class RuntimeAdmissionControl(BaseModel):
    query_max_inflight: Union[int, str] = "auto"
    query_admission_timeout_seconds: float = 1.0
    query_retry_after_seconds: int = 3
    session_lock_timeout_seconds: float = 2.0
    session_lock_retry_after_seconds: int = 2
    query_read_guard_timeout_seconds: float = 0.25
    admin_write_timeout_seconds: float = 5.0
    indexing_write_drain_timeout_seconds: float = 120.0
    indexing_retry_after_seconds: int = 5
    query_timeout_seconds: Optional[float] = None


class RuntimeLLMConcurrency(BaseModel):
    max_concurrency: int = 2
    admission_timeout_seconds: float = 6.0
    retry_after_seconds: int = 2


class RuntimeAsyncJobs(BaseModel):
    enabled: bool = True
    workers: int = 1
    queue_max_size: int = 256
    default_timeout_seconds: float = 300.0
    max_timeout_seconds: float = 900.0
    retention_seconds: int = 3600
    cleanup_interval_seconds: float = 30.0
    min_retry_delay_seconds: float = 0.1
    max_retry_delay_seconds: float = 3.0


class RuntimeProvisioningJobs(BaseModel):
    enabled: bool = True
    workers: int = 1
    queue_max_size: int = 128
    default_timeout_seconds: float = 900.0
    max_timeout_seconds: float = 3600.0
    retention_seconds: int = 86400
    cleanup_interval_seconds: float = 60.0
    min_retry_delay_seconds: float = 0.5
    max_retry_delay_seconds: float = 5.0


class RuntimeIndexing(BaseModel):
    mode: str = "online"
    dense_visibility_filter_enabled: bool = True
    write_drain_timeout_seconds: float = 120.0
    retry_after_seconds: int = 5
    backfill_ingest_state_on_startup: bool = True
    online_min_text_length: int = 200
    online_min_quality_score: float = 0.4
    online_remove_urls: bool = False
    online_remove_emails: bool = False
    online_use_unstructured_fallback: bool = True


class RuntimeRedis(BaseModel):
    enabled: bool = True
    url: str = "redis://localhost:6379/0"
    key_prefix: str = "synapse"
    socket_timeout_seconds: float = 0.2
    connect_timeout_seconds: float = 0.2
    lock_enabled: bool = True
    lock_ttl_seconds: float = 300.0
    lock_retry_interval_seconds: float = 0.05
    rate_limit_enabled: bool = True
    rate_counter_ttl_seconds: int = 360
    rate_limit_retry_after_seconds: int = 2
    session_max_inflight: int = 1
    user_max_inflight: int = 4
    session_store_enabled: bool = True
    session_ttl_seconds: int = 7 * 24 * 3600
    session_read_through_enabled: bool = True
    query_cache_enabled: bool = True
    query_cache_ttl_seconds: int = 0


class RuntimeWatchdog(BaseModel):
    enabled: bool = True
    stuck_threshold_seconds: float = 240.0
    check_interval_seconds: float = 5.0


class RuntimePolicy(BaseModel):
    schema_version: int = 1
    executors: RuntimeExecutors = Field(default_factory=RuntimeExecutors)
    admission_control: RuntimeAdmissionControl = Field(default_factory=RuntimeAdmissionControl)
    llm_concurrency: RuntimeLLMConcurrency = Field(default_factory=RuntimeLLMConcurrency)
    async_jobs: RuntimeAsyncJobs = Field(default_factory=RuntimeAsyncJobs)
    provisioning_jobs: RuntimeProvisioningJobs = Field(default_factory=RuntimeProvisioningJobs)
    indexing: RuntimeIndexing = Field(default_factory=RuntimeIndexing)
    redis: RuntimeRedis = Field(default_factory=RuntimeRedis)
    watchdog: RuntimeWatchdog = Field(default_factory=RuntimeWatchdog)


# =============================================================================
# Profile Templates (per-KB behavior)
# =============================================================================


class PromptPolicy(BaseModel):
    template: Optional[str] = None
    template_name: Optional[str] = None


class GenerationPolicy(BaseModel):
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    normalize_newlines: Optional[str] = None
    enable_min_tokens_strategy: bool = False
    min_tokens_long_response: int = 0
    long_response_max_tokens: int = 0


class RoutingPolicy(BaseModel):
    enable_enhanced_handlers: bool = True
    enable_clarification: bool = True
    structured_query_fast_mode: Optional[Union[bool, str]] = None
    structured_entity_resolution: bool = True
    structured_natural_response_style: bool = True


class GroundingPolicy(BaseModel):
    allow_general_knowledge_fallback: bool = True
    min_verification_threshold: float = 0.1
    enable_collection_query_anchoring: bool = True
    collection_anchor_terms: List[str] = Field(default_factory=list)


class QueryHandlerPolicy(BaseModel):
    canned_responses: Dict[str, str] = Field(default_factory=dict)


class ProfileTemplate(BaseModel):
    schema_version: int = 1
    profile_template_id: str = "default"
    profile_template_version: str = ""
    prompt: PromptPolicy = Field(default_factory=PromptPolicy)
    generation: GenerationPolicy = Field(default_factory=GenerationPolicy)
    routing: RoutingPolicy = Field(default_factory=RoutingPolicy)
    grounding: GroundingPolicy = Field(default_factory=GroundingPolicy)
    query_handler: QueryHandlerPolicy = Field(default_factory=QueryHandlerPolicy)


class KBConfigSnapshot(BaseModel):
    schema_version: int = 1
    kb_id: str
    collection_name: str
    profile_template_id: str
    profile_template_version: str
    resolved_at: Optional[str] = None
    prompt: PromptPolicy = Field(default_factory=PromptPolicy)
    generation: GenerationPolicy = Field(default_factory=GenerationPolicy)
    routing: RoutingPolicy = Field(default_factory=RoutingPolicy)
    grounding: GroundingPolicy = Field(default_factory=GroundingPolicy)
    query_handler: QueryHandlerPolicy = Field(default_factory=QueryHandlerPolicy)
    resolved_structured_query_fast_mode: Optional[bool] = None
    notes: Dict[str, Any] = Field(default_factory=dict)
