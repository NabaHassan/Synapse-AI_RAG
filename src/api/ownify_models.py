"""Pydantic models for Ownify AI provisioning endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator


class OwnifyDocumentInput(BaseModel):
    file_id: Optional[str] = Field(
        None,
        min_length=1,
        description="Stable document identifier from Ownify. If omitted, the AI service generates one.",
    )
    file_name: Optional[str] = Field(None, min_length=1, description="Original document filename")
    sas_url: Optional[str] = Field(None, min_length=1, description="Temporary URL used by the AI service for ingestion")
    local_path: Optional[str] = Field(
        None,
        min_length=1,
        description="Path to a document already present on the AI service filesystem.",
    )

    @model_validator(mode="after")
    def validate_source(self):
        if bool(self.sas_url) == bool(self.local_path):
            raise ValueError("Provide exactly one document source: sas_url or local_path.")
        return self


class OwnifyGenerationConfig(BaseModel):
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, ge=0.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    normalize_newlines: Optional[str] = None
    enable_min_tokens_strategy: Optional[bool] = None
    min_tokens_long_response: Optional[int] = Field(None, ge=0)
    long_response_max_tokens: Optional[int] = Field(None, ge=0)


class OwnifyRoutingConfig(BaseModel):
    enable_enhanced_handlers: Optional[bool] = None
    enable_clarification: Optional[bool] = None
    structured_query_fast_mode: Optional[Union[bool, str]] = None
    structured_entity_resolution: Optional[bool] = None
    structured_natural_response_style: Optional[bool] = None


class OwnifyGroundingConfig(BaseModel):
    allow_general_knowledge_fallback: Optional[bool] = None
    min_verification_threshold: Optional[float] = Field(None, ge=0.0)
    enable_collection_query_anchoring: Optional[bool] = None
    collection_anchor_terms: Optional[List[str]] = None


class OwnifyQueryHandlerConfig(BaseModel):
    canned_responses: Optional[Dict[str, str]] = None


class OwnifyAIConfigRequest(BaseModel):
    generation: Optional[OwnifyGenerationConfig] = None
    routing: Optional[OwnifyRoutingConfig] = None
    grounding: Optional[OwnifyGroundingConfig] = None
    query_handler: Optional[OwnifyQueryHandlerConfig] = None
    canned_responses: Optional[Dict[str, str]] = Field(
        None,
        description="Convenience alias for query_handler.canned_responses",
    )


class OwnifyProvisionRequest(BaseModel):
    display_name: str = Field(..., min_length=1)
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    ai_config: Optional[OwnifyAIConfigRequest] = None
    documents: List[OwnifyDocumentInput] = Field(
        default_factory=list,
        description="Deprecated. Provisioning rejects documents; use the documents job endpoint.",
    )
    idempotency_key: Optional[str] = None
    replace_existing: bool = False
    timeout_seconds: Optional[float] = Field(None, gt=0)


class OwnifyBatchDocumentsRequest(BaseModel):
    documents: List[OwnifyDocumentInput] = Field(..., min_items=1)
    idempotency_key: Optional[str] = None
    timeout_seconds: Optional[float] = Field(None, gt=0)


class OwnifyConfigUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    ai_config: Optional[OwnifyAIConfigRequest] = None


class OwnifyJobAcceptedResponse(BaseModel):
    status: str
    job_id: str
    tenant_id: str
    kb_id: str
    request_id: str
    job_status: str
    queued_at: str
    timeout_seconds: float


class OwnifyJobStatusResponse(BaseModel):
    job_id: str
    tenant_id: str
    kb_id: str
    job_type: str
    job_status: str
    phase: str
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    request_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    timeout_seconds: Optional[float] = None
    attempts: int = 0
    cancellation_requested: bool = False
    is_terminal: bool = False
    request_summary: Dict[str, Any] = Field(default_factory=dict)
    kb: Optional[Dict[str, Any]] = None
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class OwnifyDocumentsOperationResponse(BaseModel):
    status: str
    job_id: str
    tenant_id: str
    kb_id: str
    request_id: str
    job_status: str
    phase: str
    queued_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    timeout_seconds: float
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class OwnifyConfigUpdateResponse(BaseModel):
    status: str
    tenant_id: str
    kb_id: str
    kb: Dict[str, Any]
    snapshot_path: str
    config_version: str


class OwnifyDocumentDeleteResponse(BaseModel):
    status: str
    job_id: str
    tenant_id: str
    kb_id: str
    request_id: str
    job_status: str
    phase: str
    queued_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    timeout_seconds: float
    file_id: str
    file_name: Optional[str] = None
    deleted_count: int = 0
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None


class OwnifyTenantDeleteResponse(BaseModel):
    status: str
    job_id: str
    tenant_id: str
    kb_id: str
    request_id: str
    job_status: str
    phase: str
    queued_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    timeout_seconds: float
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None


class OwnifyAIStatusResponse(BaseModel):
    status: str
    tenant_id: str
    kb_id: str
    exists: bool
    kb: Optional[Dict[str, Any]] = None
    collection: Dict[str, Any] = Field(default_factory=dict)
    latest_job: Optional[Dict[str, Any]] = None
