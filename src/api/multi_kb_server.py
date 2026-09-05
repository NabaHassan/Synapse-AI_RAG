"""Multi-KB RAG API Server.

Provides KB management, document management, and per-KB conversational queries.
"""

import asyncio
import functools
import hashlib
import hmac
import httpx
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Query, Depends, UploadFile, File, Form, BackgroundTasks
from slack_sdk.web.async_client import AsyncWebClient
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from src.pipeline.session_doc_store import (
    store_session_docs,
    delete_session_docs,
    get_session_doc_info,
    MAX_FILE_BYTES,
    SESSION_TTL_SECONDS,
)
from src.indexing.document_loader import load_document
from src.indexing.recursive_chunker import RecursiveCharacterTextSplitter

from src.kb_management import KBRegistry, KBManager, ProvisioningJobStore
from src.kb_management.kb_manager import KBManagerConfig
from src.api.ownify_models import (
    OwnifyBatchDocumentsRequest,
    OwnifyConfigUpdateRequest,
    OwnifyConfigUpdateResponse,
    OwnifyDocumentDeleteResponse,
    OwnifyDocumentsOperationResponse,
    OwnifyAIStatusResponse,
    OwnifyJobAcceptedResponse,
    OwnifyJobStatusResponse,
    OwnifyProvisionRequest,
    OwnifyTenantDeleteResponse,
)
from src.pipeline.multi_kb_pipeline import MultiKBPipeline, MultiKBSettings, SharedResources
from src.pipeline.conversational_rag_pipeline import (
    PipelineCancelledError,
    PipelineOverloadedError,
)
from src.config import RuntimePolicy, load_runtime_policy
from src.utils.source_normalization import (
    normalize_citations_sources as _normalize_citations_sources,
    normalize_result_citations,
)
from src.concurrency import (
    DistributedSessionLockBackend,
    RateLimitDecision,
    RedisBackendUnavailable,
    RedisConnection,
    RedisInFlightRateLimiter,
    RedisRuntimeConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class CreateKBRequest(BaseModel):
    kb_id: Optional[str] = Field(None, description="Optional KB identifier (server will generate if omitted)")
    display_name: str = Field(..., description="Human-friendly name")
    description: Optional[str] = Field(None, description="Optional description")
    existing_collection: Optional[str] = Field(None, description="Existing Qdrant collection name")


class AddDocumentRequest(BaseModel):
    file_id: Optional[str] = Field(
        None,
        description="Unique file identifier from backend. If omitted, the server generates a stable source-based ID.",
    )
    file_name: Optional[str] = Field(
        None,
        description="Original filename. Required for SAS URLs when the filename cannot be inferred from the URL.",
    )
    sas_url: Optional[str] = Field(None, description="SAS URL for document download")
    local_path: Optional[str] = Field(
        None,
        description="Path to a document already present on the server filesystem. file:// URLs are also accepted.",
    )

    @model_validator(mode="after")
    def validate_source(self):
        if bool(self.sas_url) == bool(self.local_path):
            raise ValueError("Provide exactly one document source: sas_url or local_path.")
        return self


class BatchAddDocumentsRequest(BaseModel):
    documents: Optional[List[AddDocumentRequest]] = Field(
        None,
        description="Optional explicit documents to index. Each item accepts sas_url or local_path.",
    )
    directory_path: Optional[str] = Field(
        None,
        description="Optional server-local directory to scan and index into this KB.",
    )
    recursive: bool = Field(True, description="Recursively scan directory_path")
    fail_fast: bool = Field(False, description="Stop the batch after the first indexing failure")

    @model_validator(mode="after")
    def validate_batch_source(self):
        if not self.directory_path and not self.documents:
            raise ValueError("Provide directory_path, documents, or both.")
        return self


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="Optional user identifier for rate limiting")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    stream: Optional[bool] = Field(False, description="Enable streaming response")
    web: Optional[str] = Field("off", description="Enable duckduckgo web search (on/off)")
    kb: Optional[str] = Field("on", description="Enable local KB retrieval (on/off)")
    connector: Optional[str] = Field(None, description="Force connector type (email/drive/calendar/sheets/docs/presentation/slack/notion/outlook/onedrive)")
    google_file_id: Optional[str] = Field(None, description="Google Drive file ID from @ mention")
    google_file_name: Optional[str] = Field(None, description="Google Drive filename from @ mention")
    microsoft_file_id: Optional[str] = Field(None, description="OneDrive/SharePoint item ID from @ mention")
    microsoft_file_name: Optional[str] = Field(None, description="OneDrive/SharePoint filename from @ mention")
    google_calendar_id: Optional[str] = Field(None, description="Google Calendar ID from @ mention")
    google_calendar_name: Optional[str] = Field(None, description="Google Calendar title from @ mention")
    gmail_location: Optional[str] = Field(None, description="Gmail folder operator from # mention (e.g. in:spam)")
    gmail_category: Optional[str] = Field(None, description="Gmail category operator from # mention (e.g. category:promotions)")
    outlook_folder: Optional[str] = Field(None, description="Outlook mail folder from # mention (e.g. inbox, sentitems)")
    outlook_location: Optional[str] = Field(None, description="Outlook folder alias from # mention chip")
    outlook_message_id: Optional[str] = Field(None, description="Outlook message ID from @ mention")
    microsoft_drive_path: Optional[str] = Field(None, description="OneDrive/SharePoint Graph drive path (me or sites/{id})")


class QueryResponse(BaseModel):
    status: str
    query: str
    answer: str
    citations: list
    metadata: Dict[str, Any]
    processing_time: float
    session_id: str
    turn_number: int
    was_reformulated: bool = False
    reformulated_query: Optional[str] = None


class SessionResponse(BaseModel):
    status: str
    session_id: str
    message: Optional[str] = None
    created_at: Optional[str] = None
    turn_count: Optional[int] = None


class ConversationTurnResponse(BaseModel):
    turn_id: int
    timestamp: str
    query: str
    reformulated_query: Optional[str] = None
    answer: str
    citations: list
    entities: list
    confidence: float


class SessionHistoryResponse(BaseModel):
    status: str
    session_id: str
    turn_count: int
    history: List[ConversationTurnResponse]


class AsyncQueryJobRequest(BaseModel):
    query: str = Field(..., description="User query", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="Optional user identifier for rate limiting")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    stream: Optional[bool] = Field(False, description="Enable streaming response")
    web: Optional[str] = Field("off", description="Enable duckduckgo web search (on/off)")
    kb: Optional[str] = Field("on", description="Enable local KB retrieval (on/off)")
    connector: Optional[str] = Field(None, description="Force connector type (email/drive/calendar/sheets/docs/presentation)")
    timeout_seconds: Optional[float] = Field(
        None,
        description="Optional per-job timeout in seconds",
        gt=0,
    )
    google_file_id: Optional[str] = Field(None, description="Google Drive file ID from @ mention")
    google_file_name: Optional[str] = Field(None, description="Google Drive filename from @ mention")
    google_calendar_id: Optional[str] = Field(None, description="Google Calendar ID from @ mention")
    google_calendar_name: Optional[str] = Field(None, description="Google Calendar title from @ mention")
    gmail_location: Optional[str] = Field(None, description="Gmail folder operator from # mention (e.g. in:spam)")
    gmail_category: Optional[str] = Field(None, description="Gmail category operator from # mention (e.g. category:promotions)")
    outlook_folder: Optional[str] = Field(None, description="Outlook mail folder from # mention (e.g. inbox, sentitems)")
    outlook_location: Optional[str] = Field(None, description="Outlook folder alias from # mention chip")
    outlook_message_id: Optional[str] = Field(None, description="Outlook message ID from @ mention")
    microsoft_drive_path: Optional[str] = Field(None, description="OneDrive/SharePoint Graph drive path (me or sites/{id})")
    microsoft_file_id: Optional[str] = Field(None, description="OneDrive/SharePoint item ID from @ mention")
    microsoft_file_name: Optional[str] = Field(None, description="OneDrive/SharePoint filename from @ mention")


class ConnectorQueryRequest(BaseModel):
    connector: str = Field(..., description="Connector to query (email/drive/calendar/sheets/docs/presentation)")
    query: str = Field(..., description="User query", min_length=1)
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: Optional[str] = Field(None, description="Optional user identifier for rate limiting")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class ConnectorRequest(BaseModel):
    query: str = Field(..., description="User query", min_length=1)
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: Optional[str] = Field(None, description="Optional user identifier for rate limiting")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class AsyncQueryJobAcceptedResponse(BaseModel):
    status: str
    job_id: str
    kb_id: str
    request_id: str
    queued_at: str
    timeout_seconds: float


class AsyncQueryJobStatusResponse(BaseModel):
    job_id: str
    kb_id: str
    job_status: str
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    query: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    timeout_seconds: Optional[float] = None
    attempts: int = 0
    cancellation_requested: bool = False
    is_terminal: bool = False
    retry_after_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class AsyncQueryJobCancelResponse(BaseModel):
    status: str
    job_id: str
    kb_id: str
    job_status: str
    message: Optional[str] = None


# =============================================================================
# Configuration Helpers
# =============================================================================


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _extract_mcp_text_content(response: Dict[str, Any]) -> str:
    if not isinstance(response, dict):
        return ""
    content = response.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return first.get("text", "")
    return ""


async def _build_slack_autocomplete_items(user_id: str) -> List[Dict[str, Any]]:
    if _slack_mcp_client is None:
        return []

    try:
        channel_response = await _slack_mcp_client.list_channels(user_id=user_id, query="")
        dm_response = await _slack_mcp_client._list_dms(user_id=user_id)
    except Exception as exc:
        logger.warning("Slack autocomplete sync failed for user_id=%s: %s", user_id, exc)
        return []

    items: List[Dict[str, Any]] = []
    # Parse channel listings
    text = _extract_mcp_text_content(channel_response)
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- **"):
            continue
        match = re.match(r"- \*\*(?P<label>[^*]+)\*\* \((?P<kind>[^)]+)\) — ID: (?P<id>.+)$", line)
        if not match:
            continue
        label = match.group("label")
        kind = match.group("kind")
        if kind == "DM":
            continue
        display = label if label.startswith("#") else f"#{label}"
        items.append({
            "display": display,
            "value": display,
            "icon": "💬",
            "category": "Channels",
            "meta": kind,
        })

    text = _extract_mcp_text_content(dm_response)
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("- **"):
            continue
        dm_match = re.match(r"- \*\*(?:DM(?: with (?P<name>.+))?)\*\* — ID: (?P<id>.+)$", line)
        if not dm_match:
            continue
        name = dm_match.group("name") or "Direct Message"
        display = f"@{name.strip()}"
        items.append({
            "display": display,
            "value": display,
            "icon": "👤",
            "category": "Direct Messages",
            "meta": "DM",
        })

    return items


async def _build_notion_autocomplete_items(user_id: str) -> List[Dict[str, Any]]:
    if _notion_mcp_client is None:
        return []

    items: List[Dict[str, Any]] = []
    try:
        page_response = await _notion_mcp_client.search_pages(user_id=user_id, query="", limit=50)
        db_response = await _notion_mcp_client.search_databases(user_id=user_id, query="", limit=50)
    except Exception as exc:
        logger.warning("Notion autocomplete sync failed for user_id=%s: %s", user_id, exc)
        return []

    for response, default_meta in [(page_response, "Page"), (db_response, "Database")]:
        text = _extract_mcp_text_content(response)
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("- **"):
                continue
            match = re.match(r"- \*\*(?P<title>.+?)\*\*(?: \((?P<meta>[^)]+)\))?", line)
            if not match:
                continue
            title = match.group("title").strip()
            meta = match.group("meta") or default_meta
            items.append({
                "display": title,
                "value": title,
                "icon": "📚" if meta.lower() == "page" else "🗄️",
                "category": "Notion Knowledge Bases",
                "meta": meta,
            })

    return items


async def _build_google_autocomplete_items(user_id: str, auth_token: Optional[str] = None) -> List[Dict[str, Any]]:
    if _mcp_client is None:
        return []

    token = auth_token
    if token is None:
        creds = _mcp_client._load_credentials(user_id, "drive")
        if creds is None or not creds.token:
            return []
        token = creds.token

    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://www.googleapis.com/drive/v3/files",
                headers=headers,
                params={
                    "pageSize": 20,
                    "orderBy": "createdTime desc",
                    "fields": "files(id,name,mimeType,webViewLink,createdTime)",
                    "q": "trashed = false",
                },
            )
            resp.raise_for_status()
            files_data = resp.json()
    except Exception as exc:
        logger.warning("Google autocomplete sync failed for user_id=%s: %s", user_id, exc)
        return []

    items: List[Dict[str, Any]] = []
    for file_obj in files_data.get("files", []):
        name = file_obj.get("name", "Unnamed File")
        mime = file_obj.get("mimeType", "unknown")
        friendly_type = "Doc"
        if "spreadsheet" in mime:
            friendly_type = "Sheet"
        elif "presentation" in mime:
            friendly_type = "Slide"
        elif "pdf" in mime:
            friendly_type = "PDF"
        elif "folder" in mime:
            friendly_type = "Folder"
        items.append({
            "display": name,
            "value": name,
            "icon": "📄" if friendly_type in {"Doc", "Sheet", "Slide", "PDF"} else "📁",
            "category": "Google Drive Files",
            "meta": friendly_type,
        })

    return items


async def _build_microsoft_onedrive_autocomplete_items(
    user_id: str,
    auth_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if _microsoft_mcp_client is None:
        return []

    token = auth_token
    if token is None:
        token = _microsoft_mcp_client._load_access_token(user_id, "onedrive")
    if not token:
        logger.warning("No Microsoft OneDrive token for autocomplete user_id=%s", user_id)
        return []

    from src.mcp.onedrive_search import collect_onedrive_browse_items

    try:
        browse_items = collect_onedrive_browse_items(token, timeout=15.0)
    except Exception as exc:
        logger.warning("OneDrive autocomplete sync failed for user_id=%s: %s", user_id, exc)
        return []

    items: List[Dict[str, Any]] = []
    for file_obj in browse_items:
        name = file_obj.get("name", "Unnamed File")
        file_id = file_obj.get("id", "")
        parent_folder = file_obj.get("_parent_folder") or ""
        is_folder = bool(file_obj.get("folder"))
        mime = (file_obj.get("file") or {}).get("mimeType", "unknown")
        friendly_type = "File"
        if is_folder:
            friendly_type = "Folder"
        elif "pdf" in mime:
            friendly_type = "PDF"
        elif "wordprocessingml" in mime or "msword" in mime:
            friendly_type = "Word"
        elif "spreadsheetml" in mime or "excel" in mime:
            friendly_type = "Excel"
        elif "presentationml" in mime or "powerpoint" in mime:
            friendly_type = "PowerPoint"
        display_name = f"{parent_folder}/{name}" if parent_folder and not is_folder else name
        items.append({
            "display": display_name,
            "value": name,
            "file_id": file_id,
            "drive_path": "me",
            "icon": "📄" if friendly_type != "Folder" else "📁",
            "category": "OneDrive Files",
            "meta": f"{friendly_type}" + (f" · {parent_folder}" if parent_folder else ""),
        })
    logger.info("OneDrive autocomplete built %d items for user_id=%s", len(items), user_id)
    return items


async def _build_microsoft_outlook_autocomplete_items(
    user_id: str,
    auth_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Recent Outlook inbox messages for @ mention disambiguation."""
    if _microsoft_mcp_client is None:
        return []

    token = auth_token
    if token is None:
        token = _microsoft_mcp_client._load_access_token(user_id, "outlook")
    if not token:
        logger.warning("No Microsoft Outlook token for autocomplete user_id=%s", user_id)
        return []

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://graph.microsoft.com/v1.0/me/mailFolders/inbox/messages",
                headers=headers,
                params={
                    "$top": "30",
                    "$orderby": "receivedDateTime desc",
                    "$select": "id,conversationId,subject,from,receivedDateTime,bodyPreview",
                },
            )
            if resp.status_code != 200:
                logger.warning(
                    "Outlook autocomplete failed user_id=%s status=%s",
                    user_id,
                    resp.status_code,
                )
                return []
            data = resp.json()
    except Exception as exc:
        logger.warning("Outlook autocomplete sync failed for user_id=%s: %s", user_id, exc)
        return []

    items: List[Dict[str, Any]] = []
    for msg in data.get("value") or []:
        subject = (msg.get("subject") or "(no subject)").strip()
        message_id = msg.get("id") or ""
        conversation_id = msg.get("conversationId") or ""
        sender_block = msg.get("from") or {}
        email = (sender_block.get("emailAddress") or {})
        sender = (email.get("name") or email.get("address") or "Unknown").strip()
        received = (msg.get("receivedDateTime") or "")[:10]
        if not message_id:
            continue
        items.append({
            "display": subject,
            "value": subject,
            "file_id": message_id,
            "outlook_message_id": message_id,
            "outlook_conversation_id": conversation_id,
            "icon": "📧",
            "category": "Outlook Messages",
            "meta": f"{sender} · {received}",
        })
    logger.info("Outlook autocomplete built %d items for user_id=%s", len(items), user_id)
    return items


def _resolve_google_access_token(
    user_id: str,
    auth_token: Optional[str] = None,
    prefer_services: Optional[List[str]] = None,
) -> Optional[str]:
    """Resolve a Google access token for REST calls (calendar, drive, etc.)."""
    if auth_token:
        return auth_token

    if _mcp_client is None:
        logger.warning(
            "Cannot resolve Google token for user_id=%s: MCP client not initialized",
            user_id,
        )
        return None

    for svc in prefer_services or ("calendar", "drive", "gmail"):
        creds = _mcp_client._load_credentials(user_id, svc)
        if creds and creds.token:
            logger.debug(
                "Resolved Google access token for user_id=%s from service=%s",
                user_id,
                svc,
            )
            return creds.token

    logger.warning(
        "No Google access token in Redis for user_id=%s (checked %s)",
        user_id,
        prefer_services or ("calendar", "drive", "gmail"),
    )
    return None


async def _build_calendar_autocomplete_items(user_id: str, auth_token: Optional[str] = None) -> List[Dict[str, Any]]:
    token = _resolve_google_access_token(
        user_id,
        auth_token=auth_token,
        prefer_services=["calendar", "drive", "gmail"],
    )
    if not token:
        return []

    from src.mcp.calendar_search import list_user_calendars_for_autocomplete

    items = list_user_calendars_for_autocomplete(token=token, timeout=15.0)
    logger.info(
        "Calendar autocomplete for user_id=%s: %d calendar(s)",
        user_id,
        len(items),
    )
    return items


async def refresh_user_workspace_cache(user_id: str, connector: str, auth_token: Optional[str] = None):
    """
    Background worker task that pulls real upstream workspace elements, sanitizes 
    metadata payload layouts, and writes them straight to Redis cache indexes.
    """
    if connector is None or not user_id:
        return

    connector_key = str(connector).lower().strip()
    if _redis_connection is None:
        logger.warning("Redis is not configured; autocomplete cache cannot be refreshed.")
        return

    try:
        redis_client = _redis_connection.client()
    except Exception as exc:
        logger.warning("Redis autocomplete cache unavailable: %s", exc)
        return

    cache_key = f"user_cache:{connector_key}:{user_id}"
    items_summary = []

    try:
        # =========================================================================
        # 1. SLACK LIVE INTEGRATION
        # =========================================================================
        if connector_key == "slack":
            if not auth_token:
                logger.warning("Slack auth token not provided for user %s", user_id)
                return
            
            client = AsyncWebClient(token=auth_token)
            
            # Fetch Public and Private Channels
            channels_response = await client.conversations_list(
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=100
            )
            if channels_response.get("ok"):
                for chan in channels_response.get("channels", []):
                    chan_name = chan.get("name", "")
                    items_summary.append({
                        "display": f"#{chan_name}",
                        "value": f"#{chan_name}",
                        "icon": "💬",
                        "category": "Channels"
                    })
            
            # Fetch Direct Messages (IMs) and resolve usernames
            im_response = await client.conversations_list(types="im", limit=50)
            users_response = await client.users_list(limit=150)
            
            if im_response.get("ok") and users_response.get("ok"):
                # Create a quick ID mapping dict for fast lookups
                user_map = {u["id"]: u.get("real_name") or u["name"] for u in users_response.get("members", [])}
                
                for im in im_response.get("channels", []):
                    target_user_id = im.get("user")
                    if target_user_id in user_map:
                        human_name = user_map[target_user_id]
                        items_summary.append({
                            "display": f"@{human_name}",
                            "value": f"@{human_name}",
                            "icon": "👤",
                            "category": "Direct Messages"
                        })

        # =========================================================================
        # 2. NOTION LIVE INTEGRATION
        # =========================================================================
        elif connector_key == "notion":
            if not auth_token:
                logger.warning("Notion auth token not provided for user %s", user_id)
                return
            
            async with httpx.AsyncClient() as http_client:
                headers = {
                    "Authorization": f"Bearer {auth_token}",
                    "Notion-Version": "2022-06-28",
                    "Content-Type": "application/json"
                }
                # POST an empty search body query to scan all index workspaces
                response = await http_client.post(
                    "https://api.notion.com/v1/search",
                    headers=headers,
                    json={"page_size": 100}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for obj in data.get("results", []):
                        obj_type = obj.get("object") # 'page' or 'database'
                        
                        # Extract nested Notion title values cleanly
                        properties = obj.get("properties", {})
                        title_text = ""
                        
                        if obj_type == "page":
                            # Pages handle titles inside various custom user block parameters
                            title_list = properties.get("title", {}).get("title", [])
                            if not title_list and "Name" in properties:
                                title_list = properties.get("Name", {}).get("title", [])
                            title_text = "".join([t.get("plain_text", "") for t in title_list])
                        elif obj_type == "database":
                            title_list = obj.get("title", [])
                            title_text = "".join([t.get("plain_text", "") for t in title_list])
                        
                        if title_text:
                            items_summary.append({
                                "display": title_text,
                                "value": title_text,
                                "icon": "📚" if obj_type == "page" else "🗄️",
                                "category": "Notion Knowledge Bases",
                                "meta": obj_type.capitalize()
                            })

        # =========================================================================
        # 3. GOOGLE CALENDAR LIVE INTEGRATION
        # =========================================================================
        elif connector_key == "calendar":
            items_summary = await _build_calendar_autocomplete_items(
                user_id, auth_token=auth_token
            )
            if not items_summary:
                logger.warning(
                    "Calendar autocomplete cache empty for user_id=%s "
                    "(auth_token_passed=%s)",
                    user_id,
                    bool(auth_token),
                )

        # =========================================================================
        # 4. GOOGLE DRIVE LIVE INTEGRATION
        # =========================================================================
        elif connector_key == "onedrive":
            items_summary = await _build_microsoft_onedrive_autocomplete_items(
                user_id,
                auth_token=auth_token,
            )
            if not items_summary:
                logger.warning(
                    "OneDrive autocomplete cache empty for user_id=%s (auth_token_passed=%s)",
                    user_id,
                    bool(auth_token),
                )

        elif connector_key == "outlook":
            items_summary = await _build_microsoft_outlook_autocomplete_items(
                user_id,
                auth_token=auth_token,
            )
            if not items_summary:
                logger.warning(
                    "Outlook autocomplete cache empty for user_id=%s (auth_token_passed=%s)",
                    user_id,
                    bool(auth_token),
                )

        elif connector_key == "google":
            if not auth_token:
                logger.warning("Google auth token not provided for user %s", user_id)
                return
            
            async with httpx.AsyncClient() as http_client:
                headers = {"Authorization": f"Bearer {auth_token}"}
                # Query specific fields and filter out trashed resources
                url = (
                    "https://www.googleapis.com/drive/v3/files"
                    "?pageSize=100&q=trashed=false&fields=files(id,name,mimeType)"
                )
                response = await http_client.get(url, headers=headers)
                
                if response.status_code == 200:
                    mime_map = {
                        "application/vnd.google-apps.document": {"icon": "📄", "meta": "Doc"},
                        "application/vnd.google-apps.spreadsheet": {"icon": "📊", "meta": "Sheet"},
                        "application/vnd.google-apps.presentation": {"icon": "📊", "meta": "Slide"},
                        "application/pdf": {"icon": "📕", "meta": "PDF"}
                    }
                    
                    data = response.json()
                    for file in data.get("files", []):
                        mime = file.get("mimeType", "")
                        file_name = file.get("name", "")
                        
                        # Match UI icons based on incoming workspace types
                        style = mime_map.get(mime, {"icon": "📁", "meta": "File"})
                        
                        file_id = file.get("id", "")
                        items_summary.append({
                            "display": file_name,
                            "value": file_name,        # still insert the name into the query text
                            "file_id": file_id,        # carry the Drive ID separately
                            "icon": style["icon"],
                            "category": "Google Drive Files",
                            "meta": style["meta"]
                        })
        else:
            logger.warning("Unsupported autocomplete connector: %s", connector_key)
            return

        # =========================================================================
        # CACHE COMMIT ENGINE
        # =========================================================================
        if items_summary:
            # Commit payload straight to Redis cache index with a 1-hour expiration TTL
            redis_client.setex(cache_key, 3600, json.dumps(items_summary))
            logger.info(f"Successfully cached {len(items_summary)} entries for {connector_key} ({user_id})")
        else:
            logger.warning(f"No sync components discovered during extraction index sweep for {connector_key}")

    except Exception as e:
        logger.error(f"Failed background autocomplete sync cache write for user {user_id}: {e}", exc_info=True)


def _env_flag(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


# =============================================================================
# Runtime Policy (runtime.yaml + env overrides)
# =============================================================================

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUNTIME_PATH = _REPO_ROOT / "src" / "config" / "runtime.yaml"
_RUNTIME_CONFIG_PATH = Path(os.environ.get("RUNTIME_CONFIG_PATH", str(_DEFAULT_RUNTIME_PATH)))

if _RUNTIME_CONFIG_PATH.exists():
    _runtime_policy = load_runtime_policy(_RUNTIME_CONFIG_PATH)
else:
    _runtime_policy = RuntimePolicy()

# Ensure downstream components that still read env defaults stay consistent.
os.environ.setdefault("LLM_MAX_CONCURRENCY", str(_runtime_policy.llm_concurrency.max_concurrency))
os.environ.setdefault("LLM_ADMISSION_TIMEOUT_SECONDS", str(_runtime_policy.llm_concurrency.admission_timeout_seconds))
os.environ.setdefault("LLM_RETRY_AFTER_SECONDS", str(_runtime_policy.llm_concurrency.retry_after_seconds))

CONFIG_DIR = os.environ.get("CONFIG_DIR", str(_REPO_ROOT / "src" / "config"))


# =============================================================================
# Concurrency & Runtime Controls
# =============================================================================

# Dedicated executors for workload isolation
query_executor: Optional[ThreadPoolExecutor] = None
index_executor: Optional[ThreadPoolExecutor] = None

QUERY_EXECUTOR_WORKERS = int(_runtime_policy.executors.query_workers)
INDEX_EXECUTOR_WORKERS = int(_runtime_policy.executors.index_workers)
LLM_MAX_CONCURRENCY = max(1, int(_runtime_policy.llm_concurrency.max_concurrency))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
DEFAULT_QUERY_TIMEOUT_SECONDS = 180.0 if LLM_MAX_TOKENS >= 1024 else 120.0
QUERY_TIMEOUT_SECONDS = (
    float(_runtime_policy.admission_control.query_timeout_seconds)
    if _runtime_policy.admission_control.query_timeout_seconds is not None
    else DEFAULT_QUERY_TIMEOUT_SECONDS
)
ADMIN_WRITE_TIMEOUT_SECONDS = float(_runtime_policy.admission_control.admin_write_timeout_seconds)
DEFAULT_QUERY_MAX_INFLIGHT = max(1, min(QUERY_EXECUTOR_WORKERS, LLM_MAX_CONCURRENCY))
_raw_query_max_inflight = _runtime_policy.admission_control.query_max_inflight
if isinstance(_raw_query_max_inflight, str) and _raw_query_max_inflight.strip().lower() == "auto":
    QUERY_MAX_INFLIGHT = DEFAULT_QUERY_MAX_INFLIGHT
else:
    QUERY_MAX_INFLIGHT = int(_raw_query_max_inflight)
QUERY_MAX_INFLIGHT = max(1, min(QUERY_MAX_INFLIGHT, QUERY_EXECUTOR_WORKERS))
QUERY_ADMISSION_TIMEOUT_SECONDS = float(_runtime_policy.admission_control.query_admission_timeout_seconds)
QUERY_RETRY_AFTER_SECONDS = int(_runtime_policy.admission_control.query_retry_after_seconds)
SESSION_LOCK_TIMEOUT_SECONDS = float(_runtime_policy.admission_control.session_lock_timeout_seconds)
SESSION_LOCK_RETRY_AFTER_SECONDS = int(_runtime_policy.admission_control.session_lock_retry_after_seconds)
QUERY_READ_GUARD_TIMEOUT_SECONDS = float(_runtime_policy.admission_control.query_read_guard_timeout_seconds)
INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS = float(_runtime_policy.admission_control.indexing_write_drain_timeout_seconds)
INDEXING_RETRY_AFTER_SECONDS = int(_runtime_policy.admission_control.indexing_retry_after_seconds)

# exclusive for older (query/session blocked during indexing)
# online for newer (allows query/session traffic while indexing runs)
INDEXING_MODE = str(_runtime_policy.indexing.mode).strip().lower()
if INDEXING_MODE not in {"exclusive", "online"}:
    INDEXING_MODE = "exclusive"
INDEXING_ONLINE_MODE = INDEXING_MODE == "online"
INDEXING_DENSE_VISIBILITY_FILTER_ENABLED = bool(_runtime_policy.indexing.dense_visibility_filter_enabled)
INDEXING_DENSE_DEFAULT_FILTER = (
    {"ingest_state": "ready"}
    if INDEXING_ONLINE_MODE and INDEXING_DENSE_VISIBILITY_FILTER_ENABLED
    else None
)
INDEXING_ONLINE_MIN_TEXT_LENGTH = int(_runtime_policy.indexing.online_min_text_length)
INDEXING_ONLINE_MIN_QUALITY_SCORE = float(_runtime_policy.indexing.online_min_quality_score)
INDEXING_ONLINE_REMOVE_URLS = bool(_runtime_policy.indexing.online_remove_urls)
INDEXING_ONLINE_REMOVE_EMAILS = bool(_runtime_policy.indexing.online_remove_emails)
INDEXING_ONLINE_USE_UNSTRUCTURED_FALLBACK = bool(_runtime_policy.indexing.online_use_unstructured_fallback)

# Async query jobs
ASYNC_QUERY_JOBS_ENABLED = bool(_runtime_policy.async_jobs.enabled)
ASYNC_QUERY_JOB_WORKERS = max(
    1,
    min(
        int(_runtime_policy.async_jobs.workers),
        QUERY_EXECUTOR_WORKERS,
    ),
)
ASYNC_QUERY_JOB_QUEUE_MAX_SIZE = max(1, int(_runtime_policy.async_jobs.queue_max_size))
ASYNC_QUERY_JOB_DEFAULT_TIMEOUT_SECONDS = max(
    QUERY_TIMEOUT_SECONDS,
    float(_runtime_policy.async_jobs.default_timeout_seconds),
)
ASYNC_QUERY_JOB_MAX_TIMEOUT_SECONDS = max(
    ASYNC_QUERY_JOB_DEFAULT_TIMEOUT_SECONDS,
    float(_runtime_policy.async_jobs.max_timeout_seconds),
)
ASYNC_QUERY_JOB_RETENTION_SECONDS = max(60, int(_runtime_policy.async_jobs.retention_seconds))
ASYNC_QUERY_JOB_CLEANUP_INTERVAL_SECONDS = max(
    5.0, float(_runtime_policy.async_jobs.cleanup_interval_seconds)
)
ASYNC_QUERY_JOB_MIN_RETRY_DELAY_SECONDS = float(_runtime_policy.async_jobs.min_retry_delay_seconds)
ASYNC_QUERY_JOB_MAX_RETRY_DELAY_SECONDS = max(
    0.5, float(_runtime_policy.async_jobs.max_retry_delay_seconds)
)

# Ownify provisioning jobs
OWNIFY_PROVISIONING_API_KEY = os.getenv("OWNIFY_PROVISIONING_API_KEY")
PROVISIONING_JOBS_ENABLED = bool(_runtime_policy.provisioning_jobs.enabled)
PROVISIONING_JOB_WORKERS = max(
    1,
    min(
        int(_runtime_policy.provisioning_jobs.workers),
        INDEX_EXECUTOR_WORKERS,
    ),
)
PROVISIONING_JOB_QUEUE_MAX_SIZE = max(1, int(_runtime_policy.provisioning_jobs.queue_max_size))
PROVISIONING_JOB_DEFAULT_TIMEOUT_SECONDS = max(
    30.0,
    float(_runtime_policy.provisioning_jobs.default_timeout_seconds),
)
PROVISIONING_JOB_MAX_TIMEOUT_SECONDS = max(
    PROVISIONING_JOB_DEFAULT_TIMEOUT_SECONDS,
    float(_runtime_policy.provisioning_jobs.max_timeout_seconds),
)
PROVISIONING_JOB_RETENTION_SECONDS = max(300, int(_runtime_policy.provisioning_jobs.retention_seconds))
PROVISIONING_JOB_CLEANUP_INTERVAL_SECONDS = max(
    10.0, float(_runtime_policy.provisioning_jobs.cleanup_interval_seconds)
)
PROVISIONING_JOB_MIN_RETRY_DELAY_SECONDS = float(_runtime_policy.provisioning_jobs.min_retry_delay_seconds)
PROVISIONING_JOB_MAX_RETRY_DELAY_SECONDS = max(
    0.5, float(_runtime_policy.provisioning_jobs.max_retry_delay_seconds)
)

# LLM stuck-generation watchdog
LLM_STUCK_WATCHDOG_ENABLED = bool(_runtime_policy.watchdog.enabled)
LLM_STUCK_THRESHOLD_SECONDS = max(30.0, float(_runtime_policy.watchdog.stuck_threshold_seconds))
LLM_STUCK_CHECK_INTERVAL_SECONDS = max(2.0, float(_runtime_policy.watchdog.check_interval_seconds))

# Redis safety layer
REDIS_ENABLED = bool(_runtime_policy.redis.enabled)
REDIS_URL = _runtime_policy.redis.url
REDIS_KEY_PREFIX = _runtime_policy.redis.key_prefix
REDIS_SOCKET_TIMEOUT_SECONDS = float(_runtime_policy.redis.socket_timeout_seconds)
REDIS_CONNECT_TIMEOUT_SECONDS = float(_runtime_policy.redis.connect_timeout_seconds)

REDIS_LOCK_ENABLED = REDIS_ENABLED and bool(_runtime_policy.redis.lock_enabled)
REDIS_LOCK_TTL_SECONDS = float(_runtime_policy.redis.lock_ttl_seconds)
REDIS_LOCK_RETRY_INTERVAL_SECONDS = float(_runtime_policy.redis.lock_retry_interval_seconds)

REDIS_RATE_LIMIT_ENABLED = REDIS_ENABLED and bool(_runtime_policy.redis.rate_limit_enabled)
REDIS_RATE_COUNTER_TTL_SECONDS = max(
    120,
    int(float(_runtime_policy.redis.rate_counter_ttl_seconds))
)
RATE_LIMIT_RETRY_AFTER_SECONDS = int(_runtime_policy.redis.rate_limit_retry_after_seconds)
SESSION_MAX_INFLIGHT_REDIS = int(_runtime_policy.redis.session_max_inflight)
USER_MAX_INFLIGHT_REDIS = int(_runtime_policy.redis.user_max_inflight)

REDIS_SESSION_STORE_ENABLED = REDIS_ENABLED and bool(_runtime_policy.redis.session_store_enabled)
REDIS_SESSION_TTL_SECONDS = int(_runtime_policy.redis.session_ttl_seconds)
REDIS_SESSION_READ_THROUGH_ENABLED = bool(_runtime_policy.redis.session_read_through_enabled)

REDIS_QUERY_CACHE_ENABLED = REDIS_ENABLED and bool(_runtime_policy.redis.query_cache_enabled)
REDIS_QUERY_CACHE_TTL_SECONDS = int(_runtime_policy.redis.query_cache_ttl_seconds)

PIPELINE_STRUCTURED_FAST_MODE = _env_optional_bool("PIPELINE_STRUCTURED_FAST_MODE")


# =============================================================================
# Concurrency State
# =============================================================================

# In-process per-session lock registry
_session_locks: Dict[str, asyncio.Lock] = {}
_locks_meta_lock = asyncio.Lock()


@dataclass
class KbRwGate:
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    active_readers: int = 0
    writer_active: bool = False


_kb_rw_gates: Dict[str, KbRwGate] = {}
_kb_rw_gates_lock = asyncio.Lock()

# Query admission control
_query_admission_semaphore = asyncio.Semaphore(QUERY_MAX_INFLIGHT)
_query_admission_state_lock = asyncio.Lock()
_query_inflight = 0
_query_waiters = 0
_query_rejections = 0
_query_peak_inflight = 0
_session_lock_rejections = 0
_rate_limit_rejections = 0
_indexing_gate_rejections = 0

# Redis safety-layer runtime objects
_redis_connection: Optional[RedisConnection] = None
_distributed_session_lock_backend: Optional[DistributedSessionLockBackend] = None
_redis_rate_limiter: Optional[RedisInFlightRateLimiter] = None

# Async query jobs runtime
QUERY_JOB_STATUS_QUEUED = "queued"
QUERY_JOB_STATUS_RUNNING = "running"
QUERY_JOB_STATUS_CANCELLING = "cancelling"
QUERY_JOB_STATUS_SUCCEEDED = "succeeded"
QUERY_JOB_STATUS_FAILED = "failed"
QUERY_JOB_STATUS_CANCELLED = "cancelled"
QUERY_JOB_STATUS_TIMED_OUT = "timed_out"
QUERY_JOB_TERMINAL_STATES = {
    QUERY_JOB_STATUS_SUCCEEDED,
    QUERY_JOB_STATUS_FAILED,
    QUERY_JOB_STATUS_CANCELLED,
    QUERY_JOB_STATUS_TIMED_OUT,
}
QUERY_JOB_RETRYABLE_ERROR_CODES = {
    "query_overloaded",
    "llm_overloaded",
    "session_busy",
    "rate_limited",
    "lock_backend_unavailable",
    "indexing_in_progress",
}

OWNIFY_JOB_TYPE_PROVISION = "provision"
OWNIFY_JOB_TYPE_DOCUMENTS = "documents"
OWNIFY_JOB_TYPE_DOCUMENT_DELETE = "document_delete"
OWNIFY_JOB_TYPE_TENANT_DELETE = "tenant_delete"
OWNIFY_JOB_STATUS_QUEUED = "queued"
OWNIFY_JOB_STATUS_RUNNING = "running"
OWNIFY_JOB_STATUS_SUCCEEDED = "succeeded"
OWNIFY_JOB_STATUS_SUCCEEDED_WITH_ERRORS = "succeeded_with_errors"
OWNIFY_JOB_STATUS_FAILED = "failed"
OWNIFY_JOB_STATUS_CANCELLED = "cancelled"
OWNIFY_JOB_STATUS_TIMED_OUT = "timed_out"
OWNIFY_JOB_TERMINAL_STATES = {
    OWNIFY_JOB_STATUS_SUCCEEDED,
    OWNIFY_JOB_STATUS_SUCCEEDED_WITH_ERRORS,
    OWNIFY_JOB_STATUS_FAILED,
    OWNIFY_JOB_STATUS_CANCELLED,
    OWNIFY_JOB_STATUS_TIMED_OUT,
}

OWNIFY_PHASE_QUEUED = "queued"
OWNIFY_PHASE_VALIDATING = "validating"
OWNIFY_PHASE_CREATING_KB = "creating_kb"
OWNIFY_PHASE_WRITING_CONFIG = "writing_config"
OWNIFY_PHASE_INDEXING_DOCUMENTS = "indexing_documents"
OWNIFY_PHASE_DELETING_DOCUMENT = "deleting_document"
OWNIFY_PHASE_DELETING_TENANT = "deleting_tenant"
OWNIFY_PHASE_FINALIZING = "finalizing"


@dataclass
class AsyncQueryJobRecord:
    job_id: str
    kb_id: str
    request: Any
    request_id: str
    session_id: Optional[str]
    user_id: Optional[str]
    timeout_seconds: float
    status: str = QUERY_JOB_STATUS_QUEUED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    attempts: int = 0
    cancellation_requested: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    last_retry_after_seconds: Optional[float] = None


_async_query_job_queue: Optional[asyncio.Queue[str]] = None
_async_query_jobs: Dict[str, AsyncQueryJobRecord] = {}
_async_query_jobs_lock = asyncio.Lock()
_async_query_job_workers: List[asyncio.Task] = []
_async_query_job_cleanup_task: Optional[asyncio.Task] = None

# Ownify provisioning jobs runtime
_ownify_job_queue: Optional[asyncio.Queue[str]] = None
_ownify_job_workers: List[asyncio.Task] = []
_ownify_job_cleanup_task: Optional[asyncio.Task] = None
_ownify_job_store: Optional[ProvisioningJobStore] = None
_ownify_tenant_locks: Dict[str, asyncio.Lock] = {}
_ownify_tenant_locks_lock = asyncio.Lock()

# LLM watchdog task
_llm_stuck_watchdog_task: Optional[asyncio.Task] = None


# =============================================================================
# Application Lifespan
# =============================================================================

multi_kb_pipeline: Optional[MultiKBPipeline] = None
kb_manager: Optional[KBManager] = None

# Google Workspace MCP client (initialized during lifespan if enabled)
_mcp_client = None

# Slack MCP client (initialized during lifespan if enabled)
_slack_mcp_client = None

# Notion MCP client (initialized during lifespan if enabled)
_notion_mcp_client = None

# Microsoft 365 MCP client (initialized during lifespan if enabled)
_microsoft_mcp_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global multi_kb_pipeline, kb_manager, query_executor, index_executor
    global _redis_connection, _distributed_session_lock_backend, _redis_rate_limiter
    global _llm_stuck_watchdog_task, _mcp_client, _slack_mcp_client, _notion_mcp_client
    global _microsoft_mcp_client
    global _ownify_job_store

    data_dir = os.getenv("DATA_DIR", "./data")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    # Initialize dedicated executors for workload isolation.
    query_executor = ThreadPoolExecutor(
        max_workers=QUERY_EXECUTOR_WORKERS,
        thread_name_prefix="multi-query-exec",
    )
    index_executor = ThreadPoolExecutor(
        max_workers=INDEX_EXECUTOR_WORKERS,
        thread_name_prefix="multi-index-exec",
    )

    logger.info(
        "Executors initialized (query=%s, index=%s)",
        QUERY_EXECUTOR_WORKERS,
        INDEX_EXECUTOR_WORKERS,
    )

    anchor_terms_env = os.getenv("COLLECTION_ANCHOR_TERMS", "")
    anchor_terms = [term.strip() for term in anchor_terms_env.split(",") if term.strip()]

    settings = MultiKBSettings(
        qdrant_url=qdrant_url,
        data_dir=data_dir,
        llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 1024),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        llm_top_p=float(os.getenv("LLM_TOP_P", "0.8")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
        embedding_dim=_env_int("EMBEDDING_DIM", 1024),
        # Avoid probing CUDA at startup; vLLM is sensitive to early CUDA init in parent.
        use_gpu=_env_bool("USE_GPU", True),
        rerank_top_k=_env_int("RERANK_TOP_K", 7),
        nli_threshold=float(os.getenv("NLI_THRESHOLD", "0.0")),
        dedup_threshold=float(os.getenv("DEDUP_THRESHOLD", "0.85")),
        normalize_newlines=os.getenv("NORMALIZE_NEWLINES", "preserve"),
        enable_memory=_env_bool("ENABLE_MEMORY", True),
        max_turns=_env_int("MAX_TURNS", 50),
        history_in_prompt_turns=_env_int("HISTORY_IN_PROMPT_TURNS", 10),
        auto_save=_env_bool("AUTO_SAVE", True),
        enable_reformulation=_env_bool("ENABLE_REFORMULATION", True),
        use_llm_reformulation=_env_bool("USE_LLM_REFORMULATION", True),
        enable_cache=_env_bool("ENABLE_QUERY_CACHE", False),
        cache_redis_enabled=REDIS_QUERY_CACHE_ENABLED,
        cache_redis_ttl_seconds=REDIS_QUERY_CACHE_TTL_SECONDS,
        redis_enabled=REDIS_SESSION_STORE_ENABLED,
        redis_url=REDIS_URL,
        redis_key_prefix=REDIS_KEY_PREFIX,
        redis_socket_timeout_seconds=REDIS_SOCKET_TIMEOUT_SECONDS,
        redis_connect_timeout_seconds=REDIS_CONNECT_TIMEOUT_SECONDS,
        session_store_ttl_seconds=REDIS_SESSION_TTL_SECONDS,
        session_read_through_enabled=REDIS_SESSION_READ_THROUGH_ENABLED,
        allow_general_knowledge_fallback=_env_flag("ALLOW_GENERAL_KNOWLEDGE_FALLBACK", "true"),
        min_verification_threshold=float(os.getenv("MIN_VERIFICATION_THRESHOLD", "0.1")),
        enable_collection_query_anchoring=_env_flag("ENABLE_COLLECTION_QUERY_ANCHORING", "true"),
        collection_anchor_terms=anchor_terms,
        structured_query_fast_mode=PIPELINE_STRUCTURED_FAST_MODE,
        structured_entity_resolution=_env_flag("STRUCTURED_ENTITY_RESOLUTION", "true"),
        structured_natural_response_style=_env_flag("STRUCTURED_NATURAL_RESPONSE_STYLE", "true"),
        dense_default_filters=INDEXING_DENSE_DEFAULT_FILTER,
        max_active_pipelines=_env_int("MAX_ACTIVE_PIPELINES", 2),
    )

    registry_path = os.getenv("KB_REGISTRY_PATH", str(Path(data_dir) / "kb_registry.json"))
    registry = KBRegistry(registry_path=registry_path)

    init_index_embedder = _env_bool("SHARED_INDEX_EMBEDDER", True)
    shared_resources = SharedResources(settings=settings, init_index_embedder=init_index_embedder)

    kb_manager_config = KBManagerConfig(
        qdrant_url=qdrant_url,
        data_dir=data_dir,
        config_dir=CONFIG_DIR,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        embedding_cache_dir=str(Path(data_dir) / "embeddings_cache"),
        indexing_mode=INDEXING_MODE,
        backfill_ingest_state_on_startup=bool(_runtime_policy.indexing.backfill_ingest_state_on_startup),
        online_min_text_length=INDEXING_ONLINE_MIN_TEXT_LENGTH,
        online_min_quality_score=INDEXING_ONLINE_MIN_QUALITY_SCORE,
        online_remove_urls=INDEXING_ONLINE_REMOVE_URLS,
        online_remove_emails=INDEXING_ONLINE_REMOVE_EMAILS,
        online_use_unstructured_fallback=INDEXING_ONLINE_USE_UNSTRUCTURED_FALLBACK,
        enable_entity_extraction=_env_flag("ENABLE_ENTITY_EXTRACTION", "true"),
        ner_backend=os.getenv("NER_BACKEND", "spacy"),
        ner_model=os.getenv("NER_MODEL", "en_core_web_lg"),
    )

    kb_manager = KBManager(
        registry=registry,
        config=kb_manager_config,
        shared_embedding_generator=shared_resources.index_embedder,
    )

    _ownify_job_store = ProvisioningJobStore(
        jobs_dir=str(Path(data_dir) / "provisioning_jobs"),
    )

    multi_kb_pipeline = MultiKBPipeline(
        registry=registry,
        settings=settings,
        shared_resources=shared_resources,
    )

    if REDIS_ENABLED:
        redis_runtime = RedisRuntimeConfig(
            enabled=True,
            url=REDIS_URL,
            key_prefix=REDIS_KEY_PREFIX,
            socket_timeout_seconds=REDIS_SOCKET_TIMEOUT_SECONDS,
            connect_timeout_seconds=REDIS_CONNECT_TIMEOUT_SECONDS,
        )
        _redis_connection = RedisConnection(redis_runtime)
        if _redis_connection.init_error:
            logger.warning("Redis init issue: %s", _redis_connection.init_error)
        else:
            logger.info("Redis safety layer initialized")

        if REDIS_LOCK_ENABLED and _redis_connection is not None:
            _distributed_session_lock_backend = DistributedSessionLockBackend(
                connection=_redis_connection,
                lock_ttl_seconds=REDIS_LOCK_TTL_SECONDS,
                retry_sleep_seconds=REDIS_LOCK_RETRY_INTERVAL_SECONDS,
            )
            logger.info("Distributed session lock backend enabled")
        else:
            _distributed_session_lock_backend = None

        if REDIS_RATE_LIMIT_ENABLED and _redis_connection is not None:
            _redis_rate_limiter = RedisInFlightRateLimiter(
                connection=_redis_connection,
                counter_ttl_seconds=REDIS_RATE_COUNTER_TTL_SECONDS,
            )
            logger.info("Redis rate limiter enabled")
        else:
            _redis_rate_limiter = None
    else:
        _redis_connection = None
        _distributed_session_lock_backend = None
        _redis_rate_limiter = None

    if not OWNIFY_PROVISIONING_API_KEY:
        logger.warning(
            "OWNIFY_PROVISIONING_API_KEY is not set; /ownify provisioning endpoints are unauthenticated"
        )

    await _start_async_query_jobs_runtime()
    await _start_ownify_jobs_runtime()
    if LLM_STUCK_WATCHDOG_ENABLED:
        _llm_stuck_watchdog_task = asyncio.create_task(_llm_stuck_watchdog_loop())
        logger.info(
            "LLM stuck watchdog enabled (threshold=%.1fs, check_interval=%.1fs)",
            LLM_STUCK_THRESHOLD_SECONDS,
            LLM_STUCK_CHECK_INTERVAL_SECONDS,
        )

    # -------------------------------------------------------------------------
    # Google Workspace MCP client (optional — skipped if env vars not set)
    # -------------------------------------------------------------------------
    _mcp_enabled = _env_bool("GOOGLE_MCP_ENABLED", True)
    _mcp_client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
    _mcp_client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")
    _mcp_redirect_uri = os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "")

    if _mcp_enabled and _mcp_client_id and _mcp_client_secret and _mcp_redirect_uri:
        if _redis_connection is not None and not _redis_connection.init_error:
            try:
                from src.mcp.google_workspace_client import GoogleWorkspaceMCPClient
                _mcp_client = GoogleWorkspaceMCPClient(
                    client_id=_mcp_client_id,
                    client_secret=_mcp_client_secret,
                    redirect_uri=_mcp_redirect_uri,
                    redis_client=_redis_connection.client(),
                    redis_key_prefix=REDIS_KEY_PREFIX,
                    token_ttl_seconds=86400,
                    services=["gmail", "drive", "calendar"],
                )
                logger.info("Google Workspace MCP client initialized")
            except Exception as _mcp_init_err:
                logger.warning("MCP client init failed (non-fatal): %s", _mcp_init_err)
                _mcp_client = None
        else:
            logger.warning(
                "MCP client skipped: Redis is required for token storage but is unavailable"
            )
    elif _mcp_enabled:
        logger.warning(
            "MCP enabled but GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET / "
            "GOOGLE_OAUTH_REDIRECT_URI are not set — MCP disabled"
        )

    # -------------------------------------------------------------------------
    # Slack MCP client (optional — skipped if env vars not set)
    # -------------------------------------------------------------------------
    _slack_enabled = _env_bool("SLACK_MCP_ENABLED", False)
    _slack_client_id = os.getenv("SLACK_CLIENT_ID", "")
    _slack_client_secret = os.getenv("SLACK_CLIENT_SECRET", "")
    _slack_redirect_uri = os.getenv("SLACK_OAUTH_REDIRECT_URI", "")
    _slack_token_ttl = int(os.getenv("SLACK_MCP_TOKEN_TTL", str(30 * 24 * 3600)))

    if _slack_enabled and (_slack_client_id and _slack_client_secret and _slack_redirect_uri):
        if _redis_connection is not None and not _redis_connection.init_error:
            try:
                from src.mcp.slack_client import SlackMCPClient
                _slack_mcp_client = SlackMCPClient(
                    client_id=_slack_client_id,
                    client_secret=_slack_client_secret,
                    redirect_uri=_slack_redirect_uri,
                    redis_client=_redis_connection.client(),
                    redis_key_prefix=REDIS_KEY_PREFIX,
                    token_ttl_seconds=_slack_token_ttl,
                )
                logger.info("Slack MCP client initialized")
            except Exception as _slack_init_err:
                logger.warning("Slack MCP client init failed (non-fatal): %s", _slack_init_err)
                _slack_mcp_client = None
        else:
            logger.warning(
                "Slack MCP client skipped: Redis is required for token storage but is unavailable"
            )
    elif _slack_enabled:
        logger.warning(
            "Slack MCP enabled but SLACK_CLIENT_ID / SLACK_CLIENT_SECRET / "
            "SLACK_OAUTH_REDIRECT_URI are not set — Slack MCP disabled"
        )

    # -------------------------------------------------------------------------
    # Notion MCP client (optional — skipped if env vars not set)
    # -------------------------------------------------------------------------
    _notion_enabled       = _env_bool("NOTION_MCP_ENABLED", False)
    _notion_client_id     = os.getenv("NOTION_CLIENT_ID", "")
    _notion_client_secret = os.getenv("NOTION_CLIENT_SECRET", "")
    _notion_redirect_uri  = os.getenv("NOTION_OAUTH_REDIRECT_URI", "")
    _notion_token_ttl     = int(os.getenv("NOTION_MCP_TOKEN_TTL", str(30 * 24 * 3600)))

    if _notion_enabled and (_notion_client_id and _notion_client_secret and _notion_redirect_uri):
        if _redis_connection is not None and not _redis_connection.init_error:
            try:
                from src.mcp.notion_client import NotionMCPClient
                _notion_mcp_client = NotionMCPClient(
                    client_id=_notion_client_id,
                    client_secret=_notion_client_secret,
                    redirect_uri=_notion_redirect_uri,
                    redis_client=_redis_connection.client(),
                    redis_key_prefix=REDIS_KEY_PREFIX,
                    token_ttl_seconds=_notion_token_ttl,
                )
                logger.info("Notion MCP client initialized")
            except Exception as _notion_init_err:
                logger.warning("Notion MCP client init failed (non-fatal): %s", _notion_init_err)
                _notion_mcp_client = None
        else:
            logger.warning(
                "Notion MCP client skipped: Redis is required for token storage but is unavailable"
            )
    elif _notion_enabled:
        logger.warning(
            "Notion MCP enabled but NOTION_CLIENT_ID / NOTION_CLIENT_SECRET / "
            "NOTION_OAUTH_REDIRECT_URI are not set — Notion MCP disabled"
        )

    # -------------------------------------------------------------------------
    # Microsoft 365 MCP client (optional — skipped if env vars not set)
    # -------------------------------------------------------------------------
    _microsoft_enabled = _env_bool("MICROSOFT_MCP_ENABLED", False)
    _microsoft_client_id = os.getenv("MICROSOFT_OAUTH_CLIENT_ID", "")
    _microsoft_client_secret = os.getenv("MICROSOFT_OAUTH_CLIENT_SECRET", "")
    _microsoft_redirect_uri = os.getenv("MICROSOFT_OAUTH_REDIRECT_URI", "")
    _microsoft_tenant_id = (os.getenv("MICROSOFT_OAUTH_TENANT_ID", "common") or "common").split("#")[0].strip() or "common"
    _microsoft_token_ttl = int(os.getenv("MICROSOFT_MCP_TOKEN_TTL", "86400"))

    if _microsoft_enabled and (_microsoft_client_id and _microsoft_client_secret and _microsoft_redirect_uri):
        if _redis_connection is not None and not _redis_connection.init_error:
            try:
                from src.mcp.microsoft365_client import Microsoft365MCPClient
                _microsoft_mcp_client = Microsoft365MCPClient(
                    client_id=_microsoft_client_id,
                    client_secret=_microsoft_client_secret,
                    redirect_uri=_microsoft_redirect_uri,
                    redis_client=_redis_connection.client(),
                    redis_key_prefix=REDIS_KEY_PREFIX,
                    token_ttl_seconds=_microsoft_token_ttl,
                    tenant_id=_microsoft_tenant_id,
                    services=["outlook", "onedrive"],
                )
                logger.info("Microsoft 365 MCP client initialized")
            except Exception as _ms_init_err:
                logger.warning("Microsoft 365 MCP client init failed (non-fatal): %s", _ms_init_err)
                _microsoft_mcp_client = None
        else:
            logger.warning(
                "Microsoft 365 MCP client skipped: Redis is required for token storage but is unavailable"
            )
    elif _microsoft_enabled:
        logger.warning(
            "Microsoft MCP enabled but MICROSOFT_OAUTH_CLIENT_ID / MICROSOFT_OAUTH_CLIENT_SECRET / "
            "MICROSOFT_OAUTH_REDIRECT_URI are not set — Microsoft 365 MCP disabled"
        )

    try:
        yield
    finally:
        logger.info("Shutting down Multi-KB server")
        if _llm_stuck_watchdog_task is not None:
            _llm_stuck_watchdog_task.cancel()
            await asyncio.gather(_llm_stuck_watchdog_task, return_exceptions=True)
            _llm_stuck_watchdog_task = None
        await _stop_async_query_jobs_runtime()
        await _stop_ownify_jobs_runtime()
        if query_executor is not None:
            query_executor.shutdown(wait=True, cancel_futures=False)
            query_executor = None
        if index_executor is not None:
            index_executor.shutdown(wait=True, cancel_futures=False)
            index_executor = None
        _distributed_session_lock_backend = None
        _redis_rate_limiter = None
        _redis_connection = None


# =============================================================================
# Concurrency Helpers
# =============================================================================


def _kb_session_key(kb_id: str, session_id: Optional[str]) -> Optional[str]:
    if not session_id:
        return None
    return f"{kb_id}:{session_id}"


async def get_session_lock(kb_id: str, session_id: str) -> asyncio.Lock:
    """
    Get/create per-session lock with double-checked guard.
    """
    key = _kb_session_key(kb_id, session_id)
    if key is None:
        raise ValueError("session_id is required for session lock")

    if key not in _session_locks:
        async with _locks_meta_lock:
            if key not in _session_locks:
                _session_locks[key] = asyncio.Lock()
    return _session_locks[key]


def cleanup_session_lock(kb_id: str, session_id: str) -> None:
    key = _kb_session_key(kb_id, session_id)
    if key:
        _session_locks.pop(key, None)


async def _get_kb_gate(kb_id: str) -> KbRwGate:
    if kb_id not in _kb_rw_gates:
        async with _kb_rw_gates_lock:
            if kb_id not in _kb_rw_gates:
                _kb_rw_gates[kb_id] = KbRwGate()
    return _kb_rw_gates[kb_id]


class ReadGuardTimeoutError(TimeoutError):
    """Raised when read guard cannot be acquired within timeout."""


@asynccontextmanager
async def query_read_guard(kb_id: str, timeout_seconds: Optional[float] = None):
    """Readers side of query/admin RW gate."""
    gate = await _get_kb_gate(kb_id)
    loop = asyncio.get_running_loop()
    deadline = None if timeout_seconds is None else (loop.time() + max(0.0, float(timeout_seconds)))
    async with gate.condition:
        while gate.writer_active:
            if deadline is None:
                await gate.condition.wait()
                continue

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ReadGuardTimeoutError("Timed out waiting for indexing write lock to release")
            try:
                await asyncio.wait_for(gate.condition.wait(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise ReadGuardTimeoutError("Timed out waiting for indexing write lock to release") from exc
        gate.active_readers += 1
    try:
        yield
    finally:
        async with gate.condition:
            gate.active_readers -= 1
            if gate.active_readers == 0:
                gate.condition.notify_all()


@asynccontextmanager
async def admin_write_guard(kb_id: str, timeout_seconds: float = ADMIN_WRITE_TIMEOUT_SECONDS):
    """Writers side of query/admin RW gate with timeout."""
    gate = await _get_kb_gate(kb_id)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    async with gate.condition:
        while gate.writer_active or gate.active_readers > 0:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for active queries to drain")
            await asyncio.wait_for(gate.condition.wait(), timeout=remaining)
        gate.writer_active = True

    try:
        yield
    finally:
        async with gate.condition:
            gate.writer_active = False
            gate.condition.notify_all()


async def run_in_named_executor(
        executor: Optional[ThreadPoolExecutor],
        func: Callable[..., Any],
        *args,
        timeout_seconds: Optional[float] = None,
        **kwargs
) -> Any:
    """Run sync callable in selected executor with optional timeout."""
    if executor is None:
        raise RuntimeError("Executor not initialized")

    loop = asyncio.get_running_loop()
    call = functools.partial(func, *args, **kwargs)
    future = loop.run_in_executor(executor, call)
    if timeout_seconds is not None:
        return await asyncio.wait_for(future, timeout=timeout_seconds)
    return await future


@asynccontextmanager
async def query_admission_guard(request_id: str, session_id: Optional[str]):
    """Bound in-flight query work and reject overload with explicit backpressure."""
    global _query_inflight, _query_waiters, _query_rejections, _query_peak_inflight

    acquired = False
    async with _query_admission_state_lock:
        _query_waiters += 1

    try:
        try:
            await asyncio.wait_for(
                _query_admission_semaphore.acquire(),
                timeout=QUERY_ADMISSION_TIMEOUT_SECONDS
            )
            acquired = True
        except asyncio.TimeoutError:
            async with _query_admission_state_lock:
                _query_rejections += 1
                inflight_snapshot = _query_inflight
                waiters_snapshot = max(0, _query_waiters - 1)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "query_overloaded",
                    "message": "Server is busy. Please retry shortly.",
                    "request_id": request_id,
                    "session_id": session_id,
                    "max_inflight": QUERY_MAX_INFLIGHT,
                    "inflight": inflight_snapshot,
                    "waiters": waiters_snapshot,
                },
                headers={"Retry-After": str(QUERY_RETRY_AFTER_SECONDS)}
            )
    finally:
        async with _query_admission_state_lock:
            _query_waiters = max(0, _query_waiters - 1)

    async with _query_admission_state_lock:
        _query_inflight += 1
        if _query_inflight > _query_peak_inflight:
            _query_peak_inflight = _query_inflight

    try:
        yield
    finally:
        if acquired:
            async with _query_admission_state_lock:
                _query_inflight = max(0, _query_inflight - 1)
            _query_admission_semaphore.release()


@asynccontextmanager
async def session_turn_guard(session_lock: asyncio.Lock, request_id: str, session_id: str):
    """
    Bound wait time on per-session lock so a single hot session cannot build unbounded waiters.
    """
    global _session_lock_rejections
    acquired = False
    try:
        await asyncio.wait_for(session_lock.acquire(), timeout=SESSION_LOCK_TIMEOUT_SECONDS)
        acquired = True
    except asyncio.TimeoutError:
        async with _query_admission_state_lock:
            _session_lock_rejections += 1
        raise HTTPException(
            status_code=429,
            detail={
                "error": "session_busy",
                "message": "A previous request for this session is still running. Please retry shortly.",
                "request_id": request_id,
                "session_id": session_id,
                "session_lock_timeout_seconds": SESSION_LOCK_TIMEOUT_SECONDS,
            },
            headers={"Retry-After": str(SESSION_LOCK_RETRY_AFTER_SECONDS)}
        )

    try:
        yield
    finally:
        if acquired:
            session_lock.release()


@asynccontextmanager
async def distributed_session_turn_guard(session_id: str, request_id: str):
    """
    Redis-backed same-session lock.
    """
    global _session_lock_rejections

    if _distributed_session_lock_backend is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "lock_backend_unavailable",
                "message": "Session lock backend is not initialized",
                "request_id": request_id,
                "session_id": session_id,
            },
        )

    owner_token = f"{request_id}:{uuid.uuid4().hex}"
    try:
        acquired = await asyncio.to_thread(
            _distributed_session_lock_backend.acquire,
            session_id,
            owner_token,
            SESSION_LOCK_TIMEOUT_SECONDS,
        )
    except RedisBackendUnavailable as exc:
        logger.error("Distributed lock acquisition failed for session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "lock_backend_unavailable",
                "message": "Session lock backend unavailable",
                "request_id": request_id,
                "session_id": session_id,
            },
        )

    if not acquired:
        async with _query_admission_state_lock:
            _session_lock_rejections += 1
        raise HTTPException(
            status_code=429,
            detail={
                "error": "session_busy",
                "message": "A previous request for this session is still running. Please retry shortly.",
                "request_id": request_id,
                "session_id": session_id,
                "session_lock_timeout_seconds": SESSION_LOCK_TIMEOUT_SECONDS,
            },
            headers={"Retry-After": str(SESSION_LOCK_RETRY_AFTER_SECONDS)},
        )

    try:
        yield
    finally:
        try:
            await asyncio.to_thread(
                _distributed_session_lock_backend.release,
                session_id,
                owner_token,
            )
        except RedisBackendUnavailable:
            logger.warning("Distributed lock release failed for session %s", session_id, exc_info=True)


@asynccontextmanager
async def redis_rate_limit_guard(
        request_id: str,
        session_id: Optional[str],
        user_id: Optional[str],
):
    """
    Redis-backed per-session/per-user in-flight limiter.

    Failure policy:
    - limiter backend unavailable -> fail-open (request proceeds)
    """
    global _rate_limit_rejections

    if not (REDIS_ENABLED and REDIS_RATE_LIMIT_ENABLED and _redis_rate_limiter is not None):
        yield
        return

    decision: RateLimitDecision = await asyncio.to_thread(
        _redis_rate_limiter.acquire,
        session_id=session_id,
        user_id=user_id,
        max_inflight_per_session=SESSION_MAX_INFLIGHT_REDIS,
        max_inflight_per_user=USER_MAX_INFLIGHT_REDIS,
        retry_after_seconds=RATE_LIMIT_RETRY_AFTER_SECONDS,
    )

    if not decision.allowed:
        async with _query_admission_state_lock:
            _rate_limit_rejections += 1
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "reason": decision.reason,
                "message": "Rate limit exceeded. Please retry shortly.",
                "request_id": request_id,
                "session_id": session_id,
                "user_id": user_id,
                "session_inflight": decision.session_inflight,
                "user_inflight": decision.user_inflight,
            },
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )

    try:
        yield
    finally:
        await asyncio.to_thread(
            _redis_rate_limiter.release,
            session_id=session_id,
            user_id=user_id,
        )


def _query_admission_metrics() -> Dict[str, Any]:
    """Best-effort snapshot of query admission state."""
    return {
        "max_inflight": QUERY_MAX_INFLIGHT,
        "inflight": _query_inflight,
        "waiters": _query_waiters,
        "rejections": _query_rejections,
        "peak_inflight": _query_peak_inflight,
        "admission_timeout_seconds": QUERY_ADMISSION_TIMEOUT_SECONDS,
    }


def _session_lock_metrics() -> Dict[str, Any]:
    return {
        "backend": "redis" if _distributed_session_lock_backend is not None else "in_process",
        "rejections": _session_lock_rejections,
        "active_locks": len(_session_locks),
        "session_lock_timeout_seconds": SESSION_LOCK_TIMEOUT_SECONDS,
    }


def _rate_limit_metrics() -> Dict[str, Any]:
    return {
        "enabled": bool(REDIS_ENABLED and REDIS_RATE_LIMIT_ENABLED and _redis_rate_limiter is not None),
        "rejections": _rate_limit_rejections,
        "session_max_inflight": SESSION_MAX_INFLIGHT_REDIS,
        "user_max_inflight": USER_MAX_INFLIGHT_REDIS,
    }


def _indexing_gate_metrics() -> Dict[str, Any]:
    return {
        "enabled": True,
        "rejections": _indexing_gate_rejections,
        "indexing_mode": INDEXING_MODE,
    }


def _async_query_job_metrics() -> Dict[str, Any]:
    queue_depth = _async_query_job_queue.qsize() if _async_query_job_queue is not None else 0
    status_counts = {state: 0 for state in QUERY_JOB_TERMINAL_STATES | {QUERY_JOB_STATUS_QUEUED, QUERY_JOB_STATUS_RUNNING, QUERY_JOB_STATUS_CANCELLING}}
    for job in _async_query_jobs.values():
        status_counts[job.status] = status_counts.get(job.status, 0) + 1
    return {
        "queue_depth": queue_depth,
        "queue_max_size": ASYNC_QUERY_JOB_QUEUE_MAX_SIZE,
        "jobs_total": len(_async_query_jobs),
        "status_counts": status_counts,
    }


def _async_query_jobs_runtime_ready() -> bool:
    if _async_query_job_queue is None and not _async_query_job_workers:
        return not ASYNC_QUERY_JOBS_ENABLED
    if _async_query_job_queue is None:
        return False
    if len(_async_query_job_workers) != ASYNC_QUERY_JOB_WORKERS:
        return False
    return all(not worker.done() for worker in _async_query_job_workers)


async def _record_indexing_rejection() -> None:
    global _indexing_gate_rejections
    async with _query_admission_state_lock:
        _indexing_gate_rejections += 1


def _indexing_in_progress_http_exception(
        request_id: str,
        operation: str,
        kb_id: Optional[str] = None,
        session_id: Optional[str] = None,
) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "error": "indexing_in_progress",
            "message": f"{operation} blocked while indexing is running",
            "request_id": request_id,
            "kb_id": kb_id,
            "session_id": session_id,
            "indexing_mode": INDEXING_MODE,
        },
        headers={"Retry-After": str(INDEXING_RETRY_AFTER_SECONDS)},
    )


async def _execute_query_request_internal(
        kb_id: str,
        request: QueryRequest,
        *,
        request_id: str,
        session_id: Optional[str],
        user_id: Optional[str],
        cancel_event: threading.Event,
        timeout_seconds: float,
) -> QueryResponse:
    if multi_kb_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    start_time = time.time()
    composite_session_id = _kb_session_key(kb_id, session_id)

    mcp_context, mcp_service, mcp_tool = await multi_kb_pipeline.resolve_mcp_context_async(
        kb_id=kb_id,
        query=request.query,
        user_id=user_id,
        connector=request.connector,
        session_id=session_id,
        google_file_id=request.google_file_id,
        google_file_name=request.google_file_name,
        google_calendar_id=request.google_calendar_id,
        google_calendar_name=request.google_calendar_name,
        gmail_location=request.gmail_location,
        gmail_category=request.gmail_category,
        outlook_folder=request.outlook_folder,
        outlook_location=request.outlook_location,
        outlook_message_id=request.outlook_message_id,
        microsoft_file_id=request.microsoft_file_id,
        microsoft_file_name=request.microsoft_file_name,
        microsoft_drive_path=request.microsoft_drive_path,
    )
    try:
        if session_id:
            if REDIS_ENABLED and REDIS_LOCK_ENABLED and _distributed_session_lock_backend is not None:
                async with distributed_session_turn_guard(
                        session_id=composite_session_id,
                        request_id=request_id
                ):
                    async with redis_rate_limit_guard(
                            request_id=request_id,
                            session_id=composite_session_id,
                            user_id=user_id,
                    ):
                        async with query_admission_guard(request_id=request_id, session_id=session_id):
                            async with query_read_guard(kb_id, timeout_seconds=QUERY_READ_GUARD_TIMEOUT_SECONDS):
                                result = await run_in_named_executor(
                                    query_executor,
                                    multi_kb_pipeline.query,
                                    kb_id,
                                    request.query,
                                    session_id,
                                    user_id,
                                    cancel_event,
                                    timeout_seconds=timeout_seconds,
                                    web=request.web, kb=request.kb,
                                    connector=request.connector,
                                    mcp_context=mcp_context,
                                    mcp_service=mcp_service,
                                    mcp_tool=mcp_tool,
                                    google_file_id=request.google_file_id,
                                    google_file_name=request.google_file_name,
                                    google_calendar_id=request.google_calendar_id,
                                    google_calendar_name=request.google_calendar_name,
                                    gmail_location=request.gmail_location,
                                    gmail_category=request.gmail_category,
                                    outlook_folder=request.outlook_folder,
                                    outlook_location=request.outlook_location,
                                    outlook_message_id=request.outlook_message_id,
                                    microsoft_file_id=request.microsoft_file_id,
                                    microsoft_file_name=request.microsoft_file_name,
                                    microsoft_drive_path=request.microsoft_drive_path,
                                )
            else:
                session_lock = await get_session_lock(kb_id, session_id)
                async with session_turn_guard(
                        session_lock=session_lock,
                        request_id=request_id,
                        session_id=session_id
                ):
                    async with redis_rate_limit_guard(
                            request_id=request_id,
                            session_id=composite_session_id,
                            user_id=user_id,
                    ):
                        async with query_admission_guard(request_id=request_id, session_id=session_id):
                            async with query_read_guard(kb_id, timeout_seconds=QUERY_READ_GUARD_TIMEOUT_SECONDS):
                                result = await run_in_named_executor(
                                    query_executor,
                                    multi_kb_pipeline.query,
                                    kb_id,
                                    request.query,
                                    session_id,
                                    user_id,
                                    cancel_event,
                                    timeout_seconds=timeout_seconds,
                                    web=request.web, kb=request.kb,
                                    connector=request.connector,
                                    mcp_context=mcp_context,
                                    mcp_service=mcp_service,
                                    mcp_tool=mcp_tool,
                                    google_file_id=request.google_file_id,
                                    google_file_name=request.google_file_name,
                                    google_calendar_id=request.google_calendar_id,
                                    google_calendar_name=request.google_calendar_name,
                                    gmail_location=request.gmail_location,
                                    gmail_category=request.gmail_category,
                                    outlook_folder=request.outlook_folder,
                                    outlook_location=request.outlook_location,
                                    outlook_message_id=request.outlook_message_id,
                                    microsoft_file_id=request.microsoft_file_id,
                                    microsoft_file_name=request.microsoft_file_name,
                                    microsoft_drive_path=request.microsoft_drive_path,
                                )
        else:
            async with redis_rate_limit_guard(
                    request_id=request_id,
                    session_id=composite_session_id,
                    user_id=user_id,
            ):
                async with query_admission_guard(request_id=request_id, session_id=session_id):
                    async with query_read_guard(kb_id, timeout_seconds=QUERY_READ_GUARD_TIMEOUT_SECONDS):
                        result = await run_in_named_executor(
                            query_executor,
                            multi_kb_pipeline.query,
                            kb_id,
                            request.query,
                            session_id,
                            user_id,
                            cancel_event,
                            timeout_seconds=timeout_seconds,
                            web=request.web,
                            kb=request.kb,
                            connector=request.connector,
                            mcp_context=mcp_context,
                            mcp_service=mcp_service,
                            mcp_tool=mcp_tool,
                            google_file_id=request.google_file_id,
                            google_file_name=request.google_file_name,
                            google_calendar_id=request.google_calendar_id,
                            google_calendar_name=request.google_calendar_name,
                            gmail_location=request.gmail_location,
                            gmail_category=request.gmail_category,
                            outlook_folder=request.outlook_folder,
                            outlook_location=request.outlook_location,
                            outlook_message_id=request.outlook_message_id,
                            microsoft_file_id=request.microsoft_file_id,
                            microsoft_file_name=request.microsoft_file_name,
                            microsoft_drive_path=request.microsoft_drive_path,
                        )
    except ReadGuardTimeoutError:
        await _record_indexing_rejection()
        raise _indexing_in_progress_http_exception(
            request_id=request_id,
            operation="query",
            kb_id=kb_id,
            session_id=session_id,
        )

    processing_time = time.time() - start_time

    citations = _normalize_citations_sources(result.citations)
    response = QueryResponse(
        status="success",
        query=result.query,
        answer=result.answer,
        citations=citations,
        metadata={
            "confidence": result.metadata.get('confidence', 0.0),
            "query_type": result.query_classification.get('query_type'),
            "documents_used": result.context_stats.get('final', 0),
            "total_time": result.total_time,
            "knowledge_base": kb_id,
            "request_id": request_id,
            "user_id": user_id,
            "extracted_entities": result.extracted_entities,
            "reformulation_method": result.reformulation_method if result.was_reformulated else None,
            "memory_stats": result.memory_stats,
            "retrieval_stats": result.retrieval_stats,
        },
        processing_time=processing_time,
        session_id=result.session_id,
        turn_number=result.turn_number,
        was_reformulated=result.was_reformulated,
        reformulated_query=result.reformulated_query if result.was_reformulated else None
    )
    return response


def _job_status_payload(job: AsyncQueryJobRecord) -> Dict[str, Any]:
    result_payload = normalize_result_citations(job.result)

    return {
        "job_id": job.job_id,
        "kb_id": job.kb_id,
        "job_status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "query": job.request.query,
        "session_id": job.session_id,
        "user_id": job.user_id,
        "request_id": job.request_id,
        "timeout_seconds": job.timeout_seconds,
        "attempts": job.attempts,
        "cancellation_requested": job.cancellation_requested,
        "is_terminal": job.status in QUERY_JOB_TERMINAL_STATES,
        "retry_after_seconds": job.last_retry_after_seconds,
        "result": result_payload,
        "error": job.error,
    }


async def _set_job_status(
        job_id: str,
        *,
        status: Optional[str] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        attempts: Optional[int] = None,
        cancellation_requested: Optional[bool] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        last_retry_after_seconds: Optional[float] = None,
) -> Optional[AsyncQueryJobRecord]:
    async with _async_query_jobs_lock:
        job = _async_query_jobs.get(job_id)
        if job is None:
            return None
        if status is not None:
            job.status = status
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at
        if attempts is not None:
            job.attempts = attempts
        if cancellation_requested is not None:
            job.cancellation_requested = cancellation_requested
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        job.last_retry_after_seconds = last_retry_after_seconds
        job.updated_at = time.time()
        return job


async def _get_job(job_id: str) -> Optional[AsyncQueryJobRecord]:
    async with _async_query_jobs_lock:
        return _async_query_jobs.get(job_id)


async def _remove_expired_jobs() -> int:
    now = time.time()
    removed = 0
    async with _async_query_jobs_lock:
        stale_ids = [
            job_id
            for job_id, job in _async_query_jobs.items()
            if job.status in QUERY_JOB_TERMINAL_STATES and (now - job.updated_at) >= ASYNC_QUERY_JOB_RETENTION_SECONDS
        ]
        for job_id in stale_ids:
            _async_query_jobs.pop(job_id, None)
            removed += 1
    return removed


async def _async_query_job_cleanup_loop() -> None:
    try:
        while True:
            await asyncio.sleep(ASYNC_QUERY_JOB_CLEANUP_INTERVAL_SECONDS)
            removed = await _remove_expired_jobs()
            if removed > 0:
                logger.info("Async query job cleanup removed %s expired jobs", removed)
    except asyncio.CancelledError:
        logger.info("Async query job cleanup loop stopped")
        raise


async def _run_single_async_query_job(job_id: str) -> None:
    job = await _get_job(job_id)
    if job is None:
        return

    if job.status in QUERY_JOB_TERMINAL_STATES:
        return

    if job.cancel_event.is_set():
        await _set_job_status(
            job_id,
            status=QUERY_JOB_STATUS_CANCELLED,
            cancellation_requested=True,
            completed_at=time.time(),
            error={"error": "job_cancelled", "message": "Job cancelled before execution started"},
        )
        return

    await _set_job_status(
        job_id,
        status=QUERY_JOB_STATUS_RUNNING,
        started_at=time.time(),
    )

    deadline = job.created_at + job.timeout_seconds
    attempts = job.attempts

    while True:
        attempts += 1
        await _set_job_status(job_id, attempts=attempts)

        if job.cancel_event.is_set():
            await _set_job_status(
                job_id,
                status=QUERY_JOB_STATUS_CANCELLED,
                cancellation_requested=True,
                completed_at=time.time(),
                error={"error": "job_cancelled", "message": "Job cancelled before execution"},
            )
            return

        remaining_seconds = deadline - time.time()
        if remaining_seconds <= 0:
            await _set_job_status(
                job_id,
                status=QUERY_JOB_STATUS_TIMED_OUT,
                completed_at=time.time(),
                error={
                    "error": "query_timeout",
                    "message": f"Async job timed out after {job.timeout_seconds:.1f}s",
                    "request_id": job.request_id,
                    "session_id": job.session_id,
                    "timeout_seconds": job.timeout_seconds,
                },
            )
            return

        try:
            response = await _execute_query_request_internal(
                job.kb_id,
                QueryRequest(
                    query=job.request.query,
                    session_id=job.session_id,
                    user_id=job.user_id,
                    top_k=job.request.top_k,
                    stream=job.request.stream,
                    web=job.request.web,
                    kb=job.request.kb,
                    connector=getattr(job.request, "connector", None),
                    google_file_id=getattr(job.request, "google_file_id", None),
                    google_file_name=getattr(job.request, "google_file_name", None),
                    google_calendar_id=getattr(job.request, "google_calendar_id", None),
                    google_calendar_name=getattr(job.request, "google_calendar_name", None),
                    gmail_location=getattr(job.request, "gmail_location", None),
                    gmail_category=getattr(job.request, "gmail_category", None),
                    outlook_folder=getattr(job.request, "outlook_folder", None),
                    outlook_location=getattr(job.request, "outlook_location", None),
                    outlook_message_id=getattr(job.request, "outlook_message_id", None),
                    microsoft_file_id=getattr(job.request, "microsoft_file_id", None),
                    microsoft_file_name=getattr(job.request, "microsoft_file_name", None),
                    microsoft_drive_path=getattr(job.request, "microsoft_drive_path", None),
                ),
                request_id=job.request_id,
                session_id=job.session_id,
                user_id=job.user_id,
                cancel_event=job.cancel_event,
                timeout_seconds=min(job.timeout_seconds, max(1.0, remaining_seconds)),
            )
            await _set_job_status(
                job_id,
                status=QUERY_JOB_STATUS_SUCCEEDED,
                completed_at=time.time(),
                result=response.dict(),
            )
            return
        except asyncio.TimeoutError:
            job.cancel_event.set()
            await _set_job_status(
                job_id,
                status=QUERY_JOB_STATUS_TIMED_OUT,
                completed_at=time.time(),
                error={
                    "error": "query_timeout",
                    "message": f"Query timed out after {min(job.timeout_seconds, max(1.0, remaining_seconds)):.1f}s",
                    "request_id": job.request_id,
                    "session_id": job.session_id,
                    "timeout_seconds": min(job.timeout_seconds, max(1.0, remaining_seconds)),
                },
            )
            return
        except PipelineCancelledError:
            await _set_job_status(
                job_id,
                status=QUERY_JOB_STATUS_CANCELLED,
                cancellation_requested=True,
                completed_at=time.time(),
                error={"error": "job_cancelled", "message": "Query job cancelled"},
            )
            return
        except PipelineOverloadedError as exc:
            retry_after_seconds = max(
                ASYNC_QUERY_JOB_MIN_RETRY_DELAY_SECONDS,
                min(float(exc.retry_after_seconds), ASYNC_QUERY_JOB_MAX_RETRY_DELAY_SECONDS),
            )
            if time.time() + retry_after_seconds < deadline:
                await _set_job_status(
                    job_id,
                    status=QUERY_JOB_STATUS_RUNNING,
                    error={
                        "error": getattr(exc, "reason", "pipeline_overloaded"),
                        "message": str(exc),
                        "request_id": job.request_id,
                        "session_id": job.session_id,
                        "retry_after_seconds": retry_after_seconds,
                    },
                    last_retry_after_seconds=retry_after_seconds,
                )
                await asyncio.sleep(retry_after_seconds)
                continue
            await _set_job_status(
                job_id,
                status=QUERY_JOB_STATUS_FAILED,
                completed_at=time.time(),
                error={
                    "error": getattr(exc, "reason", "pipeline_overloaded"),
                    "message": str(exc),
                    "request_id": job.request_id,
                    "session_id": job.session_id,
                    "retry_after_seconds": retry_after_seconds,
                },
            )
            return
        except HTTPException as exc:
            error_payload = exc.detail if isinstance(exc.detail, dict) else {"error": "http_error", "message": str(exc)}
            if exc.status_code in {409, 429}:
                retry_after_seconds = int(exc.headers.get("Retry-After", "1")) if exc.headers else 1
                if time.time() + retry_after_seconds < deadline and not job.cancel_event.is_set():
                    await _set_job_status(
                        job_id,
                        status=QUERY_JOB_STATUS_RUNNING,
                        error=error_payload,
                        last_retry_after_seconds=retry_after_seconds,
                    )
                    await asyncio.sleep(retry_after_seconds)
                    continue

            status = QUERY_JOB_STATUS_TIMED_OUT if error_payload.get(
                "error") == "query_timeout" else QUERY_JOB_STATUS_FAILED
            await _set_job_status(
                job_id,
                status=status,
                completed_at=time.time(),
                error=error_payload,
            )
            return
        except Exception as exc:
            await _set_job_status(
                job_id,
                status=QUERY_JOB_STATUS_FAILED,
                completed_at=time.time(),
                error={
                    "error": "internal_error",
                    "message": str(exc),
                    "request_id": job.request_id,
                    "session_id": job.session_id,
                },
            )
            return


async def _async_query_job_worker(worker_index: int) -> None:
    if _async_query_job_queue is None:
        return

    logger.info("Async query job worker %s started", worker_index)
    try:
        while True:
            job_id = await _async_query_job_queue.get()
            try:
                await _run_single_async_query_job(job_id)
            finally:
                _async_query_job_queue.task_done()
    except asyncio.CancelledError:
        logger.info("Async query job worker %s stopped", worker_index)
        raise


async def _start_async_query_jobs_runtime() -> None:
    global _async_query_job_queue, _async_query_job_workers, _async_query_job_cleanup_task

    if not ASYNC_QUERY_JOBS_ENABLED:
        logger.info("Async query jobs are disabled")
        return

    if _async_query_job_queue is not None:
        return

    _async_query_job_queue = asyncio.Queue(maxsize=ASYNC_QUERY_JOB_QUEUE_MAX_SIZE)
    _async_query_job_workers = [
        asyncio.create_task(_async_query_job_worker(worker_index=i + 1))
        for i in range(ASYNC_QUERY_JOB_WORKERS)
    ]
    _async_query_job_cleanup_task = asyncio.create_task(_async_query_job_cleanup_loop())
    logger.info(
        "Async query jobs runtime initialized (workers=%s, queue_max_size=%s, default_timeout=%.1fs)",
        ASYNC_QUERY_JOB_WORKERS,
        ASYNC_QUERY_JOB_QUEUE_MAX_SIZE,
        ASYNC_QUERY_JOB_DEFAULT_TIMEOUT_SECONDS,
    )


async def _stop_async_query_jobs_runtime() -> None:
    global _async_query_job_queue, _async_query_job_workers, _async_query_job_cleanup_task

    async with _async_query_jobs_lock:
        for job in _async_query_jobs.values():
            if job.status not in QUERY_JOB_TERMINAL_STATES:
                job.cancellation_requested = True
                job.cancel_event.set()
                if job.status == QUERY_JOB_STATUS_QUEUED:
                    job.status = QUERY_JOB_STATUS_CANCELLED
                    job.completed_at = time.time()
                    job.error = {
                        "error": "job_cancelled",
                        "message": "Server shutting down",
                    }
                elif job.status == QUERY_JOB_STATUS_RUNNING:
                    job.status = QUERY_JOB_STATUS_CANCELLING
                job.updated_at = time.time()

    for task in _async_query_job_workers:
        task.cancel()
    if _async_query_job_workers:
        await asyncio.gather(*_async_query_job_workers, return_exceptions=True)
    _async_query_job_workers = []

    if _async_query_job_cleanup_task is not None:
        _async_query_job_cleanup_task.cancel()
        await asyncio.gather(_async_query_job_cleanup_task, return_exceptions=True)
        _async_query_job_cleanup_task = None

    _async_query_job_queue = None


def _utc_iso_from_timestamp(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None
    return datetime.utcfromtimestamp(timestamp).isoformat() + "Z"


def _model_payload(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if item is not None}
    return {}


def _request_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ownify_kb_id(tenant_id: str) -> str:
    return tenant_id


async def require_ownify_provisioning_auth(request: Request) -> None:
    if not OWNIFY_PROVISIONING_API_KEY:
        return

    authorization = request.headers.get("authorization", "")
    token = ""
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        token = request.headers.get("x-ownify-api-key", "").strip()

    if not token or not hmac.compare_digest(token, OWNIFY_PROVISIONING_API_KEY):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "Invalid Ownify provisioning credentials",
            },
        )


def _ownify_store() -> ProvisioningJobStore:
    if _ownify_job_store is None:
        raise HTTPException(status_code=503, detail="Ownify provisioning job store not initialized")
    return _ownify_job_store


async def _get_ownify_tenant_lock(tenant_id: str) -> asyncio.Lock:
    if tenant_id not in _ownify_tenant_locks:
        async with _ownify_tenant_locks_lock:
            if tenant_id not in _ownify_tenant_locks:
                _ownify_tenant_locks[tenant_id] = asyncio.Lock()
    return _ownify_tenant_locks[tenant_id]


def _ownify_status_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "tenant_id": job["tenant_id"],
        "kb_id": job["kb_id"],
        "job_type": job.get("job_type", OWNIFY_JOB_TYPE_PROVISION),
        "job_status": job.get("job_status", OWNIFY_JOB_STATUS_QUEUED),
        "phase": job.get("phase", OWNIFY_PHASE_QUEUED),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "request_id": job.get("request_id"),
        "idempotency_key": job.get("idempotency_key"),
        "timeout_seconds": job.get("timeout_seconds"),
        "attempts": int(job.get("attempts") or 0),
        "cancellation_requested": bool(job.get("cancellation_requested", False)),
        "is_terminal": job.get("job_status") in OWNIFY_JOB_TERMINAL_STATES,
        "request_summary": job.get("request_summary") or {},
        "kb": job.get("kb"),
        "documents": job.get("documents") or [],
        "result": job.get("result"),
        "error": job.get("error"),
    }


async def _set_ownify_job(job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    store = _ownify_store()
    updates["updated_at"] = time.time()
    return store.update(job_id, **updates)


def _ownify_terminal_response(job: Dict[str, Any]) -> OwnifyJobAcceptedResponse:
    return OwnifyJobAcceptedResponse(
        status="accepted",
        job_id=job["job_id"],
        tenant_id=job["tenant_id"],
        kb_id=job["kb_id"],
        request_id=job["request_id"],
        job_status=job.get("job_status", OWNIFY_JOB_STATUS_QUEUED),
        queued_at=_utc_iso_from_timestamp(job.get("created_at")) or datetime.utcnow().isoformat() + "Z",
        timeout_seconds=float(job.get("timeout_seconds") or PROVISIONING_JOB_DEFAULT_TIMEOUT_SECONDS),
    )


def _ownify_operation_status(job_status: str) -> str:
    if job_status == OWNIFY_JOB_STATUS_SUCCEEDED:
        return "success"
    if job_status == OWNIFY_JOB_STATUS_SUCCEEDED_WITH_ERRORS:
        return "succeeded_with_errors"
    return job_status or OWNIFY_JOB_STATUS_FAILED


def _ownify_documents_operation_response(job: Dict[str, Any]) -> OwnifyDocumentsOperationResponse:
    job_status = job.get("job_status", OWNIFY_JOB_STATUS_FAILED)
    return OwnifyDocumentsOperationResponse(
        status=_ownify_operation_status(job_status),
        job_id=job["job_id"],
        tenant_id=job["tenant_id"],
        kb_id=job["kb_id"],
        request_id=job["request_id"],
        job_status=job_status,
        phase=job.get("phase", OWNIFY_PHASE_FINALIZING),
        queued_at=_utc_iso_from_timestamp(job.get("created_at")) or datetime.utcnow().isoformat() + "Z",
        started_at=_utc_iso_from_timestamp(job.get("started_at")),
        completed_at=_utc_iso_from_timestamp(job.get("completed_at")),
        timeout_seconds=float(job.get("timeout_seconds") or PROVISIONING_JOB_DEFAULT_TIMEOUT_SECONDS),
        documents=job.get("documents") or [],
        result=job.get("result"),
        error=job.get("error"),
    )


def _ownify_document_delete_response(job: Dict[str, Any]) -> OwnifyDocumentDeleteResponse:
    result = job.get("result") or {}
    delete_result = result.get("delete") or {}
    documents = job.get("documents") or []
    document = documents[0] if documents else {}
    file_id = str(result.get("file_id") or document.get("file_id") or delete_result.get("file_id") or "")
    file_name = result.get("file_name") or document.get("file_name") or delete_result.get("file_name")
    job_status = job.get("job_status", OWNIFY_JOB_STATUS_FAILED)
    return OwnifyDocumentDeleteResponse(
        status=_ownify_operation_status(job_status),
        job_id=job["job_id"],
        tenant_id=job["tenant_id"],
        kb_id=job["kb_id"],
        request_id=job["request_id"],
        job_status=job_status,
        phase=job.get("phase", OWNIFY_PHASE_FINALIZING),
        queued_at=_utc_iso_from_timestamp(job.get("created_at")) or datetime.utcnow().isoformat() + "Z",
        started_at=_utc_iso_from_timestamp(job.get("started_at")),
        completed_at=_utc_iso_from_timestamp(job.get("completed_at")),
        timeout_seconds=float(job.get("timeout_seconds") or PROVISIONING_JOB_DEFAULT_TIMEOUT_SECONDS),
        file_id=file_id,
        file_name=file_name,
        deleted_count=int(result.get("deleted_count") or delete_result.get("deleted_count") or 0),
        result=delete_result or result,
        error=job.get("error"),
    )


def _ownify_tenant_delete_response(job: Dict[str, Any]) -> OwnifyTenantDeleteResponse:
    job_status = job.get("job_status", OWNIFY_JOB_STATUS_FAILED)
    return OwnifyTenantDeleteResponse(
        status=_ownify_operation_status(job_status),
        job_id=job["job_id"],
        tenant_id=job["tenant_id"],
        kb_id=job["kb_id"],
        request_id=job["request_id"],
        job_status=job_status,
        phase=job.get("phase", OWNIFY_PHASE_FINALIZING),
        queued_at=_utc_iso_from_timestamp(job.get("created_at")) or datetime.utcnow().isoformat() + "Z",
        started_at=_utc_iso_from_timestamp(job.get("started_at")),
        completed_at=_utc_iso_from_timestamp(job.get("completed_at")),
        timeout_seconds=float(job.get("timeout_seconds") or PROVISIONING_JOB_DEFAULT_TIMEOUT_SECONDS),
        result=job.get("result") or {},
        error=job.get("error"),
    )


def _ownify_document_summary(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "file_id": document.get("file_id"),
            "file_name": document.get("file_name"),
            "status": document.get("status", "queued"),
        }
        for document in documents
    ]


async def _enqueue_ownify_job(record: Dict[str, Any]) -> None:
    if _ownify_job_queue is None:
        raise HTTPException(status_code=503, detail="Ownify provisioning queue not initialized")
    try:
        _ownify_job_queue.put_nowait(record["job_id"])
    except asyncio.QueueFull:
        _ownify_store().delete(record["job_id"])
        raise HTTPException(
            status_code=429,
            detail={
                "error": "provisioning_queue_full",
                "message": "Ownify provisioning queue is full. Please retry shortly.",
                "request_id": record["request_id"],
            },
            headers={"Retry-After": str(INDEXING_RETRY_AFTER_SECONDS)},
        )


def _remaining_job_seconds(job: Dict[str, Any]) -> float:
    deadline = float(job.get("created_at") or time.time()) + float(job.get("timeout_seconds") or 0)
    return max(0.0, deadline - time.time())


async def _run_indexing_call_with_timeout(
    job: Dict[str, Any],
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    remaining = _remaining_job_seconds(job)
    if remaining <= 0:
        raise asyncio.TimeoutError()
    return await run_in_named_executor(
        index_executor,
        func,
        *args,
        timeout_seconds=max(1.0, remaining),
        **kwargs,
    )


async def _index_ownify_document(job: Dict[str, Any], kb_id: str, document: Dict[str, Any]) -> Dict[str, Any]:
    if kb_manager is None:
        raise RuntimeError("KB manager is not initialized")

    if INDEXING_ONLINE_MODE:
        result = await _run_indexing_call_with_timeout(
            job,
            kb_manager.add_document,
            kb_id=kb_id,
            file_id=document.get("file_id"),
            file_name=document.get("file_name"),
            sas_url=document.get("sas_url"),
            local_path=document.get("local_path"),
        )
    else:
        remaining = _remaining_job_seconds(job)
        if remaining <= 0:
            raise asyncio.TimeoutError()
        async with admin_write_guard(kb_id, timeout_seconds=min(remaining, INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS)):
            result = await _run_indexing_call_with_timeout(
                job,
                kb_manager.add_document,
                kb_id=kb_id,
                file_id=document.get("file_id"),
                file_name=document.get("file_name"),
                sas_url=document.get("sas_url"),
                local_path=document.get("local_path"),
            )

    status = "succeeded" if result.get("success") else "failed"
    if result.get("success") and multi_kb_pipeline is not None:
        multi_kb_pipeline.evict_kb(kb_id)
    return {
        "file_id": result.get("file_id") or document.get("file_id"),
        "file_name": result.get("file_name") or document.get("file_name"),
        "status": status,
        "result": result,
    }


async def _delete_ownify_document(
    kb_id: str,
    file_id: str,
    file_name: Optional[str],
    timeout_seconds: float = ADMIN_WRITE_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    if kb_manager is None:
        raise RuntimeError("KB manager is not initialized")

    if INDEXING_ONLINE_MODE:
        result = await run_in_named_executor(
            index_executor,
            kb_manager.remove_document,
            kb_id,
            file_id,
            file_name,
            timeout_seconds=timeout_seconds,
        )
    else:
        async with admin_write_guard(kb_id, timeout_seconds=INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS):
            result = await run_in_named_executor(
                index_executor,
                kb_manager.remove_document,
                kb_id,
                file_id,
                file_name,
                timeout_seconds=timeout_seconds,
            )

    if multi_kb_pipeline is not None:
        multi_kb_pipeline.evict_kb(kb_id)
    return result


async def _execute_ownify_provision_job(job: Dict[str, Any]) -> None:
    if kb_manager is None:
        raise RuntimeError("KB manager is not initialized")

    payload = job.get("request_payload") or {}
    kb_id = job["kb_id"]
    tenant_id = job["tenant_id"]
    documents = list(payload.get("documents") or [])

    await _set_ownify_job(job["job_id"], phase=OWNIFY_PHASE_CREATING_KB)
    kb = await _run_indexing_call_with_timeout(
        job,
        kb_manager.create_or_get_kb,
        kb_id,
        payload["display_name"],
        payload.get("description"),
        bool(payload.get("replace_existing", False)),
    )
    if multi_kb_pipeline is not None and payload.get("replace_existing"):
        multi_kb_pipeline.evict_kb(kb_id)
    await _set_ownify_job(job["job_id"], kb=kb)

    await _set_ownify_job(job["job_id"], phase=OWNIFY_PHASE_WRITING_CONFIG)
    config_result = await _run_indexing_call_with_timeout(
        job,
        kb_manager.write_runtime_config_snapshot,
        kb_id=kb_id,
        tenant_id=tenant_id,
        display_name=payload.get("display_name"),
        description=payload.get("description"),
        system_prompt=payload.get("system_prompt"),
        ai_config=payload.get("ai_config"),
        source="ownify_provisioning",
    )
    if multi_kb_pipeline is not None:
        multi_kb_pipeline.evict_kb(kb_id)

    document_results: List[Dict[str, Any]] = []
    if documents:
        await _set_ownify_job(
            job["job_id"],
            phase=OWNIFY_PHASE_INDEXING_DOCUMENTS,
            documents=_ownify_document_summary(documents),
        )
        for index, document in enumerate(documents):
            running_docs = list(document_results)
            running_docs.append(
                {
                    "file_id": document.get("file_id"),
                    "file_name": document.get("file_name"),
                    "status": "running",
                }
            )
            for queued in documents[index + 1:]:
                running_docs.append(
                    {
                        "file_id": queued.get("file_id"),
                        "file_name": queued.get("file_name"),
                        "status": "queued",
                    }
                )
            await _set_ownify_job(job["job_id"], documents=running_docs)
            try:
                document_result = await _index_ownify_document(job, kb_id, document)
            except asyncio.TimeoutError:
                raise
            except Exception as exc:
                logger.error(
                    "Ownify provisioning document ingestion failed (tenant=%s, file_id=%s): %s",
                    tenant_id,
                    document.get("file_id"),
                    exc,
                    exc_info=True,
                )
                document_result = {
                    "file_id": document.get("file_id"),
                    "file_name": document.get("file_name"),
                    "status": "failed",
                    "result": {
                        "success": False,
                        "error_code": "indexing_failed",
                        "error": str(exc),
                    },
                }
            document_results.append(document_result)
            await _set_ownify_job(job["job_id"], documents=document_results)

    await _set_ownify_job(job["job_id"], phase=OWNIFY_PHASE_FINALIZING)
    failed_documents = [item for item in document_results if item.get("status") != "succeeded"]
    final_status = (
        OWNIFY_JOB_STATUS_SUCCEEDED_WITH_ERRORS
        if failed_documents
        else OWNIFY_JOB_STATUS_SUCCEEDED
    )
    latest_kb = kb_manager.registry.get_kb(kb_id) or config_result.get("kb") or kb
    await _set_ownify_job(
        job["job_id"],
        job_status=final_status,
        completed_at=time.time(),
        request_payload={"redacted": True},
        kb=latest_kb,
        documents=document_results,
        result={
            "kb": latest_kb,
            "config": config_result,
            "documents_total": len(document_results),
            "documents_failed": len(failed_documents),
        },
        error={
            "error": "document_ingestion_failed",
            "message": "One or more documents failed to ingest",
        } if failed_documents else None,
    )


async def _execute_ownify_documents_job(job: Dict[str, Any]) -> None:
    if kb_manager is None:
        raise RuntimeError("KB manager is not initialized")

    kb_id = job["kb_id"]
    if not kb_manager.registry.exists(kb_id):
        raise ValueError(f"KB not found: {kb_id}")

    documents = list((job.get("request_payload") or {}).get("documents") or [])
    document_results: List[Dict[str, Any]] = []
    await _set_ownify_job(
        job["job_id"],
        phase=OWNIFY_PHASE_INDEXING_DOCUMENTS,
        documents=_ownify_document_summary(documents),
    )
    for index, document in enumerate(documents):
        running_docs = list(document_results)
        running_docs.append(
            {
                "file_id": document.get("file_id"),
                "file_name": document.get("file_name"),
                "status": "running",
            }
        )
        for queued in documents[index + 1:]:
            running_docs.append(
                {
                    "file_id": queued.get("file_id"),
                    "file_name": queued.get("file_name"),
                    "status": "queued",
                }
            )
        await _set_ownify_job(job["job_id"], documents=running_docs)
        try:
            document_result = await _index_ownify_document(job, kb_id, document)
        except asyncio.TimeoutError:
            raise
        except Exception as exc:
            logger.error(
                "Ownify document ingestion failed (tenant=%s, file_id=%s): %s",
                job["tenant_id"],
                document.get("file_id"),
                exc,
                exc_info=True,
            )
            document_result = {
                "file_id": document.get("file_id"),
                "file_name": document.get("file_name"),
                "status": "failed",
                "result": {
                    "success": False,
                    "error_code": "indexing_failed",
                    "error": str(exc),
                },
            }
        document_results.append(document_result)
        await _set_ownify_job(job["job_id"], documents=document_results)

    failed_documents = [item for item in document_results if item.get("status") != "succeeded"]
    latest_kb = kb_manager.registry.get_kb(kb_id)
    await _set_ownify_job(
        job["job_id"],
        phase=OWNIFY_PHASE_FINALIZING,
        job_status=(
            OWNIFY_JOB_STATUS_SUCCEEDED_WITH_ERRORS
            if failed_documents
            else OWNIFY_JOB_STATUS_SUCCEEDED
        ),
        completed_at=time.time(),
        request_payload={"redacted": True},
        kb=latest_kb,
        documents=document_results,
        result={
            "kb": latest_kb,
            "documents_total": len(document_results),
            "documents_failed": len(failed_documents),
        },
        error={
            "error": "document_ingestion_failed",
            "message": "One or more documents failed to ingest",
        } if failed_documents else None,
    )


async def _execute_ownify_document_delete_job(job: Dict[str, Any]) -> None:
    if kb_manager is None:
        raise RuntimeError("KB manager is not initialized")

    kb_id = job["kb_id"]
    if not kb_manager.registry.exists(kb_id):
        raise ValueError(f"KB not found: {kb_id}")

    payload = job.get("request_payload") or {}
    file_id = str(payload.get("file_id") or "")
    file_name = payload.get("file_name")
    if not file_id:
        raise ValueError("file_id is required")

    await _set_ownify_job(
        job["job_id"],
        phase=OWNIFY_PHASE_DELETING_DOCUMENT,
        documents=[
            {
                "file_id": file_id,
                "file_name": file_name,
                "status": "running",
            }
        ],
    )

    result = await _delete_ownify_document(
        kb_id,
        file_id,
        file_name,
        timeout_seconds=max(1.0, _remaining_job_seconds(job)),
    )
    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Failed to delete document")

    latest_kb = kb_manager.registry.get_kb(kb_id)
    document_result = {
        "file_id": file_id,
        "file_name": file_name,
        "status": "succeeded",
        "result": result,
    }
    await _set_ownify_job(
        job["job_id"],
        phase=OWNIFY_PHASE_FINALIZING,
        job_status=OWNIFY_JOB_STATUS_SUCCEEDED,
        completed_at=time.time(),
        request_payload={"redacted": True},
        kb=latest_kb,
        documents=[document_result],
        result={
            "kb": latest_kb,
            "file_id": file_id,
            "file_name": file_name,
            "deleted_count": int(result.get("deleted_count") or 0),
            "delete": result,
        },
        error=None,
    )


async def _execute_ownify_tenant_delete_job(job: Dict[str, Any]) -> None:
    if kb_manager is None:
        raise RuntimeError("KB manager is not initialized")

    kb_id = job["kb_id"]
    tenant_id = job["tenant_id"]
    await _set_ownify_job(job["job_id"], phase=OWNIFY_PHASE_DELETING_TENANT)

    existed = kb_manager.registry.exists(kb_id)
    delete_result: Dict[str, Any]
    if existed:
        remaining = _remaining_job_seconds(job)
        if remaining <= 0:
            raise asyncio.TimeoutError()
        async with admin_write_guard(kb_id, timeout_seconds=min(remaining, INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS)):
            delete_result = await run_in_named_executor(
                index_executor,
                kb_manager.delete_kb,
                kb_id,
                timeout_seconds=max(1.0, remaining),
            )
    else:
        delete_result = {
            "status": "success",
            "kb_id": kb_id,
            "collection_name": None,
            "already_deleted": True,
        }

    if multi_kb_pipeline is not None:
        multi_kb_pipeline.evict_kb(kb_id)

    removed_job_records = _ownify_store().delete_for_tenant(
        tenant_id,
        exclude_job_id=job["job_id"],
    )

    await _set_ownify_job(
        job["job_id"],
        phase=OWNIFY_PHASE_FINALIZING,
        job_status=OWNIFY_JOB_STATUS_SUCCEEDED,
        completed_at=time.time(),
        request_payload={"redacted": True},
        kb=None,
        documents=[],
        result={
            "kb_id": kb_id,
            "tenant_id": tenant_id,
            "existed": existed,
            "delete": delete_result,
            "removed_job_records": removed_job_records,
        },
        error=None,
    )


async def _run_single_ownify_job(job_id: str) -> None:
    store = _ownify_store()
    job = store.get(job_id)
    if job is None or job.get("job_status") in OWNIFY_JOB_TERMINAL_STATES:
        return

    tenant_lock = await _get_ownify_tenant_lock(str(job.get("tenant_id") or ""))
    async with tenant_lock:
        job = store.get(job_id)
        if job is None or job.get("job_status") in OWNIFY_JOB_TERMINAL_STATES:
            return

        if job.get("cancellation_requested"):
            await _set_ownify_job(
                job_id,
                job_status=OWNIFY_JOB_STATUS_CANCELLED,
                completed_at=time.time(),
                request_payload={"redacted": True},
                error={"error": "job_cancelled", "message": "Job cancelled before execution"},
            )
            return

        attempts = int(job.get("attempts") or 0) + 1
        await _set_ownify_job(
            job_id,
            job_status=OWNIFY_JOB_STATUS_RUNNING,
            phase=OWNIFY_PHASE_VALIDATING,
            started_at=job.get("started_at") or time.time(),
            attempts=attempts,
        )
        job = store.get(job_id) or job

        try:
            if _remaining_job_seconds(job) <= 0:
                raise asyncio.TimeoutError()

            if job.get("job_type") == OWNIFY_JOB_TYPE_DOCUMENTS:
                await _execute_ownify_documents_job(job)
            elif job.get("job_type") == OWNIFY_JOB_TYPE_DOCUMENT_DELETE:
                await _execute_ownify_document_delete_job(job)
            elif job.get("job_type") == OWNIFY_JOB_TYPE_TENANT_DELETE:
                await _execute_ownify_tenant_delete_job(job)
            else:
                await _execute_ownify_provision_job(job)
        except asyncio.TimeoutError:
            await _set_ownify_job(
                job_id,
                job_status=OWNIFY_JOB_STATUS_TIMED_OUT,
                completed_at=time.time(),
                request_payload={"redacted": True},
                error={
                    "error": "provisioning_timeout",
                    "message": f"Ownify job timed out after {float(job.get('timeout_seconds') or 0):.1f}s",
                    "request_id": job.get("request_id"),
                },
            )
        except Exception as exc:
            logger.error("Ownify job failed (job_id=%s): %s", job_id, exc, exc_info=True)
            await _set_ownify_job(
                job_id,
                job_status=OWNIFY_JOB_STATUS_FAILED,
                completed_at=time.time(),
                request_payload={"redacted": True},
                error={
                    "error": "provisioning_failed",
                    "message": str(exc),
                    "request_id": job.get("request_id"),
                },
            )


async def _ownify_job_worker(worker_index: int) -> None:
    if _ownify_job_queue is None:
        return
    logger.info("Ownify provisioning job worker %s started", worker_index)
    try:
        while True:
            job_id = await _ownify_job_queue.get()
            try:
                await _run_single_ownify_job(job_id)
            finally:
                _ownify_job_queue.task_done()
    except asyncio.CancelledError:
        logger.info("Ownify provisioning job worker %s stopped", worker_index)
        raise


async def _remove_expired_ownify_jobs() -> int:
    if _ownify_job_store is None:
        return 0
    cutoff = time.time() - PROVISIONING_JOB_RETENTION_SECONDS
    stale_ids = _ownify_job_store.terminal_older_than(OWNIFY_JOB_TERMINAL_STATES, cutoff)
    for job_id in stale_ids:
        _ownify_job_store.delete(job_id)
    return len(stale_ids)


async def _ownify_job_cleanup_loop() -> None:
    try:
        while True:
            await asyncio.sleep(PROVISIONING_JOB_CLEANUP_INTERVAL_SECONDS)
            removed = await _remove_expired_ownify_jobs()
            if removed > 0:
                logger.info("Ownify job cleanup removed %s expired jobs", removed)
    except asyncio.CancelledError:
        logger.info("Ownify provisioning job cleanup loop stopped")
        raise


async def _start_ownify_jobs_runtime() -> None:
    global _ownify_job_queue, _ownify_job_workers, _ownify_job_cleanup_task

    if not PROVISIONING_JOBS_ENABLED:
        logger.info("Ownify provisioning jobs are disabled")
        return

    if _ownify_job_queue is not None:
        return

    _ownify_job_queue = asyncio.Queue(maxsize=PROVISIONING_JOB_QUEUE_MAX_SIZE)
    _ownify_job_workers = [
        asyncio.create_task(_ownify_job_worker(worker_index=i + 1))
        for i in range(PROVISIONING_JOB_WORKERS)
    ]
    _ownify_job_cleanup_task = asyncio.create_task(_ownify_job_cleanup_loop())
    if _ownify_job_store is not None:
        requeued = 0
        for job in _ownify_job_store.list_jobs():
            if job.get("job_status") in OWNIFY_JOB_TERMINAL_STATES:
                continue
            try:
                _ownify_job_store.update(
                    job["job_id"],
                    job_status=OWNIFY_JOB_STATUS_QUEUED,
                    phase=OWNIFY_PHASE_QUEUED,
                    cancellation_requested=False,
                    updated_at=time.time(),
                    error=None,
                )
                _ownify_job_queue.put_nowait(job["job_id"])
                requeued += 1
            except asyncio.QueueFull:
                logger.warning("Ownify provisioning queue full while requeueing stored jobs")
                break
        if requeued > 0:
            logger.info("Requeued %s non-terminal Ownify provisioning jobs", requeued)
    logger.info(
        "Ownify provisioning jobs runtime initialized (workers=%s, queue_max_size=%s, default_timeout=%.1fs)",
        PROVISIONING_JOB_WORKERS,
        PROVISIONING_JOB_QUEUE_MAX_SIZE,
        PROVISIONING_JOB_DEFAULT_TIMEOUT_SECONDS,
    )


async def _stop_ownify_jobs_runtime() -> None:
    global _ownify_job_queue, _ownify_job_workers, _ownify_job_cleanup_task

    if _ownify_job_store is not None:
        shutdown_time = time.time()
        for job in _ownify_job_store.list_jobs():
            if job.get("job_status") not in OWNIFY_JOB_TERMINAL_STATES:
                _ownify_job_store.update(
                    job["job_id"],
                    job_status=OWNIFY_JOB_STATUS_CANCELLED,
                    cancellation_requested=True,
                    completed_at=shutdown_time,
                    updated_at=shutdown_time,
                    request_payload={"redacted": True},
                    error={
                        "error": "job_cancelled",
                        "message": "Server shutting down",
                    },
                )

        for task in _ownify_job_workers:
            task.cancel()
        if _ownify_job_workers:
            await asyncio.gather(*_ownify_job_workers, return_exceptions=True)
        _ownify_job_workers = []

    if _ownify_job_cleanup_task is not None:
        _ownify_job_cleanup_task.cancel()
        await asyncio.gather(_ownify_job_cleanup_task, return_exceptions=True)
        _ownify_job_cleanup_task = None

    _ownify_job_queue = None


async def _llm_stuck_watchdog_loop() -> None:
    """
    Detect a wedged process where LLM slots remain occupied while no admitted query
    is in-flight (detached/stuck generation tasks). Exit process so supervisor restarts.
    """
    while True:
        await asyncio.sleep(LLM_STUCK_CHECK_INTERVAL_SECONDS)

        if multi_kb_pipeline is None:
            continue
        llm_gen = getattr(getattr(multi_kb_pipeline, "shared_resources", None), "llm_generator", None)
        if llm_gen is None or not hasattr(llm_gen, "get_concurrency_metrics"):
            continue

        try:
            llm_metrics = llm_gen.get_concurrency_metrics()
        except Exception:
            logger.warning("LLM watchdog failed to collect metrics", exc_info=True)
            continue

        inflight = int(llm_metrics.get("inflight", 0) or 0)
        if inflight <= 0:
            continue
        oldest_inflight_seconds = float(llm_metrics.get("oldest_inflight_seconds", 0.0) or 0.0)
        if oldest_inflight_seconds < LLM_STUCK_THRESHOLD_SECONDS:
            continue

        async with _query_admission_state_lock:
            admitted_query_inflight = int(_query_inflight)

        if admitted_query_inflight > 0:
            continue

        logger.critical(
            "LLM watchdog detected stuck generations (inflight=%s, oldest=%.1fs, threshold=%.1fs, "
            "query_inflight=%s). Exiting process for self-healing restart.",
            inflight,
            oldest_inflight_seconds,
            LLM_STUCK_THRESHOLD_SECONDS,
            admitted_query_inflight,
        )
        os._exit(1)


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Multi-KB AI Infrastructure",
    description="Multi-KB RAG orchestration server",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_UI_INDEX = Path(__file__).parent.parent.parent / "index.html"


@app.get("/", include_in_schema=False)
async def serve_ui():
    if _UI_INDEX.exists():
        return FileResponse(str(_UI_INDEX), media_type="text/html")
    return JSONResponse({"status": "ok", "detail": "No UI index.html found"}, status_code=200)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info("%s %s from %s", request.method, request.url.path, request.client.host)
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info("%s %s - %s (%.2fs)", request.method, request.url.path, response.status_code, duration)
    return response


@app.get("/api/autocomplete")
async def get_workspace_autocomplete_suggestions(
    connector: str = Query(
        ...,
        description="Target workspace: slack, google, calendar, notion",
    ),
    user_id: str = Query(..., description="Unique user tracking context identifier"),
):
    """
    Fast gateway endpoint returning cached workspace navigation structures in under 5ms.
    Completely isolates client autocomplete requests from external rate-limit locks.
    """
    if _redis_connection is None:
        return JSONResponse(content={"items": []})

    connector_key = connector.lower().strip()
    cache_key = f"user_cache:{connector_key}:{user_id}"
    try:
        cached_data = _redis_connection.client().get(cache_key)
        if cached_data:
            return JSONResponse(content={"items": json.loads(cached_data)})
    except Exception as exc:
        logger.warning("Failed to read autocomplete cache for %s/%s: %s", connector, user_id, exc)

    if connector_key == "calendar":
        try:
            items = await _build_calendar_autocomplete_items(user_id)
            if items:
                _redis_connection.client().setex(cache_key, 3600, json.dumps(items))
                logger.info(
                    "Primed calendar autocomplete cache for user_id=%s (%d items)",
                    user_id,
                    len(items),
                )
                return JSONResponse(content={"items": items})
            logger.warning(
                "Calendar autocomplete live fetch returned 0 items for user_id=%s",
                user_id,
            )
        except Exception as exc:
            logger.warning(
                "Live calendar autocomplete failed for user_id=%s: %s",
                user_id,
                exc,
                exc_info=True,
            )

    if connector_key == "onedrive":
        try:
            items = await _build_microsoft_onedrive_autocomplete_items(user_id)
            if items:
                _redis_connection.client().setex(cache_key, 3600, json.dumps(items))
                logger.info(
                    "Primed OneDrive autocomplete cache for user_id=%s (%d items)",
                    user_id,
                    len(items),
                )
                return JSONResponse(content={"items": items})
            logger.warning(
                "OneDrive autocomplete live fetch returned 0 items for user_id=%s",
                user_id,
            )
        except Exception as exc:
            logger.warning(
                "Live OneDrive autocomplete failed for user_id=%s: %s",
                user_id,
                exc,
                exc_info=True,
            )

    if connector_key == "outlook":
        try:
            items = await _build_microsoft_outlook_autocomplete_items(user_id)
            if items:
                _redis_connection.client().setex(cache_key, 3600, json.dumps(items))
                logger.info(
                    "Primed Outlook autocomplete cache for user_id=%s (%d items)",
                    user_id,
                    len(items),
                )
                return JSONResponse(content={"items": items})
            logger.warning(
                "Outlook autocomplete live fetch returned 0 items for user_id=%s",
                user_id,
            )
        except Exception as exc:
            logger.warning(
                "Live Outlook autocomplete failed for user_id=%s: %s",
                user_id,
                exc,
                exc_info=True,
            )

    return JSONResponse(content={"items": []})


# =============================================================================
# Google Workspace MCP endpoints
# =============================================================================


@app.get("/mcp/auth", tags=["mcp"], summary="Start Google Workspace OAuth flow")
async def mcp_auth_start(
    user_id: str = Query(..., description="User identifier for token isolation"),
    services: str = Query("gmail,drive,calendar", description="Comma-separated services to authorize"),
):
    """
    Generate a Google OAuth 2.0 authorization URL.

    Redirect the user to the returned URL. After approval, Google will
    redirect back to /mcp/auth/callback automatically.
    """
    if _mcp_client is None:
        raise HTTPException(
            status_code=503,
            detail="MCP client is not enabled. Set GOOGLE_MCP_ENABLED=true and provide OAuth credentials.",
        )
    requested_services = [s.strip() for s in services.split(",") if s.strip()]
    auth_url = _mcp_client.get_auth_url(user_id=user_id, services=requested_services)
    return JSONResponse({"status": "ok", "auth_url": auth_url, "user_id": user_id})


@app.get("/mcp/auth/callback", tags=["mcp"], summary="Google OAuth callback — exchanges code for tokens")
async def mcp_auth_callback(
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="State parameter (encodes user_id + services)"),
    error: Optional[str] = Query(None, description="Error from Google if user denied"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    OAuth 2.0 callback endpoint. Google redirects here after the user approves.

    Exchanges the authorization code for tokens and stores them in Redis
    keyed by user_id. Redirect URI must match what was registered in GCP.
    """
    if error:
        raise HTTPException(status_code=400, detail=f"Google OAuth error: {error}")

    if _mcp_client is None:
        raise HTTPException(status_code=503, detail="MCP client is not enabled.")

    try:
        result = _mcp_client.exchange_code(code=code, state=state)
        
        # Trigger workspace cache refresh after successful authentication
        # Use the access_token directly from the response to avoid race condition with Redis
        access_token = result.get("access_token")
        if access_token:
            logger.info(f"Triggering initial workspace cache build for {result['user_id']}")
            background_tasks.add_task(
                refresh_user_workspace_cache,
                user_id=result["user_id"],
                connector="google",
                auth_token=access_token,
            )
            background_tasks.add_task(
                refresh_user_workspace_cache,
                user_id=result["user_id"],
                connector="calendar",
                auth_token=access_token,
            )
        else:
            logger.warning(f"Could not extract raw access_token for background refresh task.")
        
        return JSONResponse({
            "status": "ok",
            "message": "Authentication successful. You can close this window.",
            "user_id": result["user_id"],
            "services_authorized": result["services_authorized"],
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("MCP OAuth callback failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Token exchange failed.") from exc


@app.get("/mcp/status", tags=["mcp"], summary="Check MCP auth status for a user")
async def mcp_status(
    user_id: str = Query(..., description="User identifier"),
):
    """Return which Workspace services the user has authenticated."""
    if _mcp_client is None:
        return JSONResponse({"enabled": False})
    status = _mcp_client.auth_status(user_id=user_id)
    return JSONResponse({"enabled": True, "user_id": user_id, "auth_status": status})


@app.delete("/mcp/revoke", tags=["mcp"], summary="Revoke Google Workspace tokens for a user")
async def mcp_revoke(
    user_id: str = Query(..., description="User identifier"),
):
    """Remove all Google Workspace OAuth tokens for a user from Redis, disconnecting their Google access."""
    if _mcp_client is None:
        raise HTTPException(status_code=503, detail="MCP client is not enabled.")
    try:
        _mcp_client.revoke_tokens(user_id=user_id)
        return JSONResponse({"status": "ok", "message": f"Google Workspace tokens revoked for user_id={user_id}"})
    except Exception as exc:
        logger.error("Google MCP revoke failed for user_id=%s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to revoke Google Workspace tokens.") from exc


# =============================================================================
# Microsoft 365 MCP endpoints
# =============================================================================


@app.get("/mcp/microsoft/auth", tags=["mcp-microsoft"], summary="Start Microsoft 365 OAuth flow")
async def microsoft_mcp_auth_start(
    user_id: str = Query(..., description="User identifier for token isolation"),
    services: str = Query("outlook,onedrive", description="Comma-separated services to authorize"),
):
    """Generate a Microsoft OAuth 2.0 authorization URL for Outlook and OneDrive."""
    if _microsoft_mcp_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Microsoft 365 MCP client is not enabled. Set MICROSOFT_MCP_ENABLED=true "
                "and provide OAuth credentials."
            ),
        )
    requested_services = [s.strip() for s in services.split(",") if s.strip()]
    auth_url = _microsoft_mcp_client.get_auth_url(user_id=user_id, services=requested_services)
    return JSONResponse({"status": "ok", "auth_url": auth_url, "user_id": user_id})


@app.get(
    "/mcp/microsoft/auth/callback",
    tags=["mcp-microsoft"],
    summary="Microsoft OAuth callback — exchanges code for tokens",
)
async def microsoft_mcp_auth_callback(
    code: str = Query(..., description="Authorization code from Microsoft"),
    state: str = Query(..., description="State parameter (encodes user_id + services)"),
    error: Optional[str] = Query(None, description="Error from Microsoft if user denied"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """OAuth 2.0 callback endpoint for Microsoft 365 delegated permissions."""
    if error:
        raise HTTPException(status_code=400, detail=f"Microsoft OAuth error: {error}")

    if _microsoft_mcp_client is None:
        raise HTTPException(status_code=503, detail="Microsoft 365 MCP client is not enabled.")

    try:
        result = _microsoft_mcp_client.exchange_code(code=code, state=state)
        access_token = result.get("access_token")
        if access_token:
            background_tasks.add_task(
                refresh_user_workspace_cache,
                user_id=result["user_id"],
                connector="onedrive",
                auth_token=access_token,
            )
            background_tasks.add_task(
                refresh_user_workspace_cache,
                user_id=result["user_id"],
                connector="outlook",
                auth_token=access_token,
            )
        return JSONResponse({
            "status": "ok",
            "message": "Microsoft authentication successful. You can close this window.",
            "user_id": result["user_id"],
            "services_authorized": result["services_authorized"],
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Microsoft MCP OAuth callback failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Microsoft token exchange failed.") from exc


@app.get("/mcp/microsoft/status", tags=["mcp-microsoft"], summary="Check Microsoft 365 MCP auth status")
async def microsoft_mcp_status(
    user_id: str = Query(..., description="User identifier"),
):
    """Return which Microsoft 365 services the user has authenticated."""
    if _microsoft_mcp_client is None:
        return JSONResponse({"enabled": False})
    status = _microsoft_mcp_client.auth_status(user_id=user_id)
    return JSONResponse({"enabled": True, "user_id": user_id, "auth_status": status})


@app.delete("/mcp/microsoft/revoke", tags=["mcp-microsoft"], summary="Revoke Microsoft 365 tokens for a user")
async def microsoft_mcp_revoke(
    user_id: str = Query(..., description="User identifier"),
):
    """Remove all Microsoft 365 OAuth tokens for a user from Redis."""
    if _microsoft_mcp_client is None:
        raise HTTPException(status_code=503, detail="Microsoft 365 MCP client is not enabled.")
    try:
        _microsoft_mcp_client.revoke_tokens(user_id=user_id)
        return JSONResponse({
            "status": "ok",
            "message": f"Microsoft 365 tokens revoked for user_id={user_id}",
        })
    except Exception as exc:
        logger.error("Microsoft MCP revoke failed for user_id=%s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to revoke Microsoft 365 tokens.") from exc


# =============================================================================
# Slack MCP endpoints
# =============================================================================


@app.get("/mcp/slack/auth", tags=["mcp-slack"], summary="Start Slack OAuth flow for a user")
async def slack_mcp_auth_start(
    user_id: str = Query(..., description="User identifier for token isolation"),
):
    """
    Generate a Slack OAuth 2.0 authorization URL (user token / xoxp-).

    Redirect the user to the returned URL. After approval, Slack will
    redirect back to /mcp/slack/callback automatically.
    """
    if _slack_mcp_client is None:
        raise HTTPException(
            status_code=503,
            detail="Slack MCP client is not enabled. Set SLACK_MCP_ENABLED=true and provide OAuth credentials.",
        )
    auth_url = _slack_mcp_client.get_auth_url(user_id=user_id)
    return JSONResponse({"status": "ok", "auth_url": auth_url, "user_id": user_id})


@app.get("/mcp/slack/callback", tags=["mcp-slack"], summary="Slack OAuth callback — exchanges code for user token")
async def slack_mcp_auth_callback(
    code: str = Query(..., description="Authorization code from Slack"),
    state: str = Query(..., description="State parameter (encodes user_id + nonce)"),
    error: Optional[str] = Query(None, description="Error from Slack if user denied"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    OAuth 2.0 callback endpoint. Slack redirects here after the user approves.

    Exchanges the authorization code for a user token and stores it in Redis
    keyed by user_id. Redirect URI must match what was registered in your Slack App settings.
    """
    if error:
        raise HTTPException(status_code=400, detail=f"Slack OAuth error: {error}")

    if _slack_mcp_client is None:
        raise HTTPException(status_code=503, detail="Slack MCP client is not enabled.")

    try:
        result = _slack_mcp_client.exchange_code(code=code, state=state)
        
        # Trigger workspace cache refresh after successful authentication
        # Use the access_token directly from the response to avoid race condition with Redis
        access_token = result.get("access_token")
        if access_token:
            logger.info(f"Triggering initial workspace cache build for {result['user_id']}")
            background_tasks.add_task(
                refresh_user_workspace_cache,
                user_id=result["user_id"],
                connector="slack",
                auth_token=access_token,
            )
        else:
            logger.warning(f"Could not extract raw access_token for background refresh task.")
        
        return JSONResponse({
            "status": "ok",
            "message": "Slack authentication successful. You can close this window.",
            "user_id": result["user_id"],
            "team_name": result.get("team_name", ""),
            "slack_user_id": result.get("slack_user_id", ""),
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Slack OAuth callback failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Slack token exchange failed.") from exc


@app.get("/mcp/slack/status", tags=["mcp-slack"], summary="Check Slack auth status for a user")
async def slack_mcp_status(
    user_id: str = Query(..., description="User identifier"),
):
    """Return whether the user has authenticated with Slack and token metadata."""
    if _slack_mcp_client is None:
        return JSONResponse({"enabled": False})
    status = _slack_mcp_client.auth_status(user_id=user_id)
    return JSONResponse({"enabled": True, "user_id": user_id, "auth_status": status})


@app.delete("/mcp/slack/revoke", tags=["mcp-slack"], summary="Revoke Slack token for a user")
async def slack_mcp_revoke(
    user_id: str = Query(..., description="User identifier"),
):
    """Remove the Slack token for a user from Redis, disconnecting their Slack access."""
    if _slack_mcp_client is None:
        raise HTTPException(status_code=503, detail="Slack MCP client is not enabled.")
    try:
        _slack_mcp_client.revoke_tokens(user_id=user_id)
        return JSONResponse({"status": "ok", "message": f"Slack token revoked for user_id={user_id}"})
    except Exception as exc:
        logger.error("Slack token revoke failed for user_id=%s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to revoke Slack token.") from exc


@app.post("/mcp/slack/command", tags=["mcp-slack"])
async def slack_slash_command(
    background_tasks: BackgroundTasks,
    command: str = Form(...),
    text: str = Form(...),
    user_id: str = Form(...),
    user_name: str = Form(...),
    channel_id: str = Form(...),
    channel_name: str = Form(...),
    response_url: str = Form(...),
):
    """
    Slack Slash Command Webhook (/Synapse-ai).
    Processes query asynchronously and posts the final answer back to response_url.
    """
    if _slack_mcp_client is None:
        return {"text": "Slack MCP is not enabled on this server.", "response_type": "ephemeral"}

    # Resolve target KB (default to environment config or first active KB)
    default_kb = os.getenv("DEFAULT_SLACK_KB_ID", "Epstein_Data")
    if kb_manager is not None and not kb_manager.registry.exists(default_kb):
        existing_kbs = list(kb_manager.registry._data.keys())
        if existing_kbs:
            default_kb = existing_kbs[0]
        else:
            return {"text": "No active Knowledge Base found on this server.", "response_type": "ephemeral"}

    # Run the query task asynchronously to avoid Slack's 3-second timeout
    background_tasks.add_task(
        _process_slack_command_async,
        kb_id=default_kb,
        query=text,
        slack_user_id=user_id,
        slack_user_name=user_name,
        channel_id=channel_id,
        response_url=response_url
    )

    return {
        "text": f"Synapse-ai is thinking about your query: '{text}'...",
        "response_type": "ephemeral"
    }

@app.post("/mcp/slack/synapse-ai/epstein", tags=["mcp-slack"])
async def slack_epstein_command(
    background_tasks: BackgroundTasks,
    command: str = Form(...),
    text: str = Form(...),
    user_id: str = Form(...),
    user_name: str = Form(...),
    channel_id: str = Form(...),
    channel_name: str = Form(...),
    response_url: str = Form(...),
):
    """
    Slack Slash Command Webhook for Epstein KB (/epstein).
    Processes query asynchronously against the epstein KB.
    """
    if _slack_mcp_client is None:
        return {"text": "Slack MCP is not enabled on this server.", "response_type": "ephemeral"}

    if kb_manager is not None and not kb_manager.registry.exists("epstein"):
        return {"text": "The Epstein Knowledge Base is not configured on this server.", "response_type": "ephemeral"}

    background_tasks.add_task(
        _process_slack_command_async,
        kb_id="epstein",
        query=text,
        slack_user_id=user_id,
        slack_user_name=user_name,
        channel_id=channel_id,
        response_url=response_url
    )

    return {
        "text": f"Synapse-ai is searching the Epstein KB for: '{text}'...",
        "response_type": "ephemeral"
    }


@app.post("/mcp/slack/synapse-ai/cafl", tags=["mcp-slack"])
async def slack_cafl_command(
    background_tasks: BackgroundTasks,
    command: str = Form(...),
    text: str = Form(...),
    user_id: str = Form(...),
    user_name: str = Form(...),
    channel_id: str = Form(...),
    channel_name: str = Form(...),
    response_url: str = Form(...),
):
    """
    Slack Slash Command Webhook for CAFL KB (/synapse-ai_Cafl).
    Processes query asynchronously against the cafl KB.
    """
    if _slack_mcp_client is None:
        return {"text": "Slack MCP is not enabled on this server.", "response_type": "ephemeral"}

    if kb_manager is not None and not kb_manager.registry.exists("cafl"):
        return {"text": "The CAFL Knowledge Base is not configured on this server.", "response_type": "ephemeral"}

    background_tasks.add_task(
        _process_slack_command_async,
        kb_id="cafl",
        query=text,
        slack_user_id=user_id,
        slack_user_name=user_name,
        channel_id=channel_id,
        response_url=response_url
    )

    return {
        "text": f"Synapse-ai is searching the CAFL KB for: '{text}'...",
        "response_type": "ephemeral"
    }


async def _process_slack_command_async(
    kb_id: str,
    query: str,
    slack_user_id: str,
    slack_user_name: str,
    channel_id: str,
    response_url: str
):
    import httpx
    # Unique session per Slack channel to preserve context history
    session_id = f"slack_session_{channel_id}"
    request_id = uuid.uuid4().hex
    cancel_event = threading.Event()

    # Ensure session exists in the pipeline
    if not multi_kb_pipeline.session_exists(kb_id, session_id):
        pipeline = multi_kb_pipeline.get_pipeline_for_kb(kb_id)
        if pipeline.session_manager is not None:
            pipeline.session_manager.get_or_create_session(session_id)
            try:
                pipeline.save_session(session_id)
            except Exception as _save_exc:
                logger.warning("Failed to save Slack session: %s", _save_exc)

    query_req = QueryRequest(
        query=query,
        session_id=session_id,
        user_id=slack_user_id,
        #connector="slack"
    )

    try:
        res = await _execute_query_request_internal(
            kb_id=kb_id,
            request=query_req,
            request_id=request_id,
            session_id=session_id,
            user_id=slack_user_id,
            cancel_event=cancel_event,
            timeout_seconds=QUERY_TIMEOUT_SECONDS
        )
        answer = res.answer
    except Exception as exc:
        answer = f"Error processing query: {exc}"

    # Post back to Slack's response_url
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                response_url,
                json={
                    "text": answer,
                    "response_type": "in_channel"
                },
                timeout=10.0
            )
    except Exception as post_exc:
        logger.error("Failed to post answer to Slack response_url: %s", post_exc)


# =============================================================================
# Notion MCP endpoints
# =============================================================================


@app.get("/mcp/notion/auth", tags=["mcp-notion"], summary="Start Notion OAuth flow for a user")
async def notion_mcp_auth_start(
    user_id: str = Query(..., description="User identifier for token isolation"),
):
    """
    Generate a Notion OAuth 2.0 authorization URL.

    Redirect the user to the returned URL. After approval, Notion will
    redirect back to /mcp/notion/callback automatically.
    """
    if _notion_mcp_client is None:
        raise HTTPException(
            status_code=503,
            detail="Notion MCP client is not enabled. Set NOTION_MCP_ENABLED=true and provide OAuth credentials.",
        )
    auth_url = _notion_mcp_client.get_auth_url(user_id=user_id)
    return JSONResponse({"status": "ok", "auth_url": auth_url, "user_id": user_id})


@app.get("/mcp/notion/callback", tags=["mcp-notion"], summary="Notion OAuth callback — exchanges code for token")
async def notion_mcp_auth_callback(
    code: str = Query(..., description="Authorization code from Notion"),
    state: str = Query(..., description="State parameter (encodes user_id + nonce)"),
    error: Optional[str] = Query(None, description="Error from Notion if user denied"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    OAuth 2.0 callback endpoint. Notion redirects here after the user approves.

    Exchanges the authorization code for an access token and stores it in Redis
    keyed by user_id.
    """
    if error:
        raise HTTPException(status_code=400, detail=f"Notion OAuth error: {error}")

    if _notion_mcp_client is None:
        raise HTTPException(status_code=503, detail="Notion MCP client is not enabled.")

    try:
        result = _notion_mcp_client.exchange_code(code=code, state=state)
        
        # Trigger workspace cache refresh after successful authentication
        # Use the access_token directly from the response to avoid race condition with Redis
        access_token = result.get("access_token")
        if access_token:
            logger.info(f"Triggering initial workspace cache build for {result['user_id']}")
            background_tasks.add_task(
                refresh_user_workspace_cache,
                user_id=result["user_id"],
                connector="notion",
                auth_token=access_token,
            )
        else:
            logger.warning(f"Could not extract raw access_token for background refresh task.")
        
        return JSONResponse({
            "status":         "ok",
            "message":        "Notion authentication successful. You can close this window.",
            "user_id":        result["user_id"],
            "workspace_name": result.get("workspace_name", ""),
            "workspace_id":   result.get("workspace_id", ""),
        })
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Notion OAuth callback failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Notion token exchange failed.") from exc


@app.get("/mcp/notion/status", tags=["mcp-notion"], summary="Check Notion auth status for a user")
async def notion_mcp_status(
    user_id: str = Query(..., description="User identifier"),
):
    """Return whether the user has authenticated with Notion and workspace metadata."""
    if _notion_mcp_client is None:
        return JSONResponse({"enabled": False})
    status = _notion_mcp_client.auth_status(user_id=user_id)
    return JSONResponse({"enabled": True, "user_id": user_id, "auth_status": status})


@app.delete("/mcp/notion/revoke", tags=["mcp-notion"], summary="Revoke Notion token for a user")
async def notion_mcp_revoke(
    user_id: str = Query(..., description="User identifier"),
):
    """Remove the Notion token for a user from Redis, disconnecting their Notion access."""
    if _notion_mcp_client is None:
        raise HTTPException(status_code=503, detail="Notion MCP client is not enabled.")
    try:
        _notion_mcp_client.revoke_tokens(user_id=user_id)
        return JSONResponse({"status": "ok", "message": f"Notion token revoked for user_id={user_id}"})
    except Exception as exc:
        logger.error("Notion token revoke failed for user_id=%s: %s", user_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to revoke Notion token.") from exc


# =============================================================================
# Health & Stats
# =============================================================================

@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serve the admin UI."""
    index_path = Path(__file__).resolve().parents[2] / "index.html"
    return FileResponse(index_path)


@app.get("/health/liveness")
async def health_liveness():
    return {
        "status": "alive",
        "timestamp": time.time(),
    }


@app.get("/health/readiness")
async def health_readiness():
    qdrant_ok = False
    llm_ready = False
    redis_ready = True
    llm_generation_metrics: Dict[str, Any] = {}

    if kb_manager is not None:
        try:
            kb_manager.qdrant.get_collections()
            qdrant_ok = True
        except Exception:
            qdrant_ok = False

    if multi_kb_pipeline is not None and getattr(multi_kb_pipeline, "shared_resources", None) is not None:
        llm_gen = multi_kb_pipeline.shared_resources.llm_generator
        llm_ready = getattr(llm_gen, "model", None) is not None
        if llm_gen is not None and hasattr(llm_gen, "get_concurrency_metrics"):
            try:
                llm_generation_metrics = llm_gen.get_concurrency_metrics()
            except Exception:
                llm_generation_metrics = {"enabled": False, "error": "metric_collection_failed"}

    redis_details: Dict[str, Any] = {
        "enabled": REDIS_ENABLED,
        "lock_enabled": bool(REDIS_ENABLED and REDIS_LOCK_ENABLED),
        "rate_limit_enabled": bool(REDIS_ENABLED and REDIS_RATE_LIMIT_ENABLED),
    }
    if REDIS_ENABLED:
        if _redis_connection is None:
            redis_details["not_initialized"] = True
        else:
            redis_ready = _redis_connection.ping()
            redis_details["ready"] = redis_ready
            if _redis_connection.init_error:
                redis_details["init_error"] = _redis_connection.init_error

    async_jobs_ready = _async_query_jobs_runtime_ready()

    ready = (
        multi_kb_pipeline is not None and
        kb_manager is not None and
        query_executor is not None and
        index_executor is not None and
        qdrant_ok and
        llm_ready and
        redis_ready and
        async_jobs_ready
    )

    payload = {
        "status": "ready" if ready else "not_ready",
        "pipeline_loaded": multi_kb_pipeline is not None,
        "kb_manager_ready": kb_manager is not None,
        "query_executor_ready": query_executor is not None,
        "index_executor_ready": index_executor is not None,
        "qdrant_ready": qdrant_ok,
        "model_ready": llm_ready,
        "redis_ready": redis_ready,
        "async_query_jobs_ready": async_jobs_ready,
        "executor_metrics": {
            "query": QUERY_EXECUTOR_WORKERS,
            "index": INDEX_EXECUTOR_WORKERS,
        },
        "admission_metrics": _query_admission_metrics(),
        "session_lock_metrics": _session_lock_metrics(),
        "rate_limit_metrics": _rate_limit_metrics(),
        "indexing_gate_metrics": _indexing_gate_metrics(),
        "async_query_job_metrics": _async_query_job_metrics(),
        "llm_watchdog": {
            "enabled": LLM_STUCK_WATCHDOG_ENABLED,
            "threshold_seconds": LLM_STUCK_THRESHOLD_SECONDS,
            "check_interval_seconds": LLM_STUCK_CHECK_INTERVAL_SECONDS,
        },
        "redis": redis_details,
        "llm_generation_metrics": llm_generation_metrics,
        "timestamp": time.time(),
    }

    if not ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.get("/health")
async def health_check():
    return await health_readiness()


@app.get("/stats")
async def system_stats():
    if multi_kb_pipeline is None or kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    stats = multi_kb_pipeline.stats()
    return {
        "status": "online",
        "total_kbs": len(kb_manager.registry.list_kbs()),
        "active_pipelines": stats["active_pipelines"],
        "max_active_pipelines": stats["max_active_pipelines"],
    }


# =============================================================================
# KB Management
# =============================================================================

@app.post("/kb/create")
async def create_kb(request: CreateKBRequest):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        kb = kb_manager.create_kb(
            kb_id=request.kb_id,
            display_name=request.display_name,
            description=request.description,
            existing_collection=request.existing_collection,
        )
        return {"status": "success", "kb": kb}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to create KB: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/kb/list")
async def list_kbs():
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return kb_manager.list_kbs()


@app.get("/kb/{kb_id}")
async def get_kb_info(kb_id: str):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        return kb_manager.get_kb_info(kb_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/kb/{kb_id}")
async def delete_kb(kb_id: str):
    if kb_manager is None or multi_kb_pipeline is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        result = kb_manager.delete_kb(kb_id)
        multi_kb_pipeline.evict_kb(kb_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to delete KB: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# =============================================================================
# Document Management
# =============================================================================

def _raise_document_indexing_failure(result: Dict[str, Any]) -> None:
    """Map document-manager error codes to stable HTTP responses."""
    error_message = result.get("error", "Unknown error")
    error_code = result.get("error_code", "indexing_failed")
    file_name = result.get("file_name")

    if error_code == "unsupported_file_type":
        raise HTTPException(
            status_code=400,
            detail={
                "error": error_code,
                "message": error_message,
                "file_name": file_name,
                "supported_formats": result.get("supported_formats", []),
            },
        )

    if error_code == "invalid_document_source":
        raise HTTPException(
            status_code=400,
            detail={
                "error": error_code,
                "message": error_message,
                "file_name": file_name,
            },
        )

    if error_code in {"local_file_not_found", "directory_not_found"}:
        raise HTTPException(
            status_code=404,
            detail={
                "error": error_code,
                "message": error_message,
                "file_name": file_name,
            },
        )

    if error_code in {"not_a_file", "not_a_directory", "no_documents_to_index"}:
        raise HTTPException(
            status_code=400,
            detail={
                "error": error_code,
                "message": error_message,
                "file_name": file_name,
            },
        )

    if error_code in {
        "extraction_failed",
        "no_content_extracted",
        "empty_after_cleaning",
        "content_too_short",
        "low_quality_content",
    }:
        raise HTTPException(
            status_code=422,
            detail={
                "error": error_code,
                "message": error_message,
                "file_name": file_name,
            },
        )

    raise HTTPException(
        status_code=500,
        detail={
            "error": error_code,
            "message": f"Failed to index file: {error_message}",
            "file_name": file_name,
        },
    )

@app.post("/kb/{kb_id}/documents/add")
async def add_document(kb_id: str, request: AddDocumentRequest):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        if INDEXING_ONLINE_MODE:
            result = await run_in_named_executor(
                index_executor,
                kb_manager.add_document,
                kb_id=kb_id,
                file_id=request.file_id,
                file_name=request.file_name,
                sas_url=request.sas_url,
                local_path=request.local_path,
            )
        else:
            async with admin_write_guard(kb_id, timeout_seconds=INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS):
                result = await run_in_named_executor(
                    index_executor,
                    kb_manager.add_document,
                    kb_id=kb_id,
                    file_id=request.file_id,
                    file_name=request.file_name,
                    sas_url=request.sas_url,
                    local_path=request.local_path,
                )
        if result.get("success"):
            if multi_kb_pipeline is not None:
                multi_kb_pipeline.evict_kb(kb_id)
            return {"status": "success", **result}
        _raise_document_indexing_failure(result)
    except TimeoutError:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "indexing_drain_timeout",
                "message": "Cannot start indexing while active queries are running",
                "timeout_seconds": INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS,
            }
        )
    except ValueError as exc:
        if "KB not found" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc))
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_document_source",
                "message": str(exc),
            },
        )


@app.post("/kb/{kb_id}/documents/add-batch")
async def add_documents_batch(kb_id: str, request: BatchAddDocumentsRequest):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    documents = [
        document.model_dump(exclude_none=True)
        for document in (request.documents or [])
    ]

    try:
        if INDEXING_ONLINE_MODE:
            result = await run_in_named_executor(
                index_executor,
                kb_manager.add_documents,
                kb_id=kb_id,
                documents=documents,
                directory_path=request.directory_path,
                recursive=request.recursive,
                fail_fast=request.fail_fast,
            )
        else:
            async with admin_write_guard(kb_id, timeout_seconds=INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS):
                result = await run_in_named_executor(
                    index_executor,
                    kb_manager.add_documents,
                    kb_id=kb_id,
                    documents=documents,
                    directory_path=request.directory_path,
                    recursive=request.recursive,
                    fail_fast=request.fail_fast,
                )

        if result.get("success") or result.get("partial_success"):
            if multi_kb_pipeline is not None:
                multi_kb_pipeline.evict_kb(kb_id)
            status = "partial_success" if result.get("partial_success") else "success"
            return JSONResponse(
                status_code=207 if result.get("partial_success") else 200,
                content={"status": status, **result},
            )

        error_code = result.get("error_code")
        if error_code in {"directory_not_found", "not_a_directory", "no_documents_to_index"}:
            _raise_document_indexing_failure(result)

        return JSONResponse(
            status_code=422,
            content={"status": "failed", **result},
        )
    except TimeoutError:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "indexing_drain_timeout",
                "message": "Cannot start batch indexing while active queries are running",
                "timeout_seconds": INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS,
            }
        )
    except ValueError as exc:
        if "KB not found" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc))
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_document_source",
                "message": str(exc),
            },
        )


@app.delete("/kb/{kb_id}/documents/{file_id}")
async def delete_document(kb_id: str, file_id: str, file_name: Optional[str] = None):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        if INDEXING_ONLINE_MODE:
            result = await run_in_named_executor(
                index_executor,
                kb_manager.remove_document,
                kb_id,
                file_id,
                file_name,
            )
        else:
            async with admin_write_guard(kb_id, timeout_seconds=INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS):
                result = await run_in_named_executor(
                    index_executor,
                    kb_manager.remove_document,
                    kb_id,
                    file_id,
                    file_name,
                )
        if result.get("success"):
            if multi_kb_pipeline is not None:
                multi_kb_pipeline.evict_kb(kb_id)
            return {"status": "success", **result}
        raise HTTPException(status_code=404, detail=result.get("error", "Delete failed"))
    except TimeoutError:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "indexing_drain_timeout",
                "message": "Cannot start delete indexing operation while active queries are running",
                "timeout_seconds": INDEXING_WRITE_DRAIN_TIMEOUT_SECONDS,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/kb/{kb_id}/documents")
async def list_documents(kb_id: str):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        return kb_manager.list_documents(kb_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# =============================================================================
# Ownify AI Provisioning
# =============================================================================

def _validate_ownify_tenant_id(tenant_id: str) -> str:
    kb_id = _ownify_kb_id(tenant_id)
    try:
        KBManager._validate_kb_id(kb_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return kb_id


def _ownify_provision_request_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "display_name": payload.get("display_name"),
        "description": payload.get("description"),
        "has_system_prompt": payload.get("system_prompt") is not None,
        "has_ai_config": payload.get("ai_config") is not None,
        "documents_count": len(payload.get("documents") or []),
        "replace_existing": bool(payload.get("replace_existing", False)),
    }


def _ownify_documents_request_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "documents_count": len(payload.get("documents") or []),
        "documents": [
            {
                "file_id": item.get("file_id"),
                "file_name": item.get("file_name"),
            }
            for item in (payload.get("documents") or [])
        ],
    }


def _ownify_create_job_record(
    *,
    job_type: str,
    tenant_id: str,
    kb_id: str,
    request_payload: Dict[str, Any],
    request_summary: Dict[str, Any],
    idempotency_key: Optional[str],
    timeout_seconds: float,
) -> Dict[str, Any]:
    now = time.time()
    return {
        "job_id": uuid.uuid4().hex,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "job_type": job_type,
        "request_id": uuid.uuid4().hex,
        "idempotency_key": idempotency_key,
        "request_hash": _request_hash(request_payload),
        "job_status": OWNIFY_JOB_STATUS_QUEUED,
        "phase": OWNIFY_PHASE_QUEUED,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "attempts": 0,
        "timeout_seconds": timeout_seconds,
        "cancellation_requested": False,
        "request_payload": request_payload,
        "request_summary": request_summary,
        "kb": None,
        "documents": _ownify_document_summary(request_payload.get("documents") or []),
        "result": None,
        "error": None,
    }


async def _ownify_prepare_job(
    *,
    job_type: str,
    tenant_id: str,
    kb_id: str,
    request_payload: Dict[str, Any],
    request_summary: Dict[str, Any],
    idempotency_key: Optional[str],
    timeout_seconds: Optional[float],
    enqueue: bool = True,
) -> Dict[str, Any]:
    if not PROVISIONING_JOBS_ENABLED:
        raise HTTPException(status_code=409, detail="Ownify provisioning jobs are disabled")
    if enqueue and _ownify_job_queue is None:
        raise HTTPException(status_code=503, detail="Ownify provisioning queue not initialized")

    store = _ownify_store()
    request_hash = _request_hash(request_payload)
    existing = store.find_by_idempotency_key(
        tenant_id=tenant_id,
        idempotency_key=idempotency_key,
        job_type=job_type,
    )
    if existing:
        if existing.get("request_hash") != request_hash:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "idempotency_key_conflict",
                    "message": "The idempotency key was already used with a different request payload",
                    "job_id": existing.get("job_id"),
                },
            )
        return existing

    resolved_timeout = timeout_seconds or PROVISIONING_JOB_DEFAULT_TIMEOUT_SECONDS
    resolved_timeout = min(float(resolved_timeout), PROVISIONING_JOB_MAX_TIMEOUT_SECONDS)
    record = _ownify_create_job_record(
        job_type=job_type,
        tenant_id=tenant_id,
        kb_id=kb_id,
        request_payload=request_payload,
        request_summary=request_summary,
        idempotency_key=idempotency_key,
        timeout_seconds=resolved_timeout,
    )
    store.create(record)
    if enqueue:
        await _enqueue_ownify_job(record)
    return record


async def _wait_for_ownify_job_completion(job: Dict[str, Any]) -> Dict[str, Any]:
    job_id = job["job_id"]
    current = _ownify_store().get(job_id) or job
    if current.get("job_status") in OWNIFY_JOB_TERMINAL_STATES:
        return current

    deadline = time.monotonic() + max(1.0, _remaining_job_seconds(current))
    while time.monotonic() < deadline:
        await asyncio.sleep(0.25)
        current = _ownify_store().get(job_id) or current
        if current.get("job_status") in OWNIFY_JOB_TERMINAL_STATES:
            return current

    raise HTTPException(
        status_code=504,
        detail={
            "error": "job_wait_timeout",
            "message": "Timed out waiting for the Ownify document job to complete",
            "job_id": job_id,
        },
    )



@app.post(
    "/ownify/tenants/{tenant_id}/ai/provision",
    response_model=OwnifyJobAcceptedResponse,
    status_code=202,
)
async def ownify_provision_tenant_ai(
    tenant_id: str,
    request: OwnifyProvisionRequest,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    if request.documents:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "documents_not_allowed_in_provision",
                "message": (
                    "Provisioning only creates or updates the tenant AI workspace. "
                    "Upload documents with POST /ownify/tenants/{tenant_id}/ai/documents."
                ),
            },
        )

    kb_id = _validate_ownify_tenant_id(tenant_id)
    request_payload = _model_payload(request)
    job = await _ownify_prepare_job(
        job_type=OWNIFY_JOB_TYPE_PROVISION,
        tenant_id=tenant_id,
        kb_id=kb_id,
        request_payload=request_payload,
        request_summary=_ownify_provision_request_summary(request_payload),
        idempotency_key=request.idempotency_key,
        timeout_seconds=request.timeout_seconds,
    )
    return _ownify_terminal_response(job)


@app.get(
    "/ownify/tenants/{tenant_id}/ai/jobs/{job_id}",
    response_model=OwnifyJobStatusResponse,
)
@app.get(
    "/ownify/tenants/{tenant_id}/ai/provision/jobs/{job_id}",
    response_model=OwnifyJobStatusResponse,
)
async def ownify_get_provisioning_job(
    tenant_id: str,
    job_id: str,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    job = _ownify_store().get(job_id)
    if job is None or job.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return OwnifyJobStatusResponse(**_ownify_status_payload(job))


@app.post(
    "/ownify/tenants/{tenant_id}/ai/config",
    response_model=OwnifyConfigUpdateResponse,
)
async def ownify_update_ai_config(
    tenant_id: str,
    request: OwnifyConfigUpdateRequest,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    kb_id = _validate_ownify_tenant_id(tenant_id)
    current_kb = kb_manager.registry.get_kb(kb_id)
    if not current_kb:
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")

    result = await run_in_named_executor(
        index_executor,
        kb_manager.write_runtime_config_snapshot,
        kb_id=kb_id,
        tenant_id=tenant_id,
        display_name=request.display_name or current_kb.get("display_name", kb_id),
        description=request.description if request.description is not None else current_kb.get("description"),
        system_prompt=request.system_prompt,
        ai_config=_model_payload(request.ai_config),
        source="ownify_config_update",
        timeout_seconds=ADMIN_WRITE_TIMEOUT_SECONDS,
    )
    if multi_kb_pipeline is not None:
        multi_kb_pipeline.evict_kb(kb_id)

    return OwnifyConfigUpdateResponse(
        status="success",
        tenant_id=tenant_id,
        kb_id=kb_id,
        kb=result["kb"],
        snapshot_path=result["snapshot_path"],
        config_version=result["config_version"],
    )


@app.delete(
    "/ownify/tenants/{tenant_id}/ai",
    response_model=OwnifyTenantDeleteResponse,
)
async def ownify_delete_tenant_ai(
    tenant_id: str,
    idempotency_key: Optional[str] = None,
    timeout_seconds: Optional[float] = Query(None, gt=0),
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    kb_id = _validate_ownify_tenant_id(tenant_id)
    request_payload = {
        "delete_all": True,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
    }
    job = await _ownify_prepare_job(
        job_type=OWNIFY_JOB_TYPE_TENANT_DELETE,
        tenant_id=tenant_id,
        kb_id=kb_id,
        request_payload=request_payload,
        request_summary={"delete_all": True},
        idempotency_key=idempotency_key,
        timeout_seconds=timeout_seconds,
    )
    completed_job = await _wait_for_ownify_job_completion(job)
    response = _ownify_tenant_delete_response(completed_job)
    if completed_job.get("job_status") == OWNIFY_JOB_STATUS_SUCCEEDED:
        _ownify_store().delete(completed_job["job_id"])
    return response


@app.post(
    "/ownify/tenants/{tenant_id}/ai/documents",
    response_model=OwnifyDocumentsOperationResponse,
)
@app.post(
    "/ownify/tenants/{tenant_id}/ai/documents/jobs",
    response_model=OwnifyDocumentsOperationResponse,
)
async def ownify_submit_documents_job(
    tenant_id: str,
    request: OwnifyBatchDocumentsRequest,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    kb_id = _validate_ownify_tenant_id(tenant_id)
    if not kb_manager.registry.exists(kb_id):
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")

    request_payload = _model_payload(request)
    job = await _ownify_prepare_job(
        job_type=OWNIFY_JOB_TYPE_DOCUMENTS,
        tenant_id=tenant_id,
        kb_id=kb_id,
        request_payload=request_payload,
        request_summary=_ownify_documents_request_summary(request_payload),
        idempotency_key=request.idempotency_key,
        timeout_seconds=request.timeout_seconds,
    )
    completed_job = await _wait_for_ownify_job_completion(job)
    return _ownify_documents_operation_response(completed_job)


@app.get("/ownify/tenants/{tenant_id}/ai/documents")
async def ownify_list_documents(
    tenant_id: str,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    kb_id = _validate_ownify_tenant_id(tenant_id)
    if not kb_manager.registry.exists(kb_id):
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")

    try:
        result = await run_in_named_executor(
            index_executor,
            kb_manager.list_documents,
            kb_id,
            timeout_seconds=ADMIN_WRITE_TIMEOUT_SECONDS,
        )
        return {
            "status": "success",
            "tenant_id": tenant_id,
            "kb_id": kb_id,
            **result,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to list Ownify documents: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete(
    "/ownify/tenants/{tenant_id}/ai/documents/{file_id}",
    response_model=OwnifyDocumentDeleteResponse,
)
async def ownify_delete_document(
    tenant_id: str,
    file_id: str,
    file_name: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    timeout_seconds: Optional[float] = Query(None, gt=0),
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    kb_id = _validate_ownify_tenant_id(tenant_id)
    if not kb_manager.registry.exists(kb_id):
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")

    request_payload = {
        "file_id": file_id,
        "file_name": file_name,
    }
    job = await _ownify_prepare_job(
        job_type=OWNIFY_JOB_TYPE_DOCUMENT_DELETE,
        tenant_id=tenant_id,
        kb_id=kb_id,
        request_payload={key: value for key, value in request_payload.items() if value is not None},
        request_summary={
            "file_id": file_id,
            "file_name": file_name,
        },
        idempotency_key=idempotency_key,
        timeout_seconds=timeout_seconds,
    )
    completed_job = await _wait_for_ownify_job_completion(job)
    return _ownify_document_delete_response(completed_job)


@app.get(
    "/ownify/tenants/{tenant_id}/ai/status",
    response_model=OwnifyAIStatusResponse,
)
async def ownify_get_ai_status(
    tenant_id: str,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    if kb_manager is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    kb_id = _validate_ownify_tenant_id(tenant_id)
    latest_job = _ownify_store().latest_for_tenant(tenant_id)
    latest_job_payload = _ownify_status_payload(latest_job) if latest_job else None
    kb = kb_manager.registry.get_kb(kb_id)
    if not kb:
        return OwnifyAIStatusResponse(
            status="success",
            tenant_id=tenant_id,
            kb_id=kb_id,
            exists=False,
            latest_job=latest_job_payload,
        )

    info = await run_in_named_executor(
        index_executor,
        kb_manager.get_kb_info,
        kb_id,
        timeout_seconds=ADMIN_WRITE_TIMEOUT_SECONDS,
    )
    return OwnifyAIStatusResponse(
        status="success",
        tenant_id=tenant_id,
        kb_id=kb_id,
        exists=True,
        kb=info.get("kb"),
        collection=info.get("collection") or {},
        latest_job=latest_job_payload,
    )


@app.post("/ownify/tenants/{tenant_id}/ai/session/new", response_model=SessionResponse)
async def ownify_create_session(
    tenant_id: str,
    background_tasks: BackgroundTasks,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    if multi_kb_pipeline is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    kb_id = _validate_ownify_tenant_id(tenant_id)
    if kb_manager is not None and not kb_manager.registry.exists(kb_id):
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")

    return await create_session(kb_id, background_tasks)


@app.post(
    "/ownify/tenants/{tenant_id}/ai/query/jobs",
    response_model=AsyncQueryJobAcceptedResponse,
    status_code=202,
)
async def ownify_submit_query_job(
    tenant_id: str,
    request: AsyncQueryJobRequest,
    http_request: Request,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    kb_id = _validate_ownify_tenant_id(tenant_id)
    if kb_manager is not None and not kb_manager.registry.exists(kb_id):
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")

    return await submit_query_job(kb_id, request, http_request)


@app.get(
    "/ownify/tenants/{tenant_id}/ai/query/jobs/{job_id}",
    response_model=AsyncQueryJobStatusResponse,
)
async def ownify_get_query_job_status(
    tenant_id: str,
    job_id: str,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    kb_id = _validate_ownify_tenant_id(tenant_id)
    return await get_query_job_status(kb_id, job_id)


@app.post("/ownify/connector", response_model=QueryResponse)
async def ownify_connector_query(
    request: ConnectorQueryRequest,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    if not request.connector:
        raise HTTPException(status_code=400, detail="Missing connector field")
    connector = request.connector.lower().strip()
    if connector not in ("email", "drive", "calendar", "sheets", "docs", "presentation", "slack", "notion"):
        raise HTTPException(status_code=400, detail=f"Unsupported connector: {request.connector}")

    kb_id = _validate_ownify_tenant_id(request.tenant_id)
    if kb_manager is not None and not kb_manager.registry.exists(kb_id):
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")
    
    if request.session_id:
        if not multi_kb_pipeline.session_exists(kb_id, request.session_id):
            raise HTTPException(status_code=404, detail="Session not found for this KB")

    request_id = uuid.uuid4().hex
    timeout_seconds = QUERY_TIMEOUT_SECONDS
    cancel_event = threading.Event()

    query_req = QueryRequest(
        query=request.query,
        session_id=request.session_id,
        user_id=request.user_id,
        connector=connector,
    )

    res = await _execute_query_request_internal(
        kb_id,
        query_req,
        request_id=request_id,
        session_id=request.session_id,
        user_id=request.user_id,
        cancel_event=cancel_event,
        timeout_seconds=timeout_seconds,
    )
    return res


@app.post("/ownify/connector/{connector_type}", response_model=QueryResponse)
async def ownify_individual_connector_query(
    connector_type: str,
    request: ConnectorRequest,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    connector = connector_type.lower().strip()
    if connector not in ("email", "drive", "calendar", "sheets", "docs", "presentation", "slack", "notion"):
        raise HTTPException(status_code=400, detail=f"Unsupported connector: {connector_type}")

    kb_id = _validate_ownify_tenant_id(request.tenant_id)
    if kb_manager is not None and not kb_manager.registry.exists(kb_id):
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")
    
    if request.session_id:
        if not multi_kb_pipeline.session_exists(kb_id, request.session_id):
            raise HTTPException(status_code=404, detail="Session not found for this KB")

    request_id = uuid.uuid4().hex
    timeout_seconds = QUERY_TIMEOUT_SECONDS
    cancel_event = threading.Event()

    query_req = QueryRequest(
        query=request.query,
        session_id=request.session_id,
        user_id=request.user_id,
        connector=connector,
    )

    res = await _execute_query_request_internal(
        kb_id,
        query_req,
        request_id=request_id,
        session_id=request.session_id,
        user_id=request.user_id,
        cancel_event=cancel_event,
        timeout_seconds=timeout_seconds,
    )
    return res


# =============================================================================
# Query & Sessions
# =============================================================================

@app.post("/kb/{kb_id}/session/new", response_model=SessionResponse)
async def create_session(
    kb_id: str,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Query(
        None,
        description="Optional user identifier used to prime autocomplete cache for connector navigation",
    ),
    connector: Optional[str] = Query(
        None,
        description="Optional connector name to refresh autocomplete data for: slack, google, notion",
    ),
    auth_token: Optional[str] = Query(
        None,
        description="Optional auth token override used for background connector sync",
    ),
):
    if multi_kb_pipeline is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return _create_session_core(kb_id, user_id=user_id, connector=connector, auth_token=auth_token, background_tasks=background_tasks)


def _create_session_core(kb_id: str, user_id: Optional[str] = None, connector: Optional[str] = None, auth_token: Optional[str] = None, background_tasks: Optional[BackgroundTasks] = None) -> SessionResponse:
    if multi_kb_pipeline is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        result = multi_kb_pipeline.create_session(kb_id)
        if user_id and connector and background_tasks is not None:
            background_tasks.add_task(
                refresh_user_workspace_cache,
                user_id=user_id,
                connector=str(connector),
                auth_token=auth_token,
            )
        return SessionResponse(
            status="success",
            session_id=result["session_id"],
            message="New conversation session created",
            created_at=datetime.now().isoformat(),
            turn_count=0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/kb/{kb_id}/query", response_model=QueryResponse)
async def query_kb(kb_id: str, request: QueryRequest):
    if multi_kb_pipeline is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        if request.session_id:
            if not multi_kb_pipeline.session_exists(kb_id, request.session_id):
                raise HTTPException(status_code=404, detail="Session not found for this KB")

        request_id = uuid.uuid4().hex
        timeout_seconds = QUERY_TIMEOUT_SECONDS
        cancel_event = threading.Event()

        return await _execute_query_request_internal(
            kb_id,
            request,
            request_id=request_id,
            session_id=request.session_id,
            user_id=request.user_id,
            cancel_event=cancel_event,
            timeout_seconds=timeout_seconds,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/kb/{kb_id}/session/{session_id}/documents")
async def upload_session_document(
    kb_id: str,
    session_id: str,
    file: UploadFile = File(...),
):
    # 1. Size guard
    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_BYTES / (1024 * 1024):.1f}MB",
        )

    # 2. Text extraction
    import tempfile
    from pathlib import Path
    
    suffix = Path(file.filename).suffix if file.filename else ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    # 2. Document Processing (offloaded to index_executor to avoid blocking the event loop)
    try:
        def process_document(path, fname):
            docs = load_document(path)
            raw_text = "\n\n".join(doc.content for doc in docs)
            if not raw_text.strip():
                return None, None

            chunker = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
                keep_separator=True
            )
            chunk_texts = chunker.split_text(raw_text)

            # Cap chunks to avoid runaway latency on huge docs
            chunk_texts = chunk_texts[:200]

            # Single true batch call — all chunks in one model forward pass
            embeddings = multi_kb_pipeline.shared_resources.index_embedder.generate_embeddings_batch(
                chunk_texts
            )
            return chunk_texts, embeddings.tolist()

        chunk_texts, embeddings_list = await run_in_named_executor(
            index_executor,
            process_document,
            tmp_path,
            file.filename
        )
        
        if chunk_texts is None:
             raise HTTPException(status_code=422, detail="No readable text found in document")

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 5. Serialization and Storage
    chunks = []
    for i, (text, vector) in enumerate(zip(chunk_texts, embeddings_list)):
        chunks.append({
            "id": f"{session_id}_{i}",
            "text": text,
            "embedding": vector,
            "metadata": {
                "session_id": session_id,
                "filename": file.filename,
                "chunk_index": i
            }
        })

    file_meta = {
        "filename": file.filename,
        "upload_time": datetime.utcnow().isoformat() + "Z",
        "total_chunks": len(chunks)
    }

    redis_conn = multi_kb_pipeline.shared_resources.redis_connection
    if not redis_conn:
        raise HTTPException(status_code=503, detail="Redis connection unavailable for session storage")

    try:
        redis_client = redis_conn.client()
        store_session_docs(redis_client, session_id, chunks, file_meta)
    except Exception as e:
        logger.error(f"Redis storage failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to store session documents")

    return {
        "status": "success",
        "session_id": session_id,
        "filename": file.filename,
        "chunks_stored": len(chunks),
        "upload_time": file_meta["upload_time"],
        "expires_in_seconds": SESSION_TTL_SECONDS,
    }


@app.get("/kb/{kb_id}/session/{session_id}/documents")
async def get_session_document_info(
    kb_id: str,
    session_id: str,
):
    redis_conn = multi_kb_pipeline.shared_resources.redis_connection
    if not redis_conn:
        raise HTTPException(status_code=503, detail="Redis connection unavailable")

    try:
        redis_client = redis_conn.client()
        doc_info = get_session_doc_info(redis_client, session_id)
    except Exception as e:
        logger.error(f"Failed to read session documents for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read session documents")

    if not doc_info:
        return {
            "status": "empty",
            "session_id": session_id,
            "document": None,
        }

    return {
        "status": "ready",
        "session_id": session_id,
        "document": doc_info,
    }


@app.delete("/kb/{kb_id}/session/{session_id}/documents")
async def delete_session_documents(
    kb_id: str,
    session_id: str,
):
    redis_conn = multi_kb_pipeline.shared_resources.redis_connection
    if not redis_conn:
        raise HTTPException(status_code=503, detail="Redis connection unavailable")

    try:
        redis_client = redis_conn.client()
        delete_session_docs(redis_client, session_id)
    except Exception as e:
        logger.error(f"Redis deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session documents")
    return {"status": "success", "session_id": session_id}


@app.post("/ownify/tenants/{tenant_id}/ai/session/{session_id}/documents")
async def ownify_upload_session_document(
    tenant_id: str,
    session_id: str,
    file: UploadFile = File(...),
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    kb_id = _validate_ownify_tenant_id(tenant_id)
    if kb_manager is not None and not kb_manager.registry.exists(kb_id):
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")
    
    return await upload_session_document(kb_id, session_id, file)


@app.get("/ownify/tenants/{tenant_id}/ai/session/{session_id}/documents")
async def ownify_get_session_document_info(
    tenant_id: str,
    session_id: str,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    kb_id = _validate_ownify_tenant_id(tenant_id)
    if kb_manager is not None and not kb_manager.registry.exists(kb_id):
        raise HTTPException(status_code=404, detail=f"KB not found: {kb_id}")
    return await get_session_document_info(kb_id, session_id)


@app.delete("/ownify/tenants/{tenant_id}/ai/session/{session_id}/documents")
async def ownify_delete_session_documents(
    tenant_id: str,
    session_id: str,
    _auth: None = Depends(require_ownify_provisioning_auth),
):
    kb_id = _validate_ownify_tenant_id(tenant_id)
    return await delete_session_documents(kb_id, session_id)


@app.post("/kb/{kb_id}/query/jobs", response_model=AsyncQueryJobAcceptedResponse, status_code=202)
async def submit_query_job(kb_id: str, request: AsyncQueryJobRequest, http_request: Request):
    if multi_kb_pipeline is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    if not ASYNC_QUERY_JOBS_ENABLED:
        raise HTTPException(status_code=409, detail="Async query jobs are disabled")
    if _async_query_job_queue is None:
        raise HTTPException(status_code=503, detail="Async job queue not initialized")

    if request.session_id and not multi_kb_pipeline.session_exists(kb_id, request.session_id):
        raise HTTPException(status_code=404, detail="Session not found for this KB")

    timeout_seconds = request.timeout_seconds or ASYNC_QUERY_JOB_DEFAULT_TIMEOUT_SECONDS
    timeout_seconds = min(timeout_seconds, ASYNC_QUERY_JOB_MAX_TIMEOUT_SECONDS)

    job_id = uuid.uuid4().hex
    request_id = uuid.uuid4().hex

    job_record = AsyncQueryJobRecord(
        job_id=job_id,
        kb_id=kb_id,
        request=request,
        request_id=request_id,
        session_id=request.session_id,
        user_id=request.user_id,
        timeout_seconds=timeout_seconds,
    )

    async with _async_query_jobs_lock:
        _async_query_jobs[job_id] = job_record

    try:
        _async_query_job_queue.put_nowait(job_id)
    except asyncio.QueueFull:
        async with _async_query_jobs_lock:
            _async_query_jobs.pop(job_id, None)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "query_overloaded",
                "message": "Async job queue is full. Please retry shortly.",
                "request_id": request_id,
            },
            headers={"Retry-After": str(QUERY_RETRY_AFTER_SECONDS)}
        )

    return AsyncQueryJobAcceptedResponse(
        status="accepted",
        job_id=job_id,
        kb_id=kb_id,
        request_id=request_id,
        queued_at=datetime.utcnow().isoformat() + "Z",
        timeout_seconds=timeout_seconds,
    )


@app.get("/kb/{kb_id}/query/jobs/{job_id}", response_model=AsyncQueryJobStatusResponse)
async def get_query_job_status(kb_id: str, job_id: str):
    job = await _get_job(job_id)
    if job is None or job.kb_id != kb_id:
        raise HTTPException(status_code=404, detail="Job not found")

    payload = _job_status_payload(job)
    return AsyncQueryJobStatusResponse(**payload)


@app.post("/kb/{kb_id}/query/jobs/{job_id}/cancel", response_model=AsyncQueryJobCancelResponse)
async def cancel_query_job(kb_id: str, job_id: str):
    job = await _get_job(job_id)
    if job is None or job.kb_id != kb_id:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in QUERY_JOB_TERMINAL_STATES:
        return AsyncQueryJobCancelResponse(
            status="success",
            job_id=job_id,
            kb_id=kb_id,
            job_status=job.status,
            message="Job already completed",
        )

    job.cancel_event.set()
    await _set_job_status(
        job_id,
        status=QUERY_JOB_STATUS_CANCELLING if job.status == QUERY_JOB_STATUS_RUNNING else QUERY_JOB_STATUS_CANCELLED,
        cancellation_requested=True,
        completed_at=time.time() if job.status == QUERY_JOB_STATUS_QUEUED else None,
        error={
            "error": "job_cancelled",
            "message": "Cancellation requested",
        } if job.status == QUERY_JOB_STATUS_QUEUED else None,
    )

    return AsyncQueryJobCancelResponse(
        status="success",
        job_id=job_id,
        kb_id=kb_id,
        job_status=QUERY_JOB_STATUS_CANCELLING if job.status == QUERY_JOB_STATUS_RUNNING else QUERY_JOB_STATUS_CANCELLED,
        message="Cancellation requested",
    )


@app.get("/kb/{kb_id}/session/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(kb_id: str, session_id: str, limit: Optional[int] = Query(None, ge=1)):
    if multi_kb_pipeline is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        if not multi_kb_pipeline.session_exists(kb_id, session_id):
            raise HTTPException(status_code=404, detail="Session not found for this KB")

        history = multi_kb_pipeline.get_session_history(kb_id, session_id, n=limit)

        turns = []
        for turn in history:
            turns.append(ConversationTurnResponse(
                turn_id=turn.turn_id,
                timestamp=turn.timestamp,
                query=turn.query,
                reformulated_query=turn.reformulated_query if turn.reformulated_query != turn.query else None,
                answer=turn.answer,
                citations=_normalize_citations_sources(turn.citations),
                entities=turn.entities_mentioned,
                confidence=turn.confidence,
            ))

        return SessionHistoryResponse(
            status="success",
            session_id=session_id,
            turn_count=len(turns),
            history=turns,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to get session history: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/kb/{kb_id}/session/{session_id}")
async def delete_session(kb_id: str, session_id: str):
    if multi_kb_pipeline is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        success = multi_kb_pipeline.delete_session(kb_id, session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found for this KB")
        cleanup_session_lock(kb_id, session_id)
        return {"status": "success", "session_id": session_id}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# =============================================================================
# LLM Backend Control Endpoints
# =============================================================================

class BackendConfigRequest(BaseModel):
    backend: str  # "local" or "api"
    model_name: Optional[str] = None  # e.g., "gemini-2.5-flash-lite"

class BackendConfigResponse(BaseModel):
    status: str
    active_backend: str
    api_model_name: str
    local_model_loaded: bool

@app.get(
    "/ownify/ai/backend",
    response_model=BackendConfigResponse,
    tags=["Admin"]
)
async def get_llm_backend_status(
    _auth: None = Depends(require_ownify_provisioning_auth)
):
    """Get active LLM backend status."""
    if multi_kb_pipeline is None or getattr(multi_kb_pipeline, "shared_resources", None) is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    generator = multi_kb_pipeline.shared_resources.llm_generator
    return BackendConfigResponse(
        status="success",
        active_backend=generator.backend,
        api_model_name=generator.api_model_name,
        local_model_loaded=generator.model is not None
    )

@app.post(
    "/ownify/ai/backend",
    response_model=BackendConfigResponse,
    tags=["Admin"]
)
async def toggle_llm_backend(
    request: BackendConfigRequest,
    _auth: None = Depends(require_ownify_provisioning_auth)
):
    """Toggle LLM backend dynamically."""
    if multi_kb_pipeline is None or getattr(multi_kb_pipeline, "shared_resources", None) is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    backend_val = request.backend.lower()
    if backend_val == "api":
        raise HTTPException(status_code=400, detail="API backend is disabled. Only 'local' is supported.")
    elif backend_val != "local":
        raise HTTPException(status_code=400, detail="Backend must be 'local'")
        
    generator = multi_kb_pipeline.shared_resources.llm_generator
    
    # Update backend settings
    generator.backend = "local"
        
    # If switching to local, trigger loading immediately to avoid cold start on next query
    if generator.model is None:
        try:
            await run_in_named_executor(
                index_executor,  # Reuse index_executor to load in background thread pool safely
                generator._load_model
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load local model: {str(e)}")
            
    return BackendConfigResponse(
        status="success",
        active_backend=generator.backend,
        api_model_name=generator.api_model_name,
        local_model_loaded=generator.model is not None
    )


# =============================================================================
# Exception Handler
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc),
        },
    )


# =============================================================================
# Server Startup
# =============================================================================


def start_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    import uvicorn

    print("=" * 80)
    print("STARTING MULTI-KB RAG API SERVER")
    print("=" * 80)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Workers: {workers}")
    print()
    print("API will be accessible at:")
    print(f"   http://{host}:{port}")
    print(f"   http://localhost:{port}")
    print("=" * 80 + "\n")

    uvicorn.run(
        "src.api.multi_kb_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Multi-KB RAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()
    start_server(host=args.host, port=args.port, workers=args.workers)
