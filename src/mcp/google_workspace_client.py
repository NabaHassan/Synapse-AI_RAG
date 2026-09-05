"""
Google Workspace MCP Client — Approach A (Google Official Remote Servers).

Handles:
  - OAuth 2.0 flow (multi-user, per-user tokens in Redis)
  - Token refresh (automatic via google-auth)
  - MCP tool calls to Gmail, Drive, Calendar remote endpoints
  - Keyword-based intent detection (no LLM required)

Services supported (read-only):
  - Gmail    → https://gmailmcp.googleapis.com/mcp/v1
  - Drive    → https://drivemcp.googleapis.com/mcp/v1
  - Calendar → https://calendarmcp.googleapis.com/mcp/v1
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

logger = logging.getLogger(__name__)

# Drive /files/{id}/export only works for Google Docs Editor (native) files.
_EXPORT_UNSUPPORTED_MSG = "Export only supports Docs Editors files."


def _is_expected_export_failure(status_code: int, body: str) -> bool:
    """Non-Google-native uploads always 403 on /export — not a real error."""
    return status_code == 403 and _EXPORT_UNSUPPORTED_MSG in body

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SERVICES: Dict[str, Dict[str, Any]] = {
    "gmail": {
        "mcp_url": "https://gmailmcp.googleapis.com/mcp/v1",
        "scopes": [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
        ],
    },
    "drive": {
        "mcp_url": "https://drivemcp.googleapis.com/mcp/v1",
        "scopes": [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets.readonly",
        ],
    },
    "calendar": {
        "mcp_url": "https://calendarmcp.googleapis.com/mcp/v1",
        "scopes": [
            "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
            "https://www.googleapis.com/auth/calendar.events.freebusy",
            "https://www.googleapis.com/auth/calendar.events.readonly",
        ],
    },
    "sheets": {
        "mcp_url": "https://drivemcp.googleapis.com/mcp/v1",
        "scopes": [
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/spreadsheets.readonly",
        ],
    },
    "docs": {
        "mcp_url": "https://drivemcp.googleapis.com/mcp/v1",
        "scopes": [
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.file",
        ],
    },
    "presentation": {
        "mcp_url": "https://drivemcp.googleapis.com/mcp/v1",
        "scopes": [
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.file",
        ],
    },
}

# Token TTL in Redis (seconds). Refresh tokens last much longer; access tokens
# are refreshed automatically by google-auth when expired.
_TOKEN_TTL_SECONDS = int(os.getenv("GOOGLE_MCP_TOKEN_TTL", "86400"))  # 24h

# Drive content limits for LLM context injection
_DRIVE_FILE_CONTENT_CHAR_LIMIT = int(os.getenv("GOOGLE_DRIVE_FILE_CONTENT_CHAR_LIMIT", "4000"))
_DRIVE_MULTI_FILE_TOTAL_CHAR_LIMIT = int(os.getenv("GOOGLE_DRIVE_MULTI_FILE_TOTAL_CHAR_LIMIT", "12000"))
_DRIVE_MULTI_FILE_MAX_CONTENT_FETCH = int(os.getenv("GOOGLE_DRIVE_MULTI_FILE_MAX_FETCH", "3"))


def _truncate_drive_content(text: str, limit: int) -> str:
    """Truncate extracted Drive file text for prompt safety."""
    if not text or len(text) <= limit:
        return text or ""
    omitted = len(text) - limit
    logger.info("Truncating Drive content from %d to %d chars (%d omitted)", len(text), limit, omitted)
    return f"{text[:limit]}\n...[truncated, {omitted} more characters]"

# Trailing phrases users add after a file title (e.g. "@cover letter what is in the file")
_DRIVE_QUERY_SUFFIX_PATTERNS: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\s+what\s+is\s+in\s+the\s+file\s*$",
        r"\s+what\s+is\s+in\s+this\s+file\s*$",
        r"\s+what\s+is\s+the\s+file\s+about\s*$",
        r"\s+what\s+is\s+this\s+file\s+about\s*$",
        r"\s+what\s+is\s+the\s+content(?:\s+of\s+the\s+file)?\s*$",
        r"\s+what\s+is\s+the\s+file\s+consists?\s+of\s*$",
        r"\s+what\s+does\s+(?:this\s+)?file\s+(?:say|contain)\s*$",
        r"\s+tell\s+me\s+about\s+(?:the\s+)?file\s*$",
        r"\s+tell\s+me\s+about\s+(?:this\s+)?file\s*$",
        r"\s+what\s+is\s+in\s+the\s+document\s*$",
        r"\s+what\s+is\s+the\s+document\s+about\s*$",
        r"\s+summarize\s+(?:this\s+)?file\s*$",
        r"\s+summarize\s+(?:the\s+)?file\s*$",
        r"\s+what\s+is\s+mentioned\s*$",
        r"\s+what\s+(?:are|were)\s+the\s+results(?:\s+in\s+the\s+report)?\s*$",
        r"\s+what\s+where\s+the\s+results.*$",
        r"\s+mentioned\s+in\s+the\s+file\s*$",
        r"\s+in\s+the\s+report\s*$",
    )
]

_DRIVE_QUESTION_TAIL_SPLIT = re.compile(
    r"\s+(?:what|where|how|why|when|which|who|tell\s+me|summarize|describe|explain)\s+",
    re.IGNORECASE,
)

_DRIVE_QUERY_PREFIXES: Tuple[str, ...] = (
    "find file",
    "find document",
    "find doc",
    "search file",
    "search document",
    "search doc",
    "file about",
    "document about",
    "doc about",
    "show me",
    "show",
    "get",
    "list",
    "find",
    "search",
    "look for",
    "read",
    "open",
)

_FILENAME_WITH_EXT_PATTERN = re.compile(
    r"^([\w\s\-\.]+)\.(pdf|docx|doc|xlsx|xls|pptx|ppt|txt|csv|json|md|html|png|jpg|jpeg|gif)\b",
    re.IGNORECASE,
)


def extract_drive_search_term(query: str) -> str:
    """
    Pull a Drive file title out of conversational text.

    Examples:
      "@cover letter what is in the file" -> "cover letter"
      "Flower1.pdf tell me about the file" -> "Flower1.pdf"
    """
    if not query or not str(query).strip():
        return ""

    text = str(query).strip().lstrip("@").strip()
    if not text:
        return ""

    ext_match = _FILENAME_WITH_EXT_PATTERN.match(text)
    if ext_match:
        term = ext_match.group(0).strip()
        logger.info("extract_drive_search_term: filename with extension=%r", term)
        return term

    lowered = text.lower()
    for suffix_re in _DRIVE_QUERY_SUFFIX_PATTERNS:
        text = suffix_re.sub("", text).strip()
        lowered = text.lower()

    for prefix in _DRIVE_QUERY_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip().strip("'\"")
            lowered = text.lower()
            break

    tail_parts = _DRIVE_QUESTION_TAIL_SPLIT.split(text, maxsplit=1)
    if tail_parts and tail_parts[0].strip():
        text = tail_parts[0].strip()
        logger.info("extract_drive_search_term: stripped question tail -> %r", text)

    term = text.strip().strip("'\"")
    if term:
        logger.info("extract_drive_search_term: conversational title=%r (from query=%r)", term, query[:80])
    return term


def pick_best_drive_file_match(files: List[Dict[str, Any]], search_term: str) -> Optional[Dict[str, Any]]:
    """Prefer exact title match, then substring match, when Drive returns multiple files."""
    if not files or not search_term:
        return None

    needle = search_term.strip().lower()
    if not needle:
        return None

    exact = [f for f in files if (f.get("name") or "").strip().lower() == needle]
    if len(exact) == 1:
        logger.info("pick_best_drive_file_match: exact name match %r", exact[0].get("name"))
        return exact[0]
    if len(exact) > 1:
        logger.info("pick_best_drive_file_match: %d exact matches, using first", len(exact))
        return exact[0]

    partial = [f for f in files if needle in (f.get("name") or "").strip().lower()]
    if len(partial) == 1:
        logger.info("pick_best_drive_file_match: single partial match %r", partial[0].get("name"))
        return partial[0]
    if len(partial) > 1:
        partial.sort(key=lambda f: len((f.get("name") or "")))
        logger.info("pick_best_drive_file_match: %d partial matches, shortest name %r", len(partial), partial[0].get("name"))
        return partial[0]

    return None


# ---------------------------------------------------------------------------
# Intent patterns — pure regex, no LLM required
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: List[Tuple[str, str, str, Dict[str, Any]]] = [
    # (service, tool, regex_pattern, extra_params)

    # ── Gmail ────────────────────────────────────────────────────────────────
    ("gmail", "list_recent_inbox",
     r"\b(recent emails?|latest emails?|my emails today|emails today|check my inbox|"
     r"what emails|inbox today|mail today)\b",
     {}),
    ("gmail", "search_sent",
     r"\b(sent mail|sent email|emails? i sent|did i email|did i send|in sent)\b",
     {}),
    ("gmail", "get_thread",
     r"\b(thread[_\s-]?id|that email|that thread|tell me more about (?:the|that) (?:email|thread))\b",
     {}),
    ("gmail", "search_threads",
     r"\b(email|emails|inbox|mail|unread|message|messages|thread|threads|"
     r"sent|received|from\s+\w|subject)\b",
     {}),

    # ── Drive ────────────────────────────────────────────────────────────────
    ("drive", "search_files",
     r"\b(drive|file|files|document|documents|folder|folders|spreadsheet|"
     r"shared with me|upload|doc|docs|pdf|report)\b",
     {}),

    # ── Calendar ─────────────────────────────────────────────────────────────
    ("calendar", "list_events",
     r"\b(calendar|meeting|meetings|event|events|schedule|appointment|"
     r"busy|free|availability|call|sync|invite|invites)\b",
     {}),
]

# Pre-compile patterns for speed
_COMPILED_PATTERNS = [
    (svc, tool, re.compile(pat, re.IGNORECASE), params)
    for svc, tool, pat, params in _INTENT_PATTERNS
]


# ---------------------------------------------------------------------------
# GoogleWorkspaceMCPClient
# ---------------------------------------------------------------------------

class GoogleWorkspaceMCPClient:
    """
    Lightweight MCP client for Google Workspace.

    Stores one OAuth token set per (user_id, service) in Redis.
    All MCP calls are synchronous (called from ThreadPoolExecutor in server).
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        redis_client,  # redis.Redis instance from existing connection
        redis_key_prefix: str = "synapse",
        token_ttl_seconds: int = _TOKEN_TTL_SECONDS,
        services: Optional[List[str]] = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._redis = redis_client
        self._key_prefix = redis_key_prefix
        self._token_ttl = token_ttl_seconds
        self._services = services or list(_SERVICES.keys())

        logger.info(
            "GoogleWorkspaceMCPClient initialized (services=%s, redirect_uri=%s)",
            self._services,
            self._redirect_uri,
        )

    # -----------------------------------------------------------------------
    # OAuth helpers
    # -----------------------------------------------------------------------

    def _client_config(self) -> Dict[str, Any]:
        """Build google-auth-oauthlib client config dict.

        The Gmail/Drive/Calendar MCP APIs (gmailmcp.googleapis.com etc.) require
        tokens issued via an 'installed' (Desktop app) OAuth 2.0 client.  Using a
        'web' client type causes those endpoints to return 'The caller does not
        have permission' even when all scopes are correctly granted.
        """
        return {
            "installed": {
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self._redirect_uri],
            }
        }

    def _all_scopes(self, services: Optional[List[str]] = None) -> List[str]:
        """Collect all scopes for the requested services."""
        svcs = services or self._services
        scopes: List[str] = ["openid", "https://www.googleapis.com/auth/userinfo.email"]
        for svc in svcs:
            if svc in _SERVICES:
                scopes.extend(_SERVICES[svc]["scopes"])
        return list(dict.fromkeys(scopes))  # deduplicate, preserve order

    def get_auth_url(
        self,
        user_id: str,
        services: Optional[List[str]] = None,
    ) -> str:
        """
        Generate an OAuth 2.0 authorization URL.

        The `state` encodes user_id + requested services so the callback
        can store tokens under the right Redis key.
        """
        import urllib.parse

        svcs = services or self._services
        state_payload = json.dumps({"user_id": user_id, "services": svcs})
        state = urllib.parse.quote(state_payload)

        flow = Flow.from_client_config(
            self._client_config(),
            scopes=self._all_scopes(svcs),
            redirect_uri=self._redirect_uri,
        )
        flow.autogenerate_code_verifier = False
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state,
        )
        logger.info("Generated auth URL for user_id=%s services=%s", user_id, svcs)
        return auth_url

    def exchange_code(
        self,
        code: str,
        state: str,
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for tokens and persist to Redis.

        Returns dict with user_id and list of authorized services.
        """
        import urllib.parse

        try:
            state_payload = json.loads(urllib.parse.unquote(state))
            user_id: str = state_payload["user_id"]
            services: List[str] = state_payload.get("services", self._services)
        except Exception as exc:
            raise ValueError(f"Invalid OAuth state parameter: {exc}") from exc

        flow = Flow.from_client_config(
            self._client_config(),
            scopes=self._all_scopes(services),
            redirect_uri=self._redirect_uri,
            state=state,
        )
        flow.autogenerate_code_verifier = False
        # Suppress HTTPS requirement for localhost development
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
        # Add this line to prevent strict scope change validation crashes:
        os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
        flow.fetch_token(code=code)

        creds = flow.credentials
        token_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes) if creds.scopes else self._all_scopes(services),
            "expiry": creds.expiry.isoformat() if creds.expiry else None,
        }

        # Store one token record per service under its own Redis key
        for svc in services:
            if svc in _SERVICES:
                redis_key = self._token_key(user_id, svc)
                self._redis.setex(redis_key, self._token_ttl, json.dumps(token_data))
                logger.info("Stored MCP token for user_id=%s service=%s", user_id, svc)

        return {"user_id": user_id, "services_authorized": services, "access_token": creds.token}

    # -----------------------------------------------------------------------
    # Token management
    # -----------------------------------------------------------------------

    def _token_key(self, user_id: str, service: str) -> str:
        return f"{self._key_prefix}:mcp:token:{user_id}:{service}"

    def _load_credentials(self, user_id: str, service: str) -> Optional[Credentials]:
        """Load and optionally refresh credentials from Redis."""
        raw = self._redis.get(self._token_key(user_id, service))
        if not raw:
            return None

        try:
            data = json.loads(raw)
        except Exception:
            return None

        from datetime import datetime, timezone

        expiry = None
        if data.get("expiry"):
            try:
                expiry = datetime.fromisoformat(data["expiry"])
                if expiry.tzinfo is not None:
                    expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                pass

        creds = Credentials(
            token=data.get("token"),
            refresh_token=data.get("refresh_token"),
            token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=data.get("client_id", self._client_id),
            client_secret=data.get("client_secret", self._client_secret),
            scopes=data.get("scopes"),
            expiry=expiry,
        )

        # Refresh if expired
        if creds.expired and creds.refresh_token:
            try:
                import google.auth.transport.requests as google_requests
                creds.refresh(google_requests.Request())
                # Persist refreshed token
                data["token"] = creds.token
                data["expiry"] = creds.expiry.isoformat() if creds.expiry else None
                self._redis.setex(
                    self._token_key(user_id, service),
                    self._token_ttl,
                    json.dumps(data),
                )
                logger.debug("Refreshed MCP token for user_id=%s service=%s", user_id, service)
            except Exception as exc:
                logger.warning("Token refresh failed for user_id=%s service=%s: %s", user_id, service, exc)
                return None

        return creds

    def is_authenticated(self, user_id: str, service: str) -> bool:
        """Return True if the user has a valid (or refreshable) token for the service."""
        if not user_id:
            return False
        check_svc = service
        if service in ("sheets", "docs", "presentation"):
            check_svc = "drive"
        if check_svc not in _SERVICES:
            return False
        creds = self._load_credentials(user_id, check_svc)
        return creds is not None and (creds.valid or bool(creds.refresh_token))

    def revoke_tokens(self, user_id: str) -> None:
        """Delete all MCP tokens for a user from Redis."""
        for svc in _SERVICES:
            self._redis.delete(self._token_key(user_id, svc))
        logger.info("Revoked all MCP tokens for user_id=%s", user_id)

    def auth_status(self, user_id: str) -> Dict[str, bool]:
        """Return authentication status for each service."""
        return {svc: self.is_authenticated(user_id, svc) for svc in self._services}

    # -----------------------------------------------------------------------
    # Intent detection
    # -----------------------------------------------------------------------

    def detect_intent(
        self, query: str
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """
        Detect if the query maps to a Workspace MCP tool.

        Returns (service, tool, params) or (None, None, {}) if no match.
        Automatic intent detection has been completely disabled.
        """
        return None, None, {}

    # -----------------------------------------------------------------------
    # MCP tool call
    # -----------------------------------------------------------------------

    def call_tool(
        self,
        user_id: str,
        service: str,
        tool: str,
        params: Dict[str, Any],
        timeout_seconds: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Call a tool on the Google remote MCP server (or local REST fallback if enabled).

        Uses the MCP JSON-RPC 2.0 wire format over HTTP.
        Returns the parsed tool result or raises on error.
        """
        auth_svc = service
        if service in ("sheets", "docs", "presentation"):
            auth_svc = "drive"
        creds = self._load_credentials(user_id, auth_svc)

        # ── DEBUG: token diagnostics (safe — never logs the full token) ──────
        logger.info(
            "DEBUG MCP token check for user_id=%r service=%s: "
            "valid_token=%s token_len=%s token_prefix=%s "
            "expired=%s expiry=%s scopes_count=%s",
            user_id,
            service,
            creds is not None and bool(getattr(creds, "token", None)),
            len(getattr(creds, "token", "") or "") if creds else 0,
            ((getattr(creds, "token", "") or "")[:10] + "...") if creds and creds.token else "None",
            getattr(creds, "expired", "N/A") if creds else "N/A",
            getattr(creds, "expiry", "N/A") if creds else "N/A",
            len(list(getattr(creds, "scopes", None) or [])) if creds else 0,
        )
        # ─────────────────────────────────────────────────────────────────────

        if creds is None:
            raise PermissionError(f"No valid credentials for user={user_id} service={service}")

        # Check if local REST call path is enabled
        use_local_rest = os.getenv("GOOGLE_MCP_USE_LOCAL_REST", "true").lower() == "true"
        if use_local_rest:
            logger.info("Routing Google Workspace tool call %s:%s locally using REST APIs", service, tool)
            try:
                return self._call_tool_local(
                    user_id=user_id,
                    service=service,
                    tool=tool,
                    params=params,
                    creds=creds,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                logger.warning(
                    "Local REST API call failed for %s:%s: %s. Falling back to remote MCP.",
                    service, tool, exc
                )

        mcp_url = _SERVICES[service]["mcp_url"]

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool,
                "arguments": params,
            },
        }

        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
        }

        logger.info(
            "MCP tool call (remote): service=%s tool=%s user_id=%s params=%s",
            service, tool, user_id, json.dumps(params, default=str)[:500]
        )

        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(mcp_url, json=payload, headers=headers)

        if response.status_code == 401:
            raise PermissionError(
                f"MCP auth rejected for user={user_id} service={service}. "
                "Token may have been revoked — user should re-authenticate."
            )

        response.raise_for_status()
        result = response.json()
        logger.info(
            "MCP remote response (service=%s tool=%s): status=%s result_keys=%s content_items=%s",
            service, tool, response.status_code, list(result.keys()),
            len(result.get("result", {}).get("content", []))
        )
        logger.debug("MCP raw response (service=%s tool=%s): %s", service, tool, json.dumps(result, default=str)[:1000])

        if "error" in result:
            err = result["error"]
            raise RuntimeError(
                f"MCP tool error [{err.get('code')}]: {err.get('message', 'unknown')}"
            )

        # Google Workspace MCP servers sometimes return HTTP 200 but embed
        # permission/auth error text inside the content body instead of using
        # the JSON-RPC error field. Detect these and raise so the pipeline
        # falls back to RAG cleanly.
        _PERMISSION_PHRASES = (
            "caller does not have permission",
            "insufficient authentication scopes",
            "request had insufficient authentication",
            "access denied",
            "forbidden",
            "permission denied",
        )
        tool_result = result.get("result", {})
        content_items = tool_result.get("content") or []
        for item in content_items:
            if isinstance(item, dict) and item.get("type") == "text":
                text_lower = (item.get("text") or "").lower()
                if any(phrase in text_lower for phrase in _PERMISSION_PHRASES):
                    logger.warning(
                        "MCP permission error in content body (service=%s tool=%s): %s",
                        service, tool, item.get("text", "")[:200],
                    )
                    raise PermissionError(
                        f"MCP permission denied (service={service} tool={tool}): "
                        f"{item.get('text', '')[:200]}"
                    )

        return tool_result

    # -----------------------------------------------------------------------
    # Local REST API Implementations (Direct calls bypassing cloud MCP)
    # -----------------------------------------------------------------------

    def _call_tool_local(
        self,
        user_id: str,
        service: str,
        tool: str,
        params: Dict[str, Any],
        creds: Any,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        """Dispatch tool calls directly to Google core REST APIs."""
        logger.info(f"=== Google Workspace MCP Tool Call ===")
        logger.info(f"Service: {service}, Tool: {tool}, User: {user_id}")
        logger.info(f"Params: {params}")
        
        token = creds.token
        if not token:
            raise PermissionError(f"No active OAuth token for user_id={user_id}")

        query = params.get("query") or ""
        mime_type = params.get("mime_type")

        if service == "gmail":
            from src.mcp.gmail_search import run_gmail_tool

            embedder = params.pop("_embedder", None)
            result = run_gmail_tool(
                token=token,
                tool=tool,
                params=dict(params),
                timeout=timeout_seconds,
                embedder=embedder,
            )
        elif service == "calendar":
            from src.mcp.calendar_search import run_calendar_search

            timezone = params.get("timezone")
            calendar_id = params.get("calendar_id")
            result = run_calendar_search(
                token=token,
                query=query,
                timeout=timeout_seconds,
                timezone=timezone,
                calendar_id=calendar_id,
            )
        elif service in ("drive", "sheets", "docs", "presentation"):
            if tool == "read_drive_file":
                file_name = params.get("file_name") or query
                query = self.build_targeted_drive_query(file_name)
            result = self._local_drive_search(token, query, timeout_seconds, mime_type=mime_type, params=params)
        else:
            raise NotImplementedError(f"Local fallback not implemented for service: {service}")
        
        items = result.get("items") or []
        content_blocks = result.get("content") or []
        if items:
            logger.info("Result count: %d items", len(items))
        elif content_blocks:
            logger.info(
                "Result: text content (%d block(s), no item list)",
                len(content_blocks),
            )
        else:
            logger.info("Result count: 0 items")
        logger.info(f"=== End Google Workspace MCP Tool Call ===")
        return result

    def build_targeted_drive_query(self, file_name: str) -> str:
        """
        Build an explicit Google Drive query that targets a file title.

        Escapes single quotes and protects against general global search fallback.
        """
        sanitized_name = file_name.strip().replace("'", "\\'")
        if not sanitized_name:
            return "trashed = false"
        return f"name = '{sanitized_name}' and trashed = false"

    def _fetch_sheets_content(self, token: str, spreadsheet_id: str, timeout: float) -> str:
        """Fetch the cell values from the first sheet of a spreadsheet."""
        headers = {"Authorization": f"Bearer {token}"}
        logger.info(f"Fetching Sheets content via API v4 for spreadsheet_id={spreadsheet_id}")

        # Target the entire first sheet by default using a generic range query
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/A1:Z100"

        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                rows = data.get("values", [])
                if not rows:
                    logger.info("Spreadsheet is empty")
                    return "The spreadsheet is currently empty."

                # Format the grid rows cleanly as text lines for the LLM context
                matrix_strings = [", ".join([str(cell) for cell in row]) for row in rows]
                result = "\n".join(matrix_strings)
                logger.info(f"Successfully fetched {len(rows)} rows from spreadsheet, total chars={len(result)}")
                return result
            else:
                logger.warning(f"Sheets API request failed with status {resp.status_code}: {resp.text[:200]}")

        logger.info("Sheets API failed or empty — trying Drive CSV export fallback")
        csv_text = self._export_drive_file(token, spreadsheet_id, "text/csv", timeout)
        if isinstance(csv_text, str) and csv_text.strip():
            return csv_text

        return "[Unable to read spreadsheet values]"

    def _export_drive_file(
        self, token: str, file_id: str, export_mime: str, timeout: float
    ) -> Optional[str]:
        """Export a Drive file (Google-native or Office-compatible) to text or PDF bytes."""
        headers = {"Authorization": f"Bearer {token}"}
        export_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
        logger.info(
            "Drive export attempt file_id=%s export_mime=%s",
            file_id,
            export_mime,
        )
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                export_url, headers=headers, params={"mimeType": export_mime}
            )
            if resp.status_code == 200:
                if export_mime == "application/pdf":
                    logger.info(
                        "Drive PDF export succeeded, size=%s bytes",
                        len(resp.content),
                    )
                    return resp.content
                text = resp.text
                if text and text.strip():
                    logger.info(
                        "Drive export succeeded as %s, chars=%s",
                        export_mime,
                        len(text),
                    )
                    return text
            body_preview = resp.text[:200]
            if _is_expected_export_failure(resp.status_code, resp.text):
                logger.debug(
                    "Drive export not supported for this file type mime=%s (status=%s)",
                    export_mime,
                    resp.status_code,
                )
            else:
                logger.warning(
                    "Drive export failed mime=%s status=%s body=%s",
                    export_mime,
                    resp.status_code,
                    body_preview,
                )
        return None

    def _extract_text_from_exported_pdf(self, pdf_bytes: bytes) -> str:
        from src.mcp.drive_file_extractor import extract_drive_file_text

        extracted = extract_drive_file_text(
            pdf_bytes,
            mime_type="application/pdf",
            filename="export.pdf",
            source_label="Google Drive",
        )
        if extracted.startswith("["):
            return ""
        return extracted

    def _fetch_office_or_binary_content(
        self, token: str, file_id: str, mime_type: str, timeout: float
    ) -> str:
        """
        Read uploaded Office files (pptx/docx/xlsx) via native Drive download.

        Drive /export does not apply to uploaded OpenXML binaries; skip it to avoid 403 noise.
        """
        from src.mcp.drive_file_extractor import extract_drive_file_text

        logger.info(
            "Office/binary content fetch file_id=%s mime_type=%s (download via alt=media)",
            file_id,
            mime_type,
        )

        headers = {"Authorization": f"Bearer {token}"}
        download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(download_url, headers=headers, params={"alt": "media"})
            if resp.status_code != 200:
                logger.warning(
                    "Office file download failed status=%s",
                    resp.status_code,
                )
                return (
                    f"[Unable to read file content for type {mime_type}. "
                    "Open it in Google Drive or convert to Google Docs/Sheets/Slides.]"
                )

        return extract_drive_file_text(
            resp.content,
            mime_type=mime_type,
            filename="",
            source_label="Google Drive",
        )

    def _fetch_slides_content(self, token: str, presentation_id: str, timeout: float) -> str:
        """Export slide text footprints safely using plain text streams."""
        headers = {"Authorization": f"Bearer {token}"}
        export_url = f"https://www.googleapis.com/drive/v3/files/{presentation_id}/export"
        logger.info(f"Exporting Slides content via Drive API for presentation_id={presentation_id}")

        with httpx.Client(timeout=timeout) as client:
            resp = client.get(export_url, headers=headers, params={"mimeType": "text/plain"})
            if resp.status_code == 200:
                content = resp.text
                logger.info(f"Successfully exported presentation, total chars={len(content)}")
                return content
            else:
                logger.warning(f"Slides export failed with status {resp.status_code}: {resp.text[:200]}")

        return "[Unable to read presentation body text strings]"

    def _fetch_file_content(self, token: str, file_id: str, mime_type: str, timeout: float) -> str:
        """
        Fetch text content from a Google Drive file.

        Handles:
        - Google Docs (export to plain text)
        - Google Sheets (via Sheets API v4 for tabular data)
        - Google Slides (export to plain text)
        - Binary text files (txt, md, json) - download directly
        - PDFs and other binary formats (return placeholder)
        """
        headers = {"Authorization": f"Bearer {token}"}
        logger.info(f"Fetching content for file_id={file_id}, mime_type={mime_type}")

        # 1. Google Docs Handler
        if "google-apps.document" in mime_type:
            export_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
            logger.info(f"Exporting Google Doc via: {export_url}")
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(export_url, headers=headers, params={"mimeType": "text/plain"})
                if resp.status_code == 200:
                    content = resp.text
                    logger.info(f"Successfully exported Google Doc, length={len(content)} chars")
                    return content
                else:
                    logger.warning(f"Google Doc export failed with status {resp.status_code}: {resp.text[:200]}")
            return "[Unable to read Google Doc content]"

        # 2. Google Sheets Handler
        elif "google-apps.spreadsheet" in mime_type:
            return self._fetch_sheets_content(token, file_id, timeout)

        # 3. Google Slides Handler
        elif "google-apps.presentation" in mime_type:
            return self._fetch_slides_content(token, file_id, timeout)

        # 3b. Uploaded Microsoft Office / OpenXML (pptx, docx, xlsx on Drive)
        elif any(
            marker in mime_type
            for marker in (
                "openxmlformats-officedocument",
                "application/msword",
                "application/vnd.ms-excel",
                "application/vnd.ms-powerpoint",
            )
        ):
            return self._fetch_office_or_binary_content(token, file_id, mime_type, timeout)

        # 4. Fallback Binary Flat Files (txt, md, json, etc.)
        elif any(t in mime_type for t in ["text/plain", "text/markdown", "application/json", "application/javascript", "text/html"]):
            download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            logger.info(f"Downloading text file via: {download_url}")
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(download_url, headers=headers, params={"alt": "media"})
                if resp.status_code == 200:
                    content = resp.text
                    logger.info(f"Successfully downloaded text file, length={len(content)} chars")
                    return content
                else:
                    logger.warning(f"Text file download failed with status {resp.status_code}: {resp.text[:200]}")
            return "[Unable to read text file content]"

        # 5. PDFs and other binary formats
        elif "pdf" in mime_type.lower():
            logger.info("PDF file detected - attempting content extraction")
            download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(download_url, headers=headers, params={"alt": "media"})
                if resp.status_code == 200:
                    from src.mcp.drive_file_extractor import extract_drive_file_text

                    logger.info("Successfully downloaded PDF, size=%s bytes", len(resp.content))
                    return extract_drive_file_text(
                        resp.content,
                        mime_type=mime_type,
                        filename="file.pdf",
                        source_label="Google Drive",
                    )
                logger.warning(
                    "PDF download failed with status %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
            return "[Unable to read PDF content from Google Drive]"

        logger.warning(f"Unsupported mime_type for content extraction: {mime_type}")
        return f"[Content of type {mime_type} cannot be directly converted into standard context]"

    def _local_drive_search(self, token: str, query: str, timeout: float, mime_type: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"=== Drive Search Start ===")
        logger.info(f"Original query: {query}")
        logger.info(f"MIME type filter: {mime_type}")
        logger.info(f"Timeout: {timeout}s")
        
        # NEW: Direct file fetch if file_id is provided
        if params and params.get("file_id"):
            file_id = params["file_id"]
            logger.info(f"Direct file fetch requested for file_id: {file_id}")
            
            # Fetch file metadata first
            headers = {"Authorization": f"Bearer {token}"}
            with httpx.Client(timeout=timeout) as client:
                meta = client.get(
                    f"https://www.googleapis.com/drive/v3/files/{file_id}",
                    headers=headers,
                    params={"fields": "id,name,mimeType"}
                ).json()
            
            file_text = self._fetch_file_content(token, file_id, meta["mimeType"], timeout)
            truncated = _truncate_drive_content(file_text, _DRIVE_FILE_CONTENT_CHAR_LIMIT)
            result_text = (
                f"Found file context from Google Drive document:\n"
                f"Filename: {meta['name']}\n"
                f"--- Content Begin ---\n{truncated}\n--- Content End ---"
            )
            logger.info(
                "Direct file fetch complete: %s, raw_len=%d truncated_len=%d",
                meta["name"],
                len(file_text),
                len(truncated),
            )
            return {
                "content": [{"type": "text", "text": result_text}],
                "items": [{"name": meta["name"], "id": file_id}],
            }
        
        headers = {"Authorization": f"Bearer {token}"}
        search_term = ""

        # 1. If it's already a built targeted query structure, use it directly
        if query.strip().lower().startswith("name =") or "contains" in query.lower():
            clean_q = query.strip()
        else:
            search_term = extract_drive_search_term(query)
            if search_term:
                term_escaped = search_term.replace("'", "\\'")
                clean_q = f"name = '{term_escaped}' and trashed = false"
            else:
                clean_q = "trashed = false"

        if mime_type:
            if "," in mime_type:
                mime_types = [m.strip() for m in mime_type.split(",")]
                mime_conds = " or ".join(f"mimeType = '{m}'" for m in mime_types)
                clean_q += f" and ({mime_conds})"
            else:
                clean_q += f" and mimeType = '{mime_type}'"

        logger.info(f"Cleaned query for Drive API: {clean_q}")
        logger.info(f"Drive API URL: https://www.googleapis.com/drive/v3/files")
        logger.info(f"Drive API params: pageSize=10, orderBy=createdTime desc, fields=files(id,name,mimeType,webViewLink,createdTime), q={clean_q}")

        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                "https://www.googleapis.com/drive/v3/files",
                headers=headers,
                params={
                    "pageSize": 10,
                    "orderBy": "createdTime desc",
                    "fields": "files(id, name, mimeType, webViewLink, createdTime)",
                    "q": clean_q
                }
            )
            resp.raise_for_status()
            files_data = resp.json()
            files = files_data.get("files") or []
            logger.info(f"Drive API response status: {resp.status_code}")
            logger.info(f"Drive API returned {len(files)} files")

            # Exact title match often misses; retry with contains for conversational titles
            if not files and search_term and "name =" in clean_q:
                term_escaped = search_term.replace("'", "\\'")
                fallback_q = f"name contains '{term_escaped}' and trashed = false"
                if mime_type:
                    if "," in mime_type:
                        mime_types = [m.strip() for m in mime_type.split(",")]
                        mime_conds = " or ".join(f"mimeType = '{m}'" for m in mime_types)
                        fallback_q += f" and ({mime_conds})"
                    else:
                        fallback_q += f" and mimeType = '{mime_type}'"
                logger.info(f"No exact name match; retrying Drive query: {fallback_q}")
                resp = client.get(
                    "https://www.googleapis.com/drive/v3/files",
                    headers=headers,
                    params={
                        "pageSize": 10,
                        "orderBy": "createdTime desc",
                        "fields": "files(id, name, mimeType, webViewLink, createdTime)",
                        "q": fallback_q,
                    },
                )
                resp.raise_for_status()
                files = resp.json().get("files") or []
                logger.info(f"Drive fallback returned {len(files)} files")

            if files:
                logger.info(f"File names: {[f.get('name', 'Unnamed') for f in files[:3]]}{'...' if len(files) > 3 else ''}")

            if not files:
                return {"content": [{"type": "text", "text": "No matching files found in your Google Drive."}]}

            search_term_for_match = extract_drive_search_term(query) if not (
                query.strip().lower().startswith("name =") or "contains" in query.lower()
            ) else ""

            # If the search specifically narrowed down to 1 clear document, read its body!
            if len(files) == 1 or "name =" in query:
                target_file = files[0]
            elif search_term_for_match:
                best = pick_best_drive_file_match(files, search_term_for_match)
                target_file = best
            else:
                target_file = None

            if target_file is not None and (len(files) == 1 or "name =" in query or search_term_for_match):
                logger.info(f"Targeted match detected: fetching content for: {target_file['name']}")
                file_text = self._fetch_file_content(
                    token, target_file["id"], target_file["mimeType"], timeout
                )
                truncated = _truncate_drive_content(file_text, _DRIVE_FILE_CONTENT_CHAR_LIMIT)

                result_text = (
                    f"Found file context from Google Drive document:\n"
                    f"Filename: {target_file['name']}\n"
                    f"--- Content Begin ---\n"
                    f"{truncated}\n"
                    f"--- Content End ---"
                )
                logger.info(
                    "Returning targeted content for %s: raw_len=%d truncated_len=%d",
                    target_file["name"],
                    len(file_text),
                    len(truncated),
                )
                return {
                    "content": [{"type": "text", "text": result_text}],
                    "items": [
                        {
                            "name": target_file["name"],
                            "id": target_file.get("id", ""),
                        }
                    ],
                }

            if len(files) > 1 and search_term_for_match and target_file is None:
                logger.info(
                    "Multiple Drive matches (%d) but no confident title match for %r — listing metadata only",
                    len(files),
                    search_term_for_match,
                )

            summaries = []
            all_contents: List[str] = []
            total_content_chars = 0
            content_fetch_count = 0
            skipped_content_files: List[str] = []

            for f in files:
                name = f.get("name", "Unnamed File")
                mime = f.get("mimeType", "Unknown Type")
                link = f.get("webViewLink", "#")
                created = f.get("createdTime", "Unknown Date")
                file_id = f.get("id", "")
                logger.debug(f"Processing file: name={name}, mimeType={mime}, id={file_id}")

                friendly_type = mime.split(".")[-1] if "." in mime else mime
                if "document" in mime:
                    friendly_type = "Google Doc"
                elif "spreadsheet" in mime:
                    friendly_type = "Google Sheet"
                elif "presentation" in mime:
                    friendly_type = "Google Slide"
                elif "pdf" in mime:
                    friendly_type = "PDF"
                elif "folder" in mime:
                    friendly_type = "Folder"

                summaries.append(
                    f"- **Name**: [{name}]({link})\n"
                    f"  **Type**: {friendly_type}\n"
                    f"  **Created**: {created}\n"
                )

                # Fetch content for a bounded subset of files (skip folders)
                if not file_id or "folder" in mime.lower():
                    continue

                if content_fetch_count >= _DRIVE_MULTI_FILE_MAX_CONTENT_FETCH:
                    skipped_content_files.append(name)
                    continue
                if total_content_chars >= _DRIVE_MULTI_FILE_TOTAL_CHAR_LIMIT:
                    skipped_content_files.append(name)
                    continue

                remaining_budget = _DRIVE_MULTI_FILE_TOTAL_CHAR_LIMIT - total_content_chars
                per_file_limit = min(_DRIVE_FILE_CONTENT_CHAR_LIMIT, remaining_budget)
                if per_file_limit <= 0:
                    skipped_content_files.append(name)
                    continue

                logger.info(
                    "Fetching multi-match Drive content for %s (file %d/%d, budget=%d)",
                    name,
                    content_fetch_count + 1,
                    _DRIVE_MULTI_FILE_MAX_CONTENT_FETCH,
                    per_file_limit,
                )
                content = self._fetch_file_content(token, file_id, mime, timeout)
                if content and not content.startswith("["):
                    truncated = _truncate_drive_content(content, per_file_limit)
                    all_contents.append(f"\n\n--- File: {name} ---\n{truncated}")
                    total_content_chars += len(truncated)
                    content_fetch_count += 1
                    logger.info(
                        "Added Drive content for %s: raw_len=%d truncated_len=%d total_chars=%d",
                        name,
                        len(content),
                        len(truncated),
                        total_content_chars,
                    )
                else:
                    logger.warning(
                        "Could not extract content for %s: %s",
                        name,
                        content[:100] if content else "empty",
                    )

            result_text = "Matching Google Drive files:\n\n" + "\n".join(summaries)

            if all_contents:
                result_text += "\n\n--- File Contents ---\n" + "\n".join(all_contents)
                logger.info(
                    "Multi-file Drive response: %d file(s) with content, %d total chars",
                    content_fetch_count,
                    total_content_chars,
                )
            if skipped_content_files:
                result_text += (
                    f"\n\n*(Content not fetched for {len(skipped_content_files)} additional file(s): "
                    f"{', '.join(skipped_content_files[:5])}"
                    f"{'...' if len(skipped_content_files) > 5 else ''}. "
                    f"Refine your search or use @filename to read a specific file.)*"
                )
                logger.info(
                    "Skipped Drive content fetch for %d files due to limits",
                    len(skipped_content_files),
                )
            
            logger.info(f"Drive search complete: {len(summaries)} files summarized, {len(all_contents)} with content")
            logger.info(f"=== Drive Search End ===")
            return {
                "content": [{"type": "text", "text": result_text}],
                "items": summaries  # Add items for result count tracking
            }

