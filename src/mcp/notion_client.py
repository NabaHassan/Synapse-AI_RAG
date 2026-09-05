"""
Notion MCP Client — OAuth 2.0 (public integrations).

Handles:
  - OAuth 2.0 flow (per-user token in Redis) for public/multi-workspace integrations
  - Token storage/retrieval in Redis (per-user only; no shared fallback tokens)

Services supported (read-only):
  - search_pages     → POST /v1/search  (filter: page, full-text)
  - query_database   → POST /v1/databases/{id}/query (multi-filter)
  - search_databases → POST /v1/search  (filter: database)
  - get_page_details → GET  /v1/pages/{id} + block children (properties + body)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import httpx

from src.mcp.notion_property_extractor import (
    extract_all_properties,
    extract_page_title,
    format_page_summary,
    format_properties_block,
)
from src.mcp.notion_query_parser import (
    build_compound_filter,
    is_template_notion_title,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOTION_AUTHORIZE_URL = "https://api.notion.com/v1/oauth/authorize"
NOTION_TOKEN_URL     = "https://api.notion.com/v1/oauth/token"
NOTION_API_BASE      = "https://api.notion.com/v1"
NOTION_API_VERSION   = "2022-06-28"

# Token TTL in Redis (30 days)
_TOKEN_TTL_SECONDS = int(os.getenv("NOTION_MCP_TOKEN_TTL", str(30 * 24 * 3600)))

# Page content extraction limits (paginated block fetch)
_NOTION_BLOCK_PAGE_SIZE = 100
_NOTION_PAGE_MAX_BLOCKS = int(os.getenv("NOTION_PAGE_MAX_BLOCKS", "500"))
_NOTION_PAGE_MAX_DEPTH = int(os.getenv("NOTION_PAGE_MAX_DEPTH", "4"))
_NOTION_PAGE_MAX_CHARS = int(os.getenv("NOTION_PAGE_MAX_CHARS", "16000"))


# ---------------------------------------------------------------------------
# NotionMCPClient
# ---------------------------------------------------------------------------


class NotionMCPClient:
    """
    Lightweight Notion MCP client.

    Stores one OAuth access token per user_id in Redis.
    Users must complete OAuth before any Notion data can be read.
    All API calls are async (httpx.AsyncClient).
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        redis_client,               # redis.Redis instance
        redis_key_prefix: str = "synapse",
        token_ttl_seconds: int = _TOKEN_TTL_SECONDS,
    ) -> None:
        self._client_id     = client_id
        self._client_secret = client_secret
        self._redirect_uri  = redirect_uri
        self._redis         = redis_client
        self._key_prefix    = redis_key_prefix
        self._token_ttl     = token_ttl_seconds

        logger.info("NotionMCPClient initialized (redirect_uri=%s)", self._redirect_uri)

    # -----------------------------------------------------------------------
    # OAuth helpers
    # -----------------------------------------------------------------------

    def get_auth_url(self, user_id: str) -> str:
        """
        Generate a Notion OAuth 2.0 authorization URL.

        The `state` encodes user_id + nonce for CSRF protection and so the
        callback can store the token under the correct Redis key.
        """
        nonce = secrets.token_urlsafe(16)
        state_payload = json.dumps({"user_id": user_id, "nonce": nonce})
        state = urllib.parse.quote(state_payload)

        params = {
            "client_id":     self._client_id,
            "redirect_uri":  self._redirect_uri,
            "response_type": "code",
            "owner":         "user",
            "state":         state,
        }
        auth_url = f"{NOTION_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
        logger.info("Generated Notion auth URL for user_id=%s", user_id)
        return auth_url

    def exchange_code(self, code: str, state: str) -> Dict[str, Any]:
        """
        Exchange an authorization code for an access token and persist to Redis.

        Notion requires HTTP Basic Auth (base64 client_id:client_secret) for the
        token exchange endpoint — different from Slack's form-based approach.
        """
        try:
            state_payload = json.loads(urllib.parse.unquote(state))
            user_id: str  = state_payload["user_id"]
        except Exception as exc:
            raise ValueError(f"Invalid Notion OAuth state parameter: {exc}") from exc

        credentials = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode()
        ).decode()

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                NOTION_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type":  "application/json",
                },
                json={
                    "grant_type":   "authorization_code",
                    "code":         code,
                    "redirect_uri": self._redirect_uri,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise ValueError(f"Notion OAuth token exchange failed: {data['error']}")

        access_token = data.get("access_token")
        if not access_token:
            raise ValueError("Notion OAuth response missing access_token.")

        token_data = {
            "access_token":   access_token,
            "token_type":     data.get("token_type", "bearer"),
            "bot_id":         data.get("bot_id", ""),
            "workspace_id":   data.get("workspace_id", ""),
            "workspace_name": data.get("workspace_name", ""),
            "workspace_icon": data.get("workspace_icon", ""),
            "owner":          data.get("owner", {}),
            "web_user_id":    user_id,
            "stored_at":      time.time(),
        }

        redis_key = self._token_key(user_id)
        self._redis.setex(redis_key, self._token_ttl, json.dumps(token_data))
        logger.info(
            "Stored Notion token for user_id=%s workspace=%s",
            user_id,
            token_data["workspace_name"],
        )

        return {
            "user_id":        user_id,
            "workspace_name": token_data["workspace_name"],
            "workspace_id":   token_data["workspace_id"],
            "access_token":   access_token,
        }

    # -----------------------------------------------------------------------
    # Token management
    # -----------------------------------------------------------------------

    def _token_key(self, user_id: str) -> str:
        return f"{self._key_prefix}:mcp:notion:token:{user_id}"

    def _load_token(self, user_id: str) -> Optional[str]:
        """Load per-user OAuth access token from Redis. No shared fallback tokens."""
        if not user_id:
            logger.debug("Notion _load_token called without user_id")
            return None
        raw = self._redis.get(self._token_key(user_id))
        if not raw:
            logger.debug("No Notion token in Redis for user_id=%s", user_id)
            return None
        try:
            token = json.loads(raw).get("access_token")
            if not token:
                logger.warning("Notion token record missing access_token for user_id=%s", user_id)
            return token
        except Exception as exc:
            logger.warning("Failed to parse Notion token for user_id=%s: %s", user_id, exc)
            return None

    def _load_token_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load full token record from Redis."""
        raw = self._redis.get(self._token_key(user_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def is_authenticated(self, user_id: str) -> bool:
        """True only when this user has completed Notion OAuth (token in Redis)."""
        if not user_id:
            return False
        return self._load_token(user_id) is not None

    def revoke_tokens(self, user_id: str) -> None:
        """Delete Notion token for user from Redis."""
        self._redis.delete(self._token_key(user_id))
        logger.info("Revoked Notion token for user_id=%s", user_id)

    def auth_status(self, user_id: str) -> Dict[str, Any]:
        """Return authentication status and workspace metadata for the user."""
        if not user_id:
            return {"connected": False}
        data = self._load_token_data(user_id)
        if not data:
            return {"connected": False}
        return {
            "connected":      True,
            "workspace_name": data.get("workspace_name", ""),
            "workspace_id":   data.get("workspace_id", ""),
            "stored_at":      data.get("stored_at"),
        }

    # -----------------------------------------------------------------------
    # Notion API helpers
    # -----------------------------------------------------------------------

    def _headers(self, token: str) -> Dict[str, str]:
        return {
            "Authorization":  f"Bearer {token}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type":   "application/json",
        }

    async def _notion_post(
        self,
        token: str,
        endpoint: str,
        body: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Authenticated POST to the Notion API."""
        url = f"{NOTION_API_BASE}/{endpoint}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=self._headers(token), json=body)
            resp.raise_for_status()
        data = resp.json()
        if data.get("object") == "error":
            raise RuntimeError(f"Notion API error ({endpoint}): {data.get('message', 'unknown')}")
        return data

    async def _notion_get(
        self,
        token: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Authenticated GET to the Notion API."""
        url = f"{NOTION_API_BASE}/{endpoint}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=self._headers(token), params=params or {})
            resp.raise_for_status()
        data = resp.json()
        if data.get("object") == "error":
            raise RuntimeError(f"Notion API error ({endpoint}): {data.get('message', 'unknown')}")
        return data

    async def _paginate_block_children(
        self,
        token: str,
        block_id: str,
        timeout: float = 30.0,
        max_blocks: int = _NOTION_PAGE_MAX_BLOCKS,
    ) -> List[Dict[str, Any]]:
        """Follow Notion next_cursor until max_blocks or no more children."""
        collected: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        while len(collected) < max_blocks:
            params: Dict[str, Any] = {"page_size": min(_NOTION_BLOCK_PAGE_SIZE, max_blocks - len(collected))}
            if cursor:
                params["start_cursor"] = cursor
            data = await self._notion_get(token, f"blocks/{block_id}/children", params, timeout=timeout)
            batch = data.get("results") or []
            collected.extend(batch)
            if not data.get("has_more") or not data.get("next_cursor"):
                break
            cursor = data.get("next_cursor")
            logger.debug(
                "Notion paginate blocks/%s/children: collected=%d has_more=%s",
                block_id,
                len(collected),
                data.get("has_more"),
            )

        if len(collected) >= max_blocks:
            logger.info(
                "Notion block children capped at max_blocks=%d for block_id=%s",
                max_blocks,
                block_id,
            )
        return collected[:max_blocks]

    async def _collect_page_text_lines(
        self,
        token: str,
        block_id: str,
        timeout: float,
        *,
        depth: int = 0,
        lines: Optional[List[str]] = None,
        block_count: Optional[List[int]] = None,
        indent: int = 0,
    ) -> List[str]:
        """
        Recursively collect readable text from a page or nested block children.
        Uses block_count as a mutable single-element list to track total blocks visited.
        """
        if lines is None:
            lines = []
        if block_count is None:
            block_count = [0]

        if depth > _NOTION_PAGE_MAX_DEPTH:
            logger.debug("Notion block recursion depth limit reached at block_id=%s", block_id)
            return lines

        remaining = _NOTION_PAGE_MAX_BLOCKS - block_count[0]
        if remaining <= 0:
            return lines

        blocks = await self._paginate_block_children(
            token,
            block_id,
            timeout=timeout,
            max_blocks=remaining,
        )

        for block in blocks:
            if block_count[0] >= _NOTION_PAGE_MAX_BLOCKS:
                lines.append("...[page content truncated: block limit reached]")
                logger.info("Notion page block limit %d reached", _NOTION_PAGE_MAX_BLOCKS)
                return lines

            block_count[0] += 1
            line = _block_to_text_line(block, indent=indent)
            if line:
                lines.append(line)

            if block.get("has_children") and block.get("id"):
                child_type = block.get("type", "")
                child_indent = indent + (2 if child_type in ("bulleted_list_item", "numbered_list_item", "to_do", "toggle") else 0)
                await self._collect_page_text_lines(
                    token,
                    block["id"],
                    timeout,
                    depth=depth + 1,
                    lines=lines,
                    block_count=block_count,
                    indent=child_indent,
                )

        return lines

    # -----------------------------------------------------------------------
    # Public read operations
    # -----------------------------------------------------------------------

    async def _paginate_post(
        self,
        token: str,
        endpoint: str,
        body: Dict[str, Any],
        max_results: int = 100,
        timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """Follow Notion next_cursor until max_results or no more pages."""
        collected: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        page_size = min(100, max_results)

        while len(collected) < max_results:
            req = {**body, "page_size": min(page_size, max_results - len(collected))}
            if cursor:
                req["start_cursor"] = cursor
            data = await self._notion_post(token, endpoint, req, timeout=timeout)
            batch = data.get("results") or []
            collected.extend(batch)
            if not data.get("has_more") or not data.get("next_cursor"):
                break
            cursor = data.get("next_cursor")
            logger.debug(
                "Notion paginate %s: collected=%d has_more=%s",
                endpoint,
                len(collected),
                data.get("has_more"),
            )

        return collected[:max_results]

    def _format_page_summaries(self, pages: List[Dict[str, Any]]) -> List[str]:
        summaries: List[str] = []
        for page in pages:
            title = extract_page_title(page)
            if is_template_notion_title(title):
                logger.debug("Skipping template page in summaries: %r", title)
                continue
            summaries.append(format_page_summary(page))
        return summaries

    async def _resolve_task_database(
        self,
        token: str,
        database_hint: Optional[str],
        timeout: float,
    ) -> Optional[Dict[str, Any]]:
        """Pick the best task database (by hint or first accessible match)."""
        body: Dict[str, Any] = {
            "filter": {"property": "object", "value": "database"},
            "page_size": 50,
        }
        if database_hint:
            body["query"] = database_hint.strip()

        databases = await self._paginate_post(token, "search", body, max_results=50, timeout=timeout)
        if not databases:
            logger.warning("Notion: no databases found for hint=%r", database_hint)
            return None

        if database_hint:
            hint_lower = database_hint.lower()
            for db in databases:
                title = extract_page_title(db).lower()
                if hint_lower in title or title in hint_lower:
                    logger.info("Notion: matched database %r", extract_page_title(db))
                    return db

        for db in databases:
            title_lower = extract_page_title(db).lower()
            if "task" in title_lower or "synapse" in title_lower:
                logger.info("Notion: using task-like database %r", extract_page_title(db))
                return db

        logger.info("Notion: defaulting to database %r", extract_page_title(databases[0]))
        return databases[0]

    async def retrieve_page(
        self,
        user_id: str,
        page_id: str,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Fetch a Notion page object including all database properties."""
        import re

        token = self._load_token(user_id)
        if not token:
            raise PermissionError(f"No Notion token for user_id={user_id}. Please sign in first.")

        resolved_id = page_id
        if page_id and not re.match(
            r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$",
            page_id.lower().replace("-", ""),
        ):
            body = {"query": page_id, "page_size": 5, "filter": {"property": "object", "value": "page"}}
            data = await self._notion_post(token, "search", body, timeout=timeout)
            results = data.get("results") or []
            for row in results:
                title = extract_page_title(row)
                if title.lower() == page_id.lower() or page_id.lower() in title.lower():
                    resolved_id = row.get("id", "")
                    break
            if not resolved_id or resolved_id == page_id:
                if results:
                    resolved_id = results[0].get("id", page_id)
                else:
                    raise ValueError(f"No Notion page found matching {page_id!r}")

        logger.info("Notion retrieve_page user_id=%s page_id=%s", user_id, resolved_id)
        page = await self._notion_get(token, f"pages/{resolved_id}", timeout=timeout)
        return page

    async def get_page_details(
        self,
        user_id: str,
        page_id: str,
        focus_property: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Return full task/page details: all properties plus block body content.
        """
        token = self._load_token(user_id)
        if not token:
            raise PermissionError(f"No Notion token for user_id={user_id}. Please sign in first.")

        page = await self.retrieve_page(user_id=user_id, page_id=page_id, timeout=timeout)
        resolved_id = page.get("id", page_id)
        title = extract_page_title(page)
        properties = extract_all_properties(page)
        url = page.get("url", "")

        lines = await self._collect_page_text_lines(token, resolved_id, timeout)
        body_text = "\n".join(lines).strip()

        header = (
            f"Task: {title}\n"
            f"ID: {resolved_id}\n"
            f"URL: {url}\n"
        )
        prop_block = format_properties_block(properties, indent="")
        sections = [header]
        if prop_block:
            sections.append(prop_block)

        if focus_property:
            prop_key = None
            for key in properties:
                if key.lower() == focus_property.lower():
                    prop_key = key
                    break
            if not prop_key:
                semantic_keys = {
                    "priority": "priority",
                    "assigned": "assignee",
                    "status": "status",
                    "category": "category",
                    "due": "due",
                    "description": "description",
                }
                sem = semantic_keys.get(focus_property, focus_property)
                for key in properties:
                    if sem in key.lower():
                        prop_key = key
                        break
            if prop_key and prop_key in properties:
                sections.append(f"Focused property — {prop_key}: {properties[prop_key]}")

        if body_text:
            sections.append(f"Page body:\n{body_text}")
        elif not properties:
            sections.append("No additional page body content found.")

        if len("\n".join(sections)) > _NOTION_PAGE_MAX_CHARS:
            combined = "\n\n".join(sections)
            combined = combined[:_NOTION_PAGE_MAX_CHARS] + "\n...[truncated]"
            text = combined
        else:
            text = "\n\n".join(sections)

        logger.info(
            "Notion get_page_details complete page_id=%s title=%r props=%d body_chars=%d",
            resolved_id,
            title,
            len(properties),
            len(body_text),
        )
        return {
            "content": [{"type": "text", "text": text}],
            "page_id": resolved_id,
            "title": title,
            "properties": properties,
            "url": url,
        }

    async def query_database(
        self,
        user_id: str,
        filter_spec: Optional[Dict[str, Any]] = None,
        database_hint: Optional[str] = None,
        database_id: Optional[str] = None,
        limit: int = 100,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Query a Notion database with structured filters (status, priority, category, dates).
        """
        token = self._load_token(user_id)
        if not token:
            raise PermissionError(f"No Notion token for user_id={user_id}. Please sign in first.")

        filter_spec = filter_spec or {}
        logger.info(
            "Notion query_database user_id=%s hint=%r spec=%s",
            user_id,
            database_hint or database_id,
            filter_spec,
        )

        database: Optional[Dict[str, Any]] = None
        if database_id:
            database = await self._notion_get(token, f"databases/{database_id}", timeout=timeout)
        else:
            database = await self._resolve_task_database(
                token,
                database_hint or filter_spec.get("database_hint"),
                timeout,
            )

        if not database:
            text = f"No Notion database found matching {database_hint!r}."
            return {"content": [{"type": "text", "text": text}]}

        db_id = database.get("id", "")
        db_title = extract_page_title(database)
        notion_filter = build_compound_filter(database, filter_spec)

        body: Dict[str, Any] = {}
        if notion_filter:
            body["filter"] = notion_filter

        pages = await self._paginate_post(
            token,
            f"databases/{db_id}/query",
            body,
            max_results=limit,
            timeout=timeout,
        )

        assignee_name = (filter_spec.get("assignee") or "").strip().lower()
        if assignee_name:
            filtered_pages = []
            for page in pages:
                props = extract_all_properties(page)
                assignee_val = ""
                for key, val in props.items():
                    if "assign" in key.lower() or key.lower() in ("owner", "person"):
                        assignee_val = val.lower()
                        break
                if assignee_name in assignee_val:
                    filtered_pages.append(page)
            pages = filtered_pages
            logger.info(
                "Notion query_database assignee filter %r -> %d rows",
                filter_spec.get("assignee"),
                len(pages),
            )

        if not pages:
            filt_desc = ", ".join(
                f"{k}={v}" for k, v in filter_spec.items() if v and k != "clean_query"
            ) or "no filters"
            text = f"No tasks found in {db_title!r} matching ({filt_desc})."
            return {
                "content": [{"type": "text", "text": text}],
                "database_id": db_id,
                "database_title": db_title,
            }

        summaries = self._format_page_summaries(pages)
        filt_desc = ", ".join(
            f"{k}={v}" for k, v in filter_spec.items() if v and k not in ("clean_query", "database_hint")
        ) or "all"
        header = f"Notion tasks in {db_title!r} matching ({filt_desc}) — {len(summaries)} item(s):\n\n"
        logger.info(
            "Notion query_database returned %d rows db=%r",
            len(summaries),
            db_title,
        )
        return {
            "content": [{"type": "text", "text": header + "\n\n".join(summaries)}],
            "database_id": db_id,
            "database_title": db_title,
            "tasks": summaries,
        }

    async def search_pages(
        self,
        user_id: str,
        query: str = "",
        limit: int = 100,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Search Notion pages accessible to the integration (full-text, not Status filter).
        Returns a formatted text block ready for MCP context injection.
        """
        token = self._load_token(user_id)
        if not token:
            raise PermissionError(f"No Notion token for user_id={user_id}. Please sign in first.")

        logger.info("Notion search_pages user_id=%s query=%r limit=%d", user_id, query, limit)

        body: Dict[str, Any] = {
            "filter": {"property": "object", "value": "page"},
        }
        if query.strip():
            body["query"] = query.strip()

        results = await self._paginate_post(token, "search", body, max_results=limit, timeout=timeout)

        if not results:
            text = (
                f"No Notion pages found matching '{query}'."
                if query else "No Notion pages found."
            )
            return {"content": [{"type": "text", "text": text}]}

        summaries = self._format_page_summaries(results)
        header = f"Notion pages{' matching ' + repr(query) if query else ''}:\n\n"
        return {"content": [{"type": "text", "text": header + "\n\n".join(summaries)}]}

    async def search_databases(
        self,
        user_id: str,
        query: str = "",
        limit: int = 100,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Search Notion databases accessible to the integration.
        """
        token = self._load_token(user_id)
        if not token:
            raise PermissionError(f"No Notion token for user_id={user_id}. Please sign in first.")

        logger.info("Notion search_databases user_id=%s query=%r", user_id, query)

        body: Dict[str, Any] = {
            "filter":    {"property": "object", "value": "database"},
            "page_size": limit,
        }
        if query.strip():
            body["query"] = query.strip()

        data    = await self._notion_post(token, "search", body, timeout=timeout)
        results = data.get("results", [])

        if not results:
            text = (
                f"No Notion databases found matching '{query}'."
                if query else "No Notion databases found."
            )
            return {"content": [{"type": "text", "text": text}]}

        summaries: List[str] = []
        for db in results:
            title  = extract_page_title(db)
            db_id  = db.get("id", "")
            url    = db.get("url", "")
            props  = list(db.get("properties", {}).keys())[:5]
            summaries.append(
                f"- **{title}** (Database)\n"
                f"  URL: {url}\n"
                f"  ID: {db_id}\n"
                f"  Columns: {', '.join(props) if props else 'N/A'}"
            )

        header = f"Notion databases{' matching ' + repr(query) if query else ''}:\n\n"
        return {"content": [{"type": "text", "text": header + "\n\n".join(summaries)}]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rich_text_to_plain(rich_text: List[Dict[str, Any]]) -> str:
    return "".join(rt.get("plain_text", "") for rt in (rich_text or [])).strip()


def _block_to_text_line(block: Dict[str, Any], indent: int = 0) -> str:
    """Convert a Notion block object into a single plain-text line (or empty string)."""
    block_type = block.get("type", "")
    block_data = block.get(block_type, {}) or {}
    prefix_pad = " " * indent

    if block_type == "divider":
        return f"{prefix_pad}---"

    rich_text = block_data.get("rich_text")
    if rich_text is not None:
        text_str = _rich_text_to_plain(rich_text)
        if text_str:
            prefix = {
                "heading_1": "# ",
                "heading_2": "## ",
                "heading_3": "### ",
                "bulleted_list_item": "• ",
                "numbered_list_item": "- ",
                "to_do": ("☑ " if block_data.get("checked") else "☐ "),
                "quote": "> ",
                "code": "`",
                "callout": "💡 ",
            }.get(block_type, "")
            suffix = "`" if block_type == "code" else ""
            return f"{prefix_pad}{prefix}{text_str}{suffix}"
        return ""

    if block_type == "equation":
        expr = block_data.get("expression", "")
        return f"{prefix_pad}$ {expr}" if expr else ""

    if block_type == "bookmark":
        url = block_data.get("url", "")
        caption = _rich_text_to_plain(block_data.get("caption", []))
        if url:
            return f"{prefix_pad}🔗 {caption or url} ({url})"
        return ""

    if block_type in ("embed", "link_preview", "video", "file", "pdf"):
        url = block_data.get("url") or block_data.get("external", {}).get("url", "")
        caption = _rich_text_to_plain(block_data.get("caption", []))
        if url or caption:
            return f"{prefix_pad}🔗 {caption or url}"
        return ""

    if block_type == "table_row":
        cells = block_data.get("cells", [])
        cell_texts = [_rich_text_to_plain(cell) for cell in cells]
        joined = " | ".join(t for t in cell_texts if t)
        return f"{prefix_pad}| {joined} |" if joined else ""

    if block_type in ("child_page", "child_database"):
        title = block_data.get("title", "Untitled")
        kind = "Page" if block_type == "child_page" else "Database"
        return f"{prefix_pad}📎 {kind}: {title}"

    if block_type == "table":
        return f"{prefix_pad}[Table]"

    if block_type == "column_list":
        return ""

    logger.debug("Notion block type not mapped to text: %s", block_type)
    return ""
