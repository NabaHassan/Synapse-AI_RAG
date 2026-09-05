"""
OneDrive / SharePoint file search and read helpers — Graph API parity with Google Drive.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.mcp.google_workspace_client import (
    extract_drive_search_term,
    pick_best_drive_file_match,
)

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

_FILE_CONTENT_CHAR_LIMIT = int(os.getenv("ONEDRIVE_FILE_CONTENT_CHAR_LIMIT", "4000"))
_MULTI_FILE_TOTAL_CHAR_LIMIT = int(os.getenv("ONEDRIVE_MULTI_FILE_TOTAL_CHAR_LIMIT", "12000"))
_MULTI_FILE_MAX_FETCH = int(os.getenv("ONEDRIVE_MULTI_FILE_MAX_FETCH", "3"))
_SHAREPOINT_SITE_SEARCH_MAX = int(os.getenv("ONEDRIVE_SHAREPOINT_SITE_SEARCH_MAX", "2"))
_ONEDRIVE_BROWSE_MAX = int(os.getenv("ONEDRIVE_BROWSE_MAX", "50"))
_ONEDRIVE_BROWSE_PER_FOLDER_MAX = int(os.getenv("ONEDRIVE_BROWSE_PER_FOLDER_MAX", "15"))
_ONEDRIVE_BROWSE_FOLDER_NAMES = (
    "Documents",
    "Desktop",
    "Attachments",
    "Pictures",
    "School",
    "University",
)

_ONEDRIVE_LISTING_QUERY = re.compile(
    r"\b(list|show|recent|browse|what\s+files|my\s+files|onedrive\s+files?|files\s+in\s+onedrive|all\s+files)\b",
    re.I,
)


def _truncate_content(text: str, limit: int) -> str:
    if not text or len(text) <= limit:
        return text or ""
    omitted = len(text) - limit
    logger.info("Truncating OneDrive content from %d to %d chars (%d omitted)", len(text), limit, omitted)
    return f"{text[:limit]}\n...[truncated, {omitted} more characters]"


def _graph_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


def is_broad_onedrive_listing_query(query: str) -> bool:
    """True when the user wants a file listing rather than a targeted filename search."""
    q = (query or "").strip()
    if not q:
        return True
    if _ONEDRIVE_LISTING_QUERY.search(q):
        return True
    lowered = q.lower()
    if "onedrive" in lowered and any(token in lowered for token in ("list", "show", "recent", "file")):
        return True
    return False


def resolve_onedrive_search_term(query: str) -> str:
    """Map conversational list/browse queries to an empty search (browse mode)."""
    if is_broad_onedrive_listing_query(query):
        logger.info("resolve_onedrive_search_term: broad listing query -> browse mode")
        return ""
    term = extract_drive_search_term(query)
    logger.debug("resolve_onedrive_search_term: query=%r -> term=%r", query[:80], term[:80] if term else "")
    return term


def _collect_onedrive_browse_items(
    client: httpx.Client,
    headers: Dict[str, str],
    *,
    max_items: int = _ONEDRIVE_BROWSE_MAX,
) -> List[Dict[str, Any]]:
    """
    Collect OneDrive items for autocomplete and list queries.

    Graph /me/drive/recent often returns very few items; this also scans root and
    common folders (Documents, Desktop, etc.) like Google Drive file listing.
    """
    combined: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add_items(items: List[Dict[str, Any]], *, parent_folder: str = "") -> None:
        for raw in items or []:
            item_id = raw.get("id")
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            entry = dict(raw)
            if parent_folder:
                entry["_parent_folder"] = parent_folder
            combined.append(entry)
            if len(combined) >= max_items:
                return

    recent_resp = client.get(
        f"{GRAPH_BASE}/me/drive/recent",
        headers=headers,
        params={"$top": "25"},
    )
    if recent_resp.status_code == 200:
        _add_items(recent_resp.json().get("value") or [])
    else:
        logger.warning("OneDrive recent browse failed status=%s", recent_resp.status_code)

    root_resp = client.get(
        f"{GRAPH_BASE}/me/drive/root/children",
        headers=headers,
        params={"$top": "50", "$select": "id,name,file,folder,webUrl,size"},
    )
    if root_resp.status_code != 200:
        logger.warning("OneDrive root browse failed status=%s", root_resp.status_code)
        return combined[:max_items]

    root_items = root_resp.json().get("value") or []
    folders_to_scan: List[tuple[str, str]] = []
    for item in root_items:
        if len(combined) >= max_items:
            break
        if item.get("file"):
            _add_items([item])
        elif item.get("folder"):
            folder_name = (item.get("name") or "").strip()
            _add_items([item])
            if folder_name in _ONEDRIVE_BROWSE_FOLDER_NAMES and item.get("id"):
                folders_to_scan.append((item["id"], folder_name))

    for folder_id, folder_name in folders_to_scan:
        if len(combined) >= max_items:
            break
        child_resp = client.get(
            f"{GRAPH_BASE}/me/drive/items/{folder_id}/children",
            headers=headers,
            params={
                "$top": str(_ONEDRIVE_BROWSE_PER_FOLDER_MAX),
                "$select": "id,name,file,folder,webUrl,size",
            },
        )
        if child_resp.status_code != 200:
            logger.warning(
                "OneDrive folder browse failed folder=%s status=%s",
                folder_name,
                child_resp.status_code,
            )
            continue
        _add_items(child_resp.json().get("value") or [], parent_folder=folder_name)

    logger.info("OneDrive browse collected %d unique items", len(combined))
    return combined[:max_items]


def collect_onedrive_browse_items(token: str, timeout: float = 15.0) -> List[Dict[str, Any]]:
    """Public helper for autocomplete and background cache refresh."""
    headers = _graph_headers(token)
    with httpx.Client(timeout=timeout) as client:
        return _collect_onedrive_browse_items(client, headers)


def _search_personal_drive(
    client: httpx.Client,
    headers: Dict[str, str],
    search_term: str,
) -> List[Dict[str, Any]]:
    if not search_term:
        logger.info("OneDrive search with empty term — browsing workspace files")
        return _collect_onedrive_browse_items(client, headers)

    url = f"{GRAPH_BASE}/me/drive/root/search(q='{search_term.replace(chr(39), chr(39)+chr(39))}')"
    logger.info("OneDrive personal search term=%r", search_term)
    resp = client.get(url, headers=headers)
    if resp.status_code != 200:
        logger.warning("OneDrive search failed status=%s body=%s", resp.status_code, resp.text[:300])
        return []
    items = resp.json().get("value") or []
    logger.info("OneDrive personal search returned %d items", len(items))
    return items


def _search_sharepoint_sites(
    client: httpx.Client,
    headers: Dict[str, str],
    search_term: str,
) -> List[Dict[str, Any]]:
    """Search SharePoint document libraries (best-effort, capped)."""
    site_url = f"{GRAPH_BASE}/sites"
    site_resp = client.get(
        site_url,
        headers=headers,
        params={"$search": search_term, "$top": str(_SHAREPOINT_SITE_SEARCH_MAX)},
    )
    if site_resp.status_code != 200:
        logger.warning("SharePoint site search failed status=%s", site_resp.status_code)
        return []

    sites = site_resp.json().get("value") or []
    logger.info("SharePoint site search found %d sites for term=%r", len(sites), search_term)
    combined: List[Dict[str, Any]] = []
    for site in sites:
        site_id = site.get("id")
        if not site_id:
            continue
        escaped = search_term.replace("'", "''")
        drive_search_url = f"{GRAPH_BASE}/sites/{site_id}/drive/root/search(q='{escaped}')"
        drive_resp = client.get(drive_search_url, headers=headers)
        if drive_resp.status_code != 200:
            logger.debug(
                "SharePoint drive search failed site_id=%s status=%s",
                site_id,
                drive_resp.status_code,
            )
            continue
        for item in drive_resp.json().get("value") or []:
            item["_sharepoint_site"] = site.get("displayName") or site.get("name") or site_id
            item["_sharepoint_site_id"] = site_id
            combined.append(item)
    logger.info("SharePoint drive search combined %d items", len(combined))
    return combined


def _normalize_drive_item(item: Dict[str, Any]) -> Dict[str, Any]:
    site_id = item.get("_sharepoint_site_id") or ""
    drive_path = f"sites/{site_id}" if site_id else "me"
    parent_folder = item.get("_parent_folder") or ""
    site_label = item.get("_sharepoint_site") or "OneDrive"
    if parent_folder:
        site_label = f"{site_label}/{parent_folder}"
    return {
        "id": item.get("id"),
        "name": item.get("name") or "Unnamed",
        "mimeType": (item.get("file") or {}).get("mimeType") or item.get("mimeType") or "",
        "webUrl": item.get("webUrl") or "",
        "site": site_label,
        "size": item.get("size") or 0,
        "drive_path": drive_path,
        "site_id": site_id,
        "is_folder": bool(item.get("folder")),
    }


def prepare_onedrive_call_params(
    query: str,
    *,
    item_id: Optional[str] = None,
    file_name: Optional[str] = None,
    drive_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build tool + params for Microsoft365MCPClient.call_tool (mirrors prepare_gmail_call_params)."""
    if item_id:
        params: Dict[str, Any] = {"item_id": item_id}
        if drive_path:
            params["drive_path"] = drive_path
        logger.info(
            "prepare_onedrive_call_params: direct read item_id=%s drive_path=%s",
            item_id,
            drive_path or "me",
        )
        return "read_file", params
    if file_name:
        logger.info("prepare_onedrive_call_params: read by name=%r", file_name)
        return "read_file", {"file_name": file_name, "query": file_name}
    logger.info("prepare_onedrive_call_params: search query=%r", (query or "")[:120])
    return "search_files", {"query": query or ""}


def fetch_onedrive_file_content(
    token: str,
    item_id: str,
    mime_type: str,
    timeout: float,
    *,
    drive_path: str = "me",
) -> str:
    """
    Download and extract text from a OneDrive/SharePoint drive item.
    Uses shared drive_file_extractor (same stack as Google Drive MCP reads).
    """
    from src.mcp.drive_file_extractor import extract_drive_file_text

    headers = _graph_headers(token)
    meta_url = f"{GRAPH_BASE}/{drive_path}/drive/items/{item_id}"
    logger.info("Fetching OneDrive item metadata id=%s drive_path=%s", item_id, drive_path)

    with httpx.Client(timeout=timeout) as client:
        meta_resp = client.get(meta_url, headers=headers, params={"$select": "id,name,file,size"})
        if meta_resp.status_code != 200:
            logger.warning("OneDrive metadata failed status=%s", meta_resp.status_code)
            return "[Unable to read OneDrive file metadata]"
        meta = meta_resp.json()
        name = meta.get("name") or "file"
        file_info = meta.get("file") or {}
        mime_type = mime_type or file_info.get("mimeType") or ""

        content_url = f"{GRAPH_BASE}/{drive_path}/drive/items/{item_id}/content"
        logger.info("Downloading OneDrive content name=%r mime=%r", name, mime_type)
        content_resp = client.get(content_url, headers=headers, follow_redirects=True)
        if content_resp.status_code != 200:
            logger.warning(
                "OneDrive content download failed status=%s name=%r",
                content_resp.status_code,
                name,
            )
            return f"[Unable to download OneDrive file: {name}]"

        return extract_drive_file_text(
            content_resp.content,
            mime_type=mime_type,
            filename=name,
            source_label="OneDrive",
        )


def _format_file_listing(files: List[Dict[str, Any]]) -> str:
    lines = [f"Found {len(files)} file(s) in OneDrive/SharePoint:\n"]
    for item in files:
        lines.append(
            f"- **{item.get('name')}** (site: {item.get('site')}, id: {item.get('id')}, "
            f"mime: {item.get('mimeType') or 'unknown'})"
        )
        if item.get("webUrl"):
            lines.append(f"  Link: {item['webUrl']}")
    return "\n".join(lines)


def run_onedrive_tool(
    token: str,
    tool: str,
    params: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    """Dispatch OneDrive tools: search_files, read_file."""
    query = (params.get("query") or "").strip()
    item_id = params.get("item_id") or params.get("file_id")
    drive_path = params.get("drive_path") or "me"

    logger.info(
        "run_onedrive_tool: tool=%s query=%r item_id=%s drive_path=%s",
        tool,
        query[:120],
        item_id,
        drive_path,
    )

    headers = _graph_headers(token)

    if tool == "read_file" or item_id:
        if not item_id:
            file_name = params.get("file_name") or extract_drive_search_term(query)
            if not file_name:
                return {
                    "content": [{"type": "text", "text": "No file name or item_id provided for read_file."}],
                }
            with httpx.Client(timeout=timeout) as client:
                items = [_normalize_drive_item(i) for i in _search_personal_drive(client, headers, file_name)]
                items.extend(
                    _normalize_drive_item(i) for i in _search_sharepoint_sites(client, headers, file_name)
                )
            best = pick_best_drive_file_match(items, file_name) if items else None
            if not best:
                return {
                    "content": [{"type": "text", "text": f"No OneDrive/SharePoint file matched '{file_name}'."}],
                }
            item_id = best["id"]
            drive_path = best.get("drive_path") or "me"

        content = fetch_onedrive_file_content(
            token,
            item_id,
            params.get("mime_type") or "",
            timeout,
            drive_path=drive_path,
        )
        truncated = _truncate_content(content, _FILE_CONTENT_CHAR_LIMIT)
        result_text = (
            f"Found file context from OneDrive/SharePoint:\n"
            f"Item ID: {item_id}\n"
            f"--- Content Begin ---\n{truncated}\n--- Content End ---"
        )
        return {
            "content": [{"type": "text", "text": result_text}],
            "items": [{"id": item_id}],
        }

    # search_files (default)
    search_term = resolve_onedrive_search_term(query) if query else ""
    with httpx.Client(timeout=timeout) as client:
        personal = [_normalize_drive_item(i) for i in _search_personal_drive(client, headers, search_term)]
        sharepoint = (
            [_normalize_drive_item(i) for i in _search_sharepoint_sites(client, headers, search_term)]
            if search_term
            else []
        )
    files = [f for f in personal + sharepoint if not f.get("is_folder")]

    if not files and is_broad_onedrive_listing_query(query):
        logger.info("OneDrive search empty for listing query — falling back to browse")
        with httpx.Client(timeout=timeout) as client:
            files = [
                _normalize_drive_item(i)
                for i in _collect_onedrive_browse_items(client, headers)
                if not i.get("folder")
            ]

    if not files:
        return {
            "content": [{"type": "text", "text": "No matching files found in OneDrive or SharePoint."}],
        }

    if len(files) == 1 or search_term:
        best = pick_best_drive_file_match(files, search_term) if search_term else files[0]
        target = best or files[0]
        content = fetch_onedrive_file_content(
            token,
            target["id"],
            target.get("mimeType") or "",
            timeout,
            drive_path=target.get("drive_path") or "me",
        )
        truncated = _truncate_content(content, _FILE_CONTENT_CHAR_LIMIT)
        result_text = (
            f"Found file context from OneDrive/SharePoint:\n"
            f"Filename: {target['name']}\n"
            f"Site: {target.get('site')}\n"
            f"--- Content Begin ---\n{truncated}\n--- Content End ---"
        )
        return {
            "content": [{"type": "text", "text": result_text}],
            "items": [target],
        }

    listing = _format_file_listing(files[:10])
    total_chars = 0
    content_sections: List[str] = [listing, "\n--- File previews ---\n"]
    fetched = 0
    for item in files:
        if fetched >= _MULTI_FILE_MAX_FETCH:
            break
        if total_chars >= _MULTI_FILE_TOTAL_CHAR_LIMIT:
            break
        body = fetch_onedrive_file_content(
            token,
            item["id"],
            item.get("mimeType") or "",
            timeout,
            drive_path=item.get("drive_path") or "me",
        )
        snippet = _truncate_content(body, _FILE_CONTENT_CHAR_LIMIT)
        total_chars += len(snippet)
        fetched += 1
        content_sections.append(
            f"\n**{item['name']}** (site: {item.get('site')}):\n{snippet}\n"
        )

    return {
        "content": [{"type": "text", "text": "\n".join(content_sections)}],
        "items": files[:10],
    }
