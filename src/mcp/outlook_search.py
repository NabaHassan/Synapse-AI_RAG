"""
Outlook mail search helpers — Graph API parity with gmail_search.py (read-only).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

_INBOX_TOP = int(os.getenv("OUTLOOK_INBOX_TOP", "10"))
_SEARCH_TOP = int(os.getenv("OUTLOOK_SEARCH_TOP", "10"))
_BODY_MAX_CHARS = int(os.getenv("OUTLOOK_BODY_MAX_CHARS", "3000"))
_THREAD_MAX_MESSAGES = int(os.getenv("OUTLOOK_THREAD_MAX_MESSAGES", "10"))

_MESSAGE_ID_PATTERN = re.compile(
    r"\b(?:message[_\s-]?id|id)[=:\s]+([A-Za-z0-9_-]{20,})\b",
    re.I,
)

_CONVERSATIONAL_PREFIXES = [
    r"^(?:please\s+)?(?:can you\s+)?(?:could you\s+)?(?:would you\s+)?",
    r"^(?:tell me about|show me|give me|list|check|find|search for|look up|look at)\s+(?:my\s+)?",
    r"^(?:what are|what is|what were|what's|whats)\s+(?:my\s+)?",
    r"^(?:emails?|mail|messages?|inbox|outlook)\s+(?:about|from|regarding|on|for|with)\s+",
    r"^(?:emails?|mail|messages?|inbox|outlook)\s+",
    r"^(?:my\s+)?(?:recent\s+)?(?:emails?|mail|messages?|inbox)\s*",
]

_FROM_PATTERNS = [
    re.compile(r"\b(?:from|by|sent by)\s+([A-Za-z0-9._%+\-@ ]{2,80})", re.I),
    re.compile(r"\b(?:email(?:s)?|mail|message(?:s)?)\s+from\s+([A-Za-z0-9._%+\-@ ]{2,80})", re.I),
]

_SUBJECT_PATTERNS = [
    re.compile(r'\bsubject\s+(?:is\s+)?["\']?([^"\']+)["\']?', re.I),
    re.compile(
        r"\b(?:about|regarding|re:|on the topic of)\s+(.+?)(?:\s+(?:today|yesterday|this week|unread)|$)",
        re.I,
    ),
]

_INBOX_HINTS = re.compile(
    r"\b(recent emails?|latest emails?|my emails today|emails today|check my inbox|"
    r"what emails|inbox today|mail today|outlook inbox)\b",
    re.I,
)

_OUTLOOK_HASH_FOLDERS: Dict[str, str] = {
    "inbox": "inbox",
    "sent": "sentitems",
    "drafts": "drafts",
    "draft": "drafts",
    "trash": "deleteditems",
    "bin": "deleteditems",
    "archive": "archive",
    "junk": "junkemail",
    "spam": "junkemail",
}

_OUTLOOK_AT_DATE_ALIASES: Dict[str, str] = {
    "today": "today",
    "yesterday": "yesterday",
    "lastweek": "last week",
    "last-week": "last week",
    "thisweek": "this week",
    "this-week": "this week",
}

_OUTLOOK_FOLLOWUP_PATTERNS = [
    re.compile(r"\bwhat did (?:that|it) say\b", re.I),
    re.compile(r"\b(?:that|this|the) (?:email|thread|message|mail)\b", re.I),
    re.compile(r"\bmore (?:about|on) (?:that|it|this)(?:\s+email|\s+thread|\s+message)?\b", re.I),
    re.compile(r"\b(?:summarize|summary of|read) (?:that|it|this)(?:\s+email|\s+thread|\s+message)?\b", re.I),
    re.compile(r"^(?:that one|this one|the one)\b", re.I),
]


@dataclass
class OutlookQueryParams:
    """Structured Outlook search parameters (mirrors GmailQueryParams)."""

    original_query: str = ""
    sender: Optional[str] = None
    subject_keywords: List[str] = field(default_factory=list)
    keywords: Optional[str] = None
    date_after: Optional[str] = None
    date_before: Optional[str] = None
    is_unread: bool = False
    folder: str = "inbox"
    clean_query: str = ""
    message_id: Optional[str] = None
    conversation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutlookMessageRecord:
    message_id: str
    conversation_id: str
    subject: str
    sender: str
    received: str
    preview: str
    body: str = ""
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _graph_headers(token: str, *, search: bool = False) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if search:
        headers["ConsistencyLevel"] = "eventual"
    return headers


def _strip_conversational_noise(query: str) -> str:
    text = (query or "").strip()
    for pattern in _CONVERSATIONAL_PREFIXES:
        text = re.sub(pattern, "", text, flags=re.I).strip()
    return text


def _extract_message_id(query: str) -> Optional[str]:
    match = _MESSAGE_ID_PATTERN.search(query or "")
    if match:
        return match.group(1)
    return None


def _extract_sender(query: str) -> Optional[str]:
    for pattern in _FROM_PATTERNS:
        match = pattern.search(query or "")
        if match:
            sender = match.group(1).strip()
            if sender:
                logger.debug("Outlook sender extracted: %r", sender)
                return sender
    return None


def _extract_subject_keywords(query: str) -> List[str]:
    for pattern in _SUBJECT_PATTERNS:
        match = pattern.search(query or "")
        if match:
            subject = match.group(1).strip()
            if subject:
                return [subject]
    cleaned = _strip_conversational_noise(query)
    if cleaned and len(cleaned.split()) <= 6:
        return [cleaned]
    return []


def parse_outlook_mentions(query: str) -> Dict[str, Any]:
    """Parse @sender, @date, and #folder tokens from the composer."""
    result: Dict[str, Any] = {
        "sender": None,
        "folder": None,
        "date_after": None,
        "date_before": None,
        "is_unread": False,
        "clean_query": query,
    }

    q_lower = (query or "").lower()
    if re.search(r"\bunread\b", q_lower):
        result["is_unread"] = True

    try:
        from src.mcp.gmail_search import parse_date_range

        tz = os.getenv("OUTLOOK_USER_TIMEZONE", os.getenv("TZ", "UTC")) or "UTC"
        for at_date in re.finditer(r"@([\w-]+)", query or "", re.I):
            token = at_date.group(1).lower()
            phrase = _OUTLOOK_AT_DATE_ALIASES.get(token)
            if phrase:
                after, before = parse_date_range(phrase, tz)
                if after:
                    result["date_after"] = after.replace("/", "-")
                if before:
                    result["date_before"] = before.replace("/", "-")
                logger.debug("Outlook @date: @%s -> after=%r before=%r", token, after, before)
    except Exception as exc:
        logger.warning("Outlook date mention parse failed: %s", exc)

    hash_match = re.search(r"#([\w-]+)", query or "")
    if hash_match:
        token = hash_match.group(1).lower()
        if token in _OUTLOOK_HASH_FOLDERS:
            result["folder"] = _OUTLOOK_HASH_FOLDERS[token]
            logger.debug("Outlook # folder: #%s -> %s", token, result["folder"])
        elif token == "unread":
            result["is_unread"] = True

    at_match = re.search(
        r"@([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
        query or "",
    )
    if at_match:
        result["sender"] = at_match.group(1).strip()
    else:
        at_name = re.search(r"@([A-Za-z][A-Za-z0-9.'\-]+)", query or "")
        if at_name:
            token = at_name.group(1).strip().lower()
            if token not in _OUTLOOK_AT_DATE_ALIASES:
                result["sender"] = at_name.group(1).strip()

    clean = query or ""
    clean = re.sub(r"#[\w-]+", " ", clean)
    clean = re.sub(r"@[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", " ", clean)
    clean = re.sub(r"@[A-Za-z][A-Za-z0-9.'\-]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    result["clean_query"] = clean or query
    logger.debug(
        "parse_outlook_mentions: sender=%r folder=%r unread=%s clean=%r",
        result["sender"],
        result["folder"],
        result["is_unread"],
        result["clean_query"],
    )
    return result


def apply_outlook_connector_hints(
    params: OutlookQueryParams,
    *,
    outlook_folder: Optional[str] = None,
    outlook_location: Optional[str] = None,
) -> OutlookQueryParams:
    """Apply #folder chips from the UI (mirrors apply_gmail_connector_hints)."""
    folder_hint = outlook_folder or outlook_location
    if folder_hint:
        normalized = folder_hint.lower().strip().lstrip("#")
        if normalized == "unread":
            params.is_unread = True
            logger.info("Outlook connector hint: unread flag from chip")
        elif normalized in _OUTLOOK_HASH_FOLDERS:
            params.folder = _OUTLOOK_HASH_FOLDERS[normalized]
        elif normalized in _OUTLOOK_HASH_FOLDERS.values():
            params.folder = normalized
        else:
            params.folder = normalized
        logger.info("Outlook connector hint: folder=%s unread=%s", params.folder, params.is_unread)
    return params


def parse_outlook_query_struct(
    query: str,
    *,
    outlook_folder: Optional[str] = None,
    outlook_location: Optional[str] = None,
) -> OutlookQueryParams:
    """Rule-based structured extraction for Outlook queries."""
    mentions = parse_outlook_mentions(query)
    working_query = mentions.get("clean_query") or query
    params = OutlookQueryParams(original_query=query, clean_query=working_query)

    params.is_unread = bool(mentions.get("is_unread"))
    params.folder = mentions.get("folder") or "inbox"
    params.date_after = mentions.get("date_after")
    params.date_before = mentions.get("date_before")
    params.sender = mentions.get("sender") or _extract_sender(working_query)
    subjects = _extract_subject_keywords(working_query)
    if subjects:
        params.subject_keywords = subjects
        params.keywords = subjects[0]

    mid = _extract_message_id(query)
    if mid:
        params.message_id = mid

    params = apply_outlook_connector_hints(
        params,
        outlook_folder=outlook_folder,
        outlook_location=outlook_location,
    )
    logger.info(
        "parse_outlook_query_struct: folder=%s unread=%s sender=%r",
        params.folder,
        params.is_unread,
        params.sender,
    )
    return params


def is_outlook_follow_up_query(query: str) -> bool:
    return any(pat.search(query or "") for pat in _OUTLOOK_FOLLOWUP_PATTERNS)


def match_outlook_message_from_memory(query: str, session_metadata: Dict[str, Any]) -> Optional[str]:
    """Resolve message_id from session metadata for follow-ups."""
    messages = session_metadata.get("outlook_message_memory") or []
    active = session_metadata.get("active_outlook_message") or {}
    if not messages and not active:
        return None

    q_lower = (query or "").lower()
    stopwords = {"re", "fwd", "fw", "the", "your", "from", "email", "mail", "message", "thread", "outlook"}

    for entry in messages:
        subj = (entry.get("subject") or "").lower()
        if not subj or subj == "(no subject)":
            continue
        tokens = [w for w in re.findall(r"[a-z0-9]{3,}", subj) if w not in stopwords]
        if tokens and sum(1 for tok in tokens[:4] if tok in q_lower) >= 1:
            mid = entry.get("message_id")
            logger.info("Outlook follow-up matched subject %r -> message_id=%s", entry.get("subject"), mid)
            return mid

        sender = (entry.get("sender") or "").lower()
        sender_token = sender.split("<")[0].strip().split()[0] if sender else ""
        if sender_token and len(sender_token) > 2 and sender_token in q_lower:
            mid = entry.get("message_id")
            logger.info("Outlook follow-up matched sender %r -> message_id=%s", sender_token, mid)
            return mid

    if is_outlook_follow_up_query(query):
        mid = active.get("message_id")
        if mid:
            logger.info("Outlook follow-up phrase detected -> active message_id=%s", mid)
            return mid
    return None


def persist_outlook_thread_memory(memory: Any, outlook_messages: List[Dict[str, Any]]) -> None:
    """Store top Outlook messages in session metadata after a successful turn."""
    if memory is None or not outlook_messages:
        return

    entries: List[Dict[str, Any]] = []
    for row in outlook_messages[:5]:
        entries.append({
            "message_id": row.get("message_id"),
            "conversation_id": row.get("conversation_id"),
            "subject": row.get("subject"),
            "sender": row.get("sender"),
            "score": row.get("score", 0.0),
        })

    entries = [e for e in entries if e.get("message_id")]
    if not entries:
        return

    memory.session.metadata["outlook_message_memory"] = entries
    memory.session.metadata["active_outlook_message"] = entries[0]
    memory.session.metadata["active_mcp_tool"] = "outlook"
    logger.info(
        "Persisted Outlook message memory: active message_id=%s subject=%r (%d messages)",
        entries[0].get("message_id"),
        entries[0].get("subject"),
        len(entries),
    )


def apply_outlook_follow_up_routing(
    query: str,
    session_metadata: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Route follow-up queries to get_message using session memory."""
    message_id = match_outlook_message_from_memory(query, session_metadata)
    if not message_id:
        return None
    tool, params = prepare_outlook_call_params(query, message_id=message_id)
    logger.info("Outlook follow-up routing -> get_message(%s)", message_id)
    return tool, params


def patch_outlook_call_params(
    tool: str,
    params: Dict[str, Any],
    *,
    outlook_folder: Optional[str] = None,
    outlook_location: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Merge UI #folder hints into an existing tool/params pair."""
    if not outlook_folder and not outlook_location:
        return tool, params
    struct = parse_outlook_query_struct(
        params.get("query") or "",
        outlook_folder=outlook_folder,
        outlook_location=outlook_location,
    )
    params = dict(params)
    params["outlook_folder"] = struct.folder
    if struct.is_unread:
        params["is_unread"] = True
        if tool == "list_recent_inbox":
            tool = "list_unread"
    logger.info("patch_outlook_call_params: folder=%s tool=%s", struct.folder, tool)
    return tool, params


def build_outlook_search_query(query: str, sender: Optional[str], subjects: List[str]) -> str:
    """Build a Graph $search KQL string."""
    parts: List[str] = []
    if sender:
        if "@" in sender:
            parts.append(f"from:{sender}")
        else:
            parts.append(f"from:{sender}")
    for subject in subjects:
        parts.append(f"subject:{subject}")
    if not parts:
        cleaned = _strip_conversational_noise(query)
        if cleaned:
            parts.append(cleaned)
    search_q = " AND ".join(parts) if len(parts) > 1 else (parts[0] if parts else query)
    logger.info("build_outlook_search_query: %r -> %r", query[:80], search_q[:120])
    return search_q


def resolve_outlook_tool(query: str, message_id: Optional[str] = None, *, is_unread: bool = False) -> str:
    if message_id:
        return "get_message"
    if is_unread or re.search(r"\bunread\b", (query or "").lower()):
        return "list_unread"
    q_lower = (query or "").lower()
    if _INBOX_HINTS.search(query or ""):
        return "list_recent_inbox"
    if "inbox" in q_lower and any(
        token in q_lower for token in ("recent", "latest", "my", "check", "show", "list", "today")
    ):
        return "list_recent_inbox"
    return "search_messages"


def prepare_outlook_call_params(
    query: str,
    message_id: Optional[str] = None,
    *,
    outlook_folder: Optional[str] = None,
    outlook_location: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build tool + params for Microsoft365MCPClient.call_tool."""
    struct = parse_outlook_query_struct(
        query,
        outlook_folder=outlook_folder,
        outlook_location=outlook_location,
    )
    mid = message_id or struct.message_id or _extract_message_id(query)
    tool = resolve_outlook_tool(query, mid, is_unread=struct.is_unread)
    params: Dict[str, Any] = {
        "query": query,
        "outlook_folder": struct.folder,
    }
    if mid:
        params["message_id"] = mid
    if conversation_id:
        params["conversation_id"] = conversation_id
    if struct.is_unread:
        params["is_unread"] = True
    if struct.date_after:
        params["date_after"] = struct.date_after
    if struct.date_before:
        params["date_before"] = struct.date_before
    if tool == "search_messages":
        sender = struct.sender or _extract_sender(query)
        subjects = struct.subject_keywords or _extract_subject_keywords(query)
        params["search_query"] = build_outlook_search_query(query, sender, subjects)
    logger.info("prepare_outlook_call_params: tool=%s params_keys=%s", tool, list(params.keys()))
    return tool, params


def _sender_display(message: Dict[str, Any]) -> str:
    from_block = message.get("from", {}) or {}
    email = (from_block.get("emailAddress") or {})
    name = (email.get("name") or "").strip()
    address = (email.get("address") or "").strip()
    if name and address:
        return f"{name} <{address}>"
    return address or name or "Unknown"


def _body_text(message: Dict[str, Any], *, include_body: bool) -> str:
    if not include_body:
        return (message.get("bodyPreview") or "").strip()
    body = message.get("body") or {}
    content = (body.get("content") or "").strip()
    content_type = (body.get("contentType") or "").lower()
    if content_type == "html" and content:
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()
    if not content:
        content = (message.get("bodyPreview") or "").strip()
    if len(content) > _BODY_MAX_CHARS:
        content = content[:_BODY_MAX_CHARS] + f"\n...[truncated, {len(content) - _BODY_MAX_CHARS} more chars]"
    return content


def _message_record(message: Dict[str, Any], *, include_body: bool = False) -> OutlookMessageRecord:
    return OutlookMessageRecord(
        message_id=message.get("id") or "",
        conversation_id=message.get("conversationId") or "",
        subject=(message.get("subject") or "(no subject)").strip(),
        sender=_sender_display(message),
        received=(message.get("receivedDateTime") or "").strip(),
        preview=(message.get("bodyPreview") or "").strip(),
        body=_body_text(message, include_body=include_body),
    )


def format_outlook_results(records: List[OutlookMessageRecord]) -> str:
    if not records:
        return "No Outlook messages matched your query."
    lines = [f"Found {len(records)} Outlook message(s):\n"]
    for rec in records:
        body_block = ""
        if rec.body:
            body_block = f"\n  **Body**:\n{rec.body}\n"
        elif rec.preview:
            body_block = f"\n  **Preview**: {rec.preview}\n"
        lines.append(
            f"[Outlook Message | message_id={rec.message_id} | conversation_id={rec.conversation_id}]\n"
            f"- **From**: {rec.sender}\n"
            f"  **Subject**: {rec.subject}\n"
            f"  **Received**: {rec.received}\n"
            f"{body_block}"
        )
    return "\n".join(lines)


def _fetch_message(
    client: httpx.Client,
    headers: Dict[str, str],
    message_id: str,
    *,
    include_body: bool,
) -> Optional[OutlookMessageRecord]:
    select_fields = "id,conversationId,subject,from,receivedDateTime,bodyPreview"
    if include_body:
        select_fields += ",body"
    url = f"{GRAPH_BASE}/me/messages/{message_id}"
    logger.debug("Fetching Outlook message id=%s include_body=%s", message_id, include_body)
    resp = client.get(url, headers=headers, params={"$select": select_fields})
    if resp.status_code != 200:
        logger.warning("Outlook get message failed status=%s body=%s", resp.status_code, resp.text[:200])
        return None
    return _message_record(resp.json(), include_body=include_body)


def _fetch_conversation_messages(
    client: httpx.Client,
    headers: Dict[str, str],
    conversation_id: str,
) -> List[OutlookMessageRecord]:
    if not conversation_id:
        return []
    url = f"{GRAPH_BASE}/me/messages"
    params = {
        "$filter": f"conversationId eq '{conversation_id}'",
        "$orderby": "receivedDateTime asc",
        "$top": str(_THREAD_MAX_MESSAGES),
        "$select": "id,conversationId,subject,from,receivedDateTime,bodyPreview,body",
    }
    logger.debug("Fetching Outlook conversation id=%s", conversation_id)
    resp = client.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        logger.warning(
            "Outlook conversation fetch failed status=%s body=%s",
            resp.status_code,
            resp.text[:200],
        )
        return []
    values = resp.json().get("value") or []
    records = [_message_record(msg, include_body=True) for msg in values]
    logger.info("Outlook conversation fetched %d messages", len(records))
    return records


def _list_folder_messages(
    client: httpx.Client,
    headers: Dict[str, str],
    *,
    folder: str = "inbox",
    unread_only: bool = False,
    date_after: Optional[str] = None,
    date_before: Optional[str] = None,
    top: int = _INBOX_TOP,
) -> List[OutlookMessageRecord]:
    folder = (folder or "inbox").lower().strip()
    url = f"{GRAPH_BASE}/me/mailFolders/{folder}/messages"
    filters: List[str] = []
    if unread_only:
        filters.append("isRead eq false")
    if date_after:
        iso_after = date_after if "T" in date_after else f"{date_after}T00:00:00Z"
        filters.append(f"receivedDateTime ge {iso_after}")
    if date_before:
        iso_before = date_before if "T" in date_before else f"{date_before}T23:59:59Z"
        filters.append(f"receivedDateTime le {iso_before}")

    params: Dict[str, Any] = {
        "$top": str(top),
        "$orderby": "receivedDateTime desc",
        "$select": "id,conversationId,subject,from,receivedDateTime,bodyPreview,isRead",
    }
    if filters:
        params["$filter"] = " and ".join(filters)

    logger.info(
        "Listing Outlook folder=%s unread_only=%s top=%s filters=%s",
        folder,
        unread_only,
        top,
        filters,
    )
    resp = client.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        logger.warning("Outlook folder list failed folder=%s status=%s", folder, resp.status_code)
        return []
    values = resp.json().get("value") or []
    return [_message_record(msg, include_body=False) for msg in values]


def _list_recent_inbox(
    client: httpx.Client,
    headers: Dict[str, str],
    *,
    folder: str = "inbox",
) -> List[OutlookMessageRecord]:
    return _list_folder_messages(client, headers, folder=folder)


def _list_unread_inbox(
    client: httpx.Client,
    headers: Dict[str, str],
    *,
    folder: str = "inbox",
) -> List[OutlookMessageRecord]:
    return _list_folder_messages(client, headers, folder=folder, unread_only=True)


def _search_messages(
    client: httpx.Client,
    headers: Dict[str, str],
    search_query: str,
) -> List[OutlookMessageRecord]:
    url = f"{GRAPH_BASE}/me/messages"
    params = {
        "$search": f"\"{search_query}\"",
        "$top": str(_SEARCH_TOP),
        "$select": "id,conversationId,subject,from,receivedDateTime,bodyPreview,body",
    }
    search_headers = dict(headers)
    search_headers["ConsistencyLevel"] = "eventual"
    logger.info("Outlook $search query=%r top=%s", search_query, _SEARCH_TOP)
    resp = client.get(url, headers=search_headers, params=params)
    if resp.status_code != 200:
        logger.warning("Outlook search failed status=%s body=%s", resp.status_code, resp.text[:300])
        return []
    values = resp.json().get("value") or []
    return [_message_record(msg, include_body=idx < 3) for idx, msg in enumerate(values)]


def run_outlook_tool(
    token: str,
    tool: str,
    params: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    """Dispatch Outlook tools: list_recent_inbox, search_messages, get_message."""
    query = (params.get("query") or "").strip()
    message_id = params.get("message_id")
    search_query = (params.get("search_query") or query).strip()

    resolved_tool = tool
    if tool not in ("list_recent_inbox", "list_unread", "search_messages", "get_message"):
        resolved_tool = resolve_outlook_tool(
            query,
            message_id,
            is_unread=bool(params.get("is_unread")),
        )

    folder = (params.get("outlook_folder") or "inbox").lower().strip()
    date_after = params.get("date_after")
    date_before = params.get("date_before")

    logger.info(
        "run_outlook_tool: tool=%s resolved=%s query=%r message_id=%s folder=%s",
        tool,
        resolved_tool,
        query[:120],
        message_id,
        folder,
    )

    headers = _graph_headers(token)

    with httpx.Client(timeout=timeout) as client:
        if resolved_tool == "get_message":
            mid = message_id or _extract_message_id(query)
            if not mid:
                return {
                    "content": [{"type": "text", "text": "No message_id provided for get_message."}],
                }
            primary = _fetch_message(client, headers, mid, include_body=True)
            if not primary:
                return {
                    "content": [{"type": "text", "text": f"No Outlook message found for message_id={mid}."}],
                }
            thread = _fetch_conversation_messages(client, headers, primary.conversation_id)
            records = thread or [primary]
            text = format_outlook_results(records)
            return {
                "content": [{"type": "text", "text": text}],
                "outlook_messages": [r.to_dict() for r in records],
            }

        if resolved_tool == "list_unread":
            records = _list_unread_inbox(client, headers, folder=folder)
            if not records:
                return {"content": [{"type": "text", "text": f"No unread messages in Outlook {folder}."}]}
            text = format_outlook_results(records)
            return {
                "content": [{"type": "text", "text": text}],
                "outlook_messages": [r.to_dict() for r in records],
            }

        if resolved_tool == "list_recent_inbox":
            records = _list_folder_messages(
                client,
                headers,
                folder=folder,
                date_after=date_after,
                date_before=date_before,
            )
            if not records:
                return {"content": [{"type": "text", "text": f"No messages in Outlook {folder}."}]}
            text = format_outlook_results(records)
            return {
                "content": [{"type": "text", "text": text}],
                "outlook_messages": [r.to_dict() for r in records],
            }

        if not search_query:
            search_query = build_outlook_search_query(query, _extract_sender(query), _extract_subject_keywords(query))
        records = _search_messages(client, headers, search_query)
        if not records:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"No Outlook messages matched `{search_query}`.",
                    }
                ],
            }
        text = format_outlook_results(records)
        return {
            "content": [{"type": "text", "text": text}],
            "outlook_messages": [r.to_dict() for r in records],
        }
