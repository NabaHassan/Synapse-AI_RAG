"""
Gmail search helpers: NL → Gmail operators, multi-stage search, rerank, full body fetch.

Implements improvements #1–7 from the Gmail MCP accuracy plan.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

GMAIL_THREADS_URL = "https://gmail.googleapis.com/gmail/v1/users/me/threads"
GMAIL_PROFILE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/profile"

_FETCH_CANDIDATES = int(os.getenv("GMAIL_FETCH_CANDIDATES", "20"))
_RERANK_TOP_K = int(os.getenv("GMAIL_RERANK_TOP_K", "5"))
_BODY_TOP_K = int(os.getenv("GMAIL_BODY_TOP_K", "3"))
_BODY_MAX_CHARS = int(os.getenv("GMAIL_BODY_MAX_CHARS", "3000"))

_CONVERSATIONAL_PREFIXES = [
    r"^(?:please\s+)?(?:can you\s+)?(?:could you\s+)?(?:would you\s+)?",
    r"^(?:tell me about|show me|give me|list|check|find|search for|look up|look at)\s+(?:my\s+)?",
    r"^(?:what are|what is|what were|what's|whats)\s+(?:my\s+)?",
    r"^(?:do i have|did i get|did i receive|have i got|have i received)\s+(?:any\s+)?",
    r"^(?:any|some)\s+",
    r"^(?:emails?|mail|messages?|inbox)\s+(?:about|from|regarding|on|for|with)\s+",
    r"^(?:emails?|mail|messages?|inbox)\s+",
    r"^(?:my\s+)?(?:recent\s+)?(?:emails?|mail|messages?|inbox)\s*",
]

_FROM_PATTERNS = [
    re.compile(r"\b(?:from|by|sent by)\s+([A-Za-z0-9._%+\-@ ]{2,80})", re.I),
    re.compile(r"\b(?:email(?:s)?|mail|message(?:s)?)\s+from\s+([A-Za-z0-9._%+\-@ ]{2,80})", re.I),
    re.compile(r"\b(?:did i (?:email|send|mail)|emails? i sent to)\s+([A-Za-z0-9._%+\-@ ]{2,60})", re.I),
]

_SUBJECT_PATTERNS = [
    re.compile(r'\bsubject\s+(?:is\s+)?["\']?([^"\']+)["\']?', re.I),
    re.compile(r"\b(?:about|regarding|re:|on the topic of)\s+(.+?)(?:\s+(?:today|yesterday|this week|last week|unread|starred|with attachment)|$)", re.I),
]

_THREAD_ID_PATTERN = re.compile(r"\bthread[_\s-]?id[=:\s]+([a-f0-9]{10,})\b", re.I)

# #location / #category tokens (#10) — Slack-style narrowing for Gmail
_GMAIL_HASH_LOCATIONS = {
    "inbox": "in:inbox",
    "sent": "in:sent",
    "drafts": "in:drafts",
    "draft": "in:drafts",
    "spam": "in:spam",
    "trash": "in:trash",
    "bin": "in:trash",
    "snoozed": "in:snoozed",
    "all": "in:anywhere",
    "anywhere": "in:anywhere",
}

_GMAIL_HASH_FLAGS = {
    "starred": "is:starred",
    "important": "is:important",
    "unread": "is:unread",
}

_GMAIL_HASH_CATEGORIES = {
    "primary": "category:primary",
    "promotions": "category:promotions",
    "promo": "category:promotions",
    "social": "category:social",
    "updates": "category:updates",
    "forums": "category:forums",
}

# @date chips (Notion/Google-style); parsed before @sender
_GMAIL_AT_DATE_ALIASES: Dict[str, str] = {
    "today": "today",
    "yesterday": "yesterday",
    "lastweek": "last week",
    "last-week": "last week",
    "thisweek": "this week",
    "this-week": "this week",
}

_GMAIL_FOLLOWUP_PATTERNS = [
    re.compile(r"\bwhat did (?:that|it) say\b", re.I),
    re.compile(r"\b(?:reply|respond) to (?:that|it|this)(?:\s+email|\s+thread|\s+message)?\b", re.I),
    re.compile(r"\b(?:that|this|the) (?:email|thread|message|mail)\b", re.I),
    re.compile(r"\bmore (?:about|on) (?:that|it|this)(?:\s+email|\s+thread|\s+message)?\b", re.I),
    re.compile(r"\b(?:summarize|summary of|read) (?:that|it|this)(?:\s+email|\s+thread|\s+message)?\b", re.I),
    re.compile(r"^(?:that one|this one|the one)\b", re.I),
    re.compile(r"\bwhat (?:did|does) (?:that|it) (?:say|mean)\b", re.I),
]


@dataclass
class GmailQueryParams:
    """Structured Gmail search parameters (#2)."""

    original_query: str = ""
    tool: str = "search_threads"
    sender: Optional[str] = None
    subject_keywords: List[str] = field(default_factory=list)
    keywords: Optional[str] = None
    date_after: Optional[str] = None  # Gmail format YYYY/MM/DD
    date_before: Optional[str] = None
    is_unread: bool = False
    is_starred: bool = False
    has_attachment: bool = False
    in_sent: bool = False
    category_primary: bool = False
    include_promotions: bool = False
    location: Optional[str] = None  # Gmail operator e.g. in:spam
    category: Optional[str] = None  # Gmail operator e.g. category:promotions
    boost_important: bool = True
    thread_id: Optional[str] = None
    timezone: str = "UTC"
    clean_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _default_timezone() -> str:
    return os.getenv("GMAIL_USER_TIMEZONE", os.getenv("TZ", "UTC")) or "UTC"


def _get_tz(tz_name: str):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name)
    except Exception:
        logger.warning("Invalid timezone %r — falling back to UTC", tz_name)
        from zoneinfo import ZoneInfo
        return ZoneInfo("UTC")


def _gmail_date(dt: datetime) -> str:
    return dt.strftime("%Y/%m/%d")


def default_ytd_date_range(tz_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Default Gmail search window: Jan 1 of the current year through end of today (exclusive).
    Replaces the previous rolling 7-day default that produced narrow ranges like May 28–Jun 5.
    """
    tz_name = tz_name or _default_timezone()
    tz = _get_tz(tz_name)
    now = datetime.now(tz)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    year_start = datetime(now.year, 1, 1, tzinfo=tz)
    date_after = _gmail_date(year_start)
    date_before = _gmail_date(start_of_today + timedelta(days=1))
    logger.debug(
        "Default YTD date range in tz=%s: after=%s before=%s",
        tz_name,
        date_after,
        date_before,
    )
    return date_after, date_before


def parse_date_range(query: str, tz_name: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse time phrases (#7): today, yesterday, this week, last week, last N days.
    Returns (after, before) in Gmail YYYY/MM/DD format (before is exclusive next day).
    """
    tz_name = tz_name or _default_timezone()
    tz = _get_tz(tz_name)
    now = datetime.now(tz)
    q = query.lower()

    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_exclusive = start_of_today + timedelta(days=1)

    if re.search(r"\btoday\b", q):
        logger.debug("Date range: today in tz=%s", tz_name)
        return _gmail_date(start_of_today), _gmail_date(end_exclusive)

    if re.search(r"\byesterday\b", q):
        y_start = start_of_today - timedelta(days=1)
        y_end = start_of_today
        logger.debug("Date range: yesterday in tz=%s", tz_name)
        return _gmail_date(y_start), _gmail_date(y_end)

    if re.search(r"\bthis week\b", q):
        week_start = start_of_today - timedelta(days=start_of_today.weekday())
        week_end = week_start + timedelta(days=7)
        logger.debug("Date range: this week in tz=%s", tz_name)
        return _gmail_date(week_start), _gmail_date(week_end)

    if re.search(r"\blast week\b", q):
        week_start = start_of_today - timedelta(days=start_of_today.weekday() + 7)
        week_end = week_start + timedelta(days=7)
        logger.debug("Date range: last week in tz=%s", tz_name)
        return _gmail_date(week_start), _gmail_date(week_end)

    last_n = re.search(r"\blast\s+(\d+)\s+days?\b", q)
    if last_n:
        n = int(last_n.group(1))
        start = start_of_today - timedelta(days=n)
        logger.debug("Date range: last %d days in tz=%s", n, tz_name)
        return _gmail_date(start), _gmail_date(end_exclusive)

    return None, None


def _strip_conversational_filler(query: str) -> str:
    text = query.strip()
    for _ in range(4):
        prev = text
        for pat in _CONVERSATIONAL_PREFIXES:
            text = re.sub(pat, "", text, flags=re.I).strip()
        if text == prev:
            break
    text = re.sub(r"\s+", " ", text).strip(" ?.,!")
    return text


def _extract_sender(query: str) -> Optional[str]:
    for pat in _FROM_PATTERNS:
        m = pat.search(query)
        if m:
            sender = m.group(1).strip().strip('"\'')
            sender = re.sub(r"\s+(?:today|yesterday|this week|last week|about|regarding).*$", "", sender, flags=re.I)
            if sender:
                logger.debug("Extracted sender: %r", sender)
                return sender
    return None


def _extract_subject_keywords(query: str) -> List[str]:
    for pat in _SUBJECT_PATTERNS:
        m = pat.search(query)
        if m:
            phrase = m.group(1).strip().strip('"\'')
            phrase = re.sub(r"\s+(?:today|yesterday|this week|last week|unread|starred).*$", "", phrase, flags=re.I)
            if phrase and len(phrase) > 1:
                logger.debug("Extracted subject keywords: %r", phrase)
                return [phrase]
    return []


def _extract_free_keywords(query: str, params: GmailQueryParams) -> Optional[str]:
    text = _strip_conversational_filler(query)
    text = re.sub(r"\b(?:from|by|sent by)\s+[A-Za-z0-9._%+\-@ ]+", "", text, flags=re.I)
    text = re.sub(r"\bsubject\s+[\"']?[^\"']+[\"']?", "", text, flags=re.I)
    for phrase in ("today", "yesterday", "this week", "last week", "recent", "latest", "unread", "starred", "include promotions"):
        text = re.sub(rf"\b{re.escape(phrase)}\b", "", text, flags=re.I)
    text = re.sub(r"\b(?:with|has)\s+attachment\b", "", text, flags=re.I)
    text = re.sub(r"\b(?:in\s+)?sent\b", "", text, flags=re.I)
    text = re.sub(r"\b(?:emails?|mail|messages?|inbox|threads?)\b", "", text, flags=re.I)
    text = re.sub(r"\blist\b", "", text, flags=re.I)
    if params.location:
        text = re.sub(r"#[\w-]+", "", text, flags=re.I)
    for token in list(_GMAIL_HASH_LOCATIONS.keys()) + list(_GMAIL_HASH_CATEGORIES.keys()) + list(_GMAIL_HASH_FLAGS.keys()):
        text = re.sub(rf"\b{re.escape(token)}\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ?.,!")
    if text and len(text) > 1:
        return text
    return None


def merge_llm_gmail_struct(params: GmailQueryParams, llm_struct: Optional[Dict[str, Any]]) -> GmailQueryParams:
    """Merge LLM-extracted fields into rule-based params (#2)."""
    if not llm_struct:
        return params

    tool = llm_struct.get("tool")
    if tool in ("list_recent_inbox", "search_threads", "get_thread", "search_sent"):
        params.tool = tool

    if llm_struct.get("sender"):
        params.sender = str(llm_struct["sender"]).strip()
    if llm_struct.get("subject_keywords"):
        sk = llm_struct["subject_keywords"]
        if isinstance(sk, list):
            params.subject_keywords = [str(s).strip() for s in sk if str(s).strip()]
        elif isinstance(sk, str) and sk.strip():
            params.subject_keywords = [sk.strip()]
    if llm_struct.get("keywords"):
        params.keywords = str(llm_struct["keywords"]).strip() or params.keywords
    if llm_struct.get("thread_id"):
        params.thread_id = str(llm_struct["thread_id"]).strip()

    dr = llm_struct.get("date_range") or {}
    if isinstance(dr, dict):
        after_raw = dr.get("after")
        before_raw = dr.get("before")
        if after_raw or before_raw:
            rel_query = " ".join(
                str(v) for v in (after_raw, before_raw) if v and str(v).lower() not in ("null", "none")
            )
            rule_after, rule_before = parse_date_range(rel_query or params.original_query, params.timezone)
            if after_raw and str(after_raw).lower() not in ("null", "none"):
                if rule_after:
                    params.date_after = rule_after
                else:
                    params.date_after = _normalize_date_for_gmail(str(after_raw))
            if before_raw and str(before_raw).lower() not in ("null", "none"):
                if rule_before:
                    params.date_before = rule_before
                else:
                    params.date_before = _normalize_date_for_gmail(str(before_raw))
        elif not params.date_after and not params.date_before:
            rule_after, rule_before = parse_date_range(params.original_query, params.timezone)
            params.date_after = rule_after or params.date_after
            params.date_before = rule_before or params.date_before

    if llm_struct.get("is_unread"):
        params.is_unread = True
    if llm_struct.get("is_starred"):
        params.is_starred = True
    if llm_struct.get("has_attachment"):
        params.has_attachment = True
    if llm_struct.get("in_sent"):
        params.in_sent = True
    if llm_struct.get("location") and not params.location:
        params.location = str(llm_struct["location"]).strip()
    if llm_struct.get("category") and not params.category:
        params.category = str(llm_struct["category"]).strip()

    logger.info(
        "Merged LLM Gmail struct: tool=%s location=%r category=%r sender=%r keywords=%r",
        params.tool,
        params.location,
        params.category,
        params.sender,
        params.keywords,
    )
    return params


def parse_gmail_mentions(query: str) -> Dict[str, Any]:
    """
    Parse @sender, @date, and #location/#category tokens.

    @john or @sarah@company.com — narrow to sender
    @today @yesterday @lastweek — date range chips
    #inbox #sent #spam #drafts #trash #snoozed #starred #primary #promotions etc.
    """
    result: Dict[str, Any] = {
        "sender": None,
        "location": None,
        "category": None,
        "date_after": None,
        "date_before": None,
        "is_starred": False,
        "is_unread": False,
        "is_important": False,
        "include_promotions": False,
        "clean_query": query,
    }

    q_lower = query.lower()
    if re.search(r"\binclude promotions\b", q_lower):
        result["include_promotions"] = True

    tz = os.getenv("GMAIL_USER_TIMEZONE", os.getenv("TZ", "UTC")) or "UTC"
    for at_date in re.finditer(r"@([\w-]+)", query, re.I):
        token = at_date.group(1).lower()
        phrase = _GMAIL_AT_DATE_ALIASES.get(token)
        if phrase:
            after, before = parse_date_range(phrase, tz)
            if after:
                result["date_after"] = after
            if before:
                result["date_before"] = before
            logger.debug("Gmail @date: @%s -> after=%r before=%r", token, after, before)

    hash_match = re.search(r"#([\w-]+)", query)
    if hash_match:
        token = hash_match.group(1).lower()
        if token in _GMAIL_HASH_LOCATIONS:
            result["location"] = _GMAIL_HASH_LOCATIONS[token]
            logger.debug("Gmail # mention location: #%s -> %s", token, result["location"])
        elif token in _GMAIL_HASH_CATEGORIES:
            result["category"] = _GMAIL_HASH_CATEGORIES[token]
            if token in ("promotions", "promo"):
                result["include_promotions"] = True
            logger.debug("Gmail # mention category: #%s -> %s", token, result["category"])
        elif token in _GMAIL_HASH_FLAGS:
            flag = _GMAIL_HASH_FLAGS[token]
            if flag == "is:starred":
                result["is_starred"] = True
            elif flag == "is:unread":
                result["is_unread"] = True
            elif flag == "is:important":
                result["is_important"] = True
            logger.debug("Gmail # mention flag: #%s -> %s", token, flag)

    at_match = re.search(
        r"@([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
        query,
    )
    if at_match:
        result["sender"] = at_match.group(1).strip()
    else:
        at_name = re.search(r"@([A-Za-z][A-Za-z0-9.'\-]+)", query)
        if at_name:
            token = at_name.group(1).strip().lower()
            if token not in _GMAIL_AT_DATE_ALIASES:
                result["sender"] = at_name.group(1).strip()

    clean = query
    clean = re.sub(r"#[\w-]+", " ", clean)
    clean = re.sub(r"@[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", " ", clean)
    clean = re.sub(r"@[A-Za-z][A-Za-z0-9.'\-]+", " ", clean)
    clean = re.sub(r"\binclude promotions\b", " ", clean, flags=re.I)
    clean = re.sub(r"\s+", " ", clean).strip()
    result["clean_query"] = clean or query
    logger.debug("parse_gmail_mentions: sender=%r location=%r category=%r clean=%r", result["sender"], result["location"], result["category"], result["clean_query"])
    return result


def is_gmail_follow_up_query(query: str) -> bool:
    """True when the user likely refers to a prior Gmail thread (#8)."""
    return any(pat.search(query) for pat in _GMAIL_FOLLOWUP_PATTERNS)


def match_gmail_thread_from_memory(query: str, session_metadata: Dict[str, Any]) -> Optional[str]:
    """Resolve thread_id from session metadata for follow-ups (#8)."""
    threads = session_metadata.get("gmail_thread_memory") or []
    active = session_metadata.get("active_gmail_thread") or {}
    if not threads and not active:
        return None

    q_lower = query.lower()
    stopwords = {"re", "fwd", "fw", "the", "your", "from", "email", "mail", "message", "thread"}

    for entry in threads:
        subj = (entry.get("subject") or "").lower()
        if not subj or subj == "no subject":
            continue
        tokens = [w for w in re.findall(r"[a-z0-9]{3,}", subj) if w not in stopwords]
        if tokens and sum(1 for tok in tokens[:4] if tok in q_lower) >= 1:
            tid = entry.get("thread_id")
            logger.info("Gmail follow-up matched subject %r -> thread_id=%s", entry.get("subject"), tid)
            return tid

        sender = (entry.get("from") or entry.get("sender") or "").lower()
        sender_token = sender.split("<")[0].strip().split()[0] if sender else ""
        if sender_token and len(sender_token) > 2 and sender_token in q_lower:
            tid = entry.get("thread_id")
            logger.info("Gmail follow-up matched sender %r -> thread_id=%s", sender_token, tid)
            return tid

    if is_gmail_follow_up_query(query):
        tid = active.get("thread_id")
        if tid:
            logger.info("Gmail follow-up phrase detected -> active thread_id=%s", tid)
            return tid

    return None


def persist_gmail_thread_memory(memory: Any, gmail_threads: List[Dict[str, Any]]) -> None:
    """Store top Gmail threads in session metadata after a successful turn (#8)."""
    if memory is None or not gmail_threads:
        return

    entries: List[Dict[str, Any]] = []
    for row in gmail_threads[:5]:
        entries.append({
            "thread_id": row.get("thread_id"),
            "subject": row.get("subject"),
            "from": row.get("sender") or row.get("from"),
            "score": row.get("score", 0.0),
        })

    entries = [e for e in entries if e.get("thread_id")]
    if not entries:
        return

    memory.session.metadata["gmail_thread_memory"] = entries
    memory.session.metadata["active_gmail_thread"] = entries[0]
    memory.session.metadata["active_mcp_tool"] = "gmail"
    logger.info(
        "Persisted Gmail thread memory: active thread_id=%s subject=%r (%d threads)",
        entries[0].get("thread_id"),
        entries[0].get("subject"),
        len(entries),
    )


def apply_gmail_follow_up_routing(
    query: str,
    session_metadata: Dict[str, Any],
    embedder: Any = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Route follow-up queries to get_thread using session memory (#8)."""
    thread_id = match_gmail_thread_from_memory(query, session_metadata)
    if not thread_id:
        return None
    tool, params = prepare_gmail_call_params(query, thread_id=thread_id, embedder=embedder)
    logger.info("Gmail follow-up routing -> get_thread(%s)", thread_id)
    return tool, params


def _normalize_date_for_gmail(value: str) -> str:
    value = value.strip()
    if re.match(r"^\d{4}/\d{2}/\d{2}$", value):
        return value
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value.replace("-", "/")
    return value


def parse_gmail_query_struct(
    query: str,
    llm_struct: Optional[Dict[str, Any]] = None,
    timezone: Optional[str] = None,
) -> GmailQueryParams:
    """Rule-based + optional LLM structured extraction (#2)."""
    tz = timezone or _default_timezone()
    mentions = parse_gmail_mentions(query)
    working_query = mentions.get("clean_query") or query

    params = GmailQueryParams(original_query=query, timezone=tz, clean_query=working_query)

    q_lower = working_query.lower()
    params.is_unread = bool(re.search(r"\bunread\b", q_lower)) or mentions.get("is_unread", False)
    params.is_starred = bool(re.search(r"\bstarred\b", q_lower)) or mentions.get("is_starred", False)
    params.has_attachment = bool(re.search(r"\b(?:with|has)\s+attachment\b", q_lower))
    params.in_sent = bool(re.search(r"\b(?:in\s+sent|sent\s+(?:mail|email|message)|did i (?:send|email)|emails? i sent)\b", q_lower))
    params.include_promotions = mentions.get("include_promotions", False)
    params.location = mentions.get("location")
    params.category = mentions.get("category")
    if mentions.get("is_important"):
        params.boost_important = True

    if params.location == "in:sent":
        params.in_sent = True

    thread_m = _THREAD_ID_PATTERN.search(query)
    if thread_m:
        params.thread_id = thread_m.group(1)
        params.tool = "get_thread"

    if mentions.get("sender"):
        params.sender = mentions["sender"]
    if not params.sender:
        params.sender = _extract_sender(working_query)

    params.subject_keywords = _extract_subject_keywords(working_query)
    date_after, date_before = parse_date_range(working_query, tz)
    if not date_after and not date_before:
        date_after, date_before = parse_date_range(query, tz)
    if mentions.get("date_after"):
        date_after = mentions["date_after"]
    if mentions.get("date_before"):
        date_before = mentions["date_before"]
    params.date_after = date_after
    params.date_before = date_before

    params.keywords = _extract_free_keywords(working_query, params)

    generic_list = re.search(
        r"\b(?:recent|latest|today|yesterday|this week|what are my emails|check my inbox|my emails)\b",
        q_lower,
    )
    if generic_list:
        if (
            not params.sender
            and not params.subject_keywords
            and not params.keywords
            and not params.in_sent
            and not params.location
            and not params.category
        ):
            params.tool = "list_recent_inbox"
            if not params.category and not params.include_promotions:
                params.category_primary = True

    if params.in_sent and params.tool == "search_threads":
        params.tool = "search_sent"

    return merge_llm_gmail_struct(params, llm_struct)


def apply_gmail_connector_hints(
    params: GmailQueryParams,
    *,
    gmail_location: Optional[str] = None,
    gmail_category: Optional[str] = None,
) -> GmailQueryParams:
    """Apply UI #folder / #category selections from composer chips."""
    if gmail_location:
        if gmail_location.startswith("is:"):
            if gmail_location == "is:starred":
                params.is_starred = True
            elif gmail_location == "is:important":
                params.boost_important = True
            logger.info("Gmail connector hint: flag=%s", gmail_location)
        else:
            params.location = gmail_location
            logger.info("Gmail connector hint: location=%s", gmail_location)
    if gmail_category:
        params.category = gmail_category
        if "promotions" in (gmail_category or ""):
            params.include_promotions = True
        logger.info("Gmail connector hint: category=%s", gmail_category)
    if params.location or params.category:
        if params.tool == "list_recent_inbox":
            params.tool = "search_threads"
    return params


def resolve_gmail_tool(query: str, params: GmailQueryParams) -> str:
    """Pick Gmail tool by intent (#6)."""
    if params.thread_id or params.tool == "get_thread":
        return "get_thread"
    if params.in_sent or params.tool == "search_sent":
        return "search_sent"
    if params.location or params.category or params.is_starred or params.is_unread:
        return "search_threads"
    if params.tool == "list_recent_inbox":
        return "list_recent_inbox"
    q_lower = query.lower()
    if re.search(r"\b(?:recent|latest|today|yesterday|this week|my inbox|check my emails)\b", q_lower):
        if not params.sender and not params.subject_keywords and not params.keywords:
            return "list_recent_inbox"
    return "search_threads"


def build_gmail_search_query(params: GmailQueryParams, stage: str = "strict") -> str:
    """
    Compile structured params → Gmail q= string (#1, #10).
    stage: strict | relaxed | broad
    """
    parts: List[str] = []

    if params.location:
        parts.append(params.location)
    elif params.tool == "search_sent" or params.in_sent:
        parts.append("in:sent")
    elif params.tool == "list_recent_inbox":
        parts.append("in:inbox")
    else:
        if stage == "broad":
            parts.append("(in:inbox OR in:sent)")
        else:
            parts.append("in:inbox")

    if params.category:
        parts.append(params.category)
    elif params.category_primary and not params.include_promotions and stage != "broad":
        parts.append("category:primary")

    if stage == "strict":
        if params.sender:
            sender = params.sender.replace('"', "")
            if "@" in sender:
                parts.append(f"from:{sender}")
            else:
                parts.append(f"from:{sender.split()[0]}")
        for sk in params.subject_keywords:
            sk_clean = sk.replace('"', "")
            parts.append(f'subject:"{sk_clean}"')
        if params.keywords:
            parts.append(params.keywords)
    elif stage == "relaxed":
        if params.sender:
            sender_token = params.sender.split()[0].replace('"', "")
            parts.append(f"from:{sender_token}")
        if params.subject_keywords:
            parts.append(params.subject_keywords[0])
        elif params.keywords:
            parts.append(params.keywords)
    else:  # broad
        kw = params.keywords or (params.subject_keywords[0] if params.subject_keywords else None)
        if kw:
            parts.append(kw)
        elif params.sender:
            parts.append(params.sender.split()[0])

    if params.date_after and stage != "broad":
        parts.append(f"after:{params.date_after}")
    if params.date_before and stage != "broad":
        parts.append(f"before:{params.date_before}")
    elif stage == "relaxed" and params.date_after:
        parts.append(f"after:{params.date_after}")

    if params.is_unread:
        parts.append("is:unread")
    if params.is_starred:
        parts.append("is:starred")
    if params.has_attachment:
        parts.append("has:attachment")

    q = " ".join(p for p in parts if p).strip()
    logger.debug("build_gmail_search_query stage=%s -> %r", stage, q)
    return q


def build_search_stages(params: GmailQueryParams) -> List[Tuple[str, str]]:
    """Multi-stage queries: strict → relaxed → broad (#3)."""
    if params.tool == "list_recent_inbox":
        if not params.date_after and not params.date_before:
            params.date_after, params.date_before = default_ytd_date_range(params.timezone)
            logger.info(
                "list_recent_inbox: using year-to-date default after=%s before=%s",
                params.date_after,
                params.date_before,
            )
        q = build_gmail_search_query(params, stage="strict")
        return [("recent_inbox", q)]

    if params.tool == "search_sent":
        return [
            ("sent_strict", build_gmail_search_query(params, stage="strict")),
            ("sent_relaxed", build_gmail_search_query(params, stage="relaxed")),
            ("sent_broad", build_gmail_search_query(params, stage="broad")),
        ]

    stages = [
        ("strict", build_gmail_search_query(params, stage="strict")),
        ("relaxed", build_gmail_search_query(params, stage="relaxed")),
        ("broad", build_gmail_search_query(params, stage="broad")),
    ]
    return [(name, q) for name, q in stages if q]


def _parse_headers(payload: Dict[str, Any]) -> Dict[str, str]:
    headers = {}
    for h in payload.get("headers") or []:
        name = (h.get("name") or "").lower()
        if name:
            headers[name] = h.get("value") or ""
    return headers


def extract_plain_body(payload: Dict[str, Any], max_chars: int = _BODY_MAX_CHARS) -> str:
    """Extract plain text from a Gmail message payload (#5)."""

    def _walk(part: Dict[str, Any]) -> str:
        mime = (part.get("mimeType") or "").lower()
        body = part.get("body") or {}
        data = body.get("data")
        if mime == "text/plain" and data:
            try:
                raw = base64.urlsafe_b64decode(data + "==")
                return raw.decode("utf-8", errors="replace")
            except Exception as exc:
                logger.debug("Failed to decode text/plain body: %s", exc)
        for child in part.get("parts") or []:
            text = _walk(child)
            if text:
                return text
        if mime == "text/html" and data:
            try:
                raw = base64.urlsafe_b64decode(data + "==")
                html = raw.decode("utf-8", errors="replace")
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text).strip()
                return text
            except Exception as exc:
                logger.debug("Failed to decode text/html body: %s", exc)
        return ""

    text = _walk(payload)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n[... truncated ...]\n\n{tail}"


@dataclass
class ThreadRecord:
    thread_id: str
    subject: str
    sender: str
    date: str
    snippet: str
    body: str = ""
    score: float = 0.0
    search_stage: str = ""
    is_important: bool = False

    def card_text(self) -> str:
        return f"Subject: {self.subject}\nFrom: {self.sender}\nDate: {self.date}\n{self.snippet}\n{self.body[:500]}"


def _score_thread_structural(record: ThreadRecord, params: GmailQueryParams, query: str) -> float:
    score = 0.1
    blob = f"{record.subject} {record.sender} {record.snippet} {record.body}".lower()
    q_lower = query.lower()

    if params.sender and params.sender.lower().split()[0] in blob:
        score += 0.35
    for sk in params.subject_keywords:
        if sk.lower() in blob:
            score += 0.25
    if params.keywords and params.keywords.lower() in blob:
        score += 0.2
    if params.is_unread and "unread" in q_lower:
        score += 0.05

    if record.is_important and params.boost_important:
        score += 0.12
        logger.debug("Important boost applied to thread %s", record.thread_id[:8])

    for token in re.findall(r"[a-z0-9]{3,}", q_lower):
        if token in blob:
            score += 0.02

    return min(score, 1.0)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _embed_text(embedder: Any, text: str) -> Optional[List[float]]:
    try:
        if hasattr(embedder, "run"):
            result = embedder.run(text=text)
            if isinstance(result, dict):
                emb = result.get("embedding")
                if emb is not None:
                    return list(emb[0] if isinstance(emb[0], (list, tuple)) else emb)
        if hasattr(embedder, "encode"):
            vec = embedder.encode([text])
            if hasattr(vec, "tolist"):
                return vec[0].tolist()
            return list(vec[0])
    except Exception as exc:
        logger.warning("Embedding failed for Gmail rerank: %s", exc)
    return None


def semantic_rerank_threads(
    records: List[ThreadRecord],
    query: str,
    params: GmailQueryParams,
    embedder: Any = None,
    top_k: int = _RERANK_TOP_K,
) -> List[ThreadRecord]:
    """Rerank candidate threads (#4): structural + optional embedding similarity."""
    if not records:
        return []

    for rec in records:
        rec.score = _score_thread_structural(rec, params, query)

    if embedder is not None and len(records) > 1:
        query_emb = _embed_text(embedder, query)
        if query_emb:
            for rec in records:
                card_emb = _embed_text(embedder, rec.card_text())
                if card_emb:
                    sim = _cosine_similarity(query_emb, card_emb)
                    rec.score = 0.45 * rec.score + 0.55 * max(0.0, sim)
                    logger.debug("Thread %s semantic score=%.3f", rec.thread_id[:8], rec.score)

    records.sort(key=lambda r: r.score, reverse=True)
    top = records[:top_k]
    logger.info(
        "Gmail rerank: %d candidates -> top %d (best score=%.3f)",
        len(records),
        len(top),
        top[0].score if top else 0.0,
    )
    return top


def _fetch_thread_record(
    client: httpx.Client,
    headers: Dict[str, str],
    thread_id: str,
    include_body: bool = False,
    search_stage: str = "",
) -> Optional[ThreadRecord]:
    fmt = "full" if include_body else "metadata"
    params = {"format": fmt}
    if fmt == "metadata":
        params["metadataHeaders"] = ["Subject", "From", "Date"]

    resp = client.get(f"{GMAIL_THREADS_URL}/{thread_id}", headers=headers, params=params)
    if resp.status_code != 200:
        logger.warning("Failed to fetch thread %s: HTTP %s", thread_id, resp.status_code)
        return None

    data = resp.json()
    messages = data.get("messages") or []
    if not messages:
        return None

    msg = messages[-1]
    hdrs = _parse_headers(msg.get("payload") or {})
    body = extract_plain_body(msg.get("payload") or {}) if include_body else ""
    label_ids = msg.get("labelIds") or []
    is_important = "IMPORTANT" in label_ids

    return ThreadRecord(
        thread_id=thread_id,
        subject=hdrs.get("subject") or "No Subject",
        sender=hdrs.get("from") or "Unknown Sender",
        date=hdrs.get("date") or "Unknown Date",
        snippet=msg.get("snippet") or "",
        body=body,
        search_stage=search_stage,
        is_important=is_important,
    )


def _list_thread_ids(
    client: httpx.Client,
    headers: Dict[str, str],
    q: str,
    max_results: int,
) -> List[str]:
    resp = client.get(
        GMAIL_THREADS_URL,
        headers=headers,
        params={"maxResults": max_results, "q": q},
    )
    resp.raise_for_status()
    threads = resp.json().get("threads") or []
    ids = [t["id"] for t in threads if t.get("id")]
    logger.info("Gmail list q=%r returned %d thread ids", q[:120], len(ids))
    return ids


def multi_stage_gmail_search(
    client: httpx.Client,
    headers: Dict[str, str],
    params: GmailQueryParams,
    min_hits: int = 3,
    fetch_candidates: int = _FETCH_CANDIDATES,
) -> Tuple[List[str], str]:
    """
    Run staged searches until enough thread ids (#3).
    Returns (thread_ids, winning_stage_name). No silent inbox fallback (#9).
    """
    stages = build_search_stages(params)
    all_ids: List[str] = []
    winning_stage = ""

    for stage_name, q in stages:
        if not q:
            continue
        try:
            ids = _list_thread_ids(client, headers, q, fetch_candidates)
        except Exception as exc:
            logger.warning("Gmail search stage %s failed: %s", stage_name, exc)
            continue

        for tid in ids:
            if tid not in all_ids:
                all_ids.append(tid)

        logger.info("Gmail stage %s: q=%r cumulative_ids=%d", stage_name, q[:100], len(all_ids))
        if len(all_ids) >= min_hits:
            winning_stage = stage_name
            break
        if ids and not winning_stage:
            winning_stage = stage_name

    return all_ids[:fetch_candidates], winning_stage


def format_gmail_results(records: List[ThreadRecord]) -> str:
    if not records:
        return "No emails found matching the request."

    lines = ["Recent Gmail messages:\n"]
    for rec in records:
        body_block = ""
        if rec.body:
            body_block = f"  **Body excerpt**: {rec.body[:800]}\n"
        lines.append(
            f"[Gmail Thread | thread_id={rec.thread_id} | score={rec.score:.2f} | stage={rec.search_stage}]\n"
            f"- **From**: {rec.sender}\n"
            f"  **Subject**: {rec.subject}\n"
            f"  **Date**: {rec.date}\n"
            f"  **Snippet**: {rec.snippet}\n"
            f"{body_block}"
        )
    return "\n".join(lines)


def run_gmail_tool(
    token: str,
    tool: str,
    params: Dict[str, Any],
    timeout: float,
    embedder: Any = None,
) -> Dict[str, Any]:
    """
    Main Gmail MCP entry: dispatches list_recent_inbox, search_threads, search_sent, get_thread (#6).
    """
    query = (params.get("query") or "").strip()
    llm_struct = params.get("gmail_struct")
    timezone = params.get("timezone") or _default_timezone()
    thread_id = params.get("thread_id")

    gmail_params = parse_gmail_query_struct(query, llm_struct=llm_struct, timezone=timezone)
    if thread_id:
        gmail_params.thread_id = thread_id
        gmail_params.tool = "get_thread"

    resolved_tool = tool if tool in ("list_recent_inbox", "search_threads", "search_sent", "get_thread") else resolve_gmail_tool(query, gmail_params)
    logger.info(
        "run_gmail_tool: tool=%s resolved=%s query=%r struct=%s",
        tool,
        resolved_tool,
        query[:120],
        gmail_params.to_dict(),
    )

    headers = {"Authorization": f"Bearer {token}"}

    with httpx.Client(timeout=timeout) as client:
        if resolved_tool == "get_thread":
            tid = gmail_params.thread_id
            if not tid:
                return {"content": [{"type": "text", "text": "No thread_id provided for get_thread."}]}
            rec = _fetch_thread_record(client, headers, tid, include_body=True, search_stage="get_thread")
            if not rec:
                return {"content": [{"type": "text", "text": f"No email thread found for thread_id={tid}."}]}
            rec.score = 1.0
            text = format_gmail_results([rec])
            return {"content": [{"type": "text", "text": text}], "gmail_threads": [asdict(rec)]}

        thread_ids, winning_stage = multi_stage_gmail_search(client, headers, gmail_params)

        if not thread_ids:
            compiled = build_gmail_search_query(gmail_params, stage="strict")
            msg = f"No emails matched `{compiled}`."
            logger.info("Gmail search returned zero results (no inbox fallback)")
            return {"content": [{"type": "text", "text": msg}]}

        records: List[ThreadRecord] = []
        for idx, tid in enumerate(thread_ids):
            include_body = idx < _BODY_TOP_K
            rec = _fetch_thread_record(
                client,
                headers,
                tid,
                include_body=include_body,
                search_stage=winning_stage,
            )
            if rec:
                records.append(rec)

        ranked = semantic_rerank_threads(records, query or gmail_params.original_query, gmail_params, embedder=embedder)
        text = format_gmail_results(ranked)
        logger.info("Gmail search complete: %d threads formatted", len(ranked))
        return {
            "content": [{"type": "text", "text": text}],
            "gmail_threads": [asdict(r) for r in ranked],
        }


def patch_gmail_call_params(
    tool: str,
    call_params: Dict[str, Any],
    *,
    gmail_location: Optional[str] = None,
    gmail_category: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Merge UI #folder hints into an existing Gmail tool call."""
    if not gmail_location and not gmail_category:
        return tool, call_params
    query = (call_params.get("query") or "").strip()
    gs = call_params.get("gmail_struct") or {}
    params = parse_gmail_query_struct(query, llm_struct=gs)
    params = apply_gmail_connector_hints(
        params,
        gmail_location=gmail_location,
        gmail_category=gmail_category,
    )
    call_params["gmail_struct"] = {**gs, **params.to_dict()}
    call_params["query"] = query
    return resolve_gmail_tool(query, params), call_params


def prepare_gmail_call_params(
    query: str,
    llm_struct: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    embedder: Any = None,
    gmail_location: Optional[str] = None,
    gmail_category: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build tool + params dict for google_workspace_client.call_tool."""
    params = parse_gmail_query_struct(query, llm_struct=llm_struct)
    params = apply_gmail_connector_hints(
        params,
        gmail_location=gmail_location,
        gmail_category=gmail_category,
    )
    tool = resolve_gmail_tool(query, params)
    call_params: Dict[str, Any] = {
        "query": query,
        "gmail_struct": llm_struct or params.to_dict(),
        "timezone": params.timezone,
    }
    if thread_id:
        call_params["thread_id"] = thread_id
        tool = "get_thread"
    if embedder is not None:
        call_params["_embedder"] = embedder
    return tool, call_params
