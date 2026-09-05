"""
Google Calendar search helpers: industry-aligned @ / # tokens.

@ — when / how to look (today, weekend, morning, agenda, pipeline, …)
# — which calendar world (tasks, work, family, …) or a named Google calendar

Legacy aliases (deprecated, still parsed): @lens and #time-scope.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

CALENDAR_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
CALENDAR_LIST_URL = "https://www.googleapis.com/calendar/v3/users/me/calendarList"

_MAX_RESULTS = int(os.getenv("CALENDAR_FETCH_MAX", "25"))

# @ — life-area lenses (match calendar names + event text)
_CALENDAR_AT_LENSES: Dict[str, Dict[str, Any]] = {
    "tasks": {
        "label": "Tasks & deadlines",
        "keywords": ["task", "todo", "to-do", "deadline", "due", "reminder", "action item"],
        "calendar_patterns": [r"task", r"todo", r"reminder", r"deadline"],
    },
    "task": {
        "label": "Tasks & deadlines",
        "keywords": ["task", "todo", "deadline", "due"],
        "calendar_patterns": [r"task", r"todo"],
    },
    "birthdays": {
        "label": "Birthdays",
        "keywords": ["birthday", "bday", "born", "turns"],
        "calendar_patterns": [r"birthday", r"bday", r"birthdays"],
    },
    "birthday": {
        "label": "Birthdays",
        "keywords": ["birthday", "bday"],
        "calendar_patterns": [r"birthday", r"bday"],
    },
    "holidays": {
        "label": "Holidays",
        "keywords": ["holiday", "observance", "public holiday", "day off"],
        "calendar_patterns": [r"holiday", r"holidays", r"observance"],
    },
    "holiday": {
        "label": "Holidays",
        "keywords": ["holiday", "observance"],
        "calendar_patterns": [r"holiday"],
    },
    "family": {
        "label": "Family",
        "keywords": ["family", "mom", "dad", "parent", "kids", "spouse", "dinner with"],
        "calendar_patterns": [r"family", r"home", r"personal"],
    },
    "work": {
        "label": "Work",
        "keywords": ["work", "office", "standup", "stand-up", "1:1", "sync", "sprint", "review"],
        "calendar_patterns": [r"work", r"office", r"company", r"corp"],
    },
    "health": {
        "label": "Health",
        "keywords": ["doctor", "dentist", "medical", "therapy", "clinic", "hospital", "gym", "workout"],
        "calendar_patterns": [r"health", r"medical", r"wellness"],
    },
    "travel": {
        "label": "Travel",
        "keywords": ["flight", "hotel", "trip", "travel", "airbnb", "airport", "vacation"],
        "calendar_patterns": [r"travel", r"trips", r"vacation"],
    },
    "focus": {
        "label": "Focus time",
        "keywords": ["focus", "deep work", "heads down", "blocked", "no meetings", "maker time"],
        "calendar_patterns": [r"focus", r"blocked"],
    },
    "kids": {
        "label": "Kids & school",
        "keywords": ["school", "soccer", "playdate", "pediatric", "pta", "pickup"],
        "calendar_patterns": [r"kids", r"children", r"school", r"family"],
    },
    "social": {
        "label": "Social",
        "keywords": ["dinner", "coffee", "lunch with", "party", "drinks", "hangout"],
        "calendar_patterns": [r"social", r"friends"],
    },
}

# @ — time scopes & view modes (also accepted as deprecated #today, #week, …)
_CALENDAR_TIME_SCOPES: Dict[str, str] = {
    "today": "today",
    "tomorrow": "tomorrow",
    "week": "this_week",
    "thisweek": "this_week",
    "nextweek": "next_week",
    "month": "this_month",
    "thismonth": "this_month",
    "morning": "morning_today",
    "afternoon": "afternoon_today",
    "evening": "evening_today",
    "night": "evening_today",
    "weekend": "weekend",
    "soon": "soon_48h",
    "agenda": "agenda",
    "pipeline": "pipeline_14d",
    "ytd": "ytd_forward",
    "year": "ytd_forward",
    "allday": "all_day_only",
    "busy": "has_attendees",
    "virtual": "virtual_only",
    "inperson": "in_person_only",
    "onsite": "in_person_only",
}

# Backward compatibility alias
_CALENDAR_HASH_SCOPES = _CALENDAR_TIME_SCOPES

_CALENDAR_LENS_TOKENS: frozenset[str] = frozenset(_CALENDAR_AT_LENSES.keys())


@dataclass
class CalendarQueryParams:
    original_query: str = ""
    clean_query: str = ""
    lens: Optional[str] = None
    lens_label: Optional[str] = None
    scope: Optional[str] = None
    scope_label: Optional[str] = None
    search_terms: List[str] = field(default_factory=list)
    calendar_ids: List[str] = field(default_factory=list)
    calendar_name_hint: Optional[str] = None
    timezone: str = "UTC"
    all_day_only: bool = False
    virtual_only: bool = False
    in_person_only: bool = False
    has_attendees_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _default_timezone() -> str:
    return os.getenv("GMAIL_USER_TIMEZONE", os.getenv("TZ", "UTC")) or "UTC"


def _get_tz(tz_name: str):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name)
    except Exception:
        from zoneinfo import ZoneInfo
        return ZoneInfo("UTC")


def _apply_calendar_lens(result: Dict[str, Any], token: str) -> None:
    lens_def = _CALENDAR_AT_LENSES.get(token)
    if not lens_def:
        return
    result["lens"] = token
    result["lens_label"] = lens_def.get("label", token)
    logger.debug("Calendar lens: %s -> %s", token, result["lens_label"])


def _apply_calendar_scope(result: Dict[str, Any], token: str) -> None:
    scope_key = _CALENDAR_TIME_SCOPES.get(token)
    if not scope_key:
        return
    result["scope"] = scope_key
    result["scope_label"] = token
    logger.debug("Calendar time scope: %s -> %s", token, scope_key)


def parse_calendar_mentions(query: str) -> Dict[str, Any]:
    """
    Parse calendar @ (time) and # (lens / named calendar) tokens.

    Preferred examples:
      #work @today meetings
      #family @weekend what's on?
      @tomorrow standup

    Legacy (still supported):
      @work #today
      #today @tasks
    """
    result: Dict[str, Any] = {
        "lens": None,
        "lens_label": None,
        "scope": None,
        "scope_label": None,
        "calendar_name_hint": None,
        "clean_query": query,
    }

    for hash_match in re.finditer(r"#([\w-]+)", query, re.I):
        token = hash_match.group(1).lower()
        if token in _CALENDAR_LENS_TOKENS:
            _apply_calendar_lens(result, token)
            continue
        if token in _CALENDAR_TIME_SCOPES:
            _apply_calendar_scope(result, token)
            logger.info(
                "Deprecated calendar time token #%s; prefer @%s",
                token,
                token,
            )
            continue
        logger.debug("Calendar # token not mapped: #%s", token)

    for at_match in re.finditer(r"@([\w-]+)", query, re.I):
        token = at_match.group(1).lower()
        if token in _CALENDAR_TIME_SCOPES:
            _apply_calendar_scope(result, token)
            continue
        if token in _CALENDAR_LENS_TOKENS:
            _apply_calendar_lens(result, token)
            logger.info(
                "Deprecated calendar lens token @%s; prefer #%s",
                token,
                token,
            )
            continue
        logger.debug("Calendar @ token not mapped as scope/lens: @%s", token)

    hash_long = re.search(r"#([A-Za-z][A-Za-z0-9.'\-\s]+?)(?=\s+@|\s+#|\s*$|[,.!?])", query)
    if hash_long:
        raw_hint = hash_long.group(1).strip()
        first = raw_hint.split()[0].lower()
        if first not in _CALENDAR_LENS_TOKENS and first not in _CALENDAR_TIME_SCOPES:
            result["calendar_name_hint"] = raw_hint
            logger.debug("Calendar #calendar name hint: #%s", raw_hint)

    if not result["calendar_name_hint"]:
        at_long = re.search(r"@([A-Za-z][A-Za-z0-9.'\-\s]+?)(?=\s+#|\s+@|\s*$|[,.!?])", query)
        if at_long:
            raw_hint = at_long.group(1).strip()
            first = raw_hint.split()[0].lower()
            if first not in _CALENDAR_TIME_SCOPES and first not in _CALENDAR_LENS_TOKENS:
                result["calendar_name_hint"] = raw_hint
                logger.debug(
                    "Deprecated calendar name via @%s; prefer #%s",
                    raw_hint,
                    raw_hint,
                )

    clean = query
    clean = re.sub(r"#[\w-]+", " ", clean, flags=re.I)
    clean = re.sub(r"@[\w-]+", " ", clean, flags=re.I)
    clean = re.sub(r"#[A-Za-z][A-Za-z0-9.'\-\s]+?(?=\s|$)", " ", clean)
    clean = re.sub(r"@[A-Za-z][A-Za-z0-9.'\-\s]+?(?=\s|$)", " ", clean)
    clean = re.sub(
        r"\b(?:what'?s on|show me|check|list|my calendar|calendar|events?|schedule)\b",
        " ",
        clean,
        flags=re.I,
    )
    clean = re.sub(r"\s+", " ", clean).strip(" ?.,!")
    result["clean_query"] = clean or ""
    return result


def parse_calendar_query_struct(
    query: str,
    timezone: Optional[str] = None,
    calendar_id: Optional[str] = None,
) -> CalendarQueryParams:
    """Build structured calendar params from query + @ / # tokens."""
    tz = timezone or _default_timezone()
    mentions = parse_calendar_mentions(query)
    params = CalendarQueryParams(
        original_query=query,
        clean_query=mentions.get("clean_query") or "",
        timezone=tz,
    )

    if calendar_id:
        params.calendar_ids = [calendar_id]
        logger.info("Calendar query pinned to calendar_id=%s", calendar_id[:24])

    name_hint = mentions.get("calendar_name_hint")
    if name_hint:
        params.calendar_name_hint = name_hint

    lens_token = mentions.get("lens")
    if lens_token and lens_token in _CALENDAR_AT_LENSES:
        lens_def = _CALENDAR_AT_LENSES[lens_token]
        params.lens = lens_token
        params.lens_label = lens_def.get("label")
        params.search_terms = list(lens_def.get("keywords") or [])

    scope = mentions.get("scope")
    if scope:
        params.scope = scope
        params.scope_label = mentions.get("scope_label")

    q_lower = query.lower()
    if not params.scope:
        if "today" in q_lower:
            params.scope = "today"
        elif "tomorrow" in q_lower:
            params.scope = "tomorrow"
        elif "next week" in q_lower:
            params.scope = "next_week"
        elif "this week" in q_lower or re.search(r"\bweek\b", q_lower):
            params.scope = "this_week"
        elif "month" in q_lower:
            params.scope = "this_month"

    if params.clean_query:
        clean = params.clean_query
        for phrase in (
            "today", "tomorrow", "this week", "next week", "the week", "month",
            "morning", "afternoon", "evening", "weekend",
        ):
            clean = re.sub(rf"\b{re.escape(phrase)}\b", "", clean, flags=re.I)
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean and clean not in params.search_terms:
            params.search_terms.append(clean)
        params.clean_query = clean

    if params.scope == "all_day_only":
        params.all_day_only = True
    elif params.scope == "virtual_only":
        params.virtual_only = True
    elif params.scope == "in_person_only":
        params.in_person_only = True
    elif params.scope == "has_attendees":
        params.has_attendees_only = True

    logger.info(
        "Calendar query struct: lens=%r scope=%r terms=%r clean=%r",
        params.lens,
        params.scope,
        params.search_terms,
        params.clean_query,
    )
    return params


def build_time_range(
    params: CalendarQueryParams,
) -> Tuple[datetime, datetime]:
    """Resolve #scope (and defaults) to API timeMin/timeMax in user TZ."""
    tz = _get_tz(params.timezone)
    now = datetime.now(tz)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_today = start_of_today + timedelta(days=1)

    scope = params.scope or "this_week"
    logger.debug("Building calendar time range for scope=%s", scope)

    if scope == "today":
        return start_of_today, end_of_today

    if scope == "tomorrow":
        t0 = end_of_today
        return t0, t0 + timedelta(days=1)

    if scope == "this_week":
        week_start = start_of_today - timedelta(days=start_of_today.weekday())
        return week_start, week_start + timedelta(days=7)

    if scope == "next_week":
        week_start = start_of_today - timedelta(days=start_of_today.weekday()) + timedelta(days=7)
        return week_start, week_start + timedelta(days=7)

    if scope == "this_month":
        month_start = start_of_today.replace(day=1)
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1)
        return month_start, month_end

    if scope == "morning_today":
        return datetime.combine(start_of_today.date(), time(6, 0), tz), datetime.combine(
            start_of_today.date(), time(12, 0), tz
        )

    if scope == "afternoon_today":
        return datetime.combine(start_of_today.date(), time(12, 0), tz), datetime.combine(
            start_of_today.date(), time(18, 0), tz
        )

    if scope == "evening_today":
        return datetime.combine(start_of_today.date(), time(18, 0), tz), end_of_today

    if scope == "weekend":
        days_until_sat = (5 - start_of_today.weekday()) % 7
        sat = start_of_today + timedelta(days=days_until_sat)
        return sat, sat + timedelta(days=2)

    if scope == "soon_48h":
        return now, now + timedelta(hours=48)

    if scope == "agenda":
        return start_of_today, end_of_today + timedelta(days=1)

    if scope == "pipeline_14d":
        return start_of_today, start_of_today + timedelta(days=14)

    if scope == "ytd_forward":
        year_start = datetime(now.year, 1, 1, tzinfo=tz)
        return year_start, start_of_today + timedelta(days=365)

    # default: next 7 days from now
    return now, now + timedelta(days=7)


def _build_free_text_q(params: CalendarQueryParams) -> Optional[str]:
    terms = [t for t in params.search_terms if t and len(t) > 1]
    if not terms:
        return params.clean_query or None
    return " OR ".join(terms[:6])


def _list_user_calendars(client: httpx.Client, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    resp = client.get(
        CALENDAR_LIST_URL,
        headers=headers,
        params={"minAccessRole": "reader", "showHidden": "true"},
    )
    if resp.status_code != 200:
        logger.warning(
            "Calendar list API failed status=%s body=%s",
            resp.status_code,
            resp.text[:300],
        )
        resp.raise_for_status()
    items = resp.json().get("items") or []
    logger.info("Calendar list: %d calendars visible", len(items))
    return items


def _resolve_calendar_ids(
    all_calendars: List[Dict[str, Any]],
    params: CalendarQueryParams,
) -> List[str]:
    """Pick calendar IDs for explicit @-mention, @lens, or name hint; fall back to primary."""
    if params.calendar_ids:
        valid = {c["id"] for c in all_calendars}
        picked = [cid for cid in params.calendar_ids if cid in valid]
        if picked:
            logger.info("Using explicit calendar_id(s): %s", [p[:16] for p in picked])
            return picked[:5]
        logger.warning(
            "Explicit calendar_id not in user list — falling back to primary"
        )

    if params.calendar_name_hint:
        hint = params.calendar_name_hint.lower()
        for cal in all_calendars:
            summary = (cal.get("summary") or "").lower()
            if hint == summary or hint in summary:
                logger.info(
                    "Matched calendar by @ name hint %r -> %r",
                    params.calendar_name_hint,
                    cal.get("summary"),
                )
                return [cal["id"]]
        logger.info(
            "No calendar name match for @%r — using primary",
            params.calendar_name_hint,
        )

    if not params.lens or params.lens not in _CALENDAR_AT_LENSES:
        primary = [c["id"] for c in all_calendars if c.get("primary")]
        return primary or ["primary"]

    patterns = _CALENDAR_AT_LENSES[params.lens].get("calendar_patterns") or []
    matched: List[str] = []
    for cal in all_calendars:
        summary = (cal.get("summary") or "").lower()
        if any(re.search(pat, summary, re.I) for pat in patterns):
            matched.append(cal["id"])
            logger.debug("Lens %s matched calendar %r", params.lens, cal.get("summary"))

    if matched:
        return matched[:5]
    primary = [c["id"] for c in all_calendars if c.get("primary")]
    logger.info("No named calendar for @%s — using primary", params.lens)
    return primary or ["primary"]


def _event_passes_filters(ev: Dict[str, Any], params: CalendarQueryParams) -> bool:
    if params.all_day_only:
        start = ev.get("start") or {}
        if "dateTime" in start:
            return False
    if params.has_attendees_only:
        attendees = ev.get("attendees") or []
        if len(attendees) < 2:
            return False
    if params.virtual_only:
        loc = (ev.get("location") or "").lower()
        hang = (ev.get("hangoutLink") or "").lower()
        if not hang and "zoom" not in loc and "meet.google" not in loc and "teams" not in loc:
            return False
    if params.in_person_only:
        loc = (ev.get("location") or "").strip()
        hang = ev.get("hangoutLink")
        if not loc or hang:
            return False
    return True


def _format_event(ev: Dict[str, Any], calendar_name: str = "") -> str:
    title = ev.get("summary", "No Title")
    start_raw = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date", "")
    end_raw = ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date", "")
    location = ev.get("location") or ""
    description = (ev.get("description") or "")[:200]
    cal_tag = f" [{calendar_name}]" if calendar_name else ""

    start_disp, end_disp = start_raw, end_raw
    try:
        if "T" in start_raw:
            start_disp = datetime.fromisoformat(start_raw.replace("Z", "+00:00")).strftime("%b %d, %Y at %I:%M %p")
        if "T" in end_raw:
            end_disp = datetime.fromisoformat(end_raw.replace("Z", "+00:00")).strftime("%b %d, %Y at %I:%M %p")
    except Exception:
        pass

    loc_str = f"\n  **Location**: {location}" if location else ""
    desc_str = f"\n  **Notes**: {description}..." if description else ""
    return (
        f"- **Event**: {title}{cal_tag}\n"
        f"  **Time**: {start_disp} to {end_disp}{loc_str}{desc_str}\n"
    )


def _fetch_events_for_calendar(
    client: httpx.Client,
    headers: Dict[str, str],
    calendar_id: str,
    time_min: datetime,
    time_max: datetime,
    free_text_q: Optional[str],
) -> List[Dict[str, Any]]:
    url = CALENDAR_EVENTS_URL.format(calendar_id=calendar_id)
    api_params: Dict[str, Any] = {
        "maxResults": _MAX_RESULTS,
        "singleEvents": "true",
        "orderBy": "startTime",
        "timeMin": time_min.isoformat(),
        "timeMax": time_max.isoformat(),
    }
    if free_text_q:
        api_params["q"] = free_text_q

    resp = client.get(url, headers=headers, params=api_params)
    resp.raise_for_status()
    return resp.json().get("items") or []


def list_user_calendars_for_autocomplete(
    token: str,
    timeout: float = 15.0,
) -> List[Dict[str, Any]]:
    """Fetch the user's calendars for @-mention autocomplete (Drive-style)."""
    headers = {"Authorization": f"Bearer {token}"}
    items: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=timeout) as client:
            calendars = _list_user_calendars(client, headers)
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Calendar list for autocomplete HTTP error status=%s body=%s",
            exc.response.status_code,
            exc.response.text[:300],
        )
        return items
    except Exception as exc:
        logger.warning("Calendar list for autocomplete failed: %s", exc, exc_info=True)
        return items

    for cal in calendars:
        cal_id = cal.get("id") or ""
        summary = (cal.get("summary") or "Unnamed Calendar").strip()
        if not cal_id or not summary:
            continue
        access = (cal.get("accessRole") or "reader").replace("_", " ")
        meta_parts = [access.capitalize()]
        if cal.get("primary"):
            meta_parts.insert(0, "Primary")
        if cal.get("backgroundColor"):
            meta_parts.append(cal["backgroundColor"])
        items.append({
            "display": f"@{summary}",
            "value": f"@{summary} ",
            "calendar_id": cal_id,
            "icon": "📅" if cal.get("primary") else "🗓️",
            "category": "Google Calendars",
            "meta": " · ".join(meta_parts[:2]),
        })

    items.sort(key=lambda x: (0 if "Primary" in (x.get("meta") or "") else 1, x.get("display", "")))
    logger.info("Built %d calendar autocomplete items", len(items))
    return items


def run_calendar_search(
    token: str,
    query: str,
    timeout: float,
    timezone: Optional[str] = None,
    calendar_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Main calendar MCP entry with @ lenses and # scopes."""
    params = parse_calendar_query_struct(query, timezone=timezone, calendar_id=calendar_id)
    time_min, time_max = build_time_range(params)
    free_text_q = _build_free_text_q(params)
    headers = {"Authorization": f"Bearer {token}"}

    lens_hdr = f"@ {params.lens_label}" if params.lens_label else "all calendars"
    scope_hdr = params.scope_label or params.scope or "this week"
    logger.info(
        "run_calendar_search: lens=%s scope=%s timeMin=%s timeMax=%s q=%r",
        lens_hdr,
        scope_hdr,
        time_min.isoformat(),
        time_max.isoformat(),
        free_text_q,
    )

    with httpx.Client(timeout=timeout) as client:
        all_cals = _list_user_calendars(client, headers)
        cal_name_by_id = {c["id"]: c.get("summary", "Calendar") for c in all_cals}
        calendar_ids = _resolve_calendar_ids(all_cals, params)
        if len(calendar_ids) == 1:
            single_name = cal_name_by_id.get(calendar_ids[0])
            if single_name:
                lens_hdr = single_name

        all_events: List[Tuple[Dict[str, Any], str]] = []
        for cal_id in calendar_ids:
            try:
                events = _fetch_events_for_calendar(
                    client, headers, cal_id, time_min, time_max, free_text_q
                )
                cal_name = cal_name_by_id.get(cal_id, "")
                for ev in events:
                    if _event_passes_filters(ev, params):
                        all_events.append((ev, cal_name))
                logger.info("Calendar %s: %d events after filters", cal_id[:12], len(events))
            except Exception as exc:
                logger.warning("Failed to fetch calendar %s: %s", cal_id, exc)

    all_events.sort(
        key=lambda pair: pair[0].get("start", {}).get("dateTime")
        or pair[0].get("start", {}).get("date")
        or ""
    )

    if not all_events:
        scope_desc = f"#{scope_hdr}" if params.scope_label else scope_hdr
        lens_desc = f"@{params.lens}" if params.lens else "your calendars"
        msg = f"No events found for {lens_desc} ({scope_desc})."
        if free_text_q:
            msg += f" Search: `{free_text_q}`."
        return {"content": [{"type": "text", "text": msg}]}

    header = (
        f"Calendar ({lens_hdr} · #{scope_hdr}):\n\n"
        if params.lens or params.scope_label
        else "Upcoming Calendar events:\n\n"
    )
    summaries = [_format_event(ev, cal_name) for ev, cal_name in all_events[:_MAX_RESULTS]]
    text = header + "\n".join(summaries)
    logger.info("Calendar search complete: %d events", len(summaries))
    return {
        "content": [{"type": "text", "text": text}],
        "calendar_meta": {
            "lens": params.lens,
            "scope": params.scope,
            "event_count": len(summaries),
        },
    }


def prepare_calendar_call_params(
    query: str,
    timezone: Optional[str] = None,
    calendar_id: Optional[str] = None,
    calendar_name: Optional[str] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "query": query,
        "timezone": timezone or _default_timezone(),
    }
    if calendar_id:
        params["calendar_id"] = calendar_id
    if calendar_name:
        params["calendar_name"] = calendar_name
    return params


def get_calendar_autocomplete_items() -> Dict[str, List[Dict[str, str]]]:
    """Static @ / # suggestions for UI autocomplete."""
    at_items = [
        {"display": f"@{key}", "value": f"@{key} ", "icon": "🎯", "meta": spec.get("label", key)}
        for key, spec in _CALENDAR_AT_LENSES.items()
        if key == key.replace(" ", "") and key in (
            "tasks",
            "birthdays",
            "holidays",
            "family",
            "work",
            "health",
            "travel",
            "focus",
            "kids",
            "social",
        )
    ]
    hash_labels = {
        "today": "Today",
        "tomorrow": "Tomorrow",
        "week": "This week",
        "weekend": "This weekend",
        "morning": "This morning",
        "afternoon": "This afternoon",
        "evening": "Tonight",
        "agenda": "Today + tomorrow (agenda)",
        "pipeline": "Next 14 days",
        "soon": "Next 48 hours",
        "month": "This month",
        "allday": "All-day events only",
        "busy": "Meetings with attendees",
        "virtual": "Video calls only",
    }
    hash_items = [
        {
            "display": f"#{key}",
            "value": f"#{key} ",
            "icon": "🕐",
            "meta": hash_labels.get(key, key),
        }
        for key in hash_labels
    ]
    return {"at_lenses": at_items, "hash_scopes": hash_items}
