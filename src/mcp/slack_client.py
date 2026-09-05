"""
Slack MCP Client — OAuth 2.0 + slack_sdk WebClient.

Handles:
  - OAuth 2.0 flow (user token / xoxp-) stored per-user in Redis
  - Token storage/retrieval (tokens don't expire but we keep a TTL for rotation)
  - Reading channels, DMs, and messages on behalf of the signed-in user

Services supported (read-only):
  - Public channels  → conversations.list + conversations.history
  - Private channels → conversations.list + conversations.history
  - Direct messages  → conversations.list(types=im) + conversations.history
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLACK_AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
SLACK_TOKEN_URL = "https://slack.com/api/oauth.v2.access"
SLACK_API_BASE = "https://slack.com/api"

# User token scopes needed for reading channels + DMs
_USER_SCOPES: List[str] = [
    "channels:read",
    "channels:history",
    "groups:read",
    "groups:history",
    "im:read",
    "im:history",
    "mpim:read",
    "mpim:history",
    "users:read",
    "search:read",
]

# Token TTL in Redis (30 days — user tokens don't expire but we rotate)
_TOKEN_TTL_SECONDS = int(os.getenv("SLACK_MCP_TOKEN_TTL", str(30 * 24 * 3600)))


# ---------------------------------------------------------------------------
# SlackMCPClient
# ---------------------------------------------------------------------------


class SlackMCPClient:
    """
    Lightweight Slack MCP client.

    Stores one OAuth user token per user_id in Redis.
    All API calls are synchronous (called from ThreadPoolExecutor in server).
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        redis_client,  # redis.Redis instance from existing connection
        redis_key_prefix: str = "synapse",
        token_ttl_seconds: int = _TOKEN_TTL_SECONDS,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._redis = redis_client
        self._key_prefix = redis_key_prefix
        self._token_ttl = token_ttl_seconds

        logger.info(
            "SlackMCPClient initialized (redirect_uri=%s)", self._redirect_uri
        )

    # -----------------------------------------------------------------------
    # OAuth helpers
    # -----------------------------------------------------------------------

    def get_auth_url(self, user_id: str) -> str:
        """
        Generate a Slack OAuth 2.0 authorization URL (user token flow).

        The `state` encodes user_id + a random nonce so the callback can
        store the token under the correct Redis key and prevent CSRF.
        """
        nonce = secrets.token_urlsafe(16)
        state_payload = json.dumps({"user_id": user_id, "nonce": nonce})
        state = urllib.parse.quote(state_payload)

        params = {
            "client_id": self._client_id,
            "user_scope": ",".join(_USER_SCOPES),
            "redirect_uri": self._redirect_uri,
            "state": state,
        }
        query_string = urllib.parse.urlencode(params)
        auth_url = f"{SLACK_AUTHORIZE_URL}?{query_string}"
        logger.info("Generated Slack auth URL for user_id=%s", user_id)
        return auth_url

    def exchange_code(self, code: str, state: str) -> Dict[str, Any]:
        """
        Exchange an authorization code for a user token and persist to Redis.

        Returns dict with user_id and token metadata.
        """
        try:
            state_payload = json.loads(urllib.parse.unquote(state))
            user_id: str = state_payload["user_id"]
        except Exception as exc:
            raise ValueError(f"Invalid Slack OAuth state parameter: {exc}") from exc

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                SLACK_TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": self._redirect_uri,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if not data.get("ok"):
            error = data.get("error", "unknown_error")
            raise ValueError(f"Slack OAuth token exchange failed: {error}")

        # Extract the *user* token from authed_user block
        authed_user = data.get("authed_user", {})
        user_token = authed_user.get("access_token")
        if not user_token:
            raise ValueError(
                "Slack OAuth response missing authed_user.access_token. "
                "Ensure your app has User Token Scopes configured."
            )

        token_data = {
            "access_token": user_token,
            "token_type": authed_user.get("token_type", "user"),
            "scope": authed_user.get("scope", ""),
            "slack_user_id": authed_user.get("id", ""),
            "web_user_id": user_id,
            "team_id": data.get("team", {}).get("id", ""),
            "team_name": data.get("team", {}).get("name", ""),
            "stored_at": time.time(),
        }

        redis_key = self._token_key(user_id)
        self._redis.setex(redis_key, self._token_ttl, json.dumps(token_data))
        
        slack_user_id = token_data["slack_user_id"]
        if slack_user_id:
            slack_redis_key = self._token_key(slack_user_id)
            self._redis.setex(slack_redis_key, self._token_ttl, json.dumps(token_data))
            
        logger.info(
            "Stored Slack user token for user_id=%s, slack_user_id=%s, team=%s",
            user_id,
            slack_user_id,
            token_data["team_name"],
        )

        return {
            "user_id": user_id,
            "team_name": token_data["team_name"],
            "slack_user_id": token_data["slack_user_id"],
            "scopes": token_data["scope"],
            "access_token": user_token,
        }

    # -----------------------------------------------------------------------
    # Token management
    # -----------------------------------------------------------------------

    def _token_key(self, user_id: str) -> str:
        return f"{self._key_prefix}:mcp:slack:token:{user_id}"

    def _load_token(self, user_id: str) -> Optional[str]:
        """Load the per-user OAuth access token from Redis. No shared fallback tokens."""
        if not user_id:
            logger.debug("Slack _load_token called without user_id")
            return None
        raw = self._redis.get(self._token_key(user_id))
        if not raw:
            logger.debug("No Slack token in Redis for user_id=%s", user_id)
            return None
        try:
            data = json.loads(raw)
            token = data.get("access_token")
            if not token:
                logger.warning("Slack token record missing access_token for user_id=%s", user_id)
            return token
        except Exception as exc:
            logger.warning("Failed to parse Slack token for user_id=%s: %s", user_id, exc)
            return None

    def _load_token_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load the full token record from Redis."""
        raw = self._redis.get(self._token_key(user_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def is_authenticated(self, user_id: str) -> bool:
        """Return True only when this user has completed Slack OAuth (token in Redis)."""
        if not user_id:
            return False
        return self._load_token(user_id) is not None

    def revoke_tokens(self, user_id: str) -> None:
        """Delete Slack token for user from Redis."""
        token_data = self._load_token_data(user_id)
        if token_data:
            slack_user_id = token_data.get("slack_user_id")
            web_user_id = token_data.get("web_user_id")
            if slack_user_id:
                self._redis.delete(self._token_key(slack_user_id))
            if web_user_id:
                self._redis.delete(self._token_key(web_user_id))
        self._redis.delete(self._token_key(user_id))
        logger.info("Revoked Slack token for user_id=%s", user_id)

    def auth_status(self, user_id: str) -> Dict[str, Any]:
        """Return authentication status and metadata for the user."""
        if not user_id:
            return {"connected": False}
        data = self._load_token_data(user_id)
        if not data:
            return {"connected": False}
        return {
            "connected": True,
            "team_name": data.get("team_name", ""),
            "slack_user_id": data.get("slack_user_id", ""),
            "scopes": data.get("scope", ""),
            "stored_at": data.get("stored_at"),
        }

    # -----------------------------------------------------------------------
    # Slack API helpers
    # -----------------------------------------------------------------------

    async def _slack_get(
        self,
        token: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Make an authenticated GET request to the Slack Web API."""
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        url = f"{SLACK_API_BASE}/{endpoint}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers, params=params or {})
            resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            error = data.get("error", "unknown_error")
            raise RuntimeError(f"Slack API error ({endpoint}): {error}")
        return data

    async def _resolve_user_name(self, token: str, user_id: str) -> str:
        """Resolve a Slack user ID to a display name."""
        try:
            data = await self._slack_get(token, "users.info", {"user": user_id})
            user = data.get("user", {})
            profile = user.get("profile", {})
            return profile.get("display_name") or profile.get("real_name") or user_id
        except Exception:
            return user_id

    async def _resolve_channel_id_by_name(
        self,
        token: str,
        channel_name: str,
        timeout: float = 15.0,
    ) -> Optional[str]:
        """
        Resolve a channel or DM name to a Slack conversation ID.

        Uses conversations.list and, for DMs, resolves the associated user.
        """
        clean_name = channel_name.lstrip("#@").strip().lower().rstrip(".,!?")
        if clean_name.startswith("dm with "):
            clean_name = clean_name[8:].strip()
        elif clean_name.startswith("dm "):
            clean_name = clean_name[3:].strip()

        if not clean_name:
            return None

        try:
            data = await self._slack_get(
                token,
                "conversations.list",
                {
                    "types": "public_channel,private_channel,im,mpim",
                    "limit": 1000,
                    "exclude_archived": "true",
                },
                timeout=timeout,
            )
            channels = data.get("channels", [])
            cursor = data.get("response_metadata", {}).get("next_cursor")
            while cursor:
                data = await self._slack_get(
                    token,
                    "conversations.list",
                    {
                        "types": "public_channel,private_channel,im,mpim",
                        "limit": 1000,
                        "exclude_archived": "true",
                        "cursor": cursor,
                    },
                    timeout=timeout,
                )
                channels.extend(data.get("channels", []))
                cursor = data.get("response_metadata", {}).get("next_cursor")

            # Match public/private channels by name first
            for ch in channels:
                if ch.get("is_im") or ch.get("is_mpim"):
                    continue
                if (ch.get("name") or "").lower() == clean_name:
                    logger.info("Resolved Slack channel name %r -> %s", channel_name, ch["id"])
                    return ch["id"]

            # Match by channel id if provided
            for ch in channels:
                if ch["id"].lower() == clean_name:
                    logger.info("Resolved Slack channel id %r -> %s", channel_name, ch["id"])
                    return ch["id"]

            # Attempt DM matching by user name (exact, then substring)
            dm_user_ids = [ch["user"] for ch in channels if ch.get("is_im") and ch.get("user")]
            if dm_user_ids:
                import asyncio
                resolved = await asyncio.gather(
                    *(self._resolve_user_name(token, uid) for uid in dm_user_ids)
                )
                user_map = dict(zip(dm_user_ids, resolved))

                for ch in channels:
                    if not ch.get("is_im") or not ch.get("user"):
                        continue
                    other_user_id = ch["user"]
                    name = user_map.get(other_user_id, "")
                    name_lower = name.lower()
                    if name_lower == clean_name or other_user_id.lower() == clean_name:
                        logger.info("Resolved Slack DM user %r -> %s", channel_name, ch["id"])
                        return ch["id"]

                for ch in channels:
                    if not ch.get("is_im") or not ch.get("user"):
                        continue
                    other_user_id = ch["user"]
                    name = user_map.get(other_user_id, "")
                    name_lower = name.lower()
                    if clean_name in name_lower or name_lower in clean_name:
                        logger.info("Resolved Slack DM user (partial) %r -> %s via %r", channel_name, ch["id"], name)
                        return ch["id"]

            logger.warning("Could not resolve Slack channel/DM name %r", channel_name)
            return None
        except Exception as exc:
            logger.warning("Failed resolving Slack channel/DM %r: %s", channel_name, exc)
            return None

    # -----------------------------------------------------------------------
    # Public read operations
    # -----------------------------------------------------------------------

    async def list_channels(
        self,
        user_id: str,
        query: str = "",
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        List public + private channels + DMs the user is a member of.
        Optionally filter by query string (channel name matching).
        """
        token = self._load_token(user_id)
        if not token:
            raise PermissionError(f"No Slack token for user_id={user_id}. Please sign in first.")

        logger.info("Slack list_channels for user_id=%s query=%r", user_id, query)

        data = await self._slack_get(
            token,
            "conversations.list",
            {
                "types": "public_channel,private_channel,im,mpim",
                "limit": 1000,
                "exclude_archived": "true",
            },
            timeout=timeout,
        )

        channels = data.get("channels", [])
        cursor = data.get("response_metadata", {}).get("next_cursor")
        while cursor:
            data = await self._slack_get(
                token,
                "conversations.list",
                {
                    "types": "public_channel,private_channel,im,mpim",
                    "limit": 1000,
                    "exclude_archived": "true",
                    "cursor": cursor
                },
                timeout=timeout,
            )
            channels.extend(data.get("channels", []))
            cursor = data.get("response_metadata", {}).get("next_cursor")
        q_lower = query.lower().strip()

        result_items = []
        for ch in channels:
            name = ch.get("name") or ch.get("user", "")
            is_dm = ch.get("is_im", False) or ch.get("is_mpim", False)
            ch_type = "DM" if is_dm else ("Private" if ch.get("is_private") else "Public")

            if q_lower and q_lower not in name.lower():
                continue

            result_items.append(
                f"- **{'DM' if is_dm else '#' + name}** ({ch_type}) — ID: {ch['id']}"
            )

        if not result_items:
            text = f"No Slack channels found matching '{query}'." if query else "No Slack channels found."
        else:
            header = f"Your Slack channels{' matching ' + repr(query) if query else ''}:\n\n"
            text = header + "\n".join(result_items)

        return {"content": [{"type": "text", "text": text}]}

    async def _list_dms(
        self,
        user_id: str,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        List DM conversations the user has access to.
        """
        token = self._load_token(user_id)
        if not token:
            raise PermissionError(f"No Slack token for user_id={user_id}. Please sign in first.")

        logger.info("Slack _list_dms for user_id=%s", user_id)

        data = await self._slack_get(
            token,
            "conversations.list",
            {
                "types": "im,mpim",
                "limit": 1000,
            },
            timeout=timeout,
        )

        channels = data.get("channels", [])
        cursor = data.get("response_metadata", {}).get("next_cursor")
        while cursor:
            data = await self._slack_get(
                token,
                "conversations.list",
                {
                    "types": "im,mpim",
                    "limit": 1000,
                    "cursor": cursor
                },
                timeout=timeout,
            )
            channels.extend(data.get("channels", []))
            cursor = data.get("response_metadata", {}).get("next_cursor")
        # Collect IM user IDs to resolve in parallel
        im_user_ids = []
        for ch in channels:
            if ch.get("is_im", False) and ch.get("user"):
                im_user_ids.append(ch["user"])

        unique_uids = list(set(im_user_ids))
        user_names = {}
        if unique_uids:
            import asyncio
            resolved = await asyncio.gather(*(self._resolve_user_name(token, uid) for uid in unique_uids))
            user_names = dict(zip(unique_uids, resolved))

        result_items = []
        for ch in channels:
            is_im = ch.get("is_im", False)
            if is_im:
                other_user_id = ch.get("user", "")
                name = user_names.get(other_user_id, other_user_id)
                ch_label = f"DM with {name}"
            else:
                ch_label = f"Group DM (ID: {ch['id']})"
            
            result_items.append(f"- **{ch_label}** — ID: {ch['id']}")

        if not result_items:
            text = "No direct messages found."
        else:
            text = "Your Slack direct messages:\n\n" + "\n".join(result_items)

        return {"content": [{"type": "text", "text": text}]}

    async def search_messages(
        self,
        user_id: str,
        query: str,
        limit: int = 20,
        timeout: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Search Slack messages across all channels and DMs the user can access.

        Uses conversations.list + conversations.history to find recent relevant messages.
        Falls back to search.messages if the user has the search:read scope.
        """
        token = self._load_token(user_id)
        if not token:
            raise PermissionError(f"No Slack token for user_id={user_id}. Please sign in first.")

        logger.info("Slack search_messages for user_id=%s query=%r", user_id, query)

        # Try search.messages first (requires search:read scope — better quality)
        try:
            return await self._search_via_search_api(token, query, timeout)
        except RuntimeError as exc:
            if "missing_scope" in str(exc) or "not_allowed" in str(exc):
                logger.info("search:read scope missing, falling back to history scan")
            else:
                logger.warning("search.messages failed (%s), falling back to history scan", exc)

        # Fallback: scan recent history of the most recent channels
        return await self._search_via_history(token, query, limit, timeout)

    async def _search_via_search_api(
        self, token: str, query: str, timeout: float
    ) -> Dict[str, Any]:
        """Use Slack's search.messages API (requires search:read scope)."""
        data = await self._slack_get(
            token,
            "search.messages",
            {"query": query, "count": 10, "sort": "timestamp", "sort_dir": "desc"},
            timeout=timeout,
        )
        matches = data.get("messages", {}).get("matches", [])
        if not matches:
            return {"content": [{"type": "text", "text": f"No Slack messages found matching '{query}'."}]}

        summaries = []
        for m in matches:
            channel_name = m.get("channel", {}).get("name", "unknown-channel")
            username = m.get("username") or m.get("user", "Unknown")
            ts_raw = m.get("ts", "")
            text = m.get("text", "")
            permalink = m.get("permalink", "")

            try:
                ts_human = _format_slack_ts(ts_raw)
            except Exception:
                ts_human = ts_raw

            summaries.append(
                f"- **#{channel_name}** — {username} ({ts_human})\n"
                f"  \"{text[:200]}{'...' if len(text) > 200 else ''}\"\n"
                f"  {permalink}"
            )

        result_text = f"Slack messages matching '{query}':\n\n" + "\n\n".join(summaries)
        return {"content": [{"type": "text", "text": result_text}]}

    async def _search_via_history(
        self, token: str, query: str, limit: int, timeout: float
    ) -> Dict[str, Any]:
        """
        Fallback: list recent channels and scan their history for the query.
        Returns messages containing the query text from the most active channels.
        """
        # Fetch channels
        try:
            ch_data = await self._slack_get(
                token,
                "conversations.list",
                {
                    "types": "public_channel,private_channel,im,mpim",
                    "limit": 200,
                    "exclude_archived": "true",
                },
                timeout=timeout,
            )
            channels = ch_data.get("channels", [])
            cursor = ch_data.get("response_metadata", {}).get("next_cursor")
            while cursor and len(channels) < 500:  # Cap at 500 channels for search
                ch_data = await self._slack_get(
                    token,
                    "conversations.list",
                    {
                        "types": "public_channel,private_channel,im,mpim",
                        "limit": 200,
                        "exclude_archived": "true",
                        "cursor": cursor
                    },
                    timeout=timeout,
                )
                channels.extend(ch_data.get("channels", []))
                cursor = ch_data.get("response_metadata", {}).get("next_cursor")
        except Exception as exc:
            raise RuntimeError(f"Failed to list Slack channels: {exc}") from exc

        q_lower = query.lower().strip()
        summaries = []

        matched_items = []
        for ch in channels[:10]:  # Check up to 10 channels
            ch_id = ch["id"]
            ch_name = ch.get("name") or "DM"
            try:
                hist = await self._slack_get(
                    token,
                    "conversations.history",
                    {"channel": ch_id, "limit": 30},
                    timeout=timeout,
                )
                messages = hist.get("messages", [])
                for msg in messages:
                    text = msg.get("text", "")
                    if not q_lower or q_lower in text.lower():
                        matched_items.append({
                            "ch_label": "DM" if ch.get("is_im", False) else f"#{ch_name}",
                            "user": msg.get("user", "Unknown"),
                            "ts": msg.get("ts", ""),
                            "text": text
                        })
                        if len(matched_items) >= limit:
                            break
            except Exception as exc:
                logger.debug("Skipping channel %s during history scan: %s", ch_id, exc)
                continue

            if len(matched_items) >= limit:
                break

        if not matched_items:
            filter_note = f" matching '{query}'" if q_lower else ""
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"No Slack messages found{filter_note}.",
                    }
                ]
            }

        # Resolve user names in parallel
        unique_uids = {item["user"] for item in matched_items if item["user"] != "Unknown"}
        user_names = {}
        if unique_uids:
            import asyncio
            resolved = await asyncio.gather(*(self._resolve_user_name(token, uid) for uid in unique_uids))
            user_names = dict(zip(unique_uids, resolved))

        for item in matched_items:
            user_label = user_names.get(item["user"], item["user"])
            try:
                ts_human = _format_slack_ts(item["ts"])
            except Exception:
                ts_human = item["ts"]
            summaries.append(
                f"- **{item['ch_label']}** — {user_label} ({ts_human})\n"
                f"  \"{item['text'][:200]}{'...' if len(item['text']) > 200 else ''}\""
            )

        header = f"Recent Slack messages{' matching ' + repr(query) if q_lower else ''}:\n\n"
        result_text = header + "\n\n".join(summaries)
        return {"content": [{"type": "text", "text": result_text}]}

    async def get_channel_history(
        self,
        user_id: str,
        channel_name: str,
        limit: int = 20,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Read recent messages from a specific channel by name.
        """
        token = self._load_token(user_id)
        if not token:
            raise PermissionError(f"No Slack token for user_id={user_id}.")

        logger.info(
            "Slack get_channel_history user_id=%s channel=%s", user_id, channel_name
        )

        clean_name = channel_name.lstrip("#").strip()
        if clean_name.lower() in ("current", "this", ""):
            return await self._search_via_history(token, query="", limit=limit, timeout=timeout)

        channel_id = await self._resolve_channel_id_by_name(token, clean_name, timeout=timeout)
        if not channel_id:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Could not locate a channel or DM named '{channel_name}'.",
                    }
                ]
            }

        display_label = channel_name if channel_name.startswith("#") else f"#{clean_name}"
        if channel_id.startswith("D"):
            try:
                im_data = await self._slack_get(
                    token,
                    "conversations.info",
                    {"channel": channel_id},
                    timeout=timeout,
                )
                dm_user = im_data.get("channel", {}).get("user")
                if dm_user:
                    partner_name = await self._resolve_user_name(token, dm_user)
                    display_label = f"DM with {partner_name}"
            except Exception as exc:
                logger.warning("Failed to resolve DM partner label for %s: %s", channel_id, exc)
                display_label = f"DM with {clean_name}"

        history_data = await self._slack_get(
            token,
            "conversations.history",
            {"channel": channel_id, "limit": limit},
            timeout=timeout,
        )
        messages = history_data.get("messages", [])

        if not messages:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"No recent messages found for channel '{channel_name}'.",
                    }
                ]
            }

        summaries = []
        unique_uids = {msg.get("user") for msg in messages if msg.get("user")}
        user_names = {}
        if unique_uids:
            import asyncio
            resolved = await asyncio.gather(*(self._resolve_user_name(token, uid) for uid in unique_uids))
            user_names = dict(zip(unique_uids, resolved))

        for msg in messages[:limit]:
            text = msg.get("text", "")
            user_id = msg.get("user")
            user_label = user_names.get(user_id, user_id or "Unknown")
            ts = msg.get("ts", "")
            try:
                ts_human = _format_slack_ts(ts)
            except Exception:
                ts_human = ts
            summaries.append(
                f"- **{user_label}** ({ts_human})\n"
                f"  \"{text[:200]}{'...' if len(text) > 200 else ''}\""
            )

        header = f"Recent messages in {display_label}:\n\n"
        return {"content": [{"type": "text", "text": header + "\n\n".join(summaries)}]}

    async def get_channel_members(
        self,
        user_id: str,
        channel_name: str,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        List the members of a specific channel.
        """
        token = self._load_token(user_id)
        if not token:
            raise PermissionError(f"No Slack token for user_id={user_id}.")

        logger.info(
            "Slack get_channel_members user_id=%s channel=%s", user_id, channel_name
        )

        import re
        clean_name = channel_name.lstrip("#").strip()
        if clean_name.lower() in ("current", "this", ""):
            return {"content": [{"type": "text", "text": "Please specify the exact name of the channel you want to see the members of."}]}
        # List channels to resolve name or ID
        ch_data = await self._slack_get(
            token,
            "conversations.list",
            {
                "types": "public_channel,private_channel,im,mpim",
                "limit": 1000,
                "exclude_archived": "true",
            },
            timeout=timeout,
        )
        channels = ch_data.get("channels", [])
        cursor = ch_data.get("response_metadata", {}).get("next_cursor")
        while cursor:
            ch_data = await self._slack_get(
                token,
                "conversations.list",
                {
                    "types": "public_channel,private_channel,im,mpim",
                    "limit": 1000,
                    "exclude_archived": "true",
                    "cursor": cursor
                },
                timeout=timeout,
            )
            channels.extend(ch_data.get("channels", []))
            cursor = ch_data.get("response_metadata", {}).get("next_cursor")

        clean_name_lower = clean_name.lower()
        channel_id = None

        # 1. Match by channel name
        for ch in channels:
            if (ch.get("name") or "").lower() == clean_name_lower:
                channel_id = ch["id"]
                break

        # 2. Match by channel ID
        if not channel_id:
            for ch in channels:
                if ch["id"].lower() == clean_name_lower:
                    channel_id = ch["id"]
                    break

        # 3. Match by DM user name (real name or display name or ID)
        if not channel_id:
            target_name = clean_name_lower
            if target_name.startswith("dm with "):
                target_name = target_name[8:].strip()
            elif target_name.startswith("dm "):
                target_name = target_name[3:].strip()

            dm_user_ids = []
            for ch in channels:
                if ch.get("is_im") and ch.get("user"):
                    dm_user_ids.append(ch["user"])

            unique_uids = list(set(dm_user_ids))
            user_names = {}
            if unique_uids:
                import asyncio
                resolved = await asyncio.gather(*(self._resolve_user_name(token, uid) for uid in unique_uids))
                user_names = dict(zip(unique_uids, resolved))

            # Try exact match first
            for ch in channels:
                if ch.get("is_im") and ch.get("user"):
                    other_user_id = ch["user"]
                    name = user_names.get(other_user_id, "")
                    if name.lower() == target_name or other_user_id.lower() == target_name:
                        channel_id = ch["id"]
                        break
            # Try substring match fallback
            if not channel_id:
                for ch in channels:
                    if ch.get("is_im") and ch.get("user"):
                        other_user_id = ch["user"]
                        name = user_names.get(other_user_id, "")
                        if target_name in name.lower():
                            channel_id = ch["id"]
                            break

        # 4. Regex fallback
        if not channel_id:
            if re.match(r"^[CGD][A-Z0-9]{8,12}$", clean_name.upper()):
                channel_id = clean_name.upper()

        if not channel_id:
            return {"content": [{"type": "text", "text": f"Channel '#{clean_name}' not found or not accessible."}]}

        try:
            data = await self._slack_get(
                token,
                "conversations.members",
                {"channel": channel_id, "limit": 1000},
                timeout=timeout,
            )
            members = data.get("members", [])
            if not members:
                return {"content": [{"type": "text", "text": f"No members found in #{clean_name}."}]}
                
            names = []
            for m in members:
                name = await self._resolve_user_name(token, m)
                names.append(f"- {name} (ID: {m})")
                
            if channel_id.startswith("D"):
                dm_user = None
                for ch in channels:
                    if ch["id"] == channel_id:
                        dm_user = ch.get("user")
                        break
                if dm_user:
                    dm_name = await self._resolve_user_name(token, dm_user)
                    display_channel_name = f"DM with {dm_name}"
                else:
                    display_channel_name = f"DM ({channel_id})"
            else:
                display_channel_name = f"#{clean_name}"

            text = f"Members in {display_channel_name}:\n\n" + "\n".join(names)
            return {"content": [{"type": "text", "text": text}]}
        except Exception as e:
            if channel_id.startswith("D"):
                display_channel_name = f"DM ({channel_id})"
            else:
                display_channel_name = f"#{clean_name}"
            return {"content": [{"type": "text", "text": f"Error fetching members for {display_channel_name}: {e}"}]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_slack_mentions(query: str) -> Dict[str, Optional[str]]:
    """
    Extract Slack @person and #channel mentions from a user query.

    Rules:
    - `#channel` targets a channel
    - `@Person Name` targets a DM when no channel is present
    - When both are present, the channel takes priority for history lookups
    """
    channel: Optional[str] = None
    dm_user: Optional[str] = None

    channel_match = re.search(r"#([\w-]+)", query)
    if channel_match:
        channel = channel_match.group(1)

    dm_match = re.search(
        r"@([A-Za-z][A-Za-z0-9.'\-]*(?:\s+[A-Za-z][A-Za-z0-9.'\-]*)*?)"
        r"(?=\s+(?:last|recent|latest)\b|\s*$|[,.!?](?:\s|$))",
        query,
        re.IGNORECASE,
    )
    if not dm_match:
        dm_match = re.search(r"@([^\s#@]+(?:\s+[^\s#@]+)?)", query)
    if dm_match:
        dm_user = dm_match.group(1).strip().rstrip(".,!?")

    logger.info(
        "Parsed Slack mentions from query=%r -> channel=%r dm_user=%r",
        query,
        channel,
        dm_user,
    )
    return {"channel": channel, "dm_user": dm_user}


def _format_slack_ts(ts: str) -> str:
    """Convert a Slack timestamp (e.g. '1716384000.123456') to human-readable."""
    import datetime

    epoch = float(ts.split(".")[0])
    dt = datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc)
    return dt.strftime("%b %d, %Y at %I:%M %p UTC")
