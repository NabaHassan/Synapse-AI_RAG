"""
Microsoft 365 MCP Client — Outlook + OneDrive via Microsoft Graph (read-only).

Handles:
  - OAuth 2.0 delegated flow (per-user tokens in Redis)
  - Token refresh via refresh_token grant
  - Local Graph REST tool calls (Outlook mail + OneDrive/SharePoint files)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_TOKEN_TTL_SECONDS = int(os.getenv("MICROSOFT_MCP_TOKEN_TTL", "86400"))

_SERVICES: Dict[str, Dict[str, Any]] = {
    "outlook": {
        "scopes": [
            "https://graph.microsoft.com/Mail.Read",
            "https://graph.microsoft.com/User.Read",
            "offline_access",
        ],
    },
    "onedrive": {
        "scopes": [
            "https://graph.microsoft.com/Files.Read.All",
            "https://graph.microsoft.com/Sites.Read.All",
            "https://graph.microsoft.com/User.Read",
            "offline_access",
        ],
    },
}


class Microsoft365MCPClient:
    """
    Lightweight MCP client for Microsoft 365 (Outlook + OneDrive).

    Stores one OAuth token set per (user_id, service) in Redis.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        redis_client,
        redis_key_prefix: str = "synapse",
        token_ttl_seconds: int = _TOKEN_TTL_SECONDS,
        tenant_id: str = "common",
        services: Optional[List[str]] = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_uri = redirect_uri
        self._redis = redis_client
        self._key_prefix = redis_key_prefix
        self._token_ttl = token_ttl_seconds
        self._tenant_id = tenant_id or "common"
        self._services = services or list(_SERVICES.keys())
        self._authorize_url = (
            f"https://login.microsoftonline.com/{self._tenant_id}/oauth2/v2.0/authorize"
        )
        self._token_url = (
            f"https://login.microsoftonline.com/{self._tenant_id}/oauth2/v2.0/token"
        )

        logger.info(
            "Microsoft365MCPClient initialized (services=%s tenant=%s redirect_uri=%s)",
            self._services,
            self._tenant_id,
            self._redirect_uri,
        )

    def _all_scopes(self, services: Optional[List[str]] = None) -> List[str]:
        svcs = services or self._services
        scopes: List[str] = []
        for svc in svcs:
            if svc in _SERVICES:
                scopes.extend(_SERVICES[svc]["scopes"])
        return list(dict.fromkeys(scopes))

    def get_auth_url(
        self,
        user_id: str,
        services: Optional[List[str]] = None,
    ) -> str:
        """Generate a Microsoft OAuth 2.0 authorization URL."""
        svcs = services or self._services
        state_payload = json.dumps({"user_id": user_id, "services": svcs})
        state = urllib.parse.quote(state_payload)
        scopes = self._all_scopes(svcs)

        params = {
            "client_id": self._client_id,
            "response_type": "code",
            "redirect_uri": self._redirect_uri,
            "response_mode": "query",
            "scope": " ".join(scopes),
            "state": state,
            "prompt": "consent",
        }
        auth_url = f"{self._authorize_url}?{urllib.parse.urlencode(params)}"
        logger.info("Generated Microsoft auth URL for user_id=%s services=%s", user_id, svcs)
        return auth_url

    def exchange_code(self, code: str, state: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens and persist to Redis."""
        try:
            state_payload = json.loads(urllib.parse.unquote(state))
            user_id: str = state_payload["user_id"]
            services: List[str] = state_payload.get("services", self._services)
        except Exception as exc:
            raise ValueError(f"Invalid Microsoft OAuth state parameter: {exc}") from exc

        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                self._token_url,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": self._redirect_uri,
                    "grant_type": "authorization_code",
                    "scope": " ".join(self._all_scopes(services)),
                },
            )
            if resp.status_code != 200:
                logger.error("Microsoft token exchange failed status=%s body=%s", resp.status_code, resp.text[:300])
                resp.raise_for_status()
            token_response = resp.json()

        access_token = token_response.get("access_token")
        if not access_token:
            raise ValueError("Microsoft OAuth response missing access_token")

        expires_in = int(token_response.get("expires_in") or 3600)
        token_data = {
            "access_token": access_token,
            "refresh_token": token_response.get("refresh_token"),
            "token_type": token_response.get("token_type", "Bearer"),
            "scopes": self._all_scopes(services),
            "expires_at": time.time() + expires_in,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }

        for svc in services:
            if svc in _SERVICES:
                redis_key = self._token_key(user_id, svc)
                self._redis.setex(redis_key, self._token_ttl, json.dumps(token_data))
                logger.info("Stored Microsoft MCP token for user_id=%s service=%s", user_id, svc)

        return {
            "user_id": user_id,
            "services_authorized": services,
            "access_token": access_token,
        }

    def _token_key(self, user_id: str, service: str) -> str:
        return f"{self._key_prefix}:mcp:microsoft:token:{user_id}:{service}"

    def _load_access_token(self, user_id: str, service: str) -> Optional[str]:
        """Load and refresh access token from Redis."""
        raw = self._redis.get(self._token_key(user_id, service))
        if not raw:
            logger.debug("No Microsoft token in Redis for user_id=%s service=%s", user_id, service)
            return None

        try:
            data = json.loads(raw)
        except Exception:
            logger.warning("Invalid Microsoft token JSON for user_id=%s service=%s", user_id, service)
            return None

        access_token = data.get("access_token")
        expires_at = float(data.get("expires_at") or 0)
        if access_token and expires_at > time.time() + 60:
            return access_token

        refresh_token = data.get("refresh_token")
        if not refresh_token:
            logger.warning("Microsoft token expired and no refresh_token for user_id=%s service=%s", user_id, service)
            return access_token

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    self._token_url,
                    data={
                        "client_id": data.get("client_id", self._client_id),
                        "client_secret": data.get("client_secret", self._client_secret),
                        "refresh_token": refresh_token,
                        "grant_type": "refresh_token",
                        "scope": " ".join(data.get("scopes") or self._all_scopes([service])),
                    },
                )
                if resp.status_code != 200:
                    logger.warning(
                        "Microsoft token refresh failed user_id=%s service=%s status=%s",
                        user_id,
                        service,
                        resp.status_code,
                    )
                    return None
                refreshed = resp.json()

            new_access = refreshed.get("access_token")
            if not new_access:
                return None

            data["access_token"] = new_access
            data["expires_at"] = time.time() + int(refreshed.get("expires_in") or 3600)
            if refreshed.get("refresh_token"):
                data["refresh_token"] = refreshed["refresh_token"]

            self._redis.setex(self._token_key(user_id, service), self._token_ttl, json.dumps(data))
            logger.debug("Refreshed Microsoft token for user_id=%s service=%s", user_id, service)
            return new_access
        except Exception as exc:
            logger.warning(
                "Microsoft token refresh error user_id=%s service=%s: %s",
                user_id,
                service,
                exc,
            )
            return None

    def is_authenticated(self, user_id: str, service: str) -> bool:
        if not user_id or service not in _SERVICES:
            return False
        return self._load_access_token(user_id, service) is not None

    def revoke_tokens(self, user_id: str) -> None:
        for svc in _SERVICES:
            self._redis.delete(self._token_key(user_id, svc))
        logger.info("Revoked all Microsoft MCP tokens for user_id=%s", user_id)

    def auth_status(self, user_id: str) -> Dict[str, bool]:
        return {svc: self.is_authenticated(user_id, svc) for svc in self._services}

    def call_tool(
        self,
        user_id: str,
        service: str,
        tool: str,
        params: Dict[str, Any],
        timeout_seconds: float = 15.0,
    ) -> Dict[str, Any]:
        """Call a Microsoft 365 tool via local Graph REST."""
        if service not in _SERVICES:
            raise ValueError(f"Unsupported Microsoft service: {service}")

        token = self._load_access_token(user_id, service)
        logger.info(
            "Microsoft MCP tool call: service=%s tool=%s user_id=%s token_present=%s params=%s",
            service,
            tool,
            user_id,
            bool(token),
            json.dumps(params, default=str)[:500],
        )
        if not token:
            raise PermissionError(
                f"No valid Microsoft credentials for user={user_id} service={service}. "
                "User should re-authenticate via /mcp/microsoft/auth."
            )

        return self._call_tool_local(
            service=service,
            tool=tool,
            params=params,
            token=token,
            timeout_seconds=timeout_seconds,
        )

    def _call_tool_local(
        self,
        service: str,
        tool: str,
        params: Dict[str, Any],
        token: str,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        logger.info("=== Microsoft 365 MCP Tool Call === service=%s tool=%s", service, tool)

        if service == "outlook":
            from src.mcp.outlook_search import run_outlook_tool

            result = run_outlook_tool(
                token=token,
                tool=tool,
                params=dict(params),
                timeout=timeout_seconds,
            )
        elif service == "onedrive":
            from src.mcp.onedrive_search import run_onedrive_tool

            result = run_onedrive_tool(
                token=token,
                tool=tool,
                params=dict(params),
                timeout=timeout_seconds,
            )
        else:
            raise NotImplementedError(f"Local Graph fallback not implemented for service: {service}")

        items = result.get("items") or []
        content_blocks = result.get("content") or []
        if items:
            logger.info("Microsoft tool result: %d items", len(items))
        elif content_blocks:
            logger.info("Microsoft tool result: %d content block(s)", len(content_blocks))
        else:
            logger.info("Microsoft tool result: empty")
        logger.info("=== End Microsoft 365 MCP Tool Call ===")
        return result
