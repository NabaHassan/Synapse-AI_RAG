"""MCP client module — Google Workspace, Microsoft 365, Slack, Notion."""
from .google_workspace_client import GoogleWorkspaceMCPClient
from .microsoft365_client import Microsoft365MCPClient
from .notion_client import NotionMCPClient

__all__ = ["GoogleWorkspaceMCPClient", "Microsoft365MCPClient", "NotionMCPClient"]
