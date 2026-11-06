"""Discover remote MCP tools inside the Research package."""

import asyncio
from urllib.parse import urlparse

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPConnectionError(RuntimeError):
    """The configured MCP server could not supply its tools."""


async def discover_mcp_tools(mcp_server_url: str) -> list[BaseTool]:
    url = mcp_server_url.strip()
    if not url:
        return []
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("mcp_server_url must be an HTTP or HTTPS URL.")
    client = MultiServerMCPClient({
        "research_tools": {"url": url, "transport": "http"}
    })
    try:
        # Each returned tool opens its own session when invoked.
        return await asyncio.wait_for(client.get_tools(), timeout=15)
    except Exception as error:
        raise MCPConnectionError(
            "MCP tools are unavailable. Start the MCP server and check MCP_SERVER_URL."
        ) from error
