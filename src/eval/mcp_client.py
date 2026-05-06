"""HTTP MCP client for the Dataverse MCP server."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MCPClient:
    """Connects to a Dataverse MCP server over HTTP (Streamable HTTP transport).

    Handles tool discovery, tool invocation, and trace capture.
    Uses the MCP Streamable HTTP transport protocol:
    - POST to the MCP endpoint with JSON-RPC 2.0 messages
    """

    def __init__(self, server_url: str, auth_token: str | None = None, timeout: float = 120.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._tools: list[dict[str, Any]] = []
        self._request_id = 0

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        self._client = httpx.Client(headers=headers, timeout=timeout)
        self._session_id: str | None = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_jsonrpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC 2.0 request to the MCP server."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        headers = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        logger.debug("MCP request: %s %s", method, json.dumps(params or {})[:200])

        response = self._client.post(self.server_url, json=payload, headers=headers)

        # Capture session ID from response headers
        if "Mcp-Session-Id" in response.headers:
            self._session_id = response.headers["Mcp-Session-Id"]

        response.raise_for_status()

        # Handle SSE responses
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse_response(response.text)

        result = response.json()
        if "error" in result:
            raise MCPError(result["error"].get("message", "Unknown MCP error"), result["error"])
        return result.get("result")

    def _parse_sse_response(self, sse_text: str) -> Any:
        """Parse Server-Sent Events response to extract JSON-RPC result."""
        last_data = None
        for line in sse_text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    last_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

        if last_data is None:
            raise MCPError("No valid data in SSE response")

        if "error" in last_data:
            raise MCPError(
                last_data["error"].get("message", "Unknown MCP error"),
                last_data["error"],
            )
        return last_data.get("result", last_data)

    def initialize(self) -> dict[str, Any]:
        """Initialize the MCP session."""
        result = self._send_jsonrpc("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "dataverse-mcp-eval", "version": "0.1.0"},
        })
        # Send initialized notification
        self._client.post(
            self.server_url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={"Mcp-Session-Id": self._session_id} if self._session_id else {},
        )
        return result

    def discover_tools(self) -> list[dict[str, Any]]:
        """Discover available tools from the MCP server."""
        result = self._send_jsonrpc("tools/list")
        self._tools = result.get("tools", [])
        logger.info("Discovered %d MCP tools", len(self._tools))
        return self._tools

    def get_tools(self) -> list[dict[str, Any]]:
        """Return cached tools (call discover_tools first)."""
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool on the MCP server and return the result."""
        start = time.perf_counter()
        try:
            result = self._send_jsonrpc("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("Tool %s completed in %.0fms", tool_name, duration_ms)
            return {"result": result, "duration_ms": duration_ms, "error": None}
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("Tool %s failed after %.0fms: %s", tool_name, duration_ms, e)
            return {"result": None, "duration_ms": duration_ms, "error": str(e)}

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Convert MCP tools to OpenAI function-calling format."""
        openai_tools = []
        for tool in self._tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                },
            })
        return openai_tools

    def close(self) -> None:
        self._client.close()


class MCPError(Exception):
    """Error from the MCP server."""

    def __init__(self, message: str, detail: Any = None):
        super().__init__(message)
        self.detail = detail
