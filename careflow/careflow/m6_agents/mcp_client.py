"""M6 — MCP client used by every agent (no agent imports the tool functions directly).

One long-lived stdio session per event loop, with a per-agent allow-list
(`ScopedMCP`) so an agent can only call the tools it owns, plus a timeout, bounded
retries on transport errors and a circuit breaker.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from careflow.config import settings
from careflow.m7_observability import tracing
from careflow.m7_observability.breaker import CircuitBreaker

log = logging.getLogger(__name__)
mcp_breaker = CircuitBreaker("mcp")
MCP_TIMEOUT_S = 30.0
MCP_RETRIES = 2


class MCPClient:
    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None

    async def _ensure(self) -> ClientSession:
        loop = asyncio.get_running_loop()
        if self._session is not None and self._loop is loop:
            return self._session
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._session is not None and self._loop is loop:
                return self._session
            self._stack, self._session, self._loop = None, None, loop  # drop sessions from a dead loop
            stack = AsyncExitStack()
            params = StdioServerParameters(command=sys.executable, args=["-m", "careflow.m6_agents.mcp_server"],
                                           cwd=str(settings.data_dir.parent))
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._stack, self._session = stack, session
            return session

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:  # pragma: no cover
                log.warning("mcp close failed")
        self._stack = self._session = self._loop = None

    async def _with_retries(self, op: str, fn):
        last: Exception | None = None
        for attempt in range(MCP_RETRIES + 1):
            try:
                session = await self._ensure()
                return await mcp_breaker.call(lambda: asyncio.wait_for(fn(session), MCP_TIMEOUT_S))
            except (asyncio.TimeoutError, ConnectionError, OSError, BrokenPipeError) as exc:  # transport only
                last = exc
                log.warning("mcp %s transport error (%s); attempt %d", op, type(exc).__name__, attempt + 1)
                await self.close()
                await asyncio.sleep(0.5 * (2**attempt))
        raise ConnectionError(f"mcp {op} failed after retries: {last}")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async def go(s: ClientSession):
            return await s.call_tool(name, arguments)

        with tracing.span(f"mcp:{name}", as_type="tool", tool=name, transport="mcp"):
            res = await self._with_retries(f"tool {name}", go)
        if res.isError:
            return {"ok": False, "error": {"code": "mcp_error", "message": _text(res)[:300]}}
        if res.structuredContent is not None:
            sc = res.structuredContent
            return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
        return json.loads(_text(res))

    async def read_resource(self, uri: str) -> str:
        async def go(s: ClientSession):
            return await s.read_resource(uri)

        with tracing.span(f"mcp-resource:{uri}", as_type="tool", transport="mcp"):
            res = await self._with_retries(f"resource {uri}", go)
        return "".join(getattr(c, "text", "") for c in res.contents)

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> str:
        async def go(s: ClientSession):
            return await s.get_prompt(name, arguments)

        res = await self._with_retries(f"prompt {name}", go)
        return "\n".join(getattr(m.content, "text", "") for m in res.messages)

    async def list_tools(self) -> list[str]:
        async def go(s: ClientSession):
            return await s.list_tools()

        return [t.name for t in (await self._with_retries("list_tools", go)).tools]


def _text(res: Any) -> str:
    return "".join(getattr(c, "text", "") for c in res.content)


_client = MCPClient()


class ScopedMCP:
    """An agent's view of the MCP server: only its allowed tools."""

    def __init__(self, agent: str, allowed_tools: set[str], allowed_resources: tuple[str, ...] = ()) -> None:
        self.agent, self.allowed_tools, self.allowed_resources = agent, frozenset(allowed_tools), allowed_resources

    async def call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        if tool not in self.allowed_tools:
            raise PermissionError(f"agent '{self.agent}' may not call MCP tool '{tool}'")
        return await _client.call_tool(tool, arguments)

    async def read(self, uri: str) -> str:
        if not uri.startswith(self.allowed_resources):
            raise PermissionError(f"agent '{self.agent}' may not read '{uri}'")
        return await _client.read_resource(uri)


async def close_mcp() -> None:
    await _client.close()


get_mcp_client = lambda: _client  # noqa: E731
