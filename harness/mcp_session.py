"""
MCP session helper.
Wraps ClientSession so runner.py never imports from mcp directly.
Keeps the async context manager plumbing in one place.
"""

from __future__ import annotations

import os
import random
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def mcp_session(server_script: str | Path):
    """
    Async context manager that starts the MCP server subprocess and yields
    (session, openai_tools) ready to use.

    openai_tools is a list of dicts in OpenAI function-calling format.

    Usage:
        async with mcp_session("mcp-server/main.py") as (session, tools):
            result = await session.call_tool("calculate_median", {"collection": [1,2,3]})
    """
    server_script = Path(server_script)
    params = StdioServerParameters(
        command="uv",
        args=["run", "python", server_script.name],
        env=dict(os.environ),
        cwd=str(server_script.parent),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            openai_tools = [_to_openai_tool(t) for t in tools_result.tools]
            yield session, openai_tools


def _to_openai_tool(mcp_tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema or {},
        },
    }


def filter_tools_for_task(
    all_tools: list[dict],
    relevant_names: list[str] | str,
    num_distractors: int | None,
) -> list[dict]:
    """
    Slice the tool list based on distractor count.

    num_distractors=None  → all tools          (Standard mode)
    num_distractors=0     → only the relevant tools  (Oracle mode)
    num_distractors=N     → relevant + N random distractors

    relevant_names accepts a single string (L1/L2) or a list (L3).
    The relevant set is used for oracle/distractor filtering only —
    it has no influence on scoring.
    """
    if num_distractors is None:
        return all_tools

    if isinstance(relevant_names, str):
        relevant_names = [relevant_names]
    relevant_set = set(relevant_names)

    relevant = [t for t in all_tools if t["function"]["name"] in relevant_set]

    if num_distractors == 0:
        return relevant

    distractors = [t for t in all_tools if t["function"]["name"] not in relevant_set]
    sampled = random.sample(distractors, min(num_distractors, len(distractors)))
    mixed = relevant + sampled
    random.shuffle(mixed)
    return mixed
