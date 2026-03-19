"""
main.py

Interactive MCP client using the shared OpenAI-compatible provider path.

Key features:
- Connects to an MCP server via:
  - STDIO (recommended for local dev): client launches the server script
  - Streamable HTTP: client connects to a running server URL
- Reuses the harness backend config flow for provider/model/base_url/api_key
- Attempts native tool-calling when the model supports it
- Includes a JSON fallback for models that don't reliably emit tool_calls
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.model_client import ModelClient, resolve_model_config


def _json_dumps_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def _normalize_mcp_tool_result(result: Any) -> str:
    """
    MCP tool results vary slightly by SDK version.
    Try to convert to a compact string to feed back to the LLM.
    """

    content = getattr(result, "content", None)
    if content is not None:
        return _json_dumps_safe(content)

    # Some versions may provide model_dump()
    if hasattr(result, "model_dump"):
        return _json_dumps_safe(result.model_dump())

    return _json_dumps_safe(result)


def _mcp_tools_to_openai_tools(mcp_tools: Any) -> List[Dict[str, Any]]:
    """
    Convert MCP tool definitions into OpenAI-compatible "tools" schema.
    Ollama supports an OpenAI-compatible chat.completions endpoint.
    """
    openai_tools: List[Dict[str, Any]] = []
    for tool in mcp_tools.tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": (tool.description or ""),
                    "parameters": (tool.inputSchema or {}),
                },
            }
        )
    return openai_tools


def _maybe_parse_fallback_tool_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Fallback path: if the model doesn't emit tool_calls, we ask it to output ONLY:
      {"tool":"<name>","args":{...}}
    """
    if not text:
        return None
    s = text.strip()

    # Allow fenced JSON blocks too
    if s.startswith("```"):
        s = s.strip("`")
        # common patterns: ```json\n{...}\n```
        s = s.replace("json\n", "", 1).strip()

    if not (s.startswith("{") and s.endswith("}")):
        return None

    try:
        obj = json.loads(s)
    except Exception:
        return None

    if isinstance(obj, dict) and "tool" in obj and "args" in obj and isinstance(obj["args"], dict):
        return obj

    return None


class McpConnector:
    def __init__(self):
        self._exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

    async def connect_stdio(self, server_script_path: str) -> None:
        """
        Launch an MCP server script as a subprocess and connect over STDIO.
        """
        if not (server_script_path.endswith(".py") or server_script_path.endswith(".js")):
            raise ValueError("Server script must end with .py or .js")

        command = sys.executable if server_script_path.endswith(".py") else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()

    async def connect_http(self, url: str) -> None:
        read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
            streamable_http_client(url)
        )
        self.session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()

    async def close(self) -> None:
        await self._exit_stack.aclose()



class UniversalMcpClient:
    def __init__(self, mcp: McpConnector, llm: ModelClient):
        if mcp.session is None:
            raise ValueError("MCP session is not initialized.")
        self._mcp = mcp
        self._llm = llm

    async def _load_tools(self) -> Tuple[Any, List[Dict[str, Any]]]:
        assert self._mcp.session is not None
        mcp_tools = await self._mcp.session.list_tools()
        openai_tools = _mcp_tools_to_openai_tools(mcp_tools)
        return mcp_tools, openai_tools

    async def answer(self, user_query: str) -> str:
        """
        One-turn query handler:
        - Ask model
        - If model requests tool call(s), execute via MCP and ask model again with tool results
        - If model doesn't request tool call, allow fallback JSON tool request
        """
        assert self._mcp.session is not None
        _, openai_tools = await self._load_tools()

        system_prompt = (
            "You are a helpful assistant running locally.\n"
            "You MAY call tools when it helps.\n\n"
            "If you want to call a tool but you cannot emit a native tool call, "
            "respond with ONLY valid JSON in this exact format:\n"
            '{"tool":"<tool_name>","args":{...}}\n'
            "Otherwise, respond normally."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        first = self._llm.chat(messages, tools=openai_tools)
        first_msg = first.choices[0].message

        # Path A: native tool calling (best case)
        tool_calls = getattr(first_msg, "tool_calls", None)
        if tool_calls:
            messages.append(first_msg.model_dump())

            for call in tool_calls:
                tool_name = call.function.name
                tool_args = json.loads(call.function.arguments or "{}")

                result = await self._mcp.session.call_tool(tool_name, tool_args)
                tool_output_text = _normalize_mcp_tool_result(result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_output_text,
                    }
                )

            final = self._llm.chat(messages, tools=openai_tools)
            return (final.choices[0].message.content or "").strip()

        # Path B: fallback JSON tool request
        raw_text = (first_msg.content or "").strip()
        fallback = _maybe_parse_fallback_tool_json(raw_text)
        if fallback:
            tool_name = str(fallback["tool"])
            tool_args = fallback["args"]

            result = await self._mcp.session.call_tool(tool_name, tool_args)
            tool_output_text = _normalize_mcp_tool_result(result)

            messages.append({"role": "assistant", "content": raw_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool result: {tool_output_text}\nNow answer the original user question clearly.",
                }
            )

            final = self._llm.chat(messages, tools=openai_tools)
            return (final.choices[0].message.content or "").strip()

        # Path C: model answered normally
        return raw_text

    async def chat_loop(self) -> None:
        print("\nUniversal MCP Client started. Type 'quit' to exit.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break
            try:
                response = await self.answer(query)
                print("\n" + response)
            except Exception as exc:
                print("\nError:", exc)


# -----------------------------
# CLI / Entrypoint
# -----------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal MCP client powered by OpenAI-compatible backends.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="How to connect to the MCP server: launch via STDIO (default) or connect via Streamable HTTP.",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Path to MCP server script (.py or .js). Required for --transport stdio.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="MCP server URL (e.g., http://localhost:8000/mcp). Required for --transport http.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g. qwen2.5:7b or meta-llama/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["ollama", "vllm", "openai"],
        help="Provider backend. If omitted, prefer configs/models.yaml for this model, else default to ollama.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override API base URL. If omitted, prefer configs/models.yaml for this model.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override API key / bearer token. If omitted, prefer configs/models.yaml or LLM_API_KEY.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Deprecated alias for --base-url when using Ollama.",
    )
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()
    base_url_override = args.base_url
    if base_url_override is None and args.ollama_url != "http://localhost:11434":
        base_url_override = args.ollama_url.rstrip("/")
        if not base_url_override.endswith("/v1"):
            base_url_override += "/v1"

    model_cfg = resolve_model_config(
        args.model,
        backend=args.backend,
        base_url=base_url_override,
        api_key=args.api_key,
    )
    llm = ModelClient(model_cfg, allow_fallback=True)

    mcp = McpConnector()
    try:
        if args.transport == "stdio":
            if not args.server:
                raise SystemExit("Missing --server. Example: --transport stdio --server ../mcp-server/main.py")
            await mcp.connect_stdio(args.server)
        else:
            if not args.url:
                raise SystemExit("Missing --url. Example: --transport http --url http://localhost:8000/mcp")
            await mcp.connect_http(args.url)

        assert mcp.session is not None
        tools = (await mcp.session.list_tools()).tools
        print("Connected to MCP server. Tools:", [t.name for t in tools])
        print(f"Using model: {model_cfg.name} [{model_cfg.backend}]")
        print("LLM endpoint:", model_cfg.base_url)

        client = UniversalMcpClient(mcp=mcp, llm=llm)
        await client.chat_loop()
    finally:
        await mcp.close()


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
