"""MCP server entrypoint for local stdio and HTTP transports."""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Multi-Dataset Function Calling Tools", json_response=True)


# ============================================================================
# TOOL IMPORTS - Each module registers its own tools
# ============================================================================

# Import tools - they will auto-register via decorators
from tools import bfcl_math_tools, jefferson_stats_tools

# Pass the mcp instance to each module for registration
bfcl_math_tools.register_tools(mcp)
jefferson_stats_tools.register_tools(mcp)


# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="MCP server for function calling threshold testing."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport type for the server.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind for HTTP transports (ignored for stdio).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for HTTP transports (ignored for stdio).",
    )
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Optional ASGI mount path for HTTP transports.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.transport != "stdio":
        mcp.settings.host = args.host
        mcp.settings.port = args.port

    mcp.run(transport=args.transport, mount_path=args.mount_path)


if __name__ == "__main__":
    main()
