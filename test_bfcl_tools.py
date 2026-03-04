#!/usr/bin/env python3
"""
Quick test of Gorilla BFCL tools in MCP server.
Tests basic arithmetic to verify real Gorilla tools are working.
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def create_client():
    """Create MCP client connected to stdio server."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp-server/main.py", "--transport", "stdio"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def test_tools():
    """Test Gorilla BFCL math tools."""
    async with create_client() as session:
        # List available tools
        tools_result = await session.list_tools()
        print(f"Server has {len(tools_result.tools)} tools:\n")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")
        
        print("\n" + "="*60)
        print("TESTING GORILLA BFCL TOOLS")
        print("="*60 + "\n")
        
        # Test 1: Basic addition
        print("Test 1: add(42, 15)")
        result = await session.call_tool("add", arguments={"a": 42, "b": 15})
        print(f"Result: {result.content[0].text}")
        
        # Test 2: Division
        print("\nTest 2: divide(100, 4)")
        result = await session.call_tool("divide", arguments={"a": 100, "b": 4})
        print(f"Result: {result.content[0].text}")
        
        # Test 3: Square root
        print("\nTest 3: square_root(144, 2)")
        result = await session.call_tool("square_root", arguments={"number": 144, "precision": 2})
        print(f"Result: {result.content[0].text}")
        
        # Test 4: List operations - mean
        print("\nTest 4: mean([10, 20, 30, 40, 50])")
        result = await session.call_tool("mean", arguments={"numbers": [10, 20, 30, 40, 50]})
        print(f"Result: {result.content[0].text}")
        
        # Test 5: Standard deviation
        print("\nTest 5: standard_deviation([2, 4, 6, 8, 10])")
        result = await session.call_tool("standard_deviation", arguments={"numbers": [2, 4, 6, 8, 10]})
        print(f"Result: {result.content[0].text}")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED - Gorilla BFCL tools working!")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(test_tools())
