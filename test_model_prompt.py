#!/usr/bin/env python3
"""
Interactive model testing script.

Test any OpenAI-compatible backend with custom prompts and optional MCP tools.
Useful for quick manual validation of local Ollama models today and other
OpenAI-compatible backends later.
"""

import argparse
import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from evaluation_framework import (
    extract_result_value,
    mcp_tools_to_openai_tools,
    maybe_parse_fallback_tool_json,
)
from inference_backend import (
    BackendConfig,
    add_backend_cli_args,
    create_openai_client,
    resolve_backend_config,
    run_chat_completion,
)


def test_model_simple(backend_config: BackendConfig, prompt: str):
    """Test model with simple prompt (no tools)."""
    print("\n" + "=" * 60)
    print(f"Testing Model: {backend_config.model}")
    print("=" * 60)
    print(f"\nYour Prompt:\n{prompt}\n")
    print("Model Response:")
    print("-" * 60)

    client = create_openai_client(backend_config)

    try:
        response = run_chat_completion(
            client,
            backend_config,
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        answer = response.choices[0].message.content
        print(answer)
        print("\n" + "-" * 60)
        print("Success!\n")

    except Exception as exc:
        print(f"\nError: {exc}\n")
        return False

    return True


async def test_model_with_tools(backend_config: BackendConfig, prompt: str, use_fallback: bool = False):
    """Test model with MCP tools available."""
    print("\n" + "=" * 60)
    print(f"Testing Model with Tools: {backend_config.model}")
    print("=" * 60)
    print(f"\nYour Prompt:\n{prompt}\n")

    server_path = Path(__file__).parent / "mcp-server" / "main.py"
    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
        env=None,
    )

    print("Connecting to MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_list = await session.list_tools()
            print(f"Connected! {len(tools_list.tools)} tools available")

            openai_tools = mcp_tools_to_openai_tools(tools_list)

            print(f"\nAvailable Tools: {', '.join([t.name for t in tools_list.tools[:5]])}...")
            print("\nModel Response:")
            print("-" * 60)

            client = create_openai_client(backend_config)

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
                {"role": "user", "content": prompt},
            ]

            if use_fallback:
                messages[0]["content"] += (
                    "\n\nIf you cannot emit native tool calls, respond with JSON: "
                    '{"tool":"<tool_name>","args":{...}}'
                )

            try:
                response = run_chat_completion(
                    client,
                    backend_config,
                    messages,
                    openai_tools,
                    tool_choice="auto",
                )

                message = response.choices[0].message

                if hasattr(message, "tool_calls") and message.tool_calls:
                    print("Native Tool Call Detected!")
                    tool_call = message.tool_calls[0]
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"\nTool: {function_name}")
                    print(f"Arguments: {json.dumps(function_args, indent=2)}")

                    print("\nExecuting tool...")
                    try:
                        result = await session.call_tool(function_name, function_args)
                        result_data = extract_result_value(result)
                        print(f"Result: {json.dumps(result_data, indent=2, ensure_ascii=False)}")
                    except Exception as exc:
                        print(f"Tool execution error: {exc}")

                elif hasattr(message, "content") and message.content:
                    print("Text Response (no tool call):")
                    print(message.content)

                    if use_fallback:
                        fallback = maybe_parse_fallback_tool_json(message.content)
                        if fallback:
                            print("\nFallback Tool Call Detected!")
                            print(f"Tool: {fallback['tool']}")
                            print(f"Arguments: {json.dumps(fallback['args'], indent=2)}")

                            try:
                                result = await session.call_tool(fallback["tool"], fallback["args"])
                                result_data = extract_result_value(result)
                                print(f"Result: {json.dumps(result_data, indent=2, ensure_ascii=False)}")
                            except Exception as exc:
                                print(f"Tool execution error: {exc}")
                else:
                    print("No response content")

                print("\n" + "-" * 60)
                print("Test complete!\n")

            except Exception as exc:
                print(f"\nError: {exc}\n")
                return False

    return True


def interactive_mode(backend_config: BackendConfig, with_tools: bool = False, use_fallback: bool = False):
    """Interactive prompt loop."""
    print("\n" + "=" * 60)
    print(f"Interactive Mode - {backend_config.model}")
    print("=" * 60)
    print("Enter your prompts below. Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("Your prompt: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!\n")
                break

            if not prompt:
                continue

            if with_tools:
                asyncio.run(test_model_with_tools(backend_config, prompt, use_fallback))
            else:
                test_model_simple(backend_config, prompt)

        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as exc:
            print(f"\nError: {exc}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenAI-compatible models with custom prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_model_prompt.py --model gpt-oss:20b
  python3 test_model_prompt.py --model gpt-oss:20b --prompt "What is 2+2?"
  python3 test_model_prompt.py --model qwen2.5:7b --with-tools
  python3 test_model_prompt.py --model qwen2.5:7b --with-tools --prompt "Calculate skewness of [1,2,3]"
  python3 test_model_prompt.py --model gpt-oss:20b --with-tools --fallback --prompt "Calculate mean of [1,2,3]"
        """,
    )

    parser.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Model name (default: gpt-oss:20b)",
    )
    parser.add_argument(
        "--prompt",
        help="Single prompt to test (interactive mode if not provided)",
    )
    parser.add_argument(
        "--with-tools",
        action="store_true",
        help="Enable MCP tools",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Enable fallback JSON parsing for non-native tool models",
    )
    add_backend_cli_args(parser)

    args = parser.parse_args()
    backend_config = resolve_backend_config(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    if args.prompt:
        if args.with_tools:
            asyncio.run(test_model_with_tools(backend_config, args.prompt, args.fallback))
        else:
            test_model_simple(backend_config, args.prompt)
    else:
        interactive_mode(backend_config, args.with_tools, args.fallback)


if __name__ == "__main__":
    main()
