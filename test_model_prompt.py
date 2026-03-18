#!/usr/bin/env python3
"""
Interactive Model Testing Script

Test any Ollama model with custom prompts and optional tool calling.
Perfect for quick testing of gpt-oss:20b or other models.

Usage:
    python3 test_model_prompt.py --model gpt-oss:20b
    python3 test_model_prompt.py --model gpt-oss:20b --prompt "Your question here"
    python3 test_model_prompt.py --model gpt-oss:20b --with-tools
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from inference_backend import (
    BackendConfig,
    add_backend_cli_args,
    create_openai_client,
    resolve_backend_config,
    run_chat_completion,
)


def test_model_simple(backend_config: BackendConfig, prompt: str):
    """Test model with simple prompt (no tools)"""
    print("\n" + "=" * 60)
    print(f"Testing Model: {backend_config.model}")
    print("=" * 60)
    print(f"\n📝 Your Prompt:\n{prompt}\n")
    print("🤖 Model Response:")
    print("-" * 60)
    
    client = create_openai_client(backend_config)
    
    try:
        response = run_chat_completion(
            client,
            backend_config,
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        
        answer = response.choices[0].message.content
        print(answer)
        print("\n" + "-" * 60)
        print("✅ Success!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        return False
    
    return True


async def test_model_with_tools(backend_config: BackendConfig, prompt: str, use_fallback: bool = False):
    """Test model with MCP tools available"""
    print("\n" + "=" * 60)
    print(f"Testing Model with Tools: {backend_config.model}")
    print("=" * 60)
    print(f"\n📝 Your Prompt:\n{prompt}\n")
    
    # Connect to MCP server
    server_path = Path(__file__).parent / "mcp-server" / "main.py"
    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
        env=None
    )
    
    print("🔧 Connecting to MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools_list = await session.list_tools()
            print(f"✅ Connected! {len(tools_list.tools)} tools available")
            
            # Convert to OpenAI format
            openai_tools = []
            for tool in tools_list.tools:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema
                    }
                }
                openai_tools.append(openai_tool)
            
            print(f"\n🛠️  Available Tools: {', '.join([t.name for t in tools_list.tools[:5]])}...")
            print("\n🤖 Model Response:")
            print("-" * 60)
            
            # Call model
            client = create_openai_client(backend_config)
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
                {"role": "user", "content": prompt}
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
                
                # Check for tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    print("✅ Native Tool Call Detected!")
                    tool_call = message.tool_calls[0]
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"\n🔧 Tool: {function_name}")
                    print(f"📋 Arguments: {json.dumps(function_args, indent=2)}")
                    
                    # Execute tool
                    print(f"\n⚙️  Executing tool...")
                    try:
                        result = await session.call_tool(function_name, function_args)
                        
                        # Extract result
                        if hasattr(result, 'content') and result.content:
                            content = result.content[0]
                            if hasattr(content, 'text'):
                                result_data = json.loads(content.text)
                                print(f"✅ Result: {json.dumps(result_data, indent=2)}")
                            else:
                                print(f"✅ Result: {content}")
                        else:
                            print(f"✅ Result: {result}")
                    
                    except Exception as e:
                        print(f"❌ Tool execution error: {str(e)}")
                
                elif hasattr(message, 'content') and message.content:
                    print("📝 Text Response (no tool call):")
                    print(message.content)
                    
                    # Try to parse fallback JSON
                    if use_fallback:
                        content = message.content.strip()
                        if content.startswith("{") and "tool" in content:
                            try:
                                fallback = json.loads(content)
                                if "tool" in fallback and "args" in fallback:
                                    print("\n✅ Fallback Tool Call Detected!")
                                    print(f"🔧 Tool: {fallback['tool']}")
                                    print(f"📋 Arguments: {json.dumps(fallback['args'], indent=2)}")
                                    
                                    # Execute
                                    try:
                                        result = await session.call_tool(fallback['tool'], fallback['args'])
                                        print(f"✅ Result: {result}")
                                    except Exception as e:
                                        print(f"❌ Tool execution error: {str(e)}")
                            except:
                                pass
                else:
                    print("❓ No response content")
                
                print("\n" + "-" * 60)
                print("✅ Test complete!\n")
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
                return False
    
    return True


def interactive_mode(backend_config: BackendConfig, with_tools: bool = False, use_fallback: bool = False):
    """Interactive prompt loop"""
    print("\n" + "=" * 60)
    print(f"🎯 Interactive Mode - {backend_config.model}")
    print("=" * 60)
    print("Enter your prompts below. Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prompt = input("📝 Your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            if not prompt:
                continue
            
            if with_tools:
                asyncio.run(test_model_with_tools(backend_config, prompt, use_fallback))
            else:
                test_model_simple(backend_config, prompt)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test Ollama models with custom prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (simple)
  python3 test_model_prompt.py --model gpt-oss:20b

  # Single prompt test
  python3 test_model_prompt.py --model gpt-oss:20b --prompt "What is 2+2?"

  # Test with tools (interactive)
  python3 test_model_prompt.py --model qwen2.5:7b --with-tools

  # Test with tools (single prompt)
  python3 test_model_prompt.py --model qwen2.5:7b --with-tools --prompt "Calculate skewness of [1,2,3]"

  # Test with fallback (for non-native models)
  python3 test_model_prompt.py --model gpt-oss:20b --with-tools --fallback --prompt "Calculate mean of [1,2,3]"
        """
    )
    
    parser.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Ollama model name (default: gpt-oss:20b)"
    )
    parser.add_argument(
        "--prompt",
        help="Single prompt to test (interactive mode if not provided)"
    )
    parser.add_argument(
        "--with-tools",
        action="store_true",
        help="Enable MCP tools (32 math & stats tools)"
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Enable fallback JSON parsing for non-native tool models"
    )
    add_backend_cli_args(parser)
    
    args = parser.parse_args()
    backend_config = resolve_backend_config(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    
    # Single prompt mode
    if args.prompt:
        if args.with_tools:
            asyncio.run(test_model_with_tools(backend_config, args.prompt, args.fallback))
        else:
            test_model_simple(backend_config, args.prompt)
    else:
        # Interactive mode
        interactive_mode(backend_config, args.with_tools, args.fallback)


if __name__ == "__main__":
    main()
