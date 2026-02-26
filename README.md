# CMPUT 469 - CtrlAltElite (T8)

Lightweight open-source LLM agent tooling with the Model Context Protocol (MCP).

## Project Structure

- `mcp-server/main.py`: FastMCP server with demo tools/resources/prompts.
- `mcp-client/main.py`: Ollama-powered MCP client with tool-calling loop.

## Prerequisites

- Python 3.13+
- `uv` installed
- Ollama installed and running locally on `http://localhost:11434`
- At least one local model pulled, for example:

```powershell
ollama pull qwen2.5
```

## Install Dependencies

```powershell
cd mcp-server
uv sync

cd ../mcp-client
uv sync
```

## Run The Project (Recommended: STDIO)

Start from the client folder. The client launches the server subprocess automatically.

```powershell
cd mcp-client
uv run main.py --transport stdio --server ../mcp-server/main.py --model qwen2.5
```

At the `Query:` prompt, enter questions. Type `quit` to exit.

## Run The Project (HTTP Mode)

Use two terminals.

Terminal 1 (server):

```powershell
cd mcp-server
uv run main.py --transport streamable-http --host 127.0.0.1 --port 8012
```

Terminal 2 (client):

```powershell
cd mcp-client
uv run main.py --transport http --url http://127.0.0.1:8012/mcp --model qwen2.5
```

## Notes

- If port `8000` or `8012` is already in use, choose another port and keep client/server ports matched.
- If model loading fails, run `ollama pull <model-name>` first.
