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

## PostgreSQL Setup (for SQL Tool Evaluation)

The MCP server includes 5 SQL tools backed by the [Postgrespro demo database](https://postgrespro.com/community/demodb) (airline flights dataset).

### 1. Install PostgreSQL

Download and install from https://www.postgresql.org/download/windows/ (PostgreSQL 18 recommended).

After installation, add the bin folder to your PATH:

```powershell
[System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\PostgreSQL\18\bin", [System.EnvironmentVariableTarget]::User)
```

Restart your terminal, then verify:

```powershell
psql --version
```

### 2. Download and Restore the Demo Database

Download `demo-small-en.zip` from https://postgrespro.com/community/demodb, extract it, then restore:

```powershell
createdb -U postgres demo
psql -U postgres -d demo -f "C:\path\to\demo-small-en-20170815.sql"
```

### 3. Configure the Connection String

Copy `.env.example` to `.env` and set your password:

```
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/demo
```

### 4. Run SQL Tool Evaluation

```powershell
python evaluate_bfcl.py --model qwen3:4b --synthetic postgres_test_cases.jsonl
```

This runs 20 test cases against the demo database across all 5 SQL tools:
- `list_tables` — lists tables in a schema
- `describe_table` — returns column names and types
- `get_row_count` — counts rows in a table
- `get_foreign_keys` — returns foreign key relationships
- `execute_query` — runs a read-only SELECT query

Results are logged to Supabase automatically.

## Notes

- If port `8000` or `8012` is already in use, choose another port and keep client/server ports matched.
- If model loading fails, run `ollama pull <model-name>` first.
- `DATABASE_URL` must be set in `.env` for SQL tools to work. Math tools work without it.
