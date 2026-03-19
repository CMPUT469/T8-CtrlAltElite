# CMPUT 469 - CtrlAltElite (T8)

Lightweight open-source LLM agent tooling with the Model Context Protocol (MCP).

## Active Architecture

- `harness/runner.py`: active evaluation entrypoint
- `harness/model_client.py`: shared provider/model/base_url/api_key handling
- `harness/mcp_session.py`: MCP server lifecycle for evaluation runs
- `mcp-server/main.py`: FastMCP server entrypoint
- `mcp-server/tools/*`: active domain/tool implementations

## Legacy Code

- `mcp-client/`: legacy interactive client from the initial project setup

The active backend and evaluation architecture is the `harness/` path. Legacy paths are retained only to avoid risky deletions during stabilization work and should not be treated as the primary flow.

## Prerequisites

- Python 3.13+
- `uv` installed
- an OpenAI-compatible inference endpoint
  - local Ollama today
  - vLLM later
- for local Ollama, at least one pulled model, for example:

```powershell
ollama pull qwen2.5
```

## Install Dependencies

```powershell
pip install -r requirements-eval.txt
```

## Run Evaluations

```powershell
python -m harness.runner --dataset bfcl --model qwen2.5:7b --level L1
```

`configs/models.yaml` is the runtime model registry. Runtime resolves model defaults from that file first, then applies any CLI overrides such as `--backend`, `--base-url`, and `--api-key`.

## Threshold Sweep

```powershell
python -m harness.threshold_sweep --dataset bfcl --model qwen2.5:7b --sweep
```

## PostgreSQL Setup

The MCP server includes SQL tools backed by the [Postgrespro demo database](https://postgrespro.com/community/demodb).

### 1. Install PostgreSQL

Download and install from https://www.postgresql.org/download/windows/ .

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

```text
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/demo
```

### 4. Run SQL Tool Evaluation

```powershell
python -m harness.runner --dataset postgres --model qwen2.5:7b --level L1
```

## Notes

- Override model/provider defaults with `--backend`, `--base-url`, and `--api-key` when needed.
- If model loading fails under Ollama, run `ollama pull <model-name>` first.
- `DATABASE_URL` must be set in `.env` for SQL tools to work. Math tools work without it.
