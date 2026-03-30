**Eureka Setup Guide For T8-CtrlAltElite**

This guide walks a teammate from zero Eureka experience to a fully working evaluation setup for the CMPUT 469 capstone repo on Eureka.

This is based on the path that actually worked in practice.

**What this guide is for**
The goal is to run the project’s active evaluation pipeline on Eureka using:

- the repo’s active `harness/` architecture
- a Python 3.11 virtual environment
- Ollama running directly on Eureka in user space
- models stored under `/projects/<your_clg_username>/...`
- GPU nodes for inference
- no dependency on someone else’s laptop
- no vLLM

**Important high-level decisions**
These were learned from actual testing:

- Use the `harness/` path, not `mcp-client/`
- Use **Ollama on Eureka**, not vLLM
- Use **Python 3.11** in practice
- Use **`/projects/<your_clg_username>/...`** for storage, not `$HOME`
- Keep `uv` installed because the active MCP server flow uses `uv run python ...`
- Install a couple of packages in addition to `requirements-eval.txt`, because the MCP server imports all tool modules eagerly

---

## 1. Account And Access Setup

### What Eureka access looks like
When your account is created, you’ll be assigned a cluster username that looks like:

```text
clg_youridentifier
```

This is your Eureka username. It is not your CCID or general university login.

### First-time account/login steps
Eureka authentication uses **CILogon**. You typically authenticate in the browser during SSH login.

### What you need before you begin
You need:

- an active Eureka account
- your assigned `clg_...` username
- SSH access working from your laptop

### How to test that your account works
From your laptop terminal:

```bash
ssh clg_youridentifier@eureka.paice-ua.com
```

Expected behavior:
- You may be prompted to authenticate via CILogon in a browser.
- After successful auth, you should land on a Eureka login shell.
- Your prompt may look something like:

```bash
clg_youridentifier@eureka-login1:~$
```

### How to confirm the account is really usable
On Eureka, run:

```bash
hostname
pwd
whoami
```

Expected:
- `hostname` should look like `eureka-login1` or similar
- `pwd` should be your home directory
- `whoami` should show your `clg_...` username

### What to send the admin / teammate for shared project access
If you need access to a shared project directory, send:

- your full `clg_...` username
- the specific `/projects/...` directory you need access to

Example message:

```text
Please add clg_youridentifier to the shared project directory:
/projects/clg_a465712f90e8
```

### Security note
Do **not** share:
- your password
- your CILogon session
- SSH private keys
- tokens copied from browser auth

Only share your cluster username when requesting access.

---

## 2. Local Machine Prerequisites

### Required on your laptop
You should have:

- SSH client
- a terminal you are comfortable with
- optional but recommended: VS Code + Remote SSH

### Windows
Recommended options:
- PowerShell
- Windows Terminal
- OpenSSH client
- VS Code with Remote - SSH extension

To verify SSH exists in PowerShell:

```powershell
ssh -V
```

### macOS / Linux
Use your normal terminal.

Verify SSH:

```bash
ssh -V
```

### VS Code Remote SSH
Optional, but very helpful.

Install:
- VS Code
- Remote - SSH extension

This makes it easier to browse files and edit on Eureka.

### Shell assumptions in this guide
Most cluster commands below assume a Bash-style shell on Eureka.

When running commands on your laptop:
- PowerShell examples are fine on Windows
- shell syntax on Eureka should still be Bash

---

## 3. First Login Workflow

### Log into Eureka
From your laptop:

```bash
ssh clg_youridentifier@eureka.paice-ua.com
```

### What to expect
You should see:
- Ubuntu login banner
- Eureka welcome banner
- your prompt on a login node

Example:

```bash
clg_youridentifier@eureka-login1:~$
```

### Verify you are on a login node
Run:

```bash
hostname
```

Expected:
- `eureka-login1`
- `eureka-login2`
- similar login-node hostname

### Check your current location and storage layout
Run:

```bash
pwd
ls /projects
```

If you already know your project path, test it:

```bash
ls /projects/clg_youridentifier
```

---

## 4. Shared Project Access

### Check whether you can access the shared directory
If your project path is, for example:

```bash
/projects/clg_a465712f90e8
```

run:

```bash
ls -ld /projects/clg_a465712f90e8
cd /projects/clg_a465712f90e8
pwd
touch access_test_file
ls access_test_file
rm access_test_file
```

### Expected result
You should be able to:
- enter the directory
- create a file
- remove the file

### If access fails
If you get permission errors:
- stop
- ask the owner/admin to add your `clg_...` username to that project directory

### Why this matters
This project should use `/projects/...` for:
- repo clone
- venv
- pip cache
- Ollama install
- Ollama model files
- logs
- results

Do **not** rely on `$HOME` for big files.

---

## 5. Repo Setup On Eureka

### Clone into project-backed storage
On a login node:

```bash
cd /projects/clg_youridentifier
git clone https://github.com/CMPUT469/T8-CtrlAltElite.git
cd T8-CtrlAltElite
```

### Confirm the repo structure
Run:

```bash
ls
ls harness
ls configs
ls mcp-server
```

### What to look for
Active architecture should include:

- `harness/runner.py`
- `harness/model_client.py`
- `harness/mcp_session.py`
- `mcp-server/main.py`
- `configs/models.yaml`

### What to ignore
`mcp-client/` is legacy. Do not treat it as the main flow.

---

## 6. Python Environment Setup On The Login Node

### Why the login node
Use the login node for:
- environment setup
- package installation
- model downloads
- repo inspection

Do not waste GPU node time on those steps.

### Make cache/temp directories in project storage
From the repo root:

```bash
mkdir -p /projects/clg_youridentifier/pip-cache
mkdir -p /projects/clg_youridentifier/tmp
```

### Ensure Python 3.11 is available
Check:

```bash
python3.11 --version
```

If needed, load a module first depending on site config. If `python3.11` already works, use it directly.

### Create the virtual environment
From repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show something like:

```bash
(.venv) clg_youridentifier@eureka-login1:/projects/clg_youridentifier/T8-CtrlAltElite$
```

### Install `uv`
Even though we use `pip` for the main environment, `uv` is still needed because the MCP server path uses `uv run python ...`.

Install it on the login node:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
command -v uv
uv --version
```

Expected:
- `uv` should be found in `~/.local/bin/uv`

### Set cache/temp env vars
While in the venv:

```bash
export PATH="$HOME/.local/bin:$PATH"
export PIP_CACHE_DIR=/projects/clg_youridentifier/pip-cache
export TMPDIR=/projects/clg_youridentifier/tmp
```

### Upgrade packaging tools
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Install the repo requirements
```bash
python -m pip install -r requirements-eval.txt
```

### Important: install extra import-time dependencies
This repo needs more than `requirements-eval.txt` for the active MCP server startup, because `mcp-server/main.py` eagerly imports all tool modules.

Install these too:

```bash
python -m pip install httpx psycopg2-binary
```

### Why these extras matter
- `finance_tools.py` imports `httpx`
- `sql_tools.py` imports `psycopg2`
- even if you are not testing finance or Postgres yet, those imports happen at startup

### Verify imports
Run:

```bash
python - <<'PY'
import mcp, openai, yaml, dotenv, scipy, numpy, httpx, psycopg2
print("core imports ok")
PY
```

Expected:
```text
core imports ok
```

### Verify the MCP server environment path works
The MCP server is launched via `uv run` with working directory `mcp-server`.

Check that path explicitly:

```bash
cd mcp-server
uv run python - <<'PY'
import mcp, httpx, psycopg2
print("uv+mcp server imports ok")
PY
cd ..
```

Expected:
```text
uv+mcp server imports ok
```

---

## 7. Ollama Setup On Eureka

### Why Ollama
This is the working, cluster-local inference path.

Why not vLLM:
- the GPUs tested are GTX 1080 / Pascal / `sm_61`
- vLLM was not practically viable on this hardware
- Ollama worked

### Install Ollama in user space
Use project-backed storage.

On the login node:

```bash
cd /projects/clg_youridentifier
mkdir -p ollama-install
mkdir -p ollama-models
mkdir -p ollama-logs
cd ollama-install
```

Download and unpack Ollama:

```bash
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama-linux-amd64.tar.zst
mkdir -p root
tar --zstd -xf ollama-linux-amd64.tar.zst -C root
```

### Set Ollama runtime environment
```bash
export OLLAMA_ROOT=/projects/clg_youridentifier/ollama-install/root
export PATH="$OLLAMA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_ROOT/lib/ollama:${LD_LIBRARY_PATH}"
export OLLAMA_MODELS=/projects/clg_youridentifier/ollama-models
export OLLAMA_NO_CLOUD=1
export OLLAMA_HOST=127.0.0.1:11434
```

### Verify the binary
```bash
command -v ollama
ollama -v
```

Expected:
- `ollama` should resolve to the project-local install
- `ollama -v` may warn if no server is running yet; that is okay

### Start a temporary login-node server just to pull models
```bash
nohup ollama serve > /projects/clg_youridentifier/ollama-logs/ollama-login.log 2>&1 &
sleep 5
curl http://127.0.0.1:11434/v1/models
```

If no models exist yet, you may see empty model data. That is okay.

### Pull recommended models
Recommended based on actual testing:

- `qwen3:8b` = best default balance
- `qwen2.5:latest` = also very good
- `gpt-oss:20b` = stronger but much slower
- `llama3.1:8b` = not recommended for native tool calling

Pull models:

```bash
ollama pull qwen3:8b
ollama pull qwen2.5:latest
ollama pull gpt-oss:20b
```

You can skip `gpt-oss:20b` if storage or time is a concern, but it’s useful as a strong baseline.

### Verify model availability
```bash
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/v1/models
```

### Stop the temporary login-node server
```bash
pkill -f "ollama serve"
```

### Important note
Inference should run on Eureka itself, not through your laptop. Do not use SSH tunneling for normal use if you want cluster-local inference.

---

## 8. GPU Allocation Workflow

### Why GPU nodes matter
Login nodes are for:
- setup
- installs
- pulling models
- light coordination

Compute/GPU nodes are for:
- model serving
- real inference
- evaluations that use the self-hosted model

### Request a GPU node
From a login node:

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gpus=1 --time=02:00:00
```

### What to expect
You’ll see something like:

```text
salloc: Granted job allocation 2019
salloc: Waiting for resource configuration
salloc: Nodes eureka3 are ready for job
```

Your prompt should then change to something like:

```bash
clg_youridentifier@eureka3:...$
```

### Confirm you are on a compute/GPU node
Run:

```bash
hostname
```

Expected:
- `eureka1`
- `eureka2`
- `eureka3`
- etc.

### Check the GPU
Run:

```bash
nvidia-smi
```

Expected:
- NVIDIA GPU shown
- for the tested nodes, usually GTX 1080

### Key distinction
- login node: do setup work
- compute node: do actual model inference work

---

## 9. Starting The Model Server On Eureka

### Set the same Ollama environment on the GPU node
```bash
export OLLAMA_ROOT=/projects/clg_youridentifier/ollama-install/root
export PATH="$OLLAMA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_ROOT/lib/ollama:${LD_LIBRARY_PATH}"
export OLLAMA_MODELS=/projects/clg_youridentifier/ollama-models
export OLLAMA_NO_CLOUD=1
export OLLAMA_HOST=127.0.0.1:11434
```

### Start `ollama serve`
```bash
nohup ollama serve > /projects/clg_youridentifier/ollama-logs/ollama-compute.log 2>&1 &
sleep 5
```

### Verify the endpoint locally on Eureka
```bash
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/v1/models
```

Expected:
- JSON listing your models

### Check current loaded model status
Before first inference, `ollama ps` may be empty:

```bash
ollama ps
```

That’s normal if no model is loaded yet.

### If the server does not respond
Check logs:

```bash
tail -100 /projects/clg_youridentifier/ollama-logs/ollama-compute.log
```

---

## 10. Running The Repo

### Activate the repo environment on the GPU node
```bash
cd /projects/clg_youridentifier/T8-CtrlAltElite
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export PIP_CACHE_DIR=/projects/clg_youridentifier/pip-cache
export TMPDIR=/projects/clg_youridentifier/tmp
```

### Export harness endpoint settings
For local Ollama on the GPU node:

```bash
export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
export OLLAMA_API_KEY="ollama"
export OLLAMA_MODEL="qwen3:8b"
```

### First real smoke test
Run a BFCL L1 oracle smoke test:

```bash
python -m harness.runner \
  --dataset bfcl \
  --model "$OLLAMA_MODEL" \
  --backend ollama \
  --base-url "$OLLAMA_BASE_URL" \
  --api-key "$OLLAMA_API_KEY" \
  --level L1 \
  --limit 5 \
  --oracle
```

### Expand to a larger smoke test
```bash
python -m harness.runner \
  --dataset bfcl \
  --model "$OLLAMA_MODEL" \
  --backend ollama \
  --base-url "$OLLAMA_BASE_URL" \
  --api-key "$OLLAMA_API_KEY" \
  --level L1 \
  --limit 15 \
  --oracle
```

### A practical working default
Based on actual tests, this is the recommended default:

```bash
export OLLAMA_MODEL="qwen3:8b"
```

---

## 11. Verifying That The Pipeline Is Truly Working

### What success looks like
A successful run should show:

- dataset/model summary
- MCP server ready
- tasks being processed
- lines like:
```text
OK  actual=add | ref=add | steps=1/1 | wos=1.00
```
- final summary with nonzero WOS
- a results JSON file written under `results/`

### Example signs of real success
You want to see:
- `MCP server ready - 58 tools available`
- actual tool names in `actual=...`
- no tool call count low or zero
- results saved message

### How to distinguish failure types

#### Repo/MCP failure
Symptoms:
- `harness.runner` crashes immediately
- MCP server does not start
- import errors
- no tools listed

Check:
- repo venv activated?
- `uv` on PATH?
- extra packages installed?
- `python -m harness.runner` being run from repo root?

#### Endpoint connectivity failure
Symptoms:
- model call errors
- connection refused
- timeouts
- `curl http://127.0.0.1:11434/v1/models` fails

Check:
- is `ollama serve` running?
- are you on the GPU node?
- is `OLLAMA_HOST`/`OLLAMA_BASE_URL` correct?

#### Model not producing native tool calls
Symptoms:
- endpoint works
- run completes
- `actual=none`
- high `no tool call`

This means the model answered, but did not return native tool calls in the expected format.

### Native tool calling vs fallback
The harness only accepts JSON-in-text fallback if you pass:

```bash
--allow-fallback
```

If you do **not** pass that flag, successful tool execution means native tool calling happened.

This was important in model testing:
- `qwen3:8b`, `qwen2.5:latest`, and `gpt-oss:20b` all showed native tool calling in successful runs
- `llama3.1:8b` did not

---

## 12. Common Pitfalls / Troubleshooting

### Pitfall: using the wrong node
If you try to run model inference on a login node, things may be slow, blocked, or inappropriate.

Use:
- login node for setup
- GPU node for model serving and evaluation

### Pitfall: using `$HOME`
Do not store:
- repo clone
- model files
- large caches
- Ollama install
in `$HOME`

Use `/projects/<your_clg_username>/...`

### Pitfall: missing shared project permissions
If you cannot enter or write to the shared project directory:
- request access from the admin/owner
- provide your `clg_...` username only

### Pitfall: missing `uv`
The active MCP path uses `uv run python ...`

If `uv` is not installed or not on PATH:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Pitfall: only installing `requirements-eval.txt`
That is not enough by itself for startup in practice.

Also install:
```bash
python -m pip install httpx psycopg2-binary
```

### Pitfall: Ollama not running
If `curl http://127.0.0.1:11434/v1/models` fails:
- Ollama is not running on that machine

Start it again:

```bash
nohup ollama serve > /projects/<your_clg_username>/ollama-logs/ollama-compute.log 2>&1 &
sleep 5
```

### Pitfall: pointing the harness at the wrong endpoint
Remember:

```text
http://127.0.0.1:11434
```

means:
- the current Eureka machine itself

It does **not** mean your laptop.

### Pitfall: confusing laptop localhost with Eureka localhost
If you run on Eureka and set `base_url=http://127.0.0.1:11434/v1`, the server must be running on **Eureka**, not on your laptop.

### Pitfall: trying to use vLLM on GTX 1080
Don’t. This path was thoroughly tested and is not practically viable on Pascal for real inference.

---

## 13. Recommended Final Workflow For Day-To-Day Usage

### What to do each session on the login node
1. SSH into Eureka
2. Go to your project directory
3. Activate your repo venv if doing setup work
4. Use the login node for:
   - repo updates
   - package installs
   - pulling new Ollama models
   - checking files/logs

Typical login-node commands:

```bash
ssh clg_youridentifier@eureka.paice-ua.com
cd /projects/clg_youridentifier/T8-CtrlAltElite
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
```

### What to do each session on the GPU node
1. Request a GPU allocation
2. Set Ollama environment
3. Start `ollama serve`
4. Verify `/v1/models`
5. Activate repo venv
6. Export harness endpoint variables
7. Run evaluations

### Normal order of operations
#### On login node
```bash
ssh clg_youridentifier@eureka.paice-ua.com
cd /projects/clg_youridentifier/T8-CtrlAltElite
```

#### Get a GPU node
```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gpus=1 --time=02:00:00
```

#### On GPU node: start Ollama
```bash
export OLLAMA_ROOT=/projects/clg_youridentifier/ollama-install/root
export PATH="$OLLAMA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$OLLAMA_ROOT/lib/ollama:${LD_LIBRARY_PATH}"
export OLLAMA_MODELS=/projects/clg_youridentifier/ollama-models
export OLLAMA_NO_CLOUD=1
export OLLAMA_HOST=127.0.0.1:11434

nohup ollama serve > /projects/clg_youridentifier/ollama-logs/ollama-compute.log 2>&1 &
sleep 5
curl http://127.0.0.1:11434/v1/models
```

#### On GPU node: run repo
```bash
cd /projects/clg_youridentifier/T8-CtrlAltElite
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export PIP_CACHE_DIR=/projects/clg_youridentifier/pip-cache
export TMPDIR=/projects/clg_youridentifier/tmp

export OLLAMA_BASE_URL="http://127.0.0.1:11434/v1"
export OLLAMA_API_KEY="ollama"
export OLLAMA_MODEL="qwen3:8b"

python -m harness.runner \
  --dataset bfcl \
  --model "$OLLAMA_MODEL" \
  --backend ollama \
  --base-url "$OLLAMA_BASE_URL" \
  --api-key "$OLLAMA_API_KEY" \
  --level L1 \
  --limit 15 \
  --oracle
```

### Recommended practical defaults
Use these model defaults:

- default day-to-day: `qwen3:8b`
- also viable: `qwen2.5:latest`
- slower stronger baseline: `gpt-oss:20b`

Avoid:
- `llama3.1:8b` for native tool-calling evals
- vLLM on Eureka GTX 1080 nodes

---

## Final takeaway

A teammate new to Eureka should think of the workflow like this:

1. Get cluster access working.
2. Work from `/projects/...`, not `$HOME`.
3. Clone the repo and use the active `harness/` path.
4. Create a Python 3.11 venv and install the real working dependency set.
5. Install `uv` because the MCP flow needs it.
6. Install Ollama in user space under `/projects/...`.
7. Pull models on the login node.
8. Allocate a GPU node.
9. Start Ollama on the GPU node.
10. Point `harness.runner` at `http://127.0.0.1:11434/v1`.
11. Use `qwen3:8b` as the default working model.

That is the actual working Eureka setup we established, end to end.

If you want, I can also turn this into a shorter markdown `EUREKA_SETUP.md` style document you can drop into the repo or share directly with teammates.