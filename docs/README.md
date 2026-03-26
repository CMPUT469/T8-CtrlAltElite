# MCP Evaluation Harness

Evaluates LLM function-calling on MCP tool servers using outcome-based scoring
(MCPVerse methodology) extended with TESR (Task Efficiency-Scaled Result).

---

## Structure

```
.
├── harness/
│   ├── runner.py            ← single entry point  (replaces evaluate_jefferson.py + evaluate_bfcl.py)
│   ├── model_client.py      ← provider adapter    (Ollama now, vLLM Eureka later — one config line)
│   ├── mcp_session.py       ← MCP server lifecycle + tool helpers
│   ├── metrics.py           ← E(O,Ô), TESR, F1, auxiliary stats  (replaces evaluation_framework.py)
│   ├── threshold_sweep.py   ← distractor sweep    (replaces incremental_threshold.py)
│   └── db_logger.py         ← Supabase logger     (active logging path)
│
├── datasets/
│   ├── jefferson_stats/
│   │   ├── tasks_l1.jsonl   single-tool, params stated directly
│   │   ├── tasks_l2.jsonl   single-tool, real-world framing
│   │   └── tasks_l3.jsonl   multi-step chains (2-4 tools)
│   ├── bfcl_math/
│   │   └── tasks_l1.jsonl   (move math_test_cases.jsonl here, rename)
│   └── postgres/
│       └── tasks_l1.jsonl   (move postgres_test_cases.jsonl here, rename)
│
├── configs/
│   └── models.yaml          model registry (Ollama + Eureka/vLLM slots)
│
├── mcp-server/              unchanged — FastMCP server + tools
│   ├── main.py
│   └── tools/
│       ├── jefferson_stats_tools.py
│       ├── bfcl_math_tools.py
│       ├── finance_tools.py
│       └── sql_tools.py
│
├── mcp-client/              legacy — not part of the active evaluation architecture
│   └── main.py
│
├── schema.sql               unchanged — Supabase DDL (run ALTER TABLE below after migration)
├── run_threshold_tests.sh   updated to call harness/threshold_sweep.py
├── .env                     SUPABASE_URL, SUPABASE_KEY, LLM_API_KEY
└── results/                 auto-created, gitignored
```

### Files removed in this refactor

| Old file | Reason |
|---|---|
| `evaluate_jefferson.py` | Absorbed into `harness/runner.py` |
| `evaluate_bfcl.py` | Absorbed into `harness/runner.py` |
| `evaluation_framework.py` | Absorbed into `harness/metrics.py` |
| `incremental_threshold.py` | Replaced by `harness/threshold_sweep.py` |
| `test_bfcl_tools.py` | Ad-hoc smoke test — use `--limit 5` on runner instead |
| `test_jefferson_stats.py` | Ad-hoc smoke test — use `--limit 5` on runner instead |
| `test_model_prompt.py` | Ad-hoc smoke test — use `--limit 5` on runner instead |
| `test_threshold.py` | Ad-hoc smoke test — use `threshold_sweep --distractors 0` instead |

### Files moved

| Old location | New location |
|---|---|
| `db_logger.py` | `harness/db_logger.py` |
| `math_test_cases.jsonl` | `datasets/bfcl_math/tasks_l1.jsonl` |
| `postgres_test_cases.jsonl` | `datasets/postgres/tasks_l1.jsonl` |
| `stats_test_cases.jsonl` | split into `datasets/jefferson_stats/tasks_l1/l2/l3.jsonl` |

---

## Quickstart

### Install

```bash
pip install -r requirements-eval.txt
```

### Run (Ollama)

```bash
# All levels, standard mode (all tools visible)
python -m harness.runner --dataset jefferson --model qwen2.5:7b

# Stage-1 Jefferson variant
python -m harness.runner --dataset jefferson_stage1 --model qwen2.5:7b

# L1 only, oracle mode (only the correct tool exposed)
python -m harness.runner --dataset jefferson --model qwen2.5:7b --level L1 --oracle

# Quick smoke test — 5 tasks
python -m harness.runner --dataset jefferson --model qwen2.5:7b --limit 5
```

### Run (vLLM on Eureka)

Once your Eureka endpoint is running (`vllm serve <model> --port 8000 --api-key <token>`):

1. Update `configs/models.yaml` with the real host and token.
2. Run exactly the same command with `--backend vllm`:

```bash
python -m harness.runner \
    --dataset jefferson \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --backend vllm \
    --base-url http://eureka-node-01:8000/v1 \
    --api-key your-token \
    --level L1 L2 L3
```

No other changes needed — the harness uses the same OpenAI-compatible client for
both Ollama and vLLM.

---

## Metrics

| Metric | Description |
|---|---|
| **Outcome Accuracy** `E(O,Ô)` | Did execution produce the correct result? Primary MCPVerse metric. |
| **TESR overall/L1/L2/L3** | Outcome × (optimal_steps / actual_steps). Penalises redundant calls. |
| Function selection rate | Did the model pick the right tool? |
| Parameter accuracy | Were the arguments correct? |
| F1 / Precision / Recall | Traditional metrics (for comparison with older results). |

### TESR interpretation

- `1.0` — correct answer, no wasted steps
- `0.67` — correct answer but one extra tool call (e.g. 2 optimal / 3 actual)
- `0.0` — wrong answer, regardless of how many calls were made

---

## Task levels

| Level | Description | `optimal_steps` |
|---|---|---|
| L1 | Single tool, parameters stated explicitly | 1 |
| L2 | Single tool, parameters must be inferred from real-world framing | 1 |
| L3 | Multi-step: model must chain 2–4 tools and reason across intermediate results | 2–4 |

### Task format (JSONL)

All tasks share a common schema — no separate ground-truth file needed.

```jsonc
{
  "id":               "l1_003",
  "level":            "L1",
  "function":         "calculate_variance",   // single function name (L1/L2)
  // "functions":     ["fn_a", "fn_b"],        // list for L3
  "query":            "Compute the variance of [2, 4, 6, 8, 10].",
  "expected_params":  {"collection": [2,4,6,8,10]},   // optional; used for aux metrics
  "expected_outcome": 8.0,                    // compared against tool execution result
  "optimal_steps":    1
}
```

For L3 tasks `expected_outcome` is a dict describing the final synthesised answer,
not a single tool return value.

---

## Adding a new dataset

1. Create `datasets/<name>/tasks_l1.jsonl` (and l2/l3 as needed).
2. Add an entry to `DATASETS` in `harness/runner.py`:
   ```python
   "mydata": {
       "tasks": {"L1": "datasets/mydata/tasks_l1.jsonl"},
       "server": "mcp-server/main.py",
   }
   ```
3. Run: `python -m harness.runner --dataset mydata --model qwen2.5:7b`
