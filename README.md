# CMPUT 469 — CtrlAltElite (T8)
## Developing Effective Tooling for Lightweight Open-Source LLM Agents with MCP

### Project summary
We’re studying whether **lightweight (<20B) open-source LLMs** can reliably act as **tool-orchestrating agents** when tools are exposed through the **Model Context Protocol (MCP)**. The core challenge is mapping vague natural language into strict tool calls (tool choice + JSON args), especially as the **tool list grows** and models face **tool-space interference**. :contentReference[oaicite:0]{index=0}

---

## Goal
Evaluate if small open-source models can:
1. **Select the correct tool(s)** for a task
2. Produce **schema-valid arguments**
3. Achieve correct **task outcomes** after tool execution :contentReference[oaicite:1]{index=1}

### Saturation / threshold limit
We will measure the **saturation point**: how many tools a model can handle before tool-selection accuracy noticeably degrades. We scale the MCP tool set from **N = 5 → 100** tools. :contentReference[oaicite:2]{index=2}

---

## Inputs / Outputs
**Input**
- Natural language prompt
- MCP server containing N tools (tool space scaled over experiments) :contentReference[oaicite:3]{index=3}

**Output**
- Tool call(s) chosen by the LLM + arguments
- Tool output(s)
- Final agent answer (plus trace logs for evaluation) :contentReference[oaicite:4]{index=4}

---

## Evaluation
We focus on accuracy using two main metrics:

- **F1 Score (Tool Selection):** Harmonic mean of precision/recall over tool calls  
  - Penalizes **over-calling** (extra unnecessary tools) and **under-calling** (missing required tools)
  - Acts as a behavioral “guardrail” against excessive agency :contentReference[oaicite:5]{index=5}

- **Task Finish Score (TFS):** Binary outcome metric based on final state / correctness  
  - Measures whether the objective was achieved (even if the agent took a non-standard path) :contentReference[oaicite:6]{index=6}

We’ll also note qualitative observations that may matter in practice: **security, privacy, memory handling, and context-window effects**. :contentReference[oaicite:7]{index=7}

---

## Approach / System plan
### Phase 1 — Baseline agent + MCP tool server
- Stand up an MCP server with a small clean tool set
- Implement the agent loop: prompt → select tool(s) → call → observe → final answer
- Log traces: tool names, arguments, outputs, and final responses

### Phase 2 — Multi-model comparison + scaling tool space
- Run the same tool suite across multiple lightweight open-source models
- Scale tool space (5 → 100) to find each model’s threshold limit

### Phase 3 — If saturation happens early (optional improvement path)
If models saturate early, we’ll explore:
- **MCP gateway-style preprocessing** to filter tools before the LLM sees them (context reduction, fewer naming collisions, reduced schema noise) :contentReference[oaicite:8]{index=8}
- **Schema/format alignment** and structured preprocessing to improve argument validity :contentReference[oaicite:9]{index=9}
- Optional fine-tuning on MCP/tool-calling style data (only if needed) :contentReference[oaicite:10]{index=10}

---

## Timeline (high-level)
- **Week 1 (Feb 23–27):** Repo + MCP server setup, initial tools :contentReference[oaicite:11]{index=11}  
- **Week 2 (Mar 2–6):** First lightweight LLM integrated, find threshold limit :contentReference[oaicite:12]{index=12}  
- **Week 3 (Mar 9–13):** Evaluate additional LLMs, compare thresholds :contentReference[oaicite:13]{index=13}  
- **Week 4 (Mar 16–20):** Compute initial F1 + TFS; note other concerns :contentReference[oaicite:14]{index=14}  
- **Week 5 (Mar 23–27):** Fine-tuning (if applicable) + reruns :contentReference[oaicite:15]{index=15}  
- **Week 6 (Mar 30–Apr 3):** Final evaluation + results consolidation :contentReference[oaicite:16]{index=16}  
- **Week 7 (Apr 6–10):** Final presentation :contentReference[oaicite:17]{index=17}  
- **Week 8 (Apr 13–17):** Final report :contentReference[oaicite:18]{index=18}  

---

## Deliverables
- MCP tool server + agent client implementation
- Benchmark suite (prompts + expected outcomes)
- Evaluation logs + summary results (F1, TFS, threshold limits per model)
- Final report + presentation/demo

---

## Repo structure (suggested)
