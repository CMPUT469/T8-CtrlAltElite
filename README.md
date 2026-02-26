# CMPUT 469 — CtrlAltElite (T8)
## Developing Effective Tooling for Lightweight Open-Source LLM Agents with MCP

### Project summary
We’re studying whether **lightweight (<20B) open-source LLMs** can reliably act as **tool-orchestrating agents** when tools are exposed through the **Model Context Protocol (MCP)**. The core challenge is mapping vague natural language into strict tool calls (tool choice + JSON args), especially as the **tool list grows** and models face **tool-space interference**.

---

## Goal
Evaluate if small open-source models can:
1. **Select the correct tool(s)** for a task
2. Produce **schema-valid arguments**
3. Achieve correct **task outcomes** after tool execution

### Saturation / threshold limit
We will measure the **saturation point**: how many tools a model can handle before tool-selection accuracy noticeably degrades. We scale the MCP tool set from **N = 5 → 100** tools.

---

## Inputs / Outputs
**Input**
- Natural language prompt
- MCP server containing N tools (tool space scaled over experiments)

**Output**
- Tool call(s) chosen by the LLM + arguments
- Tool output(s)
- Final agent answer (plus trace logs for evaluation)

---

## Evaluation
We focus on accuracy using two main metrics:

- **F1 Score (Tool Selection):** Harmonic mean of precision/recall over tool calls  
  - Penalizes **over-calling** (extra unnecessary tools) and **under-calling** (missing required tools)
  - Acts as a behavioral “guardrail” against excessive agency

- **Task Finish Score (TFS):** Binary outcome metric based on final state / correctness  
  - Measures whether the objective was achieved (even if the agent took a non-standard path)

We’ll also note qualitative observations that may matter in practice: **security, privacy, memory handling, and context-window effects**.

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
- **MCP gateway-style preprocessing** to filter tools before the LLM sees them (context reduction, fewer naming collisions, reduced schema noise)
- **Schema/format alignment** and structured preprocessing to improve argument validity 
- Optional fine-tuning on MCP/tool-calling style data (only if needed)

---

## Timeline (high-level)
- **Week 1 (Feb 23–27):** Repo + MCP server setup, initial tools 
- **Week 2 (Mar 2–6):** First lightweight LLM integrated, find threshold limit 
- **Week 3 (Mar 9–13):** Evaluate additional LLMs, compare thresholds   
- **Week 4 (Mar 16–20):** Compute initial F1 + TFS; note other concerns 
- **Week 5 (Mar 23–27):** Fine-tuning (if applicable) + reruns 
- **Week 6 (Mar 30–Apr 3):** Final evaluation + results consolidation  
- **Week 7 (Apr 6–10):** Final presentation 
- **Week 8 (Apr 13–17):** Final report

---

## Deliverables
- MCP tool server + agent client implementation
- Benchmark suite (prompts + expected outcomes)
- Evaluation logs + summary results (F1, TFS, threshold limits per model)
- Final report + presentation/demo

---
