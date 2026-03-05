# Dataset Integration Plan

## Current Status (Week 2)
**Dataset:** BFCL Math Tools  
**Tools Implemented:** 50 math/calculation tools  
**Test Coverage:** 42 test cases  
**Finding:** 5.13% F1 degradation when scaling 25→50 tools

---

## Client's Recommended Datasets

### 1. **BFCL (Gorilla)** IN USE
- **Source:** https://huggingface.co/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- **Provides:** 16k+ real-world tool-use instructions
- **Current Use:** Math tools (42 tests)
- **Next:** Implement weather, movie, database tools to get 100+ tests

### 2. **AgentBench** 🔄 NEXT
- **Source:** https://github.com/THUDM/AgentBench
- **Focus:** Tool planning, decision making, failure recovery
- **Tests:**
  - When to call a tool
  - When NOT to call a tool
  - Multi-step reasoning
- **Value:** Tests agent behavior beyond just accuracy

### 3. **PlanBench**
- **Source:** https://github.com/harshakokel/PlanBench
- **Focus:** Prompt → plan → tool calls
- **Tests:** Sequential vs parallel tool execution
- **Value:** Tests planning capability

### 4. **MCPVerse** 
- **Source:** https://arxiv.org/abs/2508.16260
- **Provides:** 550+ real-world executable tools
- **Methodology:** Oracle / Standard / Max-Scale modes
- **Value:** Industry-standard threshold testing framework

---

## Implementation Strategy

### Phase 1: Expand BFCL Coverage (Current)
**Goal:** 100+ test cases from BFCL  
**Action:** Implement top BFCL functions:
- `get_current_weather` (19 tests)
- `Movies_3_FindMovies` (18 tests)
- `cmd_controller.execute` (28 tests)
- Database operations

**Tools to add:** 30-50 more from BFCL  
**Expected:** 100+ real test cases

### Phase 2: Add AgentBench Tests
**Goal:** Test agent decision-making  
**Action:** 
- Implement AgentBench tool scenarios
- Test "when not to call" behaviors
- Measure failure recovery

### Phase 3: PlanBench Integration
**Goal:** Test multi-step planning  
**Action:**
- Implement planning scenarios
- Test DAG vs sequential execution

### Phase 4: MCPVerse Alignment
**Goal:** Align with industry standard  
**Action:**
- Structure results in Oracle/Standard/Max-Scale format
- Compare qwen2.5 vs other models

---

## Week 3 Priorities

1. **Immediate:** Add 50 more BFCL tools from weather/movies/database domains
2. **Testing:** Run 100-tool evaluation to find sharper threshold
3. **Comparison:** Test llama3.1 and mistral with same methodology
4. **Documentation:** Log threshold points for each model
