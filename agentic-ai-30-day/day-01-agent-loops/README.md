# Day 1 — Agent Loops vs Chat

**Goal:** Internalize that production agents are **control loops**, not chat sessions. Implement and compare three patterns, then document failure modes for each step of `observe → reason → act → verify`.

**Time:** 3–5 hours  
**Outcome:** Runnable loop comparison + your reference loop sketch + failure-mode sheet for the capstone.

---

## Concepts (45 min)

### Chat is not an agent
A chat completion is a single turn: prompt in, text out. An **agent** repeatedly:

1. **Observe** — task, state, last tool results, errors  
2. **Reason** — decide next action (or stop)  
3. **Act** — call a tool or produce a final answer  
4. **Verify** — schema checks, policy, critics (often missing in demos)

### Three production-relevant loops

| Pattern | Control flow | Strengths | Weaknesses |
|---------|--------------|-----------|------------|
| **Tool-calling** | Model emits structured tool calls; runtime executes; results appended | Simple, matches provider APIs | Easy to loop forever; weak long-horizon planning |
| **ReAct** | Explicit Thought → Action → Observation steps | Debuggable traces; interleaves reasoning | Verbose; Thought text is not reliable control |
| **Plan-and-execute** | Plan once (or periodically), then execute steps | Better for multi-step tasks; separable planner/executor models | Stale plans; plan–world drift; higher latency |

Production systems often **compose** these: a state machine with tool-calling inside nodes, optional planner, mandatory verify/HITL on side effects.

---

## Lab (2–3 hrs)

```bash
cd agentic-ai-30-day/day-01-agent-loops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.compare
```

The demo uses a **deterministic mock LLM** (no API key). Optional: set `OPENAI_API_KEY` and `USE_REAL_LLM=1` to try the same loops against a real model (`src/llm.py`).

### Tasks

1. Run `python -m src.compare` and read the three traces.  
2. Break tools on purpose (edit `src/tools.py` to return errors / wrong shapes) and re-run — note which loop recovers.  
3. Fill in [LOOP_SKETCH.md](./LOOP_SKETCH.md) for your capstone domain.  
4. Complete [FAILURE_MODES.md](./FAILURE_MODES.md) — at least one concrete failure per step.  
5. (Stretch) Add a hard max of N tool calls and an explicit `verify` step before any side-effecting tool.

---

## Definition of done

- [ ] You can explain when you’d pick each of the three patterns  
- [ ] `compare.py` runs and you understand each trace  
- [ ] Loop sketch filled for your domain  
- [ ] Failure modes filled for observe / reason / act / verify  

**Tomorrow (Day 2):** Model the same agent as an explicit state machine / graph.
