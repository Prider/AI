# Agentic AI and Memory

This section explains how **agentic AI systems** use different kinds of **memory**, and includes a small offline demo you can run with Python 3 (no API key required).

## What is agentic AI?

An **agent** is software that can pursue a goal over multiple steps. Typical loop:

1. **Perceive** — read the user message and current state  
2. **Recall** — pull relevant short-term and long-term memory  
3. **Plan** — decide the next action (or tool call)  
4. **Act** — execute that action  
5. **Remember** — store useful outcomes for later turns  

Without memory, each turn starts from scratch. With memory, the agent can stay consistent, learn preferences, and reuse past results.

## Memory layers (practical model)

| Layer | Lifetime | Role | Example |
| --- | --- | --- | --- |
| **Working memory** | Current task | Goal, plan, scratch notes | “Book a flight to Chiang Mai” |
| **Short-term / conversation** | Session | Recent dialogue turns | Last 10 messages |
| **Episodic** | Across sessions | What happened before | “User asked about invoices last Monday” |
| **Semantic / long-term** | Durable | Facts, prefs, policies | “Budget currency is THB” |
| **Procedural** | Durable | How to do things | Tool playbooks, SOPs |

In production systems, semantic memory is often implemented with **embeddings + a vector index** (RAG). This demo uses a simple keyword store so it runs offline and stays easy to read.

## Why agents need memory

- **Continuity** — remember the user’s name, constraints, and prior answers  
- **Planning** — keep the current goal and remaining steps  
- **Learning** — store corrected facts so mistakes are not repeated  
- **Grounding** — retrieve policies or docs instead of inventing them  

## Demo: `memory_agent.py`

A tiny agent that:

- keeps a **conversation buffer** (short-term)  
- tracks the **current goal** (working memory)  
- stores **facts** in `memory_store.json` (long-term)  
- can **remember**, **recall**, and **forget** facts via natural-language commands  

### Run

```bash
cd agentic-ai-memory
python3 memory_agent.py
```

### Try

```text
you> remember my name is Pawat
you> remember preferred currency is THB
you> what do you know about me?
you> set goal: draft a weekly expense summary
you> status
you> forget preferred currency
you> quit
```

### Tests

```bash
cd agentic-ai-memory
python3 -m unittest test_memory_agent.py -v
```

## How this maps to real LLM agents

| Demo piece | Production equivalent |
| --- | --- |
| Conversation list | Context window / chat history summarizer |
| `MemoryStore` keyword match | Embedding model + vector DB (RAG) |
| `set goal` | Planner / task graph / agent scratchpad |
| `remember` / `forget` | Explicit memory write tools |
| Agent loop | Orchestrator (ReAct, tool-calling, multi-agent) |

## Design tips

1. **Separate layers** — do not dump everything into one prompt forever.  
2. **Write selectively** — store facts and outcomes, not every token.  
3. **Retrieve with purpose** — pull only memories relevant to the current goal.  
4. **Allow forget / correct** — memory must be editable or it becomes stale.  
5. **Prefer grounded retrieval** for policies and numbers (RAG over docs).  

## Further reading (concepts)

- ReAct-style agents (reason + act loops)  
- RAG: retrieve → augment prompt → generate  
- Vector indexes for approximate nearest-neighbor search over embeddings  
- Session summarization for long chats  

---

*Part of the [AI](../README.md) learning repository.*
