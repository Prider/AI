# Day 1 — Failure modes (observe → reason → act → verify)

For each step: name a failure, how you’d detect it, and how you’d mitigate it in production.
Fill the blank rows for **your** capstone domain.

## Observe

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Tool result truncated / summarized away key IDs | Schema check: required fields missing after observe | Keep raw tool payloads in scratchpad; summarize only for LLM context |
| Stale state after concurrent updates | Version / etag mismatch on resources | Optimistic locking; re-fetch before mutate |
| _Your domain:_ | | |

## Reason

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Infinite tool loop (same call repeated) | Trace: identical tool+args N times | Dedupe window; force stop or escalate after N |
| Planner invents steps that tools can’t do | Plan validator vs tool registry | Constrain plan language to registered tool names |
| _Your domain:_ | | |

## Act

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Invalid tool arguments | JSON Schema validation before execute | Repair once, then fail closed |
| Side effect without authz | Policy engine deny | Least-privilege scopes; HITL for irreversible acts |
| Partial tool success (timeout after write) | Idempotency keys; reconcile | Design tools idempotent; verify-after-write |
| _Your domain:_ | | |

## Verify

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Final answer cites non-existent tool evidence | Require citation IDs from observation log | Critic step; reject ungrounded answers |
| Critic rubber-stamps bad output | Spot-check LLM judge vs human labels | Calibrate judges; prefer deterministic checks first |
| _Your domain:_ | | |

## Loop-level (cross-cutting)

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Budget blowup ($ / latency) | Per-run cost+latency metrics | Model routing; early exit; hard caps |
| Prompt injection via tool/RAG content | Canary / policy on untrusted strings | Delimit untrusted data; never follow instructions in tool output |
| _Your domain:_ | | |
