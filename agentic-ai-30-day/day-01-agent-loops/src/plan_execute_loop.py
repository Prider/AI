"""Plan-and-execute: plan once, run tool steps, then synthesize."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import llm
from .tools import run_tool, tool_catalog_prompt


@dataclass
class LoopResult:
    pattern: str
    final: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = "completed"
    plan: dict[str, Any] | None = None


PLANNER_SYSTEM = f"""You are a planner. Output JSON only:
{{
  "goal": "...",
  "steps": [
    {{"id": 1, "tool": "<name>", "args": {{...}}, "why": "..."}}
  ]
}}

Use only these tools:
{tool_catalog_prompt()}

Do not include a final narrative step; the executor will synthesize after tools run.
"""


def run(task: str, *, max_steps: int = 8) -> LoopResult:
    plan_messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": task},
    ]
    plan_raw = llm.chat(plan_messages, mode="plan_execute")
    try:
        plan = llm.parse_json_block(plan_raw)
    except Exception as exc:  # noqa: BLE001
        return LoopResult(
            "plan_execute",
            "",
            [{"error": f"bad plan JSON: {exc}", "raw": plan_raw}],
            "invalid_plan",
        )

    trace: list[dict[str, Any]] = [{"phase": "plan", "plan": plan}]
    observations: list[str] = []

    steps = plan.get("steps") or []
    for i, step in enumerate(steps):
        if i >= max_steps:
            break
        tool = step.get("tool")
        args = step.get("args") or {}
        if not tool:
            continue
        result = run_tool(str(tool), args if isinstance(args, dict) else {})
        obs = result.to_observation()
        observations.append(f"Step {step.get('id', i)} ({tool}): {obs}")
        trace.append(
            {
                "phase": "execute",
                "step": step,
                "observation": obs,
            }
        )

    synth_messages = [
        {
            "role": "system",
            "content": (
                "Synthesize a concise final answer for the user from the plan "
                "goal and tool observations. Mention approval if refund previewed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}\nGoal: {plan.get('goal')}\n"
                + "\n".join(observations)
            ),
        },
    ]
    final = llm.chat(synth_messages, mode="plan_execute").strip()
    trace.append({"phase": "synthesize", "final": final})
    return LoopResult("plan_execute", final, trace, plan=plan)
