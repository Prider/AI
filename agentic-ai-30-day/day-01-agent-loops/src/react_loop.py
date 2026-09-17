"""ReAct loop: Thought → Action → Observation (or Final)."""

from __future__ import annotations

import json
import re
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


SYSTEM = f"""You are a ReAct agent. On every turn output exactly one of:

Thought: ...
Action: <tool_name>
Action Input: {{...json...}}

OR

Thought: ...
Final: <answer>

Tools:
{tool_catalog_prompt()}

Do not invent order facts. Refunds require approval language in the Final.
"""


_ACTION_RE = re.compile(
    r"Action:\s*(?P<tool>\w+)\s*Action Input:\s*(?P<input>\{[\s\S]*?\})",
    re.IGNORECASE,
)
_FINAL_RE = re.compile(r"Final:\s*(?P<final>.+)\s*$", re.IGNORECASE | re.DOTALL)


def run(task: str, *, max_steps: int = 6) -> LoopResult:
    transcript = f"Question: {task}\n"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": transcript},
    ]
    trace: list[dict[str, Any]] = []

    for i in range(max_steps):
        raw = llm.chat(messages, mode="react")
        messages.append({"role": "assistant", "content": raw})
        thought = _extract_thought(raw)

        has_action = re.search(r"Action:", raw, re.IGNORECASE) is not None
        has_final = re.search(r"Final:", raw, re.IGNORECASE) is not None
        if has_final and not has_action:
            final_m = _FINAL_RE.search(raw)
            answer = final_m.group("final").strip() if final_m else raw
            trace.append({"step": i, "thought": thought, "final": answer})
            return LoopResult("react", answer, trace)

        action_m = _ACTION_RE.search(raw)
        if not action_m:
            trace.append({"step": i, "thought": thought, "error": "parse_failure", "raw": raw})
            return LoopResult("react", "", trace, "invalid_model_output")

        tool = action_m.group("tool").strip()
        try:
            args = json.loads(action_m.group("input"))
        except json.JSONDecodeError as exc:
            trace.append({"step": i, "error": f"bad action input: {exc}", "raw": raw})
            return LoopResult("react", "", trace, "invalid_model_output")

        result = run_tool(tool, args)
        obs = result.to_observation()
        trace.append(
            {
                "step": i,
                "thought": thought,
                "tool": tool,
                "args": args,
                "observation": obs,
            }
        )
        transcript += raw + f"\nObservation: {obs}\n"
        messages.append({"role": "user", "content": transcript})

    return LoopResult("react", "", trace, "max_steps")


def _extract_thought(raw: str) -> str:
    m = re.search(r"Thought:\s*(.+?)(?:\n(?:Action|Final):|\Z)", raw, re.DOTALL)
    return m.group(1).strip() if m else ""
