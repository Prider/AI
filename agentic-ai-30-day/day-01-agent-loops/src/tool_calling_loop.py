"""Tool-calling loop: model emits JSON tool calls; runtime executes."""

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


SYSTEM = f"""You are a production support agent. Use tools via a single JSON object per turn:
{{"tool": "<name>", "args": {{...}}}}
or finish with:
{{"final": "<answer>"}}

Available tools:
{tool_catalog_prompt()}

Rules:
- Never invent order data; call lookup_order.
- refund_preview only previews; say approval is required.
- Respond with JSON only.
"""


def run(task: str, *, max_steps: int = 6) -> LoopResult:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]
    trace: list[dict[str, Any]] = []

    for i in range(max_steps):
        raw = llm.chat(messages, mode="tool_calling")
        messages.append({"role": "assistant", "content": raw})
        try:
            decision = llm.parse_json_block(raw)
        except Exception as exc:  # noqa: BLE001
            trace.append({"step": i, "error": f"invalid JSON: {exc}", "raw": raw})
            return LoopResult("tool_calling", "", trace, "invalid_model_output")

        if "final" in decision:
            trace.append({"step": i, "final": decision["final"]})
            return LoopResult("tool_calling", str(decision["final"]), trace)

        name = decision.get("tool")
        args = decision.get("args") or {}
        result = run_tool(str(name), args if isinstance(args, dict) else {})
        obs = result.to_observation()
        trace.append(
            {
                "step": i,
                "tool": name,
                "args": args,
                "observation": obs,
                "phase": "act",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": f"Observation from {name}: {obs}",
            }
        )

    return LoopResult("tool_calling", "", trace, "max_steps")
