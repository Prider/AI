"""Compare tool-calling, ReAct, and plan-and-execute on the same task."""

from __future__ import annotations

import json
import sys

from . import plan_execute_loop, react_loop, tool_calling_loop

DEMO_TASK = (
    "Customer asks: What is the status of order B200, and can you preview a "
    "refund because of a duplicate charge?"
)


def _print_result(title: str, result: object) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    payload = {
        "pattern": getattr(result, "pattern"),
        "stopped_reason": getattr(result, "stopped_reason"),
        "final": getattr(result, "final"),
        "steps": getattr(result, "steps"),
    }
    plan = getattr(result, "plan", None)
    if plan is not None:
        payload["plan"] = plan
    print(json.dumps(payload, indent=2))


def main() -> int:
    print("Day 1 — Agent loop comparison (mock LLM by default)")
    print(f"Task: {DEMO_TASK}")

    tc = tool_calling_loop.run(DEMO_TASK)
    _print_result("1) Tool-calling loop", tc)

    react = react_loop.run(DEMO_TASK)
    _print_result("2) ReAct loop", react)

    pe = plan_execute_loop.run(DEMO_TASK)
    _print_result("3) Plan-and-execute loop", pe)

    print("\n" + "-" * 72)
    print("Debrief prompts")
    print("-" * 72)
    print("• Where does each pattern put control: model text vs runtime code?")
    print("• Which trace would you rather debug on-call at 2am?")
    print("• Where would you insert verify / HITL before refund_preview?")
    print("• Fill LOOP_SKETCH.md and FAILURE_MODES.md next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
