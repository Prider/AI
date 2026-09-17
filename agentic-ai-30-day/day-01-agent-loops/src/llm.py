"""LLM backend: deterministic mock (default) or optional OpenAI."""

from __future__ import annotations

import json
import os
import re
from typing import Any


def use_real_llm() -> bool:
    return os.getenv("USE_REAL_LLM", "").strip() in {"1", "true", "True"} and bool(
        os.getenv("OPENAI_API_KEY")
    )


def chat(messages: list[dict[str, str]], *, mode: str) -> str:
    """Return assistant text for the given loop mode."""
    if use_real_llm():
        return _openai_chat(messages)
    return MockLLM(mode).complete(messages)


def _openai_chat(messages: list[dict[str, str]]) -> str:
    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return resp.choices[0].message.content or ""


class MockLLM:
    """Scripted policy good enough to exercise each loop on the demo task."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def complete(self, messages: list[dict[str, str]]) -> str:
        user_blob = "\n".join(m["content"] for m in messages if m["role"] != "system")
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        if self.mode == "tool_calling":
            return self._tool_calling(user_blob)
        if self.mode == "react":
            return self._react(user_blob)
        if self.mode == "plan_execute":
            return self._plan_execute(system, user_blob)
        raise ValueError(f"unknown mode: {self.mode}")

    def _tool_calling(self, blob: str) -> str:
        if "preview_refund_usd" in blob:
            return json.dumps(
                {
                    "final": (
                        "Order B200 is processing (eta 5 days, $19). "
                        "Refund preview: $19 for reason 'duplicate charge' "
                        "(requires approval before execution)."
                    )
                }
            )
        if '"status"' in blob and "lookup_order" in blob:
            return json.dumps(
                {
                    "tool": "refund_preview",
                    "args": {"order_id": "B200", "reason": "duplicate charge"},
                }
            )
        return json.dumps({"tool": "lookup_order", "args": {"order_id": "B200"}})

    def _react(self, blob: str) -> str:
        if "preview_refund_usd" in blob:
            return (
                "Thought: I have order status and a refund preview; I can answer.\n"
                "Final: Order B200 is processing (eta 5 days, $19). "
                "Refund preview $19 for 'duplicate charge' requires approval."
            )
        if "Observation:" in blob and '"status"' in blob:
            return (
                "Thought: Order found; preview refund for duplicate charge.\n"
                "Action: refund_preview\n"
                'Action Input: {"order_id":"B200","reason":"duplicate charge"}'
            )
        return (
            "Thought: I need order B200 details before discussing a refund.\n"
            "Action: lookup_order\n"
            'Action Input: {"order_id":"B200"}'
        )

    def _plan_execute(self, system: str, blob: str) -> str:
        if "You are a planner" in system:
            return json.dumps(
                {
                    "goal": "Explain B200 status and preview refund for duplicate charge",
                    "steps": [
                        {
                            "id": 1,
                            "tool": "lookup_order",
                            "args": {"order_id": "B200"},
                            "why": "Need current status and total",
                        },
                        {
                            "id": 2,
                            "tool": "refund_preview",
                            "args": {
                                "order_id": "B200",
                                "reason": "duplicate charge",
                            },
                            "why": "Produce approval-gated refund preview",
                        },
                    ],
                }
            )
        if "synthesize" in system.lower():
            return (
                "Order B200 is processing (eta 5 days, total $19). "
                "Refund preview: $19 USD for reason 'duplicate charge'. "
                "Execution requires approval."
            )
        return blob


def parse_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))
