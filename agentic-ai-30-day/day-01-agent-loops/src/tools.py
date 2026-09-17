"""Shared tool registry for Day 1 agent loops.

Tools are deliberately small and imperfect so loops can fail and recover.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    ok: bool
    data: Any
    error: str | None = None

    def to_observation(self) -> str:
        if self.ok:
            return json.dumps({"ok": True, "data": self.data}, ensure_ascii=True)
        return json.dumps({"ok": False, "error": self.error}, ensure_ascii=True)


ToolFn = Callable[[dict[str, Any]], ToolResult]


def _calc(args: dict[str, Any]) -> ToolResult:
    expr = str(args.get("expression", "")).strip()
    if not expr:
        return ToolResult(False, None, "missing expression")
    # Extremely small safe subset: digits, + - * / ( ) . and spaces
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return ToolResult(False, None, "unsupported characters in expression")
    try:
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — constrained above
    except Exception as exc:  # noqa: BLE001 — demo surface
        return ToolResult(False, None, f"calc error: {exc}")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ToolResult(False, None, "expression did not evaluate to a number")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ToolResult(False, None, "non-finite result")
    return ToolResult(True, {"value": value})


def _lookup_order(args: dict[str, Any]) -> ToolResult:
    order_id = str(args.get("order_id", "")).strip().upper()
    catalog = {
        "A100": {"status": "shipped", "eta_days": 2, "total_usd": 42.5},
        "B200": {"status": "processing", "eta_days": 5, "total_usd": 19.0},
        "C300": {"status": "cancelled", "eta_days": None, "total_usd": 0.0},
    }
    if order_id not in catalog:
        return ToolResult(False, None, f"unknown order_id: {order_id}")
    return ToolResult(True, {"order_id": order_id, **catalog[order_id]})


def _refund_preview(args: dict[str, Any]) -> ToolResult:
    """Side-effect-shaped tool that only previews — still treat as sensitive."""
    order_id = str(args.get("order_id", "")).strip().upper()
    reason = str(args.get("reason", "")).strip()
    if not order_id or not reason:
        return ToolResult(False, None, "order_id and reason are required")
    looked = _lookup_order({"order_id": order_id})
    if not looked.ok:
        return looked
    total = float(looked.data["total_usd"])
    return ToolResult(
        True,
        {
            "order_id": order_id,
            "preview_refund_usd": total,
            "reason": reason,
            "requires_approval": True,
        },
    )


TOOLS: dict[str, ToolFn] = {
    "calculator": _calc,
    "lookup_order": _lookup_order,
    "refund_preview": _refund_preview,
}

TOOL_SPECS = {
    "calculator": {
        "description": "Evaluate a basic arithmetic expression.",
        "parameters": {"expression": "string, e.g. '(19.0 + 42.5) * 1.1'"},
    },
    "lookup_order": {
        "description": "Fetch order status by id (A100, B200, C300).",
        "parameters": {"order_id": "string"},
    },
    "refund_preview": {
        "description": "Preview a refund for an order (does not execute payment).",
        "parameters": {"order_id": "string", "reason": "string"},
    },
}


def run_tool(name: str, args: dict[str, Any]) -> ToolResult:
    fn = TOOLS.get(name)
    if fn is None:
        return ToolResult(False, None, f"unknown tool: {name}")
    if not isinstance(args, dict):
        return ToolResult(False, None, "args must be an object")
    return fn(args)


def tool_catalog_prompt() -> str:
    return json.dumps(TOOL_SPECS, indent=2)
