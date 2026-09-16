#!/usr/bin/env python3
"""Offline demo: a tiny agentic loop with short-term, working, and long-term memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_STORE = Path(__file__).resolve().parent / "memory_store.json"


@dataclass
class MemoryStore:
    """Long-term semantic-ish memory (keyword retrieval stand-in for a vector index)."""

    path: Path = DEFAULT_STORE
    facts: List[str] = field(default_factory=list)

    def load(self) -> None:
        if not self.path.exists():
            self.facts = []
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.facts = list(data.get("facts", []))

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"facts": self.facts}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def remember(self, fact: str) -> str:
        cleaned = " ".join(fact.strip().split())
        if not cleaned:
            return "Nothing to remember."
        lower = cleaned.lower()
        if any(existing.lower() == lower for existing in self.facts):
            return f"Already remembered: {cleaned}"
        self.facts.append(cleaned)
        self.save()
        return f"Remembered: {cleaned}"

    def forget(self, query: str) -> str:
        q = query.strip().lower()
        if not q:
            return "Tell me what to forget."
        kept: List[str] = []
        removed: List[str] = []
        for fact in self.facts:
            if q in fact.lower():
                removed.append(fact)
            else:
                kept.append(fact)
        if not removed:
            return f"No long-term memory matched “{query.strip()}”."
        self.facts = kept
        self.save()
        return "Forgot: " + "; ".join(removed)

    def recall(self, query: str = "", limit: int = 5) -> List[str]:
        if not self.facts:
            return []
        if not query.strip():
            return self.facts[:limit]
        tokens = [t for t in re.split(r"\W+", query.lower()) if t]
        scored = []
        for fact in self.facts:
            hay = fact.lower()
            score = sum(1 for t in tokens if t in hay)
            if score:
                scored.append((score, fact))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [fact for _, fact in scored[:limit]]


@dataclass
class AgentState:
    """Working + short-term memory for the current session."""

    goal: Optional[str] = None
    conversation: List[str] = field(default_factory=list)
    max_turns: int = 12

    def add_turn(self, role: str, text: str) -> None:
        self.conversation.append(f"{role}: {text}")
        overflow = len(self.conversation) - self.max_turns
        if overflow > 0:
            del self.conversation[:overflow]


class MemoryAgent:
    """Perceive → recall → act → remember."""

    def __init__(self, store: Optional[MemoryStore] = None) -> None:
        self.store = store or MemoryStore()
        self.store.load()
        self.state = AgentState()

    def handle(self, user_text: str) -> str:
        text = user_text.strip()
        self.state.add_turn("user", text)
        reply = self._act(text)
        self.state.add_turn("agent", reply)
        return reply

    def _act(self, text: str) -> str:
        lower = text.lower()

        if lower in {"quit", "exit", "q"}:
            return "Goodbye."

        if lower in {"help", "?"}:
            return (
                "Commands: remember <fact> | recall [query] | forget <query> | "
                "set goal: <goal> | status | what do you know | quit"
            )

        m = re.match(r"^(?:remember|save)\s+(.+)$", text, re.I)
        if m:
            return self.store.remember(m.group(1))

        m = re.match(r"^(?:forget|delete)\s+(.+)$", text, re.I)
        if m:
            return self.store.forget(m.group(1))

        m = re.match(r"^(?:recall|search)\s*(.*)$", text, re.I)
        if m and (m.group(1) is not None) and (
            lower.startswith("recall") or lower.startswith("search")
        ):
            return self._format_facts(self.store.recall(m.group(1)))

        m = re.match(r"^(?:set\s+goal|goal)\s*:?\s*(.+)$", text, re.I)
        if m:
            self.state.goal = m.group(1).strip()
            return f"Working memory goal set: {self.state.goal}"

        if lower in {"status", "state"}:
            return self._status()

        if any(
            phrase in lower
            for phrase in (
                "what do you know",
                "what do you remember",
                "about me",
                "my facts",
            )
        ):
            # Broad introspective asks should surface the full long-term store.
            return self._format_facts(self.store.recall("", limit=50))

        # Default: retrieve related long-term memory and answer with context.
        related = self.store.recall(text, limit=3)
        if related:
            joined = "; ".join(related)
            goal_bit = f" Current goal: {self.state.goal}." if self.state.goal else ""
            return (
                f"I don't have tools beyond memory in this demo, but related facts: "
                f"{joined}.{goal_bit}"
            )

        if self.state.goal:
            return (
                f"No matching long-term facts. Working on goal “{self.state.goal}”. "
                "Try: remember <fact> or recall."
            )
        return "No matching memory yet. Try: remember <fact>, set goal: <goal>, or help."

    def _status(self) -> str:
        recent = self.state.conversation[-4:] or ["(empty)"]
        facts = self.store.facts or ["(none)"]
        return (
            f"Goal: {self.state.goal or '(none)'}\n"
            f"Short-term turns: {len(self.state.conversation)}\n"
            f"Recent:\n  - " + "\n  - ".join(recent) + "\n"
            f"Long-term facts ({len(self.store.facts)}):\n  - "
            + "\n  - ".join(facts)
        )

    @staticmethod
    def _format_facts(facts: Iterable[str]) -> str:
        items = list(facts)
        if not items:
            return "I don't have any matching long-term memories yet."
        return "Long-term memory:\n- " + "\n- ".join(items)


def main() -> None:
    agent = MemoryAgent()
    print("Agentic AI memory demo (type 'help' or 'quit')")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user:
            continue
        reply = agent.handle(user)
        print(f"agent> {reply}")
        if user.lower() in {"quit", "exit", "q"}:
            break


if __name__ == "__main__":
    main()
