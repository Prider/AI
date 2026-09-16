#!/usr/bin/env python3
"""Unit tests for the agentic memory demo."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory_agent import MemoryAgent, MemoryStore


class MemoryAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store_path = Path(self.tmp.name) / "memory_store.json"
        self.agent = MemoryAgent(MemoryStore(path=self.store_path))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_remember_and_recall(self) -> None:
        self.assertIn("Remembered", self.agent.handle("remember my name is Pawat"))
        self.assertIn("Remembered", self.agent.handle("remember preferred currency is THB"))
        reply = self.agent.handle("what do you know about me?")
        self.assertIn("Pawat", reply)
        self.assertIn("THB", reply)
        self.assertTrue(self.store_path.exists())

    def test_forget(self) -> None:
        self.agent.handle("remember preferred currency is THB")
        self.agent.handle("remember office is Bangkok")
        reply = self.agent.handle("forget currency")
        self.assertIn("Forgot", reply)
        remaining = self.agent.handle("recall")
        self.assertIn("Bangkok", remaining)
        self.assertNotIn("THB", remaining)

    def test_working_memory_goal(self) -> None:
        reply = self.agent.handle("set goal: weekly expense summary")
        self.assertIn("weekly expense summary", reply)
        status = self.agent.handle("status")
        self.assertIn("weekly expense summary", status)

    def test_duplicate_remember(self) -> None:
        self.agent.handle("remember team size is 5")
        again = self.agent.handle("remember team size is 5")
        self.assertIn("Already remembered", again)
        self.assertEqual(len(self.agent.store.facts), 1)

    def test_short_term_buffer_trims(self) -> None:
        self.agent.state.max_turns = 4
        for i in range(6):
            self.agent.handle(f"remember item {i}")
        self.assertLessEqual(len(self.agent.state.conversation), 4)


if __name__ == "__main__":
    unittest.main()
