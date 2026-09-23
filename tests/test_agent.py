"""Tests for src/agent.py: termination, schema, guards, and determinism."""

import json

import pytest

from src.agent import Agent, MockBackend, OpenAIBackend, validate_schema
from src.demo import TASK
from src.tools import CalculatorTool, FileReaderTool, MemoryTool, WebSearchTool


def make_agent(tmp_path, backend=None, max_steps=12, trajectory_path=None):
    tools = [WebSearchTool(), CalculatorTool(), FileReaderTool(), MemoryTool()]
    memory = tools[-1]
    memory.store_path = tmp_path / "notes.json"  # isolate test state
    memory.clear()
    return Agent(
        tools=tools,
        backend=backend or MockBackend(),
        max_steps=max_steps,
        trajectory_path=trajectory_path,
    )


def tool_names():
    return ["web_search", "calculator", "file_reader", "memory"]


# --- demo task ----------------------------------------------------------------
class TestDemoTask:
    def test_terminates_within_max_steps(self, tmp_path):
        agent = make_agent(tmp_path)
        result = agent.run(TASK)
        assert result["status"] == "success"
        assert len(result["steps"]) <= 12
        # 1 file read + 5 bugs x (calculator + memory note) = 11 steps
        assert len(result["steps"]) == 11

    def test_trajectory_validates_against_schema(self, tmp_path):
        agent = make_agent(tmp_path)
        result = agent.run(TASK)
        ok, errors = validate_schema(result, tool_names())
        assert ok, errors

    def test_final_answer_ranks_blocker_first(self, tmp_path):
        result = make_agent(tmp_path).run(TASK)
        first_bug_line = next(
            line
            for line in result["final_answer"].splitlines()
            if line.startswith("- BUG-")
        )
        assert "BUG-104" in first_bug_line  # the blocker, score 100

    def test_trajectory_file_written(self, tmp_path):
        path = tmp_path / "trajectory.jsonl"
        agent = make_agent(tmp_path, trajectory_path=path)
        agent.run(TASK)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1  # one run per file, one JSON object per line
        ok, errors = validate_schema(json.loads(lines[0]), tool_names())
        assert ok, errors

    def test_deterministic_across_runs(self, tmp_path):
        first = make_agent(tmp_path).run(TASK)
        second = make_agent(tmp_path).run(TASK)
        assert first == second
        assert first["run_id"] == second["run_id"]


# --- loop detection ------------------------------------------------------------
class RepeatingBackend:
    """Pathological backend: always emits the exact same action."""

    def decide(self, task, steps):
        return {
            "thought": "Searching again for the same thing.",
            "action": {"tool": "web_search", "args": {"query": "flaky tests"}},
            "final_answer": None,
        }


class TestLoopDetection:
    def test_same_action_3x_stops(self, tmp_path):
        agent = make_agent(tmp_path, backend=RepeatingBackend())
        result = agent.run(TASK)
        assert result["status"] == "loop_detected"
        # The 3rd identical call is never executed: only 2 steps recorded.
        assert len(result["steps"]) == 2
        ok, errors = validate_schema(result, tool_names())
        assert ok, errors

    def test_different_args_do_not_trigger(self, tmp_path):
        class WanderingBackend:
            def __init__(self):
                self.n = 0

            def decide(self, task, steps):
                self.n += 1
                return {
                    "thought": f"Trying expression {self.n}.",
                    "action": {
                        "tool": "calculator",
                        "args": {"expression": f"{self.n} + 1"},
                    },
                    "final_answer": None,
                }

        agent = make_agent(tmp_path, backend=WanderingBackend(), max_steps=3)
        result = agent.run(TASK)
        assert result["status"] == "max_steps_exceeded"
        assert len(result["steps"]) == 3


# --- schema validator -----------------------------------------------------------
class TestValidateSchema:
    def _valid(self, tmp_path):
        return make_agent(tmp_path).run(TASK)

    def test_rejects_bad_status(self, tmp_path):
        traj = self._valid(tmp_path)
        traj["status"] = "exploded"
        ok, errors = validate_schema(traj, tool_names())
        assert not ok and any("status" in e for e in errors)

    def test_rejects_unknown_tool(self, tmp_path):
        traj = self._valid(tmp_path)
        traj["steps"][0]["action"]["tool"] = "teleport"
        ok, errors = validate_schema(traj, tool_names())
        assert not ok and any("registered tool" in e for e in errors)

    def test_rejects_missing_field(self, tmp_path):
        traj = self._valid(tmp_path)
        del traj["final_answer"]
        ok, errors = validate_schema(traj, tool_names())
        assert not ok and any("final_answer" in e for e in errors)

    def test_rejects_non_dict_args(self, tmp_path):
        traj = self._valid(tmp_path)
        traj["steps"][1]["action"]["args"] = "2+2"
        ok, errors = validate_schema(traj, tool_names())
        assert not ok and any("args" in e for e in errors)

    def test_rejects_non_object(self):
        ok, errors = validate_schema(["not", "a", "dict"], tool_names())
        assert not ok and errors


# --- OpenAI backend --------------------------------------------------------------
def test_openai_backend_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIBackend()
