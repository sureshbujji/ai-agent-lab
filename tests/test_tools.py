"""Tests for src/tools.py: each tool gets a happy path and an edge case."""

import json

import pytest

from src.tools import CalculatorTool, FileReaderTool, MemoryTool, WebSearchTool


# --- WebSearchTool -----------------------------------------------------------
class TestWebSearchTool:
    def test_happy_path_returns_top_match(self):
        out = WebSearchTool().run(query="flaky tests")
        assert "Flaky test quarantine" in out

    def test_ranking_is_deterministic(self):
        tool = WebSearchTool()
        assert tool.run(query="testing") == tool.run(query="testing")

    def test_no_match_returns_message(self):
        out = WebSearchTool().run(query="zzzzqqq-no-such-topic")
        assert "No results" in out

    def test_empty_query(self):
        assert "empty query" in WebSearchTool().run(query="  ")


# --- CalculatorTool ----------------------------------------------------------
class TestCalculatorTool:
    def test_happy_path(self):
        assert CalculatorTool().run(expression="2 + 3 * 4") == "Result: 14"

    def test_parentheses_and_power(self):
        assert CalculatorTool().run(expression="2 * (3 + 4) ** 2") == "Result: 98"

    def test_rejects_import(self):
        with pytest.raises(ValueError):
            CalculatorTool().run(expression="__import__('os').system('id')")

    def test_rejects_names_and_calls(self):
        with pytest.raises(ValueError):
            CalculatorTool().run(expression="open('data/bugs.json').read()")

    def test_rejects_syntax_errors(self):
        with pytest.raises(ValueError):
            CalculatorTool().run(expression="2 +")


# --- FileReaderTool ----------------------------------------------------------
class TestFileReaderTool:
    def test_happy_path_reads_bugs_file(self):
        content = FileReaderTool().run(path="bugs.json")
        bugs = json.loads(content)
        assert len(bugs) == 5
        assert bugs[0]["id"] == "BUG-101"

    def test_blocks_path_traversal(self):
        with pytest.raises(ValueError, match="outside the data directory"):
            FileReaderTool().run(path="../README.md")

    def test_blocks_absolute_path(self):
        with pytest.raises(ValueError, match="outside the data directory"):
            FileReaderTool().run(path="/etc/passwd")

    def test_missing_file(self):
        with pytest.raises(ValueError, match="File not found"):
            FileReaderTool().run(path="does-not-exist.json")


# --- MemoryTool --------------------------------------------------------------
class TestMemoryTool:
    def test_round_trip(self, tmp_path):
        mem = MemoryTool(store_path=tmp_path / "notes.json")
        assert mem.run(operation="write_note", key="k1", text="hello") == (
            "Note saved under key 'k1'."
        )
        assert mem.run(operation="read_note", key="k1") == "hello"
        assert json.loads(mem.run(operation="list_notes")) == ["k1"]

    def test_notes_survive_across_instances(self, tmp_path):
        store = tmp_path / "notes.json"
        MemoryTool(store_path=store).run(operation="write_note", key="k", text="v")
        assert MemoryTool(store_path=store).run(operation="read_note", key="k") == "v"

    def test_read_missing_key_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No note found"):
            MemoryTool(store_path=tmp_path / "notes.json").run(
                operation="read_note", key="missing"
            )

    def test_unknown_operation_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown operation"):
            MemoryTool(store_path=tmp_path / "notes.json").run(operation="delete_all")

    def test_write_without_key_raises(self, tmp_path):
        with pytest.raises(ValueError, match="requires 'key'"):
            MemoryTool(store_path=tmp_path / "notes.json").run(
                operation="write_note", text="no key"
            )
