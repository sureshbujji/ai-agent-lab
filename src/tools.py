"""Tool implementations for the ai-agent-lab ReAct agent.

Every tool is a class exposing:
    name         - the exact string the agent uses in ``Action: <tool>``
    description  - human/LLM readable summary of what the tool does
    args_schema  - dict of arg name -> type/description (documentation contract)
    run(**kwargs) - executes the tool; returns a string observation

Tool misuse raises ``ValueError``; the agent loop catches these and feeds them
back as ``Observation: Error: ...`` lines so the agent can recover.
"""

import ast
import json
import operator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class BaseTool:
    """Shared interface. Subclasses set name/description/args_schema and run()."""

    name = ""
    description = ""
    args_schema = {}

    def run(self, **kwargs):  # pragma: no cover - overridden
        raise NotImplementedError


class WebSearchTool(BaseTool):
    """Mock web search over a small built-in QA knowledge base. Deterministic."""

    name = "web_search"
    description = (
        "Search a built-in knowledge base of QA engineering notes. "
        "Input: query (string). Returns the top matching notes, best match first."
    )
    args_schema = {"query": "string - search keywords, e.g. 'flaky tests'"}

    KNOWLEDGE_BASE = [
        {
            "title": "Flaky test quarantine",
            "content": "Quarantine flaky tests in a separate suite so they don't block CI, "
            "then fix or delete them within one sprint. Track flake rate per test to prioritize.",
        },
        {
            "title": "Visual regression testing",
            "content": "Compare screenshots against approved baselines on every PR. Keep baselines "
            "versioned and review diffs in the same tool engineers use for code review.",
        },
        {
            "title": "Contract testing",
            "content": "Consumer-driven contract tests catch API breakages without spinning up the full "
            "system. Each consumer publishes its expectations; providers verify on every build.",
        },
        {
            "title": "Test pyramid",
            "content": "Most tests should be fast unit tests, fewer integration tests, and a thin layer "
            "of end-to-end tests. Invert the pyramid and your suite becomes slow and brittle.",
        },
        {
            "title": "Mutation testing",
            "content": "Seed small faults into the code and check that the tests fail. Surviving mutants "
            "point to weak assertions, not just missing coverage.",
        },
        {
            "title": "Shift-left testing",
            "content": "Move quality checks earlier: lint, type checks, and unit tests on every commit, "
            "plus threat modeling during design reviews.",
        },
        {
            "title": "Chaos engineering for QA",
            "content": "Inject latency and failures into staging to verify retries, timeouts, and circuit "
            "breakers before production does it for you.",
        },
        {
            "title": "AI test generation",
            "content": "LLM-generated tests are great for boilerplate and edge cases, but every generated "
            "test needs a human to confirm the oracle - the expected result.",
        },
        {
            "title": "Accessibility testing",
            "content": "Automate axe-core scans in CI and pair them with manual keyboard and "
            "screen-reader passes. Automation catches roughly 30% of accessibility issues.",
        },
        {
            "title": "Performance budgets",
            "content": "Set budgets for page load, API p95 latency, and bundle size in CI. A build that "
            "busts the budget fails like any broken test.",
        },
    ]

    def run(self, query):
        query = str(query or "").strip()
        if not query:
            return "No results: empty query."
        tokens = query.lower().split()
        scored = []
        for entry in self.KNOWLEDGE_BASE:
            haystack = (entry["title"] + " " + entry["content"]).lower()
            score = sum(haystack.count(tok) for tok in tokens)
            if score:
                scored.append((score, entry))
        # Deterministic ordering: score desc, then title asc (no randomness).
        scored.sort(key=lambda item: (-item[0], item[1]["title"]))
        if not scored:
            return f"No results for '{query}'."
        lines = [f"- {entry['title']}: {entry['content']}" for _, entry in scored[:3]]
        return "Top matches:\n" + "\n".join(lines)


class CalculatorTool(BaseTool):
    """Safe arithmetic evaluator. Parses with ``ast`` - never eval() on raw input."""

    name = "calculator"
    description = (
        "Evaluate a basic arithmetic expression safely (+, -, *, /, //, %, **, parentheses). "
        "Input: expression (string). No variables, functions, or imports allowed."
    )
    args_schema = {"expression": "string - e.g. '2 * (3 + 4)'"}

    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def run(self, expression):
        try:
            tree = ast.parse(str(expression), mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid expression: {exc}")
        return f"Result: {self._eval(tree.body)}"

    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._OPERATORS:
            return self._OPERATORS[type(node.op)](
                self._eval(node.left), self._eval(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._OPERATORS:
            return self._OPERATORS[type(node.op)](self._eval(node.operand))
        # Anything else (names, calls, attributes, subscripts, ...) is rejected.
        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


class FileReaderTool(BaseTool):
    """Read UTF-8 text files under the repo's ``data/`` directory only."""

    name = "file_reader"
    description = (
        "Read a UTF-8 text file located under the repo's data/ directory. "
        "Input: path (string), relative to data/, e.g. 'bugs.json'. "
        "Path traversal outside data/ is blocked."
    )
    args_schema = {"path": "string - path relative to data/, e.g. 'bugs.json'"}

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else REPO_ROOT / "data"

    def run(self, path):
        target = (self.base_dir / str(path)).resolve()
        base = self.base_dir.resolve()
        # Guard: the resolved target must be inside base_dir. This blocks
        # '../', absolute paths like '/etc/passwd', and symlink escapes.
        if target != base and base not in target.parents:
            raise ValueError(f"Access denied: '{path}' is outside the data directory.")
        if not target.is_file():
            raise ValueError(f"File not found: '{path}'.")
        return target.read_text(encoding="utf-8")


class MemoryTool(BaseTool):
    """Persistent key-value notes, backed by a JSON file so notes survive runs."""

    name = "memory"
    description = (
        "Persistent key-value notes. operation is one of: "
        "write_note (requires key, text), read_note (requires key), list_notes."
    )
    args_schema = {
        "operation": "string - one of: write_note, read_note, list_notes",
        "key": "string - note key (required for write_note and read_note)",
        "text": "string - note text (required for write_note)",
    }

    def __init__(self, store_path=None):
        self.store_path = (
            Path(store_path) if store_path else REPO_ROOT / "memory" / "notes.json"
        )
        self._notes = self._load()

    def _load(self):
        if self.store_path.is_file():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
            return data if isinstance(data, dict) else {}
        return {}

    def _save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys keeps the file byte-identical across identical runs.
        self.store_path.write_text(
            json.dumps(self._notes, indent=2, sort_keys=True), encoding="utf-8"
        )

    def clear(self):
        """Remove all notes (used by the demo for a clean, deterministic run)."""
        self._notes = {}
        self._save()

    def run(self, operation, key=None, text=None):
        if operation == "write_note":
            if not key:
                raise ValueError("write_note requires 'key'.")
            self._notes[str(key)] = str(text or "")
            self._save()
            return f"Note saved under key '{key}'."
        if operation == "read_note":
            if key not in self._notes:
                raise ValueError(f"No note found for key '{key}'.")
            return self._notes[key]
        if operation == "list_notes":
            return json.dumps(sorted(self._notes.keys()))
        raise ValueError(
            f"Unknown operation '{operation}'. Use write_note, read_note, or list_notes."
        )
