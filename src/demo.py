#!/usr/bin/env python3
"""QA-assistant demo: triage 5 bug reports with the ReAct agent.

Runs fully in mock mode (no API key, no network). The agent:
  1. reads data/bugs.json with the file_reader tool,
  2. computes a priority score per bug with the calculator tool
     (score = severity_weight * 10),
  3. writes a triage note per bug with the memory tool,
  4. returns a priority-ordered triage summary.

Artifacts:
  reports/trajectory.jsonl  - the agent's trajectory (one run per file)
  reports/triage_summary.md  - the human-readable triage summary

The run is deterministic: memory is reset first and the run_id is a hash of
the task, so re-running produces a byte-identical trajectory file.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agent import Agent, MockBackend, validate_schema
from src.tools import CalculatorTool, FileReaderTool, MemoryTool, WebSearchTool

TASK = "Triage these 5 bug reports."


def main():
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    memory = MemoryTool()
    memory.clear()  # fresh, deterministic run

    tools = [WebSearchTool(), CalculatorTool(), FileReaderTool(), memory]
    agent = Agent(
        tools=tools,
        backend=MockBackend(),
        max_steps=12,
        trajectory_path=reports_dir / "trajectory.jsonl",
    )

    result = agent.run(TASK)

    ok, errors = validate_schema(result, [t.name for t in tools])
    if not ok:
        raise SystemExit(f"Trajectory failed schema validation: {errors}")

    summary_md = (
        "# Bug Triage Summary\n\n"
        f"{result['final_answer']}\n\n"
        f"_Run `{result['run_id']}` · status: `{result['status']}` · "
        f"{len(result['steps'])} steps._\n"
    )
    (reports_dir / "triage_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"status:        {result['status']}")
    print(f"steps:         {len(result['steps'])}")
    print(f"run_id:        {result['run_id']}")
    print(f"trajectory:    {reports_dir / 'trajectory.jsonl'}")
    print(f"summary:       {reports_dir / 'triage_summary.md'}")
    print()
    print(result["final_answer"])


if __name__ == "__main__":
    main()
