"""A ReAct-style agent built from first principles (no frameworks).

Loop:
    Thought -> Action -> Action Input -> (tool runs) -> Observation -> ...

Backends:
    MockBackend   - deterministic, rule-based stand-in for an LLM. Default; needs
                    no API key and no network. Used by the demo and all tests.
    OpenAIBackend - real extension point implemented with urllib against the
                    OpenAI chat-completions API. Opt-in via OPENAI_API_KEY.

Guards:
    max_steps             - hard cap on tool-calling steps (default 12)
    loop detection        - the same (tool, args) pair 3x -> status "loop_detected"

Every run is logged as one JSON object on one line of a JSONL file
(see TRAJECTORY_SCHEMA in the README).
"""

import hashlib
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PROMPT_TEMPLATE = """You are a careful QA assistant that reasons step by step using tools.

Available tools:
{tool_descriptions}

Use this exact format for every step:

Thought: <your reasoning about what to do next>
Action: <exactly one of: {tool_names}>
Action Input: <a JSON object matching that tool's args_schema>

The tool result is then appended as:

Observation: <tool output>

Repeat Thought / Action / Action Input / Observation as needed. When the task is
complete, output exactly:

Thought: <your final reasoning>
Final Answer: <the complete answer to the task>

Rules:
- Never invent tool output; always wait for the Observation.
- Action Input must be valid JSON matching the tool's args_schema.
- If you catch yourself repeating an identical action, stop and summarize instead.

Task: {task}

History so far:
{history}
"""

ALLOWED_STATUSES = {"success", "failed", "max_steps_exceeded", "loop_detected"}

SEVERITY_WEIGHTS = {"blocker": 10, "critical": 9, "major": 6, "minor": 3, "trivial": 1}


def priority_recommendation(score):
    if score >= 90:
        return "P0 - fix immediately"
    if score >= 60:
        return "P1 - next sprint"
    if score >= 30:
        return "P2 - backlog"
    return "P3 - low priority"


def format_history(steps):
    """Render completed steps in the Thought/Action/Observation transcript format."""
    lines = []
    for s in steps:
        lines.append(f"Thought: {s['thought']}")
        lines.append(f"Action: {s['action']['tool']}")
        lines.append(f"Action Input: {json.dumps(s['action']['args'])}")
        lines.append(f"Observation: {s['observation']}")
    return "\n".join(lines) if lines else "(no steps yet)"


def tool_descriptions(tools):
    return "\n".join(
        f"- {t.name}: {t.description} Args: {json.dumps(t.args_schema)}" for t in tools
    )


def parse_react_output(text):
    """Parse Thought/Action/Action Input (or Final Answer) lines from model output."""
    thought = _first_group(r"Thought:\s*(.+)", text)
    final = _first_group(r"Final Answer:\s*(.+)", text, flags=re.S)
    if final is not None:
        return {"thought": thought or "", "action": None, "final_answer": final.strip()}
    action = _first_group(r"Action:\s*(\S+)", text)
    raw_args = _first_group(r"Action Input:\s*(\{.*?\})", text, flags=re.S) or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {}
    return {
        "thought": thought or "",
        "action": {"tool": (action or "").strip(), "args": args},
        "final_answer": None,
    }


def _first_group(pattern, text, flags=0):
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


class MockBackend:
    """Deterministic, rule-based mock model. No API calls, no randomness.

    For bug-triage tasks it follows a fixed, sensible plan:
      1. read data/bugs.json
      2. for each bug: calculator (priority score) -> memory write_note (triage note)
      3. final answer: priority-ordered triage summary

    Any other task gets a generic fallback: one web_search, then a summary.
    """

    def decide(self, task, steps):
        lowered = task.lower()
        if "bug" in lowered and "triage" in lowered:
            return self._triage_decide(task, steps)
        return self._fallback_decide(task, steps)

    # -- bug triage plan -----------------------------------------------------
    def _bugs_from_steps(self, steps):
        """Extract the bug list from the file_reader observation, if present."""
        for s in steps:
            if s["action"]["tool"] == "file_reader":
                try:
                    bugs = json.loads(s["observation"])
                except (json.JSONDecodeError, TypeError):
                    return []
                return bugs if isinstance(bugs, list) else []
        return None  # file not read yet

    def _triage_decide(self, task, steps):
        bugs = self._bugs_from_steps(steps)
        if bugs is None:
            return {
                "thought": (
                    "I need the bug reports before I can triage them. "
                    "I'll read data/bugs.json first."
                ),
                "action": {"tool": "file_reader", "args": {"path": "bugs.json"}},
                "final_answer": None,
            }
        if not bugs:
            return {
                "thought": "The bug file could not be parsed, so I cannot triage.",
                "action": None,
                "final_answer": "Triage failed: no readable bug reports found.",
            }

        # The plan is fixed once the bugs are known: for each bug, one
        # calculator step followed by one memory note. Step n (after the file
        # read) always executes plan[n - 1], so progress is purely positional
        # and fully deterministic.
        plan = []
        for bug in bugs:
            key = f"triage_{bug['id']}"
            weight = SEVERITY_WEIGHTS.get(str(bug.get("severity_hint", "")).lower(), 1)
            score = weight * 10
            rec = priority_recommendation(score)
            plan.append(
                {
                    "thought": (
                        f"Computing the priority score for {bug['id']} "
                        f"(severity hint: {bug.get('severity_hint')})."
                    ),
                    "action": {
                        "tool": "calculator",
                        "args": {"expression": f"{weight} * 10"},
                    },
                    "final_answer": None,
                }
            )
            plan.append(
                {
                    "thought": f"Recording the triage decision for {bug['id']} in memory.",
                    "action": {
                        "tool": "memory",
                        "args": {
                            "operation": "write_note",
                            "key": key,
                            "text": (
                                f"{bug['id']}: {bug['title']} | "
                                f"severity={bug.get('severity_hint')} | "
                                f"priority_score={score} | {rec}"
                            ),
                        },
                    },
                    "final_answer": None,
                }
            )

        next_index = len(steps) - 1  # steps[0] was the file read
        if next_index < len(plan):
            return plan[next_index]
        return {
            "thought": "All bug reports are triaged and recorded. Summarizing the results.",
            "action": None,
            "final_answer": self._triage_summary(bugs),
        }

    def _triage_summary(self, bugs):
        ranked = sorted(
            (
                SEVERITY_WEIGHTS.get(str(b.get("severity_hint", "")).lower(), 1) * 10,
                b,
            )
            for b in bugs
        )
        ranked.sort(key=lambda item: -item[0])
        lines = ["Bug triage complete. Priority order (highest first):", ""]
        for score, bug in ranked:
            lines.append(
                f"- {bug['id']} (score {score}, {bug.get('severity_hint')}): "
                f"{bug['title']} - {priority_recommendation(score)}"
            )
        keys = ", ".join(f"triage_{b['id']}" for _, b in ranked)
        lines += ["", f"Triage notes saved to memory under keys: {keys}."]
        return "\n".join(lines)

    # -- generic fallback ----------------------------------------------------
    def _fallback_decide(self, task, steps):
        if not steps:
            return {
                "thought": "This isn't a bug-triage task. I'll search the QA knowledge base first.",
                "action": {"tool": "web_search", "args": {"query": task}},
                "final_answer": None,
            }
        observation = steps[-1]["observation"] if steps else "no observation"
        return {
            "thought": "I have the search results. Summarizing for the user.",
            "action": None,
            "final_answer": f"Here's what I found for '{task}':\n{observation}",
        }


class OpenAIBackend:
    """Real LLM backend (extension point), implemented with urllib.

    Reads the API key from the ``OPENAI_API_KEY`` environment variable.
    Uses temperature 0 for near-deterministic output. Raises RuntimeError if no
    key is configured. The demo and tests never touch this class.
    """

    def __init__(self, model="gpt-4o-mini", api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it (see .env.example) or use MockBackend."
            )
        self.model = model
        self._tool_docs = ""

    def bind_tools(self, tools):
        self._tool_docs = tool_descriptions(tools)
        self._tool_names = ", ".join(t.name for t in tools)

    def decide(self, task, steps):
        import urllib.request

        prompt = PROMPT_TEMPLATE.format(
            task=task,
            tool_descriptions=self._tool_docs,
            tool_names=getattr(self, "_tool_names", ""),
            history=format_history(steps),
        )
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        text = payload["choices"][0]["message"]["content"]
        return parse_react_output(text)


class Agent:
    """Runs the ReAct loop: Thought -> Action -> Observation until done."""

    def __init__(self, tools, backend, max_steps=12, trajectory_path=None):
        self.tools = {t.name: t for t in tools}
        self.backend = backend
        self.max_steps = max_steps
        self.trajectory_path = Path(trajectory_path) if trajectory_path else None

    def run(self, task):
        # run_id is a hash of the task only: identical tasks -> identical runs.
        run_id = "run-" + hashlib.sha256(task.encode("utf-8")).hexdigest()[:12]
        steps = []
        seen_counts = {}  # (tool, canonical-args-json) -> times used
        status = "success"
        final_answer = ""

        for step_no in range(1, self.max_steps + 1):
            decision = self.backend.decide(task, [dict(s) for s in steps])
            thought = decision.get("thought", "")
            action = decision.get("action")

            if action is None:  # backend signalled completion
                final_answer = decision.get("final_answer", "")
                status = "success"
                break

            tool_name = action.get("tool", "")
            args = action.get("args", {}) or {}
            repeat_key = (tool_name, json.dumps(args, sort_keys=True, default=str))
            if seen_counts.get(repeat_key, 0) >= 2:
                # This would be the 3rd identical (tool, args) call: stop.
                status = "loop_detected"
                final_answer = (
                    f"Stopped: the agent repeated the same action 3 times "
                    f"({tool_name} with {args})."
                )
                break
            seen_counts[repeat_key] = seen_counts.get(repeat_key, 0) + 1

            tool = self.tools.get(tool_name)
            if tool is None:
                observation = (
                    f"Error: unknown tool '{tool_name}'. "
                    f"Registered tools: {sorted(self.tools)}."
                )
                steps.append(
                    {
                        "step": step_no,
                        "thought": thought,
                        "action": {"tool": tool_name, "args": args},
                        "observation": observation,
                    }
                )
                status = "failed"
                final_answer = observation
                break

            try:
                observation = str(tool.run(**args))
            except Exception as exc:  # tool misuse -> recoverable observation
                observation = f"Error: {type(exc).__name__}: {exc}"

            steps.append(
                {
                    "step": step_no,
                    "thought": thought,
                    "action": {"tool": tool_name, "args": args},
                    "observation": observation,
                }
            )
        else:
            status = "max_steps_exceeded"
            final_answer = (
                "Stopped: exceeded max_steps without producing a final answer."
            )

        trajectory = {
            "run_id": run_id,
            "task": task,
            "steps": steps,
            "final_answer": final_answer,
            "status": status,
        }
        if self.trajectory_path:
            self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            # One run per file: overwrite with the single JSON object for this run.
            self.trajectory_path.write_text(
                json.dumps(trajectory) + "\n", encoding="utf-8"
            )
        return trajectory


def validate_schema(trajectory, tool_names):
    """Validate a trajectory dict against the trajectory JSONL schema.

    Returns (ok: bool, errors: list[str]).
    """
    errors = []
    if not isinstance(trajectory, dict):
        return False, ["trajectory must be a JSON object"]

    for field in ("run_id", "task", "steps", "final_answer", "status"):
        if field not in trajectory:
            errors.append(f"missing field: {field}")

    if "run_id" in trajectory and not isinstance(trajectory["run_id"], str):
        errors.append("run_id must be a string")
    if "task" in trajectory and not isinstance(trajectory["task"], str):
        errors.append("task must be a string")
    if "final_answer" in trajectory and not isinstance(
        trajectory["final_answer"], str
    ):
        errors.append("final_answer must be a string")
    if "status" in trajectory and trajectory["status"] not in ALLOWED_STATUSES:
        errors.append(
            f"status must be one of {sorted(ALLOWED_STATUSES)}, "
            f"got {trajectory['status']!r}"
        )

    steps = trajectory.get("steps")
    if isinstance(steps, list):
        for i, step in enumerate(steps, start=1):
            prefix = f"steps[{i}]"
            if not isinstance(step, dict):
                errors.append(f"{prefix} must be an object")
                continue
            if step.get("step") != i:
                errors.append(f"{prefix}.step must be {i}")
            if not isinstance(step.get("thought"), str):
                errors.append(f"{prefix}.thought must be a string")
            action = step.get("action")
            if not isinstance(action, dict):
                errors.append(f"{prefix}.action must be an object")
            else:
                if action.get("tool") not in tool_names:
                    errors.append(
                        f"{prefix}.action.tool {action.get('tool')!r} "
                        f"is not a registered tool"
                    )
                if not isinstance(action.get("args"), dict):
                    errors.append(f"{prefix}.action.args must be a JSON object")
            if not isinstance(step.get("observation"), str):
                errors.append(f"{prefix}.observation must be a string")
    elif "steps" in trajectory:
        errors.append("steps must be a list")

    return (len(errors) == 0), errors
