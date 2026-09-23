# ai-agent-lab

I built a ReAct-style AI agent from scratch — no LangChain, no crewAI, no agent
frameworks. Just a Thought → Action → Observation loop, four hand-written tools,
and a deterministic rule-based mock model so the whole thing runs offline with
zero API keys. I'm a QA Lead, so I built it the way I'd want to test one: every
run is logged as a validated trajectory, and the demo is a bug-triage task —
the kind of work I do every day.

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │            ReAct LOOP                 │
                        │                                     │
  ┌──────────┐   task   │   ┌───────────────────────────┐     │
  │  demo.py │─────────▶│   │  MockBackend.decide()     │     │
  │ "Triage  │          │   │  rule-based "model":      │     │
  │ these 5  │          │   │  emits Thought + Action   │     │
  │ bugs"    │          │   └─────────────┬─────────────┘     │
  └──────────┘          │                 │ Thought: ...        │
                        │                 │ Action: <tool>      │
                        │                 │ Action Input: {...} │
                        │                 ▼                     │
                        │   ┌───────────────────────────┐     │
                        │   │  Agent.run() guard rails  │     │
                        │   │  - max_steps (default 12) │     │
                        │   │  - same (tool,args) 3x →  │     │
                        │   │    status "loop_detected" │     │
                        │   └─────────────┬─────────────┘     │
                        │                 │                     │
                        │                 ▼                     │
                        │   ┌───────────────────────────┐     │
                        │   │        TOOLS              │     │
                        │   │  web_search  (mock KB)    │     │
                        │   │  calculator  (safe AST)   │     │
                        │   │  file_reader (data/ only) │     │
                        │   │  memory      (JSON notes) │     │
                        │   └─────────────┬─────────────┘     │
                        │                 │ Observation: ...    │
                        │                 ▼                     │
                        │   repeat until Final Answer           │
                        └─────────────────────────────────────┘
                                          │
                                          ▼
                        reports/trajectory.jsonl  (validated schema)
                        reports/triage_summary.md (human-readable result)
```

## Quickstart

```bash
git clone <this-repo> && cd ai-agent-lab
pip install -r requirements.txt   # just pytest; everything else is stdlib
python src/demo.py                # runs the bug-triage demo, no API key needed
pytest -q                         # 31 tests
```

`python src/demo.py` prints the triage summary and writes two artifacts:

- `reports/trajectory.jsonl` — the agent's full Thought/Action/Observation trace
- `reports/triage_summary.md` — the priority-ordered triage result

## How it works

**`src/tools.py`** — four tools, each a class with `name`, `description`,
`args_schema`, and `run(**kwargs)`:

| Tool | What it does |
|---|---|
| `web_search` | Mock search over a built-in knowledge base of 10 QA notes (flaky-test quarantine, contract testing, mutation testing, …). Keyword-scored, deterministic ordering. |
| `calculator` | Safe arithmetic via `ast` parsing — never `eval()` on raw input. Only numbers and `+ - * / // % **` are allowed; `__import__('os')` and friends raise `ValueError`. |
| `file_reader` | Reads UTF-8 files under `data/` only. Resolves the path and rejects anything outside `data/` (`../`, absolute paths, symlink escapes). |
| `memory` | Key-value notes (`write_note` / `read_note` / `list_notes`) persisted to `memory/notes.json`, so notes survive across runs. |

**`src/agent.py`** — the ReAct loop from first principles:

- `PROMPT_TEMPLATE` renders `Thought:` / `Action:` / `Action Input:` lines and
  appends each tool result as `Observation:`. (Used for real by
  `OpenAIBackend`; the mock backend works on structured decisions directly.)
- `MockBackend` is a deterministic, rule-based stand-in for an LLM. For the
  triage task it follows a fixed sensible plan: read `data/bugs.json` →
  per bug: `calculator` (priority score = severity weight × 10) →
  `memory` (`write_note` with the triage decision) → final priority-ordered
  summary. Other tasks get a generic fallback: one `web_search`, then summarize.
- `OpenAIBackend` is a real, working extension point implemented with `urllib`
  against the chat-completions API (temperature 0). It reads `OPENAI_API_KEY`
  from the environment (see `.env.example`) and raises a clear error without
  one. Honest note: I implemented it for real (~35 lines), but the demo, the
  tests, and CI all run on the mock backend — nothing here needs a key.
- Guards: `max_steps` (default 12) → `max_steps_exceeded`; the same
  `(tool, args)` pair about to run a 3rd time → `loop_detected`. Tool misuse
  becomes an `Observation: Error: ...` line so the agent can recover.
- `validate_schema()` checks a trajectory dict against the schema below and
  returns `(ok, errors)`.

**`src/demo.py`** — "Triage these 5 bug reports." 11 steps: 1 file read +
5 × (score + note). Memory is reset first so the run is byte-for-byte
deterministic (verified: two runs → identical `trajectory.jsonl`).

## Sample output

`python src/demo.py`:

```
status:        success
steps:         11
run_id:        run-b7a63df5dfc4
trajectory:    reports/trajectory.jsonl
summary:       reports/triage_summary.md

Bug triage complete. Priority order (highest first):

- BUG-104 (score 100, blocker): Data loss when app is backgrounded during upload - P0 - fix immediately
- BUG-101 (score 90, critical): Login page crashes on Safari 17 - P0 - fix immediately
- BUG-102 (score 60, major): Checkout total off by one cent when coupon applied - P1 - next sprint
- BUG-105 (score 30, minor): Dark mode toggle flickers on settings page - P2 - backlog
- BUG-103 (score 10, trivial): Typo in footer copyright text - P3 - low priority

Triage notes saved to memory under keys: triage_BUG-104, triage_BUG-101, triage_BUG-102, triage_BUG-105, triage_BUG-103.
```

First three trajectory steps (`reports/trajectory.jsonl`, pretty-printed):

```json
{"step": 1,
 "thought": "I need the bug reports before I can triage them. I'll read data/bugs.json first.",
 "action": {"tool": "file_reader", "args": {"path": "bugs.json"}},
 "observation": "[{\"id\":\"BUG-101\",\"title\":\"Login page crashes on Safari 17\", ...}, ...]"}
{"step": 2,
 "thought": "Computing the priority score for BUG-101 (severity hint: critical).",
 "action": {"tool": "calculator", "args": {"expression": "9 * 10"}},
 "observation": "Result: 90"}
{"step": 3,
 "thought": "Recording the triage decision for BUG-101 in memory.",
 "action": {"tool": "memory",
            "args": {"operation": "write_note", "key": "triage_BUG-101",
                     "text": "BUG-101: Login page crashes on Safari 17 | severity=critical | priority_score=90 | P0 - fix immediately"}},
 "observation": "Note saved under key 'triage_BUG-101'."}
```

## Trajectory schema

Every run is logged as **one JSON object on one line** of the JSONL file
(one run per file — `demo.py` overwrites `reports/trajectory.jsonl` each run):

```json
{"run_id": "string",
 "task": "string",
 "steps": [{"step": 1,
            "thought": "string",
            "action": {"tool": "string", "args": {}},
            "observation": "string"}],
 "final_answer": "string",
 "status": "success|failed|max_steps_exceeded|loop_detected"}
```

Rules enforced by `validate_schema()`:

- `run_id`, `task`, `final_answer` are strings; `steps` is a list.
- `steps[i].step` is 1-indexed and sequential; `thought` and `observation`
  are strings.
- `action.tool` must exactly match a registered tool name
  (`web_search`, `calculator`, `file_reader`, `memory`); `action.args` must
  be a JSON object.
- `status` is one of `success`, `failed`, `max_steps_exceeded`,
  `loop_detected`.

## Determinism

No randomness anywhere, no timestamps in trajectories. `run_id` is
`run-` + the first 12 hex chars of `sha256(task)` (hashing is only used for a
stable id — documented here). The demo resets the memory store before each
run, and the mock backend's plan is purely positional. Re-running the demo
produces a byte-identical `trajectory.jsonl` (I verify this with `diff`).

## Tests

```
pytest -q   # 31 tests
```

- Each tool: happy path + edge case (calculator rejects `__import__`,
  file reader blocks `../` and absolute paths, memory round-trips and
  persists across instances).
- The agent terminates within `max_steps` on the demo task and its trajectory
  validates against the schema.
- Loop detection triggers on a deliberately repetitive backend; a wandering
  backend (always-new args) hits `max_steps_exceeded` instead.
- The schema validator itself is tested against bad status, unknown tool,
  missing field, and non-object args.
- Determinism: two identical runs produce identical trajectories.
- `OpenAIBackend` raises a clear `RuntimeError` when `OPENAI_API_KEY` is unset.

## Roadmap

- Point `OpenAIBackend` at the real ReAct prompt end-to-end and add a
  recorded (canned-response) test so the LLM path is covered without a key.
- Streaming step output (`--verbose`) that prints Thought/Action/Observation
  live.
- A second demo: flaky-test triage using `web_search` + `memory` together.
- Trajectory diff tooling (`traj diff run-a run-b`) for regression-testing
  agent behavior across prompt changes.
- Multi-agent sketch: a "reviewer" agent that critiques the triager's summary
  before it's finalized.
