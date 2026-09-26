# Using the Akasha tool gate

This guide is for **application developers** who want to put a safety gate in
front of tool calls (shell, files, mail, payments, …). It is not a free-form
chat agent: Akasha scores a **proposed** tool call and returns
`ready` / `abstain` / `blocked`. **Your app executes** only when the plan is
`ready`.

For thresholds, demos, and training details see
[examples/gate/README.md](../examples/gate/README.md).

---

## What you get

| Piece | Role |
|-------|------|
| Your System 2 (LLM / policy) | Proposes `tool_name` + `arguments` |
| `akasha_model` scorer + gate | Authorizes, abstains, or blocks |
| Your runtime (Akasha OS or other) | Runs the tool **only** if `plan.status == "ready"` |

Akasha **never** executes tools. There is no built-in tool runner.

Statuses:

| Status | Meaning | What your host should do |
|--------|---------|---------------------------|
| `ready` | Gates passed | Run your own permission checks, then execute |
| `abstain` | Low confidence / unclear choice | Ask the user, or ask System 2 to reformulate |
| `blocked` | Policy / risk / schema / capability | Do **not** execute; show `plan.reason` |

---

## Install

```sh
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

Python 3.10+ and PyTorch are required (see the root README).

---

## Path A — Gate with explicit signals (fastest)

Use this when **you** already have authorization / risk scores (rules, another
model, or tests). No checkpoint needed.

```python
from akasha_model import (
    ToolSpec,
    ToolProposal,
    GateSignals,
    evaluate_gate,
    describe_plan,
)

tools = {
    "fs.read": ToolSpec(
        "fs.read",
        "Read a workspace file.",
        parameters={
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        },
        required_capability="workspace_access",
    ),
    "fs.delete": ToolSpec(
        "fs.delete",
        "Delete a workspace file.",
        required_capability="workspace_access",
        irreversible=True,
    ),
}

proposal = ToolProposal("fs.read", {"path": "notes.txt"})
signals = GateSignals(
    authorized=0.95,
    sufficient_context=0.92,
    capability_present=0.94,
    confirmation_needed=0.05,
)

plan = evaluate_gate(tools, proposal, signals)
print(describe_plan(plan))
# READY: fs.read — all planner gates passed

if plan.executable:
    # YOUR code — not part of akasha_model
    host_executor.execute(plan.tool_name, plan.arguments)
else:
    # abstain → ask user; blocked → refuse and log plan.reason
    ...
```

Try the scripted scenarios:

```sh
python examples/gate/demo.py
```

---

## Path B — Gate with the tiny trained scorer

Use this when you want the **model** to fill risk (`Score`) and authorization
nouls from the situation text. The Choice used by the planner is peaked on the
System 2 proposal (Akasha authorizes that call; it does not invent a different
tool by default).

```python
import torch
from akasha_model import ToolSpec, ToolProposal
from akasha_model.gate_multitask import load_gate_multitask, plan_scored_proposal
from akasha_model.multitask import MultiQuestionCollator

# At least two ToolSpec entries (same idea as examples/gate/catalog.py).
tools = {
    "fs.read": ToolSpec(
        "fs.read", "Read a workspace file.",
        parameters={
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        },
        required_capability="workspace_access",
    ),
    "fs.delete": ToolSpec(
        "fs.delete", "Delete a workspace file.",
        required_capability="workspace_access",
        irreversible=True,
    ),
}

device = torch.device("cpu")
model, config = load_gate_multitask(
    "examples/gate/checkpoints/gate-tiny.pt", device,
)
collator = MultiQuestionCollator(
    config["context_tokens"],
    config["question_tokens"],
    config.get("option_tokens", 128),
    config.get("max_score_levels", 10),
)

proposal = ToolProposal("fs.read", {"path": "notes.txt"})
context = {
    "proposal": proposal.tool_name,
    "arguments": proposal.arguments,
    "confirmation_given": False,
    "situation": "User asks to open their personal notes file.",
}

plan = plan_scored_proposal(
    model, collator, tools, proposal,
    context=context,
    device=device,
    confirmation_given=False,
)

if plan.executable:
    host_executor.execute(plan.tool_name, plan.arguments)
```

One-shot demo (uses the committed checkpoint):

```sh
python examples/gate/scored_demo.py \
  --checkpoint examples/gate/checkpoints/gate-tiny.pt
```

### Retrain on your own situations

```sh
python examples/gate/generate_data.py --output data/gate   # template format
# edit / replace data/gate/*.jsonl with your rows, then:
python examples/gate/train_tiny.py --data data/gate --output runs/gate-tiny.pt
python examples/gate/scored_demo.py --checkpoint runs/gate-tiny.pt
```

Each JSONL row is a multitask object: `context` + `questions` (`choice`,
`score`, `noul`). Keep related contexts in one split (anti-leakage). The root
README describes the multitask format in full.

---

## Reading the plan

```python
plan.status          # "ready" | "abstain" | "blocked"
plan.executable      # True only for ready
plan.tool_name       # selected tool or None
plan.arguments       # dict or None
plan.reason          # human-readable why
plan.risk_score      # optional scalar from Score
plan.choice_probability
plan.choice_confidence
```

Default thresholds (change only after measuring false positives on held-out
data):

| Setting | Default | If violated |
|---------|---------|-------------|
| Min choice probability | 0.55 | `abstain` |
| Min choice confidence | 0.50 | `abstain` |
| Noul threshold | 0.70 | `blocked` |
| Max risk score (levels 0–2) | 1.5 | `blocked` |

Also `blocked`: unknown tool, bad arguments, missing human confirmation when
the tool is irreversible / requires confirmation.

Override with a custom `ToolCallPlanner(...)` passed into `evaluate_gate` /
`plan_scored_proposal`.

---

## Path C — Host dispatch (Akasha OS contract)

Akasha OS is **not** vendored in this repository. Integrate by implementing
`ToolHost` (`check_permissions` + `execute`) and calling `dispatch_plan` /
`run_gated_call`. The package never executes tools itself.

```python
from akasha_model import (
    ToolProposal, GateSignals, run_gated_call, describe_outcome,
)

class MyOsHost:
    def check_permissions(self, tool_name, arguments):
        # Final OS / IAM checks (capabilities, confirmations, ACLs).
        return True, "ok"

    def execute(self, tool_name, arguments):
        # Real side effects live HERE (Akasha OS), not in akasha_model.
        return os_runtime.invoke(tool_name, arguments)

outcome = run_gated_call(tools, proposal, signals, MyOsHost())
print(describe_outcome(outcome))
# EXECUTED / SKIPPED_ABSTAIN / SKIPPED_BLOCKED / REJECTED_BY_HOST
```

| `outcome.action` | Meaning |
|------------------|---------|
| `executed` | Gate was `ready` and host permissions passed → `execute` ran |
| `skipped_abstain` | Gate abstained → no execute |
| `skipped_blocked` | Gate blocked → no execute |
| `rejected_by_host` | Gate ready but host `check_permissions` failed → no execute |

Scored path: `run_scored_gated_call(...)` (same host contract).

Out-of-OS demo (in-memory fake side effects only):

```sh
python examples/gate/host_demo.py
```

### Minimal host loop

```text
1. Build a ToolSpec catalog (names, schemas, capabilities).
2. System 2 proposes tool + args (+ optional situation text).
3. Call run_gated_call(...) or run_scored_gated_call(...).
4. Handle HostOutcome (executed / skipped_* / rejected_by_host).
```

Akasha OS (or any other host) stays the source of truth for permissions and
side effects.

---

## What this is not

- Not a chatbot that “sorts your email” end-to-end without a proposal + catalog
- Not a replacement for OS / IAM permission checks
- Not Jev/Laya parity on public typed-decisions leaderboards (different product
  goal: **tool authorization gate**)
- The committed `gate-tiny.pt` is a **small demo** scorer on synthetic data —
  train on your traffic before production

---

## Where to look next

| Need | Location |
|------|----------|
| Run demos / thresholds / go-no-go | [examples/gate/README.md](../examples/gate/README.md) |
| Package API | `akasha_model.gate`, `gate_multitask`, `host`, `tool_calling` |
| Host demo (fake OS) | `python examples/gate/host_demo.py` |
| Multitask / RLCD text training | Root [README.md](../README.md) |
| Vision games (lab only) | [examples/doom](../examples/doom/), [examples/chess](../examples/chess/) |
