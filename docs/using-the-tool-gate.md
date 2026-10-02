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

### Fail-closed under injection

Path A is a **deterministic** policy-enforcement point. Natural language in
tool arguments, catalog descriptions, or pasted prior tool outputs **cannot**
change `GateSignals`, grant capabilities, skip confirmation, or force
`ready`. Only typed signals + catalog schema + planner thresholds matter.

Red-team fixtures (must stay `blocked` / `abstain` under honest deny signals):

```sh
pytest -q tests/adversarial/
```

---

## Install

Prefer [PyPI](https://pypi.org/project/akasha-model/) when you want a normal
`pip` / `uv` pin. The git tag remains the minimum OS unblock and a fallback when
you need a commit that is not on PyPI yet.

### Path A only (lightweight — no torch)

OS hosts that use scripted `GateSignals` + `evaluate_gate` / `run_gated_call`
do **not** need PyTorch:

```sh
uv pip install "akasha-model==0.3.0"
# or pin the git tag:
uv pip install "akasha-model @ git+https://github.com/azerothl/akasha-model.git@v0.3.0"
# or from a checkout:
uv pip install -e '.[dev]'
```

This installs the planner, host dispatch, outcomes, and wire modules. Importing
`akasha_model.gate` / `.host` / `.tool_calling` / `.outcomes` / `.wire` does not
load torch. Prefer a release tag over feature branches such as `cursor/gate-host-*`.

See [CHANGELOG.md](../CHANGELOG.md). Integrator surface: `evaluate_gate`,
`run_gated_call`, `HostOutcome`, `ToolHost`.

### Full package (Path B / train / vision)

```sh
uv pip install "akasha-model[torch]==0.3.0"
# or pin the git tag:
uv pip install "akasha-model[torch] @ git+https://github.com/azerothl/akasha-model.git@v0.3.0"
# or from a checkout:
uv pip install -e '.[dev,torch]'
```

### Release checklist

1. Bump `version` in `pyproject.toml` and update [CHANGELOG.md](../CHANGELOG.md).
2. Tag and push (`vX.Y.Z`).
3. Publish the matching [GitHub Release](https://github.com/azerothl/akasha-model/releases)
   (or run **Actions → Publish → Run workflow**).
4. Confirm the build appears on [PyPI](https://pypi.org/project/akasha-model/).

Publishing uses Trusted Publishing (OIDC) via `.github/workflows/publish.yml` —
no long-lived PyPI API token.

---

## Path A — Gate with explicit signals (fastest)

Use this when **you** already have authorization / risk scores (rules, another
model, or tests). No checkpoint needed.

Rust hosts (Akasha OS `aos-agent`) can call the same Path A logic in-process via
the [`akasha-gate`](../crates/akasha-gate/) crate — see
[docs/rust-gate.md](rust-gate.md). Path B scorers remain Python-only.

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

Try the first-run walkthrough, then the fuller scenario list:

```sh
python examples/usage/visual_demo.py --open  # state + probability bars
python examples/usage/demo.py               # text-only fallback
python examples/gate/demo.py
```

### Invocation budgets (host-owned)

The OS owns rate-limit counters. Pass the remaining allowance on each call:

```python
signals = GateSignals(
    authorized=0.95,
    sufficient_context=0.9,
    capability_present=0.92,
    budget_remaining=0,  # host says quota exhausted
)
plan = evaluate_gate(tools, proposal, signals)
# plan.status == "blocked", plan.reason == "budget_exceeded"
```

When `budget_remaining` is omitted (`None`), the gate does not enforce a
budget. The model never increments or resets counters.

### Path A latency (measure, don't flake)

```sh
python examples/gate/bench_latency.py --iters 2000
```

Reports p50/p95 for `evaluate_gate` and `dispatch_plan` on the smoke catalog
(CPU, no torch / no GPU). **Aspirational target:** p95 `evaluate_gate` < 5 ms
for a small catalog on a modern CPU — gate ≪ System 2 proposal time. Treat the
number as an SLO input for hosts, not a flaky CI fail gate.

### Catalog constraints (phase 3, optional)

| Layer | Responsibility |
|-------|----------------|
| `CatalogPolicy` | Allow/deny tool names; required placement tags on a surface |
| `ToolSpec` | Schema, capability, confirmation, irreversible |
| OS `check_permissions` | Final ACL / grants — always required on `ready` |

```python
from akasha_model import CatalogPolicy, evaluate_gate

policy = CatalogPolicy(
    allowlist=frozenset({"fs.read", "fs.delete"}),
    denylist=frozenset({"shell.run"}),
    required_placement=frozenset({"preview"}),
)
plan = evaluate_gate(tools, proposal, signals, catalog_policy=policy)
```

Opt-in: omit `catalog_policy` for unconstrained catalogs (demos keep working).
Stable blocked reason prefixes: `catalog_allowlist`, `catalog_denylist`,
`catalog_placement`. Outcomes suggestions never auto-write these rules.

### Authority profiles (optional)

Optional MIDAS-inspired profiles distinguish **escalate** (human review),
**reject** (hard deny), and **clarify** (missing context) without exploding the
`ready` / `abstain` / `blocked` status enum. Statuses stay compatible; reasons
gain stable prefixes when a profile is passed:

| Prefix | Meaning | Typical status |
|--------|---------|----------------|
| `escalate:` | Consequence / authority over profile | `abstain` |
| `reject:` | Hard deny (auth / risk / policy) | `blocked` |
| `clarify:` | Missing context keys / low context | `abstain` |

```python
from akasha_model import AuthorityProfile, evaluate_gate

profile = AuthorityProfile(
    name="payments",
    max_consequence=0.5,
    required_context_keys=("user_id",),
)
plan = evaluate_gate(tools, proposal, signals, authority_profile=profile)
```

Omit `authority_profile` for today's behaviour. Profiles do not replace OS
`check_permissions` or catalog allowlists. Wire-schema bump is needed only if
hosts start parsing prefixes as a new enum (#12). Outcomes never auto-write
`DEFAULT_*` or profile fields.

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

### MASK layout (schema-first vs state-first)

Path B MASK rows default to **schema-first**: question + options, then the
changing state. That lets serving amortize a shared prefix when the authorize
schema is fixed and only the proposal/context changes (KV / prefix cache).

| Layout | Flag | When |
|--------|------|------|
| `schema_first` (default) | `--mask-layout schema_first` | Hosts with a fixed Choice/Score/Noul authorize schema |
| `state_first` | `--mask-layout state_first` | Compatibility / ablation |
| Mix at train | `--mask-layout mix --layout-mix 0.5` | One checkpoint that serves both |

API: `build_sequence(..., layout=...)` and `MaskCollator(..., layout=..., layout_mix=...)`.
Eval / serving should stay on `schema_first` unless you intentionally trained
state-first only.

Latency note (CPU tiny encoder, 2026-09): tokenising two states that share the
same authorize schema yields an identical prefix through the option markers
under `schema_first` (measured in unit tests). Wall-clock win on the byte
encoder is modest; the win is for larger bidirectional encoders where the
shared prefix can stay in cache across many proposals. Prefer schema-first for
production Path B authorize traffic.

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

## Path D — Outcomes (calibrate thresholds)

After each gated call, log what happened so you can tune thresholds on **real**
results (not Hub typed-decisions scores).

```python
from akasha_model import (
    append_outcome, record_from_host_outcome,
    load_outcomes, summarize_outcomes, suggest_threshold_updates,
)

outcome = run_gated_call(tools, proposal, signals, host)
append_outcome(
    "data/gate/outcomes.jsonl",
    record_from_host_outcome(
        outcome,
        success=True,              # did the tool actually help?
        user_forced=False,         # did a human override a block/abstain?
        notes="optional",
        signals=signals,
    ),
)

rows = load_outcomes("data/gate/outcomes.jsonl")
print(summarize_outcomes(rows))
print(suggest_threshold_updates(rows))  # suggestions only — never auto-applies
```

Demo:

```sh
python examples/gate/outcomes_demo.py --log data/gate/outcomes.jsonl
```

Akasha OS should append the same JSONL (or equivalent) from production traffic.
Do **not** change `DEFAULT_*` thresholds until the summary shows a clear
false-positive or risky-failure pattern.

---

## Wire JSON contract (Rust / CLI bridges)

Hosts that serialize plans over JSON (e.g. Akasha OS `aos-agent` → Python CLI)
should use the package schemas and helpers instead of hand-rolled field names:

| Piece | Location |
|-------|----------|
| Schemas | `akasha_model/schemas/*.schema.json` |
| Helpers | `akasha_model.wire` (`plan_to_dict`, `outcome_to_dict`, `request_to_dict`, …) |
| Version | `CONTRACT_VERSION` (currently `1`) on every envelope |
| Goldens | `tests/contracts/fixtures/` |
| Notes | [docs/contracts/README.md](contracts/README.md) |

```python
from akasha_model import evaluate_gate, plan_to_dict, CONTRACT_VERSION

plan = evaluate_gate(tools, proposal, signals)
wire = plan_to_dict(plan, request_id="req-001")
assert wire["contract_version"] == CONTRACT_VERSION
```

Breaking renames require a version bump + migration note in
`docs/contracts/README.md`.

### Running contract tests as a host integrator

Path A matrix fixtures (catalog → plan status → `dispatch_plan` action) live
under `tests/contracts/` and do not require torch:

```sh
uv pip install -e '.[dev]'   # or '.[dev,torch]' if you already use the full package
pytest -q tests/contracts/
```

`tests/contracts/fixtures/path_a_matrix.json` is the published suite OS CI can
vendor or run in-place. Multi-capability tools use
`ToolSpec(required_capabilities=(...))` with optional per-token
`GateSignals.capability_signals`.

### Audit envelopes + request_id

```python
from akasha_model import build_audit_envelope, evaluate_gate, plan_to_dict

plan = evaluate_gate(tools, proposal, signals, request_id="req-001")
wire = plan_to_dict(plan, request_id="req-001")  # echoes id; model does not persist
envelope = build_audit_envelope(
    plan=plan, proposal=proposal, signals=signals, tools=tools, request_id="req-001",
)
# envelope.content_sha256 — hash-only integrity; OS may add signatures later
```

**Dedup is host-owned.** Retries from async daemons must use the same
`request_id`; akasha-model only propagates the key. Persist envelopes next to
outcomes JSONL; `suggest_threshold_updates` still never auto-writes `DEFAULT_*`.

---

## What this is not

- Not a chatbot that “sorts your email” end-to-end without a proposal + catalog
- Not a replacement for OS / IAM permission checks
- Not Jev/Laya parity on public typed-decisions leaderboards (different product
  goal: **tool authorization gate**)
- The committed `gate-tiny.pt` is a **small demo** scorer on synthetic data —
  train on your traffic before production (see the **Production Path B
  checklist** in [examples/gate/README.md](../examples/gate/README.md) and
  [path-b-real-traffic.md](path-b-real-traffic.md))
- Optional Path B description-reading probe (not a gate cutover): see
  [dataset-zeroshot-decision.md](dataset-zeroshot-decision.md) and
  `scripts/generate_zeroshot_decision_dataset.py` (data under `data/`, gitignored)

---

## Where to look next

| Need | Location |
|------|----------|
| First-run Path A walkthrough | [examples/usage](../examples/usage/README.md) · `python examples/usage/visual_demo.py --open` |
| Run demos / thresholds / go-no-go | [examples/gate/README.md](../examples/gate/README.md) |
| Path B on host traffic (beyond gate-tiny) | [path-b-real-traffic.md](path-b-real-traffic.md) |
| Other concrete menus (triage, OS routing, checklist, ensemble, vision) | [use-cases.md](use-cases.md) |
| Package API | `akasha_model.gate`, `gate_multitask`, `host`, `tool_calling`, `wire` |
| Wire schemas / Rust bridge | [contracts/README.md](contracts/README.md) · `akasha_model/schemas/` |
| Host demo (fake OS) | `python examples/gate/host_demo.py` |
| Outcomes / calibration | `python examples/gate/outcomes_demo.py` · `akasha_model.outcomes` |
| Multitask / RLCD text training | Root [README.md](../README.md) |
| Vision games (lab only) | [examples/doom](../examples/doom/), [examples/chess](../examples/chess/) |
