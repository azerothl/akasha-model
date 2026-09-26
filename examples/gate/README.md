# Tool gate example

Product direction: a System 2 **proposes** a tool call; Akasha **authorizes**,
**abstains**, or **blocks**. Nothing is executed. The deterministic boundary
remains `ToolCallPlanner` via `evaluate_gate`.

**Start here for end-to-end usage:** [docs/using-the-tool-gate.md](../../docs/using-the-tool-gate.md)
(install, Path A scripted signals, Path B scored checkpoint, host loop).

## Defaults (planner thresholds)

Exported from `akasha_model.gate` / `default_gate_planner()`:

| Constant | Default | Effect |
|----------|---------|--------|
| `DEFAULT_MIN_CHOICE_PROBABILITY` | `0.55` | Below → `abstain` |
| `DEFAULT_MIN_CHOICE_CONFIDENCE` | `0.50` | Below → `abstain` |
| `DEFAULT_NOUL_THRESHOLD` | `0.70` | `authorized` / `sufficient_context` / `capability_present` below → `blocked` |
| `DEFAULT_MAX_RISK_SCORE` | `1.5` | Expected risk (levels 0–2) above → `blocked` |

Also blocked: unknown tool, failed argument schema, missing confirmation when
`requires_confirmation` / `irreversible` / high `confirmation_needed` noul.

### Go / no-go (false positives & negatives)

Treat these as release checks for the wedge:

| Case | Fail if |
|------|---------|
| Safe read / confirmed broadcast | status is not `ready` (**false positive** block) |
| Delete without confirmation | status is `ready` (**false negative**) |
| High-risk payment / shell without capability | status is `ready` |
| Ambiguous mail | status is `ready` (must `abstain` or `blocked`) |

Tune thresholds on a held-out authorize-tool-call split before changing defaults.
Latency target for a host integration: gate ≪ System 2 proposal time.

## Scripted signals (no trained scorer)

```sh
uv pip install -e '.[dev]'
python examples/gate/demo.py
```

| Scenario | Typical status |
|----------|----------------|
| Safe read | `ready` |
| Delete without confirmation | `blocked` |
| Ambiguous mail action | `abstain` |
| High-risk payment | `blocked` (risk) |
| Shell without capability | `blocked` |
| Broadcast with confirmation | `ready` |

## Scored path (tiny multitask → gate)

JSONL is synthetic authorize-tool-call data (`safe` / `dangerous` / `ambiguous`),
split by disjoint `family:variant` contexts (anti-leakage). Files go under
`data/gate/` (gitignored).

```sh
python examples/gate/generate_data.py --output data/gate
python examples/gate/train_tiny.py --data data/gate --output runs/gate-tiny.pt
python examples/gate/scored_demo.py --checkpoint examples/gate/checkpoints/gate-tiny.pt
# refresh checkpoint (writes runs/ then copy if desired):
python examples/gate/scored_demo.py --train --epochs 40 --checkpoint runs/gate-tiny.pt
```

A small trained checkpoint is committed at
`examples/gate/checkpoints/gate-tiny.pt` for CI / quick demos.

`akasha_model.gate_multitask.plan_scored_proposal` fills Score / Noul from the
tiny multitask scorer, peaks Choice on the System 2 proposal (the gate
authorizes a proposed call), then calls `evaluate_gate`. Still **no executor**.
Pass `use_model_choice=True` only once the route head is strong enough.

## Host / Akasha OS contract

Akasha OS is external. Implement `ToolHost` and use:

```python
from akasha_model import run_gated_call, run_scored_gated_call, dispatch_plan
# run_gated_call(tools, proposal, signals, host) → HostOutcome
# run_scored_gated_call(model, collator, tools, proposal, host, context=..., device=...)
```

`dispatch_plan` calls `host.execute` **only** after `ready` **and**
`host.check_permissions` succeeds. Demo without a real OS:

```sh
python examples/gate/host_demo.py
```

See [docs/using-the-tool-gate.md](../../docs/using-the-tool-gate.md) Path C.

## Outcomes (threshold calibration)

Log post-hoc success / user overrides, then summarize:

```sh
python examples/gate/outcomes_demo.py --log data/gate/outcomes.jsonl
```

API: `append_outcome`, `record_from_host_outcome`, `summarize_outcomes`,
`suggest_threshold_updates` (`akasha_model.outcomes`). Suggestions never
auto-write new `DEFAULT_*` values — Path D in the user guide.
