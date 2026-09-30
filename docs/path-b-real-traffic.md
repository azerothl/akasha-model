# Path B: train on host authorize traffic (beyond gate-tiny)

**Issue:** [#44](https://github.com/azerothl/akasha-model/issues/44)  
**Status:** ops / host-side — optional until a host needs Path B in production.  
**Not a release blocker** for the Path A gate wedge.

The committed `examples/gate/checkpoints/gate-tiny.pt` is **demo-only**
(synthetic JSONL). Production Path B needs labeled authorize rows from the
host (Akasha OS or another runtime). **Do not** commit private traffic,
credentials, or production weights to this repository (`data/` is gitignored).

## Checkpoint policy

| Checkpoint | Role |
|------------|------|
| `examples/gate/checkpoints/gate-tiny.pt` | Demo / CI only |
| Host-trained weights under `runs/` or host secret store | Production Path B |

Document which file the host loads at serve time. Never replace the committed
demo checkpoint with private weights in a public PR.

## Host export shape (hooks)

Export one JSON object per authorize decision (or per proposal+outcome). The
importer in `examples/gate/import_authorize_export.py` accepts this **public
schema** (no OS-private fields required):

```json
{
  "situation_id": "sess-42:fs.read:notes",
  "split_group": "family:safe_read",
  "situation": "User asks to open their notes.",
  "proposal": {"tool_name": "fs.read", "arguments": {"path": "notes.txt"}},
  "confirmation_given": false,
  "expected_status": "ready",
  "labels": {
    "route_tool": "fs.read",
    "risk": 0,
    "authorized": 1,
    "sufficient_context": 1,
    "capability_present": 1,
    "confirmation_needed": 0
  }
}
```

Notes:

- `situation_id` / `split_group` must be stable keys for **anti-leakage**
  splits (train / val / test must not share the same identity).
- Labels are teacher / human / post-hoc judgments for multitask heads.
- Prefer scrubbing PII before the file leaves the host trust boundary.
- Do **not** mix Hub typed-decisions leaderboard rows into this train set.

Synthetic fixture (shape only, not real OS traffic):
`examples/gate/fixtures/host_authorize_export.jsonl`.

## Pipeline

```sh
# 1) On the host: export scrubbed JSONL (stay out of git).
# 2) Import into multitask authorize splits:
python examples/gate/import_authorize_export.py \
  --input data/gate/host_export.jsonl \
  --output data/gate/from_host \
  --catalog examples/gate  # uses examples/gate/catalog.py tool set

# 3) Train / eval (same as synthetic Path B):
uv pip install -e '.[dev,torch]'
python examples/gate/train_tiny.py --data data/gate/from_host --output runs/gate-prod.pt
python examples/gate/scored_demo.py --checkpoint runs/gate-prod.pt

# 4) Go/no-go on held-out split + outcomes calibration:
python examples/gate/outcomes_demo.py --log data/gate/outcomes.jsonl
```

Full checklist: [examples/gate/README.md](../examples/gate/README.md)
(Production Path B). User guide: [using-the-tool-gate.md](using-the-tool-gate.md).

## Go / no-go

Re-run the false-positive / false-negative matrix on **held-out** host
traffic (not the training batch). Record whether `gate-tiny` is replaced for
that host. Fail if safe reads are blocked or deletes without confirmation are
`ready`.

## What this repo will not do

- Invent or commit Akasha OS private authorize logs
- Ship production checkpoints
- Claim the demo tiny scorer is calibrated for real traffic
