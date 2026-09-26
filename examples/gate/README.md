# Tool gate example

Shows the product direction where a System 2 **proposes** a tool call and
Akasha **authorizes**, **abstains**, or **blocks**. Nothing is executed.

The helper is `akasha_model.gate.evaluate_gate`; the deterministic boundary
remains `ToolCallPlanner`.

## Run

From the repository root:

```sh
uv pip install -e '.[dev]'
python examples/gate/demo.py
```

Expected flavours in the output:

| Scenario | Typical status |
|----------|----------------|
| Safe read | `ready` |
| Delete without confirmation | `blocked` |
| Ambiguous mail action | `abstain` |
| High-risk payment | `blocked` (risk) |
| Shell without capability | `blocked` |
| Broadcast with confirmation | `ready` |

Wire a real host the same way Akasha OS does: call your executor only when
`plan.status == "ready"` (or `plan.executable`), and repeat your own permission
checks.
