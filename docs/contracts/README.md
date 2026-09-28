# Tool-gate wire contracts

Versioned JSON Schemas and Python helpers for Akasha OS (and other hosts) that
bridge into Path A without scraping Python dataclasses.

| Artifact | Path |
|----------|------|
| Request envelope | [`akasha_model/schemas/tool-gate-request.schema.json`](../../akasha_model/schemas/tool-gate-request.schema.json) |
| Plan response | [`akasha_model/schemas/tool-call-plan.schema.json`](../../akasha_model/schemas/tool-call-plan.schema.json) |
| Host outcome | [`akasha_model/schemas/host-outcome.schema.json`](../../akasha_model/schemas/host-outcome.schema.json) |
| Python helpers | `akasha_model.wire` (`plan_to_dict` / `plan_from_dict` / `outcome_to_dict` / `request_to_dict`) |
| Golden fixtures | [`tests/contracts/fixtures/`](../../tests/contracts/fixtures/) |

Current `contract_version` / `CONTRACT_VERSION`: **1**.

## Compatibility

- Additive optional fields (e.g. `request_id`) may appear without a version bump
  when they are ignored by older readers.
- Renaming fields, changing status/action enums, or changing required keys
  **requires** bumping `CONTRACT_VERSION` and a short migration note in this file.
- Path A still never executes tools; hosts own `check_permissions` + side effects.

## Vendor into Rust

Copy or submodule the three `*.schema.json` files. Validate CLI JSON against them
before mapping into `ToolCallPlanView` / `HostOutcomeView`. Prefer the Python
helpers (or matching serde types) over hand-rolled field names.

## Migration notes

### v1 (initial)

First published wire shape matching `ToolSpec`, `ToolProposal`, `GateSignals`,
`ToolCallPlan`, and `HostOutcome` as of the gate-host merge.
