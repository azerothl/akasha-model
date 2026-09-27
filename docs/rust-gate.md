# Rust Path A gate (`akasha-gate`)

## Why

Akasha OS agent runtimes are Rust-first. Shelling out to
`python -m aos_gate.cli evaluate` per call adds process spawn, env probing, and
fails closed when Python is missing. Path A with **explicit** `GateSignals` is
deterministic (`ToolCallPlanner` + shallow JSON-schema checks) and does not
need torch — a natural native port.

## Split

| Concern | Lives in |
|---------|----------|
| Path A `evaluate_gate` / `dispatch_plan` / `ToolSpec` | Rust crate `akasha-gate` **and** Python `akasha_model` (parity) |
| Path B scored proposals / MASK / RLCD | Python only (`akasha_model[torch]`) |
| OS permissions / ACLs / execution | Akasha OS host (`ToolHost` / `AkashaOsToolHost`) |
| Wire JSON | `akasha_model/schemas/` + `CONTRACT_VERSION` (see #12) |

**Non-negotiable:** no tool executor inside this crate. `dispatch_plan` calls a
host-supplied `ToolHost` trait only after `ready` + `check_permissions`.

## OS integration

```toml
akasha-gate = { git = "https://github.com/azerothl/akasha-model", package = "akasha-gate" }
```

Replace `Command("python -m aos_gate.cli evaluate")` with an in-process
`evaluate_gate` / `run_gated_call`. Keep Python for Path B and training.

## Parity policy

Rust and Python Path A must agree on `status` and reason **class** for the
shared golden fixtures (`crates/akasha-gate/tests/fixtures/` and
`tests/contracts/fixtures/` when present). Exact floating confidence strings
may differ slightly; status and block/abstain reasons are authoritative.

## Follow-ups

- Multi-cap `required_capabilities` + wire `contract_version` once #12/#14 land
- Optional FFI `cdylib` if a non-Rust host needs a `.so` without embedding CPython
