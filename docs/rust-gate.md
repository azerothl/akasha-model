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
| Wire JSON | `akasha_model/schemas/` + `CONTRACT_VERSION` (`docs/contracts/`) |

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

## Status

<<<<<<< HEAD
Base Path A is in the crate: `evaluate_gate`, multi-cap `required_capabilities`,
and wire `contract_version`. Shared golden fixtures live under
`crates/akasha-gate/tests/fixtures/` and `tests/contracts/fixtures/`.

### C ABI (`cdylib`) for non-Rust hosts

Build a shared library (no CPython, no torch, no tool executor):

```sh
cargo build -p akasha-gate --release
# → target/release/libakasha_gate.so (Linux) / .dylib (macOS) / .dll (Windows)
```

Header: [`crates/akasha-gate/include/akasha_gate.h`](../crates/akasha-gate/include/akasha_gate.h).

| Symbol | Role |
|--------|------|
| `akasha_gate_abi_version` | Static `"1"` string (do not free) |
| `akasha_gate_contract_version` | Wire `CONTRACT_VERSION` (`1`) |
| `akasha_gate_evaluate_json` | UTF-8 JSON request → heap JSON response |
| `akasha_gate_string_free` | Free evaluate responses only |

Request shape matches the Path A wire envelope (`tools` + `proposal` +
`signals`, optional `contract_version` / `request_id`). Response includes
`ok`, `status`, plan fields, and echoed `request_id`. Failures return
`{"ok":false,"error":...}` (still allocated). **Never** executes tools —
hosts keep ACLs and side effects outside the library.

Threading: re-entrant; no process-wide mutable gate state.

Smoke:

```sh
cargo build -p akasha-gate
cargo test -p akasha-gate
python3 crates/akasha-gate/scripts/ffi_ctypes_smoke.py
```

## Follow-ups (optional)

- Extended Path A parity with Python: `CatalogPolicy`, `AuthorityProfile`,
  `budget_remaining` (and optionally audit envelopes) — [#45](https://github.com/azerothl/akasha-model/issues/45)
=======
Base Path A plus extended parity with Python:

- `evaluate_gate` / `evaluate_gate_with`
- multi-cap `required_capabilities`
- wire `contract_version`
- `CatalogPolicy` (allow / deny / placement)
- `AuthorityProfile` (escalate / reject / clarify reason codes)
- `budget_remaining` on `GateSignals` (`budget_exceeded` when ≤ 0)

Shared golden fixtures live under `crates/akasha-gate/tests/` and
`tests/contracts/fixtures/`. Audit envelopes remain optional / host-side.

## Follow-ups (optional)

- Optional FFI `cdylib` for non-Rust hosts — [#49](https://github.com/azerothl/akasha-model/issues/49)
- Audit envelope fields on the Rust wire (optional; Python has them)
>>>>>>> 71408b3 (Add Rust Path A catalog, authority, and budget parity.)
- Publish `akasha-gate` on crates.io (git dep is enough today)
