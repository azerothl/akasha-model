# akasha-gate (Rust)

Native Path A authorization gate for Akasha OS — `evaluate_gate` /
`dispatch_plan` without spawning Python.

| Surface | This crate | Python package |
|---------|------------|----------------|
| Path A scripted signals | ✅ | `akasha_model.gate` / `host` / `tool_calling` |
| Path B scored multitask | ❌ (stay in Python + torch) | `akasha_model.gate_multitask` |
| Tool execution | ❌ (host owns side effects) | ❌ |

## Depend from Akasha OS

```toml
# Cargo.toml (path or git; crates.io later)
akasha-gate = { git = "https://github.com/azerothl/akasha-model", package = "akasha-gate" }
```

From a checkout:

```toml
akasha-gate = { path = "../akasha-model/crates/akasha-gate" }
```

## Parity

Golden fixtures under `tests/fixtures/` match Python Path A status/reason
classes. Run:

```sh
cargo test -p akasha-gate
```

## C ABI (`cdylib`)

Non-Rust hosts can `dlopen` the shared library (header
[`include/akasha_gate.h`](include/akasha_gate.h)):

```sh
cargo build -p akasha-gate --release
python3 crates/akasha-gate/scripts/ffi_ctypes_smoke.py
```

`akasha_gate_evaluate_json` authorizes only — no tool executor inside the
library. Details: [docs/rust-gate.md](../../docs/rust-gate.md).
