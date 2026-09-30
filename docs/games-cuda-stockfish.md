# Games lab: CUDA / device flags + Stockfish PATH

Lab Doom and chess sit outside the gate product wedge. This note is the
committed env/device contract so agents and humans can run the examples
without tribal knowledge.

Core **text** code stays multi-device (CPU / MPS / CUDA) without forcing the
game scripts onto a GPU. Game CLIs default to small-machine defaults
(`mps` or `cpu`); pass `--device` explicitly for CUDA.

Do **not** commit machine-specific absolute paths, credentials, or run logs.

## Device flags

PyTorch device string passed to `torch.device(...)`. Supported values in
practice: `cpu`, `mps` (Apple Silicon), `cuda` (NVIDIA when available).

| Script | Flag | Default | Notes |
|--------|------|---------|-------|
| `examples/doom/play.py` | `--device` | `mps` | choices: `cpu`, `mps`, `cuda` |
| `examples/doom/audit.py` | `--device` | `mps` | choices: `cpu`, `mps`, `cuda` |
| `examples/doom/train_imitation.py` | `--device` | `mps` | choices: `cpu`, `mps`, `cuda` |
| `examples/doom/train_dagger.py` | `--device` | `mps` | choices: `cpu`, `mps`, `cuda` |
| `examples/doom/train_ppo.py` | `--device` | `mps` | choices: `cpu`, `mps`, `cuda` |
| `examples/doom/train_joint.py` | `--device` | `mps` | choices: `cpu`, `mps`, `cuda` |
| `examples/doom/train_joint_ppo.py` | `--device` | `mps` | choices: `cpu`, `mps`, `cuda` |
| `examples/chess/play.py` | `--device` | `mps` | free string → `torch.device` |
| `examples/chess/eval.py` | `--device` | `mps` | free string → `torch.device` |
| `examples/chess/dagger.py` | `--device` | `cpu` | free string → `torch.device` |
| `examples/chess/train.py` | `--device` | `mps` | free string → `torch.device` |
| Smoke tests | (hardcoded) | `cpu` | CI / local verify |

Examples:

```sh
# Doom on NVIDIA
python examples/doom/play.py examples/doom/checkpoints/deadly-dagger.pt \
  --episodes 2 --game-seconds 10 --device cuda --greedy

# Chess eval on CPU
cd examples/chess
python eval.py checkpoints/chess-dagger1.pt --games 2 --quick --device cpu --sample
```

Smoke and Path A gate tests stay on CPU. CUDA is optional for the vision lab
only.

## Stockfish

Chess labelling and evaluation shells out to a Stockfish UCI engine.

| Expectation | Detail |
|-------------|--------|
| Install | Platform package manager (`stockfish` / `Stockfish` binary) |
| Discovery | `shutil.which("stockfish")`, else the literal `"stockfish"` |
| Env override | None today — put the binary on `PATH` (no absolute paths in repo) |
| Used by | `positions.py`, `dagger.py`, `eval.py` (and anything that opens UCI) |
| Failure | `FileNotFoundError` / engine popen failure if the binary is missing |

Verify:

```sh
command -v stockfish
stockfish <<<'uci' | head -n 5
```

Install the games extra from the repo root, then run chess scripts from
`examples/chess`:

```sh
uv pip install -e '.[dev,games]'
# Install Stockfish and ensure it is on PATH (see verify above).
cd examples/chess && python test_smoke.py
```

`test_smoke.py` exercises the visual scorer on CPU and does **not** require
Stockfish. Position generation and Stockfish opponents do.

## Follow-ups

- [x] Document Doom + chess `--device` / Stockfish PATH (this doc).
- [x] Add `cuda` to Doom CLI `choices` (was CPU/MPS-only).
- [x] Add `--device` to `examples/chess/train.py` (was hardcoded `mps`).
- [ ] Optional: `STOCKFISH` / `AKASHA_STOCKFISH` env override if hosts need a
      non-`PATH` binary without editing scripts (still no machine-specific
      paths committed).
- [ ] Optional: default `--device` to `cpu` on machines without MPS (today
      several defaults are `mps`; pass `--device cpu` explicitly on Linux CI).

## Related

- [AGENTS.md](../AGENTS.md) — Doom / Chess / install
- [examples/doom/README.md](../examples/doom/README.md)
- [examples/chess/README.md](../examples/chess/README.md)
