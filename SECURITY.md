# Security

## Reporting

Report vulnerabilities privately to the repository owner via a
[GitHub security advisory](https://github.com/azerothl/akasha-model/security/advisories/new)
rather than a public issue.

## Checkpoint files are code-bearing artifacts

Historically, Akasha checkpoints were read with
`torch.load(..., weights_only=False)`, which executes arbitrary Python
during unpickling. Since this change:

- Every package loader reads checkpoints with `weights_only=True` by
  default and refuses files that require full pickle.
- The unsafe path exists only behind an explicit
  `allow_unsafe_pickle=True` argument, intended solely for legacy
  checkpoints you built yourself or fully trust.
- Never load a `.pt` file from an untrusted source with that flag.

Treat checkpoint files like executables for provenance purposes: record
where they came from (see `docs/licences-poids.md` for shipped weights).

## Scope notes

- Path A (gate/host/outcomes) never executes tool calls itself; hosts do.
- Offline Path B loading denies network access while reading local weights
  (`akasha_model.offline.deny_network`); it is a safety net, not a sandbox.
