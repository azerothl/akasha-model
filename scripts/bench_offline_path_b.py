#!/usr/bin/env python3
"""Benchmark offline Path B checkpoint load (no network)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from akasha_model.offline import (
    deny_network,
    load_offline_torch_checkpoint,
    measure_rss_mb,
    resolve_local_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("examples/gate/checkpoints/gate-tiny.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/offline_path_b.json"))
    args = parser.parse_args()

    resolve_local_checkpoint(args.checkpoint)
    rss_before = measure_rss_mb()
    t0 = time.perf_counter()
    with deny_network(True):
        payload, meta = load_offline_torch_checkpoint(args.checkpoint, deny_net=False)
    elapsed = time.perf_counter() - t0
    rss_after = measure_rss_mb()
    report = {
        "checkpoint": meta.path,
        "format": meta.format,
        "bytes": meta.bytes,
        "load_latency_s": round(elapsed, 4),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "has_state_dict": isinstance(payload, dict) and "state_dict" in payload,
        "network": "denied_during_guard_test",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
