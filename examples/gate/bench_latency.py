"""Measure Path A evaluate_gate / dispatch_plan latency (CPU, no torch).

Reports p50 / p95 in milliseconds for the smoke catalog. Aspirational SLO
(not a CI flake gate): p95 evaluate ≪ typical System 2 proposal time;
target p95 evaluate < 5 ms on a small catalog on modern CPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from akasha_model.gate import GateSignals, ToolProposal, evaluate_gate
from akasha_model.host import dispatch_plan
from akasha_model.tool_calling import ToolSpec


def _catalog() -> dict[str, ToolSpec]:
    return {
        "fs.read": ToolSpec(
            "fs.read",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capability="workspace_access",
        ),
        "fs.delete": ToolSpec("fs.delete", irreversible=True),
        "mail.reply": ToolSpec("mail.reply"),
        "shell.run": ToolSpec("shell.run", required_capability="shell"),
    }


class _NullHost:
    def check_permissions(self, tool_name, arguments):
        return True, "ok"

    def execute(self, tool_name, arguments):
        return {"ok": True}


def _percentile(sorted_ms: list[float], p: float) -> float:
    if not sorted_ms:
        return 0.0
    index = min(len(sorted_ms) - 1, max(0, int(round((p / 100.0) * (len(sorted_ms) - 1)))))
    return sorted_ms[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()

    tools = _catalog()
    proposal = ToolProposal("fs.read", {"path": "notes.txt"})
    signals = GateSignals(
        authorized=0.95,
        sufficient_context=0.9,
        capability_present=0.92,
        confirmation_needed=0.05,
    )
    host = _NullHost()

    for _ in range(args.warmup):
        plan = evaluate_gate(tools, proposal, signals)
        dispatch_plan(plan, host)

    eval_ms: list[float] = []
    dispatch_ms: list[float] = []
    for _ in range(args.iters):
        start = time.perf_counter()
        plan = evaluate_gate(tools, proposal, signals)
        eval_ms.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        dispatch_plan(plan, host)
        dispatch_ms.append((time.perf_counter() - start) * 1000.0)

    eval_ms.sort()
    dispatch_ms.sort()
    report = {
        "iters": args.iters,
        "catalog_size": len(tools),
        "evaluate_ms": {
            "p50": _percentile(eval_ms, 50),
            "p95": _percentile(eval_ms, 95),
            "mean": statistics.fmean(eval_ms),
        },
        "dispatch_ms": {
            "p50": _percentile(dispatch_ms, 50),
            "p95": _percentile(dispatch_ms, 95),
            "mean": statistics.fmean(dispatch_ms),
        },
        "target": {
            "evaluate_p95_ms": 5.0,
            "note": "aspirational on small catalogs; not a CI fail gate",
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
