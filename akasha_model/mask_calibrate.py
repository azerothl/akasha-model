"""Per-type temperature calibration for MASK checkpoints (#87).

Fits one strictly positive temperature per question type (choice/score/noul)
on a held-out calibration split, then writes a new checkpoint whose config
and ``temperature`` buffer agree. Fitting collects logits with all
temperatures set to 1, so an already-calibrated checkpoint is refitted from
its raw scores. The gate thresholds and deterministic rules are untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpoints import ARCH_MASK, load_checkpoint_payload, stamp_payload
from .decision import MaskCollator, load_decision_checkpoint, save_decision_checkpoint
from .model import select_device
from .multitask import MultiQuestionDataset
from .sequence import QTYPES


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive temperature by minimising NLL of ``logits / T``."""
    if logits.numel() == 0:
        return 1.0
    log_temperature = torch.zeros((), requires_grad=True)
    optimiser = torch.optim.Adam([log_temperature], lr=0.05)
    for _ in range(300):
        optimiser.zero_grad()
        temperature = log_temperature.clamp(-5.0, 5.0).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            log_temperature.clamp_(-5.0, 5.0)
    return float(log_temperature.detach().clamp(-5.0, 5.0).exp())


@torch.no_grad()
def collect_logits(model, loader, device) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Collect raw (T=1) logits and labels grouped by question type."""
    previous = model.temperature.detach().clone()
    model.set_temperature([1.0, 1.0, 1.0])
    model.eval()
    collected: dict[int, list[list[torch.Tensor]]] = {kind: [[], []] for kind in QTYPES.values()}
    try:
        for host_batch in loader:
            batch = {
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in host_batch.items()
            }
            logits = model(
                batch["input_ids"], batch["attention_mask"],
                batch["marker_pos"], batch["marker_mask"], batch["qtype"],
            )
            masked = logits.masked_fill(~batch["marker_mask"], -1e4)
            for kind in QTYPES.values():
                rows = batch["qtype"] == kind
                if bool(rows.any()):
                    collected[kind][0].append(masked[rows].cpu())
                    collected[kind][1].append(batch["label"][rows].cpu())
    finally:
        model.temperature.copy_(previous)
    result = {}
    for kind, (logits_rows, label_rows) in collected.items():
        if logits_rows:
            width = max(row.shape[1] for row in logits_rows)
            padded = [
                F.pad(row, (0, width - row.shape[1]), value=-1e4) for row in logits_rows
            ]
            result[kind] = (torch.cat(padded), torch.cat(label_rows))
    return result


def fit_temperatures(model, loader, device) -> list[float]:
    """Return ``[choice, score, noul]`` temperatures (1.0 when a type is absent)."""
    collected = collect_logits(model, loader, device)
    temperatures = [1.0, 1.0, 1.0]
    for kind, (logits, labels) in collected.items():
        temperatures[kind] = fit_temperature(logits, labels)
    return temperatures


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a MASK checkpoint per question type")
    parser.add_argument("checkpoint")
    parser.add_argument("calibration_data", help="held-out JSONL split (not train/test)")
    parser.add_argument("--output", default="runs/mask-calibrated.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--allow-unsafe-pickle", action="store_true",
        help="only for a fully trusted legacy checkpoint that weights_only cannot read",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    model, tokenizer, config = load_decision_checkpoint(
        args.checkpoint, device, allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    collator = MaskCollator(
        tokenizer,
        max_len=int(config.get("max_len", 768)),
        head_max_len=int(config.get("head_max_len", 384)),
        option_max_len=int(config.get("option_max_len", 48)),
        group_size=1,
        layout=str(config.get("mask_layout", "schema_first")),
        mix_compact_state=bool(config.get("mix_compact_state")),
    )
    loader = DataLoader(
        MultiQuestionDataset(args.calibration_data),
        batch_size=args.batch_size, shuffle=False, collate_fn=collator,
    )
    temperatures = fit_temperatures(model, loader, device)
    model.set_temperature(temperatures)
    new_config = {**config, "temperature": temperatures}
    output = Path(args.output)
    # save_decision_checkpoint re-stamps metadata and syncs config temperature.
    save_decision_checkpoint(output, model, new_config)
    # Keep a record of which split fitted the temperatures.
    payload = load_checkpoint_payload(output)
    payload["calibration"] = {"fit_split": str(args.calibration_data), "temperatures": temperatures}
    torch.save(stamp_payload(payload, ARCH_MASK), output)
    print(json.dumps({
        "checkpoint": str(output),
        "temperatures": {"choice": temperatures[0], "score": temperatures[1], "noul": temperatures[2]},
        "device": str(device),
    }))


if __name__ == "__main__":
    main()
