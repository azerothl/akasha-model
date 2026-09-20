"""Post-hoc temperature calibration for one-pass choice models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import JsonlDataset
from .model import load_checkpoint, select_device
from .train import move


def _pad_logits(items: list[torch.Tensor]) -> torch.Tensor:
    width = max(item.shape[1] for item in items)
    fill = -1e4
    return torch.cat([
        F.pad(item.clamp_min(fill), (0, width - item.shape[1]), value=fill)
        for item in items
    ])


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for host_batch in loader:
        batch = move(host_batch, device)
        logits.append(model(batch).cpu())
        labels.append(batch["labels"].cpu())
    return _pad_logits(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive temperature by minimizing held-out NLL."""
    log_temperature = torch.zeros((), requires_grad=True)
    optimiser = torch.optim.Adam([log_temperature], lr=0.05)
    for _ in range(200):
        optimiser.zero_grad()
        temperature = log_temperature.clamp(-5.0, 5.0).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            log_temperature.clamp_(-5.0, 5.0)
    return float(log_temperature.detach().clamp(-5.0, 5.0).exp())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("calibration_data")
    parser.add_argument("--output", default="runs/calibrated.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    model, collator, config = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(
        JsonlDataset(args.calibration_data), batch_size=args.batch_size,
        collate_fn=collator,
    )
    logits, labels = collect_logits(model, loader, device)
    old_temperature = float(config.get("temperature", 1.0))
    temperature = fit_temperature(logits, labels) * old_temperature

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    payload["config"]["temperature"] = temperature
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({
        "checkpoint": str(output),
        "calibration_examples": int(labels.numel()),
        "old_temperature": old_temperature,
        "temperature": temperature,
        "device": str(device),
    }))


if __name__ == "__main__":
    main()
