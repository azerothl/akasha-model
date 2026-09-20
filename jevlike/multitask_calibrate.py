"""Fit post-hoc temperatures and a Platt calibrator for multitask heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import select_device
from .multitask import MultiQuestionCollator, MultiQuestionDataset, MultiQuestionTinyScorer


def _load_checkpoint(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = MultiQuestionTinyScorer(
        width=config["width"], rank=config["rank"],
        context_tokens=config["context_tokens"],
        question_tokens=config["question_tokens"],
        max_score_levels=config.get("max_score_levels", 10),
        encoder=config.get("encoder", "mean"),
        lexical_multiplier=config.get("lexical_multiplier", 64.0),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


def _move(batch: dict, device: torch.device) -> dict:
    return {kind: {name: value.to(device) if isinstance(value, torch.Tensor) else value
                   for name, value in values.items()}
            for kind, values in batch.items()}


def _masked(logits: list[torch.Tensor], masks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(item.shape[1] for item in logits)
    padded, padded_masks = [], []
    for values, mask in zip(logits, masks):
        padding = width - values.shape[1]
        padded.append(F.pad(values, (0, padding), value=-1e4))
        padded_masks.append(F.pad(mask, (0, padding), value=False))
    return torch.cat(padded), torch.cat(padded_masks)


@torch.no_grad()
def collect(model, path: str | Path, device: torch.device, batch_size: int) -> dict[str, tuple[torch.Tensor, ...]]:
    dataset = MultiQuestionDataset(path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=MultiQuestionCollator(
            model.context_position.num_embeddings,
            model.question_position.num_embeddings,
            384, model.max_score_levels,
        ),
    )
    collected = {"choice": [[], [], []], "score": [[], [], []], "noul": [[], []]}
    model.eval()
    for host_batch in loader:
        batch = _move(host_batch, device)
        outputs = model(batch)
        for kind in ("choice", "score"):
            if kind in outputs:
                collected[kind][0].append(outputs[kind].cpu())
                collected[kind][1].append(batch[kind]["labels"].cpu())
                mask_name = "option_mask" if kind == "choice" else "level_mask"
                collected[kind][2].append(batch[kind][mask_name].cpu())
        if "noul" in outputs:
            collected["noul"][0].append(outputs["noul"].cpu())
            collected["noul"][1].append(batch["noul"]["labels"].cpu())
    result = {}
    for kind in ("choice", "score"):
        logits, masks = _masked(collected[kind][0], collected[kind][2])
        result[kind] = (logits, torch.cat(collected[kind][1]), masks)
    result["noul"] = (torch.cat(collected["noul"][0]), torch.cat(collected["noul"][1]))
    return result


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    parameter = torch.zeros((), requires_grad=True)
    optimiser = torch.optim.Adam([parameter], lr=0.05)
    masked_logits = logits.masked_fill(~mask, -1e4)
    for _ in range(300):
        optimiser.zero_grad()
        temperature = parameter.clamp(-5.0, 5.0).exp()
        loss = F.cross_entropy(masked_logits / temperature, labels)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            parameter.clamp_(-5.0, 5.0)
    return float(parameter.detach().clamp(-5.0, 5.0).exp())


def fit_platt(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    log_scale = torch.zeros((), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimiser = torch.optim.Adam([log_scale, bias], lr=0.03)
    labels = labels.float()
    for _ in range(400):
        optimiser.zero_grad()
        scale = log_scale.clamp(-5.0, 5.0).exp()
        loss = F.binary_cross_entropy_with_logits(logits * scale + bias, labels)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            log_scale.clamp_(-5.0, 5.0)
            bias.clamp_(-10.0, 10.0)
    return float(log_scale.detach().clamp(-5.0, 5.0).exp()), float(bias.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("calibration_data")
    parser.add_argument("--output", default="runs/akasha_os_multi_v3_calibrated.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    model, payload = _load_checkpoint(args.checkpoint, device)
    collected = collect(model, args.calibration_data, device, args.batch_size)
    choice_temperature = fit_temperature(collected["choice"][0], collected["choice"][1], collected["choice"][2])
    score_temperature = fit_temperature(collected["score"][0], collected["score"][1], collected["score"][2])
    noul_scale, noul_bias = fit_platt(*collected["noul"])
    calibration = {
        "choice_temperature": choice_temperature,
        "score_temperature": score_temperature,
        "noul_scale": noul_scale,
        "noul_bias": noul_bias,
        "fit_split": str(args.calibration_data),
    }
    payload["config"].setdefault("encoder", model.encoder)
    payload["config"].setdefault("lexical_multiplier", model.lexical_multiplier)
    payload["config"]["calibration"] = calibration
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(json.dumps({"checkpoint": str(output), "device": str(device), "calibration": calibration}, ensure_ascii=False))


if __name__ == "__main__":
    main()
