"""Score one context against command-line options."""

from __future__ import annotations

import argparse
import json

import torch

from .data import ChoiceExample
from .model import load_checkpoint, select_device
from .primitives import ChoiceQuestion, OptionSpec, choice_result
from .train import move


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--context", required=True)
    parser.add_argument("--option", action="append", required=True)
    parser.add_argument("--option-description", action="append", default=[],
                        help="description matching each --option, in the same order")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="abstain below this probability")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if len(args.option) < 2:
        parser.error("pass --option at least twice")
    if len(args.option_description) not in (0, len(args.option)):
        parser.error("pass one --option-description for every --option")
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")
    device = select_device(args.device)
    model, collator, _ = load_checkpoint(args.checkpoint, device)
    descriptions = tuple(args.option_description or [""] * len(args.option))
    batch = move(collator([ChoiceExample(
        args.context, tuple(args.option), 0, descriptions
    )]), device)
    model.eval()
    with torch.no_grad():
        probabilities = model(batch).softmax(-1)[0, :len(args.option)].cpu().tolist()
    question = ChoiceQuestion(
        question_id="prediction",
        instructions="Select the best matching option.",
        options=tuple(OptionSpec(name, description)
                      for name, description in zip(args.option, descriptions)),
    )
    result = choice_result(
        question, dict(zip(args.option, probabilities)),
        threshold=args.min_confidence,
    )
    print(json.dumps({
        "selected": result.selected,
        "abstained": result.abstained,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
        "question_id": result.question_id,
    }, indent=2))


if __name__ == "__main__":
    main()
