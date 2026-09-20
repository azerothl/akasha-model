"""JSONL loading, byte tokenisation and small public data builders."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import torch
from torch.utils.data import Dataset

from .primitives import OptionSpec


@dataclass(frozen=True)
class ChoiceExample:
    context: str
    options: tuple[str, ...]
    label: int
    option_descriptions: tuple[str, ...] = ()

    def option_specs(self) -> tuple[OptionSpec, ...]:
        descriptions = self.option_descriptions or ("",) * len(self.options)
        return tuple(
            OptionSpec(name=name, description=description)
            for name, description in zip(self.options, descriptions)
        )

    def option_texts(self) -> tuple[str, ...]:
        return tuple(option.model_text() for option in self.option_specs())


def validate(payload: dict) -> ChoiceExample:
    context, options, label = payload.get("context"), payload.get("options"), payload.get("label")
    if not isinstance(context, str) or not isinstance(options, list):
        raise ValueError("each row needs string context and list options")
    if len(options) < 2 or len(options) > 255:
        raise ValueError("options must contain between 2 and 255 entries")
    names, descriptions = [], []
    for option in options:
        if isinstance(option, str):
            name, description = option, ""
        elif isinstance(option, dict):
            name, description = option.get("name"), option.get("description", "")
            if not isinstance(description, str):
                raise ValueError("option description must be a string")
        else:
            raise ValueError("options must contain strings or option objects")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("option names must be non-empty strings")
        names.append(name)
        descriptions.append(description)
    if len(set(names)) != len(names):
        raise ValueError("option names must be unique")
    if not isinstance(label, int) or not 0 <= label < len(options):
        raise ValueError("label must be an option index")
    return ChoiceExample(context, tuple(names), label, tuple(descriptions))


class JsonlDataset(Dataset[ChoiceExample]):
    def __init__(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as handle:
            self.examples = [validate(json.loads(line)) for line in handle if line.strip()]
        if not self.examples:
            raise ValueError(f"no examples in {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _bytes(text: str, length: int) -> list[int]:
    return [byte + 1 for byte in text.encode("utf-8", errors="replace")[:length]]


class ByteCollator:
    def __init__(self, context_tokens: int, option_tokens: int) -> None:
        self.context_tokens = context_tokens
        self.option_tokens = option_tokens

    def __call__(self, examples: list[ChoiceExample]):
        contexts = [_bytes(item.context, self.context_tokens) for item in examples]
        option_rows = [[_bytes(option, self.option_tokens) for option in item.option_texts()]
                       for item in examples]
        return _tensor_batch(examples, contexts, option_rows, 0)


class HuggingFaceCollator:
    def __init__(self, tokenizer, context_tokens: int, option_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.context_tokens = context_tokens
        self.option_tokens = option_tokens

    def __call__(self, examples: list[ChoiceExample]):
        contexts = self.tokenizer(
            [item.context for item in examples], truncation=True,
            max_length=self.context_tokens, add_special_tokens=True,
        )["input_ids"]
        flat = [option for item in examples for option in item.option_texts()]
        encoded = self.tokenizer(
            flat, truncation=True, max_length=self.option_tokens, add_special_tokens=True,
        )["input_ids"]
        rows, offset = [], 0
        for item in examples:
            rows.append(encoded[offset:offset + len(item.options)])
            offset += len(item.options)
        return _tensor_batch(examples, contexts, rows, self.tokenizer.pad_token_id)


def _tensor_batch(examples, contexts, option_rows, pad_id):
    batch, max_context = len(examples), max(map(len, contexts))
    max_options = max(len(row) for row in option_rows)
    max_option_tokens = max(len(tokens) for row in option_rows for tokens in row)
    context_ids = torch.full((batch, max_context), pad_id, dtype=torch.long)
    option_ids = torch.full(
        (batch, max_options, max_option_tokens), pad_id, dtype=torch.long
    )
    option_mask = torch.zeros((batch, max_options), dtype=torch.bool)
    for row, tokens in enumerate(contexts):
        context_ids[row, :len(tokens)] = torch.tensor(tokens)
    for row, options in enumerate(option_rows):
        option_mask[row, :len(options)] = True
        for column, tokens in enumerate(options):
            option_ids[row, column, :len(tokens)] = torch.tensor(tokens)
    return {
        "context_ids": context_ids, "context_mask": context_ids.ne(pad_id),
        "option_ids": option_ids, "option_token_mask": option_ids.ne(pad_id),
        "option_mask": option_mask,
        "labels": torch.tensor([item.label for item in examples], dtype=torch.long),
    }


COLOURS = ("amber", "azure", "bronze", "coral", "crimson", "gold", "green", "indigo")
ANIMALS = ("badger", "crane", "dolphin", "falcon", "gecko", "heron", "ibis", "jaguar")


AKASHA_ACTIONS = (
    "session.open", "session.fork", "memory.lookup", "notes.search",
    "notes.delete", "task.schedule", "agent.run", "skill.invoke",
    "tool.request", "model.load", "model.migrate", "media.image",
    "media.audio", "canvas.compose", "canvas.export", "device.capture",
    "device.usb", "capability.check", "module.install", "module.ui",
    "health.snapshot", "troubleshoot.report", "update.apply", "mcp.bridge",
    "harness.run", "feedback.send",
)


def _akasha_sample(seed: int) -> tuple[ChoiceExample, str, str]:
    """Build one synthetic policy decision inspired by azerothl/akasha-os.

    The label is generated from an explicit, documented routing policy rather
    than from a language model. This makes the dataset reproducible and keeps
    it suitable for testing a choice model without copying project data.
    """
    rng = random.Random(seed)
    scenarios = (
        ("start a new conversation", "chat", "local", "trusted", "no session is selected", "open a persisted session", "session.open"),
        ("continue from an earlier message", "chat", "local", "trusted", "the thread has a useful branch", "fork the conversation here", "session.fork"),
        ("recall a user's preference", "memory", "local", "trusted", "facts exist from previous turns", "bootstrap the agent with saved facts", "memory.lookup"),
        ("find a note by tag", "notes", "local", "trusted", "the collection is large", "search the notes index", "notes.search"),
        ("remove an obsolete note", "notes", "local", "trusted", "the user confirmed deletion", "delete the selected note", "notes.delete"),
        ("run a reminder tomorrow", "tasks", "local", "trusted", "the schedule is durable", "create a scheduled task", "task.schedule"),
        ("complete a multi-step goal", "agents", "local", "trusted", "the goal loop is enabled", "run a bounded background agent", "agent.run"),
        ("summarize a document", "chat", "local", "trusted", "a matching skill is installed", "invoke the workflow skill", "skill.invoke"),
        ("fetch a public webpage", "chat", "network", "untrusted", "network access is opt-in", "request the web tool", "tool.request"),
        ("load a model for a low VRAM machine", "models", "gpu", "trusted", "the selected pack is not resident", "load the compatible model pack", "model.load"),
        ("move an active completion from GPU to CPU", "models", "gpu", "trusted", "the GPU is under pressure", "migrate the running model", "model.migrate"),
        ("generate a poster", "create", "gpu", "trusted", "an image pack is installed", "generate an image", "media.image"),
        ("read a generated answer aloud", "chat", "local", "trusted", "a voice pack is available", "generate audio with TTS", "media.audio"),
        ("arrange shapes on a visual board", "canvas", "gpu", "trusted", "scene validation is enabled", "compose the canvas scene", "canvas.compose"),
        ("save a board for another app", "canvas", "local", "trusted", "the scene is valid", "export PNG, SVG and JSON", "canvas.export"),
        ("take a webcam snapshot", "device", "local", "restricted", "the user has not granted permission", "request device confirmation", "device.capture"),
        ("read a serial sensor", "device", "usb", "restricted", "USB capability is absent", "request the USB capability", "device.usb"),
        ("write to a protected folder", "security", "local", "restricted", "the path is outside the allowlist", "verify capabilities before execution", "capability.check"),
        ("install a community module", "settings", "network", "untrusted", "the signature has not been checked", "review and verify the module", "module.install"),
        ("show a module's form and table", "module", "local", "trusted", "the module declares declarative UI", "render its widget tree", "module.ui"),
        ("inspect a slow inference service", "diagnostics", "local", "trusted", "latency residuals are abnormal", "collect a health snapshot", "health.snapshot"),
        ("prepare a bug report", "diagnostics", "local", "trusted", "diagnostics found actionable findings", "open a GitHub report", "troubleshoot.report"),
        ("apply a downloaded release", "updates", "local", "trusted", "the overlay hash is valid", "apply the update on next launch", "update.apply"),
        ("connect an external IDE", "integration", "local", "trusted", "the MCP façade is enabled", "serve the model and memory tools", "mcp.bridge"),
        ("let a coding CLI edit a repository", "agents", "local", "restricted", "the harness is explicitly enabled", "run the external coding harness", "harness.run"),
        ("send a structured product report", "feedback", "network", "untrusted", "the user attached diagnostics", "submit local feedback", "feedback.send"),
    )
    task, surface, resource, trust, state, goal, target = rng.choice(scenarios)
    candidates = [target]
    candidates.extend(rng.sample([item for item in AKASHA_ACTIONS if item != target], 7))
    rng.shuffle(candidates)
    signal = rng.choice((
        "fresh request", "after restart", "user confirmed", "user cancelled",
        "low context", "high context", "queue empty", "queue busy",
        "model ready", "model missing", "permission pending", "permission granted",
        "network disabled", "network allowed", "disk low", "disk healthy",
        "gpu idle", "gpu busy", "audit required", "audit clean",
        "retryable failure", "signed artifact", "offline mode", "interactive mode",
    ))
    templates = (
        "Request={task}; surface={surface}; resource={resource}; trust={trust}; state={state}; signal={signal}. Action={goal}.",
        "User asks: {task}. surface={surface}, resource={resource}, trust={trust}, state={state}, signal={signal}. Choose: {goal}.",
        "Route {task}. Runtime: {surface}/{resource}; policy={trust}; state={state}; signal={signal}. Next: {goal}.",
        "Task={task}. Capability review applies. surface={surface}; resource={resource}; trust={trust}; state={state}; signal={signal}. Goal={goal}.",
    )
    item = ChoiceExample(
        context=rng.choice(templates).format(
            task=task, surface=surface, resource=resource, trust=trust,
            state=state, signal=signal, goal=goal,
        ) + " Offline-first, capability-based OS.",
        options=tuple(candidates),
        label=candidates.index(target),
    )
    return item, task, signal


def akasha_example(seed: int) -> ChoiceExample:
    """Build one synthetic Akasha-OS-style routing example."""
    return _akasha_sample(seed)[0]


def write_akasha_os(
    output: Path, sizes: dict[str, int], seed: int, split_by: str = "family"
) -> None:
    """Write grouped splits without putting a family/context in two files."""
    if split_by not in {"family", "context"}:
        raise ValueError("split_by must be 'family' or 'context'")
    output.mkdir(parents=True, exist_ok=True)
    rows = {split: [] for split in sizes}
    attempts = 0
    while any(len(rows[split]) < size for split, size in sizes.items()):
        item, family, signal = _akasha_sample(seed + attempts * 104729)
        group = family if split_by == "family" else f"{family}|{signal}"
        bucket = _stable(group + ":akasha-split") % 10
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        if len(rows[split]) < sizes[split]:
            rows[split].append({
                "context": item.context, "options": item.options, "label": item.label,
            })
        attempts += 1
        if attempts > max(sizes.values()) * 1000:
            raise RuntimeError("could not fill grouped Akasha-OS splits")
    for split, payloads in rows.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload) + "\n")


def synthetic_example(seed: int) -> ChoiceExample:
    rng = random.Random(seed)
    target = f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}"
    count = rng.randint(2, 8)
    options = {target}
    while len(options) < count:
        options.add(f"{rng.choice(COLOURS)} {rng.choice(ANIMALS)}")
    options = list(options)
    rng.shuffle(options)
    notes = " ".join(rng.choice(("north", "south", "east", "west")) for _ in range(8))
    context = f"Choose the exact badge {target}. Notes: {notes}. Badge: {target}."
    return ChoiceExample(context, tuple(options), options.index(target))


def write_synthetic(output: Path, sizes: dict[str, int], seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    offset = 0
    for split, size in sizes.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(size):
                item = synthetic_example(seed + offset + index * 104729)
                handle.write(json.dumps({
                    "context": item.context, "options": item.options, "label": item.label,
                }) + "\n")
        offset += size * 104729


def _stable(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _title(text: str) -> str:
    return unquote(text).replace("_", " ")


def build_wikispeedia(root: Path, output: Path, max_options: int = 64) -> None:
    graph_dir = root / "wikispeedia_paths-and-graph"
    outgoing: dict[str, list[str]] = {}
    for line in (graph_dir / "links.tsv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            source, target = line.split("\t")
            outgoing.setdefault(source, []).append(target)
    output.mkdir(parents=True, exist_ok=True)
    handles = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in ("train", "validation", "test")}
    counts = {name: 0 for name in handles}
    try:
        lines = (graph_dir / "paths_finished.tsv").read_text(encoding="utf-8").splitlines()
        for row, line in enumerate(lines):
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            path = []
            for node in fields[3].split(";"):
                if node == "<":
                    if len(path) > 1:
                        path.pop()
                else:
                    path.append(node)
            if len(path) < 2:
                continue
            step = _stable(f"{fields[0]}:{fields[1]}:{row}") % (len(path) - 1)
            current, click, target = path[step], path[step + 1], path[-1]
            candidates = list(dict.fromkeys(outgoing.get(current, ())))
            if len(candidates) < 2 or click not in candidates:
                continue
            rng = random.Random(_stable(f"{row}:{target}:menu"))
            others = [item for item in candidates if item != click]
            rng.shuffle(others)
            menu = [click] + others[:max_options - 1]
            rng.shuffle(menu)
            article = root / "plaintext_articles" / f"{current}.txt"
            body = " ".join(article.read_text(encoding="utf-8", errors="replace").split())
            payload = {
                "context": f"Target article: {_title(target)}\nCurrent article: {_title(current)}\n{body[:2048]}",
                "options": [_title(item) for item in menu], "label": menu.index(click),
            }
            bucket = _stable(target + ":split") % 10
            split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
            handles[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    print(json.dumps(counts, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("--output", type=Path, default=Path("data/synthetic"))
    synthetic.add_argument("--train", type=int, default=2000)
    synthetic.add_argument("--validation", type=int, default=400)
    synthetic.add_argument("--test", type=int, default=400)
    synthetic.add_argument("--seed", type=int, default=17)
    akasha = commands.add_parser("akasha-os", help="build an Akasha-OS-style routing dataset")
    akasha.add_argument("--output", type=Path, default=Path("data/akasha_os"))
    akasha.add_argument("--train", type=int, default=2600)
    akasha.add_argument("--validation", type=int, default=520)
    akasha.add_argument("--test", type=int, default=520)
    akasha.add_argument("--seed", type=int, default=23)
    akasha.add_argument(
        "--split-by", choices=("family", "context"), default="family",
        help="keep each scenario family or family+runtime-context group in one split",
    )
    wiki = commands.add_parser("wikispeedia")
    wiki.add_argument("--root", type=Path, required=True)
    wiki.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "synthetic":
        write_synthetic(args.output, {
            "train": args.train, "validation": args.validation, "test": args.test,
        }, args.seed)
    elif args.command == "akasha-os":
        write_akasha_os(args.output, {
            "train": args.train, "validation": args.validation, "test": args.test,
        }, args.seed, args.split_by)
    else:
        build_wikispeedia(args.root, args.output)


if __name__ == "__main__":
    main()
