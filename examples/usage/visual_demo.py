"""Build a browser HTML walkthrough of Path A gate + ticket decisions.

Run from the repository root after ``uv pip install -e '.[dev]'``:

```sh
python examples/usage/visual_demo.py
```

Writes a self-contained HTML file (default ``examples/usage/out/demo.html``)
with state, typed options, probability bars, and ready/abstain/blocked
rationale. Optional ``--open`` launches the system browser. No torch / Hub.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from decision_trace import build_trace

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "visual.html"
DEFAULT_OUT = ROOT / "out" / "demo.html"


def render_html(trace: dict, template: Path = TEMPLATE) -> str:
    raw = template.read_text(encoding="utf-8")
    marker = "/* DATA */"
    if marker not in raw:
        raise SystemExit(f"template missing {marker!r}: {template}")
    payload = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    return raw.replace(marker, payload, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="HTML output path (default: examples/usage/out/demo.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the written HTML in the default browser",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the structured trace JSON to this path",
    )
    args = parser.parse_args(argv)

    trace = build_trace()
    html = render_html(trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {args.out}")
    print(f"Steps: {len(trace['steps'])} · path {trace['path']}")
    print("Open the HTML in a browser to follow state → signals → verdict.")
    if args.open:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
