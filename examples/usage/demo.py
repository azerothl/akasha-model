"""First-run usage demo: Path A tool gate + ticket-router API shape.

CPU only, no torch, no Hub download, no side effects.

Prefer the visual walkthrough (browser HTML with probability bars):

```sh
python examples/usage/visual_demo.py --open
```

Text-only (this script):

```sh
python examples/usage/demo.py
```

Deeper gate demos live under ``examples/gate/``; training / Path B need the
``[torch]`` extra.
"""

from __future__ import annotations

from decision_trace import build_trace


def main() -> None:
    trace = build_trace()
    print("Akasha Model — first-run usage demo")
    print("Path A only: no torch, no Hub, no side effects.")
    print("Tip: python examples/usage/visual_demo.py --open\n")

    current_section = None
    for index, step in enumerate(trace["steps"], start=1):
        if step["section"] != current_section:
            current_section = step["section"]
            print(f"=== {current_section} ===")
            if current_section == "Tool gate":
                print(
                    "System 2 proposes a call; evaluate_gate returns "
                    "ready/abstain/blocked.\n"
                )
            else:
                print("Same typed-menu API; host policy owns escalation.\n")
        verdict = step["verdict"]
        print(f"## {index}. {step['title']}")
        print(f"  {verdict['summary']} | host: {verdict['host']}")
        print()

    print("Next steps")
    print("  python examples/usage/visual_demo.py --open  # probability bars")
    print("  python examples/gate/demo.py               # more gate scenarios")
    print("  python examples/gate/host_demo.py          # fake ToolHost dispatch")
    print("  docs/using-the-tool-gate.md                # Path A–D integrator guide")
    print("  docs/use-cases.md                          # triage / multitask / vision")


if __name__ == "__main__":
    main()
