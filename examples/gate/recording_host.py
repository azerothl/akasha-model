"""Minimal out-of-OS host: permission stub + in-memory fake side effects.

Stand-in for Akasha OS while integrating the gate. Safe for demos: no real
filesystem, shell, mail, or payment calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class RecordingHost:
    """ToolHost that records executions and never touches the real world."""

    capabilities: set[str] = field(default_factory=lambda: {
        "workspace_access", "mail_send",
    })
    # tool_name → required capability (mirror of ToolSpec.required_capability)
    tool_capabilities: dict[str, str | None] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)

    def check_permissions(
        self, tool_name: str, arguments: Mapping[str, Any] | None,
    ) -> tuple[bool, str]:
        required = self.tool_capabilities.get(tool_name)
        if required and required not in self.capabilities:
            return False, f"host capability missing: {required}"
        if tool_name == "shell.run":
            return False, "host policy forbids shell.run in this demo"
        return True, "host permissions ok"

    def execute(
        self, tool_name: str, arguments: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        record = {
            "tool": tool_name,
            "arguments": dict(arguments or {}),
            "note": "fake side effect (demo host only)",
        }
        self.log.append(record)
        return {"ok": True, "echo": record}
