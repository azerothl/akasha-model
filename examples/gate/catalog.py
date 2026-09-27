"""Shared tool catalog for gate demos and authorize-tool-call data."""

from __future__ import annotations

from akasha_model import ToolSpec


def tool_catalog() -> dict[str, ToolSpec]:
    return {
        "fs.read": ToolSpec(
            "fs.read",
            "Read a workspace file.",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capability="workspace_access",
        ),
        "fs.delete": ToolSpec(
            "fs.delete",
            "Delete a workspace file.",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capability="workspace_access",
            irreversible=True,
        ),
        "mail.reply": ToolSpec(
            "mail.reply",
            "Reply to one sender.",
            parameters={
                "type": "object",
                "required": ["to", "body"],
                "properties": {
                    "to": {"type": "string"},
                    "body": {"type": "string"},
                },
                "additionalProperties": False,
            },
            required_capability="mail_send",
        ),
        "mail.broadcast": ToolSpec(
            "mail.broadcast",
            "Send mail to a large distribution list.",
            parameters={
                "type": "object",
                "required": ["list", "body"],
                "properties": {
                    "list": {"type": "string"},
                    "body": {"type": "string"},
                },
                "additionalProperties": False,
            },
            required_capability="mail_send",
            requires_confirmation=True,
        ),
        "payments.charge": ToolSpec(
            "payments.charge",
            "Charge a customer card.",
            parameters={
                "type": "object",
                "required": ["amount_eur", "customer_id"],
                "properties": {
                    "amount_eur": {"type": "number"},
                    "customer_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
            required_capability="payments",
            irreversible=True,
            requires_confirmation=True,
        ),
        "shell.run": ToolSpec(
            "shell.run",
            "Run a shell command in the workspace.",
            parameters={
                "type": "object",
                "required": ["command"],
                "properties": {"command": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capability="shell",
            irreversible=True,
        ),
    }
