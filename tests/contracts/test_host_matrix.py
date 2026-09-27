"""Published Path A matrix fixtures for OS / host integrator CI (no torch)."""

from __future__ import annotations

import json
from pathlib import Path

from akasha_model.gate import evaluate_gate
from akasha_model.host import dispatch_plan
from akasha_model.tool_calling import ToolCallPlanner, ToolSpec
from akasha_model.wire import CONTRACT_VERSION, proposal_from_dict, signals_from_dict

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _base_catalog() -> dict[str, ToolSpec]:
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
            required_capabilities=("workspace_access",),
            irreversible=True,
        ),
    }


def _catalog(kind: str = "base") -> dict[str, ToolSpec]:
    tools = _base_catalog()
    if kind == "with_shell":
        tools["shell.run"] = ToolSpec(
            "shell.run",
            "Run a shell command.",
            parameters={
                "type": "object",
                "required": ["cmd"],
                "properties": {"cmd": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capabilities=("shell", "network"),
        )
    return tools


def test_published_path_a_matrix() -> None:
    matrix = json.loads((FIXTURES / "path_a_matrix.json").read_text(encoding="utf-8"))
    assert matrix["contract_version"] == CONTRACT_VERSION

    for case in matrix["cases"]:
        tools = _catalog(case.get("catalog", "base"))
        proposal = proposal_from_dict(case["proposal"])
        signals_payload = dict(case["signals"])
        if case.get("capability_signals") is not None:
            signals_payload["capability_signals"] = case["capability_signals"]
        signals = signals_from_dict(signals_payload)
        planner_kwargs = case.get("planner") or {}
        planner = ToolCallPlanner(**planner_kwargs) if planner_kwargs else None
        plan = evaluate_gate(tools, proposal, signals, planner=planner)

        assert plan.status == case["expect_plan_status"], case["id"]
        if case.get("expect_reason_substring"):
            assert case["expect_reason_substring"] in plan.reason, case["id"]

        class MatrixHost:
            def check_permissions(self, tool_name, arguments):
                if case["host_permissions"]:
                    return True, "ok"
                return False, "denied by ACL"

            def execute(self, tool_name, arguments):
                return {"ok": True}

        outcome = dispatch_plan(plan, MatrixHost())
        assert outcome.action == case["expect_host_action"], case["id"]


def test_required_capabilities_alias_roundtrip() -> None:
    multi = ToolSpec(
        "pay",
        required_capabilities=("billing", "admin"),
    )
    assert multi.required_capabilities == ("billing", "admin")
    assert multi.required_capability == "billing"

    single = ToolSpec("read", required_capability="workspace_access")
    assert single.required_capabilities == ("workspace_access",)
    assert single.required_capability == "workspace_access"
