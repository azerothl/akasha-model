from akasha_model import CatalogPolicy, ToolProposal, GateSignals, ToolSpec, evaluate_gate


def _tools():
    return {
        "fs.read": ToolSpec(
            "fs.read",
            placement_tags=("workspace", "preview"),
            required_capability="workspace_access",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        "shell.run": ToolSpec(
            "shell.run",
            placement_tags=("workspace",),
            required_capability="shell",
        ),
    }


def _ok_signals(**kwargs):
    base = dict(
        authorized=0.95,
        sufficient_context=0.9,
        capability_present=0.95,
        confirmation_needed=0.05,
    )
    base.update(kwargs)
    return GateSignals(**base)


def test_unconstrained_catalog_still_ready() -> None:
    plan = evaluate_gate(
        _tools(),
        ToolProposal("fs.read", {"path": "a.txt"}),
        _ok_signals(),
    )
    assert plan.status == "ready"


def test_allowlist_blocks_unknown_proposal() -> None:
    policy = CatalogPolicy(allowlist=frozenset({"fs.read"}))
    plan = evaluate_gate(
        _tools(),
        ToolProposal("shell.run", {"cmd": "ls"}),
        _ok_signals(),
        catalog_policy=policy,
    )
    assert plan.status == "blocked"
    assert plan.reason.startswith("catalog_allowlist")


def test_denylist_blocks() -> None:
    policy = CatalogPolicy(denylist=frozenset({"shell.run"}))
    plan = evaluate_gate(
        _tools(),
        ToolProposal("shell.run", {"cmd": "ls"}),
        _ok_signals(),
        catalog_policy=policy,
    )
    assert plan.status == "blocked"
    assert plan.reason.startswith("catalog_denylist")


def test_placement_tags_required() -> None:
    policy = CatalogPolicy(required_placement=frozenset({"preview"}))
    plan = evaluate_gate(
        _tools(),
        ToolProposal("shell.run", {"cmd": "ls"}),
        _ok_signals(),
        catalog_policy=policy,
    )
    assert plan.status == "blocked"
    assert plan.reason.startswith("catalog_placement")
    ready = evaluate_gate(
        _tools(),
        ToolProposal("fs.read", {"path": "a.txt"}),
        _ok_signals(),
        catalog_policy=policy,
    )
    assert ready.status == "ready"
