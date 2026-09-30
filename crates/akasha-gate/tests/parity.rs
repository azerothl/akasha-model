use akasha_gate::{
    dispatch_plan, evaluate_gate, HostAction, PlanStatus, ToolHost, ToolProposal, ToolSpec,
    GateSignals, ToolCallPlanner,
};
use serde_json::json;
use std::collections::BTreeMap;

fn catalog() -> BTreeMap<String, ToolSpec> {
    let mut tools = BTreeMap::new();
    tools.insert(
        "fs.read".into(),
        ToolSpec::new("fs.read")
            .with_capability("workspace_access")
            .with_parameters(json!({
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": false
            })),
    );
    tools.insert(
        "fs.delete".into(),
        ToolSpec::new("fs.delete")
            .with_capability("workspace_access")
            .irreversible(),
    );
    tools
}

struct OkHost;
impl ToolHost for OkHost {
    fn check_permissions(
        &self,
        _tool_name: &str,
        _arguments: Option<&BTreeMap<String, serde_json::Value>>,
    ) -> (bool, String) {
        (true, "ok".into())
    }

    fn execute(
        &self,
        _tool_name: &str,
        _arguments: Option<&BTreeMap<String, serde_json::Value>>,
    ) -> serde_json::Value {
        json!({"ok": true})
    }
}

struct DenyHost;
impl ToolHost for DenyHost {
    fn check_permissions(
        &self,
        _tool_name: &str,
        _arguments: Option<&BTreeMap<String, serde_json::Value>>,
    ) -> (bool, String) {
        (false, "denied by ACL".into())
    }

    fn execute(
        &self,
        _tool_name: &str,
        _arguments: Option<&BTreeMap<String, serde_json::Value>>,
    ) -> serde_json::Value {
        panic!("execute must not run")
    }
}

#[test]
fn ready_read_matches_python_status_class() {
    let tools = catalog();
    let mut args = BTreeMap::new();
    args.insert("path".into(), json!("notes.txt"));
    let proposal = ToolProposal {
        tool_name: "fs.read".into(),
        arguments: Some(args),
        choice_probabilities: None,
    };
    let signals = GateSignals {
        authorized: 0.95,
        sufficient_context: 0.9,
        capability_present: Some(0.92),
        confirmation_needed: Some(0.05),
        risk_score: None,
        confirmation_given: false,
        capability_signals: None,
        budget_remaining: None,
        consequence: None,
        context: None,
    };
    let plan = evaluate_gate(&tools, &proposal, &signals, None).unwrap();
    assert_eq!(plan.status, PlanStatus::Ready);
    assert!(plan.executable());
    assert_eq!(plan.reason, "all planner gates passed");
    let outcome = dispatch_plan(&plan, &OkHost);
    assert_eq!(outcome.action, HostAction::Executed);
}

#[test]
fn delete_without_confirmation_blocked() {
    let tools = catalog();
    let proposal = ToolProposal {
        tool_name: "fs.delete".into(),
        arguments: None,
        choice_probabilities: None,
    };
    let signals = GateSignals {
        authorized: 0.95,
        sufficient_context: 0.95,
        capability_present: Some(0.95),
        confirmation_needed: None,
        risk_score: None,
        confirmation_given: false,
        capability_signals: None,
        budget_remaining: None,
        consequence: None,
        context: None,
    };
    let plan = evaluate_gate(&tools, &proposal, &signals, None).unwrap();
    assert_eq!(plan.status, PlanStatus::Blocked);
    assert!(plan.reason.contains("confirmation"));
    assert_eq!(dispatch_plan(&plan, &OkHost).action, HostAction::SkippedBlocked);
}

#[test]
fn weak_choice_abstains() {
    let tools = catalog();
    let mut probs = BTreeMap::new();
    probs.insert("fs.read".into(), 0.52);
    probs.insert("fs.delete".into(), 0.48);
    let proposal = ToolProposal {
        tool_name: "fs.read".into(),
        arguments: None,
        choice_probabilities: Some(probs),
    };
    let signals = GateSignals {
        authorized: 1.0,
        sufficient_context: 1.0,
        capability_present: Some(1.0),
        confirmation_needed: None,
        risk_score: None,
        confirmation_given: false,
        capability_signals: None,
        budget_remaining: None,
        consequence: None,
        context: None,
    };
    let planner = ToolCallPlanner {
        min_choice_probability: 0.60,
        ..ToolCallPlanner::default()
    };
    let plan = evaluate_gate(&tools, &proposal, &signals, Some(&planner)).unwrap();
    assert_eq!(plan.status, PlanStatus::Abstain);
    assert_eq!(
        dispatch_plan(&plan, &OkHost).action,
        HostAction::SkippedAbstain
    );
}

#[test]
fn host_permission_denial() {
    let tools = catalog();
    let mut args = BTreeMap::new();
    args.insert("path".into(), json!("notes.txt"));
    let proposal = ToolProposal {
        tool_name: "fs.read".into(),
        arguments: Some(args),
        choice_probabilities: None,
    };
    let signals = GateSignals {
        authorized: 0.95,
        sufficient_context: 0.9,
        capability_present: Some(0.92),
        confirmation_needed: Some(0.05),
        risk_score: None,
        confirmation_given: false,
        capability_signals: None,
        budget_remaining: None,
        consequence: None,
        context: None,
    };
    let plan = evaluate_gate(&tools, &proposal, &signals, None).unwrap();
    let outcome = dispatch_plan(&plan, &DenyHost);
    assert_eq!(outcome.action, HostAction::RejectedByHost);
    assert!(outcome.host_reason.contains("ACL"));
}

#[test]
fn multi_cap_partial_block() {
    let mut tools = catalog();
    tools.insert(
        "shell.run".into(),
        ToolSpec::new("shell.run")
            .with_capabilities(["shell", "network"])
            .with_parameters(json!({
                "type": "object",
                "required": ["cmd"],
                "properties": {"cmd": {"type": "string"}},
                "additionalProperties": false
            })),
    );
    let mut args = BTreeMap::new();
    args.insert("cmd".into(), json!("ls"));
    let proposal = ToolProposal {
        tool_name: "shell.run".into(),
        arguments: Some(args),
        choice_probabilities: None,
    };
    let mut caps = BTreeMap::new();
    caps.insert("shell".into(), 0.95);
    caps.insert("network".into(), 0.1);
    let signals = GateSignals {
        authorized: 0.95,
        sufficient_context: 0.95,
        capability_present: None,
        confirmation_needed: None,
        risk_score: None,
        confirmation_given: true,
        capability_signals: Some(caps),
        budget_remaining: None,
        consequence: None,
        context: None,
    };
    let plan = evaluate_gate(&tools, &proposal, &signals, None).unwrap();
    assert_eq!(plan.status, PlanStatus::Blocked);
    assert!(plan.reason.contains("network"));
}
