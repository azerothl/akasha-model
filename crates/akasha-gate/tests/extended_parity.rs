//! Extended Path A parity: catalog, authority, budget (Python #15/#18/#22).

use akasha_gate::{
    evaluate_gate, evaluate_gate_with, AuthorityProfile, CatalogPolicy, EvaluateOptions,
    GateSignals, PlanStatus, ToolProposal, ToolSpec,
};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

fn tools() -> BTreeMap<String, ToolSpec> {
    let mut map = BTreeMap::new();
    map.insert(
        "fs.read".into(),
        ToolSpec::new("fs.read")
            .with_capability("workspace_access")
            .with_placement_tags(["workspace", "preview"])
            .with_parameters(json!({
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": false
            })),
    );
    map.insert(
        "shell.run".into(),
        ToolSpec::new("shell.run")
            .with_capability("shell")
            .with_placement_tags(["workspace"]),
    );
    map.insert(
        "payments.charge".into(),
        ToolSpec::new("payments.charge")
            .with_capability("billing")
            .irreversible(),
    );
    map
}

fn ok_signals() -> GateSignals {
    GateSignals {
        authorized: 0.95,
        sufficient_context: 0.9,
        capability_present: Some(0.95),
        confirmation_needed: Some(0.05),
        risk_score: None,
        confirmation_given: false,
        capability_signals: None,
        budget_remaining: None,
        consequence: None,
        context: None,
    }
}

fn proposal(tool: &str) -> ToolProposal {
    let mut args = BTreeMap::new();
    if tool == "fs.read" {
        args.insert("path".into(), json!("a.txt"));
    } else if tool == "shell.run" {
        args.insert("cmd".into(), json!("ls"));
    }
    ToolProposal {
        tool_name: tool.into(),
        arguments: if args.is_empty() { None } else { Some(args) },
        choice_probabilities: None,
    }
}

#[test]
fn catalog_allowlist_blocks() {
    let policy = CatalogPolicy {
        allowlist: Some(BTreeSet::from(["fs.read".into()])),
        ..CatalogPolicy::default()
    };
    let plan = evaluate_gate_with(
        &tools(),
        &proposal("shell.run"),
        &ok_signals(),
        EvaluateOptions {
            catalog_policy: Some(&policy),
            ..EvaluateOptions::default()
        },
    )
    .unwrap();
    assert_eq!(plan.status, PlanStatus::Blocked);
    assert!(plan.reason.starts_with("catalog_allowlist"));
}

#[test]
fn catalog_denylist_blocks() {
    let policy = CatalogPolicy {
        denylist: BTreeSet::from(["shell.run".into()]),
        ..CatalogPolicy::default()
    };
    let plan = evaluate_gate_with(
        &tools(),
        &proposal("shell.run"),
        &ok_signals(),
        EvaluateOptions {
            catalog_policy: Some(&policy),
            ..EvaluateOptions::default()
        },
    )
    .unwrap();
    assert_eq!(plan.status, PlanStatus::Blocked);
    assert!(plan.reason.starts_with("catalog_denylist"));
}

#[test]
fn catalog_placement_required() {
    let policy = CatalogPolicy {
        required_placement: BTreeSet::from(["preview".into()]),
        ..CatalogPolicy::default()
    };
    let blocked = evaluate_gate_with(
        &tools(),
        &proposal("shell.run"),
        &ok_signals(),
        EvaluateOptions {
            catalog_policy: Some(&policy),
            ..EvaluateOptions::default()
        },
    )
    .unwrap();
    assert_eq!(blocked.status, PlanStatus::Blocked);
    assert!(blocked.reason.starts_with("catalog_placement"));
    let ready = evaluate_gate_with(
        &tools(),
        &proposal("fs.read"),
        &ok_signals(),
        EvaluateOptions {
            catalog_policy: Some(&policy),
            ..EvaluateOptions::default()
        },
    )
    .unwrap();
    assert_eq!(ready.status, PlanStatus::Ready);
}

#[test]
fn budget_exceeded_blocks() {
    let mut signals = ok_signals();
    signals.budget_remaining = Some(0.0);
    let plan = evaluate_gate(&tools(), &proposal("fs.read"), &signals, None).unwrap();
    assert_eq!(plan.status, PlanStatus::Blocked);
    assert_eq!(plan.reason, "budget_exceeded");
    assert!(!plan.executable());
}

#[test]
fn positive_budget_ready() {
    let mut signals = ok_signals();
    signals.budget_remaining = Some(3.0);
    let plan = evaluate_gate(&tools(), &proposal("fs.read"), &signals, None).unwrap();
    assert_eq!(plan.status, PlanStatus::Ready);
}

#[test]
fn authority_escalate_on_consequence() {
    let profile = AuthorityProfile {
        name: "payments".into(),
        max_consequence: Some(0.4),
        ..AuthorityProfile::default()
    };
    let mut signals = ok_signals();
    signals.consequence = Some(0.9);
    let plan = evaluate_gate_with(
        &tools(),
        &proposal("fs.read"),
        &signals,
        EvaluateOptions {
            authority_profile: Some(&profile),
            ..EvaluateOptions::default()
        },
    )
    .unwrap();
    assert_eq!(plan.status, PlanStatus::Abstain);
    assert!(plan.reason.starts_with("escalate:"));
}

#[test]
fn authority_reject_low_authorized() {
    let profile = AuthorityProfile {
        name: "strict".into(),
        min_authorized: Some(0.95),
        ..AuthorityProfile::default()
    };
    let mut signals = ok_signals();
    signals.authorized = 0.2;
    signals.sufficient_context = 0.95;
    signals.capability_present = Some(0.95);
    let plan = evaluate_gate_with(
        &tools(),
        &proposal("fs.read"),
        &signals,
        EvaluateOptions {
            authority_profile: Some(&profile),
            ..EvaluateOptions::default()
        },
    )
    .unwrap();
    assert_eq!(plan.status, PlanStatus::Blocked);
    assert!(plan.reason.starts_with("reject:"));
}

#[test]
fn authority_clarify_missing_context() {
    let profile = AuthorityProfile {
        name: "mail".into(),
        required_context_keys: vec!["user_id".into(), "thread_id".into()],
        ..AuthorityProfile::default()
    };
    let mut signals = ok_signals();
    signals.sufficient_context = 0.95;
    signals.capability_present = Some(0.95);
    let mut ctx = BTreeMap::new();
    ctx.insert("user_id".into(), json!("u1"));
    signals.context = Some(ctx);
    let plan = evaluate_gate_with(
        &tools(),
        &proposal("fs.read"),
        &signals,
        EvaluateOptions {
            authority_profile: Some(&profile),
            ..EvaluateOptions::default()
        },
    )
    .unwrap();
    assert_eq!(plan.status, PlanStatus::Abstain);
    assert!(plan.reason.starts_with("clarify:"));
    assert!(plan.reason.contains("thread_id"));
}

#[test]
fn authority_reject_high_risk() {
    let profile = AuthorityProfile {
        name: "safe".into(),
        max_risk_score: Some(1.0),
        ..AuthorityProfile::default()
    };
    let mut signals = ok_signals();
    signals.sufficient_context = 0.95;
    signals.capability_present = Some(0.95);
    signals.risk_score = Some(2.0);
    let plan = evaluate_gate_with(
        &tools(),
        &proposal("fs.read"),
        &signals,
        EvaluateOptions {
            authority_profile: Some(&profile),
            ..EvaluateOptions::default()
        },
    )
    .unwrap();
    // Planner max_risk (1.5) may block first; with profile, reason is reject-coded.
    assert_eq!(plan.status, PlanStatus::Blocked);
    assert!(
        plan.reason.starts_with("reject:")
            || plan.reason.contains("Risk score")
            || plan.reason.contains("risk score")
    );
}
