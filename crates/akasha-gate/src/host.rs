use crate::planner::{evaluate_gate, GateSignals, ToolCallPlanner, ToolProposal};
use crate::types::{PlanStatus, ToolCallPlan, ToolSpec};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostAction {
    Executed,
    SkippedAbstain,
    SkippedBlocked,
    RejectedByHost,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostOutcome {
    pub action: HostAction,
    pub plan: ToolCallPlan,
    pub result: Option<serde_json::Value>,
    pub host_reason: String,
}

impl HostOutcome {
    pub fn did_execute(&self) -> bool {
        self.action == HostAction::Executed
    }
}

/// Application / OS side of the gate. Implementations must not be invoked for
/// side effects except via [`dispatch_plan`] after `ready` + permission check.
pub trait ToolHost {
    fn check_permissions(
        &self,
        tool_name: &str,
        arguments: Option<&BTreeMap<String, serde_json::Value>>,
    ) -> (bool, String);

    fn execute(
        &self,
        tool_name: &str,
        arguments: Option<&BTreeMap<String, serde_json::Value>>,
    ) -> serde_json::Value;
}

pub fn dispatch_plan(plan: &ToolCallPlan, host: &dyn ToolHost) -> HostOutcome {
    if plan.status == PlanStatus::Abstain {
        return HostOutcome {
            action: HostAction::SkippedAbstain,
            plan: plan.clone(),
            result: None,
            host_reason: plan.reason.clone(),
        };
    }
    if plan.status != PlanStatus::Ready || !plan.executable() {
        return HostOutcome {
            action: HostAction::SkippedBlocked,
            plan: plan.clone(),
            result: None,
            host_reason: plan.reason.clone(),
        };
    }
    let Some(tool_name) = plan.tool_name.as_deref() else {
        return HostOutcome {
            action: HostAction::SkippedBlocked,
            plan: plan.clone(),
            result: None,
            host_reason: "ready plan missing tool_name".into(),
        };
    };
    let (allowed, reason) = host.check_permissions(tool_name, plan.arguments.as_ref());
    if !allowed {
        return HostOutcome {
            action: HostAction::RejectedByHost,
            plan: plan.clone(),
            result: None,
            host_reason: reason,
        };
    }
    let result = host.execute(tool_name, plan.arguments.as_ref());
    HostOutcome {
        action: HostAction::Executed,
        plan: plan.clone(),
        result: Some(result),
        host_reason: "host executed after ready and permission check".into(),
    }
}

pub fn run_gated_call(
    tools: &BTreeMap<String, ToolSpec>,
    proposal: &ToolProposal,
    signals: &GateSignals,
    host: &dyn ToolHost,
    planner: Option<&ToolCallPlanner>,
) -> Result<HostOutcome, crate::planner::GateError> {
    let plan = evaluate_gate(tools, proposal, signals, planner)?;
    Ok(dispatch_plan(&plan, host))
}
