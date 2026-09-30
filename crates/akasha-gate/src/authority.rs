//! Optional authority profiles (Python `authority.py`).

use crate::types::{PlanStatus, ToolCallPlan};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Optional thresholds bound to a decision surface / tool class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthorityProfile {
    #[serde(default = "default_profile_name")]
    pub name: String,
    #[serde(default)]
    pub min_authorized: Option<f64>,
    #[serde(default)]
    pub min_sufficient_context: Option<f64>,
    #[serde(default)]
    pub max_risk_score: Option<f64>,
    #[serde(default)]
    pub max_consequence: Option<f64>,
    #[serde(default)]
    pub required_context_keys: Vec<String>,
}

fn default_profile_name() -> String {
    "default".into()
}

impl Default for AuthorityProfile {
    fn default() -> Self {
        Self {
            name: default_profile_name(),
            min_authorized: None,
            min_sufficient_context: None,
            max_risk_score: None,
            max_consequence: None,
            required_context_keys: Vec::new(),
        }
    }
}

pub fn annotate_reason(code: &str, detail: &str) -> String {
    format!("{code}: {detail}")
}

fn remap(
    status: PlanStatus,
    plan: &ToolCallPlan,
    reason: String,
    risk_score: Option<f64>,
) -> ToolCallPlan {
    ToolCallPlan {
        status,
        tool_name: plan.tool_name.clone(),
        arguments: if status == PlanStatus::Ready {
            plan.arguments.clone()
        } else {
            None
        },
        reason,
        choice_probability: plan.choice_probability,
        choice_confidence: plan.choice_confidence,
        risk_score: risk_score.or(plan.risk_score),
    }
}

/// Re-map or annotate a plan when an authority profile is active.
pub fn apply_authority_profile(
    plan: ToolCallPlan,
    profile: Option<&AuthorityProfile>,
    authorized: f64,
    sufficient_context: f64,
    risk_score: Option<f64>,
    consequence: Option<f64>,
    context: Option<&BTreeMap<String, Value>>,
) -> ToolCallPlan {
    let Some(profile) = profile else {
        return plan;
    };

    if !profile.required_context_keys.is_empty() {
        let ctx = context;
        let missing: Vec<&str> = profile
            .required_context_keys
            .iter()
            .map(String::as_str)
            .filter(|key| ctx.map(|c| !c.contains_key(*key)).unwrap_or(true))
            .collect();
        if !missing.is_empty() {
            return remap(
                PlanStatus::Abstain,
                &plan,
                annotate_reason(
                    "clarify",
                    &format!("missing required context key(s): {}", missing.join(", ")),
                ),
                None,
            );
        }
    }

    if let Some(min_ctx) = profile.min_sufficient_context {
        if sufficient_context < min_ctx {
            return remap(
                PlanStatus::Abstain,
                &plan,
                annotate_reason("clarify", "sufficient_context below authority profile minimum"),
                None,
            );
        }
    }

    if let Some(min_auth) = profile.min_authorized {
        if authorized < min_auth {
            return remap(
                PlanStatus::Blocked,
                &plan,
                annotate_reason("reject", "authorized below authority profile minimum"),
                None,
            );
        }
    }

    if let (Some(max_risk), Some(score)) = (profile.max_risk_score, risk_score) {
        if score > max_risk {
            return remap(
                PlanStatus::Blocked,
                &plan,
                annotate_reason("reject", "risk score exceeds authority profile maximum"),
                Some(score),
            );
        }
    }

    if let (Some(max_c), Some(c)) = (profile.max_consequence, consequence) {
        if c > max_c {
            return remap(
                PlanStatus::Abstain,
                &plan,
                annotate_reason(
                    "escalate",
                    "consequence exceeds authority profile; human review required",
                ),
                None,
            );
        }
    }

    if plan.status == PlanStatus::Blocked
        && (plan.reason.starts_with("authorization")
            || plan.reason.starts_with("Risk score")
            || plan.reason.starts_with("required capability")
            || plan.reason.starts_with("explicit human"))
    {
        return ToolCallPlan {
            reason: annotate_reason("reject", &plan.reason),
            ..plan
        };
    }
    if plan.status == PlanStatus::Blocked && plan.reason.contains("context sufficiency") {
        return remap(
            PlanStatus::Abstain,
            &plan,
            annotate_reason("clarify", &plan.reason),
            None,
        );
    }
    plan
}
