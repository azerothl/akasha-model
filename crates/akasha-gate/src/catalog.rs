//! Optional catalog allow/deny/placement constraints (Python `catalog.py`).

use crate::types::{PlanStatus, ToolCallPlan, ToolSpec};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Optional constraints applied before planner thresholds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogPolicy {
    /// If set, proposal tool must be in the set.
    #[serde(default)]
    pub allowlist: Option<BTreeSet<String>>,
    /// Proposal tool must not be in the set.
    #[serde(default)]
    pub denylist: BTreeSet<String>,
    /// Every listed tag must appear on the tool's `placement_tags`.
    #[serde(default)]
    pub required_placement: BTreeSet<String>,
}

fn blocked(tool_name: &str, reason: String) -> ToolCallPlan {
    ToolCallPlan {
        status: PlanStatus::Blocked,
        tool_name: Some(tool_name.to_string()),
        arguments: None,
        reason,
        choice_probability: 0.0,
        choice_confidence: 0.0,
        risk_score: None,
    }
}

/// Return a blocked plan when policy rejects the proposal, else `None`.
pub fn check_catalog_policy(
    tools: &BTreeMap<String, ToolSpec>,
    tool_name: &str,
    policy: Option<&CatalogPolicy>,
) -> Option<ToolCallPlan> {
    let policy = policy?;
    if let Some(allow) = &policy.allowlist {
        if !allow.contains(tool_name) {
            return Some(blocked(
                tool_name,
                format!("catalog_allowlist: tool not permitted: {tool_name}"),
            ));
        }
    }
    if policy.denylist.contains(tool_name) {
        return Some(blocked(
            tool_name,
            format!("catalog_denylist: tool denied: {tool_name}"),
        ));
    }
    if !policy.required_placement.is_empty() {
        let tags: BTreeSet<&str> = tools
            .get(tool_name)
            .map(|spec| spec.placement_tags.iter().map(String::as_str).collect())
            .unwrap_or_default();
        let mut missing: Vec<&str> = policy
            .required_placement
            .iter()
            .map(String::as_str)
            .filter(|tag| !tags.contains(tag))
            .collect();
        missing.sort();
        if !missing.is_empty() {
            return Some(blocked(
                tool_name,
                format!("catalog_placement: missing tag(s): {}", missing.join(", ")),
            ));
        }
    }
    None
}
