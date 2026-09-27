use crate::types::{PlanStatus, ToolCallPlan, ToolSpec};
use crate::validate::validate_tool_arguments;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const DEFAULT_MIN_CHOICE_PROBABILITY: f64 = 0.55;
pub const DEFAULT_MIN_CHOICE_CONFIDENCE: f64 = 0.50;
pub const DEFAULT_NOUL_THRESHOLD: f64 = 0.70;
pub const DEFAULT_MAX_RISK_SCORE: f64 = 1.5;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolProposal {
    pub tool_name: String,
    #[serde(default)]
    pub arguments: Option<BTreeMap<String, serde_json::Value>>,
    #[serde(default)]
    pub choice_probabilities: Option<BTreeMap<String, f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateSignals {
    pub authorized: f64,
    pub sufficient_context: f64,
    #[serde(default)]
    pub capability_present: Option<f64>,
    #[serde(default)]
    pub confirmation_needed: Option<f64>,
    #[serde(default)]
    pub risk_score: Option<f64>,
    #[serde(default)]
    pub confirmation_given: bool,
    #[serde(default)]
    pub capability_signals: Option<BTreeMap<String, f64>>,
}

#[derive(Debug, Clone)]
pub struct ChoicePeak {
    pub selected: Option<String>,
    pub probabilities: BTreeMap<String, f64>,
    pub confidence: f64,
    pub abstained: bool,
}

#[derive(Debug, Error)]
pub enum GateError {
    #[error("{0}")]
    Invalid(String),
}

#[derive(Debug, Clone)]
pub struct ToolCallPlanner {
    pub min_choice_probability: f64,
    pub min_choice_confidence: f64,
    pub noul_threshold: f64,
    pub max_risk_score: Option<f64>,
}

impl Default for ToolCallPlanner {
    fn default() -> Self {
        Self {
            min_choice_probability: DEFAULT_MIN_CHOICE_PROBABILITY,
            min_choice_confidence: DEFAULT_MIN_CHOICE_CONFIDENCE,
            noul_threshold: DEFAULT_NOUL_THRESHOLD,
            max_risk_score: Some(DEFAULT_MAX_RISK_SCORE),
        }
    }
}

pub fn default_gate_planner() -> ToolCallPlanner {
    ToolCallPlanner::default()
}

impl ToolCallPlanner {
    pub fn plan(
        &self,
        choice: &ChoicePeak,
        tools: &BTreeMap<String, ToolSpec>,
        arguments: Option<&BTreeMap<String, serde_json::Value>>,
        nouls: &BTreeMap<String, f64>,
        risk_score: Option<f64>,
        confirmation_given: bool,
    ) -> ToolCallPlan {
        if choice.abstained || choice.selected.is_none() {
            return ToolCallPlan {
                status: PlanStatus::Abstain,
                tool_name: None,
                arguments: None,
                reason: "Choice abstained".into(),
                choice_probability: 0.0,
                choice_confidence: choice.confidence,
                risk_score,
            };
        }
        let selected = choice.selected.as_ref().unwrap();
        let probability = *choice.probabilities.get(selected).unwrap_or(&0.0);
        if choice.confidence < self.min_choice_confidence
            || probability < self.min_choice_probability
        {
            return ToolCallPlan {
                status: PlanStatus::Abstain,
                tool_name: Some(selected.clone()),
                arguments: None,
                reason: "Choice confidence or probability is below the execution threshold"
                    .into(),
                choice_probability: probability,
                choice_confidence: choice.confidence,
                risk_score,
            };
        }
        let Some(spec) = tools.get(selected) else {
            return ToolCallPlan {
                status: PlanStatus::Blocked,
                tool_name: Some(selected.clone()),
                arguments: None,
                reason: "Selected tool is not in the host tool catalog".into(),
                choice_probability: probability,
                choice_confidence: choice.confidence,
                risk_score,
            };
        };
        if let (Some(max_risk), Some(score)) = (self.max_risk_score, risk_score) {
            if score > max_risk {
                return ToolCallPlan {
                    status: PlanStatus::Blocked,
                    tool_name: Some(spec.name.clone()),
                    arguments: None,
                    reason: "Risk score exceeds the planner limit".into(),
                    choice_probability: probability,
                    choice_confidence: choice.confidence,
                    risk_score,
                };
            }
        }
        for (question_id, message) in [
            ("authorized", "authorization gate is not positive"),
            ("sufficient_context", "context sufficiency gate is not positive"),
        ] {
            let gate = nouls.get(question_id).copied();
            if gate.map(|g| g < self.noul_threshold).unwrap_or(true) {
                return ToolCallPlan {
                    status: PlanStatus::Blocked,
                    tool_name: Some(spec.name.clone()),
                    arguments: None,
                    reason: message.into(),
                    choice_probability: probability,
                    choice_confidence: choice.confidence,
                    risk_score,
                };
            }
        }
        let caps = spec.capabilities();
        if !caps.is_empty() {
            let mut missing = Vec::new();
            for cap in &caps {
                let per = nouls.get(&format!("capability:{cap}")).copied();
                if let Some(value) = per {
                    if value < self.noul_threshold {
                        missing.push(cap.clone());
                    }
                    continue;
                }
                let shared = nouls.get("capability_present").copied();
                if shared.map(|g| g < self.noul_threshold).unwrap_or(true) {
                    missing.push(cap.clone());
                }
            }
            if !missing.is_empty() {
                return ToolCallPlan {
                    status: PlanStatus::Blocked,
                    tool_name: Some(spec.name.clone()),
                    arguments: None,
                    reason: format!(
                        "required capability is not positive: {}",
                        missing.join(", ")
                    ),
                    choice_probability: probability,
                    choice_confidence: choice.confidence,
                    risk_score,
                };
            }
        }
        let mut confirmation_needed = spec.requires_confirmation || spec.irreversible;
        if let Some(model_confirmation) = nouls.get("confirmation_needed").copied() {
            confirmation_needed =
                confirmation_needed || model_confirmation >= self.noul_threshold;
        }
        if confirmation_needed && !confirmation_given {
            return ToolCallPlan {
                status: PlanStatus::Blocked,
                tool_name: Some(spec.name.clone()),
                arguments: None,
                reason: "explicit human confirmation is required".into(),
                choice_probability: probability,
                choice_confidence: choice.confidence,
                risk_score,
            };
        }
        if let Err(reason) = validate_tool_arguments(spec, arguments) {
            return ToolCallPlan {
                status: PlanStatus::Blocked,
                tool_name: Some(spec.name.clone()),
                arguments: None,
                reason,
                choice_probability: probability,
                choice_confidence: choice.confidence,
                risk_score,
            };
        }
        ToolCallPlan {
            status: PlanStatus::Ready,
            tool_name: Some(spec.name.clone()),
            arguments: Some(arguments.cloned().unwrap_or_default()),
            reason: "all planner gates passed".into(),
            choice_probability: probability,
            choice_confidence: choice.confidence,
            risk_score,
        }
    }
}

fn distribution_confidence(probabilities: &BTreeMap<String, f64>) -> f64 {
    let n = probabilities.len().max(2) as f64;
    let entropy: f64 = probabilities
        .values()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let normalised = entropy / n.ln();
    (1.0 - normalised).clamp(0.0, 1.0)
}

pub fn choice_from_proposal(
    tools: &BTreeMap<String, ToolSpec>,
    proposal: &ToolProposal,
    peak: f64,
) -> Result<ChoicePeak, GateError> {
    if tools.len() < 2 {
        return Err(GateError::Invalid(
            "gate choice requires at least two tools in the catalog".into(),
        ));
    }
    if let Some(dist) = &proposal.choice_probabilities {
        let names: Vec<_> = tools.keys().cloned().collect();
        if dist.len() != names.len() || !names.iter().all(|n| dist.contains_key(n)) {
            return Err(GateError::Invalid(
                "Choice probabilities must contain exactly all option names".into(),
            ));
        }
        let confidence = distribution_confidence(dist);
        let selected = dist
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| k.clone());
        let abstained = confidence <= 0.0;
        return Ok(ChoicePeak {
            selected,
            probabilities: dist.clone(),
            confidence,
            abstained,
        });
    }
    if !tools.contains_key(&proposal.tool_name) {
        return Err(GateError::Invalid(format!(
            "proposed tool '{}' is not in the tool catalog",
            proposal.tool_name
        )));
    }
    if !(0.5 < peak && peak < 1.0) {
        return Err(GateError::Invalid("peak must be in (0.5, 1)".into()));
    }
    let names: Vec<_> = tools.keys().cloned().collect();
    let rest = (1.0 - peak) / (names.len() as f64 - 1.0);
    let mut probabilities = BTreeMap::new();
    for name in &names {
        probabilities.insert(
            name.clone(),
            if name == &proposal.tool_name {
                peak
            } else {
                rest
            },
        );
    }
    let confidence = distribution_confidence(&probabilities);
    Ok(ChoicePeak {
        selected: Some(proposal.tool_name.clone()),
        probabilities,
        confidence,
        abstained: false,
    })
}

pub fn evaluate_gate(
    tools: &BTreeMap<String, ToolSpec>,
    proposal: &ToolProposal,
    signals: &GateSignals,
    planner: Option<&ToolCallPlanner>,
) -> Result<ToolCallPlan, GateError> {
    let default = default_gate_planner();
    let active = planner.unwrap_or(&default);
    let choice = choice_from_proposal(tools, proposal, 0.90)?;
    let mut nouls = BTreeMap::new();
    nouls.insert("authorized".into(), signals.authorized);
    nouls.insert("sufficient_context".into(), signals.sufficient_context);
    if let Some(v) = signals.capability_present {
        nouls.insert("capability_present".into(), v);
    }
    if let Some(map) = &signals.capability_signals {
        for (cap, value) in map {
            nouls.insert(format!("capability:{cap}"), *value);
        }
    }
    if let Some(v) = signals.confirmation_needed {
        nouls.insert("confirmation_needed".into(), v);
    }
    Ok(active.plan(
        &choice,
        tools,
        proposal.arguments.as_ref(),
        &nouls,
        signals.risk_score,
        signals.confirmation_given,
    ))
}
