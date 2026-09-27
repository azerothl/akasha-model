use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Ready,
    Abstain,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub required_capability: Option<String>,
    #[serde(default)]
    pub required_capabilities: Vec<String>,
    #[serde(default)]
    pub requires_confirmation: bool,
    #[serde(default)]
    pub irreversible: bool,
}

impl ToolSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            parameters: None,
            required_capability: None,
            required_capabilities: Vec::new(),
            requires_confirmation: false,
            irreversible: false,
        }
    }

    pub fn with_capability(mut self, cap: impl Into<String>) -> Self {
        let cap = cap.into();
        self.required_capability = Some(cap.clone());
        if !self.required_capabilities.contains(&cap) {
            self.required_capabilities.push(cap);
        }
        self
    }

    pub fn with_capabilities<I, S>(mut self, caps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for cap in caps {
            let cap = cap.into();
            if !self.required_capabilities.contains(&cap) {
                self.required_capabilities.push(cap);
            }
        }
        if self.required_capability.is_none() {
            self.required_capability = self.required_capabilities.first().cloned();
        }
        self
    }

    pub fn irreversible(mut self) -> Self {
        self.irreversible = true;
        self
    }

    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = Some(parameters);
        self
    }

    pub fn capabilities(&self) -> Vec<String> {
        if !self.required_capabilities.is_empty() {
            return self.required_capabilities.clone();
        }
        self.required_capability
            .clone()
            .into_iter()
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallPlan {
    pub status: PlanStatus,
    pub tool_name: Option<String>,
    pub arguments: Option<BTreeMap<String, serde_json::Value>>,
    pub reason: String,
    pub choice_probability: f64,
    pub choice_confidence: f64,
    pub risk_score: Option<f64>,
}

impl ToolCallPlan {
    pub fn executable(&self) -> bool {
        self.status == PlanStatus::Ready
    }
}
