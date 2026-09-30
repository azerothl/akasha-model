//! Path A tool gate: typed signals in, [`ToolCallPlan`] out, never execute.
//!
//! Mirrors `akasha_model.tool_calling` / `gate` / `host` for scripted
//! authorization. Path B scorers stay in the Python package.
//!
//! Optional C ABI (`cdylib`): see [`ffi`] and `include/akasha_gate.h`.

mod ffi;
mod host;
mod planner;
mod types;
mod validate;

pub use host::{dispatch_plan, run_gated_call, HostAction, HostOutcome, ToolHost};
pub use planner::{
    default_gate_planner, evaluate_gate, ChoicePeak, GateSignals, ToolCallPlanner, ToolProposal,
    DEFAULT_MAX_RISK_SCORE, DEFAULT_MIN_CHOICE_CONFIDENCE, DEFAULT_MIN_CHOICE_PROBABILITY,
    DEFAULT_NOUL_THRESHOLD,
};
pub use types::{PlanStatus, ToolCallPlan, ToolSpec};
pub use validate::validate_tool_arguments;

/// Wire / semantic contract version aligned with Python `akasha_model.wire`
/// when that module is present; bumped only on breaking plan/status changes.
pub const CONTRACT_VERSION: u32 = 1;
