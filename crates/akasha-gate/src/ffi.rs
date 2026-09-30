//! C ABI for non-Rust hosts (`cdylib`).
//!
//! Evaluate Path A only — no tool executor. Hosts own side effects.
//!
//! # Threading
//!
//! Functions are re-entrant and allocate per call. The library holds no
//! process-wide mutable gate state. Callers may invoke
//! [`akasha_gate_evaluate_json`] from multiple threads concurrently.
//!
//! # Ownership
//!
//! Returned JSON strings are heap-allocated (`malloc` / Rust allocator via
//! [`CString::into_raw`]). Free **only** with [`akasha_gate_string_free`].
//! Do not mix with the host's `free` unless the host uses the same allocator
//! (not guaranteed across platforms).

use crate::planner::{evaluate_gate, GateSignals, ToolProposal};
use crate::types::{PlanStatus, ToolCallPlan, ToolSpec};
use crate::CONTRACT_VERSION;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

const ABI_VERSION: &str = "1";

#[derive(Debug, Deserialize)]
struct FfiEvaluateRequest {
    #[serde(default)]
    contract_version: Option<u32>,
    tools: BTreeMap<String, ToolSpec>,
    proposal: ToolProposal,
    signals: FfiSignals,
    #[serde(default)]
    request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FfiSignals {
    authorized: f64,
    sufficient_context: f64,
    #[serde(default)]
    capability_present: Option<f64>,
    #[serde(default)]
    confirmation_needed: Option<f64>,
    #[serde(default)]
    confirmation_given: bool,
    #[serde(default)]
    capability_signals: Option<BTreeMap<String, f64>>,
    /// Wire envelope uses nested `risk.score`; flat `risk_score` also accepted.
    #[serde(default)]
    risk: Option<FfiRisk>,
    #[serde(default)]
    risk_score: Option<f64>,
    #[serde(default)]
    budget_remaining: Option<f64>,
    #[serde(default)]
    consequence: Option<f64>,
    #[serde(default)]
    context: Option<BTreeMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct FfiRisk {
    score: f64,
}

impl FfiSignals {
    fn into_gate_signals(self) -> GateSignals {
        let risk_score = self.risk_score.or_else(|| self.risk.map(|r| r.score));
        GateSignals {
            authorized: self.authorized,
            sufficient_context: self.sufficient_context,
            capability_present: self.capability_present,
            confirmation_needed: self.confirmation_needed,
            risk_score,
            confirmation_given: self.confirmation_given,
            capability_signals: self.capability_signals,
            budget_remaining: self.budget_remaining,
            consequence: self.consequence,
            context: self.context,
        }
    }
}

#[derive(Debug, Serialize)]
struct FfiEvaluateResponse<'a> {
    ok: bool,
    contract_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<&'a BTreeMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    choice_probability: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    choice_confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    risk_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    executable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn status_str(status: PlanStatus) -> &'static str {
    match status {
        PlanStatus::Ready => "ready",
        PlanStatus::Abstain => "abstain",
        PlanStatus::Blocked => "blocked",
    }
}

fn plan_to_response<'a>(
    plan: &'a ToolCallPlan,
    request_id: Option<&'a str>,
) -> FfiEvaluateResponse<'a> {
    FfiEvaluateResponse {
        ok: true,
        contract_version: CONTRACT_VERSION,
        status: Some(status_str(plan.status)),
        tool_name: plan.tool_name.as_deref(),
        arguments: plan.arguments.as_ref(),
        reason: Some(plan.reason.as_str()),
        choice_probability: Some(plan.choice_probability),
        choice_confidence: Some(plan.choice_confidence),
        risk_score: plan.risk_score,
        executable: Some(plan.executable()),
        request_id,
        error: None,
    }
}

fn error_response(message: impl Into<String>) -> String {
    json!({
        "ok": false,
        "contract_version": CONTRACT_VERSION,
        "error": message.into(),
    })
    .to_string()
}

fn evaluate_request_json(raw: &str) -> String {
    let request: FfiEvaluateRequest = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(err) => return error_response(format!("invalid request JSON: {err}")),
    };
    if let Some(version) = request.contract_version {
        if version != CONTRACT_VERSION {
            return error_response(format!(
                "unsupported contract_version {version}; library supports {CONTRACT_VERSION}"
            ));
        }
    }
    let signals = request.signals.into_gate_signals();
    match evaluate_gate(&request.tools, &request.proposal, &signals, None) {
        Ok(plan) => match serde_json::to_string(&plan_to_response(
            &plan,
            request.request_id.as_deref(),
        )) {
            Ok(s) => s,
            Err(err) => error_response(format!("serialize plan failed: {err}")),
        },
        Err(err) => error_response(err.to_string()),
    }
}

fn to_c_string(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => {
            // Embedded NUL — return a minimal error JSON without NULs.
            CString::new(error_response("response contained interior NUL"))
                .map(CString::into_raw)
                .unwrap_or(ptr::null_mut())
        }
    }
}

/// ABI version string (`"1"`). Static; do not free.
#[no_mangle]
pub extern "C" fn akasha_gate_abi_version() -> *const c_char {
    static VERSION: &[u8] = b"1\0";
    debug_assert_eq!(
        core::str::from_utf8(&VERSION[..VERSION.len() - 1]).unwrap(),
        ABI_VERSION
    );
    VERSION.as_ptr() as *const c_char
}

/// Wire `CONTRACT_VERSION` (currently `1`).
#[no_mangle]
pub extern "C" fn akasha_gate_contract_version() -> u32 {
    CONTRACT_VERSION
}

/// Evaluate Path A from a UTF-8 JSON request (wire-shaped).
///
/// Returns a heap-allocated UTF-8 JSON string. Always free with
/// [`akasha_gate_string_free`]. Never executes tools.
///
/// On parse/evaluate failure the JSON has `"ok": false` and an `"error"`
/// field (still allocated). Returns null only if allocation fails.
#[no_mangle]
pub extern "C" fn akasha_gate_evaluate_json(request_json: *const c_char) -> *mut c_char {
    if request_json.is_null() {
        return to_c_string(error_response("request_json was null"));
    }
    let cstr = unsafe { CStr::from_ptr(request_json) };
    let raw = match cstr.to_str() {
        Ok(s) => s,
        Err(err) => return to_c_string(error_response(format!("request is not UTF-8: {err}"))),
    };
    to_c_string(evaluate_request_json(raw))
}

/// Free a string returned by [`akasha_gate_evaluate_json`].
///
/// No-op on null. Must not be used on pointers from other allocators.
#[no_mangle]
pub extern "C" fn akasha_gate_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(ptr));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn ffi_ready_read_via_json() {
        let request = include_str!("../../../tests/contracts/fixtures/path_a_ready_request.json");
        let out = evaluate_request_json(request);
        let value: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(value["ok"], true);
        assert_eq!(value["status"], "ready");
        assert_eq!(value["tool_name"], "fs.read");
        assert_eq!(value["request_id"], "req-ready-001");
    }

    #[test]
    fn ffi_c_entry_roundtrip() {
        let request = CString::new(include_str!(
            "../../../tests/contracts/fixtures/path_a_ready_request.json"
        ))
        .unwrap();
        let ptr = akasha_gate_evaluate_json(request.as_ptr());
        assert!(!ptr.is_null());
        let json = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned();
        akasha_gate_string_free(ptr);
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["status"], "ready");
        assert_eq!(unsafe { CStr::from_ptr(akasha_gate_abi_version()) }.to_str().unwrap(), ABI_VERSION);
        assert_eq!(akasha_gate_contract_version(), 1);
    }
}
