use crate::types::ToolSpec;
use serde_json::Value;
use std::collections::BTreeMap;

/// Validate the common JSON-schema subset used by Akasha tool specs.
pub fn validate_tool_arguments(
    spec: &ToolSpec,
    arguments: Option<&BTreeMap<String, Value>>,
) -> Result<(), String> {
    let Some(schema) = spec.parameters.as_ref() else {
        return Ok(());
    };
    let obj = schema
        .as_object()
        .ok_or_else(|| "invalid tool parameter schema".to_string())?;
    let properties = obj
        .get("properties")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "invalid tool parameter schema".to_string())?;
    let required = obj
        .get("required")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let values = arguments.cloned().unwrap_or_default();

    let mut missing = Vec::new();
    for name in &required {
        let key = name.as_str().unwrap_or_default();
        if !values.contains_key(key) {
            missing.push(key.to_string());
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "missing required argument(s): {}",
            missing.join(", ")
        ));
    }

    if obj.get("additionalProperties") == Some(&Value::Bool(false)) {
        let unknown: Vec<_> = values
            .keys()
            .filter(|k| !properties.contains_key(k.as_str()))
            .cloned()
            .collect();
        if !unknown.is_empty() {
            return Err(format!("unknown argument(s): {}", unknown.join(", ")));
        }
    }

    for (name, value) in &values {
        let Some(rule) = properties.get(name) else {
            continue;
        };
        if let Some(expected) = rule.get("type").and_then(|v| v.as_str()) {
            if !matches_type(value, expected) {
                return Err(format!("argument '{name}' must be {expected}"));
            }
        }
        if let Some(enum_vals) = rule.get("enum").and_then(|v| v.as_array()) {
            if !enum_vals.iter().any(|item| item == value) {
                return Err(format!("argument '{name}' is outside its enum"));
            }
        }
    }
    Ok(())
}

fn matches_type(value: &Value, expected: &str) -> bool {
    match expected {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.is_number(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "null" => value.is_null(),
        _ => true,
    }
}
