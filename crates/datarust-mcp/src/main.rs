//! datarust-mcp — Model Context Protocol (MCP) server for datarust
//!
//! Exposes classical machine learning and data profiling tools over stdio
//! JSON-RPC 2.0 (compatible with Claude Desktop, Cursor, Glama, and Smithery).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

use datarust_mcp::execute_tool;

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

fn success_response(id: Value, result: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(result),
        error: None,
    }
}

fn error_response(id: Value, code: i64, message: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(json!({
            "code": code,
            "message": message
        })),
    }
}

fn tool_content(text: String) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": text
            }
        ]
    })
}

fn get_tools_manifest() -> Value {
    json!({
        "tools": [
            {
                "name": "profile_dataset",
                "description": "Profiles tabular data from CSV content, generating descriptive statistics, distribution metrics, pairwise correlations, and data quality checks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "csv_data": {
                            "type": "string",
                            "description": "The CSV content to profile (including header row)"
                        },
                        "target_column": {
                            "type": "string",
                            "description": "Optional name of the target column for leakage detection and associations"
                        }
                    },
                    "required": ["csv_data"]
                }
            },
            {
                "name": "train_linear_regression",
                "description": "Fits an ordinary least squares linear regression model on feature matrix X and target vector y.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "array",
                            "items": { "type": "array", "items": { "type": "number" } },
                            "description": "2D array of numeric features (rows x columns)"
                        },
                        "y": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "1D array of numeric target values"
                        }
                    },
                    "required": ["x", "y"]
                }
            },
            {
                "name": "scale_features",
                "description": "Transforms numerical features using standard, minmax, or robust scaling.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "array",
                            "items": { "type": "array", "items": { "type": "number" } },
                            "description": "2D array of features to scale"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["standard", "minmax", "robust"],
                            "description": "Scaler method to apply (default: 'standard')"
                        }
                    },
                    "required": ["x"]
                }
            },
            {
                "name": "evaluate_metrics",
                "description": "Computes regression metrics (R2, MSE, RMSE, MAE) or classification metrics (accuracy).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "y_true": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "Ground truth values"
                        },
                        "y_pred": {
                            "type": "array",
                            "items": { "type": "number" },
                            "description": "Predicted values"
                        },
                        "task": {
                            "type": "string",
                            "enum": ["regression", "classification"],
                            "description": "Task type (default: 'regression')"
                        }
                    },
                    "required": ["y_true", "y_pred"]
                }
            }
        ]
    })
}

fn handle_tool_call(name: &str, args: &Value) -> Result<String, String> {
    let result = execute_tool(name, args)?;
    serde_json::to_string_pretty(&result).map_err(|error| error.to_string())
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line_res in stdin.lock().lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let err = error_response(Value::Null, -32700, &format!("Parse error: {e}"));
                let out = serde_json::to_string(&err).unwrap();
                writeln!(stdout, "{out}")?;
                stdout.flush()?;
                continue;
            }
        };

        let id = request.id.clone().unwrap_or(Value::Null);

        match request.method.as_str() {
            "initialize" => {
                let resp = success_response(
                    id,
                    json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "datarust",
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    }),
                );
                let out = serde_json::to_string(&resp).unwrap();
                writeln!(stdout, "{out}")?;
                stdout.flush()?;
            }

            "notifications/initialized" => {
                // Client acknowledgment, no response needed.
            }

            "ping" => {
                let resp = success_response(id, json!({}));
                let out = serde_json::to_string(&resp).unwrap();
                writeln!(stdout, "{out}")?;
                stdout.flush()?;
            }

            "tools/list" => {
                let resp = success_response(id, get_tools_manifest());
                let out = serde_json::to_string(&resp).unwrap();
                writeln!(stdout, "{out}")?;
                stdout.flush()?;
            }

            "tools/call" => {
                let tool_name = request.params["name"].as_str().unwrap_or_default();
                let tool_args = &request.params["arguments"];

                let resp = match handle_tool_call(tool_name, tool_args) {
                    Ok(text) => success_response(id, tool_content(text)),
                    Err(err) => success_response(
                        id,
                        json!({
                            "isError": true,
                            "content": [
                                {
                                    "type": "text",
                                    "text": format!("Error: {err}")
                                }
                            ]
                        }),
                    ),
                };
                let out = serde_json::to_string(&resp).unwrap();
                writeln!(stdout, "{out}")?;
                stdout.flush()?;
            }

            other => {
                if request.id.is_some() {
                    let resp = error_response(id, -32601, &format!("Method not found: '{other}'"));
                    let out = serde_json::to_string(&resp).unwrap();
                    writeln!(stdout, "{out}")?;
                    stdout.flush()?;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tools_manifest_has_tools() {
        let manifest = get_tools_manifest();
        let tools = manifest["tools"].as_array().expect("tools array");
        assert!(tools.len() >= 4);
    }

    #[test]
    fn test_profile_dataset_tool() {
        let args = json!({
            "csv_data": "a,b\n1,2\n3,4\n"
        });
        let res = handle_tool_call("profile_dataset", &args);
        assert!(res.is_ok());
        let output = res.unwrap();
        assert!(output.contains("\"n_rows\": 2"));
    }

    #[test]
    fn test_linear_regression_tool() {
        let args = json!({
            "x": [[1.0], [2.0], [3.0]],
            "y": [2.0, 4.0, 6.0]
        });
        let res = handle_tool_call("train_linear_regression", &args);
        assert!(res.is_ok());
        let output = res.unwrap();
        assert!(output.contains("coefficients"));
    }
}
