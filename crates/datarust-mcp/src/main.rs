//! datarust-mcp — Model Context Protocol (MCP) server for datarust
//!
//! Exposes classical machine learning and data profiling tools over stdio
//! JSON-RPC 2.0 (compatible with Claude Desktop, Cursor, Glama, and Smithery).

use std::io::{self, BufRead, Write};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;
use datarust_profile::csv::{read_csv_reader, CsvReadOptions};
use datarust_profile::report::JsonReport;

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
    match name {
        "profile_dataset" => {
            let csv_data = args["csv_data"]
                .as_str()
                .ok_or_else(|| "Missing required 'csv_data' string argument".to_string())?;
            let target_column = args["target_column"].as_str();

            let reader = io::Cursor::new(csv_data.as_bytes());
            let table = read_csv_reader(reader, &CsvReadOptions::default())
                .map_err(|e| format!("CSV parsing error: {e}"))?;

            let mut profile = datarust_profile::profile_str_matrix(&table.rows, table.header.as_deref())
                .map_err(|e| format!("Profiling error: {e}"))?;

            if let Some(target) = target_column {
                profile = profile.with_target(target);
            }

            let report = JsonReport::from_profile(&profile);
            serde_json::to_string_pretty(&report)
                .map_err(|e| format!("Failed to serialize report: {e}"))
        }

        "train_linear_regression" => {
            let x_rows = args["x"]
                .as_array()
                .ok_or_else(|| "Missing 'x' matrix".to_string())?;
            let y_vec = args["y"]
                .as_array()
                .ok_or_else(|| "Missing 'y' vector".to_string())?;

            let mut x_parsed = Vec::with_capacity(x_rows.len());
            for row in x_rows {
                let row_arr = row.as_array().ok_or_else(|| "Row must be an array of numbers".to_string())?;
                let mut r = Vec::with_capacity(row_arr.len());
                for val in row_arr {
                    r.push(val.as_f64().ok_or_else(|| "Value must be a number".to_string())?);
                }
                x_parsed.push(r);
            }

            let mut y_parsed = Vec::with_capacity(y_vec.len());
            for val in y_vec {
                y_parsed.push(val.as_f64().ok_or_else(|| "Target value must be a number".to_string())?);
            }

            let matrix = Matrix::from_rows(x_parsed)
                .map_err(|e| format!("Matrix error: {e}"))?;

            let mut model = datarust::LinearRegression::new();
            model.fit(&matrix, &y_parsed)
                .map_err(|e| format!("Fitting error: {e}"))?;

            let predictions = model.predict(&matrix)
                .map_err(|e| format!("Prediction error: {e}"))?;

            let r2 = datarust::metrics::regression::r2_score(&y_parsed, &predictions)
                .unwrap_or(0.0);
            let mse = datarust::metrics::regression::mean_squared_error(&y_parsed, &predictions, true)
                .unwrap_or(0.0);

            let res = json!({
                "coefficients": model.coef(),
                "intercept": model.intercept(),
                "r2_score": r2,
                "mean_squared_error": mse,
                "predictions_sample": predictions.iter().take(10).collect::<Vec<_>>()
            });
            serde_json::to_string_pretty(&res).map_err(|e| e.to_string())
        }

        "scale_features" => {
            let x_rows = args["x"]
                .as_array()
                .ok_or_else(|| "Missing 'x' matrix".to_string())?;
            let method = args["method"].as_str().unwrap_or("standard");

            let mut x_parsed = Vec::with_capacity(x_rows.len());
            for row in x_rows {
                let row_arr = row.as_array().ok_or_else(|| "Row must be an array of numbers".to_string())?;
                let mut r = Vec::with_capacity(row_arr.len());
                for val in row_arr {
                    r.push(val.as_f64().ok_or_else(|| "Value must be a number".to_string())?);
                }
                x_parsed.push(r);
            }

            let matrix = Matrix::from_rows(x_parsed)
                .map_err(|e| format!("Matrix error: {e}"))?;

            let transformed = match method {
                "minmax" => {
                    let mut scaler = datarust::scaler::MinMaxScaler::new();
                    scaler.fit_transform(&matrix).map_err(|e| e.to_string())?
                }
                "robust" => {
                    let mut scaler = datarust::scaler::RobustScaler::new();
                    scaler.fit_transform(&matrix).map_err(|e| e.to_string())?
                }
                _ => {
                    let mut scaler = datarust::scaler::StandardScaler::new();
                    scaler.fit_transform(&matrix).map_err(|e| e.to_string())?
                }
            };

            let mut out = Vec::with_capacity(transformed.nrows());
            for i in 0..transformed.nrows() {
                out.push(transformed.row(i).to_vec());
            }

            let res = json!({
                "method": method,
                "n_rows": transformed.nrows(),
                "n_cols": transformed.ncols(),
                "scaled_features": out
            });
            serde_json::to_string_pretty(&res).map_err(|e| e.to_string())
        }

        "evaluate_metrics" => {
            let y_true_arr = args["y_true"]
                .as_array()
                .ok_or_else(|| "Missing 'y_true'".to_string())?;
            let y_pred_arr = args["y_pred"]
                .as_array()
                .ok_or_else(|| "Missing 'y_pred'".to_string())?;
            let task = args["task"].as_str().unwrap_or("regression");

            let mut y_true = Vec::with_capacity(y_true_arr.len());
            for v in y_true_arr {
                y_true.push(v.as_f64().ok_or_else(|| "Must be number".to_string())?);
            }

            let mut y_pred = Vec::with_capacity(y_pred_arr.len());
            for v in y_pred_arr {
                y_pred.push(v.as_f64().ok_or_else(|| "Must be number".to_string())?);
            }

            if task == "classification" {
                let acc = datarust::metrics::classification::accuracy_score(&y_true, &y_pred)
                    .map_err(|e| e.to_string())?;
                let res = json!({
                    "task": "classification",
                    "accuracy": acc
                });
                serde_json::to_string_pretty(&res).map_err(|e| e.to_string())
            } else {
                let r2 = datarust::metrics::regression::r2_score(&y_true, &y_pred)
                    .map_err(|e| e.to_string())?;
                let mse = datarust::metrics::regression::mean_squared_error(&y_true, &y_pred, true)
                    .map_err(|e| e.to_string())?;
                let mae = datarust::metrics::regression::mean_absolute_error(&y_true, &y_pred)
                    .map_err(|e| e.to_string())?;
                let rmse = mse.sqrt();

                let res = json!({
                    "task": "regression",
                    "r2_score": r2,
                    "mean_squared_error": mse,
                    "root_mean_squared_error": rmse,
                    "mean_absolute_error": mae
                });
                serde_json::to_string_pretty(&res).map_err(|e| e.to_string())
            }
        }

        _ => Err(format!("Unknown tool: '{name}'")),
    }
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
                            "version": "0.7.0"
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
