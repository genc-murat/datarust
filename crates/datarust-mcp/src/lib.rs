//! Tabular tools shared by the stdio server and the WebAssembly Worker.

use datarust::traits::{Predictor, Transformer};
use datarust::{LinearRegression, Matrix, RandomForestRegressor};
use datarust_profile::csv::{read_csv_reader, CsvReadOptions};
use datarust_profile::report::JsonReport;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};

type ToolResult<T> = Result<T, String>;

fn parse<T: DeserializeOwned>(args: &Value) -> ToolResult<T> {
    serde_json::from_value(args.clone()).map_err(|error| error.to_string())
}

fn matrix(rows: Vec<Vec<f64>>) -> ToolResult<Matrix> {
    #[cfg(target_arch = "wasm32")]
    if rows.len() > 10_000
        || rows.first().is_some_and(|row| row.len() > 64)
        || rows.iter().map(Vec::len).sum::<usize>() > 50_000
    {
        return Err("x is limited to 10,000 rows, 64 columns, and 50,000 values".into());
    }
    Matrix::from_rows(rows).map_err(|error| error.to_string())
}

/// Execute a tabular operation and return its structured JSON result.
pub fn execute_tool(name: &str, args: &Value) -> ToolResult<Value> {
    match name {
        "profile_dataset" => profile_dataset(parse(args)?),
        "train_linear_regression" => {
            let mut request: TrainRequest = parse(args)?;
            request.model = Model::LinearRegression;
            train(request)
        }
        "train" => train(parse(args)?),
        "scale_features" => scale(parse(args)?),
        "evaluate_metrics" => evaluate(parse(args)?),
        _ => Err(format!("Unknown tool: '{name}'")),
    }
}

#[derive(Deserialize)]
struct ProfileRequest {
    csv_data: String,
    target_column: Option<String>,
}

fn profile_dataset(request: ProfileRequest) -> ToolResult<Value> {
    let reader = std::io::Cursor::new(request.csv_data.as_bytes());
    let table = read_csv_reader(reader, &CsvReadOptions::default())
        .map_err(|error| format!("CSV parsing error: {error}"))?;
    #[cfg(target_arch = "wasm32")]
    if table.ncols() > 64 || table.nrows() > 10_000 {
        return Err("CSV data is limited to 10,000 rows and 64 columns".into());
    }
    let mut profile = datarust_profile::profile_str_matrix(&table.rows, table.header.as_deref())
        .map_err(|error| format!("Profiling error: {error}"))?;
    if let Some(target) = request.target_column {
        if !profile.columns.iter().any(|column| column.name == target) {
            return Err(format!("Unknown target column: '{target}'"));
        }
        profile = profile.with_target(&target);
    }
    serde_json::to_value(JsonReport::from_profile(&profile)).map_err(|error| error.to_string())
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Model {
    #[default]
    LinearRegression,
    DecisionTree,
    RandomForest,
}

#[derive(Deserialize)]
struct TrainRequest {
    #[serde(default)]
    model: Model,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
}

fn fit_and_predict(model: &mut impl Predictor, x: &Matrix, y: &[f64]) -> ToolResult<Vec<f64>> {
    model
        .fit(x, y)
        .map_err(|error| format!("Fitting error: {error}"))?;
    model
        .predict(x)
        .map_err(|error| format!("Prediction error: {error}"))
}

fn train(request: TrainRequest) -> ToolResult<Value> {
    let x = matrix(request.x)?;
    let y = request.y;
    let (model_name, predictions, parameters) = match request.model {
        Model::LinearRegression => {
            let mut model = LinearRegression::new();
            let predictions = fit_and_predict(&mut model, &x, &y)?;
            (
                "linear_regression",
                predictions,
                json!({ "coefficients": model.coef(), "intercept": model.intercept() }),
            )
        }
        Model::DecisionTree => {
            let mut model = datarust::tree::DecisionTreeRegressor::new().with_max_depth(Some(8));
            (
                "decision_tree",
                fit_and_predict(&mut model, &x, &y)?,
                json!({}),
            )
        }
        Model::RandomForest => {
            let mut model = RandomForestRegressor::new()
                .with_n_estimators(10)
                .with_max_depth(Some(8));
            (
                "random_forest",
                fit_and_predict(&mut model, &x, &y)?,
                json!({}),
            )
        }
    };
    let scores = regression_metrics(&y, &predictions)?;
    let mut output = parameters;
    output["model"] = json!(model_name);
    output["status"] = json!("fitted");
    output["r2_score"] = scores["r2_score"].clone();
    output["mean_squared_error"] = scores["mean_squared_error"].clone();
    output["predictions_sample"] = json!(&predictions[..predictions.len().min(10)]);
    Ok(output)
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ScaleMethod {
    #[default]
    Standard,
    Minmax,
    Robust,
}

#[derive(Deserialize)]
struct ScaleRequest {
    #[serde(default)]
    method: ScaleMethod,
    x: Vec<Vec<f64>>,
}

fn scale(request: ScaleRequest) -> ToolResult<Value> {
    let x = matrix(request.x)?;
    let (method, result) = match request.method {
        ScaleMethod::Standard => (
            "standard",
            datarust::scaler::StandardScaler::new().fit_transform(&x),
        ),
        ScaleMethod::Minmax => (
            "minmax",
            datarust::scaler::MinMaxScaler::new().fit_transform(&x),
        ),
        ScaleMethod::Robust => (
            "robust",
            datarust::scaler::RobustScaler::new().fit_transform(&x),
        ),
    };
    let transformed = result.map_err(|error| error.to_string())?;
    transformed
        .validate_finite()
        .map_err(|error| error.to_string())?;
    let out: Vec<&[f64]> = (0..transformed.nrows())
        .map(|i| transformed.row(i))
        .collect();
    Ok(json!({
        "method": method,
        "n_rows": transformed.nrows(),
        "n_cols": transformed.ncols(),
        "scaled_features": out,
    }))
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MetricTask {
    #[default]
    Regression,
    Classification,
}

#[derive(Deserialize)]
struct EvaluateRequest {
    #[serde(default)]
    task: MetricTask,
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
}

fn evaluate(request: EvaluateRequest) -> ToolResult<Value> {
    match request.task {
        MetricTask::Regression => regression_metrics(&request.y_true, &request.y_pred),
        MetricTask::Classification => {
            let accuracy =
                datarust::metrics::classification::accuracy_score(&request.y_true, &request.y_pred)
                    .map_err(|error| error.to_string())?;
            Ok(json!({ "task": "classification", "accuracy": accuracy }))
        }
    }
}

fn regression_metrics(y: &[f64], pred: &[f64]) -> ToolResult<Value> {
    use datarust::metrics::regression::{mean_absolute_error, mean_squared_error, r2_score};
    let r2 = (y.len() >= 2)
        .then(|| r2_score(y, pred))
        .transpose()
        .map_err(|error| error.to_string())?;
    let mse = mean_squared_error(y, pred, true).map_err(|error| error.to_string())?;
    let mae = mean_absolute_error(y, pred).map_err(|error| error.to_string())?;
    if ![mse, mae].into_iter().chain(r2).all(f64::is_finite) {
        return Err("Regression metrics exceed the finite numeric range".into());
    }
    Ok(json!({
        "task": "regression",
        "r2_score": r2,
        "mean_squared_error": mse,
        "root_mean_squared_error": mse.sqrt(),
        "mean_absolute_error": mae,
    }))
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
        static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }

    #[derive(Deserialize)]
    struct Request {
        name: String,
        arguments: Value,
    }

    // The Worker fills this buffer, calls execute synchronously and reads the
    // result before invoking the instance again. Rust retains both allocations.
    #[no_mangle]
    extern "C" fn datarust_input(len: usize) -> *mut u8 {
        INPUT.with(|input| {
            let mut input = input.borrow_mut();
            input.resize(len, 0);
            input.as_mut_ptr()
        })
    }

    #[no_mangle]
    extern "C" fn datarust_execute() -> usize {
        let result = INPUT.with(|input| {
            let request: Request =
                serde_json::from_slice(&input.borrow()).map_err(|error| error.to_string())?;
            execute_tool(&request.name, &request.arguments)
        });
        let output = match result {
            Ok(value) => json!({ "result": value }),
            Err(message) => json!({ "error": message }),
        };
        OUTPUT.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            *buffer = output.to_string().into_bytes();
            buffer.len()
        })
    }

    #[no_mangle]
    extern "C" fn datarust_output() -> *const u8 {
        OUTPUT.with(|output| output.borrow().as_ptr())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn singleton_regression_preserves_defined_metrics() {
        let result = super::regression_metrics(&[1.], &[3.]).unwrap();
        assert!(result["r2_score"].is_null());
        assert_eq!(result["mean_squared_error"], 4.);
        assert_eq!(result["mean_absolute_error"], 2.);
        assert!(super::regression_metrics(&[1.], &[]).is_err());
    }
}
