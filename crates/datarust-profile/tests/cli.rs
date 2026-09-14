//! CLI integration tests: exercise the real `datarust-profile` binary via
//! `CARGO_BIN_EXE_datrust-profile` (requires the `cli` feature).

#![cfg(feature = "cli")]

use std::process::Command;

fn bin() -> String {
    std::env::var("CARGO_BIN_EXE_datarust-profile")
        .expect("CARGO_BIN_EXE_datarust-profile is set for integration tests")
}

fn write_csv(dir: &tempfile::TempDir, name: &str, content: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, content).unwrap();
    path
}

const SAMPLE: &str =
    "age,income,city\n34,58000,Istanbul\n45,92000,Ankara\n29,NA,Izmir\n52,110000,Istanbul\n";

#[test]
fn help_and_version_succeed() {
    for flag in ["--help", "--version"] {
        let out = Command::new(bin()).arg(flag).output().unwrap();
        assert!(out.status.success(), "{flag} failed");
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("datarust-profile"), "{flag}: {text}");
    }
}

#[test]
fn html_output_written() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csv(&dir, "in.csv", SAMPLE);
    let out_path = dir.path().join("out.html");

    let out = Command::new(bin())
        .arg(&input)
        .arg("-o")
        .arg(&out_path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let html = std::fs::read_to_string(&out_path).unwrap();
    assert!(html.contains("<!DOCTYPE html") || html.contains("<html"));
    // stderr mentions the written file.
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("wrote"), "stderr: {err}");
}

#[test]
fn json_output_written() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csv(&dir, "in.csv", SAMPLE);
    let out_path = dir.path().join("out.json");

    let out = Command::new(bin())
        .arg(&input)
        .arg("--json")
        .arg(&out_path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let json = std::fs::read_to_string(&out_path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(value.get("profile").is_some());
    assert!(value.get("quality").is_some());
}

#[test]
fn pretty_prints_to_stdout() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csv(&dir, "in.csv", SAMPLE);

    let out = Command::new(bin())
        .arg(&input)
        .arg("--pretty")
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("Dataset profile"));
    assert!(text.contains("age"));
    assert!(text.contains("Findings"));
}

#[test]
fn default_without_flags_prints_pretty() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csv(&dir, "in.csv", SAMPLE);

    let out = Command::new(bin()).arg(&input).output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("Dataset profile"));
}

#[test]
fn streaming_matches_default_shape() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csv(&dir, "in.csv", SAMPLE);

    let out = Command::new(bin())
        .arg(&input)
        .arg("--streaming")
        .arg("--pretty")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("4 rows"));
}

#[test]
fn target_flag_flags_leakage() {
    let dir = tempfile::tempdir().unwrap();
    // income correlates perfectly with age ⇒ leakage hint when age is target.
    let input = write_csv(
        &dir,
        "leak.csv",
        "age,income\n30,30000\n40,40000\n50,50000\n60,60000\n",
    );

    let out = Command::new(bin())
        .arg(&input)
        .arg("--target")
        .arg("age")
        .arg("--pretty")
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("leakage"), "output: {text}");
}

#[test]
fn unknown_target_is_usage_error_exit_2() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csv(&dir, "in.csv", SAMPLE);

    let out = Command::new(bin())
        .arg(&input)
        .arg("--target")
        .arg("nope")
        .output()
        .unwrap();
    // Usage-class error: the header is checked before the file is profiled.
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("not found"), "stderr: {err}");
    assert!(err.contains("age"), "stderr: {err}");
}

#[test]
fn missing_input_is_runtime_error_exit_1() {
    let out = Command::new(bin())
        .arg("/nonexistent/nope.csv")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn ragged_csv_is_runtime_error_exit_1() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csv(&dir, "bad.csv", "a,b\n1,2,3\n");

    let out = Command::new(bin()).arg(&input).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("expected 2"), "stderr: {err}");
}

#[test]
fn types_flag_forces_schema() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csv(&dir, "in.csv", SAMPLE);

    let out = Command::new(bin())
        .arg(&input)
        .arg("--types")
        .arg("numeric,categorical,categorical")
        .arg("--pretty")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    // income was forced categorical.
    assert!(text.contains("income"), "output: {text}");
}

#[test]
fn types_with_wrong_length_is_error() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csv(&dir, "in.csv", SAMPLE);

    let out = Command::new(bin())
        .arg(&input)
        .arg("--types")
        .arg("numeric")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("schema has 1 entries"), "stderr: {err}");
}

#[test]
fn delimiter_flag_parses_semicolons() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csv(&dir, "semi.csv", "a;b\n1;2\n3;4\n");

    let out = Command::new(bin())
        .arg(&input)
        .arg("--delimiter")
        .arg(";")
        .arg("--pretty")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("2 rows"), "output: {text}");
}
