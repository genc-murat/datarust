//! `datarust-profile` command-line interface (the `cli` feature).
//!
//! Profiles a CSV file and writes an HTML report, a JSON report, and/or a
//! terminal summary. Exit codes: `0` success, `1` runtime error (I/O or
//! parsing), `2` usage error.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;

use datarust_profile::report;
use datarust_profile::types::ColumnType;
use datarust_profile::{
    profile_csv_streaming_with, profile_csv_with, profile_csv_with_types, StreamingOptions,
};

/// One-call data profiling and data-quality reports for CSV files.
#[derive(Debug, Parser)]
#[command(name = "datarust-profile", version, about)]
struct Args {
    /// Path to the CSV file to profile.
    input: PathBuf,

    /// Write a self-contained HTML report to this path.
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Write a JSON report to this path (enables profile + quality findings).
    #[arg(long, value_name = "FILE")]
    json: Option<PathBuf>,

    /// Print a terminal summary to stdout (auto color: TTY + no NO_COLOR).
    #[arg(long)]
    pretty: bool,

    /// Force ANSI color for `--pretty` (default: only on a TTY).
    #[arg(long, requires = "pretty")]
    color: bool,

    /// Profile with the streaming single-pass algorithm (bounded memory).
    #[arg(long)]
    streaming: bool,

    /// Designate a target column for leakage hints (name must match the
    /// header exactly).
    #[arg(long, value_name = "COLUMN")]
    target: Option<String>,

    /// Field delimiter (single ASCII byte; default `,`).
    #[arg(long, value_name = "CHAR", default_value = ",")]
    delimiter: String,

    /// Treat the file as headerless (columns become x0..x{n-1}).
    #[arg(long)]
    no_header: bool,

    /// Force column types, e.g. `numeric,categorical,numeric` (must match the
    /// column count).
    #[arg(long, value_name = "TYPES")]
    types: Option<String>,
}

/// Command-line failure: usage errors exit `2`, runtime errors exit `1`.
struct CliError {
    message: String,
    usage: bool,
}

impl CliError {
    fn usage(message: impl Into<String>) -> Self {
        CliError {
            message: message.into(),
            usage: true,
        }
    }

    fn runtime(message: impl Into<String>) -> Self {
        CliError {
            message: message.into(),
            usage: false,
        }
    }

    fn code(&self) -> u8 {
        if self.usage {
            2
        } else {
            1
        }
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e.message);
            ExitCode::from(e.code())
        }
    }
}

fn run(args: &Args) -> Result<(), CliError> {
    if args.delimiter.len() != 1 || !args.delimiter.is_ascii() {
        return Err(CliError::usage(format!(
            "--delimiter must be a single ASCII byte, got {:?}",
            args.delimiter
        )));
    }
    let delimiter = args.delimiter.as_bytes()[0];
    let opts = datarust_profile::CsvReadOptions {
        delimiter,
        quote: Some(b'"'),
        has_header: !args.no_header,
        trim: false,
    };

    if args.streaming && (delimiter != b',' || args.no_header || args.types.is_some()) {
        return Err(CliError::usage(
            "--streaming currently requires the default options (comma delimiter, header row, no --types)",
        ));
    }

    if let Some(spec) = &args.types {
        if delimiter != b',' {
            return Err(CliError::usage(
                "--types currently requires the default comma delimiter",
            ));
        }
        parse_schema(spec)?;
    }

    // Validate --target against the header before profiling the whole file.
    if let Some(target) = &args.target {
        let header = datarust_profile::csv::read_csv_header(&args.input, &opts)
            .map_err(|e| CliError::runtime(e.to_string()))?;
        let names: Vec<String> = if opts.has_header {
            header
        } else {
            (0..header.len()).map(|j| format!("x{j}")).collect()
        };
        if !names.iter().any(|n| n == target) {
            return Err(CliError::usage(format!(
                "--target column '{target}' not found; columns are: {}",
                summarize_names(&names)
            )));
        }
    }

    let mut profile = if args.streaming {
        let s = StreamingOptions::default();
        profile_csv_streaming_with(&args.input, &s).map_err(|e| CliError::runtime(e.to_string()))?
    } else if let Some(spec) = &args.types {
        let schema = parse_schema(spec).map_err(|e| CliError::runtime(e.message.clone()))?;
        profile_csv_with_types(&args.input, &schema)
            .map_err(|e| CliError::runtime(e.to_string()))?
    } else {
        profile_csv_with(&args.input, &opts, None).map_err(|e| CliError::runtime(e.to_string()))?
    };
    if let Some(target) = &args.target {
        profile = profile.with_target(target);
    }

    let wrote_something = args.output.is_some() || args.json.is_some() || args.pretty;

    if let Some(path) = &args.output {
        let html = report::to_html(&profile);
        write_out(path, html.as_bytes())?;
        eprintln!("wrote {} ({} bytes)", path.display(), html.len());
    }
    if let Some(path) = &args.json {
        let json = report::to_json(&report::JsonReport::from_profile(&profile))
            .map_err(|e| CliError::runtime(e.to_string()))?;
        write_out(path, json.as_bytes())?;
        eprintln!("wrote {} ({} bytes)", path.display(), json.len());
    }
    if args.pretty || !wrote_something {
        let text = if args.color {
            datarust_profile::report::to_terminal(&profile, true)
        } else {
            report::to_terminal_auto(&profile)
        };
        let mut stdout = std::io::stdout();
        stdout
            .write_all(text.as_bytes())
            .map_err(|e| CliError::runtime(e.to_string()))?;
        stdout
            .flush()
            .map_err(|e| CliError::runtime(e.to_string()))?;
    }
    Ok(())
}

fn write_out(path: &Path, bytes: &[u8]) -> Result<(), CliError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(CliError::runtime(format!(
                "output directory {} does not exist",
                parent.display()
            )));
        }
    }
    std::fs::write(path, bytes)
        .map_err(|e| CliError::runtime(format!("cannot write {}: {e}", path.display())))
}

/// Joins column names for error messages, truncated after 20 entries so a
/// wide file cannot flood stderr.
fn summarize_names(names: &[String]) -> String {
    const MAX: usize = 20;
    let mut joined = names
        .iter()
        .take(MAX)
        .map(|n| n.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > MAX {
        joined.push_str(&format!(", … ({} more)", names.len() - MAX));
    }
    joined
}

/// Parses `numeric,categorical,...` into a `Vec<ColumnType>`.
fn parse_schema(spec: &str) -> Result<Vec<ColumnType>, CliError> {
    spec.split(',')
        .map(|t| match t.trim().to_ascii_lowercase().as_str() {
            "numeric" | "num" => Ok(ColumnType::Numeric),
            "categorical" | "cat" => Ok(ColumnType::Categorical),
            other => Err(CliError::usage(format!(
                "unknown column type '{other}' (expected 'numeric' or 'categorical')"
            ))),
        })
        .collect()
}
