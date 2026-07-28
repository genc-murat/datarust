//! A dependency-free, self-contained HTML report.
//!
//! The output is a single HTML document with inline CSS — no external
//! assets, no JavaScript framework. It is intentionally minimal so it can be
//! embedded in Jupyter-style widgets, emailed, or served as a static file.
//!
//! Unlike [`super::json`], the HTML renderer works without the `serde`
//! feature because it reads the profile fields directly.

use crate::profile::{CategoricalStats, ColumnProfile, DatasetProfile, NumericStats};
use crate::quality::checks::run_checks;
use crate::quality::{QualityIssue, Thresholds};
use crate::types::{ColumnType, Severity};

/// Escapes `s` for safe inclusion in HTML text content / attributes.
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Formats an `f64` for display, rendering `NaN`/infinite values as blanks.
fn fmt_num(v: f64) -> String {
    if !v.is_finite() {
        "—".to_string()
    } else if v.abs() >= 1e6 || (v.abs() < 1e-4 && v != 0.0) {
        format!("{:.3e}", v)
    } else {
        format!("{:.4}", v)
    }
}

const CSS: &str = r#"
:root { color-scheme: light dark; }
body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0 auto; max-width: 72rem; padding: 2rem; color: #1b1b1b;
       background: #fafaf7; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .5rem; border-bottom: 1px solid #ddd; padding-bottom: .25rem; }
.summary { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }
.summary div { background: #fff; border: 1px solid #e3e3e3; border-radius: 6px;
               padding: .75rem 1rem; min-width: 9rem; }
.summary dt { font-size: .75rem; text-transform: uppercase; letter-spacing: .03em;
              color: #777; margin: 0; }
.summary dd { margin: .25rem 0 0; font-size: 1.15rem; font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; }
th, td { padding: .35rem .5rem; text-align: right; border-bottom: 1px solid #eee;
         white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #f2f2ee; font-weight: 600; position: sticky; top: 0; }
tbody tr:hover { background: #f7f7f2; }
.badge { display: inline-block; padding: .05rem .4rem; border-radius: 3px;
         font-size: .7rem; text-transform: uppercase; letter-spacing: .03em; }
.badge.numeric { background: #e8f0fe; color: #1a56c4; }
.badge.categorical { background: #f6e8fe; color: #8a2be2; }
.findings { list-style: none; padding: 0; }
.findings li { padding: .4rem .6rem; margin: .25rem 0; border-radius: 4px; border: 1px solid #eee; }
.findings .critical { border-left: 4px solid #c62828; background: #fdecea; }
.findings .warning  { border-left: 4px solid #ef6c00; background: #fff4e5; }
.findings .info     { border-left: 4px solid #1976d2; background: #e8f0fe; }
@media (prefers-color-scheme: dark) {
  body { background: #161616; color: #e6e6e6; }
  .summary div { background: #1e1e1e; border-color: #333; }
  thead th { background: #242424; }
  tbody tr:hover { background: #1e1e1e; }
}
"#;

/// Renders `profile` and its default-threshold quality findings as a single
/// self-contained HTML document.
pub fn to_html(profile: &DatasetProfile) -> String {
    let findings = run_checks(profile, &Thresholds::default());
    to_html_with(profile, &findings)
}

/// Renders `profile` together with an explicit list of `findings`.
pub fn to_html_with(profile: &DatasetProfile, findings: &[QualityIssue]) -> String {
    let mut html = String::with_capacity(16 * 1024);
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    html.push_str("<title>datarust-profile report</title>\n<style>");
    html.push_str(CSS);
    html.push_str("</style>\n</head>\n<body>\n");

    html.push_str("<h1>Dataset profile</h1>\n");

    // --- Summary cards -----------------------------------------------------
    html.push_str("<dl class=\"summary\">\n");
    summary_card(&mut html, "Rows", profile.n_rows.to_string());
    summary_card(&mut html, "Columns", profile.n_columns.to_string());
    summary_card(&mut html, "Memory", format_bytes(profile.memory_bytes));
    summary_card(
        &mut html,
        "Duplicate rows",
        format!(
            "{} ({:.1}%)",
            profile.duplicate_rows,
            profile.duplicate_fraction * 100.0
        ),
    );
    summary_card(&mut html, "Findings", findings.len().to_string());
    html.push_str("</dl>\n");

    // --- Quality findings --------------------------------------------------
    if findings.is_empty() {
        html.push_str("<p>No data-quality findings.</p>\n");
    } else {
        html.push_str("<h2>Data quality findings</h2>\n<ul class=\"findings\">\n");
        for issue in findings {
            let class = match issue.severity {
                Severity::Critical => "critical",
                Severity::Warning => "warning",
                Severity::Info => "info",
            };
            html.push_str(&format!(
                "<li class=\"{class}\"><strong>{sev}</strong> — {msg}</li>\n",
                class = class,
                sev = esc(&issue.severity.to_string()),
                msg = esc(&issue.message)
            ));
        }
        html.push_str("</ul>\n");
    }

    // --- Column table ------------------------------------------------------
    html.push_str("<h2>Per-column profile</h2>\n<table>\n<thead>\n<tr>\n");
    for header in [
        "column", "type", "count", "missing", "mean", "std", "min", "Q1", "median", "Q3", "max",
        "unique", "top", "freq",
    ] {
        html.push_str(&format!("<th>{header}</th>\n"));
    }
    html.push_str("</tr>\n</thead>\n<tbody>\n");

    for col in &profile.columns {
        render_column_row(&mut html, col);
    }
    html.push_str("</tbody>\n</table>\n");
    html.push_str("</body>\n</html>\n");
    html
}

fn summary_card(html: &mut String, label: &str, value: String) {
    html.push_str(&format!(
        "<div><dt>{label}</dt><dd>{value}</dd></div>\n",
        label = esc(label),
        value = esc(&value)
    ));
}

fn render_column_row(html: &mut String, col: &ColumnProfile) {
    html.push_str("<tr>");
    html.push_str(&format!("<td>{}</td>", esc(&col.name)));

    let (badge, class) = match col.column_type {
        ColumnType::Numeric => ("numeric", "numeric"),
        ColumnType::Categorical => ("categorical", "categorical"),
    };
    html.push_str(&format!(
        "<td><span class=\"badge {class}\">{badge}</span></td>"
    ));

    // count + missing (common to both types).
    html.push_str(&format!("<td>{}</td>", col.count));
    let missing = format!(
        "{} ({:.1}%)",
        col.missing_count,
        col.missing_fraction * 100.0
    );
    html.push_str(&format!("<td>{}</td>", esc(&missing)));

    match (&col.numeric, &col.categorical) {
        (Some(n), None) => render_numeric_cells(html, n),
        (None, Some(c)) => render_categorical_cells(html, c),
        _ => {
            // All-missing numeric column or empty categorical: blank cells.
            for _ in 0..9 {
                html.push_str("<td>—</td>");
            }
        }
    }
    html.push_str("</tr>\n");
}

fn render_numeric_cells(html: &mut String, n: &NumericStats) {
    let cells = [
        fmt_num(n.mean),
        fmt_num(n.std),
        fmt_num(n.five.min),
        fmt_num(n.five.q1),
        fmt_num(n.five.median),
        fmt_num(n.five.q3),
        fmt_num(n.five.max),
    ];
    for c in cells {
        html.push_str(&format!("<td>{}</td>", esc(&c)));
    }
    // Two trailing categorical columns are empty for numeric rows.
    html.push_str("<td>—</td><td>—</td>");
}

fn render_categorical_cells(html: &mut String, c: &CategoricalStats) {
    // Six numeric cells empty for categorical rows.
    for _ in 0..6 {
        html.push_str("<td>—</td>");
    }
    html.push_str(&format!("<td>{}</td>", c.unique));
    html.push_str(&format!("<td>{}</td>", esc(&c.top)));
    html.push_str(&format!("<td>{}</td>", c.freq));
}

/// Formats a byte count as a short human-readable size.
fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{:.1} {}", value, UNITS[unit])
    }
}
