//! Terminal-friendly plain-text rendering of a profile (the `pretty` feature).
//!
//! A fixed-width summary designed for quick shell inspection — the textual
//! counterpart of the HTML report. Color is ANSI-only and disabled
//! automatically on non-terminal output and when `NO_COLOR` is set; no
//! external dependency is involved.

use std::io::IsTerminal;

use crate::profile::{ColumnProfile, DatasetProfile};
use crate::quality::{run_checks, QualityIssue, Thresholds};
use crate::types::{ColumnType, Severity};

/// Renders the profile and its quality findings as a fixed-width
/// terminal table. `color` forces ANSI styling; callers that want
/// automatic TTY detection should use [`to_terminal_auto`].
pub fn to_terminal(profile: &DatasetProfile, color: bool) -> String {
    let findings = run_checks(profile, &Thresholds::default());
    to_terminal_with(profile, &findings, color)
}

/// Like [`to_terminal`], but enables ANSI color only when stdout is a
/// terminal and `NO_COLOR` is unset.
pub fn to_terminal_auto(profile: &DatasetProfile) -> String {
    let use_color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    to_terminal(profile, use_color)
}

/// Renders the profile with pre-computed findings.
pub fn to_terminal_with(
    profile: &DatasetProfile,
    findings: &[QualityIssue],
    color: bool,
) -> String {
    let mut out = String::with_capacity(4096);
    let c = if color { Palette::on() } else { Palette::off() };

    // ── Header ─────────────────────────────────────────────────────────
    let title = "Dataset profile";
    out.push_str(&c.bold.paint(title));
    out.push('\n');
    let dup = format!(
        "{} row{} × {} column{}, {} duplicate row{}, {}",
        profile.n_rows,
        plural(profile.n_rows),
        profile.n_columns,
        plural(profile.n_columns),
        profile.duplicate_rows,
        plural(profile.duplicate_rows),
        human_bytes(profile.memory_bytes),
    );
    out.push_str(&c.dim.paint(&dup));
    out.push_str("\n\n");

    // ── Column table ───────────────────────────────────────────────────
    let mut name_w = "column".len();
    let mut type_w = "type".len();
    let mut miss_w = "missing".len();
    let mut sum_w = "summary".len();
    let rows: Vec<(String, String, String, String)> = profile
        .columns
        .iter()
        .map(|col| {
            (
                sanitize(&col.name),
                col.column_type.to_string(),
                format!(
                    "{:>4} ({:>4.1}%)",
                    col.missing_count,
                    col.missing_fraction * 100.0
                ),
                summary(col),
            )
        })
        .collect();
    for (name, ty, miss, sum) in &rows {
        name_w = name_w.max(measure(name));
        type_w = type_w.max(measure(ty));
        miss_w = miss_w.max(measure(miss));
        sum_w = sum_w.max(measure(sum));
    }
    let header_row = (
        "column".to_string(),
        "type".to_string(),
        "missing".to_string(),
        "summary".to_string(),
    );
    let divider = "-".repeat(name_w + type_w + miss_w + sum_w + 9);
    out.push_str(&c.dim.paint(&divider));
    out.push('\n');
    out.push_str(&paint_row(
        &header_row,
        name_w,
        type_w,
        miss_w,
        sum_w,
        &c.bold,
    ));
    out.push('\n');
    out.push_str(&c.dim.paint(&divider));
    out.push('\n');
    for (name, ty, miss, sum) in &rows {
        let ty_style = match ty.as_str() {
            "numeric" => &c.numeric,
            _ => &c.categorical,
        };
        out.push_str(&paint_row3(
            name, name_w, ty, type_w, ty_style, miss, miss_w,
        ));
        out.push_str("  ");
        out.push_str(sum);
        out.push('\n');
    }
    out.push_str(&c.dim.paint(&divider));
    out.push('\n');

    // ── Findings ───────────────────────────────────────────────────────
    out.push('\n');
    out.push_str(&c.bold.paint("Findings"));
    out.push('\n');
    if findings.is_empty() {
        out.push_str(&c.dim.paint("  (none — no quality issues detected)"));
        out.push('\n');
    } else {
        for issue in findings {
            let style = match issue.severity {
                Severity::Critical => &c.critical,
                Severity::Warning => &c.warning,
                Severity::Info => &c.info,
            };
            let tag = match issue.severity {
                Severity::Critical => "CRIT",
                Severity::Warning => "WARN",
                Severity::Info => "INFO",
            };
            let scope = sanitize(issue.column.as_deref().unwrap_or("dataset"));
            out.push_str("  ");
            out.push_str(&style.paint(tag));
            out.push_str(&format!(" {} ", pad(&scope, 9)));
            out.push_str(&sanitize(&issue.message));
            out.push('\n');
        }
    }
    out
}

/// One padded table row: name, type, missing, summary.
fn paint_row(
    cells: &(String, String, String, String),
    name_w: usize,
    type_w: usize,
    miss_w: usize,
    sum_w: usize,
    style: &Style,
) -> String {
    let mut s = String::new();
    s.push_str(&style.paint(&pad(&cells.0, name_w)));
    s.push_str("  ");
    s.push_str(&style.paint(&pad(&cells.1, type_w)));
    s.push_str("  ");
    s.push_str(&style.paint(&pad(&cells.2, miss_w)));
    s.push_str("  ");
    s.push_str(&style.paint(&pad(&cells.3, sum_w)));
    s
}

/// Variant taking a pre-styled type cell.
fn paint_row3(
    name: &str,
    name_w: usize,
    ty: &str,
    type_w: usize,
    ty_style: &Style,
    miss: &str,
    miss_w: usize,
) -> String {
    let mut s = String::new();
    s.push_str(&pad(name, name_w));
    s.push_str("  ");
    s.push_str(&ty_style.paint(&pad(ty, type_w)));
    s.push_str("  ");
    s.push_str(&pad(miss, miss_w));
    s.push_str("  ");
    s
}

/// Strips terminal-control characters from user data: column names and
/// quality messages can originate in a CSV file, and a header like
/// `\x1b]0;pwned\x07` must not reach the terminal even when color is off.
fn sanitize(s: &str) -> String {
    if !s.chars().any(char::is_control) {
        return s.to_string();
    }
    s.chars()
        .map(|c| if c.is_control() { '\u{fffd}' } else { c })
        .collect()
}

/// Per-column summary cell.
fn summary(col: &ColumnProfile) -> String {
    match col.column_type {
        ColumnType::Numeric => match &col.numeric {
            Some(n) => format!(
                "μ={:.4} σ={:.4} [{:.4}, {:.4}]",
                n.mean, n.std, n.five.min, n.five.max
            ),
            None => "(all missing)".to_string(),
        },
        ColumnType::Categorical => match &col.categorical {
            Some(c) => format!(
                "{} unique, top {:?} {:.1}%",
                c.unique,
                c.top,
                c.imbalance_ratio * 100.0
            ),
            None => "(all missing)".to_string(),
        },
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// Display width: ANSI escapes and control characters count as zero.
fn measure(s: &str) -> usize {
    let mut width = 0usize;
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            // Skip to 'm' (SGR terminator).
            for c in chars.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            width += 1;
        }
    }
    width
}

/// Right-pads `s` with spaces to `width` display columns.
fn pad(s: &str, width: usize) -> String {
    let w = measure(s);
    if w >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - w))
    }
}

/// Human-readable byte count (matching the crate's examples).
fn human_bytes(n: usize) -> String {
    if n >= 1_048_576 {
        format!("{:.1} MiB", n as f64 / 1_048_576.0)
    } else if n >= 1024 {
        format!("{:.1} KiB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

/// Minimal ANSI SGR style helper.
struct Style {
    codes: &'static str,
}

impl Style {
    fn paint(&self, s: &str) -> String {
        if self.codes.is_empty() {
            s.to_string()
        } else {
            format!("\u{1b}[{}m{s}\u{1b}[0m", self.codes)
        }
    }
}

/// Color palette: `Palette::on()` emits ANSI codes, `Palette::off()` is a
/// no-op pass-through.
struct Palette {
    bold: Style,
    dim: Style,
    numeric: Style,
    categorical: Style,
    critical: Style,
    warning: Style,
    info: Style,
}

impl Palette {
    fn on() -> Self {
        Palette {
            bold: Style { codes: "1" },
            dim: Style { codes: "2" },
            numeric: Style { codes: "36" },
            categorical: Style { codes: "35" },
            critical: Style { codes: "1;31" },
            warning: Style { codes: "33" },
            info: Style { codes: "36" },
        }
    }

    fn off() -> Self {
        Palette {
            bold: Style { codes: "" },
            dim: Style { codes: "" },
            numeric: Style { codes: "" },
            categorical: Style { codes: "" },
            critical: Style { codes: "" },
            warning: Style { codes: "" },
            info: Style { codes: "" },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datarust::Matrix;

    fn sample_profile() -> DatasetProfile {
        let m =
            Matrix::from_rows(vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, f64::NAN]]).unwrap();
        DatasetProfile::from_matrix(&m, None).unwrap()
    }

    #[test]
    fn renders_header_and_columns() {
        let profile = sample_profile();
        let text = to_terminal(&profile, false);
        assert!(text.contains("Dataset profile"));
        assert!(text.contains("3 rows"));
        assert!(text.contains("x0"));
        assert!(text.contains("x1"));
        assert!(text.contains("missing"));
    }

    #[test]
    fn color_off_produces_no_escapes() {
        let profile = sample_profile();
        let text = to_terminal(&profile, false);
        assert!(!text.contains('\u{1b}'));
    }

    #[test]
    fn color_on_produces_ansi() {
        let profile = sample_profile();
        let text = to_terminal(&profile, true);
        assert!(text.contains('\u{1b}'));
    }

    #[test]
    fn measure_ignores_ansi() {
        assert_eq!(measure("abc"), 3);
        assert_eq!(measure("\u{1b}[1mabc\u{1b}[0m"), 3);
    }

    #[test]
    fn pad_pads_to_width() {
        assert_eq!(pad("a", 3), "a  ");
        assert_eq!(pad("abcd", 2), "abcd");
    }

    #[test]
    fn findings_rendered_with_tags() {
        let profile = sample_profile();
        let findings = run_checks(&profile, &Thresholds::default());
        let text = to_terminal_with(&profile, &findings, false);
        // Outliers / duplicates may or may not fire; the section always exists.
        assert!(text.contains("Findings"));
    }

    #[test]
    fn header_row_prints_summary_label() {
        let profile = sample_profile();
        let text = to_terminal(&profile, false);
        let header = text
            .lines()
            .find(|l| l.trim_start().starts_with("column"))
            .unwrap();
        assert!(header.contains("summary"), "header: {header}");
    }

    #[test]
    fn control_characters_in_column_names_are_neutralized() {
        // A CSV header can smuggle ANSI escapes into terminal output; they
        // must be replaced before rendering, even with color off.
        let m = Matrix::from_rows(vec![vec![1.0], vec![2.0]]).unwrap();
        let profile =
            DatasetProfile::from_matrix(&m, Some(&["\u{1b}]0;pwned\u{7}".to_string()])).unwrap();
        let text = to_terminal(&profile, false);
        assert!(!text.contains('\u{1b}'), "{text:?}");
        assert!(!text.contains('\u{7}'), "{text:?}");
    }
}
