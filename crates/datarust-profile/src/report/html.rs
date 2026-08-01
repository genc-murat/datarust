//! A dependency-free, self-contained HTML report.
//!
//! The output is a single HTML document with inline CSS — no external
//! assets, no JavaScript framework. It is intentionally minimal so it can be
//! embedded in Jupyter-style widgets, emailed, or served as a static file.
//!
//! Columns are rendered as a responsive card grid (one card per column) rather
//! than a single wide table: numeric cards carry the summary statistics, a
//! CSS-div mini histogram, and the IQR outlier count; categorical cards carry
//! the top value and a frequency bar chart. This avoids the cell-counting
//! fragility of a heterogeneous table and keeps each column's picture
//! self-contained.
//!
//! Unlike [`super::json`], the HTML renderer works without the `serde`
//! feature because it reads the profile fields directly.

use crate::profile::{
    CategoricalStats, ColumnProfile, CorrelationMatrix, DatasetProfile, Histogram, NumericStats,
    PointBiserialEntry, Relationships,
};
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
h3 { font-size: 0.95rem; margin: 1rem 0 .4rem; }
.summary { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }
.summary div { background: #fff; border: 1px solid #e3e3e3; border-radius: 6px;
               padding: .75rem 1rem; min-width: 9rem; }
.summary dt { font-size: .75rem; text-transform: uppercase; letter-spacing: .03em;
              color: #777; margin: 0; }
.summary dd { margin: .25rem 0 0; font-size: 1.15rem; font-weight: 600; }
.findings { list-style: none; padding: 0; }
.findings li { padding: .4rem .6rem; margin: .25rem 0; border-radius: 4px; border: 1px solid #eee; }
.findings .critical { border-left: 4px solid #c62828; background: #fdecea; }
.findings .warning  { border-left: 4px solid #ef6c00; background: #fff4e5; }
.findings .info     { border-left: 4px solid #1976d2; background: #e8f0fe; }

/* Relationships & Heatmaps */
.rel-container { display: flex; flex-direction: column; gap: 1.5rem; margin: 1rem 0; }
.heatmap-wrapper { overflow-x: auto; background: #fff; border: 1px solid #e3e3e3; border-radius: 6px; padding: 1rem; }
table.heatmap { border-collapse: collapse; font-size: .8rem; text-align: center; }
table.heatmap th, table.heatmap td { padding: .4rem .6rem; border: 1px solid #eee; }
table.heatmap th { font-weight: 600; background: #f7f7f7; color: #444; }
table.heatmap td { font-variant-numeric: tabular-nums; font-weight: 500; }
table.rel-table { border-collapse: collapse; font-size: .82rem; width: 100%; max-width: 36rem; }
table.rel-table th, table.rel-table td { padding: .4rem .6rem; border: 1px solid #eee; text-align: left; }
table.rel-table th { background: #f7f7f7; font-weight: 600; }

/* Per-column card grid */
.col-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(20rem, 1fr));
            gap: 1rem; margin-top: .5rem; }
.col-card { background: #fff; border: 1px solid #e3e3e3; border-radius: 6px;
            padding: .75rem 1rem; display: flex; flex-direction: column; gap: .5rem; }
.col-card > .head { display: flex; align-items: center; gap: .5rem;
                    justify-content: space-between; }
.col-card h3 { margin: 0; font-size: 1rem; font-weight: 600; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; }
.col-card .missing { font-size: .75rem; color: #777; white-space: nowrap; }
.col-card dl.stats { display: grid; grid-template-columns: auto 1fr auto 1fr;
                     gap: .15rem .6rem; margin: 0; font-size: .82rem; }
.col-card dl.stats dt { color: #777; }
.col-card dl.stats dd { margin: 0; font-variant-numeric: tabular-nums;
                        text-align: right; }
.col-card .full { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: .15rem 1rem;
                  font-size: .82rem; }
.col-card .full span b { color: #555; font-weight: 500; }

/* Histogram / frequency bar charts (pure CSS, no JS) */
.chart { display: flex; align-items: flex-end; gap: 2px; height: 3.5rem;
         padding-top: .25rem; }
.chart .bar { flex: 1 1 0; min-width: 4px; min-height: 2px; border-radius: 2px 2px 0 0;
              background: #1a56c4; }
.chart.cat .bar { background: #8a2be2; }
.chart .bar.zero { background: #eee; min-height: 1px; }
.chart-labels { display: flex; justify-content: space-between; font-size: .7rem;
                color: #999; margin-top: .1rem; }
.cat-list { list-style: none; padding: 0; margin: 0; font-size: .8rem; }
.cat-list li { display: flex; align-items: center; gap: .4rem; margin: .15rem 0; }
.cat-list .swatch { width: .55rem; height: .55rem; border-radius: 2px; background: #8a2be2;
                    flex: 0 0 auto; }
.cat-list .label { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis;
                   white-space: nowrap; }
.cat-list .count { font-variant-numeric: tabular-nums; color: #555; }

.badge { display: inline-block; padding: .05rem .4rem; border-radius: 3px;
         font-size: .7rem; text-transform: uppercase; letter-spacing: .03em; }
.badge.numeric { background: #e8f0fe; color: #1a56c4; }
.badge.categorical { background: #f6e8fe; color: #8a2be2; }
.note { font-size: .75rem; color: #999; }

@media (prefers-color-scheme: dark) {
  body { background: #161616; color: #e6e6e6; }
  .summary div, .col-card, .heatmap-wrapper { background: #1e1e1e; border-color: #333; }
  .col-card dl.stats dt, .col-card .missing, .cat-list .count, .chart-labels { color: #999; }
  .col-card .full span b, .cat-list .count { color: #aaa; }
  .chart .bar.zero { background: #333; }
  table.heatmap th, table.rel-table th { background: #2a2a2a; color: #ddd; border-color: #333; }
  table.heatmap td, table.rel-table td { border-color: #333; }
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
    let mut html = String::with_capacity(24 * 1024);
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

    // --- Relationships & Heatmaps ------------------------------------------
    if let Some(rels) = &profile.relationships {
        render_relationships(&mut html, rels);
    }

    // --- Per-column card grid ---------------------------------------------
    html.push_str("<h2>Per-column profile</h2>\n<div class=\"col-grid\">\n");
    for col in &profile.columns {
        render_col_card(&mut html, col);
    }
    html.push_str("</div>\n");
    html.push_str("</body>\n</html>\n");
    html
}

fn render_relationships(html: &mut String, rels: &Relationships) {
    if rels.pearson.is_none() && rels.cramers_v.is_none() && rels.point_biserial.is_empty() {
        return;
    }

    html.push_str("<h2>Relationships &amp; Interaction</h2>\n<div class=\"rel-container\">\n");

    if let Some(pearson) = &rels.pearson {
        html.push_str(
            "<div class=\"heatmap-wrapper\"><h3>Pearson correlation matrix (numeric)</h3>\n",
        );
        render_matrix_heatmap(html, pearson, false);
        html.push_str("</div>\n");
    }

    if let Some(cramers) = &rels.cramers_v {
        html.push_str("<div class=\"heatmap-wrapper\"><h3>Cramér's V matrix (categorical)</h3>\n");
        render_matrix_heatmap(html, cramers, true);
        html.push_str("</div>\n");
    }

    if !rels.point_biserial.is_empty() {
        html.push_str("<div class=\"heatmap-wrapper\"><h3>Point-biserial correlation (binary categorical &#8596; numeric)</h3>\n");
        render_point_biserial_table(html, &rels.point_biserial);
        html.push_str("</div>\n");
    }

    html.push_str("</div>\n");
}

fn render_matrix_heatmap(html: &mut String, matrix: &CorrelationMatrix, is_cramers: bool) {
    html.push_str("<table class=\"heatmap\">\n<thead><tr><th></th>");
    for label in &matrix.labels {
        html.push_str(&format!("<th>{}</th>", esc(label)));
    }
    html.push_str("</tr></thead>\n<tbody>\n");

    for (i, row_label) in matrix.labels.iter().enumerate() {
        html.push_str(&format!("<tr><th>{}</th>", esc(row_label)));
        for (j, &val) in matrix.values[i].iter().enumerate() {
            let bg_style = if is_cramers {
                // Cramér's V in [0.0, 1.0] -> White to Purple
                let opacity = val.clamp(0.0, 1.0);
                format!(
                    "background-color: rgba(138, 43, 226, {:.2}); color: {};",
                    opacity * 0.7,
                    if opacity > 0.5 { "#fff" } else { "inherit" }
                )
            } else {
                // Pearson in [-1.0, 1.0] -> Red (-1) .. White (0) .. Blue (+1)
                let r = val.clamp(-1.0, 1.0);
                if i == j {
                    "background-color: rgba(26, 86, 196, 0.2);".to_string()
                } else if r >= 0.0 {
                    format!(
                        "background-color: rgba(26, 86, 196, {:.2}); color: {};",
                        r * 0.7,
                        if r > 0.5 { "#fff" } else { "inherit" }
                    )
                } else {
                    let r_abs = r.abs();
                    format!(
                        "background-color: rgba(198, 40, 40, {:.2}); color: {};",
                        r_abs * 0.7,
                        if r_abs > 0.5 { "#fff" } else { "inherit" }
                    )
                }
            };
            html.push_str(&format!("<td style=\"{}\">{:.2}</td>", bg_style, val));
        }
        html.push_str("</tr>\n");
    }
    html.push_str("</tbody></table>\n");
}

fn render_point_biserial_table(html: &mut String, entries: &[PointBiserialEntry]) {
    html.push_str("<table class=\"rel-table\">\n<thead><tr><th>Categorical (binary)</th><th>Numeric</th><th>Correlation (r)</th></tr></thead>\n<tbody>\n");
    for entry in entries {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{:.3}</td></tr>\n",
            esc(&entry.categorical),
            esc(&entry.numeric),
            entry.correlation
        ));
    }
    html.push_str("</tbody></table>\n");
}

fn summary_card(html: &mut String, label: &str, value: String) {
    html.push_str(&format!(
        "<div><dt>{label}</dt><dd>{value}</dd></div>\n",
        label = esc(label),
        value = esc(&value)
    ));
}

/// Renders one column card, dispatching on type.
fn render_col_card(html: &mut String, col: &ColumnProfile) {
    html.push_str("<div class=\"col-card\">");
    // Head: name + badge on the left, missing on the right.
    let (badge, class) = match col.column_type {
        ColumnType::Numeric => ("numeric", "numeric"),
        ColumnType::Categorical => ("categorical", "categorical"),
    };
    let missing = format!(
        "{} missing ({:.1}%)",
        col.missing_count,
        col.missing_fraction * 100.0
    );
    html.push_str(&format!(
        "<div class=\"head\"><h3>{name}</h3><span class=\"badge {class}\">{badge}</span></div>",
        name = esc(&col.name),
        class = class,
        badge = badge
    ));
    html.push_str(&format!("<div class=\"missing\">{}</div>", esc(&missing)));

    match (&col.numeric, &col.categorical) {
        (Some(n), None) => render_numeric_card(html, n),
        (None, Some(c)) => render_categorical_card(html, c),
        _ => html.push_str("<p class=\"note\">No non-missing values.</p>"),
    }
    html.push_str("</div>\n");
}

/// Renders the body of a numeric column card.
fn render_numeric_card(html: &mut String, n: &NumericStats) {
    // Top stats grid: mean / std / skew / kurt.
    html.push_str("<dl class=\"stats\">\n");
    stat_pair(html, "mean", &fmt_num(n.mean));
    stat_pair(html, "std", &fmt_num(n.std));
    stat_pair(html, "skew", &fmt_num(n.skewness));
    stat_pair(html, "kurt", &fmt_num(n.kurtosis));
    html.push_str("</dl>\n");

    // Five-number summary on one line.
    html.push_str("<div class=\"full\">");
    html.push_str(&format!(
        "<span><b>min</b> {}</span><span><b>Q1</b> {}</span><span><b>med</b> {}</span><span><b>Q3</b> {}</span><span><b>max</b> {}</span>",
        esc(&fmt_num(n.five.min)),
        esc(&fmt_num(n.five.q1)),
        esc(&fmt_num(n.five.median)),
        esc(&fmt_num(n.five.q3)),
        esc(&fmt_num(n.five.max)),
    ));
    html.push_str("</div>\n");

    // Histogram.
    render_histogram(html, &n.histogram);

    // Outlier summary line.
    html.push_str(&format!(
        "<div class=\"note\">outliers: {} ({:.1}%) beyond IQR fences</div>",
        n.outlier_count,
        n.outlier_fraction * 100.0
    ));
}

/// Renders the body of a categorical column card.
fn render_categorical_card(html: &mut String, c: &CategoricalStats) {
    html.push_str("<dl class=\"stats\">\n");
    stat_pair(html, "unique", &c.unique.to_string());
    stat_pair(html, "imbalance", &format!("{:.3}", c.imbalance_ratio));
    html.push_str("</dl>\n");

    html.push_str(&format!(
        "<div class=\"full\"><span><b>top</b> {} ({}, {:.1}%)</span></div>",
        esc(&c.top),
        c.freq,
        c.imbalance_ratio * 100.0
    ));

    // Frequency bar chart + list of top values.
    if c.top_values.is_empty() {
        html.push_str("<p class=\"note\">No non-missing values.</p>");
    } else {
        render_cat_bars(html, &c.top_values);
    }
}

/// Emits a `<dt>label</dt><dd>value</dd>` pair inside a stats grid.
fn stat_pair(html: &mut String, label: &str, value: &str) {
    html.push_str(&format!(
        "<dt>{label}</dt><dd>{value}</dd>\n",
        label = esc(label),
        value = esc(value)
    ));
}

/// Renders a histogram as a flex row of CSS bars.
///
/// Bar heights are proportional to each bin count relative to the tallest
/// bin; empty bins get a faint baseline so the chart keeps its shape.
fn render_histogram(html: &mut String, hist: &Histogram) {
    if hist.counts.is_empty() {
        html.push_str("<p class=\"note\">histogram: empty</p>");
        return;
    }
    let max = hist.max_count().max(1) as f64;
    html.push_str("<div class=\"chart\">\n");
    for &c in &hist.counts {
        let pct = (c as f64 / max) * 100.0;
        let class = if c == 0 { "bar zero" } else { "bar" };
        // Height as an inline style percentage of the chart's height.
        html.push_str(&format!(
            "<div class=\"{class}\" style=\"height:{pct:.1}%\" title=\"{c}\"></div>\n",
            class = class,
            pct = pct.max(2.0),
            c = c,
        ));
    }
    html.push_str("</div>\n");
    // Edge labels (min / max) for context.
    if hist.edges.len() >= 2 {
        let lo = hist.edges.first().copied().unwrap_or(f64::NAN);
        let hi = hist.edges.last().copied().unwrap_or(f64::NAN);
        html.push_str(&format!(
            "<div class=\"chart-labels\"><span>{}</span><span>{}</span></div>\n",
            esc(&fmt_num(lo)),
            esc(&fmt_num(hi))
        ));
    }
}

/// Renders a categorical frequency bar chart plus a compact legend list.
fn render_cat_bars(html: &mut String, top_values: &[(String, usize)]) {
    let max = top_values.iter().map(|(_, c)| *c).max().unwrap_or(0).max(1) as f64;
    html.push_str("<div class=\"chart cat\">\n");
    for (_, c) in top_values {
        let pct = (*c as f64 / max) * 100.0;
        let class = if *c == 0 { "bar zero" } else { "bar" };
        html.push_str(&format!(
            "<div class=\"{class}\" style=\"height:{pct:.1}%\" title=\"{c}\"></div>\n",
            class = class,
            pct = pct.max(2.0),
            c = c,
        ));
    }
    html.push_str("</div>\n");

    html.push_str("<ul class=\"cat-list\">\n");
    for (label, c) in top_values {
        html.push_str(&format!(
            "<li><span class=\"swatch\"></span><span class=\"label\">{label}</span><span class=\"count\">{c}</span></li>\n",
            label = esc(label),
            c = c,
        ));
    }
    html.push_str("</ul>\n");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn esc_empty() {
        assert_eq!(esc(""), "");
    }

    #[test]
    fn esc_special_chars() {
        assert_eq!(esc("&"), "&amp;");
        assert_eq!(esc("<"), "&lt;");
        assert_eq!(esc(">"), "&gt;");
        assert_eq!(esc("\""), "&quot;");
        assert_eq!(esc("'"), "&#39;");
    }

    #[test]
    fn esc_mixed() {
        assert_eq!(esc("a<b&c\"d'e"), "a&lt;b&amp;c&quot;d&#39;e");
    }

    #[test]
    fn esc_plain_text() {
        assert_eq!(esc("hello world"), "hello world");
    }

    #[test]
    fn fmt_num_normal() {
        assert_eq!(fmt_num(1.2345), "1.2345");
    }

    #[test]
    fn fmt_num_nan() {
        assert_eq!(fmt_num(f64::NAN), "—");
    }

    #[test]
    fn fmt_num_infinity() {
        assert_eq!(fmt_num(f64::INFINITY), "—");
        assert_eq!(fmt_num(f64::NEG_INFINITY), "—");
    }

    #[test]
    fn fmt_num_zero() {
        assert_eq!(fmt_num(0.0), "0.0000");
    }

    #[test]
    fn fmt_num_large() {
        assert!(fmt_num(1e7).contains("e"));
    }

    #[test]
    fn fmt_num_small() {
        assert!(fmt_num(1e-5).contains("e"));
    }

    #[test]
    fn format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn format_bytes_bytes() {
        assert_eq!(format_bytes(512), "512 B");
    }

    #[test]
    fn format_bytes_kib() {
        assert_eq!(format_bytes(2048), "2.0 KiB");
    }

    #[test]
    fn format_bytes_mib() {
        let v = 1024 * 1024;
        assert_eq!(format_bytes(v), "1.0 MiB");
    }

    #[test]
    fn format_bytes_gib() {
        let v = 1024 * 1024 * 1024;
        assert_eq!(format_bytes(v), "1.0 GiB");
    }

    #[test]
    fn format_bytes_tib() {
        let v = 1024 * 1024 * 1024 * 1024;
        assert_eq!(format_bytes(v), "1.0 TiB");
    }
}
