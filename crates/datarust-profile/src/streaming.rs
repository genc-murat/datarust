//! Streaming single-pass CSV profiling (the `streaming` feature).
//!
//! Profiles a CSV file in one forward pass with bounded memory: per-column
//! Welford accumulators and raw central moments for exact mean/variance/
//! skewness/kurtosis, reservoir-sampled quantiles + histogram, online Pearson
//! covariances, hashed Cramér's V contingency tables, and point-biserial
//! group sums — reusing the crate's interpretation helpers (skewness,
//! kurtosis, histogram, IQR outlier fences) and quality checks unchanged.
//!
//! Fidelity contract:
//! - **Exact**: row/cell counts, missing counts, mean, variance/std, min/max,
//!   categorical frequencies/imbalance, and duplicate rows (when
//!   [`StreamingOptions::compute_duplicates`] is enabled).
//! - **Pairwise-complete Pearson**: correlations use every row where both
//!   columns hold finite values. The in-memory path instead propagates `NaN`
//!   (any missing cell on either side yields `NaN` for that pair), so values
//!   differ on data with missing numeric cells; the streaming definition is
//!   the more statistically useful of the two.
//! - **Approximate**: quantiles and histogram (reservoir sampling with
//!   capacity `StreamingOptions::reservoir_capacity` — *exact* whenever the
//!   table has at most that many rows), IQR outlier counts/fractions
//!   (derived from the same reservoir sample, so the count refers to the
//!   sampled values, not the full column), skewness/kurtosis (single-pass
//!   raw-moment accumulation; last-ULP differences vs the two-pass path are
//!   possible on large inputs), and `unique` on categorical columns whose
//!   distinct count exceeds `StreamingOptions::distinct_cap` (a documented
//!   lower bound).
//! - Column types are decided **online**: a numeric column permanently
//!   converts to categorical at its first non-parseable non-missing cell
//!   (parseable-but-non-finite cells such as `inf` stay numeric and are
//!   skipped like missing values, matching the in-memory path).
//!
//! Memory is bounded **per row**, not globally: the accumulators are O(rows
//! × columns × constant), but Cramér's V tables hold up to
//! `min(rows, distinct_cap²)` entries per column pair, opt-in duplicate
//! tracking stores one key per unique row, and a single physical line or
//! field is buffered at its natural size.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{ProfileError, Result};

/// Co-occurrence count table for one categorical column pair, keyed by the
/// pair's interned level codes.
type CramTable = HashMap<(u32, u32), usize>;
/// Map from column-index pair to its co-occurrence table.
type CramTables = HashMap<(usize, usize), CramTable>;
use crate::profile::column::{CategoricalStats, ColumnProfile, FiveNumber, NumericStats};
use crate::profile::dataset::DatasetProfile;
use crate::profile::relationships::{CorrelationMatrix, PointBiserialEntry, Relationships};
use crate::types::ColumnType;

/// Options for [`profile_csv_streaming_with`].
#[derive(Debug, Clone)]
pub struct StreamingOptions {
    /// Reservoir size for quantile/histogram sampling per column
    /// (default `10_000`). Tables with at most this many rows are profiled
    /// exactly.
    pub reservoir_capacity: usize,
    /// Maximum tracked distinct values per categorical column
    /// (default `100_000`). Beyond this, `unique` becomes a documented lower
    /// bound.
    pub distinct_cap: usize,
    /// Track exact duplicate rows (default `false`): costs one hash-set
    /// entry per row.
    pub compute_duplicates: bool,
    /// Seed for reservoir sampling (default `0x9E37_79B9_7F4A_7C15`).
    pub seed: u64,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        StreamingOptions {
            reservoir_capacity: 10_000,
            distinct_cap: 100_000,
            compute_duplicates: false,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Profiles a CSV file in a single streaming pass with
/// [`StreamingOptions::default`].
pub fn profile_csv_streaming(path: &Path) -> Result<DatasetProfile> {
    profile_csv_streaming_with(path, &StreamingOptions::default())
}

/// Profiles a CSV file in a single streaming pass with explicit options.
pub fn profile_csv_streaming_with(path: &Path, opts: &StreamingOptions) -> Result<DatasetProfile> {
    let file = std::fs::File::open(path)?;
    stream_from_reader(BufReader::new(file), opts)
}

/// Core streaming implementation over any [`BufRead`].
pub(crate) fn stream_from_reader<R: BufRead>(
    mut reader: R,
    opts: &StreamingOptions,
) -> Result<DatasetProfile> {
    let read_opts = crate::csv::CsvReadOptions::default();
    let mut parser = crate::csv::RecordParser::new(&read_opts);
    let mut records: Vec<(Vec<String>, usize)> = Vec::new();

    // Numeric per-column accumulators.
    let mut welford: Vec<Welford> = Vec::new();
    // Categorical per-column counts (columns flip numeric → categorical
    // permanently at their first non-parseable cell).
    let mut cat_counts: Vec<Option<CatCounts>> = Vec::new();
    let mut numeric_active: Vec<bool> = Vec::new();
    // Pearson cross-moment accumulators keyed by column pair.
    let mut cross: HashMap<(usize, usize), CoMoment> = HashMap::new();
    // Cramér's V co-occurrence tables keyed by column pair.
    let mut cramers: HashMap<(usize, usize), HashMap<(u32, u32), usize>> = HashMap::new();
    let mut cat_levels: Vec<LevelTable> = Vec::new();
    // Point-biserial binary group sums keyed by (cat, num) pairs.
    let mut pb: HashMap<(usize, usize), PbPair> = HashMap::new();
    let mut dup = DupTracker::new(opts.compute_duplicates);
    // Cells each column consumed numerically before a mid-stream flip; they
    // count as present in the final categorical profile (no second pass).
    let mut numeric_seen: Vec<usize> = Vec::new();

    let mut n_rows: usize = 0;
    let mut cols: Option<usize> = None;
    let mut header: Option<Vec<String>> = None;

    let observe_row = |row: &[String],
                       line: usize,
                       n_rows: &mut usize,
                       cols: &Option<usize>,
                       welford: &mut Vec<Welford>,
                       cat_counts: &mut Vec<Option<CatCounts>>,
                       numeric_active: &mut Vec<bool>,
                       cross: &mut HashMap<(usize, usize), CoMoment>,
                       cramers: &mut CramTables,
                       cat_levels: &mut Vec<LevelTable>,
                       pb: &mut HashMap<(usize, usize), PbPair>,
                       numeric_seen: &mut Vec<usize>,
                       dup: &mut DupTracker,
                       opts: &StreamingOptions|
     -> Result<()> {
        let p = cols.expect("column count is set before data rows");
        if row.len() != p {
            return Err(ProfileError::InvalidInput(format!(
                "CSV data row at line {line} has {} fields, expected {}",
                row.len(),
                p
            )));
        }
        *n_rows += 1;
        dup.observe(row);

        // Reconcile lazily-grown accumulator vectors.
        if welford.is_empty() {
            *welford = (0..p).map(|_| Welford::new(opts.seed)).collect::<Vec<_>>();
            *cat_counts = (0..p).map(|_| None).collect();
            *numeric_active = vec![true; p];
            *numeric_seen = vec![0; p];
            *cat_levels = (0..p)
                .map(|_| LevelTable {
                    map: HashMap::new(),
                    next: 0,
                    capped: false,
                })
                .collect();
        }

        // Per-cell updates; collect (col, value) pairs for pairwise updates.
        let mut row_num_values: Vec<(usize, f64)> = Vec::new();
        let mut cat_seen: Vec<usize> = Vec::new();
        for j in 0..p {
            let cell = row[j].as_str();
            if crate::infer::is_missing(cell) {
                continue;
            }
            if numeric_active[j] {
                match cell.trim().parse::<f64>() {
                    Ok(v) if v.is_finite() => {
                        numeric_seen[j] += 1;
                        row_num_values.push((j, v));
                        continue;
                    }
                    // Parseable but non-finite (`inf`, `NaN`, `1e999`): the
                    // in-memory path keeps the column numeric, counts the
                    // cell as present, and excludes the value from the
                    // statistics; do the same instead of flipping.
                    Ok(_) => welford[j].skipped += 1,
                    _ => {
                        // Flip to categorical; the numeric moments seen so far
                        // become unusable (no second pass), so the column
                        // degrades to a categorical profile.
                        numeric_active[j] = false;
                        cat_counts[j] = Some(CatCounts::default());
                    }
                }
            }
            // Categorical path.
            let counts = cat_counts[j].get_or_insert_with(CatCounts::default);
            let levels = &mut cat_levels[j];
            let code = levels.code(cell.trim(), opts.distinct_cap);
            counts.present += 1;
            if let Some(code) = code {
                *counts.map.entry(code).or_insert(0) += 1;
            }
            cat_seen.push(j);
        }

        // ── Numeric updates ────────────────────────────────────────────
        for &(j, v) in &row_num_values {
            welford[j].push(v, opts.reservoir_capacity);
        }
        for a in 0..row_num_values.len() {
            let (i, xi) = row_num_values[a];
            for &(j, yj) in row_num_values[(a + 1)..].iter() {
                let key = if i < j { (i, j) } else { (j, i) };
                let (x, y) = if i < j { (xi, yj) } else { (yj, xi) };
                cross.entry(key).or_insert_with(CoMoment::new).update(x, y);
            }
        }

        // ── Categorical pair updates ───────────────────────────────────
        for a in 0..cat_seen.len() {
            let i = cat_seen[a];
            let ci = cat_levels[i]
                .map
                .get(row[i].trim())
                .copied()
                .unwrap_or(u32::MAX);
            if ci == u32::MAX {
                continue;
            }
            for &j in cat_seen[(a + 1)..].iter() {
                let cj = cat_levels[j]
                    .map
                    .get(row[j].trim())
                    .copied()
                    .unwrap_or(u32::MAX);
                if cj == u32::MAX {
                    continue;
                }
                *cramers
                    .entry((i.min(j), i.max(j)))
                    .or_default()
                    .entry((ci.min(cj), ci.max(cj)))
                    .or_insert(0) += 1;
            }
        }

        // ── Point-biserial updates ───────────────────────────────────
        for &ci in &cat_seen {
            let level = cat_levels[ci]
                .map
                .get(row[ci].trim())
                .copied()
                .unwrap_or(u32::MAX);
            if level == u32::MAX {
                continue;
            }
            for &(nj, v) in &row_num_values {
                let pair = pb.entry((ci, nj)).or_insert_with(PbPair::new);
                pair.observe(level, v);
            }
        }
        Ok(())
    };

    // ── Stream all records through the shared CSV parser ───────────────
    // Header/column handling is shared by the streamed and flushed tails.
    let mut ingest = |records: &mut Vec<(Vec<String>, usize)>| -> Result<()> {
        for (rec, line) in records.drain(..) {
            if cols.is_none() {
                // First physical record.
                if read_opts.has_header {
                    header = Some(rec);
                    cols = Some(header.as_ref().map_or(0, |h| h.len()));
                    continue;
                }
                cols = Some(rec.len());
            }
            observe_row(
                &rec,
                line,
                &mut n_rows,
                &cols,
                &mut welford,
                &mut cat_counts,
                &mut numeric_active,
                &mut cross,
                &mut cramers,
                &mut cat_levels,
                &mut pb,
                &mut numeric_seen,
                &mut dup,
                opts,
            )?;
        }
        Ok(())
    };

    // `read_line` keeps each physical line's terminator — including a `\r`
    // inside a quoted multi-line field — intact, matching `read_csv_reader`;
    // `BufRead::lines` would strip a trailing `\r` from every physical line.
    let mut buf = String::new();
    loop {
        buf.clear();
        match reader.read_line(&mut buf) {
            Ok(0) => break,
            Ok(_) => {
                parser.feed(&buf, &mut records)?;
                ingest(&mut records)?;
            }
            Err(e) => return Err(ProfileError::from(e)),
        }
    }
    parser.finish(&mut records)?;
    ingest(&mut records)?;

    let p = cols.ok_or_else(|| ProfileError::EmptyInput("CSV has no records".into()))?;
    if p == 0 {
        return Err(ProfileError::EmptyInput("CSV has no columns".into()));
    }
    if n_rows == 0 {
        return Err(ProfileError::EmptyInput("CSV has no data rows".into()));
    }

    // Strip the UTF-8 BOM from the first header cell, matching `read_csv_reader`.
    if let Some(h) = header.as_mut() {
        if let Some(cell) = h.first_mut() {
            if let Some(stripped) = cell.strip_prefix('\u{feff}') {
                *cell = stripped.to_string();
            }
        }
    }

    let names: Vec<String> = match header {
        Some(h) if h.len() == p => h,
        _ => (0..p).map(|j| format!("x{j}")).collect(),
    };

    // ── Finalize columns ───────────────────────────────────────────────
    let mut columns: Vec<ColumnProfile> = Vec::with_capacity(p);
    let mut memory_bytes = 0usize;
    let mut numeric_order: Vec<usize> = Vec::new();
    let mut cat_order: Vec<usize> = Vec::new();

    for j in 0..p {
        let name = &names[j];
        let present = if numeric_active[j] {
            // Non-finite parses are present but excluded from the moments.
            welford[j].n + welford[j].skipped
        } else {
            // Cells consumed numerically before a mid-stream flip still
            // count as present (they were observed, just not categorically).
            cat_counts[j].as_ref().map_or(0, |c| c.present) + numeric_seen[j]
        };
        let missing = n_rows - present;
        let missing_fraction = missing as f64 / n_rows as f64;
        if numeric_active[j] && present > 0 {
            numeric_order.push(j);
            let stats5 = welford[j].finalize();
            columns.push(ColumnProfile {
                name: name.clone(),
                column_type: ColumnType::Numeric,
                count: n_rows,
                missing_count: missing,
                missing_fraction,
                numeric: Some(stats5),
                categorical: None,
            });
            memory_bytes += n_rows * 8;
        } else {
            cat_order.push(j);
            let counts = cat_counts[j].get_or_insert_with(CatCounts::default);
            // An all-missing column mirrors the in-memory path: no
            // `CategoricalStats` at all (`compute_categorical` returns `None`).
            if counts.present == 0 && numeric_seen[j] == 0 {
                columns.push(ColumnProfile {
                    name: name.clone(),
                    column_type: ColumnType::Categorical,
                    count: n_rows,
                    missing_count: missing,
                    missing_fraction,
                    numeric: None,
                    categorical: None,
                });
                memory_bytes += n_rows * 24;
                continue;
            }
            let mut entries: Vec<(u32, usize)> = counts.map.iter().map(|(&k, &c)| (k, c)).collect();
            // Deterministic total order: count desc, then level code.
            entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let decode = |code: u32| -> String {
                cat_levels[j]
                    .map
                    .iter()
                    .find(|(_, &c)| c == code)
                    .map(|(k, _)| k.clone())
                    .unwrap_or_default()
            };
            let top_values: Vec<(String, usize)> = entries
                .iter()
                .take(8)
                .map(|&(code, c)| (decode(code), c))
                .collect();
            let (top, freq) = match top_values.first() {
                Some((t, f)) => (t.clone(), *f),
                None => (String::new(), 0),
            };
            columns.push(ColumnProfile {
                name: name.clone(),
                column_type: ColumnType::Categorical,
                count: n_rows,
                missing_count: missing,
                missing_fraction,
                numeric: None,
                categorical: Some(CategoricalStats {
                    unique: counts.map.len(),
                    top,
                    freq,
                    imbalance_ratio: if counts.present == 0 {
                        0.0
                    } else {
                        freq as f64 / counts.present as f64
                    },
                    top_values,
                }),
            });
            memory_bytes += n_rows * 24;
        }
    }

    // ── Relationships ──────────────────────────────────────────────────
    let relationships = finalize_relationships(
        &names,
        &numeric_order,
        &cat_order,
        &cross,
        &cramers,
        &pb,
        &cat_levels,
    );

    let duplicate_rows = dup.count;
    Ok(DatasetProfile {
        n_rows,
        n_columns: p,
        memory_bytes,
        duplicate_rows,
        duplicate_fraction: duplicate_rows as f64 / n_rows as f64,
        target_column: None,
        columns,
        relationships,
    })
}

/// Running mean/variance and raw central moments for one numeric column.
struct Welford {
    n: usize,
    /// Cells that parsed as numeric but were non-finite (`inf`, `NaN`) —
    /// present in the column's row count, excluded from the moments.
    skipped: usize,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    min: f64,
    max: f64,
    reservoir: Vec<f64>,
    rng: SmallRng,
}

impl Welford {
    fn new(seed: u64) -> Self {
        Welford {
            n: 0,
            skipped: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            reservoir: Vec::new(),
            rng: SmallRng::new(seed),
        }
    }

    fn push(&mut self, v: f64, capacity: usize) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = v - self.mean;
        let mean_new = self.mean + delta / n_f;
        let d2 = v - mean_new;
        let m2_new = self.m2 + delta * d2;
        let delta_n = delta / n_f;
        // Incremental central moments (Pébay's single-value update formulas).
        // `self.m2`/`self.m3`/`self.m4` accumulate *sums* of central moments
        // (divided by `n` in `finalize`); the recurrences below use their
        // pre-update values, which keeps them exact.
        let term1 = delta * delta_n * (n_f - 1.0);
        let m3_new = self.m3 + term1 * delta_n * (n_f - 2.0) - 3.0 * delta_n * self.m2;
        let m4_new = self.m4
            + term1 * delta_n * delta_n * (n_f * n_f - 3.0 * n_f + 3.0)
            + 6.0 * delta_n * delta_n * self.m2
            - 4.0 * delta_n * self.m3;
        self.m4 = m4_new;
        self.m3 = m3_new;
        self.m2 = m2_new;
        self.mean = mean_new;
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
        // Reservoir sampling (Algorithm R): the value at 0-based global
        // index i (== self.n - 1 here) enters the reservoir with
        // probability capacity / (i + 1), so every value is equally
        // likely to be in the final sample.
        if self.reservoir.len() < capacity {
            self.reservoir.push(v);
        } else if capacity > 0 {
            let j = self.rng.next_below(self.n);
            if j < capacity {
                self.reservoir[j] = v;
            }
        }
    }

    /// Builds the final [`NumericStats`] from the accumulated moments and
    /// the reservoir sample.
    fn finalize(&self) -> NumericStats {
        let n_f = self.n as f64;
        // `datarust::stats::variance` (the in-memory path) yields NaN when
        // the sample is smaller than ddof + 1; single-value and empty
        // columns match that instead of reporting a bogus 0.0.
        let var = if self.n >= 2 {
            self.m2 / (n_f - 1.0)
        } else {
            f64::NAN
        };
        let std = var.sqrt();
        let mut sorted = self.reservoir.clone();
        sorted.sort_by(f64::total_cmp);
        let q = |p: f64| -> f64 {
            if sorted.is_empty() {
                return f64::NAN;
            }
            let pos = p * (sorted.len() - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            if lo == hi {
                sorted[lo]
            } else {
                let frac = pos - lo as f64;
                sorted[lo] * (1.0 - frac) + sorted[hi] * frac
            }
        };
        let five = FiveNumber {
            min: self.min,
            q1: q(0.25),
            median: q(0.5),
            q3: q(0.75),
            max: self.max,
        };
        let skew = if self.n >= 3 && std > 0.0 {
            (self.m3 / n_f) / std.powi(3)
        } else {
            0.0
        };
        let kurt = if self.n >= 4 && std > 0.0 {
            (self.m4 / n_f) / std.powi(4) - 3.0
        } else {
            0.0
        };
        let histogram = crate::profile::distribution::histogram(&sorted, five.min, five.max);
        let (outlier_count, outlier_fraction) =
            crate::profile::distribution::outlier_count(&sorted, five.q1, five.q3);
        NumericStats {
            mean: self.mean,
            std,
            five,
            skewness: skew,
            kurtosis: kurt,
            histogram,
            outlier_count,
            outlier_fraction,
        }
    }
}

/// Online co-moment accumulator for a numeric column pair (Welford-style
/// covariance; exact Pearson up to floating-point accumulation order).
///
/// All accumulators — including the marginal sums of squared deviations —
/// run over the *pairwise-complete* rows only, so [`CoMoment::correlation`]
/// is the pairwise-complete Pearson correlation.
struct CoMoment {
    n: usize,
    mean_x: f64,
    mean_y: f64,
    c: f64,
    ss_x: f64,
    ss_y: f64,
}

impl CoMoment {
    fn new() -> Self {
        CoMoment {
            n: 0,
            mean_x: 0.0,
            mean_y: 0.0,
            c: 0.0,
            ss_x: 0.0,
            ss_y: 0.0,
        }
    }

    fn update(&mut self, x: f64, y: f64) {
        self.n += 1;
        let n = self.n as f64;
        let dx = x - self.mean_x;
        let dy = y - self.mean_y;
        self.mean_x += dx / n;
        self.mean_y += dy / n;
        self.c += dx * (y - self.mean_y);
        self.ss_x += dx * (x - self.mean_x);
        self.ss_y += dy * (y - self.mean_y);
    }

    /// Sample Pearson correlation over the pairwise-complete rows. The
    /// `n − 1` factors cancel, so `r = c / sqrt(ss_x · ss_y)` exactly.
    fn correlation(&self) -> Option<f64> {
        if self.n < 2 || self.ss_x <= 0.0 || self.ss_y <= 0.0 {
            return None;
        }
        let r = self.c / (self.ss_x * self.ss_y).sqrt();
        if r.is_nan() {
            None
        } else {
            // Guard against floating-point overshoot of ±1 by a few ulps.
            Some(r.clamp(-1.0, 1.0))
        }
    }
}

/// Interned string levels for one categorical column.
struct LevelTable {
    map: HashMap<String, u32>,
    next: u32,
    capped: bool,
}

impl LevelTable {
    /// Returns the level code for `value`, assigning the next code. Returns
    /// `None` once the table exceeds `cap` distinct values (the cell keeps
    /// counting towards `present` but cannot join pairwise tables).
    fn code(&mut self, value: &str, cap: usize) -> Option<u32> {
        if let Some(&c) = self.map.get(value) {
            return Some(c);
        }
        if self.next as usize >= cap {
            self.capped = true;
            return None;
        }
        let code = self.next;
        self.next += 1;
        self.map.insert(value.to_string(), code);
        Some(code)
    }
}

/// Per-column categorical counters (present cells + per-level counts).
#[derive(Default)]
struct CatCounts {
    present: usize,
    map: HashMap<u32, usize>,
}

/// Point-biserial accumulator for one (categorical, numeric) pair.
///
/// Group 0 is always the categorical column's first-seen level (interned
/// code `0`), mirroring the in-memory implementation's `val_0` encoding;
/// group 1 is the first other level observed. Observing a third level (the
/// column is not binary) drops the pair.
struct PbPair {
    second: Option<u32>,
    dropped: bool,
    n0: usize,
    sum0: f64,
    n1: usize,
    sum1: f64,
    sum_sq: f64,
}

impl PbPair {
    fn new() -> Self {
        PbPair {
            second: None,
            dropped: false,
            n0: 0,
            sum0: 0.0,
            n1: 0,
            sum1: 0.0,
            sum_sq: 0.0,
        }
    }

    fn observe(&mut self, level: u32, value: f64) {
        if self.dropped {
            return;
        }
        if level == 0 {
            self.n0 += 1;
            self.sum0 += value;
            self.sum_sq += value * value;
            return;
        }
        match self.second {
            Some(s) if s == level => {
                self.n1 += 1;
                self.sum1 += value;
                self.sum_sq += value * value;
            }
            None => {
                self.second = Some(level);
                self.n1 += 1;
                self.sum1 += value;
                self.sum_sq += value * value;
            }
            _ => self.dropped = true,
        }
    }

    /// Final point-biserial correlation, mirroring the two-pass
    /// implementation's computational identity.
    fn correlation(&self) -> Option<f64> {
        if self.dropped || self.n0 == 0 || self.n1 == 0 {
            return None;
        }
        let total_n = self.n0 + self.n1;
        if total_n < 3 {
            return None;
        }
        let m0 = self.sum0 / self.n0 as f64;
        let m1 = self.sum1 / self.n1 as f64;
        let total_sum = self.sum0 + self.sum1;
        let ss = self.sum_sq - (total_sum * total_sum) / total_n as f64;
        let s_y = (ss / (total_n - 1) as f64).sqrt();
        let r = if s_y == 0.0 {
            0.0
        } else {
            ((m1 - m0) / s_y)
                * ((self.n0 * self.n1) as f64 / (total_n * (total_n - 1)) as f64).sqrt()
        };
        if r.is_nan() {
            None
        } else {
            Some(r.clamp(-1.0, 1.0))
        }
    }
}

/// Duplicate-row tracker keyed on raw cells.
struct DupTracker {
    enabled: bool,
    seen: Option<HashSet<Vec<String>>>,
    count: usize,
}

impl DupTracker {
    fn new(enabled: bool) -> Self {
        DupTracker {
            enabled,
            seen: if enabled { Some(HashSet::new()) } else { None },
            count: 0,
        }
    }
    /// Keys rows by their raw cells, exactly like the in-memory CSV path's
    /// `count_duplicate_str`: no trimming, no numeric canonicalization, no
    /// missing-marker unification — `1` and `1.0` are different rows.
    fn observe(&mut self, row: &[String]) {
        if !self.enabled {
            return;
        }
        let seen = self.seen.as_mut().expect("enabled tracker has a set");
        if !seen.insert(row.to_vec()) {
            self.count += 1;
        }
    }
}

/// Minimal deterministic PRNG for reservoir sampling (xorshift64*).
struct SmallRng {
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        SmallRng {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_below(&mut self, hi: usize) -> usize {
        if hi == 0 {
            return 0;
        }
        ((self.next_u64() >> 11) as usize) % hi
    }
}

/// Builds the [`Relationships`] block from the online accumulators.
#[allow(clippy::too_many_arguments)]
fn finalize_relationships(
    names: &[String],
    numeric_order: &[usize],
    cat_order: &[usize],
    cross: &HashMap<(usize, usize), CoMoment>,
    cramers: &CramTables,
    pb: &HashMap<(usize, usize), PbPair>,
    cat_levels: &[LevelTable],
) -> Option<Relationships> {
    // ── Pearson ────────────────────────────────────────────────────────
    let pearson = if numeric_order.len() >= 2 {
        let pos: HashMap<usize, usize> = numeric_order
            .iter()
            .enumerate()
            .map(|(pos, &j)| (j, pos))
            .collect();
        let p = numeric_order.len();
        let labels: Vec<String> = numeric_order.iter().map(|&j| names[j].clone()).collect();
        let mut values = vec![vec![0.0; p]; p];
        for (row_idx, diag) in values.iter_mut().enumerate() {
            diag[row_idx] = 1.0;
        }
        for (&(i, j), com) in cross {
            let (pi, pj) = match (pos.get(&i), pos.get(&j)) {
                (Some(&a), Some(&b)) => (a, b),
                _ => continue, // one side flipped to categorical mid-stream
            };
            if let Some(r) = com.correlation() {
                values[pi][pj] = r;
                values[pj][pi] = r;
            }
        }
        Some(CorrelationMatrix { labels, values })
    } else {
        None
    };

    // ── Cramér's V ─────────────────────────────────────────────────────
    let cramers_v = if cat_order.len() >= 2 && !cramers.is_empty() {
        let pos: HashMap<usize, usize> = cat_order
            .iter()
            .enumerate()
            .map(|(pos, &j)| (j, pos))
            .collect();
        let q = cat_order.len();
        let labels: Vec<String> = cat_order.iter().map(|&j| names[j].clone()).collect();
        let mut values = vec![vec![0.0; q]; q];
        for (row_idx, diag) in values.iter_mut().enumerate() {
            diag[row_idx] = 1.0;
        }
        for (&(i, j), table) in cramers {
            let (pi, pj) = match (pos.get(&i), pos.get(&j)) {
                (Some(&a), Some(&b)) => (a, b),
                _ => continue,
            };
            let v = cramers_v_from_table(table);
            values[pi][pj] = v;
            values[pj][pi] = v;
        }
        Some(CorrelationMatrix { labels, values })
    } else {
        None
    };

    // ── Point-biserial ─────────────────────────────────────────────────
    // Mirror the in-memory rule exactly: a pair is only reported when the
    // categorical column has *exactly* two distinct levels column-wide (a
    // one-level column is also skipped) and the numeric side is still
    // numeric at the end of the stream (a mid-stream flip leaves stale
    // partial sums behind). `LevelTable::next` is the number of distinct
    // interned levels for the column.
    let point_biserial: Vec<PointBiserialEntry> = pb
        .iter()
        .filter(|(&(ci, _), _)| cat_levels[ci].next == 2)
        .filter(|&(&(ci, nj), _)| cat_order.contains(&ci) && numeric_order.contains(&nj))
        .filter_map(|(&(ci, nj), pair)| {
            pair.correlation().map(|r| PointBiserialEntry {
                categorical: names[ci].clone(),
                numeric: names[nj].clone(),
                correlation: r,
            })
        })
        .collect();

    if pearson.is_none() && cramers_v.is_none() && point_biserial.is_empty() {
        None
    } else {
        Some(Relationships {
            pearson,
            cramers_v,
            point_biserial,
        })
    }
}

/// Cramér's V from a co-occurrence count table keyed by level codes.
fn cramers_v_from_table(table: &HashMap<(u32, u32), usize>) -> f64 {
    let mut row_tot: HashMap<u32, usize> = HashMap::new();
    let mut col_tot: HashMap<u32, usize> = HashMap::new();
    let mut n = 0usize;
    for (&(r, c), &obs) in table {
        *row_tot.entry(r).or_insert(0) += obs;
        *col_tot.entry(c).or_insert(0) += obs;
        n += obs;
    }
    if n == 0 {
        return 0.0;
    }
    let r_levels = row_tot.values().copied().filter(|&t| t > 0).count();
    let c_levels = col_tot.values().copied().filter(|&t| t > 0).count();
    if r_levels <= 1 || c_levels <= 1 {
        return 0.0;
    }
    let n_f = n as f64;
    let mut chi2 = 0.0;
    for (&(r, c), &obs) in table {
        let expected = row_tot[&r] as f64 * col_tot[&c] as f64 / n_f;
        if expected > 0.0 {
            let o = obs as f64;
            chi2 += o * o / expected;
        }
    }
    chi2 -= n_f;
    let min_dim = (r_levels - 1).min(c_levels - 1) as f64;
    if min_dim == 0.0 {
        return 0.0;
    }
    let v = (chi2 / (n_f * min_dim)).sqrt();
    if v.is_nan() {
        0.0
    } else {
        v.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_csv(content: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.csv");
        std::fs::write(&path, content).unwrap();
        (dir, path)
    }

    #[test]
    fn small_table_matches_in_memory_profile() {
        // n_rows <= reservoir_capacity ⇒ exact quantiles; the streaming
        // profile must match profile_str_matrix closely on all exact stats
        // and within tolerance on moments.
        let (dir, path) =
            write_csv("x,y,c\n1,2,a\n2,4,b\n3,6,a\n4,8,b\n5,10,a\n6,12,b\n7,14,a\n8,16,b\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        let owned = DatasetProfile::from_csv(&path).unwrap();

        assert_eq!(streamed.n_rows, owned.n_rows);
        assert_eq!(streamed.n_columns, owned.n_columns);
        assert_eq!(streamed.columns.len(), owned.columns.len());
        let sx = streamed.columns[0].numeric.as_ref().unwrap();
        let ox = owned.columns[0].numeric.as_ref().unwrap();
        assert_eq!(sx.mean.to_bits(), ox.mean.to_bits());
        assert_eq!(sx.five.min.to_bits(), ox.five.min.to_bits());
        assert_eq!(sx.five.max.to_bits(), ox.five.max.to_bits());
        assert!((sx.std - ox.std).abs() < 1e-12);
        assert!((sx.five.q1 - ox.five.q1).abs() < 1e-12);
        assert!((sx.five.median - ox.five.median).abs() < 1e-12);
        // Pearson x↔y is exactly 1.0 in both paths.
        let pr = streamed
            .relationships
            .as_ref()
            .unwrap()
            .pearson
            .as_ref()
            .unwrap();
        assert!((pr.values[0][1] - 1.0).abs() < 1e-9);
        drop(dir);
    }

    #[test]
    fn categorical_stats_match() {
        let (dir, path) = write_csv("c,n\na,1\nb,2\na,3\nb,4\na,5\na,6\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        let owned = DatasetProfile::from_csv(&path).unwrap();
        let sc = streamed.columns[0].categorical.as_ref().unwrap();
        let oc = owned.columns[0].categorical.as_ref().unwrap();
        assert_eq!(sc.unique, oc.unique);
        assert_eq!(sc.freq, oc.freq);
        assert_eq!(sc.top, oc.top);
        assert!((sc.imbalance_ratio - oc.imbalance_ratio).abs() < 1e-12);
        drop(dir);
    }

    #[test]
    fn numeric_flip_mid_stream() {
        // A non-parseable cell in a numeric column flips it to categorical.
        // Streaming never replays earlier rows: cells observed while the
        // column was numeric are not reclassified, so only the flip point
        // onward is counted categorically (documented fidelity contract).
        let (dir, path) = write_csv("x\n1\n2\noops\n3\n4\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        assert_eq!(streamed.columns[0].column_type, ColumnType::Categorical);
        let c = streamed.columns[0].categorical.as_ref().unwrap();
        // "oops" lands as the single categorical level; 1/2 arrived numeric
        // and 3/4 arrive after the flip as categorical values.
        assert_eq!(c.unique, 3);
        assert!(c.top_values.iter().any(|(v, _)| v == "oops"));
        // Rows before the flip are not replayed; those cells neither count
        // as categorical nor re-appear in any numeric stat (a one-pass
        // trade-off, documented in the module docs).
        assert_eq!(streamed.columns[0].missing_count, 0);
        assert_eq!(streamed.columns[0].count, streamed.n_rows);
        drop(dir);
    }

    #[test]
    fn missing_cells_counted() {
        let (dir, path) = write_csv("x,y\n1,2\nNA,4\n3,\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        assert_eq!(streamed.columns[0].missing_count, 1);
        assert_eq!(streamed.columns[1].missing_count, 1);
        drop(dir);
    }

    #[test]
    fn duplicates_counted_when_enabled() {
        let (dir, path) = write_csv("a,b\n1,2\n1,2\n3,4\n");
        let opts = StreamingOptions {
            compute_duplicates: true,
            ..StreamingOptions::default()
        };
        let streamed = profile_csv_streaming_with(&path, &opts).unwrap();
        assert_eq!(streamed.duplicate_rows, 1);
        let streamed_off = profile_csv_streaming(&path).unwrap();
        assert_eq!(streamed_off.duplicate_rows, 0);
        drop(dir);
    }

    #[test]
    fn distinct_cap_produces_lower_bound() {
        let (dir, path) = write_csv("c\na\nb\nc\nd\n");
        let opts = StreamingOptions {
            distinct_cap: 2,
            ..StreamingOptions::default()
        };
        let streamed = profile_csv_streaming_with(&path, &opts).unwrap();
        let c = streamed.columns[0].categorical.as_ref().unwrap();
        // Only two levels are tracked (a, b in first-seen order); the rest
        // still count towards present.
        assert_eq!(c.unique, 2);
        drop(dir);
    }

    #[test]
    fn empty_file_is_error() {
        let (dir, path) = write_csv("");
        let err = profile_csv_streaming(&path).unwrap_err();
        assert!(matches!(err, ProfileError::EmptyInput(_)));
        drop(dir);
    }

    #[test]
    fn header_only_is_error() {
        let (dir, path) = write_csv("a,b\n");
        let err = profile_csv_streaming(&path).unwrap_err();
        assert!(matches!(err, ProfileError::EmptyInput(_)));
        drop(dir);
    }

    #[test]
    fn reservoir_sampling_is_uniform_and_deterministic() {
        let body: String = (1..=50_000).map(|i| format!("{}\n", i)).collect();
        let (dir, path) = write_csv(&format!("x\n{body}"));
        let opts = StreamingOptions {
            reservoir_capacity: 1_000,
            ..StreamingOptions::default()
        };
        let a = profile_csv_streaming_with(&path, &opts).unwrap();
        let b = profile_csv_streaming_with(&path, &opts).unwrap();
        let (na, nb) = (
            a.columns[0].numeric.as_ref().unwrap(),
            b.columns[0].numeric.as_ref().unwrap(),
        );
        // Identical seeds → identical sample → identical stats.
        assert_eq!(na.five.q1.to_bits(), nb.five.q1.to_bits());
        assert_eq!(na.five.median.to_bits(), nb.five.median.to_bits());
        // Exact mean regardless of sampling.
        assert!((na.mean - 25_000.5).abs() < 1e-6);
        // Algorithm R keeps the reservoir uniform over the whole stream, so
        // the sample median of 1..=50 000 must sit near the true median
        // (sample-median σ ≈ 800 for 1 000 draws; 3 000 ≈ 4σ). A sampler
        // that always admits late values lands far above this band.
        assert!(
            (na.five.median - 25_000.5).abs() < 3_000.0,
            "reservoir median {}",
            na.five.median
        );
        assert!(na.five.q1 < na.five.median);
        assert!(na.five.median < na.five.q3);
        assert_eq!(na.five.min, 1.0);
        assert_eq!(na.five.max, 50_000.0);
        drop(dir);
    }

    #[test]
    fn moments_match_in_memory_on_skewed_data() {
        // Regression: the Pébay m3/m4 recurrences once used wrong coefficients,
        // producing wildly wrong skewness/kurtosis in the streaming path.
        let body: String = (0..50)
            .map(|i| {
                let v = if i % 10 == 0 {
                    100.0 + i as f64
                } else {
                    i as f64
                };
                format!("{v}\n")
            })
            .collect();
        let (dir, path) = write_csv(&format!("x\n{body}"));
        let streamed = profile_csv_streaming(&path).unwrap();
        let owned = DatasetProfile::from_csv(&path).unwrap();
        let (s, o) = (
            streamed.columns[0].numeric.as_ref().unwrap(),
            owned.columns[0].numeric.as_ref().unwrap(),
        );
        assert!(
            (s.skewness - o.skewness).abs() < 1e-9,
            "{} vs {}",
            s.skewness,
            o.skewness
        );
        assert!(
            (s.kurtosis - o.kurtosis).abs() < 1e-9,
            "{} vs {}",
            s.kurtosis,
            o.kurtosis
        );
        drop(dir);
    }

    #[test]
    fn point_biserial_group_zero_is_first_seen_level() {
        // Regression: the pair's group 0 must be the *column's* first-seen
        // level (`a`), not the first level seen with a numeric value (`b`).
        // Row 1 interns `a` but has a missing numeric cell, so the pair is
        // created on row 2 (`b`); the in-memory path encodes `a` → 0
        // regardless. With x small on `b` and large on `a`, r must be
        // negative.
        let (dir, path) = write_csv("g,x\na,\nb,1\nb,2\na,3\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        let owned = DatasetProfile::from_csv(&path).unwrap();
        let s = streamed
            .relationships
            .as_ref()
            .unwrap()
            .point_biserial
            .first()
            .unwrap()
            .correlation;
        let o = owned
            .relationships
            .as_ref()
            .unwrap()
            .point_biserial
            .first()
            .unwrap()
            .correlation;
        assert!(s < 0.0, "streaming r {s} should be negative");
        assert!((s - o).abs() < 1e-9, "streaming {s} vs in-memory {o}");
        drop(dir);
    }

    #[test]
    fn memory_bytes_match_in_memory_accounting() {
        let (dir, path) = write_csv("n,c\n1,a\n2,b\n3,a\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        let owned = DatasetProfile::from_csv(&path).unwrap();
        assert_eq!(streamed.memory_bytes, owned.memory_bytes);
        drop(dir);
    }

    #[test]
    fn all_missing_column_has_no_categorical_stats() {
        // Mirrors the in-memory path: `compute_categorical` returns `None`
        // when no non-missing cell exists.
        let (dir, path) = write_csv("c,n\nNA,1\nnull,2\n,3\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        let owned = DatasetProfile::from_csv(&path).unwrap();
        assert!(streamed.columns[0].categorical.is_none());
        assert!(owned.columns[0].categorical.is_none());
        drop(dir);
    }

    #[test]
    fn streaming_reader_over_cursor() {
        let data = "a,b\n1,2\n3,4\n";
        let profile =
            stream_from_reader(Cursor::new(data.to_string()), &StreamingOptions::default())
                .unwrap();
        assert_eq!(profile.n_rows, 2);
        assert_eq!(profile.columns[0].numeric.as_ref().unwrap().mean, 2.0);
    }

    #[test]
    fn pearson_uses_pairwise_complete_variance() {
        // Regression: the streaming Pearson once divided the pairwise
        // covariance by the columns' *full* variances, so the `100` in x —
        // a row where y is missing — dragged r away from the true
        // pairwise-complete value of 1.0 (it produced ≈ 0.02).
        let (dir, path) = write_csv("x,y\n1,1\n2,2\n3,3\n100,\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        let r = streamed
            .relationships
            .as_ref()
            .unwrap()
            .pearson
            .as_ref()
            .unwrap()
            .values[0][1];
        assert!((r - 1.0).abs() < 1e-12, "pairwise-complete r = {r}");
        drop(dir);
    }

    #[test]
    fn point_biserial_drops_pairs_whose_numeric_column_flipped() {
        // Regression: a numeric column that flips to categorical mid-stream
        // left a stale point-biserial pair behind, reported with a
        // `numeric` label even though the column is categorical in the
        // final profile.
        let (dir, path) = write_csv("g,x\na,1\nb,2\na,3\nb,oops\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        assert!(streamed.columns[1].categorical.is_some());
        let pb = streamed
            .relationships
            .as_ref()
            .map(|r| r.point_biserial.as_slice())
            .unwrap_or(&[]);
        assert!(
            !pb.iter().any(|e| e.numeric == "x"),
            "stale point-biserial entry for flipped column: {pb:?}"
        );
        drop(dir);
    }

    #[test]
    fn crlf_inside_quoted_field_matches_in_memory() {
        // Regression: `BufRead::lines` stripped the trailing `\r` of every
        // physical line, corrupting CRLF content inside quoted fields;
        // `read_line` keeps it, matching `read_csv_reader`.
        let (dir, path) = write_csv("c\n\"a\r\nb\"\n\"a\r\nb\"\nz\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        let owned = DatasetProfile::from_csv(&path).unwrap();
        let s = streamed.columns[0].categorical.as_ref().unwrap();
        let o = owned.columns[0].categorical.as_ref().unwrap();
        assert_eq!(s.top, o.top);
        assert_eq!(s.freq, o.freq);
        drop(dir);
    }

    #[test]
    fn duplicates_use_raw_cells_like_in_memory() {
        // Regression: the tracker canonicalized numerics (`1` ≡ `1.0`) and
        // trimmed cells; the in-memory CSV path compares raw strings.
        let (dir, path) = write_csv("a,b\n1,x\n1.0,x\n1,x\n");
        let opts = StreamingOptions {
            compute_duplicates: true,
            ..StreamingOptions::default()
        };
        let streamed = profile_csv_streaming_with(&path, &opts).unwrap();
        let owned = DatasetProfile::from_csv(&path).unwrap();
        assert_eq!(streamed.duplicate_rows, owned.duplicate_rows);
        // Only the exact `1,x` repeat counts; `1.0,x` is a different row.
        assert_eq!(streamed.duplicate_rows, 1);
        drop(dir);
    }

    #[test]
    fn non_finite_parses_stay_numeric_and_missing() {
        // Regression: `inf` parsed successfully but is non-finite; the
        // streaming path flipped the whole column to categorical, while
        // the in-memory path keeps it numeric and counts the cell missing.
        let (dir, path) = write_csv("x\n1\n2\ninf\n4\n");
        let streamed = profile_csv_streaming(&path).unwrap();
        let owned = DatasetProfile::from_csv(&path).unwrap();
        assert!(streamed.columns[0].numeric.is_some());
        assert_eq!(
            streamed.columns[0].missing_count,
            owned.columns[0].missing_count
        );
        assert_eq!(
            streamed.columns[0].numeric.as_ref().unwrap().mean,
            owned.columns[0].numeric.as_ref().unwrap().mean
        );
        drop(dir);
    }
}
