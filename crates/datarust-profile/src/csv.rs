//! Pure-Rust CSV reading for the `csv` feature.
//!
//! A small RFC-4180-subset reader with zero external dependencies: quoted
//! fields, embedded delimiters/newlines/quotes (`""` escapes), LF and CRLF
//! line endings, and UTF-8 BOM stripping. Parsing is intentionally strict: a
//! row with a different field count than the first data row is an error
//! rather than a silent misalignment.
//!
//! The reader feeds exactly the same [`StrMatrix`] path as
//! [`crate::profile_str_matrix`], so `profile_csv(path)` and
//! `profile_str_matrix` produce bit-identical profiles on the same data.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use datarust::StrMatrix;

use crate::error::{ProfileError, Result};

/// Options controlling CSV parsing.
#[derive(Debug, Clone)]
pub struct CsvReadOptions {
    /// Field delimiter byte (default `b','`). Must be ASCII.
    pub delimiter: u8,
    /// Quote byte (default `Some(b'"')`). `None` disables quoting entirely.
    /// Must be ASCII when present.
    pub quote: Option<u8>,
    /// Whether the first physical record is a header (default `true`).
    pub has_header: bool,
    /// Trim ASCII whitespace from every unquoted field (default `false`).
    pub trim: bool,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        CsvReadOptions {
            delimiter: b',',
            quote: Some(b'"'),
            has_header: true,
            trim: false,
        }
    }
}

/// One parsed record: its fields plus the 1-based physical file line where
/// the record started (blank lines and quoted newlines are counted).
pub(crate) type Record = (Vec<String>, usize);

/// A list of parsed records.
pub(crate) type Records = Vec<Record>;

/// A parsed CSV document: optional header plus a rectangular data block.
#[derive(Debug, Clone)]
pub struct CsvTable {
    /// Header names when the source had a header row, `None` otherwise.
    pub header: Option<Vec<String>>,
    /// Data rows (excluding the header), as a [`StrMatrix`].
    pub rows: StrMatrix,
}

impl CsvTable {
    /// Number of data rows.
    pub fn nrows(&self) -> usize {
        self.rows.nrows()
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.rows.ncols()
    }

    /// Column names: the header when present (and of matching length),
    /// `x0..x{n-1}` otherwise.
    pub fn column_names(&self) -> Vec<String> {
        match &self.header {
            Some(h) if h.len() == self.ncols() => h.clone(),
            _ => (0..self.ncols()).map(|j| format!("x{j}")).collect(),
        }
    }
}

/// Reads and parses a CSV file with [`CsvReadOptions::default`].
pub fn read_csv(path: &Path) -> Result<CsvTable> {
    read_csv_with(path, &CsvReadOptions::default())
}

/// Reads and parses a CSV file with explicit options.
pub fn read_csv_with(path: &Path, opts: &CsvReadOptions) -> Result<CsvTable> {
    let file = File::open(path)?;
    read_csv_reader(BufReader::new(file), opts)
}

/// Reads only the first CSV record (the header when `has_header`) without
/// parsing the rest of the file — cheap up-front validation of arguments
/// such as a target column name. The UTF-8 BOM is stripped like
/// [`read_csv_reader`].
pub fn read_csv_header(path: &Path, opts: &CsvReadOptions) -> Result<Vec<String>> {
    let file = File::open(path)?;
    read_csv_header_reader(BufReader::new(file), opts)
}

/// Reader-based counterpart of [`read_csv_header`].
pub fn read_csv_header_reader<R: BufRead>(
    mut reader: R,
    opts: &CsvReadOptions,
) -> Result<Vec<String>> {
    validate_options(opts)?;
    let mut parser = RecordParser::new(opts);
    let mut records: Vec<(Vec<String>, usize)> = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        parser.feed(&buf, &mut records)?;
        // One physical line ends at most one record, so the first record
        // is complete as soon as it appears.
        if let Some((row, _)) = records.first() {
            let mut row = row.clone();
            if let Some(cell) = row.first_mut() {
                if let Some(stripped) = cell.strip_prefix('\u{feff}') {
                    *cell = stripped.to_string();
                }
            }
            return Ok(row);
        }
    }
    Err(ProfileError::EmptyInput("CSV has no records".into()))
}

/// Rejects option combinations the record parser cannot represent.
fn validate_options(opts: &CsvReadOptions) -> Result<()> {
    if !opts.delimiter.is_ascii() {
        return Err(ProfileError::InvalidInput(
            "CSV delimiter must be an ASCII byte".into(),
        ));
    }
    // CR/LF terminate records; accepting them as delimiter or quote would
    // make record termination impossible.
    if matches!(opts.delimiter, b'\r' | b'\n') {
        return Err(ProfileError::InvalidInput(
            "CSV delimiter must not be CR or LF".into(),
        ));
    }
    if let Some(q) = opts.quote {
        if !q.is_ascii() {
            return Err(ProfileError::InvalidInput(
                "CSV quote must be an ASCII byte".into(),
            ));
        }
        if matches!(q, b'\r' | b'\n') {
            return Err(ProfileError::InvalidInput(
                "CSV quote must not be CR or LF".into(),
            ));
        }
        if q == opts.delimiter {
            return Err(ProfileError::InvalidInput(
                "CSV delimiter and quote byte must differ".into(),
            ));
        }
    }
    Ok(())
}

/// Parses CSV records from any [`BufRead`] with explicit options.
///
/// The delimiter and quote bytes must be ASCII and differ. A trailing newline
/// is optional; fully blank records are dropped, and a header-only file is an
/// `EmptyInput` error. Each parsed record also carries the 1-based physical
/// file line where it started (blank lines and quoted newlines are counted).
pub fn read_csv_reader<R: BufRead>(mut reader: R, opts: &CsvReadOptions) -> Result<CsvTable> {
    validate_options(opts)?;

    let mut parser = RecordParser::new(opts);
    let mut records: Vec<(Vec<String>, usize)> = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        parser.feed(&buf, &mut records)?;
    }
    parser.finish(&mut records)?;

    // Strip the UTF-8 BOM from the very first cell.
    if let Some((row, _)) = records.first_mut() {
        if let Some(cell) = row.first_mut() {
            if let Some(stripped) = cell.strip_prefix('\u{feff}') {
                *cell = stripped.to_string();
            }
        }
    }

    if records.is_empty() {
        return Err(ProfileError::EmptyInput("CSV has no records".into()));
    }

    let (header, data): (Option<Vec<String>>, Records) = if opts.has_header {
        (Some(records.remove(0).0), records)
    } else {
        (None, records)
    };
    if data.is_empty() {
        return Err(ProfileError::EmptyInput("CSV has no data rows".into()));
    }
    // The expected width is the header's when present, else the first data
    // row's (which then defines the table width).
    let cols = header.as_ref().map_or(data[0].0.len(), |h| h.len());
    if cols == 0 {
        return Err(ProfileError::EmptyInput("CSV has no columns".into()));
    }
    if let Some((len, line)) = data
        .iter()
        .find(|(r, _)| r.len() != cols)
        .map(|(r, l)| (r.len(), *l))
    {
        // `line` is the physical file line where the record started, tracked
        // by the parser (blank lines never shift it).
        return Err(ProfileError::InvalidInput(format!(
            "CSV line {line} has {len} fields, expected {cols}"
        )));
    }
    let rows = StrMatrix::from_strings(data.into_iter().map(|(row, _)| row))?;
    Ok(CsvTable { header, rows })
}

/// Incremental RFC-4180 record parser used by [`read_csv_reader`] and the
/// streaming profiler.
///
/// Feeds line-shaped chunks and emits complete records into `out`. Quoted
/// fields may span lines; quote state is tracked across `feed` calls.
pub(crate) struct RecordParser {
    delimiter: u8,
    quote: Option<u8>,
    trim: bool,
    /// Current physical line, 1-based; each `feed` call is one line.
    line: usize,
    /// Physical line where the current record started (`0` = not started).
    record_line: usize,
    record: Vec<String>,
    field: String,
    /// Whether `field` was quoted (disables `trim` and blank-record dropping).
    field_quoted: bool,
    /// Whether the current record contained at least one quoted field.
    record_has_quote: bool,
    /// Whether we are inside a quoted field (possibly spanning lines).
    in_quotes: bool,
}

impl RecordParser {
    pub(crate) fn new(opts: &CsvReadOptions) -> Self {
        RecordParser {
            delimiter: opts.delimiter,
            quote: opts.quote,
            trim: opts.trim,
            line: 0,
            record_line: 0,
            record: Vec::new(),
            field: String::new(),
            field_quoted: false,
            record_has_quote: false,
            in_quotes: false,
        }
    }

    /// Feeds one physical line (including its terminator). Quoted fields
    /// spanning lines keep their newline as field data. Callers must feed
    /// exactly one line per call so that record line numbers stay accurate.
    pub(crate) fn feed(&mut self, line: &str, out: &mut Vec<(Vec<String>, usize)>) -> Result<()> {
        self.line += 1;
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            if self.in_quotes {
                if self.quote.is_some_and(|q| c == q as char) {
                    let quote = self.quote.unwrap() as char;
                    if chars.peek() == Some(&quote) {
                        // Escaped quote (""), part of the field value.
                        chars.next();
                        self.field.push(c);
                    } else {
                        self.in_quotes = false;
                    }
                } else {
                    self.field.push(c);
                }
                continue;
            }
            if self.record_line == 0 {
                self.record_line = self.line;
            }
            if self.quote.is_some_and(|q| c == q as char) {
                if !self.field.is_empty() {
                    return Err(ProfileError::InvalidInput(
                        "CSV quote appears inside an unquoted field".into(),
                    ));
                }
                self.in_quotes = true;
                self.field_quoted = true;
                self.record_has_quote = true;
                continue;
            }
            if c == self.delimiter as char {
                self.end_field();
                continue;
            }
            if c == '\r' && chars.peek() == Some(&'\n') {
                chars.next();
                self.end_field();
                self.end_record(out);
                continue;
            }
            if c == '\n' {
                self.end_field();
                self.end_record(out);
                continue;
            }
            self.field.push(c);
        }
        Ok(())
    }

    /// Flushes the final record when the input ends without a trailing
    /// newline. Errors when the input ended inside a quoted field.
    pub(crate) fn finish(&mut self, out: &mut Vec<(Vec<String>, usize)>) -> Result<()> {
        if self.in_quotes {
            return Err(ProfileError::InvalidInput(
                "CSV input ended inside a quoted field".into(),
            ));
        }
        // `field_quoted` covers a final *empty* quoted field (`""` with no
        // trailing newline), which must still complete its record.
        if !self.field.is_empty() || !self.record.is_empty() || self.field_quoted {
            self.end_field();
            self.end_record(out);
        }
        Ok(())
    }

    fn end_field(&mut self) {
        let mut value = std::mem::take(&mut self.field);
        if self.trim && !self.field_quoted {
            value = value.trim().to_string();
        }
        self.record.push(value);
        self.field_quoted = false;
    }

    fn end_record(&mut self, out: &mut Vec<(Vec<String>, usize)>) {
        // A single empty unquoted field is a blank line, not a record.
        if self.record.len() == 1 && self.record[0].is_empty() && !self.record_has_quote {
            self.record.clear();
            self.record_line = 0;
            return;
        }
        out.push((std::mem::take(&mut self.record), self.record_line.max(1)));
        self.record_line = 0;
        self.record_has_quote = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse(input: &str) -> Result<CsvTable> {
        read_csv_reader(Cursor::new(input.to_string()), &CsvReadOptions::default())
    }

    #[test]
    fn parses_plain_rows_with_header() {
        let t = parse("a,b,c\n1,2,3\n4,5,6\n").unwrap();
        assert_eq!(t.nrows(), 2);
        assert_eq!(t.ncols(), 3);
        assert_eq!(
            t.header.as_deref().unwrap(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(t.rows.get(0, 0), "1");
        assert_eq!(t.rows.get(1, 2), "6");
    }

    #[test]
    fn parses_quoted_fields_and_escapes() {
        let t = parse("name,note\n\"Doe, John\",\"said \"\"hi\"\"\"\n").unwrap();
        assert_eq!(t.rows.get(0, 0), "Doe, John");
        assert_eq!(t.rows.get(0, 1), "said \"hi\"");
    }

    #[test]
    fn quoted_multiline_field() {
        let t = parse("a,b\n\"line1\nline2\",x\n").unwrap();
        assert_eq!(t.nrows(), 1);
        assert_eq!(t.rows.get(0, 0), "line1\nline2");
    }

    #[test]
    fn crlf_and_lf_identical() {
        let lf = parse("a,b\n1,2\n").unwrap();
        let crlf = parse("a,b\r\n1,2\r\n").unwrap();
        assert_eq!(lf.rows.get(0, 0), crlf.rows.get(0, 0));
        assert_eq!(lf.rows.get(0, 1), crlf.rows.get(0, 1));
    }

    #[test]
    fn bom_is_stripped() {
        let t = parse("\u{feff}a,b\n1,2\n").unwrap();
        assert_eq!(t.header.as_deref().unwrap()[0], "a");
    }

    #[test]
    fn no_header_option() {
        let opts = CsvReadOptions {
            has_header: false,
            ..Default::default()
        };
        let t = read_csv_reader(Cursor::new("1,2\n3,4\n".to_string()), &opts).unwrap();
        assert!(t.header.is_none());
        assert_eq!(t.nrows(), 2);
        assert_eq!(t.column_names(), vec!["x0".to_string(), "x1".to_string()]);
    }

    #[test]
    fn blank_lines_skipped() {
        let t = parse("a,b\n\n1,2\n\n\n3,4\n").unwrap();
        assert_eq!(t.nrows(), 2);
    }

    #[test]
    fn trailing_newline_optional() {
        let t = parse("a,b\n1,2").unwrap();
        assert_eq!(t.nrows(), 1);
    }

    #[test]
    fn ragged_row_is_rejected_with_line_number() {
        let err = parse("a,b\n1,2,3\n").unwrap_err();
        assert!(err.to_string().contains("line 2 has 3 fields, expected 2"));
    }

    #[test]
    fn non_ascii_chars_matching_ascii_low_byte_stay_intact() {
        // `Ĭ` (U+012C) and `Ģ` (U+0122) have ',' and '"' as their low bytes;
        // `c as u8` comparisons would truncate them into delimiter/quote.
        let t = parse("a,b\nĬ,x\n").unwrap();
        assert_eq!(t.nrows(), 1);
        assert_eq!(t.rows.get(0, 0), "Ĭ");
        assert_eq!(t.rows.get(0, 1), "x");

        let t = parse("a\nĢ\n").unwrap();
        assert_eq!(t.rows.get(0, 0), "Ģ");
    }

    #[test]
    fn cr_or_lf_delimiter_and_quote_are_rejected() {
        for bad in [b'\r', b'\n'] {
            let opts = CsvReadOptions {
                delimiter: bad,
                ..Default::default()
            };
            let err = read_csv_reader(Cursor::new(String::new()), &opts).unwrap_err();
            assert!(err.to_string().contains("CR or LF"), "{err}");

            let opts = CsvReadOptions {
                quote: Some(bad),
                ..Default::default()
            };
            let err = read_csv_reader(Cursor::new(String::new()), &opts).unwrap_err();
            assert!(err.to_string().contains("CR or LF"), "{err}");
        }
    }

    #[test]
    fn ragged_line_number_accounts_for_blank_lines() {
        // The blank line is skipped, but physical line numbers do not shift.
        let err = parse("a,b\n\n1,2,3\n").unwrap_err();
        assert!(err.to_string().contains("line 3"), "{err}");
    }

    #[test]
    fn quoted_empty_field_without_trailing_newline() {
        // Headerless: the single record is data, not a header.
        let opts = CsvReadOptions {
            has_header: false,
            ..Default::default()
        };
        let t = read_csv_reader(Cursor::new("\"\"".to_string()), &opts).unwrap();
        assert_eq!(t.nrows(), 1);
        assert_eq!(t.ncols(), 1);
        assert_eq!(t.rows.get(0, 0), "");
    }

    #[test]
    fn non_ascii_delimiter_is_rejected() {
        let opts = CsvReadOptions {
            delimiter: 0xFF,
            ..Default::default()
        };
        let err = read_csv_reader(Cursor::new("a\n".to_string()), &opts).unwrap_err();
        assert!(matches!(err, ProfileError::InvalidInput(_)));
    }

    #[test]
    fn delimiter_and_quote_configurable() {
        let opts = CsvReadOptions {
            delimiter: b';',
            ..Default::default()
        };
        let t = read_csv_reader(Cursor::new("a;b\n1;2\n".to_string()), &opts).unwrap();
        assert_eq!(t.rows.get(0, 1), "2");
    }

    #[test]
    fn quoted_delimiter_stays_in_field() {
        let opts = CsvReadOptions {
            delimiter: b';',
            ..Default::default()
        };
        let t = read_csv_reader(Cursor::new("a;b\n\"1;2\";3\n".to_string()), &opts).unwrap();
        assert_eq!(t.rows.get(0, 0), "1;2");
        assert_eq!(t.rows.get(0, 1), "3");
    }

    #[test]
    fn empty_input_is_error() {
        let err = parse("").unwrap_err();
        assert!(matches!(err, ProfileError::EmptyInput(_)));
    }

    #[test]
    fn header_only_is_empty_input() {
        let err = parse("a,b\n").unwrap_err();
        assert!(matches!(err, ProfileError::EmptyInput(_)));
    }

    #[test]
    fn trim_option_applies_to_unquoted_fields() {
        let opts = CsvReadOptions {
            trim: true,
            ..Default::default()
        };
        let t = read_csv_reader(Cursor::new("a , b\n x ,\" y \"\n".to_string()), &opts).unwrap();
        assert_eq!(t.rows.get(0, 0), "x");
        // Quoted fields are never trimmed.
        assert_eq!(t.rows.get(0, 1), " y ");
    }

    #[test]
    fn quote_none_keeps_quote_characters() {
        let opts = CsvReadOptions {
            quote: None,
            ..Default::default()
        };
        let t = read_csv_reader(Cursor::new("a,b\n\"x\",2\n".to_string()), &opts).unwrap();
        assert_eq!(t.rows.get(0, 0), "\"x\"");
        assert_eq!(t.rows.get(0, 1), "2");
    }

    #[test]
    fn same_delimiter_and_quote_is_rejected() {
        let opts = CsvReadOptions {
            delimiter: b',',
            quote: Some(b','),
            ..Default::default()
        };
        let err = read_csv_reader(Cursor::new("a,b\n".to_string()), &opts).unwrap_err();
        assert!(matches!(err, ProfileError::InvalidInput(_)));
    }

    #[test]
    fn unterminated_quote_is_rejected() {
        let err = parse("a,b\n\"open,2\n").unwrap_err();
        assert!(err.to_string().contains("ended inside a quoted field"));
    }

    #[test]
    fn quoted_empty_field_is_kept() {
        let t = parse("a,b\n\"\",x\n").unwrap();
        assert_eq!(t.rows.get(0, 0), "");
        assert_eq!(t.rows.get(0, 1), "x");
    }

    #[test]
    fn read_csv_file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.csv");
        std::fs::write(&path, "a,b\n1,2\n").unwrap();
        let t = read_csv(&path).unwrap();
        assert_eq!(t.nrows(), 1);
        assert_eq!(t.rows.get(0, 0), "1");
    }
}
