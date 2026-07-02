use std::collections::HashMap;

use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Scoring function for `SelectKBest`, mirroring sklearn's score functions.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScoreFunc {
    /// ANOVA F-value between label/feature for classification (`f_classif`).
    FClassif,
    /// Chi-square statistic for non-negative features (`chi2`).
    Chi2,
    /// Mutual information for discrete classification (`mutual_info_classif`).
    /// Uses a histogram-based estimator. The number of bins defaults to
    /// `ceil(sqrt(n_samples))`.
    MutualInformation,
}

/// Select the `k` best features according to a scoring function, mirroring
/// `sklearn.feature_selection.SelectKBest`.
///
/// Labels (`y`) are provided via a side-channel at `fit_with_labels` time.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SelectKBest {
    score_func: ScoreFunc,
    k: usize,
    scores: Vec<f64>,
    support_mask: Vec<bool>,
    fitted: bool,
}

impl SelectKBest {
    /// Creates a new selector keeping the `k` best features per `score_func`.
    pub fn new(score_func: ScoreFunc, k: usize) -> Result<Self> {
        if k == 0 {
            return Err(DatarustError::InvalidConfig("k must be > 0".into()));
        }
        Ok(Self {
            score_func,
            k,
            scores: vec![],
            support_mask: vec![],
            fitted: false,
        })
    }

    /// Returns the per-feature scores computed during fit.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Returns the boolean mask of selected features.
    pub fn get_support(&self) -> &[bool] {
        &self.support_mask
    }

    /// Fit using features `x` and class labels `y` (strings or any labels
    /// converted to string form).
    pub fn fit_with_labels<S: AsRef<str>>(&mut self, x: &Matrix, y: &[S]) -> Result<()> {
        if y.len() != x.nrows() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} labels", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }
        let labels: Vec<&str> = y.iter().map(|s| s.as_ref()).collect();
        self.scores = match self.score_func {
            ScoreFunc::FClassif => f_classif(x, &labels)?,
            ScoreFunc::Chi2 => chi2(x, &labels)?,
            ScoreFunc::MutualInformation => mi_classif(x, &labels)?,
        };
        self.compute_support(x.ncols())?;
        self.fitted = true;
        Ok(())
    }

    fn compute_support(&mut self, n_features: usize) -> Result<()> {
        if self.k > n_features {
            return Err(DatarustError::InvalidConfig(format!(
                "k={} must be <= n_features={}",
                self.k, n_features
            )));
        }
        // Rank feature indices by score descending; ties broken by lower index.
        let mut idx: Vec<usize> = (0..n_features).collect();
        idx.sort_by(|&a, &b| {
            self.scores[b]
                .partial_cmp(&self.scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let keep: std::collections::HashSet<usize> = idx.into_iter().take(self.k).collect();
        self.support_mask = (0..n_features).map(|j| keep.contains(&j)).collect();
        Ok(())
    }
}

/// Default: keep 10 best features using F-test scoring.
impl Default for SelectKBest {
    fn default() -> Self {
        Self {
            score_func: ScoreFunc::FClassif,
            k: 10,
            scores: vec![],
            support_mask: vec![],
            fitted: false,
        }
    }
}

impl Transformer for SelectKBest {
    fn name(&self) -> &'static str {
        "SelectKBest"
    }

    fn fit(&mut self, _x: &Matrix) -> Result<()> {
        // No labels available via the standard trait; require labeled fit.
        Err(DatarustError::InvalidInput(
            "SelectKBest requires labels; use fit_with_labels".into(),
        ))
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("SelectKBest".into()));
        }
        if self.support_mask.len() != x.ncols() {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("{} features", self.support_mask.len()),
                actual: format!("{} features", x.ncols()),
            });
        }
        let kept: Vec<usize> = self
            .support_mask
            .iter()
            .enumerate()
            .filter(|(_, &keep)| keep)
            .map(|(j, _)| j)
            .collect();
        let mut out = vec![vec![0.0; kept.len()]; x.nrows()];
        for (i, out_row) in out.iter_mut().enumerate() {
            let x_row = x.row(i);
            for (k, &j) in kept.iter().enumerate() {
                out_row[k] = x_row[j];
            }
        }
        Matrix::new(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl FeatureNames for SelectKBest {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let names: Vec<String> = match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.support_mask.len()),
        };
        names
            .iter()
            .enumerate()
            .filter(|(i, _)| self.support_mask.get(*i).copied().unwrap_or(false))
            .map(|(_, n)| n.clone())
            .collect()
    }
}

/// ANOVA F-value for each feature, given class labels.
#[allow(clippy::needless_range_loop)]
fn f_classif(x: &Matrix, labels: &[&str]) -> Result<Vec<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        groups.entry(l).or_default().push(i);
    }
    let k = groups.len();
    if k < 2 {
        return Err(DatarustError::InvalidInput(format!(
            "f_classif needs >= 2 classes, got {}",
            k
        )));
    }
    let df_between = (k - 1) as f64;
    let df_within = (n - k) as f64;
    if df_within <= 0.0 {
        return Err(DatarustError::InvalidInput(
            "degrees of freedom within must be positive".into(),
        ));
    }
    let mut scores = vec![0.0; p];
    let group_vals: Vec<(&str, &Vec<usize>)> = groups.iter().map(|(k, v)| (*k, v)).collect();
    for j in 0..p {
        // overall mean
        let overall_mean: f64 = (0..n).map(|i| x.get(i, j)).sum::<f64>() / n as f64;
        let mut ssb = 0.0;
        let mut ssw = 0.0;
        for (_label, idxs) in &group_vals {
            let nk = idxs.len() as f64;
            let gm: f64 = idxs.iter().map(|&i| x.get(i, j)).sum::<f64>() / nk;
            ssb += nk * (gm - overall_mean).powi(2);
            for &i in idxs.iter() {
                ssw += (x.get(i, j) - gm).powi(2);
            }
        }
        let msb = ssb / df_between;
        let msw = ssw / df_within;
        scores[j] = if msw > 0.0 { msb / msw } else { f64::INFINITY };
    }
    Ok(scores)
}

/// Chi-square statistic for each (non-negative) feature, given class labels.
/// Mirrors sklearn's `chi2`: observed = sum of feature values per class.
#[allow(clippy::needless_range_loop)]
fn chi2(x: &Matrix, labels: &[&str]) -> Result<Vec<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    for i in 0..n {
        for j in 0..p {
            if x.get(i, j) < 0.0 {
                return Err(DatarustError::InvalidInput(format!(
                    "chi2 requires non-negative features; negative at {},{}",
                    i, j
                )));
            }
        }
    }
    let mut class_counts: HashMap<&str, f64> = HashMap::new();
    for &l in labels {
        *class_counts.entry(l).or_insert(0.0) += 1.0;
    }
    let total: f64 = n as f64;
    let mut feature_totals = vec![0.0; p];
    for j in 0..p {
        feature_totals[j] = (0..n).map(|i| x.get(i, j)).sum();
    }
    let mut scores = vec![0.0; p];
    for j in 0..p {
        let ft = feature_totals[j];
        if ft == 0.0 {
            scores[j] = 0.0;
            continue;
        }
        let mut s = 0.0;
        for (label, &class_count) in &class_counts {
            let expected = ft * class_count / total;
            if expected <= 0.0 {
                continue;
            }
            // observed = sum of feature j over samples in class
            let observed: f64 = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == *label)
                .map(|(i, _)| x.get(i, j))
                .sum();
            let diff = observed - expected;
            s += diff * diff / expected;
        }
        scores[j] = s;
    }
    Ok(scores)
}

/// Mutual information for discrete classification using a histogram estimator.
///
/// For each feature:
/// 1. Bin the continuous feature values into `n_bins` bins.
/// 2. Build a joint histogram `P(feature_bin, class)`.
/// 3. Compute `MI = Σ P(i,j) log( P(i,j) / (P(i) · P(j)) )`.
#[allow(clippy::needless_range_loop)]
fn mi_classif(x: &Matrix, labels: &[&str]) -> Result<Vec<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    let n_bins = ((n as f64).sqrt().ceil() as usize).max(2);

    // Encode labels as contiguous integers.
    let mut label_map: HashMap<&str, usize> = HashMap::new();
    let mut y_enc = Vec::with_capacity(n);
    for &l in labels {
        let next_idx = label_map.len();
        let idx = *label_map.entry(l).or_insert(next_idx);
        y_enc.push(idx);
    }
    let n_classes = label_map.len();

    let mut scores = vec![0.0; p];
    for j in 0..p {
        // Collect feature column and bin edges.
        let col: Vec<f64> = (0..n).map(|i| x.get(i, j)).collect();
        let mut sorted_col = col.clone();
        sorted_col.sort_by(|a, b| a.total_cmp(b));
        let lo = sorted_col[0];
        let hi = sorted_col[n - 1];
        let range = hi - lo;
        let bin_width = if range == 0.0 {
            1.0
        } else {
            range / n_bins as f64
        };

        // Bin each value. A value equal to `hi` goes into the last bin.
        let bin_of = |v: f64| -> usize {
            if range == 0.0 {
                0
            } else {
                let b = ((v - lo) / bin_width).floor() as usize;
                b.min(n_bins - 1)
            }
        };

        // Joint histogram: P(feature_bin, class)
        let mut joint = vec![vec![0.0f64; n_bins]; n_classes];
        for i in 0..n {
            let b = bin_of(col[i]);
            joint[y_enc[i]][b] += 1.0;
        }

        // Marginal: P(class)
        let class_count: Vec<f64> = (0..n_classes)
            .map(|c| joint[c].iter().sum::<f64>())
            .collect();

        // Marginal: P(feature_bin)
        let mut bin_count = vec![0.0f64; n_bins];
        for b in 0..n_bins {
            bin_count[b] = (0..n_classes).map(|c| joint[c][b]).sum();
        }

        // MI
        let mut mi = 0.0;
        for c in 0..n_classes {
            for b in 0..n_bins {
                let p_joint = joint[c][b] / n as f64;
                if p_joint <= 0.0 {
                    continue;
                }
                let p_x = bin_count[b] / n as f64;
                let p_y = class_count[c] / n as f64;
                if p_x > 0.0 && p_y > 0.0 {
                    mi += p_joint * (p_joint / (p_x * p_y)).ln();
                }
            }
        }
        scores[j] = mi;
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn separated_data() -> (Matrix, Vec<&'static str>) {
        // feature 0 separates classes well; feature 1 is random/noise
        let x = Matrix::new(vec![
            vec![0.1, 5.0],
            vec![0.2, 3.0],
            vec![0.1, 4.0],
            vec![10.0, 2.0],
            vec![9.5, 6.0],
            vec![10.5, 1.0],
        ])
        .unwrap();
        let y = vec!["a", "a", "a", "b", "b", "b"];
        (x, y)
    }

    #[test]
    fn f_classif_picks_separating_feature() {
        let (x, y) = separated_data();
        let mut skb = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
        skb.fit_with_labels(&x, &y).unwrap();
        assert_eq!(skb.get_support(), &[true, false]);
        // feature 0 should have a much higher score
        assert!(skb.scores()[0] > skb.scores()[1]);
    }

    #[test]
    fn f_classif_k2_keeps_both() {
        let (x, y) = separated_data();
        let mut skb = SelectKBest::new(ScoreFunc::FClassif, 2).unwrap();
        skb.fit_with_labels(&x, &y).unwrap();
        assert_eq!(skb.get_support(), &[true, true]);
    }

    #[test]
    fn transform_drops_low_score() {
        let (x, y) = separated_data();
        let mut skb = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
        skb.fit_with_labels(&x, &y).unwrap();
        let out = skb.transform(&x).unwrap();
        assert_eq!(out.ncols(), 1);
        assert_eq!(out.row(3), [10.0]);
    }

    #[test]
    fn f_classif_hand_computed() {
        // Two classes, single feature
        // class a: [1,2,3] mean 2 ; class b: [10,11,12] mean 11
        // overall mean = 6.5
        // SSB = 3*(2-6.5)^2 + 3*(11-6.5)^2 = 3*20.25 + 3*20.25 = 121.5
        // SSW = (1+0+1)+(1+0+1) = 4
        // F = (121.5/1)/(4/4) = 121.5
        let x = Matrix::new(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![10.0],
            vec![11.0],
            vec![12.0],
        ])
        .unwrap();
        let y = vec!["a", "a", "a", "b", "b", "b"];
        let scores = f_classif(&x, &y).unwrap();
        assert!(approx(scores[0], 121.5, 1e-6));
    }

    #[test]
    fn chi2_requires_non_negative() {
        let x = Matrix::new(vec![vec![-1.0], vec![2.0]]).unwrap();
        let y = vec!["a", "b"];
        assert!(chi2(&x, &y).is_err());
    }

    #[test]
    fn chi2_basic() {
        // feature strongly associated with classes -> high chi2
        let x = Matrix::new(vec![
            vec![10.0, 1.0],
            vec![10.0, 1.0],
            vec![0.0, 10.0],
            vec![0.0, 10.0],
        ])
        .unwrap();
        let y = vec!["a", "a", "b", "b"];
        let scores = chi2(&x, &y).unwrap();
        // both features separate perfectly here; both should be large
        assert!(scores[0] > 5.0);
        assert!(scores[1] > 5.0);
    }

    #[test]
    fn chi2_picks_separating_feature() {
        let (x, y) = separated_data();
        let mut skb = SelectKBest::new(ScoreFunc::Chi2, 1).unwrap();
        skb.fit_with_labels(&x, &y).unwrap();
        assert_eq!(skb.get_support(), &[true, false]);
    }

    #[test]
    fn mi_picks_separating_feature() {
        let (x, y) = separated_data();
        let mut skb = SelectKBest::new(ScoreFunc::MutualInformation, 1).unwrap();
        skb.fit_with_labels(&x, &y).unwrap();
        assert_eq!(skb.get_support(), &[true, false]);
        assert!(skb.scores()[0] > skb.scores()[1]);
    }

    #[test]
    fn mi_independent_feature_low_score() {
        // feature 0 is constant -> MI should be 0
        // feature 1 varies with class -> MI > 0
        let x = Matrix::new(vec![
            vec![1.0, 0.1],
            vec![1.0, 0.2],
            vec![1.0, 10.0],
            vec![1.0, 11.0],
        ])
        .unwrap();
        let y = vec!["a", "a", "b", "b"];
        let mut skb = SelectKBest::new(ScoreFunc::MutualInformation, 1).unwrap();
        skb.fit_with_labels(&x, &y).unwrap();
        // feature 0 (constant) has MI=0, feature 1 has MI>0
        assert!(skb.scores()[0] < skb.scores()[1]);
    }

    #[test]
    fn k_too_large_errors() {
        let (x, y) = separated_data();
        let mut skb = SelectKBest::new(ScoreFunc::FClassif, 5).unwrap();
        assert!(skb.fit_with_labels(&x, &y).is_err());
    }

    #[test]
    fn single_class_errors() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let y = vec!["a", "a"];
        let mut skb = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
        assert!(skb.fit_with_labels(&x, &y).is_err());
    }

    #[test]
    fn label_count_mismatch() {
        let (x, _y) = separated_data();
        let y_short = vec!["a", "a"];
        let mut skb = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
        assert!(skb.fit_with_labels(&x, &y_short).is_err());
    }

    #[test]
    fn transform_before_fit_errors() {
        let skb = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(matches!(
            skb.transform(&x),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn tie_breaks_by_lower_index() {
        // both features identical -> tie -> k=1 keeps the lower index
        let x = Matrix::new(vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ])
        .unwrap();
        let y = vec!["a", "a", "b", "b"];
        let mut skb = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
        skb.fit_with_labels(&x, &y).unwrap();
        assert_eq!(skb.get_support(), &[true, false]);
    }

    #[test]
    fn feature_names_filters_selected() {
        let (x, y) = separated_data();
        let mut skb = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
        skb.fit_with_labels(&x, &y).unwrap();
        let names = skb.feature_names_out(Some(&["good".into(), "noise".into()]));
        assert_eq!(names, vec!["good"]);
    }

    #[test]
    fn feature_names_default_keeps_selected() {
        let (x, y) = separated_data();
        let mut skb = SelectKBest::new(ScoreFunc::FClassif, 1).unwrap();
        skb.fit_with_labels(&x, &y).unwrap();
        let names = skb.feature_names_out(None);
        assert_eq!(names, vec!["x0"]);
    }
}
