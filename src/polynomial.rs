use crate::error::{DatarustError, Result};
use crate::matrix::Matrix;
use crate::traits::{default_input_names, FeatureNames};
use crate::Transformer;

/// Generate polynomial and interaction features, mirroring
/// `sklearn.preprocessing.PolynomialFeatures`.
///
/// For `degree=2` and input `[x1, x2]` the output (with bias) is:
/// `[1, x1, x2, x1^2, x1*x2, x2^2]`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PolynomialFeatures {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
    combinations: Vec<Vec<usize>>,
    input_n_features: usize,
    fitted: bool,
}

impl PolynomialFeatures {
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            include_bias: true,
            interaction_only: false,
            combinations: vec![],
            input_n_features: 0,
            fitted: false,
        }
    }

    pub fn include_bias(mut self, b: bool) -> Self {
        self.include_bias = b;
        self
    }

    pub fn interaction_only(mut self, b: bool) -> Self {
        self.interaction_only = b;
        self
    }

    pub fn n_output_features(&self) -> usize {
        self.combinations.len()
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Generate all combinations (multisets) of feature indices with total
    /// cardinality `d`, in lexicographic (non-decreasing) order — matching
    /// `itertools.combinations_with_replacement(range(n), d)`.
    fn combinations_with_replacement(n: usize, d: usize) -> Vec<Vec<usize>> {
        if d == 0 {
            return vec![vec![]];
        }
        let mut out = vec![];
        let mut current = vec![0usize; d];
        Self::recurse(n, d, 0, 0, &mut current, &mut out);
        out
    }

    fn recurse(
        n: usize,
        d: usize,
        pos: usize,
        start: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if pos == d {
            out.push(current.clone());
            return;
        }
        for i in start..n {
            current[pos] = i;
            Self::recurse(n, d, pos + 1, i, current, out);
        }
    }

    fn build_combinations(
        n_features: usize,
        degree: usize,
        include_bias: bool,
        interaction_only: bool,
    ) -> Vec<Vec<usize>> {
        let mut combos = vec![];
        for d in 0..=degree {
            let cs = Self::combinations_with_replacement(n_features, d);
            for c in cs {
                if d == 0 {
                    if include_bias {
                        combos.push(c);
                    }
                    continue;
                }
                if interaction_only {
                    // reject any combination with a repeated index
                    let mut sorted = c.clone();
                    sorted.sort_unstable();
                    let has_dup = sorted.windows(2).any(|w| w[0] == w[1]);
                    if has_dup {
                        continue;
                    }
                }
                combos.push(c);
            }
        }
        combos
    }
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self::new(2)
    }
}

impl FeatureNames for PolynomialFeatures {
    fn feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        let names: Vec<String> = match input_features {
            Some(fs) => fs.to_vec(),
            None => default_input_names(self.input_n_features),
        };
        self.combinations
            .iter()
            .map(|combo| combo_name(combo, &names))
            .collect()
    }
}

fn combo_name(combo: &[usize], names: &[String]) -> String {
    if combo.is_empty() {
        return "1".to_string();
    }
    // group consecutive equal indices (combo is non-decreasing)
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < combo.len() {
        let idx = combo[i];
        let mut count = 1;
        while i + count < combo.len() && combo[i + count] == idx {
            count += 1;
        }
        let base = names
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("x{}", idx));
        if count == 1 {
            parts.push(base);
        } else {
            parts.push(format!("{}^{}", base, count));
        }
        i += count;
    }
    parts.join(" ")
}

impl Transformer for PolynomialFeatures {
    fn name(&self) -> &'static str {
        "PolynomialFeatures"
    }

    fn fit(&mut self, x: &Matrix) -> Result<()> {
        if self.degree == 0 && !self.include_bias {
            return Err(DatarustError::InvalidConfig(
                "degree=0 with include_bias=false produces no features".into(),
            ));
        }
        self.combinations = Self::build_combinations(
            x.ncols(),
            self.degree,
            self.include_bias,
            self.interaction_only,
        );
        if self.combinations.is_empty() {
            return Err(DatarustError::InvalidConfig(
                "no output features generated".into(),
            ));
        }
        self.input_n_features = x.ncols();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, x: &Matrix) -> Result<Matrix> {
        if !self.fitted {
            return Err(DatarustError::NotFitted("PolynomialFeatures".into()));
        }
        let nrows = x.nrows();
        let n_out = self.combinations.len();
        let mut out = vec![vec![1.0; n_out]; nrows];
        for (i, out_row) in out.iter_mut().enumerate() {
            let row = x.row(i);
            for (k, combo) in self.combinations.iter().enumerate() {
                let mut prod = 1.0;
                for &idx in combo {
                    prod *= row[idx];
                }
                out_row[k] = prod;
            }
        }
        Matrix::new(out)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m1() -> Matrix {
        Matrix::new(vec![vec![2.0, 3.0]]).unwrap()
    }

    #[test]
    fn degree2_with_bias() {
        let mut pf = PolynomialFeatures::new(2);
        let out = pf.fit_transform(&m1()).unwrap();
        // expected: [1, x1, x2, x1^2, x1*x2, x2^2] = [1, 2, 3, 4, 6, 9]
        assert_eq!(out.row(0), [1.0, 2.0, 3.0, 4.0, 6.0, 9.0]);
        assert_eq!(pf.n_output_features(), 6);
    }

    #[test]
    fn degree2_without_bias() {
        let mut pf = PolynomialFeatures::new(2).include_bias(false);
        let out = pf.fit_transform(&m1()).unwrap();
        // [x1, x2, x1^2, x1*x2, x2^2]
        assert_eq!(out.row(0), [2.0, 3.0, 4.0, 6.0, 9.0]);
        assert_eq!(pf.n_output_features(), 5);
    }

    #[test]
    fn interaction_only() {
        let mut pf = PolynomialFeatures::new(2).interaction_only(true);
        let out = pf.fit_transform(&m1()).unwrap();
        // [1, x1, x2, x1*x2]
        assert_eq!(out.row(0), [1.0, 2.0, 3.0, 6.0]);
        assert_eq!(pf.n_output_features(), 4);
    }

    #[test]
    fn interaction_only_no_bias() {
        let mut pf = PolynomialFeatures::new(2)
            .interaction_only(true)
            .include_bias(false);
        let out = pf.fit_transform(&m1()).unwrap();
        // [x1, x2, x1*x2]
        assert_eq!(out.row(0), [2.0, 3.0, 6.0]);
    }

    #[test]
    fn degree3_three_features() {
        let x = Matrix::new(vec![vec![2.0, 3.0, 4.0]]).unwrap();
        let mut pf = PolynomialFeatures::new(3);
        let out = pf.fit_transform(&x).unwrap();
        // For n=3, degree=3: total = C(3+3,3) = 20 features
        assert_eq!(pf.n_output_features(), 20);
        // spot-check bias and the pure x1^3
        assert!((out.get(0, 0) - 1.0).abs() < 1e-12);
        // last feature x3^3 = 64
        assert!((out.get(0, 19) - 64.0).abs() < 1e-12);
    }

    #[test]
    fn degree1() {
        let mut pf = PolynomialFeatures::new(1);
        let out = pf.fit_transform(&m1()).unwrap();
        // [1, x1, x2]
        assert_eq!(out.row(0), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn multi_row_consistent() {
        let x = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let mut pf = PolynomialFeatures::new(2).include_bias(false);
        let out = pf.fit_transform(&x).unwrap();
        // row0: [1, 2, 1, 2, 4]
        assert_eq!(out.row(0), [1.0, 2.0, 1.0, 2.0, 4.0]);
        // row1: [3, 4, 9, 12, 16]
        assert_eq!(out.row(1), [3.0, 4.0, 9.0, 12.0, 16.0]);
    }

    #[test]
    fn n_output_count_formula() {
        // for n features and degree d, with bias, count = C(n+d, d)
        let x = Matrix::new(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap(); // n=4
        let mut pf = PolynomialFeatures::new(2);
        pf.fit(&x).unwrap();
        // C(4+2, 2) = 15
        assert_eq!(pf.n_output_features(), 15);
    }

    #[test]
    fn transform_before_fit_errors() {
        let pf = PolynomialFeatures::new(2);
        assert!(matches!(
            pf.transform(&m1()),
            Err(DatarustError::NotFitted(_))
        ));
    }

    #[test]
    fn zero_degree_with_bias_constant_one() {
        let mut pf = PolynomialFeatures::new(0);
        let out = pf.fit_transform(&m1()).unwrap();
        // only bias column
        assert_eq!(out.row(0), [1.0]);
    }

    #[test]
    fn zero_degree_without_bias_errors() {
        let mut pf = PolynomialFeatures::new(0).include_bias(false);
        assert!(pf.fit(&m1()).is_err());
    }

    #[test]
    fn feature_names_degree2() {
        let mut pf = PolynomialFeatures::new(2);
        pf.fit(&Matrix::new(vec![vec![2.0, 3.0]]).unwrap()).unwrap();
        let names = pf.feature_names_out(None);
        // [1, x0, x1, x0^2, x0 x1, x1^2]
        assert_eq!(names, vec!["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]);
    }

    #[test]
    fn feature_names_custom_inputs() {
        let mut pf = PolynomialFeatures::new(2).include_bias(false);
        pf.fit(&Matrix::new(vec![vec![2.0, 3.0]]).unwrap()).unwrap();
        let names = pf.feature_names_out(Some(&["age".to_string(), "salary".to_string()]));
        assert_eq!(
            names,
            vec!["age", "salary", "age^2", "age salary", "salary^2"]
        );
    }

    #[test]
    fn feature_names_degree3_groups() {
        let mut pf = PolynomialFeatures::new(3);
        pf.fit(&Matrix::new(vec![vec![1.0, 2.0]]).unwrap()).unwrap();
        let names = pf.feature_names_out(None);
        // degree3 combos for n=2: [],[0],[1],[0,0],[0,1],[1,1],[0,0,0],[0,0,1],[0,1,1],[1,1,1]
        assert_eq!(
            names,
            vec!["1", "x0", "x1", "x0^2", "x0 x1", "x1^2", "x0^3", "x0^2 x1", "x0 x1^2", "x1^3"]
        );
    }
}
