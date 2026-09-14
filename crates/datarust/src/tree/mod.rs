//! Decision-tree estimators: an exact port of scikit-learn 1.6.1's CART
//! (`DecisionTreeClassifier` / `DecisionTreeRegressor`).
//!
//! The splitter is a faithful transliteration of sklearn's
//! `BestSplitter` + `DensePartitioner` (see `crate::tree::splitter`): float32
//! feature quantization, the per-split Fisher-Yates feature draws (ties
//! resolve through the RNG stream, exactly like sklearn's
//! `splitter="best"`), introsort sample ordering, and sklearn's criterion
//! arithmetic. The builder mirrors `DepthFirstTreeBuilder`, including the
//! `EPSILON` tolerance on parent impurity and improvement.
//!
//! # Parity contract
//!
//! With the same `X`, `y`, integer `random_state`, and matching parameters,
//! the fitted tree matches sklearn's node-for-node: split features,
//! thresholds, node impurities, and `feature_importances_` (verified
//! bit-for-bit by `tests/tree_parity.rs` against fixtures generated with
//! scikit-learn 1.6.1 / numpy 2.0.2).
//!
//! Divergences (by design):
//!
//! - No public `sample_weight` (an internal weighted-fit path serves the
//!   forest's bootstrap), `monotonic_cst`, missing-value routing, or sparse
//!   input; `X` must be finite.
//! - `ccp_alpha` (cost-complexity pruning) and the `random`/`mae` splitters
//!   and criteria are not ported.
//! - Only single-output targets.

mod estimators;
mod node;
mod splitter;

pub use estimators::{DecisionTreeClassifier, DecisionTreeRegressor};

use node::ValueSummary;
use splitter::SkBestSplitter;

use crate::error::{DatarustError, Result};
use crate::label_space::LabelSpace;
use crate::matrix::Matrix;
use crate::traits::ParamValue;

/// Split quality criterion.
///
/// Classification nodes use `Gini` (default) or `Entropy`; regression nodes
/// use squared error — sklearn's `criterion="squared_error"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    /// Gini impurity: `1 − Σ p_k²` (classification default).
    Gini,
    /// Shannon entropy: `−Σ p_k log₂ p_k` (sklearn's tree criterion uses
    /// base-2 logs; the port keeps the exact `ln(p)/ln(2)` expression).
    Entropy,
    /// Squared error / variance reduction (regression).
    SquaredError,
}

impl Criterion {
    /// Textual name, mirroring sklearn's parameter spelling.
    pub fn as_str(&self) -> &'static str {
        match self {
            Criterion::Gini => "gini",
            Criterion::Entropy => "entropy",
            Criterion::SquaredError => "squared_error",
        }
    }
}

impl std::str::FromStr for Criterion {
    type Err = DatarustError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "gini" => Ok(Criterion::Gini),
            "entropy" => Ok(Criterion::Entropy),
            "squared_error" | "mse" => Ok(Criterion::SquaredError),
            other => Err(DatarustError::InvalidConfig(format!(
                "unknown criterion '{other}' (expected gini, entropy, or squared_error)"
            ))),
        }
    }
}

impl std::fmt::Display for Criterion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Maximum number of features considered at each split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaxFeatures {
    /// All features (sklearn's tree default).
    All,
    /// `max(1, int(sqrt(n_features)))` (sklearn's conversion).
    Sqrt,
    /// `max(1, int(log2(n_features)))` (sklearn's conversion).
    Log2,
    /// An explicit feature count.
    Count(usize),
}

impl MaxFeatures {
    /// Resolves to a concrete feature count for `n_features`.
    pub fn resolve(&self, n_features: usize) -> usize {
        if n_features == 0 {
            return 0;
        }
        let n = match self {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => ((n_features as f64).sqrt() as usize).max(1),
            MaxFeatures::Log2 => ((n_features as f64).log2() as usize).max(1),
            MaxFeatures::Count(k) => (*k).clamp(1, n_features),
        };
        n.min(n_features).max(1)
    }
}

impl std::str::FromStr for MaxFeatures {
    type Err = DatarustError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "all" => Ok(MaxFeatures::All),
            "sqrt" | "auto" => Ok(MaxFeatures::Sqrt),
            "log2" => Ok(MaxFeatures::Log2),
            other => Err(DatarustError::InvalidConfig(format!(
                "unknown max_features '{other}' (expected all, sqrt, or log2)"
            ))),
        }
    }
}

/// Shared stopping/behavior knobs for both tree estimators.
///
/// Mirrors sklearn's `DecisionTree{Classifier,Regressor}` parameters;
/// setters use the crate's `with_*` builder convention.
#[derive(Debug, Clone, PartialEq)]
struct TreeOptions {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    min_impurity_decrease: f64,
    max_features: MaxFeatures,
    random_state: u64,
}

impl Default for TreeOptions {
    fn default() -> Self {
        TreeOptions {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            max_features: MaxFeatures::All,
            random_state: 0,
        }
    }
}

impl TreeOptions {
    /// Validates the option combination.
    fn validate(&self) -> Result<()> {
        if let Some(d) = self.max_depth {
            if d == 0 {
                return Err(DatarustError::InvalidConfig(
                    "max_depth must be at least 1".into(),
                ));
            }
        }
        if self.min_samples_split < 2 {
            return Err(DatarustError::InvalidConfig(
                "min_samples_split must be at least 2".into(),
            ));
        }
        if self.min_samples_leaf == 0 {
            return Err(DatarustError::InvalidConfig(
                "min_samples_leaf must be at least 1".into(),
            ));
        }
        if self.min_impurity_decrease.is_nan() || self.min_impurity_decrease < 0.0 {
            return Err(DatarustError::InvalidConfig(
                "min_impurity_decrease must be non-negative (positive infinity is allowed)".into(),
            ));
        }
        if self.random_state > u32::MAX as u64 {
            return Err(DatarustError::InvalidConfig(
                "random_state must fit in sklearn's 32-bit seed range".into(),
            ));
        }
        if matches!(self.max_features, MaxFeatures::Count(0)) {
            return Err(DatarustError::InvalidConfig(
                "max_features count must be at least 1".into(),
            ));
        }
        Ok(())
    }

    /// Applies a shared named parameter; `Ok(false)` when the name is not
    /// shared between the two estimators.
    fn set_shared_param(&mut self, name: &str, value: ParamValue) -> Result<bool> {
        match (name, value) {
            ("min_samples_split", ParamValue::Int(v)) => {
                if v < 2 {
                    return Err(DatarustError::InvalidConfig(
                        "min_samples_split must be at least 2".into(),
                    ));
                }
                self.min_samples_split = v;
                Ok(true)
            }
            ("min_samples_leaf", ParamValue::Int(v)) => {
                if v == 0 {
                    return Err(DatarustError::InvalidConfig(
                        "min_samples_leaf must be at least 1".into(),
                    ));
                }
                self.min_samples_leaf = v;
                Ok(true)
            }
            ("min_impurity_decrease", ParamValue::Float(v)) => {
                if v.is_nan() || v < 0.0 {
                    return Err(DatarustError::InvalidConfig(
                        "min_impurity_decrease must be non-negative".into(),
                    ));
                }
                self.min_impurity_decrease = v;
                Ok(true)
            }
            ("random_state", ParamValue::Int(v)) => {
                if (v as u64) > u32::MAX as u64 {
                    return Err(DatarustError::InvalidConfig(
                        "random_state must fit in sklearn's 32-bit seed range".into(),
                    ));
                }
                self.random_state = v as u64;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Shared `Params` entries.
    fn shared_params(&self) -> Vec<(&'static str, ParamValue)> {
        vec![
            ("min_samples_split", ParamValue::Int(self.min_samples_split)),
            ("min_samples_leaf", ParamValue::Int(self.min_samples_leaf)),
            (
                "min_impurity_decrease",
                ParamValue::Float(self.min_impurity_decrease),
            ),
            ("random_state", ParamValue::Int(self.random_state as usize)),
        ]
    }
}

/// Sentinel child index for leaves (sklearn's `_TREE_LEAF`).
pub(crate) const TREE_LEAF: usize = usize::MAX;

/// A node of the fitted tree (arena layout identical to sklearn's `Node`).
#[derive(Debug, Clone)]
pub(crate) struct TreeNode {
    /// Feature index the node splits on (`None`-equivalent for leaves —
    /// sklearn stores `_TREE_UNDEFINED`, we keep 0 on leaves).
    pub feature: usize,
    /// Split threshold: left `<= threshold`, right `> threshold`.
    pub threshold: f64,
    /// Left child index ([`TREE_LEAF`] for leaves).
    pub left: usize,
    /// Right child index ([`TREE_LEAF`] for leaves).
    pub right: usize,
    /// Node impurity at fit time.
    pub impurity: f64,
    /// Weighted sample count (`n_samples` under uniform weights).
    pub weighted_n: f64,
    /// Impurity improvement over the parent, `n_node/n_total`-normalized
    /// like sklearn (0 at the root).
    pub impurity_improvement: f64,
    /// Number of training samples reaching this node.
    pub n_samples: usize,
    /// Node value summary (class histogram or regression moments).
    pub value: ValueSummary,
}

impl TreeNode {
    /// Whether the node is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.left == TREE_LEAF
    }
}

/// The fitted tree: an arena of nodes plus the label vocabulary that
/// interprets classification leaves.
#[derive(Debug, Clone)]
pub struct TreeStructure {
    /// Arena of nodes; index 0 is the root. Children reference indices;
    /// leaves carry [`TREE_LEAF`] child indices.
    pub(crate) nodes: Vec<TreeNode>,
    /// Original class labels in sorted order (classification only).
    pub(crate) classes: Option<LabelSpace>,
    /// Highest split feature index + 1, cached at construction so per-row
    /// routing does not rescan the arena.
    pub(crate) n_features: usize,
}

impl TreeStructure {
    /// Builds a tree from the builder's node arena, caching the feature
    /// width implied by the split features.
    pub(crate) fn new(nodes: Vec<TreeNode>, classes: Option<LabelSpace>) -> Self {
        let n_features = nodes
            .iter()
            .filter(|n| !n.is_leaf())
            .map(|n| n.feature + 1)
            .max()
            .unwrap_or(0);
        TreeStructure {
            nodes,
            classes,
            n_features,
        }
    }

    /// Number of nodes in the tree.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Maximum depth of the tree (a single leaf has depth 0).
    pub fn get_depth(&self) -> usize {
        fn walk(nodes: &[TreeNode], idx: usize) -> usize {
            let node = &nodes[idx];
            if node.is_leaf() {
                0
            } else {
                1 + walk(nodes, node.left).max(walk(nodes, node.right))
            }
        }
        if self.nodes.is_empty() {
            0
        } else {
            walk(&self.nodes, 0)
        }
    }

    /// Number of leaves.
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Sorted original class labels (classification only).
    pub fn classes(&self) -> Option<&[f64]> {
        self.classes.as_ref().map(|ls| ls.labels())
    }

    /// Routes a feature row down the tree and returns the leaf index.
    ///
    /// Returns an error when `row` has the wrong width, contains a non-finite
    /// value, or the fitted arena is malformed. This keeps direct tree
    /// inspection as safe as estimator prediction.
    pub fn leaf_for(&self, row: &[f64]) -> Result<usize> {
        if self.nodes.is_empty() {
            return Err(DatarustError::InvalidInput("tree has no nodes".into()));
        }
        let feature_count = self.n_features;
        if row.len() < feature_count {
            return Err(DatarustError::ShapeMismatch {
                expected: format!("at least {feature_count} features"),
                actual: format!("{} features", row.len()),
            });
        }
        if let Some((index, value)) = row
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(DatarustError::InvalidInput(format!(
                "non-finite prediction value {value} at feature {index}"
            )));
        }

        let mut idx = 0usize;
        loop {
            let node = self.nodes.get(idx).ok_or_else(|| {
                DatarustError::InvalidInput("tree contains an invalid child index".into())
            })?;
            if node.is_leaf() {
                return Ok(idx);
            }
            let next = if row[node.feature] <= node.threshold {
                node.left
            } else {
                node.right
            };
            if next == TREE_LEAF {
                return Err(DatarustError::InvalidInput(
                    "internal tree node has a leaf sentinel child".into(),
                ));
            }
            idx = next;
        }
    }

    /// Split feature of node `idx` (internal nodes only).
    pub fn node_feature(&self, idx: usize) -> Option<usize> {
        let node = self.nodes.get(idx)?;
        (!node.is_leaf()).then_some(node.feature)
    }

    /// The root node's per-class counts (classification only), in class
    /// label order — the fixture-comparable `values[0]` of sklearn's tree.
    pub fn root_class_counts(&self) -> Option<Vec<f64>> {
        match &self.nodes[0].value {
            ValueSummary::Classification(h) => Some(h.counts.clone()),
            ValueSummary::Regression { .. } => None,
        }
    }

    /// Split threshold of node `idx` (internal nodes only).
    pub fn node_threshold(&self, idx: usize) -> Option<f64> {
        let node = self.nodes.get(idx)?;
        (!node.is_leaf()).then_some(node.threshold)
    }

    /// Node value of node `idx`: the per-class counts (classification) or
    /// the target mean (regression).
    pub fn node_regression_value(&self, idx: usize) -> Option<f64> {
        match &self.nodes.get(idx)?.value {
            ValueSummary::Regression { sum, n, .. } => Some(sum / *n),
            ValueSummary::Classification(_) => None,
        }
    }

    /// Impurity of node `idx` at fit time.
    pub fn node_impurity(&self, idx: usize) -> Option<f64> {
        self.nodes.get(idx).map(|n| n.impurity)
    }

    /// Impurity improvement over the parent (`n_node/n_total`-normalized;
    /// 0 at the root and on leaves).
    pub fn node_improvement(&self, idx: usize) -> Option<f64> {
        self.nodes.get(idx).map(|n| n.impurity_improvement)
    }

    /// Number of training samples that reached node `idx`.
    pub fn node_n_samples(&self, idx: usize) -> Option<usize> {
        self.nodes.get(idx).map(|n| n.n_samples)
    }

    /// Per-feature normalized impurity reduction — sklearn's
    /// `feature_importances_` computed from node impurities:
    ///
    /// `Σ_internal (n_node·imp_node − n_L·imp_L − n_R·imp_R) / n_root`,
    /// then normalized to sum 1 with **numpy's pairwise summation**
    /// (reproduced exactly — a sequential fold diverges in the last ulp
    /// for ≥ 8 features).
    ///
    /// sklearn's Cython is compiled with FP contraction, so the per-node
    /// contribution `n_node·imp_node − n_L·imp_L` is a single fused
    /// multiply-add (`fma(n_node, imp_node, −(n_L·imp_L))`); the plain
    /// two-step rounding diverges in the last ulp. The second subtraction
    /// stays unfused, matching the emitted machine code.
    pub fn feature_importances(&self, n_features: usize) -> Vec<f64> {
        let mut importances = vec![0.0; n_features];
        let nodes = &self.nodes;
        for node in nodes {
            if !node.is_leaf() {
                let left = &nodes[node.left];
                let right = &nodes[node.right];
                importances[node.feature] += f64::mul_add(
                    node.weighted_n,
                    node.impurity,
                    -(left.weighted_n * left.impurity),
                ) - right.weighted_n * right.impurity;
            }
        }
        let root_weighted = nodes[0].weighted_n;
        for v in &mut importances {
            *v /= root_weighted;
        }
        let normalizer = numpy_pairwise_sum(&importances);
        if normalizer > 0.0 {
            for v in &mut importances {
                *v /= normalizer;
            }
        }
        importances
    }
}

/// Exact port of numpy's `pairwise_sum` for float64 additions
/// (`numpy/core/src/umath/loops.c`): sequential below 8 elements, 8
/// unrolled accumulators combined as a balanced tree up to 128, and
/// recursive halving (at multiples of 8) beyond. `np.sum` uses this for
/// contiguous float64 arrays, and sklearn's importance normalization calls
/// it — bit parity requires the same rounding.
pub(crate) fn numpy_pairwise_sum(a: &[f64]) -> f64 {
    const PW_BLOCKSIZE: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        let unrolled_end = n - (n % 8);
        while i < unrolled_end {
            r[0] += a[i];
            r[1] += a[i + 1];
            r[2] += a[i + 2];
            r[3] += a[i + 3];
            r[4] += a[i + 4];
            r[5] += a[i + 5];
            r[6] += a[i + 6];
            r[7] += a[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_pairwise_sum(&a[..n2]) + numpy_pairwise_sum(&a[n2..])
    }
}

/// Stack record for the depth-first builder (sklearn's `StackRecord`).
struct StackRecord {
    start: usize,
    end: usize,
    depth: usize,
    parent: usize,
    is_left: bool,
    impurity: f64,
    n_constant_features: usize,
}

/// Exact port of `DepthFirstTreeBuilder` on top of [`SkBestSplitter`].
struct TreeBuilder<'a> {
    splitter: SkBestSplitter<'a>,
    options: TreeOptions,
    min_weight_leaf: f64,
    nodes: Vec<TreeNode>,
}

impl<'a> TreeBuilder<'a> {
    /// Creates the builder using the supplied row weights.
    fn new_with_weights(
        x: &'a Matrix,
        y: &'a [f64],
        labels: Option<&'a [usize]>,
        n_classes: usize,
        criterion: Criterion,
        options: TreeOptions,
        sample_weights: Option<Vec<f64>>,
    ) -> Self {
        let splitter = SkBestSplitter::new_with_weights(
            x,
            y,
            labels,
            n_classes,
            criterion,
            options.max_features,
            options.min_samples_leaf,
            0.0, // min_weight_leaf: no user-supplied minimum
            options.random_state,
            sample_weights,
        );
        TreeBuilder {
            splitter,
            options,
            min_weight_leaf: 0.0,
            nodes: Vec::new(),
        }
    }

    /// Appends a node and returns its index.
    fn add_node(&mut self, node: TreeNode) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// Grows the tree (exact `DepthFirstTreeBuilder.build`).
    fn build(&mut self) {
        let n_node_samples = self.splitter.n_samples();
        let max_depth = self.options.max_depth.unwrap_or(usize::MAX);
        let min_samples_split = self.options.min_samples_split;
        let min_samples_leaf = self.options.min_samples_leaf;
        let min_weight_leaf = self.min_weight_leaf;
        let min_impurity_decrease = self.options.min_impurity_decrease;

        let mut stack = vec![StackRecord {
            start: 0,
            end: n_node_samples,
            depth: 0,
            parent: TREE_LEAF,
            is_left: false,
            impurity: f64::INFINITY,
            n_constant_features: 0,
        }];
        let mut first = true;

        while let Some(record) = stack.pop() {
            let start = record.start;
            let end = record.end;
            let depth = record.depth;
            let parent = record.parent;
            let is_left = record.is_left;
            let mut n_constant_features = record.n_constant_features;

            let n_node_samples = end - start;
            let weighted_n_node_samples = self.splitter.node_reset(start, end);

            let mut is_leaf = depth >= max_depth
                || n_node_samples < min_samples_split
                || n_node_samples < min_samples_leaf.saturating_mul(2)
                || weighted_n_node_samples < 2.0 * min_weight_leaf;

            // sklearn's `first` flag: the root's impurity comes from the
            // criterion at the root, not from a parent split record.
            let node_impurity = if first {
                first = false;
                self.splitter.node_impurity()
            } else {
                record.impurity
            };

            // impurity == 0 with tolerance due to rounding errors
            is_leaf = is_leaf || node_impurity <= f64::EPSILON;

            let mut split = splitter::SplitRecord::init(end);
            if !is_leaf {
                let (s, n_const) = self.splitter.node_split(node_impurity, n_constant_features);
                split = s;
                n_constant_features = n_const;
                // If EPSILON=0 in the below comparison, float precision
                // issues stop splitting, producing trees dissimilar to
                // sklearn v0.18 — the tolerance is part of the contract.
                is_leaf = is_leaf
                    || split.pos >= end
                    || (split.improvement + f64::EPSILON < min_impurity_decrease);
            }

            let node_id = self.add_node(TreeNode {
                feature: split.feature,
                threshold: if is_leaf { 0.0 } else { split.threshold },
                left: TREE_LEAF,
                right: TREE_LEAF,
                impurity: node_impurity,
                weighted_n: weighted_n_node_samples,
                impurity_improvement: if is_leaf { 0.0 } else { split.improvement },
                n_samples: n_node_samples,
                value: self.splitter.node_value(),
            });

            if parent != TREE_LEAF {
                let parent_node = &mut self.nodes[parent];
                if is_left {
                    parent_node.left = node_id;
                } else {
                    parent_node.right = node_id;
                }
            }

            if !is_leaf {
                // Push right child first so the left subtree is processed
                // first (matching sklearn's stack order and node numbering).
                stack.push(StackRecord {
                    start: split.pos,
                    end,
                    depth: depth + 1,
                    parent: node_id,
                    is_left: false,
                    impurity: split.impurity_right,
                    n_constant_features,
                });
                stack.push(StackRecord {
                    start,
                    end: split.pos,
                    depth: depth + 1,
                    parent: node_id,
                    is_left: true,
                    impurity: split.impurity_left,
                    n_constant_features,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::Predictor;
    use estimators::{DecisionTreeClassifier, DecisionTreeRegressor};

    #[test]
    fn perfect_axis_aligned_separation() {
        // Both features separate the classes perfectly; whichever feature the
        // exact draw loop picks, the root threshold is the midpoint of the
        // separating boundary and all leaves are pure.
        let x = Matrix::new(vec![
            vec![1.0, 5.0],
            vec![2.0, 6.0],
            vec![3.0, 7.0],
            vec![8.0, 1.0],
            vec![9.0, 2.0],
            vec![10.0, 3.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();
        assert_eq!(clf.predict(&x).unwrap(), y);
        let tree = clf.tree().unwrap();
        assert_eq!(tree.n_leaves(), 2);
        assert_eq!(tree.get_depth(), 1);
        let feature = tree.node_feature(0).unwrap();
        let threshold = tree.node_threshold(0).unwrap();
        // Verified against sklearn 1.6.1 (random_state=0): the exact draw
        // loop picks feature 1 with threshold 4.0 (midpoint of the sorted
        // boundary values 3.0 and 5.0 in float32).
        assert_eq!(feature, 1);
        assert_eq!(threshold, 4.0);
    }

    #[test]
    fn gini_vs_entropy_split_choice() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        for criterion in [Criterion::Gini, Criterion::Entropy] {
            let mut clf = DecisionTreeClassifier::new().with_criterion(criterion);
            clf.fit(&x, &y).unwrap();
            assert_eq!(clf.predict(&x).unwrap(), y);
        }
    }

    #[test]
    fn max_depth_limits_growth() {
        let x = Matrix::new(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let mut clf = DecisionTreeClassifier::new().with_max_depth(Some(1));
        clf.fit(&x, &y).unwrap();
        assert_eq!(clf.tree().unwrap().get_depth(), 1);
        // With depth 1, accuracy < 1 on non-separable data.
        assert!(clf.predict(&x).unwrap() != y);
    }

    #[test]
    fn min_samples_leaf_forces_leaf_merging() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        // min_samples_leaf = 3 forbids any split of 4 samples.
        let mut clf = DecisionTreeClassifier::new().with_min_samples_leaf(3);
        clf.fit(&x, &y).unwrap();
        assert_eq!(clf.tree().unwrap().n_nodes(), 1);
    }

    #[test]
    fn regression_reduces_variance_monotonically() {
        let x = Matrix::new(vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![10.0],
            vec![11.0],
            vec![12.0],
        ])
        .unwrap();
        let y = vec![1.0, 2.0, 1.5, 10.0, 11.0, 10.5];
        let mut reg = DecisionTreeRegressor::new();
        reg.fit(&x, &y).unwrap();
        let pred = reg.predict(&x).unwrap();
        // With default settings the tree grows to pure leaves, so training
        // predictions reproduce the targets exactly (sklearn parity).
        for i in 0..y.len() {
            assert!((pred[i] - y[i]).abs() < 1e-12);
        }
        // R² on training data with a perfect piecewise fit is 1.0.
        let r2 = crate::traits::Regressor::score(&reg, &x, &y).unwrap();
        assert!((r2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn feature_importances_sum_to_one() {
        let x = Matrix::new(vec![
            vec![1.0, 9.0],
            vec![2.0, 8.0],
            vec![3.0, 7.0],
            vec![9.0, 1.0],
            vec![8.0, 2.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();
        let imp = clf.feature_importances().unwrap();
        let total: f64 = imp.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unfitted_predict_errors() {
        let clf = DecisionTreeClassifier::new();
        let x = Matrix::new(vec![vec![1.0]]).unwrap();
        assert!(clf.predict(&x).is_err());
        assert!(!clf.is_fitted());
    }

    #[test]
    fn label_validation_via_label_space() {
        // Non-integer labels are rejected by the LabelSpace contract.
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let y = vec![0.0, 0.5];
        let mut clf = DecisionTreeClassifier::new();
        assert!(clf.fit(&x, &y).is_err());
    }

    #[test]
    fn shape_mismatch_is_reported() {
        let x = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();
        let mut clf = DecisionTreeClassifier::new();
        assert!(clf.fit(&x, &[0.0]).is_err());
    }

    #[test]
    fn predict_proba_rows_sum_to_one() {
        let x = Matrix::new(vec![
            vec![1.0, 5.0],
            vec![2.0, 6.0],
            vec![8.0, 1.0],
            vec![9.0, 2.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();
        let proba = crate::traits::PredictProba::predict_proba(&clf, &x).unwrap();
        assert_eq!(proba.ncols(), 2);
        for i in 0..proba.nrows() {
            let s: f64 = (0..2).map(|j| proba.get(i, j)).sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn deterministic_given_random_state() {
        // max_features forces RNG use; the same seed must give identical fits.
        let x = Matrix::new(vec![
            vec![1.0, 5.0, 2.0],
            vec![2.0, 6.0, 3.0],
            vec![3.0, 7.0, 4.0],
            vec![8.0, 1.0, 9.0],
            vec![9.0, 2.0, 8.0],
            vec![10.0, 3.0, 7.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let build = || {
            let mut clf = DecisionTreeClassifier::new()
                .with_max_features(MaxFeatures::Count(1))
                .with_random_state(7);
            clf.fit(&x, &y).unwrap();
            clf.predict(&x).unwrap()
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn pure_node_never_splits() {
        // All labels identical: the root's impurity is 0 (≤ EPSILON), so the
        // tree is a single leaf regardless of feature structure.
        let x = Matrix::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        let y = vec![1.0, 1.0, 1.0, 1.0];
        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();
        let tree = clf.tree().unwrap();
        assert_eq!(tree.n_nodes(), 1);
        assert_eq!(tree.n_leaves(), 1);
    }

    #[test]
    fn criterion_and_max_features_parse_from_str() {
        assert_eq!("gini".parse::<Criterion>().unwrap(), Criterion::Gini);
        assert_eq!(
            "squared_error".parse::<Criterion>().unwrap(),
            Criterion::SquaredError
        );
        assert!("bogus".parse::<Criterion>().is_err());
        assert_eq!("sqrt".parse::<MaxFeatures>().unwrap(), MaxFeatures::Sqrt);
        assert!("nope".parse::<MaxFeatures>().is_err());
    }
}
