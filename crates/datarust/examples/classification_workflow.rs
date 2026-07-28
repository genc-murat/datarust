//! Classification workflow: mixed (numeric + categorical) data → preprocessing →
//! logistic regression → classification metrics → threshold tuning →
//! stratified K-fold cross-validation.
//!
//! Scenario: customer churn prediction. Predict whether a customer will leave
//! (1/0) from tenure (months), monthly charge, age, and contract type.
//!
//! Run: `cargo run --example classification_workflow`

use datarust::compose::{ColumnTransformer, Remainder, Table};
use datarust::encoder::{HandleUnknown, OneHotEncoder};
use datarust::linear_model::{LogisticRegression, LogisticSolver};
use datarust::metrics::classification::{
    accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score,
};
use datarust::model_selection::StratifiedKFold;
use datarust::scaler::StandardScaler;
use datarust::traits::Predictor;
use datarust::transformer_kind::TransformerKind;
use datarust::CategoricalTransformerKind;
use datarust::{Matrix, StrMatrix};

/// Simple deterministic PRNG (xorshift64) for reproducible synthetic data.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Generate synthetic churn data ───────────────────────────────
    // Simple rule: short tenure + high monthly charge → high churn risk.
    let mut rng = Rng::new(2024);
    let n = 200;
    let mut num_rows: Vec<Vec<f64>> = Vec::with_capacity(n); // tenure, monthly_charge, age
    let mut cat_rows: Vec<Vec<&'static str>> = Vec::with_capacity(n); // contract_type
    let mut y: Vec<f64> = Vec::with_capacity(n);
    let contract_types: [&'static str; 3] = ["MonthToMonth", "OneYear", "TwoYear"];

    for _ in 0..n {
        let tenure = rng.next_f64() * 72.0; // 0–72 months
        let monthly = 20.0 + rng.next_f64() * 100.0; // 20–120 $
        let age = 18.0 + rng.next_f64() * 60.0; // 18–78 years
                                                // Contract type is random; shorter contracts increase churn risk.
        let ct_idx = (rng.next_f64() * 3.0) as usize;
        let ct = contract_types[ct_idx.min(2)];
        num_rows.push(vec![tenure, monthly, age]);
        cat_rows.push(vec![ct]);

        // Churn probability is not random — it is derived from the features:
        // short tenure, high charge, and month-to-month contracts raise it.
        let contract_boost = match ct {
            "MonthToMonth" => 2.5,
            "OneYear" => 0.0,
            _ => -2.5,
        };
        let score = -0.08 * tenure + 0.03 * monthly + contract_boost;
        let churn = score > -1.0;
        y.push(if churn { 1.0 } else { 0.0 });
    }
    let pos = y.iter().filter(|&&v| v == 1.0).count();
    let numeric = Matrix::new(num_rows)?;
    let categorical = StrMatrix::from_strings(cat_rows.to_vec())?;
    let table = Table::new(numeric, categorical)?;
    println!("=== Customer Churn Classification ===");
    println!(
        "Data: {n} samples ({:.0}% positive)",
        100.0 * pos as f64 / n as f64
    );
    println!();

    // ── 2. Preprocessing: StandardScaler on numeric, OneHot on categorical
    // ColumnTransformer, like its sklearn counterpart, applies different
    // transforms to different columns. It processes all data at once and emits
    // a single numeric Matrix ready to feed into LogisticRegression.
    let mut ct = ColumnTransformer::new()
        .remainder(Remainder::Drop)
        .add_numeric(
            "num_scaled",
            vec![0, 1, 2],
            TransformerKind::StandardScaler(StandardScaler::new()),
        )
        .add_categorical(
            "contract_ohe",
            vec![0],
            CategoricalTransformerKind::OneHotEncoder(
                OneHotEncoder::new().handle_unknown(HandleUnknown::Ignore),
            ),
        );
    let x = ct.fit_transform(&table)?;
    println!(
        "Feature matrix after preprocessing: {} × {}",
        x.nrows(),
        x.ncols()
    );

    // ── 3. Train logistic regression (on all data, for simplicity) ─────
    // The SVD solver is more robust on rank-deficient inputs (e.g. collinear
    // one-hot columns).
    let mut model = LogisticRegression::new().with_solver(LogisticSolver::Svd);
    model.fit(&x, &y)?;
    let preds = model.predict(&x)?; // {0.0, 1.0} labels
    let proba_pos = model.predict_positive_proba(&x)?; // P(y=1)
    println!("Training complete ({} IRLS iterations).\n", model.n_iter());

    // ── 4. Classification metrics ──────────────────────────────────────
    let acc = accuracy_score(&y, &preds)?;
    let prec = precision_score(&y, &preds)?;
    let rec = recall_score(&y, &preds)?;
    let f1 = f1_score(&y, &preds)?;
    let cm = confusion_matrix(&y, &preds)?;
    let ll = log_loss(&y, &proba_pos, 1e-15)?;
    println!("=== Classification Metrics (training set) ===");
    println!("Accuracy  : {acc:.4}");
    println!("Precision : {prec:.4}  — of predicted positives, how many are truly positive");
    println!("Recall    : {rec:.4}  — of true positives, how many were caught");
    println!("F1 score  : {f1:.4}  — harmonic mean of precision and recall");
    println!("Log loss  : {ll:.4}  — probability calibration (lower is better)");
    println!(
        "Confusion matrix: [[TN={}, FP={}], [FN={}, TP={}]]",
        cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    );
    println!();

    // ── 5. Threshold tuning: precision ↔ recall trade-off ──────────────
    // The default threshold is 0.5. For problems like churn where the cost of
    // a missed positive is high, lowering the threshold raises recall (but
    // lowers precision).
    println!("=== Threshold Comparison ===");
    println!(
        "{:<10} {:<10} {:<10} {:<10}",
        "Threshold", "Precision", "Recall", "F1"
    );
    for &thr in &[0.3, 0.5, 0.7] {
        let custom_pred: Vec<f64> = proba_pos
            .iter()
            .map(|&p| if p >= thr { 1.0 } else { 0.0 })
            .collect();
        let p = precision_score(&y, &custom_pred)?;
        let r = recall_score(&y, &custom_pred)?;
        let f = f1_score(&y, &custom_pred)?;
        println!("{:<10.1} {:<10.4} {:<10.4} {:<10.4}", thr, p, r, f);
    }
    println!("(Lower threshold → more aggressive positive prediction → recall ↑, precision ↓)\n");

    // ── 6. Stratified K-fold cross-validation ──────────────────────────
    // cross_val_score only supports KFold; for imbalanced classes,
    // StratifiedKFold is preferred. We therefore drive the split loop manually
    // and refit the model on each fold — the class ratio is preserved per fold.
    let skf = StratifiedKFold::new()
        .with_n_splits(5)
        .with_shuffle(true)
        .with_random_state(3);
    let mut fold_accs: Vec<f64> = Vec::new();
    for (train_idx, test_idx) in skf.split(&y)? {
        let x_tr = x.select_rows(&train_idx)?;
        let x_te = x.select_rows(&test_idx)?;
        let y_tr: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let y_te: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();
        let mut m = LogisticRegression::new().with_solver(LogisticSolver::Svd);
        m.fit(&x_tr, &y_tr)?;
        let p = m.predict(&x_te)?;
        fold_accs.push(accuracy_score(&y_te, &p)?);
    }
    let mean_acc = fold_accs.iter().sum::<f64>() / fold_accs.len() as f64;
    println!("=== 5-Fold Stratified CV (Accuracy) ===");
    print!("Fold scores: ");
    for a in &fold_accs {
        print!("{a:.4}  ");
    }
    println!();
    println!("Mean CV accuracy: {mean_acc:.4}");

    Ok(())
}
