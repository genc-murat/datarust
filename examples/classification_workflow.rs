//! Sınıflandırma iş akışı: karışık (sayısal + kategorik) veri → ön işleme →
//! lojistik regresyon → sınıflandırma metrikleri → threshold ayarı →
//! stratified K-fold çapraz doğrulama.
//!
//! Senaryo: Müşteri churn (kaybı) tahmini. Tenure (ay), aylık ödeme, yaş ve
//! sözleşme tipi gözlemlerinden müşterinin ayrılıp ayrılmayacağını (1/0) tahmin et.
//!
//! Çalıştırma: `cargo run --example classification_workflow`

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

/// Basit deterministik PRNG (xorshift64) — tekrarlanabilir sentetik veri için.
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
    // ── 1. Sentetik churn verisi üret ──────────────────────────────────
    // Basit kural: tenure kısa + aylık ödeme yüksekse → churn riski yüksek.
    let mut rng = Rng::new(2024);
    let n = 200;
    let mut num_rows: Vec<Vec<f64>> = Vec::with_capacity(n); // tenure, monthly_charge, age
    let mut cat_rows: Vec<Vec<&'static str>> = Vec::with_capacity(n); // contract_type
    let mut y: Vec<f64> = Vec::with_capacity(n);
    let contract_types: [&'static str; 3] = ["MonthToMonth", "OneYear", "TwoYear"];

    for _ in 0..n {
        let tenure = rng.next_f64() * 72.0; // 0–72 ay
        let monthly = 20.0 + rng.next_f64() * 100.0; // 20–120 $
        let age = 18.0 + rng.next_f64() * 60.0; // 18–78 yaş
                                                // Sözleşme tipi rasgele; kısa sözleşme churn riskini artırır.
        let ct_idx = (rng.next_f64() * 3.0) as usize;
        let ct = contract_types[ct_idx.min(2)];
        num_rows.push(vec![tenure, monthly, age]);
        cat_rows.push(vec![ct]);

        // Churn olasılığı rasgele değil, gözlemlerden belirlenir:
        // kısa tenure, yüksek ödeme, aylık sözleşme artış; az gürültü ekle.
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
    println!("=== Müşteri Churn Sınıflandırması ===");
    println!(
        "Veri: {n} örnek (%{:.0} pozitif)",
        100.0 * pos as f64 / n as f64
    );
    println!();

    // ── 2. Ön işleme: sayısala StandardScaler, kategorik'e OneHot ──────
    // ColumnTransformer, sklearn'deki gibi farklı sütunlara farklı dönüşüm
    // uygular. Tüm veriyi tek seferde işleyip LogisticRegression'a beslenebilecek
    // sayısal bir Matrix üretir.
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
        "Ön işleme sonrası özellik matrisi: {} × {}",
        x.nrows(),
        x.ncols()
    );

    // ── 3. Lojistik regresyon eğit (tüm veride, basitlik için) ─────────
    // SVD çözücüsü rank-eksik verilerde (örn. kolinear one-hot) daha sağlamdır.
    let mut model = LogisticRegression::new().with_solver(LogisticSolver::Svd);
    model.fit(&x, &y)?;
    let preds = model.predict(&x)?; // {0.0, 1.0} etiketleri
    let proba_pos = model.predict_positive_proba(&x)?; // P(y=1)
    println!("Eğitim tamamlandı ({} IRLS iterasyonu).\n", model.n_iter());

    // ── 4. Sınıflandırma metrikleri ────────────────────────────────────
    let acc = accuracy_score(&y, &preds)?;
    let prec = precision_score(&y, &preds)?;
    let rec = recall_score(&y, &preds)?;
    let f1 = f1_score(&y, &preds)?;
    let cm = confusion_matrix(&y, &preds)?;
    let ll = log_loss(&y, &proba_pos, 1e-15)?;
    println!("=== Sınıflandırma Metrikleri (eğitim verisi) ===");
    println!("Doğruluk (accuracy) : {acc:.4}");
    println!(
        "Kesinlik (precision): {prec:.4}  — pozitif tahmin edilenlerin kaçı gerçekten pozitif"
    );
    println!("Duyarlılık (recall) : {rec:.4}  — gerçek pozitiflerin kaçı yakalandı");
    println!("F1 skoru            : {f1:.4}  — precision ile recall'un harmonik ortalaması");
    println!("Log loss            : {ll:.4}  — olasılık kalibrasyonu (daha düşük = daha iyi)");
    println!(
        "Karmaşıklık matrisi : [[TN={}, FP={}], [FN={}, TP={}]]",
        cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    );
    println!();

    // ── 5. Threshold (eşik) ayarı: precision ↔ recall değiş tokuşu ─────
    // Varsayılan threshold 0.5'tir. Churn gibi "kaçırma" maliyeti yüksek
    // problemlerde threshold'u düşürmek recall'ı artırır (ama precision düşer).
    println!("=== Threshold Karşılaştırması ===");
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
    println!("(Düşük threshold → daha agresif pozitif tahmin → recall ↑, precision ↓)\n");

    // ── 6. Stratified K-fold çapraz doğrulama ──────────────────────────
    // cross_val_score sadece KFold destekler; dengesiz sınıflarda StratifiedKFold
    // tercih edilir. Bu yüzden split döngüsünü elle yürütüp her fold'da modeli
    // yeniden fit ederiz — sınıf oranı her fold'da korunur.
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
    println!("=== 5-Fold Stratified CV (Doğruluk) ===");
    print!("Fold skorları: ");
    for a in &fold_accs {
        print!("{a:.4}  ");
    }
    println!();
    println!("Ortalama CV doğruluğu: {mean_acc:.4}");

    Ok(())
}
