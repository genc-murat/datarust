//! Uçtan uca regresyon iş akışı: veri üretimi → train/test split → ölçekleme
//! (sadece train'de fit) → Ridge eğitimi → test R² → K-fold çapraz doğrulama.
//!
//! Senaryo: Ev fiyatı tahmini. Alan (m²), oda sayısı ve bina yaşı gözlemlerinden
//! fiyatı tahmin et. Gerçek ilişki `y = 3·alan + 2·oda − 1.5·yas + 50 + gürültü`.
//!
//! Çalıştırma: `cargo run --example regression_workflow`

use datarust::linear_model::Ridge;
use datarust::metrics::regression::r2_score;
use datarust::model_selection::{cross_val_score, KFold};
use datarust::scaler::StandardScaler;
use datarust::traits::{Predictor, Transformer};
use datarust::Matrix;

/// Basit deterministik PRNG (xorshift64) — sabit seed ile her çalıştırmada
/// aynı gürültü dizisini üretir, böylece sonuçlar tekrarlanabilir olur.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    /// [0, 1) aralığında üniform rasgele sayı.
    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // Üst bitleri kullanarak [0,1) üret.
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Ortalama 0, standart sapma `sigma` olan yaklaşık normal dağılım
    /// (Box–Muller'in basitleştirilmiş hâli).
    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64();
        let v = self.next_f64();
        sigma * (u.ln() * -2.0).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Sentetik veri üret ──────────────────────────────────────────
    // Gerçek (bilinen) katsayılar — modelin bunları geri kazanıp kazanamadığına
    // bakacağız.
    let true_coef = [3.0, 2.0, -1.5];
    let true_intercept = 50.0;

    let n = 120; // örnek sayısı
    let mut rng = Rng::new(42);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        // Özellikler: alan 50–250 m², oda 1–6, yaş 0–50 yıl.
        let area = 50.0 + rng.next_f64() * 200.0;
        let rooms = 1.0 + (rng.next_f64() * 5.0).round();
        let age = rng.next_f64() * 50.0;
        rows.push(vec![area, rooms, age]);
        // Fiyat = gerçek ilişki + ölçülebilir ama sinyalden küçük gürültü.
        // Sinyal aralığı ~600 birim; gürültü std ~80 → sinyal baskın, model
        // yüksek R² alır ama mükemmel değildir (gerçekçi).
        let price = true_intercept
            + true_coef[0] * area
            + true_coef[1] * rooms
            + true_coef[2] * age
            + rng.normal(80.0);
        y.push(price);
    }
    let x = Matrix::new(rows)?;
    println!("=== Ev Fiyatı Regresyonu ===");
    println!("Veri: {n} örnek, {} özellik (alan, oda, yaş)", x.ncols());
    println!("Gerçek katsayılar: {:?}", true_coef);
    println!("Gerçek intercept:  {true_intercept}\n");

    // ── 2. Train/test split (deterministik seed ile) ───────────────────
    let (x_tr, x_te, y_tr, y_te) =
        datarust::model_selection::TrainTestSplit::new()
            .with_test_size(0.25)
            .with_random_state(7)
            .split(&x, &y)?;
    println!(
        "Split: {} train / {} test\n",
        x_tr.nrows(),
        x_te.nrows()
    );

    // ── 3. Ölçekleme — fit SADECE train'de (veri sızıntısını önler) ────
    // StandardScaler'ı eğitim verisine fit et, sonra aynı parametrelerle
    // hem train'i hem test'i dönüştür. Test istatistiklerini fit'e katmak
    // modelin gerçek genelleme performansını abartır (data leakage).
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_tr)?;
    let x_tr_s = scaler.transform(&x_tr)?;
    let x_te_s = scaler.transform(&x_te)?;

    // ── 4. Ridge regresyonu eğit ───────────────────────────────────────
    let mut model = Ridge::new().with_alpha(1.0);
    model.fit(&x_tr_s, &y_tr)?;

    println!("=== Eğitilmiş Ridge (alpha=1.0) ===");
    println!("Öğrenilen katsayılar: {:?}", model.coef());
    println!("Öğrenilen intercept:  {:.4}", model.intercept());
    // Not: katsayılar ölçeklenmiş özellik uzayında olduğundan true_coef ile
    // birebir aynı olmaz; işaret ve görece büyüklükler korunur.
    println!();

    // ── 5. Test performansı ────────────────────────────────────────────
    let preds_te = model.predict(&x_te_s)?;
    let r2_te = r2_score(&y_te, &preds_te)?;
    // Regressor::score da R² döner — tutarlılığı gösterelim.
    let r2_score_method = model.score(&x_te_s, &y_te)?;
    println!("=== Test Performansı ===");
    println!("R² (r2_score fn) : {r2_te:.4}");
    println!("R² (model.score) : {r2_score_method:.4}");
    println!("(R² 1.0'a ne kadar yakınsa o kadar iyi; 0 = ortalama tahmini kadar.)\n");

    // ── 6. K-fold çapraz doğrulama ─────────────────────────────────────
    // Tek bir train/test spliti şansa bağlı olabilir. K-fold CV, veriyi K
    // parçaya bölüp her parçayı sırayla test eder ve daha sağlam bir tahmin
    // verir. cross_val_score her fold'da modeli klonlayıp yeniden fit eder;
    // bu yüzden ölçeklenMEMİŞ ham X'i veriyoruz (her fold kendi içinde
    // StandardScaler'ı pipeline ile bekleyebilir, ama burada sade tutuyoruz).
    let cv = KFold::new().with_n_splits(5).with_shuffle(true).with_random_state(1);
    let scores = cross_val_score(
        &Ridge::new().with_alpha(1.0),
        &x,
        &y,
        &cv,
        r2_score,
    )?;
    let mean_r2 = scores.iter().sum::<f64>() / scores.len() as f64;
    println!("=== 5-Fold Çapraz Doğrulama ===");
    print!("Fold R²'leri: ");
    for s in &scores {
        print!("{s:.4}  ");
    }
    println!();
    println!("Ortalama CV R²: {mean_r2:.4}");

    Ok(())
}
