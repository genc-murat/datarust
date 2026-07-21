//! Model kalıcılığı (persistence): eğitilmiş bir SupervisedPipeline'ı JSON'a
//! kaydet, sonra refit'siz yükle ve aynı tahminleri üret.
//!
//! Senaryo: Üretim dağıtımı. Model bir makinede eğitilir, diske (JSON) yazılır,
//! sonra servise yüklenir ve eğitim verisine tekrar erişmeden tahmin üretir.
//!
//! Bu örnek `serde` feature'ı gerektirir:
//!   `cargo run --example model_persistence --features serde`

use datarust::decomposition::{PCA, PCAComponents};
use datarust::linear_model::Ridge;
use datarust::pipeline::Pipeline;
use datarust::scaler::StandardScaler;
use datarust::transformer_kind::TransformerKind;
use datarust::traits::{Predictor, Regressor};
use datarust::Matrix;

#[cfg(not(feature = "serde"))]
fn main() {
    eprintln!(
        "Bu örnek `serde` feature'ını gerektirir.\n\
         Şu komutla çalıştırın:\n  \
         cargo run --example model_persistence --features serde"
    );
}

#[cfg(feature = "serde")]
struct Rng {
    state: u64,
}

#[cfg(feature = "serde")]
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
    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64();
        let v = self.next_f64();
        sigma * (u.ln() * -2.0).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

#[cfg(feature = "serde")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Sentetik regresyon verisi üret ──────────────────────────────
    // 6 özellik; gerçek ilişki ilk 4'üne bağlı, son 2'si gürültü.
    let n = 100;
    let mut rng = Rng::new(7);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let row: Vec<f64> = (0..6).map(|_| rng.normal(1.0)).collect();
        let target = 1.5 * row[0] - 2.0 * row[1] + 0.7 * row[2] + row[3] + rng.normal(0.3);
        rows.push(row);
        y.push(target);
    }
    let x = Matrix::new(rows)?;
    println!("=== Model Kalıcılığı (Persistence) ===");
    println!("Veri: {n} örnek, {} özellik\n", x.ncols());

    // ── 2. SupervisedPipeline kur: scale → PCA → Ridge ─────────────────
    // Bir önişleme zinciri (StandardScaler + PCA) ve son tahminci (Ridge).
    // with_estimator, önişleme Pipeline'ını SupervisedPipeline'a dönüştürür.
    let mut pipeline = Pipeline::new()
        .push("scaler", TransformerKind::StandardScaler(StandardScaler::new()))
        .push("pca", TransformerKind::PCA(PCA::new(PCAComponents::Count(4))))
        .with_estimator(Ridge::new().with_alpha(1.0));

    pipeline.fit(&x, &y)?;
    let preds_original = pipeline.predict(&x)?;
    let r2 = pipeline.score(&x, &y)?;
    println!("Eğitim tamamlandı.");
    println!("Eğitim R²: {r2:.4}");
    println!(
        "Pipeline fitted? {}",
        if pipeline.is_fitted() { "evet" } else { "hayır" }
    );
    println!();

    // ── 3. JSON olarak diske kaydet ────────────────────────────────────
    // save_json, fitted durumu (scaler mean/std, PCA eigvektörleri, Ridge
    // katsayıları) dahil tüm parametreleri pretty JSON'a yazar.
    let path = std::env::temp_dir().join("datarust_model_persistence_demo.json");
    // Önceki bir çalıştırmadan kalabilecek dosyayı temizle.
    let _ = std::fs::remove_file(&path);
    datarust::serialize::save_json(&pipeline, &path)?;
    let file_size = std::fs::metadata(&path)?.len();
    println!(
        "Model kaydedildi: {}",
        path.display()
    );
    println!("Dosya boyutu: {file_size} bayt\n");

    // JSON önizlemesi (ilk ~400 karakter) — fitted parametrelerin serileştiğini gör.
    let json_str = std::fs::read_to_string(&path)?;
    let preview: String = json_str.chars().take(400).collect();
    println!("=== JSON Önizleme ===\n{preview}...\n");

    // ── 4. Modeli yükle ve refit'siz tahmin üret ───────────────────────
    // load_json, orijinal tipi geri yükler. Önemli: yüklenen pipeline
    // `is_fitted() == true` ile gelir — eğitim verisine gerek yok.
    let restored: datarust::pipeline::SupervisedPipeline<Ridge> =
        datarust::serialize::load_json(&path)?;
    println!(
        "Yüklenen model fitted? {}",
        if restored.is_fitted() { "evet" } else { "hayır" }
    );

    // ── 5. Orijinal ve yüklenen modelin tahminlerini karşılaştır ───────
    let preds_restored = restored.predict(&x)?;
    let max_diff = preds_original
        .iter()
        .zip(preds_restored.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("Orijinal vs yüklenen tahminler arası maks. fark: {max_diff:.2e}");
    assert!(
        max_diff < 1e-10,
        "yüklenen model farklı tahmin üretiyor"
    );
    println!("✓ Yüklenen model orijinal ile birebir aynı tahminleri üretiyor.");

    // ── 6. Temizlik ────────────────────────────────────────────────────
    let _ = std::fs::remove_file(&path);
    println!("\nGeçici dosya temizlendi.");

    Ok(())
}
