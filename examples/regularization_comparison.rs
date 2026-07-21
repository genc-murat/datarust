//! Ridge (L2) ile Lasso (L1) regularizasyon karşılaştırması.
//!
//! Senaryo: 8 özellikli bir veri setinde sadece 3 özellik anlamlı, biri
//! kolinear (LinearRegression'ı zorlayacak), diğerleri saf gürültü. Ridge ve
//! Lasso'nun farklı alpha değerlerinde nasıl davrandığını karşılaştırır:
//! Ridge tüm katsayıları küçültür (ama sıfırlamaz), Lasso gereksizleri tam
//! sıfıra indirerek örtük özellik seçimi yapar.
//!
//! Çalıştırma: `cargo run --example regularization_comparison`

use datarust::linear_model::{Lasso, Ridge};
use datarust::traits::{Predictor, Regressor};
use datarust::Matrix;

/// Basit deterministik PRNG (xorshift64) — tekrarlanabilir veri için.
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
    fn normal(&mut self, sigma: f64) -> f64 {
        let u = self.next_f64();
        let v = self.next_f64();
        sigma * (u.ln() * -2.0).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Bir fitted modelin R²'sini ve sıfır olmayan katsayı sayısını hesapla.
struct FitSummary {
    r2: f64,
    nonzero: usize,
    coefs: Vec<f64>,
}

/// Jenerik yardımcı: herhangi bir Regressor benzeri (coef/score) için
/// özet üret. Lasso ve Ridge'in `score` ve `coef` metodları aynı imzaya sahip.
trait Inspect {
    fn coef(&self) -> &[f64];
    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64, datarust::error::DatarustError>;
}

impl Inspect for Ridge {
    fn coef(&self) -> &[f64] {
        self.coef()
    }
    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64, datarust::error::DatarustError> {
        Regressor::score(self, x, y)
    }
}

impl Inspect for Lasso {
    fn coef(&self) -> &[f64] {
        self.coef()
    }
    fn score(&self, x: &Matrix, y: &[f64]) -> Result<f64, datarust::error::DatarustError> {
        // Lasso `score` inherent metodu — Regressor trait değil ama imza aynı.
        Lasso::score(self, x, y)
    }
}

fn summarize(
    model: &dyn Inspect,
    x: &Matrix,
    y: &[f64],
) -> Result<FitSummary, Box<dyn std::error::Error>> {
    let coefs = model.coef().to_vec();
    let nonzero = coefs.iter().filter(|c| c.abs() > 1e-10).count();
    Ok(FitSummary {
        r2: model.score(x, y)?,
        nonzero,
        coefs,
    })
}

fn print_coefs(coefs: &[f64]) -> String {
    let parts: Vec<String> = coefs
        .iter()
        .map(|c| {
            if c.abs() < 1e-10 {
                "   0.00".to_string()
            } else {
                format!("{:7.3}", c)
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── 1. Sentetik veri üret ──────────────────────────────────────────
    // 8 özellik:
    //   x0, x1, x2 → anlamlı (gerçek katsayılar 2, 3, -1)
    //   x3         → anlamlı (katsayı 1)
    //   x4, x5     → saf gürültü (katsayı 0)
    //   x6         → x0 + x1 + küçük gürültü (kolinear — X'X singülerleşir)
    //   x7         → saf gürültü (katsayı 0)
    let true_coef = [2.0, 3.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let n = 150;
    let mut rng = Rng::new(99);
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let x0 = rng.normal(1.0);
        let x1 = rng.normal(1.0);
        let x2 = rng.normal(1.0);
        let x3 = rng.normal(1.0);
        let x4 = rng.normal(1.0);
        let x5 = rng.normal(1.0);
        let x6 = x0 + x1 + rng.normal(0.05); // kolinear
        let x7 = rng.normal(1.0);
        let row = vec![x0, x1, x2, x3, x4, x5, x6, x7];
        let target: f64 = row
            .iter()
            .zip(true_coef.iter())
            .map(|(xi, ci)| xi * ci)
            .sum::<f64>()
            + rng.normal(0.5);
        rows.push(row);
        y.push(target);
    }
    let x = Matrix::new(rows)?;
    println!("=== Ridge (L2) vs Lasso (L1) Karşılaştırması ===");
    println!("Veri: {n} örnek, {} özellik", x.ncols());
    println!("Gerçek katsayılar: {}", print_coefs(&true_coef));
    println!("Not: x6 = x0 + x1 (kolinear) — x4, x5, x7 saf gürültü.\n");

    // ── 2. Ridge ile alpha taraması ────────────────────────────────────
    // Ridge (L2) cezası ||β||²'dir; büyük alpha tüm katsayıları serbestçe
    // küçültür ama hiçbirini tam sıfıra indirmez. Kolinearite yüzünden
    // LinearRegression burada singüler olur; Ridge her zaman çözülür.
    println!("── Ridge (L2): α ||β||² cezası ──");
    println!(
        "{:<8} {:<8} {:<48} {:<10}",
        "Alpha", "R²", "Katsayılar", "Nonzero"
    );
    for &alpha in &[0.01, 1.0, 100.0] {
        let mut m = Ridge::new().with_alpha(alpha);
        m.fit(&x, &y)?;
        let s = summarize(&m, &x, &y)?;
        println!(
            "{:<8.2} {:<8.4} {:<48} {:<10}",
            alpha,
            s.r2,
            print_coefs(&s.coefs),
            s.nonzero
        );
    }
    println!(
        "→ Ridge tüm katsayıları küçültür ama gürültü özelliklerini (x4,x5,x7) sıfıra indirmez.\n"
    );

    // ── 3. Lasso ile alpha taraması ────────────────────────────────────
    // Lasso (L1) cezası ||β||₁'dir; soft-thresholding bazı katsayıları TAM
    // sıfıra iter — bu örtük özellik seçimi sağlar.
    println!("── Lasso (L1): α ||β||₁ cezası ──");
    println!(
        "{:<8} {:<8} {:<48} {:<10}",
        "Alpha", "R²", "Katsayılar", "Nonzero"
    );
    for &alpha in &[0.01, 0.5, 5.0] {
        let mut m = Lasso::new().with_alpha(alpha).with_max_iter(2000);
        m.fit(&x, &y)?;
        let s = summarize(&m, &x, &y)?;
        println!(
            "{:<8.2} {:<8.4} {:<48} {:<10}",
            alpha,
            s.r2,
            print_coefs(&s.coefs),
            s.nonzero
        );
    }
    println!("→ Lasso büyük alpha'da gürültü özelliklerini (x4,x5,x7) TAM sıfıra iter = otomatik özellik seçimi.\n");

    // ── 4. Özet yorum ──────────────────────────────────────────────────
    println!("=== Yorum ===");
    println!(
        "• Ridge:  hiçbir katsayı sıfırlanmaz, ama α büyüdükçe hepsi küçülür. Tahmin kararlıdır."
    );
    println!("• Lasso:  gereksiz özellikler dışlanır (sparse model). Yorumlanabilirlik ve özellik seçimi için ideal.");
    println!("• Kolinearite (x6 ≈ x0+x1): LinearRegression burada singüler olurdu; hem Ridge hem Lasso sorunsuz çözer.");

    Ok(())
}
