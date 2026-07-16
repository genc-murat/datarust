# datarust: GitHub Pages Dokümantasyon Sitesi (mdBook + CI)

## Stratejik Gerekçe

datarust'un README'i 1447 satır — etkili ama bir landing page değil. crates.io benimseme için profesyonel bir dokümantasyon sitesi, bir ML kütüphanesini "ciddi" kılan kritik farklılaştırıcı. mdBook, Rust ekosisteminin standardı (Rust'un kendi The Book'u,mdBook Documentation — Rust kullanıcılarına aşina). CI ile otomatik deploy: main'e push → site güncel.

## Mimari

### Araç: mdBook
- Rust tabanlı, tek ikili, `cargo install mdbook` ile kurulur
- Markdown kaynak → statik HTML site
- `SUMMARY.md` içindekiler tablosu (nav) → sol kenar çubuğu
- Tema, arama, sözdizimi vurgulama yerleşik
- `mdbook serve` ile yerel önizleme

### Site Yapısı (`book/` dizini)

```
book/
├── book.toml              # mdBook config (başlık, tema, çıktı)
├── theme/
│   └── index.hbs          # (opsiyonel) özelleştirilmiş ana sayfa şablonu
└── src/
    ├── SUMMARY.md         # İçindekiler (sol nav tanımı)
    ├── README.md          # Ana sayfa (landing — hero, features tablosu, hızlı başlangıç)
    ├── quickstart.md      # 5 dakikada ilk pipeline
    ├── installation.md    # Cargo.toml, feature bayrakları, MSRV
    ├── concepts.md        # Core Concepts: Matrix, Transformer/Regressor traits, error handling
    ├── guide/             # Modül rehberleri (her modül ayrı sayfa)
    │   ├── scalers.md     # 9 scaler, formüller, örnekler
    │   ├── encoders.md    # 5 encoder + sparse output
    │   ├── imputers.md    # SimpleImputer + KnnImputer
    │   ├── decomposition.md # PCA + TruncatedSVD + randomized SVD
    │   ├── linear-models.md # LinearRegression + Ridge + Lasso + LogisticRegression
    │   ├── metrics.md     # Regression + classification metrics
    │   ├── model-selection.md # train_test_split + KFold + cross_val_score
    │   └── compose.md     # Pipeline + ColumnTransformer
    ├── performance.md     # Benchmark tabloları + "hız nasıl elde edildi"
    ├── comparison.md      # sklearn feature karşılaştırma tablosu
    ├── examples.md        # 4 tam örnek (basic, pipeline, target_encoding, bench_compare)
    ├── architecture.md    # Tasarım felsefesi, modül ağacı, trait hiyerarşisi
    ├── changelog.md       # CHANGELOG.md'nin kopyası (link veya include)
    └── api.md             # docs.rs'ye yönlendirme (API referansı)
```

### İçerik Stratejisi

README'in 1447 satırlık içeriğini **mantıksal sayfalara böl**:
- **Ana sayfa**: Hero (tagline + kod örneği), features tablosu, "neden datarust?" mermesi, hızlı linkler
- **Modül rehberleri**: Her modül için API referansı + örnekler + ne zaman kullanılır
- **docs.rs link**: Tam API referansı docs.rs'de zaten var (tüm feature'lar açık), site bunu tekrarlamaz, link verir

### CI Workflow (`.github/workflows/pages.yml`)

```yaml
name: Deploy docs to GitHub Pages
on:
  push:
    branches: [main]
    paths: ['book/**', '.github/workflows/pages.yml']
permissions:
  contents: read
  pages: write
  id-token: write
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install mdBook
        run: cargo install mdbook --no-default-features
      - name: Build book
        run: mdbook build book
      - uses: actions/upload-pages-artifact@v3
        with:
          path: book/book  # mdBook çıktısı book/book/html
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
    steps:
      - uses: actions/deploy-pages@v4
```

**Tetikleyici:** `main`'e push, sadece `book/` veya workflow değişince → docs build'i gereksiz çalışmaz.

---

## Uygulama Adımları

### Adım 1 — mdBook kurulumu + iskelet
- `cargo install mdbook` (lokal önizleme için)
- `book/` dizini oluştur, `book.toml`, `src/SUMMARY.md`

### Adım 2 — Ana sayfa (src/README.md)
- Hero: tagline + 5 satırlık Quick Start kodu
- Features tablosu (README'den uyarlanmış)
- "Neden datarust?" — sıfır bağımlılık, Rust-native, JSON serializasyon, performans
- Badge'ler: crates.io, docs.rs, CI, license

### Adım 3 — Modül rehber sayfaları (src/guide/*.md)
- Her modül: overview, ne zaman kullanılır, API tablosu, örnekler, docs.rs linki
- README'deki ilgili bölümlerden uyarlanmış (kopyala değil, düzenle)
- Kod örnekleri `rust` blok işaretli, sözdizimi vurgulaması çalışır

### Adım 4 — Performans + karşılaştırma sayfaları
- `performance.md`: Benchmark tabloları (50K×200), "hız nasıl elde edildi"
- `comparison.md`: sklearn feature karşılaştırma matrisi

### Adım 5 — Örnekler + mimari + changelog
- `examples.md`: 4 tam örnek
- `architecture.md`: ARCHITECTURE.md'den uyarlanmış, güncellenmiş modül ağacı
- `changelog.md`: CHANGELOG.md referansı

### Adım 6 — book.toml konfigürasyonu
- Başlık, yazar, dil (en)
- Çıktı: HTML, ayrıca "linkcheck" (opsiyonel)
- Tema: default (Rust'a uygun)
- docs.rs/crates.io linkleri

### Adım 7 — CI workflow
- `.github/workflows/pages.yml` oluştur
- Pages ayarları için README'de kısa talimat (repo Settings → Pages → Source: GitHub Actions)

### Adım 8 — Cargo.toml + README güncellemesi
- `homepage = "https://genc-murat.github.io/datarust/"` ekle
- README'ye docs badge + site linki ekle

---

## Çıktı Özeti

- ✅ Profesyonel, çok sayfalı mdBook dokümantasyon sitesi
- ✅ Tam otomatik CI deploy (main push → Pages)
- ✅ 12+ sayfa: ana sayfa, rehberler, performans, karşılaştırma, örnekler
- ✅ crates.io/docs.rs/CI badge'leri
- ✅ Yerel önizleme (`mdbook serve book`)

**Kapsam dışı:** özel tema/CSS (varsayılan mdBook teması yeterli), interaktif playground, blog. ✓