# datarust workspace

This repository is a **Cargo workspace** hosting the datarust ecosystem: a
collection of independent, separately-versioned crates for classical
data-science and machine-learning workloads in Rust.

## Crates

| Crate | Path | Description |
|---|---|---|
| **[datarust]** | [`crates/datarust/`](crates/datarust) | Scikit-learn-style preprocessing and classical ML in Rust. Standard/minmax/robust scalers, encoders, imputers, PCA, linear models, clustering, pipelines — zero external dependencies by default. |
| **[datarust-profile]** | [`crates/datarust-profile/`](crates/datarust-profile) | One-call data profiling and data-quality reports. Column statistics, distribution shape, pairwise relationships (Pearson, Cramér's V, point-biserial), target-leakage hints, quality findings, HTML/JSON output. |

Each crate has its own `README.md`, `CHANGELOG.md`, version, and release track,
and is published independently to crates.io.

## Documentation

- **Website:** <https://datarust.dev>
- **Docs (book):** [`book/`](book) — rendered at <https://datarust.dev/docs/>
- **Blog:** [`blog/`](blog) — release stories and field notes
- **Architecture:** [`crates/datarust/ARCHITECTURE.md`](crates/datarust/ARCHITECTURE.md)
- **Roadmap:** [`crates/datarust/ROADMAP.md`](crates/datarust/ROADMAP.md)

## Working in this workspace

```sh
# Build / test every crate
cargo build --workspace
cargo test  --workspace

# Work on a single crate
cargo test -p datarust --all-features
cargo test -p datarust-profile --features serde

# Build the static docs/blog site (Node 22+)
npm install
npm run build && npm run check
```

The workspace root `Cargo.toml` carries a `[patch.crates-io]` entry that points
`datarust-profile` at the local source tree of `datarust` during development.
This patch is ignored by `cargo publish`, so published crates depend on the
crates.io release as normal.

## License

MIT — see each crate's `LICENSE`.
