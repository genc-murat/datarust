# Release Plan — next cycle (0.8 / 0.9)

Status: v0.8 split approved by the user 2026-11-20; roadmaps renumbered the
same day. Living document; update checkboxes as work lands.
Canonical algorithm roadmaps stay in `crates/datarust/ROADMAP.md` and
`crates/datarust-profile/ROADMAP.md`; this file sequences the *releases*.

## Snapshot

| Crate | Current | Published | Next |
|---|---|---|---|
| `datarust` | 0.7.0 (2026-09-14) | crates.io ✅ | **0.8.0** |
| `datarust-profile` | 0.4.0 (2026-09-13) | crates.io ✅ | **0.5.0** |
| `datarust-mcp` | 0.7.0 (unpublished placeholder) | crates.io ❌ (404) | **0.1.0** (first publish) |

Since the 0.7.0 release commit (`70fb8cc`), 20 commits landed — almost all
site/agent-ecosystem work (live MCP HTTP/SSE worker, OpenAPI 3.1, A2A agent
card, Web Bot Auth, SEO, developer portal, AdSense). That work is already
deployed and does **not** gate the crate releases. The only unreleased crate
code is the new `datarust-mcp` crate itself plus clippy/doc fixes.

Hygiene gaps found:

- **`v0.7.0` git tag is missing** (latest tag is `v0.6.5`). Backfill on
  `70fb8cc` before tagging anything new.
- `datarust-mcp` cannot be published as-is: both path dependencies lack
  `version` keys, and the crate has no `LICENSE` or `CHANGELOG.md`.
- `crates/datarust-profile/ROADMAP.md` header still says "Where we are today
  (v0.3.0)" and the v0.5 section body says "v0.4 promotes" — both stale.

## Key scope decision (approved 2026-11-20): split roadmap v0.8 into two releases

The roadmap's v0.8 ("Model selection & text") bundles hyperparameter search,
curves, three text vectorizers, sparse linear algebra, KNN, and naive Bayes —
plus the unfinished v0.7 items (`ExtraTrees*`, `Bagging`). That is roughly 3×
the v0.7 scope. Decision (approved): cut it into two themed releases, matching the
0.6 / 0.7 cadence:

- **0.8.0 — Model selection & ensemble completion.** Closes the v0.7 theme
  and ships the most-requested missing piece (`GridSearchCV`). No new
  numerical kernels required; everything builds on the existing tree kernel,
  `Params` trait, `cross_val_score`, and `KnnImputer` distances.
- **0.9.0 — Text & sparse.** Sparse CSR arithmetic first (it is the
  prerequisite kernel), then the vectorizers and naive Bayes. This is where
  the genuinely new numerics live, so it gets its own release and parity
  budget.

Roadmap renumbering (applied 2026-11-20): old v0.9 "Depth & breadth" →
**v0.10**; v1.0 "Stability" unchanged. `crates/datarust/ROADMAP.md` and
`book/src/roadmap.md` updated; the profile roadmap keeps its own numbering.

Cut-line rule: if an item slips, move the **whole item** to the next minor.
Never ship an estimator without its sklearn-parity fixtures.

---

## Track A — `datarust` 0.8.0 "Model selection & ensemble completion"

### M1 — Close the v0.7 ensemble theme

- [ ] `ExtraTreesClassifier` / `ExtraTreesRegressor` in `src/ensemble.rs`.
      Random-split variant of the existing CART kernel: per-feature uniform
      threshold draws from the same MT19937 "legacy RandomState" stream the
      forests already clone. Note sklearn 1.6.1 defaults (`max_features`
      differs from RandomForest) — pin behavior with fixtures, not memory.
- [ ] `Bagging` meta-estimator, generic over any `Predictor` (bootstrap +
      optional `max_samples` / `max_features`). First cut mirrors the forest
      release contract: no OOB scoring, no sample weights, no parallel fit.
- [ ] Parity fixtures (sklearn 1.6.1 / numpy 2.0.2, seeded) in the
      `tests/tree_fixtures.rs` / `tree_parity.rs` style, plus integration and
      proptest invariants (determinism under fixed seed, probability
      normalization, depth/sample constraints).
- [ ] `Params` impls for all four new estimators; book `guide/ensembles.md`
      update.

Size: M (kernel reuse; the work is RNG-stream fidelity + fixtures).

### M2 — Hyperparameter search & diagnostics

- [ ] `Params` for `Pipeline` / `ColumnTransformer` (today only trees,
      forests, and KMeans implement it; linear models carry the methods).
      Nested `step__param` addressing, including inside `ColumnTransformer`
      transformer lists. Settle the estimator-duplication design first:
      `GridSearchCV<E: Estimator + Params + Clone>` vs. rebuild-from-params;
      check `Clone` coverage across estimators before committing.
- [ ] `GridSearchCV` on top of `Params` + `cross_val_score`
      (`src/model_selection/cross_val.rs:35`): exhaustive product,
      `cv_results_`-style table, tie-break to first best candidate, `refit`
      best-estimator behavior, pipeline param grids.
- [ ] `RandomizedSearchCV`: `n_iter` sampling over param distributions via
      the shared `Rng` (`src/rng.rs`). Bit-exact stream parity with sklearn's
      `ParameterSampler` is a stretch goal; document the sampling contract
      either way.
- [ ] `validation_curve` and `learning_curve` diagnostics reusing the CV loop.
- [ ] Parity fixtures for search ordering/tie-breaks and curve values;
      integration tests with a `Pipeline` grid (`scaler__with_mean`-style
      addressing).
- [ ] Book: extend `guide/model-selection.md`; runnable example in
      `examples/`.

Size: L (the API-design milestone of the release; do the `Params`/clone
design spike before writing search code).

### M3 — Neighbors

- [ ] `src/neighbors/`: `KNeighborsClassifier` / `KNeighborsRegressor`,
      brute-force first, reusing the distance infrastructure behind
      `KnnImputer`. `weights = uniform | distance`, sklearn tie-breaking.
      Ball/KD-tree acceleration is explicitly post-0.8.
- [ ] Parity fixtures (iris / diabetes, tie-heavy cases), proptest
      invariants, book `guide/neighbors.md` + SUMMARY entry.

Size: S–M.

**→ Cut 0.8.0** when M1–M3 are green (release checklist below).

---

## Track B — `datarust` 0.9.0 "Text & sparse"

### M4 — Sparse CSR arithmetic (prerequisite kernel)

- [ ] Sparse × dense matmul and sparse + sparse addition on the existing
      read-only CSR `SparseMatrix`. Today it has no linear algebra; TF-IDF
      output and large one-hot features need it.
- [ ] Decide the `matrixmultiply`-feature interaction (dense backing store
      vs. pure-Rust fallback) and benchmark both; keep the default build
      zero-dependency.
- [ ] Property tests against dense equivalents (bit-tolerance policy
      documented), allocation-count regression tests in the 0.6.6 style.

Size: L.

### M5 — Text feature extraction

- [ ] `src/feature_extraction/text/`: `CountVectorizer` (tokenizer,
      `ngram_range`, `min_df`/`max_df`, vocabulary, lowercase),
      `TfidfTransformer` (`smooth_idf`, `sublinear_tf`, L2 norm),
      `TfidfVectorizer`, `HashingVectorizer` (stateless).
- [ ] Zero-dep parity risks to solve here: sklearn's default token pattern
      `(?u)\b\w\w+\b` needs a hand-rolled tokenizer (propose ASCII-word-first
      with documented divergence, or a compact Unicode word-char table), and
      `HashingVectorizer` parity needs a pure-Rust MurmurHash3-32
      implementation (~100 lines, well-specified).
- [ ] Parity fixtures against sklearn 1.6.1 vectorizer output (vocabulary
      order, idf values, hashed indices/signs); sparse-output integration
      with M4.
- [ ] Book `guide/text.md` + SUMMARY; document classification example.

Size: L.

### M6 — Naive Bayes

- [ ] `src/naive_bayes/`: `GaussianNB` (`var_smoothing`) and `MultinomialNB`
      (Laplace `alpha`) — the natural pairing with M5. `partial_fit` is a
      stretch goal, not a gate.
- [ ] Parity fixtures (`feature_log_prob_`, `class_log_prior_`, predictions),
      book `guide/naive-bayes.md`.

Size: S–M.

**→ Cut 0.9.0** when M4–M6 are green. Then renumber old v0.9 catalog work as
v0.10 in the roadmaps.

---

## Track C — `datarust-profile` 0.5.0 "Missingness & comparison" (parallel)

Independent of Tracks A/B (builds on `datarust::stats`, no new 0.8 APIs
needed). Can run alongside M1–M3; publish after `datarust` 0.8.0 so the
workspace `[patch.crates-io]` and dependency versions stay consistent.

- [ ] Missingness analysis: per-column fractions (present) + missing-pattern
      co-occurrence summary; `QualityKind::MissingPattern` finding.
- [ ] `compare_profiles(a, b) -> DatasetDiff`: schema diff
      (added/removed/renamed), numeric drift via PSI, categorical
      frequency-table distance; `QualityKind::Drift` over threshold.
- [ ] `compare_reports` HTML/JSON renderer highlighting drifted columns
      (pure-CSS convention, no JS).
- [ ] Parity tests: PSI against hand-computed references; pandas/ydata
      reference values where applicable. Streaming-mode comparison is out of
      scope for 0.5.
- [ ] Fix the stale roadmap header (v0.3.0 → v0.4.0) and the "v0.4 promotes"
      sentence in the v0.5 section; check the v0.5 boxes as items land.
- [ ] Book `profile/` pages + `blog/medium-profile-v0.5.md`.

Size: M.

---

## Track D — `datarust-mcp` 0.1.0 (first publish, early)

Unblock and ship this **before** 0.8.0 — it is independent, and the ecosystem
visibility (Smithery listing, agent configs) already points at it.

- [ ] Packaging fixes (publish blockers):
  - Add `version` keys to both path deps:
    `datarust = { path = "../datarust", version = "0.7", features = ["serde"] }`
    and `datarust-profile = { path = "../datarust-profile", version = "0.4", features = ["serde", "csv"] }`
    (bump to `"0.8"` / `"0.5"` after those releases).
  - Add `LICENSE` (MIT, same text as sibling crates) and `CHANGELOG.md`.
  - Verify `cargo publish --dry-run -p datarust-mcp` (the workspace
    `[patch.crates-io]` is ignored on publish).
- [ ] Version decision: publish as **0.1.0** — the current 0.7.0 is an
      unpublished placeholder; 0.1.0 signals a first release honestly.
- [ ] README: `cargo install datarust-mcp` instructions alongside the
      cargo-run setup; verify the Smithery listing resolves the published
      crate.
- [ ] Follow-up (0.2.0, after 0.8.0): expose the new APIs as tools
      (`train_random_forest` / tree tools, `grid_search`), and keep tool
      parity with the live HTTP/SSE worker in `src/worker.js`.

Size: S.

---

## Release checklist (per crate cut)

Observed conventions from the 0.6.x / 0.7.0 releases:

1. `cargo fmt --all -- --check`, clippy `-D warnings` across the CI feature
   matrix, `cargo test --workspace`, MSRV 1.70 job — all green.
2. Parity fixtures committed and pinned (sklearn 1.6.1 / numpy 2.0.2);
   proptest regressions clean.
3. Bump `Cargo.toml` version; update `version` keys in dependent crates.
4. `CHANGELOG.md`: `[Unreleased]` → `[x.y.z] - YYYY-MM-DD`, Keep a Changelog
   format, parity claims stated precisely (bit-exact vs. tolerance).
5. `ROADMAP.md`: tick boxes, add "Recently shipped" section; keep the book
   mirror (`book/src/roadmap.md`, `book/src/changelog.md`) in sync.
6. Refresh README feature/benchmark tables where the release changes them.
7. Book: new/updated guide pages + `SUMMARY.md`; `mdbook build book` clean.
8. Blog release post (`blog/medium-v0.8.md` naming pattern); numbered field
   note if there is a performance story.
9. Site: `npm run build && npm run check`, deploy worker (`wrangler deploy`).
10. Publish in dependency order: `datarust` → `datarust-profile` →
    `datarust-mcp`; dry-run each first.
11. Tag `vX.Y.Z` (backfill the missing `v0.7.0` tag on `70fb8cc` first) and
    push tags; verify crates.io + docs.rs renders.

## Risks

- **RNG-stream fidelity** (ExtraTrees, RandomizedSearchCV): sklearn's
  MT19937 draw order is unforgiving. Fixture-first: generate references
  before implementing, same workflow that made the 0.7 trees bit-exact.
- **Zero-dep text parity**: tokenizer Unicode semantics and MurmurHash3 are
  the two places where "pure Rust" can silently diverge from sklearn.
  Decide and document the contract in M5 before writing vectorizers.
- **Sparse kernel performance** without BLAS: benchmark against the dense
  path early in M4; the `matrixmultiply` feature must remain strictly
  opt-in.
- **MSRV 1.70**: no recent std APIs in new code; CI enforces this.
- **Scope creep / solo-dev bandwidth**: the split above exists to keep each
  release shippable. Apply the cut-line rule ruthlessly; GridSearchCV (M2)
  is the one item worth protecting in 0.8.0.

## Suggested order of work

M0 hygiene (tag backfill, roadmap header fixes, MCP packaging) → Track D
publish → M1 → M2 → M3 → **0.8.0 cut** → Track C in parallel anywhere after
M0 → **profile 0.5.0 cut** → M4 → M5 → M6 → **0.9.0 cut** → MCP 0.2.0 tools.
