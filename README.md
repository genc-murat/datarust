# datarust workspace

This repository is a **Cargo workspace** hosting the datarust ecosystem: a
collection of independent, separately-versioned crates for classical
data-science and machine-learning workloads in Rust.

## Crates

| Crate | Path | Description |
|---|---|---|
| **[datarust]** | [`crates/datarust/`](crates/datarust) | Scikit-learn-style preprocessing and classical ML in Rust. Standard/minmax/robust scalers, encoders, imputers, PCA, linear models, trees/forests/ExtraTrees/Bagging (bit-exact sklearn parity), k-NN, GridSearchCV/RandomizedSearchCV, CV curves, clustering, pipelines — zero external dependencies by default. |
| **[datarust-profile]** | [`crates/datarust-profile/`](crates/datarust-profile) | One-call data profiling and data-quality reports. Column statistics, distribution shape, pairwise relationships (Pearson, Cramér's V, point-biserial), target-leakage hints, quality findings, CSV ingestion, single-pass streaming, a CLI — HTML/JSON/terminal output. |
| **[datarust-mcp]** | [`crates/datarust-mcp/`](crates/datarust-mcp) | Model Context Protocol (MCP) server implementation for Claude Desktop, Cursor, and AI agents. Exposes data profiling, training, and preprocessing over stdio JSON-RPC. |

Each crate has its own `README.md`, `CHANGELOG.md`, version, and release track,
and is published independently to crates.io.

## Model Context Protocol (MCP) Server

This repository contains the official MCP server implementation in [`crates/datarust-mcp`](crates/datarust-mcp).

### Running Locally (Stdio)

Install the published crate and run the binary:

```sh
cargo install datarust-mcp
datarust-mcp
```

Or run from a checkout:

```sh
cargo run --release -p datarust-mcp
```

### Claude Desktop / Cursor Setup

Add to your `claude_desktop_config.json` (using the installed binary):

```json
{
  "mcpServers": {
    "datarust": {
      "command": "datarust-mcp"
    }
  }
}
```

To run from a checkout instead, use `"command": "cargo"` with
`"args": ["run", "--release", "--manifest-path", "/path/to/datarust/Cargo.toml", "-p", "datarust-mcp"]`.

### Available Tools
- `profile_dataset`: Profiles CSV data, computing descriptive stats, distribution shapes, and quality checks.
- `train_linear_regression`: Fits OLS linear regression with exact scikit-learn parity.
- `scale_features`: Feature scaling via StandardScaler, MinMaxScaler, or RobustScaler.
- `evaluate_metrics`: Computes R², MSE, RMSE, MAE, and accuracy metrics.

## Agent Platform Configs & Official Skills

This repository provides pre-configured agent instructions and official Agent Skills for autonomous coding agents:
- **Cursor Rules:** [`.cursorrules`](.cursorrules) & [`.cursor/rules/datarust.mdc`](.cursor/rules/datarust.mdc)
- **Claude Instructions:** [`CLAUDE.md`](CLAUDE.md) & [`.claude/CLAUDE.md`](.claude/CLAUDE.md)
- **Windsurf Rules:** [`.windsurf/rules/rules.md`](.windsurf/rules/rules.md)
- **Copilot Instructions:** [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
- **Agent Plugins Manifest:** [`plugin.json`](plugin.json)
- **Official Skills (skills.sh):** [`skills.json`](skills.json) — self-published skills in [`skills/`](skills/):
  - [`data-profiling`](skills/data-profiling/SKILL.md)
  - [`classical-ml`](skills/classical-ml/SKILL.md)

## Documentation

- **Website:** <https://datarust.dev>
- **Developer Portal:** <https://datarust.dev/developers/>
- **Docs (book):** [`book/`](book) — rendered at <https://datarust.dev/docs/>
- **MCP Server:** <https://datarust.dev/docs/mcp/> — verified listing on [Smithery](https://smithery.ai/server/@genc-murat/datarust)
- **OpenAPI 3.1 Spec:** <https://datarust.dev/openapi.json>
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

## Hosted API and Worker development

The Worker runs the same Rust tools as the stdio server using WebAssembly.
Install the target and run the local Worker tests after building the site.

```sh
rustup target add wasm32-unknown-unknown
npm run build
npm run test:worker
```

The hosted REST API supports profiling, scaling, metric evaluation, and
regression training. See the [OpenAPI contract](site/static/openapi.json).
Clients can submit a job with `POST /v1/jobs`, then poll the returned
`Location` using the private `X-Job-Token` response header. Reuse the token
to submit more jobs to the same collection and list them with `GET /v1/jobs`.
Job records and idempotent responses expire after 30 minutes. Use an
unpredictable `Idempotency-Key` for requests that may be retried.

The hosted Worker exposes MCP at `/mcp` and A2A JSON-RPC 1.0 at `/a2a`.
A2A accepts documentation questions or a data part with `tool` and
`arguments` matching an MCP tool. The deployment uses the SQLite-backed
`API_STATE` Durable Object declared in `wrangler.jsonc`.

To publish an HTTP Message Signatures Directory, configure
`WEB_BOT_AUTH_PRIVATE_JWK` as a Cloudflare Worker secret containing a
private Ed25519 JWK. The directory publishes the public key and signs its
response. The directory is available once the secret is configured.

## License

MIT — see each crate's `LICENSE`.
