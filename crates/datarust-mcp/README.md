# datarust-mcp

Model Context Protocol (MCP) server for [datarust](https://datarust.dev) and [datarust-profile](https://crates.io/crates/datarust-profile).

Provides tabular machine learning, preprocessing, and automated dataset profiling tools over standard stdio JSON-RPC for Claude Desktop, Cursor, and autonomous AI agents.

## Installation

```sh
cargo install datarust-mcp
```

## Running the Server

Run the installed binary:

```sh
datarust-mcp
```

Or from a workspace checkout:

```sh
cargo run --release -p datarust-mcp
```

## Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "datarust": {
      "command": "datarust-mcp"
    }
  }
}
```

From a workspace checkout, use `"command": "cargo"` with
`"args": ["run", "--release", "--manifest-path", "/PATH/TO/datarust/Cargo.toml", "-p", "datarust-mcp"]`
instead.

## Exposed MCP Tools

1. **`profile_dataset`**: Profiles CSV datasets, extracting column stats, distributions, quality issues, and correlations.
2. **`train_linear_regression`**: Fits an ordinary least squares linear regression model with scikit-learn bit-level parity.
3. **`scale_features`**: Scales numerical features via `StandardScaler`, `MinMaxScaler`, or `RobustScaler`.
4. **`evaluate_metrics`**: Computes regression (R², MSE, RMSE, MAE) and classification (accuracy) metrics.

## License

MIT
