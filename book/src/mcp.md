# datarust MCP Server (Model Context Protocol)

`datarust.dev` provides an official **datarust Model Context Protocol (MCP)** server enabling AI agents, LLMs, and coding assistants (Cursor, Claude Desktop, Antigravity, Cline) to directly invoke datarust machine learning, preprocessing, and dataset profiling tools.

## MCP Server Endpoint & Manifest

- **MCP Server Endpoint** `https://datarust.dev/mcp` uses stateless Streamable HTTP with JSON and SSE responses to POST requests.
- **In-Repository Implementation**: Built in Rust in [`crates/datarust-mcp`](https://github.com/genc-murat/datarust/tree/main/crates/datarust-mcp)
- **MCP Server Card (SEP-1649)**: [`https://datarust.dev/.well-known/mcp/server-card.json`](https://datarust.dev/.well-known/mcp/server-card.json)
- **Agentic Resource Discovery (ARD)**: Indexed in [`https://datarust.dev/.well-known/ard.json`](https://datarust.dev/.well-known/ard.json) under `urn:air:datarust.dev:mcp:datarust`

## Available Capabilities & Tools

The datarust MCP server exposes tools across classical machine learning and data profiling:

| Tool | Input | Result |
|---|---|---|
| `profile_dataset` | Header-bearing `csv_data` and optional `target_column` | Column statistics, pairwise relationships, and quality findings |
| `train_linear_regression` | Numeric feature matrix `x` and target vector `y` | Coefficients, intercept, R², MSE, and up to ten sample predictions |
| `scale_features` | Numeric matrix `x` and optional `method` of `standard`, `minmax`, or `robust` | Scaled matrix and dimensions |
| `evaluate_metrics` | Numeric `y_true` and `y_pred` vectors and optional `task` | R², MSE, RMSE, and MAE for regression, or accuracy for classification |

Hosted requests accept up to 256 KiB of JSON. A numeric matrix can contain up to 10,000 rows, 64 columns, and 50,000 values. CSV profiling accepts up to 10,000 rows and 64 columns.

## Client Configuration

### Registry Listings & Direct Installation

The datarust MCP server is listed in major MCP directories:

- **Smithery Registry**: [`https://smithery.ai/server/@genc-murat/datarust`](https://smithery.ai/server/@genc-murat/datarust)
  ```sh
  npx -y @smithery/cli install @genc-murat/datarust --client claude
  ```
- **Agentic Resource Discovery (ARD)**: [`https://datarust.dev/.well-known/ard.json`](https://datarust.dev/.well-known/ard.json)

### Claude Desktop / Cursor
Add the datarust MCP server to your configuration (e.g. `claude_desktop_config.json` or Cursor settings):

```json
{
  "mcpServers": {
    "datarust": {
      "url": "https://datarust.dev/mcp"
    }
  }
}
```

### In-Browser WebMCP
On the `datarust.dev` website, client-side WebMCP registers documentation search, crate information, and navigation via `navigator.modelContext`.
