# Model Context Protocol (MCP) Server

`datarust.dev` provides a **Model Context Protocol (MCP)** server enabling AI agents, LLMs, and coding assistants (Cursor, Claude Desktop, Antigravity, Cline) to directly invoke datarust machine learning, preprocessing, and dataset profiling capabilities.

## MCP Server Endpoint & Manifest

- **MCP Server Endpoint**: `https://datarust.dev/mcp`
- **In-Repository Implementation**: Built in Rust in [`crates/datarust-mcp`](https://github.com/genc-murat/datarust/tree/main/crates/datarust-mcp)
- **MCP Server Card (SEP-1649)**: [`https://datarust.dev/.well-known/mcp/server-card.json`](https://datarust.dev/.well-known/mcp/server-card.json)
- **Agentic Resource Discovery (ARD)**: Indexed in [`https://datarust.dev/.well-known/ard.json`](https://datarust.dev/.well-known/ard.json) under `urn:air:datarust.dev:mcp:datarust`
- **Link Header**: Every response advertises `rel="ard"` and `rel="api-catalog"` discovery headers.

## Available Capabilities & Tools

The datarust MCP server exposes tools across classical machine learning and data profiling:

### 1. `profile_dataset`
Generates a comprehensive statistical profile of a dataset from CSV or raw tabular data.
- **Inputs**: CSV text, table rows, or file reference.
- **Outputs**: Column statistics (mean, std, median, min, max, missingness, cardinality), distribution shape, quality issues, target-leakage hints, and pairwise relationships (Pearson, Cramér's V, point-biserial).

### 2. `preprocess`
Applies tabular preprocessing transformations with zero external dependencies.
- **Encoders**: OneHotEncoder, OrdinalEncoder, TargetEncoder, FrequencyEncoder.
- **Scalers**: StandardScaler, MinMaxScaler, RobustScaler, Normalizer.
- **Imputers**: SimpleImputer (mean, median, most_frequent, constant).

### 3. `train_model`
Trains classical estimators on feature matrix `X` and target vector `y` with exact scikit-learn parity.
- **Models**: LinearRegression, Ridge, Lasso, LogisticRegression, DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor, KMeans.
- **Outputs**: Model coefficients, tree structure, training metrics, and serialization payloads.

### 4. `evaluate_metrics`
Computes regression metrics (MSE, RMSE, MAE, R², max error) or classification metrics (accuracy, F1, precision, recall, confusion matrix, ROC-AUC, log loss).

## Client Configuration

### Registry Listings & Direct Installation

The datarust MCP server is listed in major MCP directories:

- **Smithery Registry**: [`https://smithery.ai/server/@genc-murat/datarust`](https://smithery.ai/server/@genc-murat/datarust)
  ```sh
  npx -y @smithery/cli install @genc-murat/datarust --client claude
  ```
- **mcp.so Registry**: [`https://mcp.so/server/datarust`](https://mcp.so/server/datarust)
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
On the `datarust.dev` website, client-side WebMCP is registered via `navigator.modelContext` so browser agents can invoke datarust estimators directly in-page.
