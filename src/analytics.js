import wasm from '../target/worker/wasm32-unknown-unknown/release/datarust_mcp.wasm';

export const OPERATIONS = new Map([
  ['profile', 'profile_dataset'],
  ['train', 'train'],
  ['scale', 'scale_features'],
  ['evaluate', 'evaluate_metrics'],
]);

const encoder = new TextEncoder();
const decoder = new TextDecoder();
let kernel;

export class InvalidInput extends Error {}

export function executeTool(name, args) {
  if (!kernel) kernel = new WebAssembly.Instance(wasm).exports;
  const input = encoder.encode(JSON.stringify({ name, arguments: args }));
  const pointer = kernel.datarust_input(input.length);
  new Uint8Array(kernel.memory.buffer, pointer, input.length).set(input);
  const size = kernel.datarust_execute();
  const output = new Uint8Array(kernel.memory.buffer, kernel.datarust_output(), size);
  const response = JSON.parse(decoder.decode(output));
  if (response.error !== undefined) throw new InvalidInput(response.error);
  return response.result;
}

export function executeOperation(operation, args) {
  const result = executeTool(OPERATIONS.get(operation), args);
  if (operation === 'profile') {
    const { profile, quality } = result;
    const pearson = profile.relationships?.pearson;
    const correlations = {};
    for (const [i, left] of (pearson?.labels ?? []).entries()) {
      for (let j = i + 1; j < pearson.labels.length; j++) {
        correlations[left + ':' + pearson.labels[j]] = pearson.values[i][j];
      }
    }
    return {
      row_count: profile.n_rows,
      column_count: profile.n_columns,
      columns: profile.columns.map((column) => ({
        ...column,
        inferred_type: column.column_type.toLowerCase(),
        null_count: column.missing_count,
        ...(column.numeric ? { mean: column.numeric.mean, std: column.numeric.std } : {}),
      })),
      quality_warnings: quality.map((item) => item.message),
      correlations,
      relationships: profile.relationships,
      target_column: profile.target_column,
    };
  }
  if (operation === 'scale') {
    return { method: result.method, scaled_x: result.scaled_features, status: 'transformed' };
  }
  if (operation === 'evaluate' && result.task === 'regression') {
    return {
      task: result.task,
      r2: result.r2_score,
      mse: result.mean_squared_error,
      rmse: result.root_mean_squared_error,
      mae: result.mean_absolute_error,
    };
  }
  return result;
}
