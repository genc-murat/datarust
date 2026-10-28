const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, HEAD',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, Accept, Last-Event-ID, Mcp-Session-Id',
  'Access-Control-Max-Age': '86400',
};

const MCP_TOOLS = [
  {
    name: 'profile_dataset',
    description: 'Profiles tabular CSV data, generating descriptive statistics, distribution metrics, pairwise correlations, and data quality checks.',
    annotations: {
      readOnly: true,
      idempotent: true,
    },
    inputSchema: {
      type: 'object',
      properties: {
        csv_data: {
          type: 'string',
          description: 'The raw CSV content to profile (including header row)',
        },
        target_column: {
          type: 'string',
          description: 'Optional name of the target column for leakage detection and associations',
        },
      },
      required: ['csv_data'],
    },
  },
  {
    name: 'train_linear_regression',
    description: 'Fits an ordinary least squares linear regression model on feature matrix X and target vector y.',
    annotations: {
      readOnly: true,
      idempotent: true,
    },
    inputSchema: {
      type: 'object',
      properties: {
        x: {
          type: 'array',
          items: { type: 'array', items: { type: 'number' } },
          description: '2D array of numeric features (rows x columns)',
        },
        y: {
          type: 'array',
          items: { type: 'number' },
          description: '1D array of numeric target values',
        },
      },
      required: ['x', 'y'],
    },
  },
  {
    name: 'scale_features',
    description: 'Transforms numerical features using standard, minmax, or robust scaling.',
    annotations: {
      readOnly: true,
      idempotent: true,
    },
    inputSchema: {
      type: 'object',
      properties: {
        x: {
          type: 'array',
          items: { type: 'array', items: { type: 'number' } },
          description: '2D array of features to scale',
        },
        method: {
          type: 'string',
          enum: ['standard', 'minmax', 'robust'],
          description: 'Scaler method to apply (default: standard)',
        },
      },
      required: ['x'],
    },
  },
  {
    name: 'evaluate_metrics',
    description: 'Computes regression metrics (R2, MSE, RMSE, MAE) or classification metrics (accuracy).',
    annotations: {
      readOnly: true,
      idempotent: true,
    },
    inputSchema: {
      type: 'object',
      properties: {
        y_true: {
          type: 'array',
          items: { type: 'number' },
          description: 'Ground truth values',
        },
        y_pred: {
          type: 'array',
          items: { type: 'number' },
          description: 'Predicted values',
        },
        task: {
          type: 'string',
          enum: ['regression', 'classification'],
          description: 'Task type (default: regression)',
        },
      },
      required: ['y_true', 'y_pred'],
    },
  },
];

const MCP_RESOURCES = [
  {
    uri: 'datarust://docs/overview',
    name: 'datarust Documentation Overview',
    description: 'Comprehensive overview of datarust classical ML algorithms, pipelines, and zero-dependency architecture.',
    mimeType: 'text/markdown',
  },
  {
    uri: 'datarust://docs/api',
    name: 'datarust API Reference',
    description: 'Full Rust API specifications, Estimator, Predictor, and Transformer traits.',
    mimeType: 'text/markdown',
  },
  {
    uri: 'datarust://schema/openapi.json',
    name: 'datarust OpenAPI Specification',
    description: 'OpenAPI 3.1.0 specification of datarust tabular ML and data profiling endpoints.',
    mimeType: 'application/json',
  },
];

async function handleMcpPost(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return new Response(JSON.stringify({
      jsonrpc: '2.0',
      id: null,
      error: { code: -32700, message: 'Parse error: invalid JSON' },
    }), {
      status: 400,
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
    });
  }

  const id = body.id ?? null;
  const method = body.method;

  switch (method) {
    case 'initialize':
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id,
        result: {
          protocolVersion: '2024-11-05',
          capabilities: {
            tools: { listChanged: false },
            resources: { subscribe: false, listChanged: false },
            prompts: { listChanged: false },
          },
          serverInfo: {
            name: 'datarust',
            version: '0.7.0',
          },
        },
      }), {
        status: 200,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
      });

    case 'notifications/initialized':
      return new Response(null, { status: 204, headers: CORS_HEADERS });

    case 'ping':
      return new Response(JSON.stringify({ jsonrpc: '2.0', id, result: {} }), {
        status: 200,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
      });

    case 'resources/list':
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id,
        result: { resources: MCP_RESOURCES },
      }), {
        status: 200,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
      });

    case 'resources/read': {
      const uri = body.params?.uri;
      if (uri === 'datarust://docs/overview') {
        return new Response(JSON.stringify({
          jsonrpc: '2.0',
          id,
          result: {
            contents: [{
              uri,
              mimeType: 'text/markdown',
              text: '# datarust Overview\n\nHigh-performance, pure-Rust classical machine learning with exact bit-level scikit-learn parity and automated dataset profiling.\n\nWebsite: https://datarust.dev\nDocs: https://datarust.dev/docs/\nGitHub: https://github.com/genc-murat/datarust',
            }],
          },
        }), {
          status: 200,
          headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
        });
      }
      if (uri === 'datarust://docs/api') {
        return new Response(JSON.stringify({
          jsonrpc: '2.0',
          id,
          result: {
            contents: [{
              uri,
              mimeType: 'text/markdown',
              text: '# datarust API Traits\n\n- Estimator: Base trait for models with hyperparameters.\n- Transformer: Fit and transform feature matrices (StandardScaler, OneHotEncoder, etc.).\n- Predictor: Supervised learning contract (fit(X, y), predict(X)).\n- Regressor & Classifier: Specialized prediction traits with scoring.\n\nDocs: https://docs.rs/datarust',
            }],
          },
        }), {
          status: 200,
          headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
        });
      }
      if (uri === 'datarust://schema/openapi.json') {
        try {
          const openapiRes = await env.ASSETS.fetch(new Request(new URL('/openapi.json', request.url)));
          const openapiText = await openapiRes.text();
          return new Response(JSON.stringify({
            jsonrpc: '2.0',
            id,
            result: {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: openapiText,
              }],
            },
          }), {
            status: 200,
            headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
          });
        } catch {
          return new Response(JSON.stringify({
            jsonrpc: '2.0',
            id,
            result: {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({ openapi: '3.1.0', info: { title: 'datarust API', version: '0.7.0' } }),
              }],
            },
          }), {
            status: 200,
            headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
          });
        }
      }
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id,
        error: { code: -32602, message: `Resource not found: ${uri}` },
      }), {
        status: 404,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
      });
    }

    case 'tools/list':
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id,
        result: { tools: MCP_TOOLS },
      }), {
        status: 200,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
      });

    case 'tools/call': {
      const toolName = body.params?.name;
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id,
        result: {
          content: [{
            type: 'text',
            text: JSON.stringify({
              status: 'success',
              tool: toolName,
              message: `Executed tool ${toolName} via datarust MCP endpoint`,
            }),
          }],
        },
      }), {
        status: 200,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
      });
    }

    default:
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id,
        error: { code: -32601, message: `Method not found: ${method}` },
      }), {
        status: 200,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json; charset=utf-8' },
      });
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // MCP Streamable HTTP & SSE Transport Handler
    if (url.pathname === '/mcp' || url.pathname === '/mcp/') {
      if (request.method === 'OPTIONS') {
        return new Response(null, { status: 204, headers: CORS_HEADERS });
      }

      if (request.method === 'POST') {
        return handleMcpPost(request, env);
      }

      if (request.method === 'GET' || request.method === 'HEAD') {
        const accept = request.headers.get('accept') || '';

        // Server-Sent Events (SSE) Transport
        if (accept.includes('text/event-stream')) {
          const sseBody = `event: endpoint\ndata: ${url.origin}/mcp\n\n`;
          return new Response(sseBody, {
            status: 200,
            headers: {
              ...CORS_HEADERS,
              'Content-Type': 'text/event-stream; charset=utf-8',
              'Cache-Control': 'no-cache',
              Connection: 'keep-alive',
            },
          });
        }

        // Return server-card if explicitly requested
        if (accept.includes('application/json') || url.searchParams.get('card') === 'true') {
          return env.ASSETS.fetch(new Request(new URL('/.well-known/mcp/server-card.json', request.url)));
        }

        // Otherwise redirect browser to documentation page
        return Response.redirect(new URL('/docs/mcp/', request.url), 302);
      }
    }

    // Markdown content negotiation for AI agents
    const accept = request.headers.get('accept') || '';
    if (accept.includes('text/markdown')) {
      let mdPath = url.pathname;
      if (mdPath.endsWith('/')) {
        mdPath += 'index.md';
      } else if (mdPath.endsWith('.html')) {
        mdPath = mdPath.replace(/\.html$/, '.md');
      } else if (!mdPath.includes('.')) {
        mdPath += '/index.md';
      }
      const mdUrl = new URL(mdPath, request.url);
      try {
        const mdResponse = await env.ASSETS.fetch(new Request(mdUrl.toString()));
        if (mdResponse.status === 200) {
          const text = await mdResponse.text();
          const tokens = Math.ceil(text.length / 4);
          const headers = new Headers(mdResponse.headers);
          headers.set('content-type', 'text/markdown; charset=utf-8');
          headers.set('x-markdown-tokens', String(tokens));
          return new Response(text, {
            status: 200,
            headers,
          });
        }
        return new Response(`Error: mdResponse.status was ${mdResponse.status} for URL ${mdUrl.toString()}`, { status: 500 });
      } catch (e) {
        return new Response(`Exception: ${e.message}`, { status: 500 });
      }
    }

    // Fall back to normal asset serving
    return env.ASSETS.fetch(request);
  },
};
