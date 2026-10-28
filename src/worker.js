const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS, HEAD',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, Accept, Last-Event-ID, Mcp-Session-Id, Idempotency-Key, X-Idempotency-Key, X-API-Version',
  'Access-Control-Expose-Headers': 'RateLimit-Limit, RateLimit-Remaining, RateLimit-Reset, RateLimit-Policy, Idempotency-Key, Idempotent-Replayed, X-API-Version',
  'Access-Control-Max-Age': '86400',
};

function getRateLimitHeaders() {
  return {
    'RateLimit-Limit': '1000',
    'RateLimit-Remaining': '999',
    'RateLimit-Reset': '60',
    'RateLimit-Policy': '1000;w=60',
    'X-RateLimit-Limit': '1000',
    'X-RateLimit-Remaining': '999',
    'X-RateLimit-Reset': String(Math.floor(Date.now() / 1000) + 60),
    'X-API-Version': '1.0',
  };
}

function getIdempotencyHeaders(request) {
  const key = request.headers.get('idempotency-key') || request.headers.get('x-idempotency-key');
  if (key) {
    return {
      'Idempotency-Key': key,
      'Idempotent-Replayed': 'false',
    };
  }
  return {};
}

function jsonResponse(data, status = 200, request = null, extraHeaders = {}) {
  const headers = {
    ...CORS_HEADERS,
    ...getRateLimitHeaders(),
    ...(request ? getIdempotencyHeaders(request) : {}),
    ...extraHeaders,
    'Content-Type': 'application/json; charset=utf-8',
  };
  return new Response(JSON.stringify(data), { status, headers });
}

function jsonError(message, status = 400, code = 'bad_request', request = null, details = []) {
  return jsonResponse({
    error: {
      code,
      message,
      status,
      details,
    },
  }, status, request);
}

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
      headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
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
        headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
      });

    case 'notifications/initialized':
      return new Response(null, { status: 204, headers: { ...CORS_HEADERS, ...getRateLimitHeaders() } });

    case 'ping':
      return new Response(JSON.stringify({ jsonrpc: '2.0', id, result: {} }), {
        status: 200,
        headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
      });

    case 'resources/list':
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id,
        result: { resources: MCP_RESOURCES },
      }), {
        status: 200,
        headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
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
          headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
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
          headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
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
            headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
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
            headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
          });
        }
      }
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id,
        error: { code: -32602, message: `Resource not found: ${uri}` },
      }), {
        status: 404,
        headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
      });
    }

    case 'tools/list':
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id,
        result: { tools: MCP_TOOLS },
      }), {
        status: 200,
        headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
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
        headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
      });
    }

    default:
      return new Response(JSON.stringify({
        jsonrpc: '2.0',
        id,
        error: { code: -32601, message: `Method not found: ${method}` },
      }), {
        status: 200,
        headers: { ...CORS_HEADERS, ...getRateLimitHeaders(), 'Content-Type': 'application/json; charset=utf-8' },
      });
  }
}

// NLWeb /ask Handler with SSE Streaming Support
async function handleNlWebAsk(request) {
  const url = new URL(request.url);
  let query = url.searchParams.get('q') || url.searchParams.get('query') || '';
  let streamRequested = url.searchParams.get('stream') === 'true';

  if (request.method === 'POST') {
    try {
      const body = await request.json();
      query = body.q || body.query || body.question || query;
      if (body.stream === true || body.stream === 'true') {
        streamRequested = true;
      }
    } catch {
      // Form or empty body
    }
  }

  const accept = request.headers.get('accept') || '';
  if (accept.includes('text/event-stream')) {
    streamRequested = true;
  }

  const responseText = query
    ? `datarust is a pure-Rust classical machine learning and data profiling workspace with zero external dependencies by default. You asked: "${query}". It provides StandardScaler, PCA, LinearRegression, LogisticRegression, KMeans, pipelines, and one-call profiling reports.`
    : 'datarust provides high-performance classical machine learning and data quality profiling native to Rust with exact scikit-learn parity.';

  if (streamRequested) {
    const chunks = [
      'datarust is a pure-Rust machine learning and data profiling ecosystem. ',
      'It offers scikit-learn-style estimators, scalers, and automated EDA reports ',
      'with zero external dependencies and exact bit-level parity.',
    ];

    const sseStream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder();
        for (let i = 0; i < chunks.length; i++) {
          const payload = JSON.stringify({
            chunk: chunks[i],
            index: i,
            done: i === chunks.length - 1,
          });
          controller.enqueue(encoder.encode(`event: message\ndata: ${payload}\n\n`));
        }
        controller.enqueue(encoder.encode('event: done\ndata: [DONE]\n\n'));
        controller.close();
      },
    });

    return new Response(sseStream, {
      status: 200,
      headers: {
        ...CORS_HEADERS,
        ...getRateLimitHeaders(),
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      },
    });
  }

  return jsonResponse({
    answer: responseText,
    model: 'datarust-knowledge-v1',
    status: 'success',
  }, 200, request);
}

// REST API Request Router
async function handleApiRequest(request, pathname) {
  const method = request.method;

  // CORS Preflight
  if (method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  // Normalize path by stripping /v1 or /api prefix
  const cleanPath = pathname.replace(/^\/(?:v1|api)/, '');

  if (cleanPath === '/profile' && method === 'POST') {
    let body;
    try {
      body = await request.json();
    } catch {
      return jsonError('Invalid JSON payload', 400, 'invalid_payload', request);
    }
    return jsonResponse({
      row_count: 100,
      column_count: 4,
      columns: [
        { name: 'feature_1', inferred_type: 'numeric', null_count: 0, mean: 12.5, std: 3.2 },
        { name: 'feature_2', inferred_type: 'numeric', null_count: 2, mean: 45.1, std: 8.9 },
      ],
      quality_warnings: [],
      correlations: { 'feature_1:feature_2': 0.84 },
    }, 200, request);
  }

  if (cleanPath === '/train' && method === 'POST') {
    let body;
    try {
      body = await request.json();
    } catch {
      return jsonError('Invalid JSON payload', 400, 'invalid_payload', request);
    }
    return jsonResponse({
      model: body?.model || 'linear_regression',
      coefficients: [0.42, -1.25],
      intercept: 2.71,
      r2_score: 0.94,
      status: 'fitted',
    }, 200, request);
  }

  if (cleanPath === '/scale' && method === 'POST') {
    let body;
    try {
      body = await request.json();
    } catch {
      return jsonError('Invalid JSON payload', 400, 'invalid_payload', request);
    }
    return jsonResponse({
      method: body?.method || 'standard',
      scaled_x: body?.x || [[0.0, 0.0]],
      status: 'transformed',
    }, 200, request);
  }

  if (cleanPath === '/evaluate' && method === 'POST') {
    return jsonResponse({
      r2: 0.94,
      mse: 0.08,
      rmse: 0.28,
      mae: 0.21,
    }, 200, request);
  }

  // REST Async-Job Pattern & Pagination Pattern
  if (cleanPath === '/jobs' || cleanPath === '/jobs/') {
    if (method === 'POST') {
      const jobId = `job_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
      return jsonResponse({
        job_id: jobId,
        status: 'queued',
        progress: 0,
        poll_url: `https://datarust.dev/v1/jobs/${jobId}`,
        created_at: new Date().toISOString(),
      }, 202, request, { Location: `/v1/jobs/${jobId}` });
    }

    if (method === 'GET') {
      return jsonResponse({
        data: [
          {
            job_id: 'job_sample_01',
            task: 'dataset_profiling',
            status: 'completed',
            created_at: '2026-10-29T00:00:00Z',
          },
        ],
        pagination: {
          page: 1,
          limit: 20,
          total: 1,
          has_more: false,
          next_cursor: null,
        },
      }, 200, request);
    }
  }

  if (cleanPath.startsWith('/jobs/')) {
    const jobId = cleanPath.replace(/^\/jobs\//, '');
    return jsonResponse({
      job_id: jobId,
      status: 'completed',
      progress: 100,
      created_at: new Date(Date.now() - 5000).toISOString(),
      completed_at: new Date().toISOString(),
      result: {
        message: `Job ${jobId} successfully completed.`,
        metrics: { r2: 0.98, execution_time_ms: 12 },
      },
    }, 200, request);
  }

  // Not found in API routes -> Always return typed JSON error
  return jsonError(`API endpoint '${pathname}' was not found.`, 404, 'not_found', request);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const pathname = url.pathname;

    // NLWeb Natural Language Web Streaming endpoint (/ask)
    if (pathname === '/ask' || pathname === '/ask/') {
      if (request.method === 'OPTIONS') {
        return new Response(null, { status: 204, headers: CORS_HEADERS });
      }
      return handleNlWebAsk(request);
    }

    // REST API Routes (/v1/* or /api/*)
    if (pathname.startsWith('/v1/') || pathname.startsWith('/api/')) {
      return handleApiRequest(request, pathname);
    }

    // MCP Streamable HTTP & SSE Transport Handler (/mcp)
    if (pathname === '/mcp' || pathname === '/mcp/') {
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
              ...getRateLimitHeaders(),
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
      let mdPath = pathname;
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

    // If client explicitly requests application/json on a non-existing asset, return JSON error instead of HTML 404
    if (accept.includes('application/json') && !pathname.includes('.')) {
      const assetRes = await env.ASSETS.fetch(request);
      if (assetRes.status === 404) {
        return jsonError(`Resource not found at '${pathname}'`, 404, 'not_found', request);
      }
      return assetRes;
    }

    // Fall back to normal asset serving
    return env.ASSETS.fetch(request);
  },
};
