import { executeTool, InvalidInput, OPERATIONS } from './analytics.js';
import { ApiState, operationResult } from './api-state.js';
import { apiError, base64, base64url, CORS_HEADERS, fingerprint, isObject, jsonResponse, readJson, RequestError, sha256 } from './http.js';

export { ApiState };

const MCP_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05'];
const JOB_TOKEN_PATTERN = /^[A-Za-z0-9_-]{43}$/;
const JOB_ID_PATTERN = /^job_\d+_[0-9a-f-]{36}$/;
const API_HEADERS = { 'X-API-Version': '1.0' };
const MCP_TOOLS = [
  {
    name: 'profile_dataset',
    description: 'Profiles tabular CSV data, generating descriptive statistics, distribution metrics, pairwise correlations, and data quality checks.',
    annotations: {
      readOnlyHint: true,
      idempotentHint: true,
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
      readOnlyHint: true,
      idempotentHint: true,
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
      readOnlyHint: true,
      idempotentHint: true,
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
      readOnlyHint: true,
      idempotentHint: true,
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

function rpcResponse(id, result) {
  return { jsonrpc: '2.0', id, result };
}

function rpcError(id, code, message) {
  return { jsonrpc: '2.0', id, error: { code, message } };
}

function sseResponse(events, headers = {}) {
  const body = events.map((event) => {
    const data = typeof event.data === 'string' ? event.data : JSON.stringify(event.data);
    return (event.type ? `event: ${event.type}\n` : '') + `data: ${data}\n\n`;
  }).join('');
  return new Response(body, {
    headers: { ...CORS_HEADERS, ...headers, 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store' },
  });
}

async function stateCall(env, scope, action, args = {}) {
  const object = env.API_STATE.get(env.API_STATE.idFromName(scope));
  const response = await object.fetch('https://api-state.internal/' + action, {
    method: 'POST',
    body: JSON.stringify(args),
  });
  if (!response.ok) throw new Error('Durable Object request failed');
  return response.json();
}

function rateHeaders(rate) {
  return {
    'RateLimit-Limit': String(rate.limit),
    'RateLimit-Remaining': String(rate.remaining),
    'RateLimit-Reset': String(rate.reset),
    'RateLimit-Policy': rate.limit + ';w=60',
    ...(!rate.allowed ? { 'Retry-After': String(rate.reset) } : {}),
  };
}

function idempotencyKey(request) {
  const key = request.headers.get('idempotency-key') || request.headers.get('x-idempotency-key');
  if (key && (key.length > 128 || !/^[!-~]+$/.test(key))) {
    throw new RequestError(400, 'invalid_idempotency_key', 'Idempotency-Key must contain 1–128 visible ASCII characters.');
  }
  return key;
}

// Object field order is insignificant. Normalize it before fingerprinting.
function canonicalize(value, depth = 0) {
  if (depth >= 64) throw new RequestError(400, 'invalid_payload', 'JSON nesting is limited to 64 levels.');
  if (Array.isArray(value)) return value.map((entry) => canonicalize(entry, depth + 1));
  if (!isObject(value)) return value;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalize(value[key], depth + 1)]));
}

async function requestIdentity(key, operation, body) {
  if (!key) return {};
  return {
    key: await fingerprint(operation + '\n' + key),
    digest: await fingerprint(JSON.stringify(canonicalize(body))),
  };
}

function apiResult(result, headers, key) {
  const replay = key && result.replayed !== undefined ? {
    'Idempotency-Key': key,
    'Idempotent-Replayed': String(result.replayed),
  } : {};
  return jsonResponse(result.body, result.status, { ...headers, ...replay });
}

function paginationNumber(url, name, fallback, max) {
  const raw = url.searchParams.get(name);
  if (raw === null) return fallback;
  const value = Number(raw);
  if (!/^\d+$/.test(raw) || !Number.isSafeInteger(value) || value < 1 || value > max) {
    throw new RequestError(400, 'invalid_pagination', `'${name}' must be an integer in 1–${max}.`);
  }
  return value;
}

async function handleJob(request, url, path, env, client, headers) {
  const key = request.method === 'POST' ? idempotencyKey(request) : null;
  let token = request.headers.get('x-job-token');
  if (token && !JOB_TOKEN_PATTERN.test(token)) {
    throw new RequestError(400, 'invalid_job_token', 'X-Job-Token must be a base64url-encoded 32-byte token.');
  }
  if (path === '/jobs' && request.method === 'POST') {
    const body = await readJson(request);
    if (!isObject(body) || !OPERATIONS.has(body.task) || !isObject(body.parameters)) {
      throw new RequestError(422, 'invalid_job', 'Provide a supported task and a parameters object.');
    }
    const identity = await requestIdentity(key, 'submit', body);
    if (!token) {
      const allocation = await stateCall(env, client, 'job-token', identity);
      if (allocation.status !== 200) return apiResult(allocation, headers, key);
      token = allocation.body.token;
    }
    const result = await stateCall(env, 'jobs:' + await fingerprint(token), 'submit', { body, ...identity });
    if (result.status === 202) {
      const pollPath = '/v1/jobs/' + result.body.job_id;
      result.body = { ...result.body, poll_url: new URL(pollPath, url).href };
      headers = { ...headers, Location: pollPath, 'X-Job-Token': token };
    }
    return apiResult(result, headers, key);
  }
  if (request.method !== 'GET') {
    return jsonResponse(apiError(405, 'method_not_allowed', 'Method not allowed.'), 405, {
      ...headers, Allow: path === '/jobs' ? 'GET, POST' : 'GET',
    });
  }
  if (!token) throw new RequestError(401, 'job_token_required', 'Send the X-Job-Token from the job submission response.');
  const scope = 'jobs:' + await fingerprint(token);
  if (path !== '/jobs') {
    const id = path.slice('/jobs/'.length);
    if (!JOB_ID_PATTERN.test(id)) return jsonResponse(apiError(404, 'job_not_found', 'Job was not found.'), 404, headers);
    return apiResult(await stateCall(env, scope, 'poll', { id }), headers);
  }
  const page = paginationNumber(url, 'page', 1, Number.MAX_SAFE_INTEGER);
  const limit = paginationNumber(url, 'limit', 20, 100);
  const cursor = url.searchParams.get('cursor');
  if (cursor !== null && (!JOB_ID_PATTERN.test(cursor) || url.searchParams.has('page'))) {
    throw new RequestError(400, 'invalid_pagination', 'Use a cursor from a previous page or a page number.');
  }
  return apiResult(await stateCall(env, scope, 'list', { page, limit, cursor }), headers);
}

async function handleApi(request, url, env, client, headers) {
  const path = url.pathname.replace(/^\/(v1|api)(?=\/|$)/, '').replace(/\/$/, '');
  if (path === '/jobs' || path.startsWith('/jobs/')) {
    return handleJob(request, url, path, env, client, headers);
  }
  const operation = path.slice(1);
  if (!OPERATIONS.has(operation)) {
    return jsonResponse(apiError(404, 'not_found', 'API endpoint not found.'), 404, headers);
  }
  if (request.method !== 'POST') {
    return jsonResponse(apiError(405, 'method_not_allowed', 'This endpoint accepts POST.'), 405, {
      ...headers, Allow: 'POST',
    });
  }
  const body = await readJson(request);
  const key = idempotencyKey(request);
  const result = key
    ? await stateCall(env, client, 'operation', { operation, body, ...await requestIdentity(key, operation, body) })
    : operationResult(operation, body);
  return apiResult(result, headers, key);
}

function validRpc(body) {
  return isObject(body) && body.jsonrpc === '2.0' && typeof body.method === 'string'
    && (!Object.hasOwn(body, 'id') || typeof body.id === 'string' || Number.isFinite(body.id));
}

async function handleMcp(request, url, env, headers) {
  const origin = request.headers.get('origin');
  if (origin && origin !== url.origin) {
    return jsonResponse(rpcError(null, -32600, 'Origin is not allowed.'), 403, headers);
  }
  const accept = request.headers.get('accept') || '';
  if (request.method === 'GET' || request.method === 'HEAD') {
    if (accept.includes('text/event-stream')) return new Response(null, { status: 405, headers: { ...headers, Allow: 'POST' } });
    if (accept.includes('application/json') || url.searchParams.get('card') === 'true') {
      return env.ASSETS.fetch(new Request(new URL('/.well-known/mcp/server-card.json', url), { method: request.method }));
    }
    return Response.redirect(new URL('/docs/mcp/', url), 302);
  }
  if (request.method !== 'POST') return new Response(null, { status: 405, headers: { ...headers, Allow: 'GET, HEAD, POST' } });
  const version = request.headers.get('mcp-protocol-version');
  if (version && !MCP_VERSIONS.includes(version)) {
    return jsonResponse(rpcError(null, -32602, 'Unsupported MCP protocol version.'), 400, headers);
  }

  let body;
  try {
    body = await readJson(request);
  } catch (error) {
    return jsonResponse(rpcError(null, -32700, error.message), error.status, headers);
  }

  const reply = (message, status = 200) => (
    accept.includes('text/event-stream') && !accept.includes('application/json') && status === 200
      ? sseResponse([{ type: 'message', data: message }], headers)
      : jsonResponse(message, status, headers)
  );
  if (!validRpc(body)) {
    if (isObject(body) && body.jsonrpc === '2.0' && !('method' in body) && ('result' in body || 'error' in body)) {
      return new Response(null, { status: 202, headers });
    }
    return reply(rpcError(null, -32600, 'Invalid JSON-RPC request.'), 400);
  }
  if (!Object.hasOwn(body, 'id')) return new Response(null, { status: 202, headers });

  const { id, method, params } = body;
  const success = (result) => reply(rpcResponse(id, result));
  switch (method) {
    case 'initialize': {
      const requested = params?.protocolVersion;
      return success({
        protocolVersion: MCP_VERSIONS.includes(requested) ? requested : MCP_VERSIONS[0],
        capabilities: { tools: {}, resources: {} },
        serverInfo: { name: 'datarust', version: '0.8.0' },
      });
    }
    case 'ping':
      return success({});
    case 'tools/list':
      return success({ tools: MCP_TOOLS });
    case 'tools/call': {
      const name = params?.name;
      if (!MCP_TOOLS.some((tool) => tool.name === name)) {
        return reply(rpcError(id, -32602, 'Unknown tool.'));
      }
      try {
        const result = executeTool(name, params.arguments ?? {});
        return success({ content: [{ type: 'text', text: JSON.stringify(result) }] });
      } catch (error) {
        if (!(error instanceof InvalidInput)) throw error;
        return success({ isError: true, content: [{ type: 'text', text: error.message }] });
      }
    }
    case 'resources/list':
      return success({ resources: MCP_RESOURCES });
    case 'resources/read': {
      const resource = MCP_RESOURCES.find((entry) => entry.uri === params?.uri);
      if (!resource) return reply(rpcError(id, -32002, 'Resource not found.'));
      const paths = {
        'datarust://docs/overview': '/docs/index.md',
        'datarust://docs/api': '/docs/api/index.md',
        'datarust://schema/openapi.json': '/openapi.json',
      };
      const response = await env.ASSETS.fetch(new Request(new URL(paths[resource.uri], url)));
      if (!response.ok) return reply(rpcError(id, -32002, 'Resource not found.'));
      const { uri, mimeType } = resource;
      return success({ contents: [{ uri, mimeType, text: await response.text() }] });
    }
    default:
      return reply(rpcError(id, -32601, 'Method not found.'));
  }
}

async function answerFromDocs(query, env, url) {
  const fallback = 'datarust provides Rust tools for tabular preprocessing, machine learning and dataset profiling. See ' + new URL('/docs/', url).href;
  const terms = [...new Set(query.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [])]
    .filter((term) => term.length >= 3 && !['datarust', 'how', 'what', 'the', 'and', 'for', 'can', 'use', 'which', 'does'].includes(term));
  if (!terms.length) return fallback;
  const response = await env.ASSETS.fetch(new Request(new URL('/search.json', url)));
  if (!response.ok) return fallback;
  const matches = (await response.json()).map((entry) => {
    const title = entry.title.toLowerCase();
    const text = [entry.description, ...(entry.headings || [])].join(' ').toLowerCase();
    const score = terms.reduce((total, term) => total + (title.includes(term) ? 3 : text.includes(term) ? 1 : 0), 0);
    return { entry, score };
  }).filter(({ score }) => score > 0).sort((a, b) => b.score - a.score).slice(0, 3);
  if (!matches.length) return fallback;
  return matches.map(({ entry }) => `${entry.title} — ${entry.description || entry.headings?.[0] || 'Documentation'} (${new URL(entry.url, url).href})`).join('\n');
}

async function handleAsk(request, url, env, headers) {
  if (!['GET', 'POST'].includes(request.method)) return new Response(null, { status: 405, headers });
  const body = request.method === 'POST' ? await readJson(request) : {};
  const query = body?.q ?? body?.query ?? body?.question ?? url.searchParams.get('q') ?? url.searchParams.get('query') ?? '';
  if (typeof query !== 'string' || query.length > 2000) {
    throw new RequestError(400, 'invalid_query', 'Provide a text query of at most 2,000 characters.');
  }
  const answer = await answerFromDocs(query, env, url);
  if (String(body?.stream ?? url.searchParams.get('stream')) === 'true' || request.headers.get('accept')?.includes('text/event-stream')) {
    const events = answer.split('\n').map((chunk, index, parts) => ({
      type: 'message', data: { chunk: index ? '\n' + chunk : chunk, index, done: index === parts.length - 1 },
    }));
    return sseResponse([...events, { type: 'done', data: '[DONE]' }], headers);
  }
  return jsonResponse({ answer, model: 'datarust-knowledge-v1', status: 'success' }, 200, headers);
}

async function handleA2A(request, url, env, headers) {
  const reply = (message, status = 200) => jsonResponse(message, status, headers);
  if (request.method !== 'POST') return new Response(null, { status: 405, headers: { ...headers, Allow: 'POST' } });
  let body;
  try {
    body = await readJson(request);
  } catch (error) {
    return reply(rpcError(null, -32700, error.message), error.status);
  }
  if (!validRpc(body) || !Object.hasOwn(body, 'id')) return reply(rpcError(null, -32600, 'Invalid JSON-RPC request.'), 400);
  const { id, method, params } = body;
  const version = request.headers.get('a2a-version');
  if (version && version !== '1.0') return reply(rpcError(id, -32009, 'Unsupported A2A version.'));
  if (!['SendMessage', 'SendStreamingMessage'].includes(method)) {
    return reply(rpcError(id, -32601, 'Method not found.'));
  }
  const message = params?.message;
  if (!isObject(message) || message.role !== 'ROLE_USER' || typeof message.messageId !== 'string'
    || !message.messageId || !Array.isArray(message.parts) || !message.parts.length || message.parts.length > 16) {
    return reply(rpcError(id, -32602, 'Provide a user message with a messageId and 1–16 parts.'));
  }
  const parts = [];
  try {
    for (const part of message.parts) {
      if (typeof part?.text === 'string' && part.text.length <= 2000) {
        parts.push({ text: await answerFromDocs(part.text, env, url) });
      } else if (isObject(part?.data) && MCP_TOOLS.some((tool) => tool.name === part.data.tool)) {
        parts.push({ data: executeTool(part.data.tool, part.data.arguments ?? {}) });
      } else {
        return reply(rpcError(id, -32602, 'Provide a text part or a data part with a supported tool and arguments.'));
      }
    }
  } catch (error) {
    if (!(error instanceof InvalidInput)) throw error;
    return reply(rpcError(id, -32602, error.message));
  }
  const result = rpcResponse(id, { message: {
    messageId: crypto.randomUUID(),
    contextId: typeof message.contextId === 'string' ? message.contextId : crypto.randomUUID(),
    role: 'ROLE_AGENT',
    parts,
  } });
  return method === 'SendStreamingMessage'
    ? sseResponse([{ data: result }], headers)
    : reply(result);
}

async function handleSignaturesDirectory(request, url, env) {
  if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: CORS_HEADERS });
  if (!['GET', 'HEAD'].includes(request.method)) return new Response(null, { status: 405, headers: { Allow: 'GET, HEAD' } });
  const secret = env.WEB_BOT_AUTH_PRIVATE_JWK;
  if (!secret) return jsonResponse(apiError(404, 'not_found', 'No signing key directory is configured.'), 404);
  const { kty, crv, x, d } = JSON.parse(secret);
  if (kty !== 'OKP' || crv !== 'Ed25519' || !JOB_TOKEN_PATTERN.test(x) || !JOB_TOKEN_PATTERN.test(d)) {
    throw new Error('Invalid signing key configuration');
  }
  const key = await crypto.subtle.importKey('jwk', { kty, crv, x, d }, 'Ed25519', false, ['sign']);
  const kid = await fingerprint(JSON.stringify({ crv, kty, x }));
  const body = JSON.stringify({ keys: [{ kty, crv, kid, x, use: 'sig' }] });
  const headers = {
    ...CORS_HEADERS,
    'Content-Type': 'application/http-message-signatures-directory+json',
    'Cache-Control': 'public, max-age=300, no-transform',
  };
  if (request.method === 'HEAD') return new Response(null, { headers });
  const digest = `sha-256=:${base64(await sha256(body))}:`;
  const created = Math.floor(Date.now() / 1000);
  const params = `("@authority";req "content-digest");created=${created};expires=${created + 600};keyid="${kid}";tag="http-message-signatures-directory"`;
  const signatureBase = `"@authority";req: ${url.host}\n"content-digest": ${digest}\n"@signature-params": ${params}`;
  const signature = await crypto.subtle.sign('Ed25519', key, new TextEncoder().encode(signatureBase));
  return new Response(body, { headers: {
    ...headers,
    'Content-Digest': digest,
    'Signature-Input': 'binding=' + params,
    Signature: `binding=:${base64(signature)}:`,
  } });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const pathname = url.pathname;
    if (/^\/(?:\.well-known\/)?http-message-signatures-directory(?:\.json)?\/?$/.test(pathname)) {
      try {
        return await handleSignaturesDirectory(request, url, env);
      } catch {
        return jsonResponse(apiError(503, 'signing_unavailable', 'The signing directory is unavailable.'), 503);
      }
    }

    const api = /^\/(v1|api)(\/|$)/.test(pathname);
    const protocol = /^\/(mcp|a2a|ask)\/?$/.exec(pathname)?.[1];
    if (api || protocol) {
      if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: CORS_HEADERS });
      let headers = { ...CORS_HEADERS, ...API_HEADERS };
      try {
        const client = 'client:' + (request.headers.get('cf-connecting-ip') || 'local');
        // Assets, discovery cards and browser redirects do not consume API quota.
        if (api || protocol !== 'mcp' || request.method === 'POST') {
          const rate = await stateCall(env, client, 'rate');
          headers = { ...headers, ...rateHeaders(rate) };
          if (!rate.allowed) return jsonResponse(apiError(429, 'rate_limit', 'Request quota exhausted.'), 429, headers);
        }
        if (api) return await handleApi(request, url, env, client, headers);
        if (protocol === 'mcp') return await handleMcp(request, url, env, headers);
        if (protocol === 'a2a') return await handleA2A(request, url, env, headers);
        return await handleAsk(request, url, env, headers);
      } catch (error) {
        if (error instanceof RequestError) {
          return jsonResponse(apiError(error.status, error.code, error.message), error.status, headers);
        }
        console.error('Worker request failed', error.name);
        return jsonResponse(apiError(500, 'internal_error', 'Request execution failed.'), 500, headers);
      }
    }

    // Markdown 404 handler for AI agents
    function markdown404(requestedPath) {
      const mdBody = `# 404 Not Found\n\nThe requested path '${requestedPath}' could not be found on datarust.dev.\n\n## Developer Resources\n- [Home](https://datarust.dev/)\n- [Developer Portal](https://datarust.dev/developers/)\n- [Documentation](https://datarust.dev/docs/)\n- [Model Context Protocol (MCP)](https://datarust.dev/docs/mcp/)\n- [API Reference](https://datarust.dev/docs/api/)\n- [OpenAPI Specification](https://datarust.dev/openapi.json)\n- [Field Notes](https://datarust.dev/blog/)\n`;
      return new Response(mdBody, {
        status: 404,
        headers: {
          ...CORS_HEADERS,
          'Content-Type': 'text/markdown; charset=utf-8',
          'X-Markdown-Tokens': String(Math.ceil(mdBody.length / 4)),
        },
      });
    }

    const accept = request.headers.get('accept') || '';
    const userAgent = (request.headers.get('user-agent') || '').toLowerCase();
    const isAgentProbe = pathname.includes('probe') || userAgent.includes('ora') || userAgent.includes('agent') || userAgent.includes('bot');

    // Markdown content negotiation for AI agents
    if (accept.includes('text/markdown') || (isAgentProbe && !pathname.includes('.'))) {
      let mdPath = pathname;
      if (mdPath.endsWith('/')) {
        mdPath += 'index.md';
      } else if (mdPath.endsWith('.html')) {
        mdPath = mdPath.replace(/\.html$/, '.md');
      } else if (!mdPath.includes('.')) {
        mdPath += '/index.md';
      }
      const mdUrl = new URL(request.url);
      mdUrl.pathname = mdPath;
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
      } catch {
        // Fall back to markdown 404
      }
      return markdown404(pathname);
    }

    // Fall back to normal asset serving, with intelligent 404 negotiation
    const assetRes = await env.ASSETS.fetch(request);
    if (assetRes.status === 404) {
      if (accept.includes('text/markdown') || isAgentProbe) {
        return markdown404(pathname);
      }
      if (accept.includes('application/json')) {
        return jsonResponse(apiError(404, 'not_found', `Resource not found at '${pathname}'`), 404);
      }
    }
    return assetRes;
  },
};
