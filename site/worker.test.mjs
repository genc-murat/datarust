import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { unstable_dev } from 'wrangler';

let worker;
let signingKey;

before(async () => {
  signingKey = await crypto.subtle.generateKey('Ed25519', true, ['sign', 'verify']);
  const jwk = await crypto.subtle.exportKey('jwk', signingKey.privateKey);
  worker = await unstable_dev('src/worker.js', {
    config: 'wrangler.jsonc',
    ip: '127.0.0.1',
    port: 0,
    local: true,
    persist: false,
    logLevel: 'error',
    vars: { WEB_BOT_AUTH_PRIVATE_JWK: JSON.stringify(jwk) },
    experimental: { disableExperimentalWarning: true, disableDevRegistry: true, watch: false },
  });
}, { timeout: 30_000 });

after(async () => {
  await worker?.stop();
});

function post(path, body, headers = {}) {
  return worker.fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });
}

function rpc(path, method, params = {}, headers = {}) {
  return post(path, { jsonrpc: '2.0', id: 1, method, params }, headers);
}

test('REST calculates real results and rejects invalid input', async () => {
  const x = [[1], [2], [3]];
  for (const slope of [2, 3]) {
    const response = await post('/v1/train', { x, y: x.map(([v]) => v * slope) });
    assert.equal(response.status, 200);
    const output = await response.json();
    assert.ok(Math.abs(output.coefficients[0] - slope) < 1e-10);
    assert.equal(output.r2_score, 1);
  }
  const scale = await post('/v1/scale', { x, method: 'minmax' });
  assert.deepEqual((await scale.json()).scaled_x, [[0], [0.5], [1]]);

  const evalResponse = await post('/v1/evaluate', { y_true: [1, 3], y_pred: [1, 1] });
  const metrics = await evalResponse.json();
  assert.deepEqual({ mse: metrics.mse, rmse: metrics.rmse, mae: metrics.mae }, { mse: 2, rmse: Math.sqrt(2), mae: 1 });

  const singleton = await post('/v1/evaluate', { y_true: [1], y_pred: [3] });
  assert.equal(singleton.status, 200);
  assert.deepEqual(await singleton.json(), { task: 'regression', r2: null, mse: 4, rmse: 2, mae: 2 });

  const profile = await post('/v1/profile', { csv_data: 'age,weight\n1,2\n2,4\n3,6' });
  assert.equal(profile.status, 200);
  const report = await profile.json();
  assert.equal(report.row_count, 3);
  assert.equal(report.columns[0].mean, 2);
  assert.ok(report.correlations['age:weight'] > 0.99);
  const bad = await post('/v1/scale', { x: [[1], [2, 3]] });
  assert.equal(bad.status, 422);
  assert.equal((await bad.json()).error.code, 'invalid_input');
});

test('REST replays an idempotent request and rejects different input under the same key', async () => {
  const headers = { 'Idempotency-Key': crypto.randomUUID() };
  const one = await post('/v1/scale', { x: [[1], [2]], method: 'minmax' }, headers);
  const two = await post('/v1/scale', { method: 'minmax', x: [[1], [2]] }, headers);
  assert.equal(one.status, 200);
  assert.equal(two.status, 200);
  assert.equal(one.headers.get('idempotent-replayed'), 'false');
  assert.equal(two.headers.get('idempotent-replayed'), 'true');
  assert.deepEqual(await two.json(), await one.json());
  if (Number(one.headers.get('ratelimit-reset')) > 1) {
    assert.equal(Number(one.headers.get('ratelimit-remaining')) - Number(two.headers.get('ratelimit-remaining')), 1);
  }
  const conflict = await post('/v1/scale', { x: [[1], [3]], method: 'minmax' }, headers);
  assert.equal(conflict.status, 409);
});

test('Job submission is idempotent, scoped, processed by an alarm, and paginated', async () => {
  const request = { task: 'scale', parameters: { x: [[4], [8]], method: 'minmax' } };
  const headers = { 'Idempotency-Key': crypto.randomUUID() };
  const responses = await Promise.all([
    post('/v1/jobs', request, headers),
    post('/v1/jobs', request, headers),
  ]);
  assert.deepEqual(responses.map((response) => response.status), [202, 202]);
  const jobs = await Promise.all(responses.map((response) => response.json()));
  assert.equal(jobs[0].job_id, jobs[1].job_id);
  assert.equal(responses[0].headers.get('X-Job-Token'), responses[1].headers.get('X-Job-Token'));
  assert.deepEqual(responses.map((response) => response.headers.get('idempotent-replayed')).sort(), ['false', 'true']);
  const token = responses[0].headers.get('X-Job-Token');
  const credentials = { 'X-Job-Token': token };
  const location = responses[0].headers.get('location');
  assert.equal(new URL(jobs[0].poll_url).pathname, location);
  assert.equal((await worker.fetch(location)).status, 401);
  assert.equal((await worker.fetch(location, { headers: { 'X-Job-Token': 'a'.repeat(43) } })).status, 404);

  let finished;
  for (let attempt = 0; attempt < 50; attempt++) {
    const response = await worker.fetch(location, { headers: credentials });
    assert.equal(response.status, 200);
    finished = await response.json();
    if (finished.status === 'completed' || finished.status === 'failed') break;
    await delay(100);
  }
  assert.equal(finished.status, 'completed');
  assert.deepEqual(finished.result.scaled_x, [[0], [1]]);
  assert.equal('parameters' in finished, false);

  const next = await post('/v1/jobs', { task: 'evaluate', parameters: { y_true: [1, 2], y_pred: [1, 2] } }, credentials);
  assert.equal(next.status, 202);
  const listing = await worker.fetch('/v1/jobs?limit=1', { headers: credentials });
  const page = await listing.json();
  assert.equal(page.data.length, 1);
  assert.equal(page.pagination.total, 2);
  const last = await worker.fetch('/v1/jobs?limit=1&cursor=' + page.pagination.next_cursor, { headers: credentials });
  const lastPage = await last.json();
  assert.equal(lastPage.data.length, 1);
  assert.notEqual(lastPage.data[0].job_id, page.data[0].job_id);
  assert.equal(lastPage.pagination.has_more, false);
});

test('MCP discovery and agent documentation match the hosted server', async () => {
  const init = await (await rpc('/mcp', 'initialize', { protocolVersion: '2025-11-25' })).json();
  const listing = await (await rpc('/mcp', 'tools/list')).json();
  const toolNames = listing.result.tools.map((tool) => tool.name).sort();
  const capabilities = Object.keys(init.result.capabilities).sort();
  for (const path of ['/.well-known/mcp/server-card.json', '/.well-known/mcp/server.json', '/.well-known/mcp.json', '/server.json', '/mcp.json']) {
    const card = await (await worker.fetch(path)).json();
    assert.deepEqual(card.tools.map((tool) => tool.name).sort(), toolNames, path);
    assert.deepEqual(Object.keys(card.capabilities).sort(), capabilities, path);
  }

  const guide = await (await worker.fetch('/llms-full.txt')).text();
  const section = guide.match(/^#### Available MCP tools\n([\s\S]*?)^#/m);
  assert.ok(section, 'llms-full.txt must document available MCP tools');
  const documented = [...section[1].matchAll(/^- `([^`]+)`/gm)].map((match) => match[1]);
  assert.deepEqual(documented.sort(), toolNames);
});

test('MCP executes Rust tools and supports typed errors and SSE responses', async () => {
  const init = await rpc('/mcp', 'initialize', { protocolVersion: '2025-11-25' });
  assert.equal((await init.json()).result.protocolVersion, '2025-11-25');
  const response = await rpc('/mcp', 'tools/call', {
    name: 'scale_features',
    arguments: { x: [[2], [4]], method: 'minmax' },
  });
  const { result } = await response.json();
  assert.deepEqual(JSON.parse(result.content[0].text).scaled_features, [[0], [1]]);
  const invalid = await rpc('/mcp', 'tools/call', { name: 'scale_features', arguments: { x: [] } });
  assert.equal((await invalid.json()).result.isError, true);
  const unsupported = await rpc('/mcp', 'tools/call', { name: 'unknown' });
  assert.equal((await unsupported.json()).error.code, -32602);
  const stream = await rpc('/mcp', 'ping', {}, { accept: 'text/event-stream' });
  assert.match(stream.headers.get('content-type'), /^text\/event-stream/);
  assert.match(await stream.text(), /"result":\{\}/);
});

test('A2A card advertises the working JSON-RPC binding and its stream', async () => {
  const card = await (await worker.fetch('/.well-known/agent-card.json')).json();
  const path = new URL(card.supportedInterfaces[0].url).pathname;
  const request = {
    message: { messageId: 'test', role: 'ROLE_USER', parts: [
      { data: { tool: 'evaluate_metrics', arguments: { task: 'classification', y_true: [2, 3], y_pred: [2, 2] } } },
    ] },
  };
  const response = await rpc(path, 'SendMessage', request, { 'A2A-Version': '1.0' });
  const message = (await response.json()).result.message;
  assert.equal(message.role, 'ROLE_AGENT');
  assert.equal(message.parts[0].data.accuracy, 0.5);
  const streaming = await rpc(path, 'SendStreamingMessage', {
    message: { messageId: 'text', role: 'ROLE_USER', parts: [{ text: 'nearest neighbors' }] },
  }, { 'A2A-Version': '1.0' });
  assert.match(streaming.headers.get('content-type'), /^text\/event-stream/);
  assert.match(await streaming.text(), /nearest neighbors/i);
});

test('Web Bot Auth directory binds the public JWK and body to the request authority', async () => {
  const url = new URL('https://datarust.dev/.well-known/http-message-signatures-directory');
  const response = await worker.fetch(url);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('Content-Type'), 'application/http-message-signatures-directory+json');
  const body = await response.text();
  const { keys: [publicJwk] } = JSON.parse(body);
  assert.equal('d' in publicJwk, false);
  const digest = response.headers.get('Content-Digest');
  assert.equal(digest, `sha-256=:${Buffer.from(await crypto.subtle.digest('SHA-256', Buffer.from(body))).toString('base64')}:`);
  const params = response.headers.get('Signature-Input').slice('binding='.length);
  const signature = Buffer.from(response.headers.get('Signature').slice('binding=:'.length, -1), 'base64');
  const signed = Buffer.from(`"@authority";req: ${url.host}\n"content-digest": ${digest}\n"@signature-params": ${params}`);
  const key = await crypto.subtle.importKey('jwk', publicJwk, 'Ed25519', false, ['verify']);
  assert.equal(await crypto.subtle.verify('Ed25519', key, signature, signed), true);
  const api = await post('/v1/scale', { x: [[1]] });
  assert.equal(api.headers.has('signature-input'), false);
});

test('the rate limit rejects requests after the client exhausts its quota', async () => {
  for (let attempt = 0; attempt < 55; attempt++) {
    const batch = await Promise.all(Array.from({ length: 40 }, () => worker.fetch('/v1/unknown')));
    const limited = batch.find((response) => response.status === 429);
    if (limited) {
      assert.equal(limited.headers.get('RateLimit-Remaining'), '0');
      assert.ok(Number(limited.headers.get('Retry-After')) > 0);
      assert.equal((await limited.json()).error.code, 'rate_limit');
      return;
    }
    assert.ok(batch.every((response) => response.status === 404));
  }
  assert.fail('Expected the API to enforce its request quota.');
});
