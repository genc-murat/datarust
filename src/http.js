export const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, HEAD',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, Accept, MCP-Protocol-Version, Idempotency-Key, X-Idempotency-Key, X-Job-Token, A2A-Version',
  'Access-Control-Expose-Headers': 'RateLimit-Limit, RateLimit-Remaining, RateLimit-Reset, RateLimit-Policy, Retry-After, Idempotency-Key, Idempotent-Replayed, Location, X-Job-Token, X-API-Version, Content-Digest, Signature-Input, Signature',
  'Access-Control-Max-Age': '86400',
};

export const RETENTION_MS = 30 * 60_000;
const MAX_BODY_BYTES = 256 * 1024;

export function jsonResponse(body, status = 200, headers = {}) {
  return Response.json(body, {
    status,
    headers: { ...CORS_HEADERS, 'Cache-Control': 'no-store', ...headers },
  });
}

export function apiError(status, code, message) {
  return { error: { code, message, status, details: [] } };
}

export class RequestError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export async function readJson(request) {
  if (!request.body) throw new RequestError(400, 'invalid_payload', 'A JSON request body is required.');
  const reader = request.body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let size = 0;
  let text = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY_BYTES) {
        throw new RequestError(413, 'payload_too_large', 'Request bodies are limited to 256 KiB.');
      }
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
    return JSON.parse(text);
  } catch (error) {
    await reader.cancel().catch(() => {});
    if (error instanceof RequestError) throw error;
    throw new RequestError(400, 'invalid_payload', 'Malformed JSON request body.');
  } finally {
    reader.releaseLock();
  }
}

export function base64(bytes) {
  return btoa(String.fromCharCode(...new Uint8Array(bytes)));
}

export function base64url(bytes) {
  return base64(bytes).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function sha256(text) {
  return crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
}

export function fingerprint(text) {
  return sha256(text).then(base64url);
}
