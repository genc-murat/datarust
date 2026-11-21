import { executeOperation, InvalidInput } from './analytics.js';
import { apiError, base64url, jsonResponse, RETENTION_MS } from './http.js';

const REQUESTS_PER_MINUTE = 1000;
const WINDOW_MS = 60_000;
const MAX_JOBS = 100;
const MAX_CACHED_REQUESTS = 100;

function record(body, status = 200) {
  return { body, status };
}

export function operationResult(operation, args) {
  try {
    return record(executeOperation(operation, args));
  } catch (error) {
    if (!(error instanceof InvalidInput)) throw error;
    return record(apiError(422, 'invalid_input', error.message), 422);
  }
}

function unexpired(entries, now) {
  return [...entries.values()].filter((value) => value.expires > now);
}

function jobResult(job) {
  const { parameters, expires, ...result } = job;
  return result;
}

export class ApiState {
  constructor(state) {
    this.storage = state.storage;
  }

  async fetch(request) {
    const action = new URL(request.url).pathname;
    const args = await request.json();
    switch (action) {
      case '/rate':
        return jsonResponse(await this.takeRate());
      case '/operation':
        return jsonResponse(await this.idempotent(args, () => operationResult(args.operation, args.body)));
      case '/job-token':
        return jsonResponse(await this.idempotent(args, () => record({
          token: base64url(crypto.getRandomValues(new Uint8Array(32))),
        })));
      case '/submit':
        // The alarm and job record are committed in the same transaction.
        return jsonResponse(await this.idempotent(args, (txn, now) => this.submit(txn, args.body, now)));
      case '/list':
        return jsonResponse(await this.listJobs(args));
      case '/poll':
        return jsonResponse(await this.pollJob(args.id));
      default:
        return new Response(null, { status: 404 });
    }
  }

  async takeRate() {
    const now = Date.now();
    const reset = (Math.floor(now / WINDOW_MS) + 1) * WINDOW_MS;
    return this.storage.transaction(async (txn) => {
      const previous = await txn.get('rate');
      const count = previous?.reset === reset ? previous.count : 0;
      if (count < REQUESTS_PER_MINUTE) {
        await txn.put('rate', { count: count + 1, reset, expires: reset });
        if (count === 0) await this.schedule(txn, reset);
      }
      return {
        allowed: count < REQUESTS_PER_MINUTE,
        limit: REQUESTS_PER_MINUTE,
        remaining: Math.max(0, REQUESTS_PER_MINUTE - count - 1),
        reset: Math.ceil((reset - now) / 1000),
      };
    });
  }

  async schedule(txn, time) {
    const scheduled = await txn.getAlarm();
    if (scheduled === null || time < scheduled) await txn.setAlarm(time);
  }

  async idempotent({ key, digest }, run) {
    return this.storage.transaction(async (txn) => {
      const now = Date.now();
      if (!key) return run(txn, now);
      const entryKey = 'request:' + key;
      const previous = await txn.get(entryKey);
      if (previous?.expires > now) {
        if (previous.digest !== digest) {
          return record(apiError(409, 'idempotency_conflict', 'The idempotency key was already used with different input.'), 409);
        }
        return { ...previous.result, replayed: true };
      }
      const entries = unexpired(await txn.list({ prefix: 'request:' }), now);
      if (entries.length >= MAX_CACHED_REQUESTS) {
        return record(apiError(429, 'idempotency_quota', 'This client has too many retained idempotency keys.'), 429);
      }
      const result = await run(txn, now);
      if (result.status < 500 && result.status !== 429) {
        const expires = now + RETENTION_MS;
        await txn.put(entryKey, { digest, result, expires });
        await this.schedule(txn, expires);
      }
      return { ...result, replayed: false };
    });
  }

  async submit(txn, { task, parameters }, now) {
    const entries = unexpired(await txn.list({ prefix: 'job:' }), now);
    if (entries.length >= MAX_JOBS) {
      return record(apiError(429, 'job_quota', 'This job token has too many retained jobs.'), 429);
    }
    const job_id = `job_${now}_${crypto.randomUUID()}`;
    const created_at = new Date(now).toISOString();
    const status = 'queued';
    await txn.put('job:' + job_id, {
      job_id, task, status, progress: 0, created_at, parameters, expires: now + RETENTION_MS,
    });
    await this.schedule(txn, now + 10);
    return record({ job_id, status, progress: 0, created_at }, 202);
  }

  async pollJob(id) {
    const job = await this.storage.get('job:' + id);
    if (!job || job.expires <= Date.now()) {
      return record(apiError(404, 'job_not_found', 'Job was not found.'), 404);
    }
    return record(jobResult(job));
  }

  async listJobs({ page, limit, cursor }) {
    const entries = await this.storage.list({ prefix: 'job:', reverse: true });
    const jobs = unexpired(entries, Date.now());
    const cursorIndex = cursor ? jobs.findIndex((job) => job.job_id === cursor) : -1;
    if (cursor && cursorIndex < 0) return record(apiError(400, 'invalid_cursor', 'Job cursor is invalid or expired.'), 400);
    const offset = cursor ? cursorIndex + 1 : (page - 1) * limit;
    const data = jobs.slice(offset, offset + limit).map(({ job_id, task, status, created_at }) => ({
      job_id, task, status, created_at,
    }));
    const has_more = offset + limit < jobs.length;
    return record({ data, pagination: {
      page: cursor ? Math.floor(offset / limit) + 1 : page,
      limit,
      total: jobs.length,
      has_more,
      next_cursor: has_more ? data.at(-1).job_id : null,
    } });
  }

  async alarm() {
    const now = Date.now();
    const { pending, nextExpiry } = await this.storage.transaction(async (txn) => {
      const entries = await txn.list();
      const expired = [...entries].filter(([, value]) => value.expires <= now).map(([key]) => key);
      if (expired.length) await txn.delete(expired);
      const pending = [...entries].filter(([key, job]) =>
        key.startsWith('job:') && job.expires > now && (job.status === 'queued' || job.status === 'running'));
      return { pending, nextExpiry: Math.min(...unexpired(entries, now).map((entry) => entry.expires)) };
    });
    if (pending.length) {
      const [key, job] = pending[0];
      await this.storage.put(key, { ...job, status: 'running', progress: 50 });
      let outcome;
      try {
        outcome = operationResult(job.task, job.parameters);
      } catch (error) {
        console.error('Async operation failed', error.name);
        outcome = record(apiError(500, 'internal_error', 'Job execution failed.'), 500);
      }
      const { parameters, ...completed } = job;
      await this.storage.put(key, {
        ...completed,
        status: outcome.status < 400 ? 'completed' : 'failed',
        progress: 100,
        completed_at: new Date().toISOString(),
        ...(outcome.status < 400 ? { result: outcome.body } : outcome.body),
      });
    }
    const next = pending.length > 1 ? Math.min(nextExpiry, Date.now() + 10) : nextExpiry;
    if (Number.isFinite(next)) {
      await this.storage.transaction((txn) => this.schedule(txn, next));
    }
  }
}
