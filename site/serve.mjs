import { createReadStream } from 'node:fs';
import { access, readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');
const HOST = '127.0.0.1';
const PORT = Number(process.env.PORT || 4173);
const TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webmanifest': 'application/manifest+json',
  '.xml': 'application/xml; charset=utf-8',
};

async function getCustomHeaders(pathname) {
  try {
    const content = await readFile(path.join(DIST, '_headers'), 'utf8');
    const headers = {};
    let currentPattern = null;

    for (const line of content.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      if (!line.startsWith(' ') && !line.startsWith('\t')) {
        currentPattern = trimmed;
      } else if (currentPattern) {
        const colonIdx = trimmed.indexOf(':');
        if (colonIdx !== -1) {
          const key = trimmed.slice(0, colonIdx).trim();
          const val = trimmed.slice(colonIdx + 1).trim();

          let match = false;
          if (currentPattern === '/*') {
            match = true;
          } else if (currentPattern === '/') {
            match = pathname === '/' || pathname === '/index.html';
          } else if (currentPattern.endsWith('/*')) {
            const prefix = currentPattern.slice(0, -1);
            match = pathname.startsWith(prefix);
          } else {
            match = pathname === currentPattern;
          }

          if (match) {
            headers[key] = val;
          }
        }
      }
    }
    return headers;
  } catch {
    return {};
  }
}

async function existingFile(pathname) {
  const decoded = decodeURIComponent(pathname).replace(/^\/+/, '');
  const relative = !decoded || pathname.endsWith('/') ? path.join(decoded, 'index.html') : decoded;
  const candidates = [path.join(DIST, relative), path.join(DIST, `${relative}.html`), path.join(DIST, relative, 'index.html')];
  for (const candidate of candidates) {
    if (!candidate.startsWith(`${DIST}${path.sep}`) && candidate !== path.join(DIST, 'index.html')) continue;
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Keep looking.
    }
  }
  return null;
}

createServer(async (request, response) => {
  const pathname = new URL(request.url, `http://${HOST}`).pathname;
  const file = await existingFile(pathname);
  if (!file) {
    response.writeHead(404, { 'Content-Type': TYPES['.html'] });
    createReadStream(path.join(DIST, '404.html')).pipe(response);
    return;
  }
  const customHeaders = await getCustomHeaders(pathname);
  const defaultContentType = TYPES[path.extname(file)] || 'application/octet-stream';
  const headers = {
    'Content-Type': customHeaders['Content-Type'] || defaultContentType,
    ...customHeaders,
  };
  response.writeHead(200, headers);
  createReadStream(file).pipe(response);
}).listen(PORT, HOST, () => {
  console.log(`datarust site preview: http://${HOST}:${PORT}`);
});

