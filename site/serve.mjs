import { createReadStream } from 'node:fs';
import { access } from 'node:fs/promises';
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
  response.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
  createReadStream(file).pipe(response);
}).listen(PORT, HOST, () => {
  console.log(`datarust site preview: http://${HOST}:${PORT}`);
});
