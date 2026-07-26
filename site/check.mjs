import { access, readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');

async function filesBelow(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = await Promise.all(entries.map((entry) => {
    const target = path.join(directory, entry.name);
    return entry.isDirectory() ? filesBelow(target) : [target];
  }));
  return files.flat();
}

function routeForFile(file) {
  const relative = path.relative(DIST, file).replaceAll(path.sep, '/');
  if (relative === 'index.html') return '/';
  if (relative === '404.html') return '/404.html';
  return `/${relative.replace(/index\.html$/, '')}`;
}

async function resolvePublicPath(pathname) {
  let relative = decodeURIComponent(pathname).replace(/^\/+/, '');
  if (!relative || pathname.endsWith('/')) relative = path.join(relative, 'index.html');
  const candidates = [
    path.join(DIST, relative),
    path.join(DIST, `${relative}.html`),
    path.join(DIST, relative, 'index.html'),
  ];
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next Pages-compatible route shape.
    }
  }
  return null;
}

function idsIn(html) {
  return [...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]);
}

async function check() {
  const files = await filesBelow(DIST);
  const htmlFiles = files.filter((file) => file.endsWith('.html'));
  const errors = [];
  const htmlByFile = new Map();

  for (const file of htmlFiles) {
    const html = await readFile(file, 'utf8');
    htmlByFile.set(file, html);
    const relative = path.relative(ROOT, file);
    if (!/<title>[^<]+<\/title>/.test(html)) errors.push(`${relative}: missing title`);
    if (!/<meta name="description" content="[^"]+">/.test(html)) errors.push(`${relative}: missing description`);
    if (!/<h1[ >]/.test(html)) errors.push(`${relative}: missing h1`);
    const ids = idsIn(html);
    const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index);
    if (duplicates.length) errors.push(`${relative}: duplicate ids ${[...new Set(duplicates)].join(', ')}`);
  }

  for (const [file, html] of htmlByFile) {
    const currentRoute = routeForFile(file);
    const references = [...html.matchAll(/\s(?:href|src)="([^"]+)"/g)].map((match) => match[1]);
    for (const reference of references) {
      if (/^(https?:|mailto:|tel:|data:)/i.test(reference)) continue;
      if (reference.includes('.md')) {
        errors.push(`${currentRoute}: source Markdown link leaked into output (${reference})`);
        continue;
      }
      const url = new URL(reference, `https://local.test${currentRoute}`);
      const target = url.pathname === currentRoute.split('#')[0] ? file : await resolvePublicPath(url.pathname);
      if (!target) {
        errors.push(`${currentRoute}: broken link ${reference}`);
        continue;
      }
      if (url.hash && target.endsWith('.html')) {
        const targetHtml = htmlByFile.get(target) || await readFile(target, 'utf8');
        const id = decodeURIComponent(url.hash.slice(1));
        if (!idsIn(targetHtml).includes(id)) errors.push(`${currentRoute}: missing fragment ${reference}`);
      }
    }
  }

  const numberedSources = (await readdir(path.join(ROOT, 'blog'))).filter((name) => /^\d+-.*\.md$/.test(name));
  const numberedPages = htmlFiles.filter((file) => /\/blog\/[^/]+\/index\.html$/.test(file)).length - 1;
  if (numberedPages !== numberedSources.length) {
    errors.push(`blog count mismatch: ${numberedSources.length} numbered sources, ${numberedPages} generated pages`);
  }

  if (errors.length) {
    console.error(`Site check failed with ${errors.length} error(s):\n${errors.map((error) => `- ${error}`).join('\n')}`);
    process.exitCode = 1;
    return;
  }
  console.log(`Site check passed: ${htmlFiles.length} HTML pages and ${files.length} total files.`);
}

await check();
