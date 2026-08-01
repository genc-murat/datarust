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

  const blogSources = (await readdir(path.join(ROOT, 'blog'))).filter((name) => name.endsWith('.md'));
  const blogPages = htmlFiles.filter((file) => /\/blog\/[^/]+\/index\.html$/.test(file)).length;
  if (blogPages !== blogSources.length) {
    errors.push(`blog count mismatch: ${blogSources.length} sources, ${blogPages} generated pages`);
  }

  const headersFile = path.join(DIST, '_headers');
  try {
    const headersContent = await readFile(headersFile, 'utf8');
    let inHomepageSection = false;
    let homepageLinkHeader = null;
    for (const line of headersContent.split('\n')) {
      const trimmed = line.trim();
      if (!line.startsWith(' ') && !line.startsWith('\t')) {
        inHomepageSection = (trimmed === '/');
      } else if (inHomepageSection && trimmed.startsWith('Link:')) {
        homepageLinkHeader = trimmed.slice(5).trim();
      }
    }
    if (!homepageLinkHeader) {
      errors.push('_headers: missing Link header for homepage (/)');
    } else {
      const linkUris = [...homepageLinkHeader.matchAll(/<([^>]+)>/g)].map((m) => m[1]);
      if (linkUris.length === 0) {
        errors.push('_headers: invalid Link header syntax for homepage');
      }
      for (const uri of linkUris) {
        const target = await resolvePublicPath(uri);
        if (!target) {
          errors.push(`_headers Link header: broken link ${uri}`);
        }
      }
    }
  } catch {
    errors.push('_headers: missing _headers file in dist');
  }

  for (const htmlFile of htmlFiles) {
    const mdFile = htmlFile.replace(/\.html$/, '.md');
    try {
      const mdContent = await readFile(mdFile, 'utf8');
      if (!mdContent.trim()) {
        errors.push(`${path.relative(ROOT, mdFile)}: empty markdown file`);
      }
    } catch {
      errors.push(`${path.relative(ROOT, htmlFile)}: missing corresponding markdown file (.md) for agent negotiation`);
    }
  }

  const middlewareFile = path.join(DIST, 'functions', '_middleware.js');
  try {
    await access(middlewareFile);
  } catch {
    errors.push('functions/_middleware.js: missing Cloudflare Pages markdown negotiation middleware');
  }

  const apiCatalogFile = path.join(DIST, '.well-known', 'api-catalog');
  try {
    const catalogRaw = await readFile(apiCatalogFile, 'utf8');
    const catalog = JSON.parse(catalogRaw);
    if (!Array.isArray(catalog.linkset) || catalog.linkset.length === 0) {
      errors.push('.well-known/api-catalog: linkset array is missing or empty');
    } else {
      for (const entry of catalog.linkset) {
        if (!entry.anchor) {
          errors.push('.well-known/api-catalog: entry missing anchor');
        }
        if (!entry['service-doc']) {
          errors.push('.well-known/api-catalog: entry missing service-doc relation');
        }
      }
    }
  } catch (err) {
    errors.push(`.well-known/api-catalog: failed to read or parse RFC 9727 linkset (${err.message})`);
  }

  const mcpServerCardFile = path.join(DIST, '.well-known', 'mcp', 'server-card.json');
  try {
    const cardRaw = await readFile(mcpServerCardFile, 'utf8');
    const card = JSON.parse(cardRaw);
    if (!card.serverInfo || !card.serverInfo.name || !card.serverInfo.version) {
      errors.push('.well-known/mcp/server-card.json: serverInfo object with name and version is required');
    }
    if (!card.endpoint) {
      errors.push('.well-known/mcp/server-card.json: endpoint property is required');
    }
    if (!card.capabilities) {
      errors.push('.well-known/mcp/server-card.json: capabilities property is required');
    }
  } catch (err) {
    errors.push(`.well-known/mcp/server-card.json: failed to read or parse SEP-1649 MCP server card (${err.message})`);
  }


  if (errors.length) {

    console.error(`Site check failed with ${errors.length} error(s):\n${errors.map((error) => `- ${error}`).join('\n')}`);
    process.exitCode = 1;
    return;
  }
  console.log(`Site check passed: ${htmlFiles.length} HTML pages (and matching .md pages) across ${files.length} total files.`);


}

await check();
