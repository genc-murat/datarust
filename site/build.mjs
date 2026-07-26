import { execFileSync } from 'node:child_process';
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { marked } from 'marked';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');
const DOCS_ROOT = path.join(ROOT, 'book', 'src');
const BLOG_ROOT = path.join(ROOT, 'blog');
const STATIC_ROOT = path.join(ROOT, 'site', 'static');
const SITE_URL = normalizeOrigin(process.env.SITE_URL || 'https://datarust.dev');
const GITHUB_URL = 'https://github.com/genc-murat/datarust';
const CRATES_URL = 'https://crates.io/crates/datarust';

marked.setOptions({ gfm: true });

function normalizeOrigin(value) {
  const url = new URL(value);
  return url.origin + url.pathname.replace(/\/+$/, '');
}

function escapeHtml(value = '') {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function escapeXml(value = '') {
  return escapeHtml(value);
}

function decodeEntities(value = '') {
  return value
    .replaceAll('&amp;', '&')
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'")
    .replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)));
}

function plainText(value = '') {
  return decodeEntities(value)
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/^\[[^\]]+\]:\s+\S+.*$/gm, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\[[^\]]+\]/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/[*_~>|]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function slugify(value) {
  return plainText(value)
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'section';
}

function excerpt(value, max = 180) {
  const text = plainText(value);
  if (text.length <= max) return text;
  return `${text.slice(0, max).replace(/\s+\S*$/, '')}…`;
}

function getGitDate(relativePath) {
  try {
    const value = execFileSync(
      'git',
      ['log', '-1', '--format=%cI', '--', relativePath],
      { cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] },
    ).trim();
    return value ? new Date(value) : null;
  } catch {
    return null;
  }
}

function formatDate(date) {
  if (!date) return '';
  return new Intl.DateTimeFormat('en', {
    month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC',
  }).format(date);
}

function splitLead(markdown, { subtitle = false } = {}) {
  const lines = markdown.replace(/^\uFEFF/, '').split('\n');
  const headingIndex = lines.findIndex((line) => /^#\s+/.test(line));
  const title = headingIndex >= 0
    ? plainText(lines[headingIndex].replace(/^#\s+/, ''))
    : 'Untitled';

  if (headingIndex >= 0) lines.splice(headingIndex, 1);
  while (lines[0]?.trim() === '') lines.shift();

  let description = '';
  if (subtitle && /^\*[^*].*\*$/.test(lines[0]?.trim() || '')) {
    description = plainText(lines.shift());
    while (lines[0]?.trim() === '') lines.shift();
  }
  if (lines[0]?.trim() === '---') lines.shift();
  while (lines[0]?.trim() === '') lines.shift();

  // A page already gets its semantic H1 from the layout.
  const body = lines.join('\n').replace(/^#\s+/gm, '## ');
  return { title, description, body };
}

function docRoute(relativePath) {
  const normalized = relativePath.replaceAll('\\', '/').replace(/^\.\//, '');
  if (normalized.toLowerCase() === 'readme.md') return '/docs/';
  return `/docs/${normalized.replace(/\.md$/i, '')}/`;
}

function blogSlug(filename) {
  const base = filename
    .replace(/\.md$/i, '')
    .replace(/^\d+-medium-/, '')
    .replace(/^medium-/, '');
  return slugify(base);
}

function splitHref(value) {
  const match = value.match(/^([^?#]*)(\?[^#]*)?(#.*)?$/);
  return { pathname: match?.[1] || '', query: match?.[2] || '', hash: match?.[3] || '' };
}

function rewriteHref(href, context) {
  if (!href || href.startsWith('#') || /^(mailto:|tel:|data:|javascript:)/i.test(href)) return href;

  const legacy = 'https://genc-murat.github.io/datarust';
  if (href === legacy || href === `${legacy}/`) return '/docs/';
  if (href.startsWith(`${legacy}/`)) {
    const legacyPath = href.slice(legacy.length).replace(/\.html(?=($|#|\?))/i, '/');
    return `/docs${legacyPath}`.replace(/\/{2,}/g, '/');
  }
  if (/^https?:\/\//i.test(href)) return href;

  const { pathname, query, hash } = splitHref(href);
  if (context.kind === 'blog' && pathname.startsWith('img/')) {
    return `/blog/${pathname}${query}${hash}`;
  }
  if (context.kind === 'docs' && /\.md$/i.test(pathname)) {
    const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(context.source), pathname));
    return `${docRoute(resolved)}${query}${hash}`;
  }
  if (context.kind === 'docs' && pathname === 'LICENSE') {
    return `${GITHUB_URL}/blob/main/LICENSE`;
  }
  return href;
}

function addHeadingIds(html) {
  const used = new Map();
  const toc = [];
  const rendered = html.replace(/<h([2-4])>([\s\S]*?)<\/h\1>/g, (_, level, inner) => {
    const text = plainText(inner);
    const base = slugify(text);
    const count = used.get(base) || 0;
    used.set(base, count + 1);
    const id = count ? `${base}-${count + 1}` : base;
    toc.push({ level: Number(level), text, id });
    return `<h${level} id="${id}"><a class="heading-anchor" href="#${id}" aria-label="Link to ${escapeHtml(text)}">#</a>${inner}</h${level}>`;
  });
  return { html: rendered, toc };
}

function renderMarkdown(markdown, context) {
  let html = marked.parse(markdown);
  html = html.replace(/(<a\s+[^>]*href=")([^"]+)("[^>]*>)/g, (match, start, href, end) => {
    const nextHref = rewriteHref(decodeEntities(href), context);
    const external = /^https?:\/\//i.test(nextHref);
    const attributes = external ? ' target="_blank" rel="noreferrer"' : '';
    return `${start}${escapeHtml(nextHref)}${end.slice(0, -1)}${attributes}>`;
  });
  html = html.replace(/(<img\s+[^>]*src=")([^"]+)("[^>]*>)/g, (match, start, src, end) => {
    const nextSrc = rewriteHref(decodeEntities(src), context);
    return `${start}${escapeHtml(nextSrc)}${end.slice(0, -1)} loading="lazy" decoding="async">`;
  });
  html = html.replace(/<table>/g, '<div class="table-scroll"><table>')
    .replace(/<\/table>/g, '</table></div>');
  return addHeadingIds(html);
}

function readingMinutes(markdown) {
  const words = plainText(markdown).split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.ceil(words / 220));
}

async function loadDocs() {
  const summary = await readFile(path.join(DOCS_ROOT, 'SUMMARY.md'), 'utf8');
  const sections = [];
  let current = { title: 'Overview', items: [] };
  sections.push(current);

  for (const line of summary.split('\n')) {
    const sectionMatch = line.match(/^#\s+(.+)/);
    if (sectionMatch && sectionMatch[1] !== 'Summary') {
      current = { title: plainText(sectionMatch[1]), items: [] };
      sections.push(current);
      continue;
    }
    const linkMatch = line.match(/^(?:-\s+)?\[([^\]]+)\]\(([^)]+\.md)\)/);
    if (!linkMatch) continue;
    const source = path.posix.normalize(linkMatch[2].replace(/^\.\//, ''));
    current.items.push({ label: plainText(linkMatch[1]), source, route: docRoute(source) });
  }

  const pages = [];
  for (const item of sections.flatMap((section) => section.items)) {
    const markdown = await readFile(path.join(DOCS_ROOT, item.source), 'utf8');
    const lead = splitLead(markdown);
    const rendered = renderMarkdown(lead.body, { kind: 'docs', source: item.source });
    pages.push({
      ...item,
      title: lead.title,
      description: excerpt(lead.body),
      markdown,
      html: rendered.html,
      toc: rendered.toc,
      date: getGitDate(path.posix.join('book/src', item.source)),
    });
  }
  return { sections, pages };
}

async function loadBlog() {
  const filenames = (await readdir(BLOG_ROOT))
    .filter((name) => name.endsWith('.md'))
    .sort((a, b) => a.localeCompare(b, 'en', { numeric: true }));
  const posts = [];

  for (const filename of filenames) {
    const markdown = await readFile(path.join(BLOG_ROOT, filename), 'utf8');
    const lead = splitLead(markdown, { subtitle: true });
    const rendered = renderMarkdown(lead.body, { kind: 'blog', source: filename });
    const numberMatch = filename.match(/^(\d+)-/);
    const slug = blogSlug(filename);
    posts.push({
      filename,
      number: numberMatch ? Number(numberMatch[1]) : null,
      isRelease: !numberMatch,
      slug,
      route: `/blog/${slug}/`,
      title: lead.title,
      description: lead.description || excerpt(lead.body),
      markdown,
      html: rendered.html,
      toc: rendered.toc,
      minutes: readingMinutes(markdown),
      date: getGitDate(path.posix.join('blog', filename)),
    });
  }
  return posts;
}

function navLink(label, href, currentPath) {
  const active = href === '/' ? currentPath === '/' : currentPath.startsWith(href);
  return `<a href="${href}"${active ? ' aria-current="page"' : ''}>${label}</a>`;
}

function icon(name) {
  const paths = {
    search: '<circle cx="11" cy="11" r="7"></circle><path d="m20 20-4-4"></path>',
    sun: '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.66 6.34l1.41-1.41"></path>',
    menu: '<path d="M4 7h16M4 12h16M4 17h16"></path>',
    arrow: '<path d="M5 12h14M14 7l5 5-5 5"></path>',
  };
  return `<svg aria-hidden="true" viewBox="0 0 24 24">${paths[name]}</svg>`;
}

function pageShell({ title, description, pathname, body, bodyClass = '' }) {
  const fullTitle = title === 'datarust' ? title : `${title} · datarust`;
  const canonical = new URL(pathname, `${SITE_URL}/`).href;
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(fullTitle)}</title>
  <meta name="description" content="${escapeHtml(description)}">
  <meta name="theme-color" content="#151514">
  <meta property="og:type" content="website">
  <meta property="og:title" content="${escapeHtml(fullTitle)}">
  <meta property="og:description" content="${escapeHtml(description)}">
  <meta property="og:url" content="${escapeHtml(canonical)}">
  <meta property="og:image" content="${escapeHtml(`${SITE_URL}/og.png`)}">
  <meta property="og:image:width" content="1731">
  <meta property="og:image:height" content="909">
  <meta property="og:image:alt" content="datarust — Classical ML, native to Rust">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:image" content="${escapeHtml(`${SITE_URL}/og.png`)}">
  <link rel="canonical" href="${escapeHtml(canonical)}">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="manifest" href="/manifest.webmanifest">
  <link rel="alternate" type="application/atom+xml" title="datarust field notes" href="/feed.xml">
  <link rel="preload" href="/assets/styles.css" as="style">
  <link rel="stylesheet" href="/assets/styles.css">
  <script src="/assets/app.js" defer></script>
</head>
<body class="${bodyClass}">
  <a class="skip-link" href="#main-content">Skip to content</a>
  <header class="site-header">
    <div class="header-inner">
      <a class="brand" href="/" aria-label="datarust home"><span class="brand-mark">dr</span><span>datarust</span></a>
      <button class="icon-button menu-button" type="button" aria-label="Open navigation" aria-expanded="false" data-menu-toggle>${icon('menu')}</button>
      <nav class="site-nav" aria-label="Primary navigation" data-menu>
        ${navLink('Docs', '/docs/', pathname)}
        ${navLink('Field notes', '/blog/', pathname)}
        <a href="https://docs.rs/datarust" target="_blank" rel="noreferrer">API</a>
        <a href="${GITHUB_URL}" target="_blank" rel="noreferrer">GitHub</a>
      </nav>
      <div class="header-actions">
        <button class="search-button" type="button" data-search-open>${icon('search')}<span>Search</span><kbd>⌘K</kbd></button>
        <button class="icon-button" type="button" aria-label="Change color theme" data-theme-toggle>${icon('sun')}</button>
      </div>
    </div>
  </header>
  <main id="main-content">${body}</main>
  <footer class="site-footer">
    <div><a class="brand footer-brand" href="/"><span class="brand-mark">dr</span><span>datarust</span></a><p>Scikit-learn-style preprocessing and classical ML, built for Rust.</p></div>
    <div class="footer-links"><a href="/docs/quickstart/">Quick start</a><a href="/blog/">Field notes</a><a href="${CRATES_URL}">crates.io</a><a href="${GITHUB_URL}">Source</a></div>
    <p class="footer-meta">MIT licensed · Built from the Markdown in this repository.</p>
  </footer>
  <dialog class="search-dialog" data-search-dialog>
    <div class="search-panel">
      <div class="search-input-wrap">${icon('search')}<input type="search" placeholder="Search docs and field notes…" aria-label="Search" autocomplete="off" data-search-input><button type="button" data-search-close aria-label="Close search">Esc</button></div>
      <div class="search-results" data-search-results><p class="search-hint">Try “pipeline”, “encoder”, or “cross-validation”.</p></div>
    </div>
  </dialog>
</body>
</html>`;
}

function articleCard(post, { compact = false } = {}) {
  const label = post.number ? `Field note ${String(post.number).padStart(2, '0')}` : 'Release story';
  const meta = [formatDate(post.date), `${post.minutes} min read`].filter(Boolean).join(' · ');
  return `<article class="article-card${compact ? ' compact' : ''}" data-blog-card data-searchable="${escapeHtml(`${post.title} ${post.description}`.toLowerCase())}">
    <a class="card-link" href="${post.route}" aria-label="Read ${escapeHtml(post.title)}"></a>
    <div class="article-label">${escapeHtml(label)}</div>
    <h2>${escapeHtml(post.title)}</h2>
    <p>${escapeHtml(post.description)}</p>
    <div class="card-meta"><span>${escapeHtml(meta)}</span><span class="read-arrow">Read ${icon('arrow')}</span></div>
  </article>`;
}

function homePage(posts, docs) {
  const numbered = posts.filter((post) => post.number).sort((a, b) => b.number - a.number);
  const latest = numbered.slice(0, 3);
  const docsCount = docs.pages.length;
  const notesCount = numbered.length;
  return pageShell({
    title: 'datarust',
    description: 'Scikit-learn-style preprocessing and classical machine learning in pure Rust.',
    pathname: '/',
    bodyClass: 'home-page',
    body: `
      <section class="hero shell">
        <div class="hero-copy">
          <p class="eyebrow">Classical ML, native to Rust</p>
          <h1>Ship the pipeline.<br><span>Skip the runtime baggage.</span></h1>
          <p class="hero-lede">datarust brings the familiar scikit-learn workflow to a small, transparent Rust library—from preprocessing and feature selection to models, metrics, and persistence.</p>
          <div class="hero-actions"><a class="button primary" href="/docs/quickstart/">Start building ${icon('arrow')}</a><a class="button secondary" href="/blog/">Read the field notes</a></div>
          <div class="install-line"><code>cargo add datarust</code><button type="button" data-copy-text="cargo add datarust">Copy</button></div>
        </div>
        <div class="hero-visual" aria-label="A datarust machine-learning pipeline">
          <div class="pipeline-window"><div class="window-bar"><span></span><span></span><span></span><small>pipeline.rs</small></div><pre><code><span class="code-muted">// one fitted object, end to end</span>
<span class="code-keyword">let mut</span> model = Pipeline::new()
    .push(<span class="code-string">"scale"</span>, StandardScaler::new())
    .with_estimator(LogisticRegression::new());

model.fit(&amp;x_train, &amp;y_train)?;
<span class="code-keyword">let</span> predictions = model.predict(&amp;x_test)?;</code></pre></div>
          <div class="pipeline-steps"><span>raw rows</span><b>→</b><span>transform</span><b>→</b><span>predict</span></div>
        </div>
      </section>
      <section class="stats-band"><div class="shell stats"><div><strong>0</strong><span>default dependencies</span></div><div><strong>${docsCount}</strong><span>documentation pages</span></div><div><strong>${notesCount}</strong><span>practical field notes</span></div><div><strong>1.70+</strong><span>supported Rust</span></div></div></section>
      <section class="section shell">
        <div class="section-heading"><div><p class="eyebrow">Learn the shape of the library</p><h2>From first matrix to fitted pipeline</h2></div><a class="text-link" href="/docs/">All documentation ${icon('arrow')}</a></div>
        <div class="path-grid">
          <a href="/docs/quickstart/"><span>01</span><h3>Start small</h3><p>Create a matrix, fit a transformer, train a model, and evaluate it.</p></a>
          <a href="/docs/guide/compose/"><span>02</span><h3>Compose the workflow</h3><p>Keep preprocessing and estimation together with pipelines and column transforms.</p></a>
          <a href="/docs/performance/"><span>03</span><h3>Understand the tradeoffs</h3><p>See the benchmarks, feature flags, and places where another tool may fit better.</p></a>
        </div>
      </section>
      <section class="section shell notes-preview">
        <div class="section-heading"><div><p class="eyebrow">Latest field notes</p><h2>The failures that teach the useful parts</h2></div><a class="text-link" href="/blog/">Browse all ${notesCount} ${icon('arrow')}</a></div>
        <div class="article-grid">${latest.map((post) => articleCard(post)).join('')}</div>
      </section>
      <section class="cta-band shell"><div><p class="eyebrow">No ceremony required</p><h2>Build your first Rust ML workflow.</h2></div><a class="button light" href="/docs/quickstart/">Open the quick start ${icon('arrow')}</a></section>
    `,
  });
}

function blogIndex(posts) {
  const numbered = posts.filter((post) => post.number).sort((a, b) => b.number - a.number);
  const release = posts.find((post) => post.isRelease);
  return pageShell({
    title: 'Field notes',
    description: 'Practical, human-written lessons from building preprocessing and classical machine-learning workflows in Rust.',
    pathname: '/blog/',
    bodyClass: 'listing-page',
    body: `
      <section class="page-hero shell narrow-left"><p class="eyebrow">Field notes · ${numbered.length} articles</p><h1>Small experiments.<br>Expensive lessons.</h1><p>Practical stories about models that ran, metrics that looked convincing, and the details that changed what the result actually meant.</p></section>
      <section class="shell blog-controls"><label for="blog-filter">Find a field note</label><div class="filter-input">${icon('search')}<input id="blog-filter" type="search" placeholder="Filter by topic…" data-blog-filter></div><p aria-live="polite" data-blog-count>${numbered.length} field notes</p></section>
      ${release ? `<section class="shell featured-story"><div><p class="eyebrow">Release story</p><h2>${escapeHtml(release.title)}</h2><p>${escapeHtml(release.description)}</p><a class="text-link" href="${release.route}">Read the story ${icon('arrow')}</a></div><div class="feature-number">v0.6</div></section>` : ''}
      <section class="shell section article-grid blog-grid" data-blog-grid>${numbered.map((post) => articleCard(post)).join('')}<p class="empty-state" data-blog-empty hidden>No field note matched that search.</p></section>
    `,
  });
}

function tocMarkup(toc) {
  const items = toc.filter((heading) => heading.level <= 3);
  if (items.length < 2) return '';
  return `<aside class="page-toc" aria-label="On this page"><p>On this page</p><ol>${items.map((item) => `<li class="level-${item.level}"><a href="#${item.id}">${escapeHtml(item.text)}</a></li>`).join('')}</ol></aside>`;
}

function docSidebar(sections, activeRoute) {
  return `<aside class="docs-sidebar" aria-label="Documentation"><div class="sidebar-heading"><span>Documentation</span><button type="button" data-sidebar-toggle aria-expanded="false">Browse</button></div><nav data-sidebar-nav>${sections.filter((section) => section.items.length).map((section) => `<section><h2>${escapeHtml(section.title)}</h2>${section.items.map((item) => `<a href="${item.route}"${item.route === activeRoute ? ' aria-current="page"' : ''}>${escapeHtml(item.label)}</a>`).join('')}</section>`).join('')}</nav></aside>`;
}

function pageNeighbors(pages, index, kind) {
  const previous = pages[index - 1];
  const next = pages[index + 1];
  if (!previous && !next) return '';
  return `<nav class="page-neighbors" aria-label="${kind} navigation">
    ${previous ? `<a class="previous" href="${previous.route}"><span>← Previous</span><strong>${escapeHtml(previous.title)}</strong></a>` : '<span></span>'}
    ${next ? `<a class="next" href="${next.route}"><span>Next →</span><strong>${escapeHtml(next.title)}</strong></a>` : '<span></span>'}
  </nav>`;
}

function docsPage(page, index, docs) {
  const updated = formatDate(page.date);
  return pageShell({
    title: page.title,
    description: page.description,
    pathname: page.route,
    bodyClass: 'content-page docs-page',
    body: `<div class="docs-layout shell-wide">
      ${docSidebar(docs.sections, page.route)}
      <article class="prose-wrap"><header class="article-header"><p class="eyebrow">Documentation</p><h1>${escapeHtml(page.title)}</h1>${updated ? `<p class="article-meta">Updated ${escapeHtml(updated)}</p>` : ''}</header><div class="prose">${page.html}</div><div class="edit-note"><span>Something unclear?</span><a href="${GITHUB_URL}/edit/main/book/src/${page.source}">Edit this page on GitHub →</a></div>${pageNeighbors(docs.pages, index, 'Documentation')}</article>
      ${tocMarkup(page.toc)}
    </div>`,
  });
}

function blogPage(post, index, orderedPosts) {
  const label = post.number ? `Field note ${String(post.number).padStart(2, '0')}` : 'Release story';
  const meta = [formatDate(post.date), `${post.minutes} min read`].filter(Boolean).join(' · ');
  return pageShell({
    title: post.title,
    description: post.description,
    pathname: post.route,
    bodyClass: 'content-page blog-post',
    body: `<div class="article-layout shell-wide">
      <article class="prose-wrap blog-prose"><header class="article-header"><a class="back-link" href="/blog/">← All field notes</a><p class="eyebrow">${escapeHtml(label)}</p><h1>${escapeHtml(post.title)}</h1><p class="article-deck">${escapeHtml(post.description)}</p><p class="article-meta">${escapeHtml(meta)}</p></header><div class="prose">${post.html}</div>${pageNeighbors(orderedPosts, index, 'Field note')}</article>
      ${tocMarkup(post.toc)}
    </div>`,
  });
}

function notFoundPage() {
  return pageShell({
    title: 'Page not found',
    description: 'The requested datarust page could not be found.',
    pathname: '/404.html',
    bodyClass: 'not-found-page',
    body: `<section class="not-found shell"><p class="error-code">404</p><h1>This row is out of bounds.</h1><p>The page may have moved, or the link may be pointing at something that no longer exists.</p><div><a class="button primary" href="/docs/">Open the docs</a><a class="button secondary" href="/blog/">Browse field notes</a></div></section>`,
  });
}

async function writePage(route, html) {
  const destination = route === '/'
    ? path.join(DIST, 'index.html')
    : route === '/404.html'
      ? path.join(DIST, '404.html')
      : path.join(DIST, route.replace(/^\//, ''), 'index.html');
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, html);
}

function feedXml(posts) {
  const ordered = posts.filter((post) => post.number).sort((a, b) => b.number - a.number);
  const updated = ordered.find((post) => post.date)?.date?.toISOString() || '2026-01-01T00:00:00.000Z';
  return `<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>datarust field notes</title>
  <subtitle>Practical lessons from classical machine learning in Rust.</subtitle>
  <link href="${escapeXml(`${SITE_URL}/blog/`)}"/>
  <link href="${escapeXml(`${SITE_URL}/feed.xml`)}" rel="self"/>
  <id>${escapeXml(`${SITE_URL}/blog/`)}</id>
  <updated>${updated}</updated>
  ${ordered.slice(0, 20).map((post) => `<entry><title>${escapeXml(post.title)}</title><link href="${escapeXml(`${SITE_URL}${post.route}`)}"/><id>${escapeXml(`${SITE_URL}${post.route}`)}</id><updated>${post.date?.toISOString() || updated}</updated><summary>${escapeXml(post.description)}</summary></entry>`).join('\n  ')}
</feed>`;
}

function sitemapXml(entries) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${entries.map(({ route, date }) => `  <url><loc>${escapeXml(`${SITE_URL}${route}`)}</loc>${date ? `<lastmod>${date.toISOString().slice(0, 10)}</lastmod>` : ''}</url>`).join('\n')}
</urlset>`;
}

async function build() {
  const [docs, posts, cargoToml] = await Promise.all([
    loadDocs(),
    loadBlog(),
    readFile(path.join(ROOT, 'Cargo.toml'), 'utf8'),
  ]);
  const crateVersion = cargoToml.match(/^version\s*=\s*"([^"]+)"/m)?.[1] || 'latest';

  await rm(DIST, { recursive: true, force: true });
  await mkdir(DIST, { recursive: true });
  await cp(STATIC_ROOT, DIST, { recursive: true });
  await cp(path.join(ROOT, 'site', 'assets'), path.join(DIST, 'assets'), { recursive: true });
  await cp(path.join(BLOG_ROOT, 'img'), path.join(DIST, 'blog', 'img'), { recursive: true });

  await writePage('/', homePage(posts, docs));
  await writePage('/blog/', blogIndex(posts));
  await writePage('/404.html', notFoundPage());

  for (const [index, page] of docs.pages.entries()) {
    await writePage(page.route, docsPage(page, index, docs));
  }

  const orderedPosts = posts
    .filter((post) => post.number)
    .sort((a, b) => a.number - b.number);
  for (const post of posts) {
    const neighbors = post.number ? orderedPosts : posts.filter((candidate) => candidate.isRelease);
    const index = neighbors.findIndex((candidate) => candidate.route === post.route);
    await writePage(post.route, blogPage(post, index, neighbors));
  }

  const searchEntries = [
    ...docs.pages.map((page) => ({ type: 'Docs', title: page.title, description: page.description, url: page.route, headings: page.toc.map((item) => item.text) })),
    ...posts.map((post) => ({ type: post.number ? `Field note ${String(post.number).padStart(2, '0')}` : 'Release story', title: post.title, description: post.description, url: post.route, headings: post.toc.map((item) => item.text) })),
  ];
  await writeFile(path.join(DIST, 'search.json'), JSON.stringify(searchEntries));
  await writeFile(path.join(DIST, 'feed.xml'), feedXml(posts));
  await writeFile(path.join(DIST, 'sitemap.xml'), sitemapXml([
    { route: '/', date: null },
    { route: '/docs/', date: docs.pages[0]?.date },
    { route: '/blog/', date: orderedPosts.at(-1)?.date },
    ...docs.pages.slice(1).map((page) => ({ route: page.route, date: page.date })),
    ...posts.map((post) => ({ route: post.route, date: post.date })),
  ]));
  await writeFile(path.join(DIST, 'robots.txt'), `User-agent: *\nAllow: /\n\nSitemap: ${SITE_URL}/sitemap.xml\n`);
  await writeFile(path.join(DIST, 'manifest.webmanifest'), JSON.stringify({
    name: 'datarust', short_name: 'datarust', start_url: '/', display: 'standalone',
    background_color: '#f5f2ea', theme_color: '#151514',
    icons: [{ src: '/favicon.svg', sizes: 'any', type: 'image/svg+xml' }],
  }, null, 2));

  console.log(`Built datarust v${crateVersion}: ${docs.pages.length} docs pages and ${posts.length} blog posts → dist/`);
}

await build();
