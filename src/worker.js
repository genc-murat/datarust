/**
 * Cloudflare Worker entry point for datarust.dev
 *
 * Handles Accept: text/markdown content negotiation by serving pre-built
 * markdown files alongside the default HTML static assets.
 *
 * When a request includes Accept: text/markdown, the worker looks for a
 * co-located .md file (e.g. /docs/guide/ → /docs/guide/index.md) and
 * returns it with Content-Type: text/markdown and an x-markdown-tokens
 * header. All other requests pass through to static assets unchanged.
 */

export default {
  async fetch(request, env) {
    const accept = request.headers.get('accept') || '';

    if (accept.includes('text/markdown')) {
      const url = new URL(request.url);
      let mdPath = url.pathname;

      if (mdPath.endsWith('/')) {
        mdPath += 'index.md';
      } else if (mdPath.endsWith('.html')) {
        mdPath = mdPath.replace(/\.html$/, '.md');
      } else if (!mdPath.includes('.')) {
        mdPath += '/index.md';
      }

      const mdUrl = new URL(mdPath, request.url);

      try {
        // Fetch static asset by URL string to avoid passing Accept: text/markdown header into ASSETS fetch
        const mdResponse = await env.ASSETS.fetch(mdUrl.toString());

        if (mdResponse.ok) {
          const text = await mdResponse.text();
          const tokens = Math.ceil(text.length / 4);
          const headers = new Headers(mdResponse.headers);
          headers.set('content-type', 'text/markdown; charset=utf-8');
          headers.set('x-markdown-tokens', String(tokens));
          headers.set('vary', 'Accept');
          return new Response(text, { status: 200, headers });
        }
      } catch {
        // Fall through to default HTML asset on error.
      }
    }

    return env.ASSETS.fetch(request);
  },
};
