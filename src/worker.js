export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === '/mcp' || url.pathname === '/mcp/') {
      const accept = request.headers.get('accept') || '';
      if (accept.includes('application/json')) {
        return env.ASSETS.fetch(new Request(new URL('/.well-known/mcp/server-card.json', request.url)));
      }
      return Response.redirect(new URL('/docs/mcp/', request.url), 302);
    }
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
        return new Response(`Error: mdResponse.status was ${mdResponse.status} for URL ${mdUrl.toString()}`, { status: 500 });
      } catch (e) {
        return new Response(`Exception: ${e.message}`, { status: 500 });
      }
    }
    // Fall back to normal asset serving
    return env.ASSETS.fetch(request);
  }
};
