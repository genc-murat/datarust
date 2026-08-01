export async function onRequest(context) {
  const { request, env } = context;
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
    } catch {
      // Fall through to HTML asset if fetching markdown asset fails.
    }
  }
  return context.next();
}
