# Publishing the datarust site on Cloudflare Pages

The site is generated from the existing Markdown sources. Documentation remains in `book/src/`, and field notes remain in `blog/`; there is no second content copy to keep in sync.

## Build locally

```sh
npm ci
npm run build
npm run check
npm run preview
```

The deploy-ready static site is written to `dist/`. The preview server listens on `http://127.0.0.1:4173` by default.

Set `SITE_URL` when you want canonical URLs, the sitemap, the feed, and `robots.txt` to use a custom domain:

```sh
SITE_URL=https://datarust.example.com npm run build
```

If it is not set, the build uses `https://datarust.pages.dev`.

## Recommended: Cloudflare Pages Git integration

Import this repository in **Workers & Pages → Create application → Pages → Import an existing Git repository**, then use:

| Setting | Value |
|---|---|
| Production branch | `main` |
| Framework preset | `None` |
| Build command | `npm run build` |
| Build output directory | `dist` |
| Root directory | leave blank (repository root) |
| Build system | v3 |

Add a production environment variable named `SITE_URL` with the final `https://…` origin. Node 22 is pinned by `.node-version`, so local and Pages builds use the same major runtime.

Every push to `main` will create a production deployment. Other branches and pull requests get isolated preview deployments through the Pages Git integration.

## Alternative: Direct Upload with Wrangler

Build first, authenticate Wrangler, and upload the output directory:

```sh
npm ci
npm run build
npx wrangler login
npx wrangler pages project create
npx wrangler pages deploy dist
```

After the project has been created once, `npm run deploy` rebuilds and uploads `dist/`. Cloudflare treats Git-integrated and Direct Upload projects as different setup choices, so choose the workflow you want before creating the Pages project.

## What the build publishes

- `/` — product and learning-path landing page
- `/docs/` — documentation generated from the mdBook source order
- `/blog/` — newest-first index of the numbered field notes
- `/blog/<slug>/` — clean, shareable article URLs
- `/search.json` — client-side search index for docs and articles
- `/feed.xml`, `/sitemap.xml`, `/robots.txt` — discovery metadata
- `/_headers`, `/_redirects`, `/404.html` — Cloudflare Pages response rules and fallback page

Run `npm run check` after changing the generator or content. It checks generated pages, internal links, anchors, metadata, and the numbered article count.
