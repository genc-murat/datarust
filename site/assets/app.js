const root = document.documentElement;
const themeButton = document.querySelector('[data-theme-toggle]');
const savedTheme = localStorage.getItem('datarust-theme');

if (savedTheme === 'light' || savedTheme === 'dark') root.dataset.theme = savedTheme;

themeButton?.addEventListener('click', () => {
  const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const current = root.dataset.theme || (systemDark ? 'dark' : 'light');
  const next = current === 'dark' ? 'light' : 'dark';
  root.dataset.theme = next;
  localStorage.setItem('datarust-theme', next);
});

const menuButton = document.querySelector('[data-menu-toggle]');
const menu = document.querySelector('[data-menu]');
menuButton?.addEventListener('click', () => {
  const open = menu?.classList.toggle('open') || false;
  menuButton.setAttribute('aria-expanded', String(open));
});

const sidebarButton = document.querySelector('[data-sidebar-toggle]');
const sidebar = document.querySelector('[data-sidebar-nav]');
sidebarButton?.addEventListener('click', () => {
  const open = sidebar?.classList.toggle('open') || false;
  sidebarButton.setAttribute('aria-expanded', String(open));
});

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const previous = button.textContent;
    button.textContent = 'Copied';
    window.setTimeout(() => { button.textContent = previous; }, 1300);
  } catch {
    button.textContent = 'Copy failed';
  }
}

document.querySelectorAll('[data-copy-text]').forEach((button) => {
  button.addEventListener('click', () => copyText(button.dataset.copyText, button));
});

document.querySelectorAll('.prose pre').forEach((pre) => {
  const code = pre.querySelector('code');
  if (!code) return;
  const button = document.createElement('button');
  button.className = 'copy-code';
  button.type = 'button';
  button.textContent = 'Copy';
  button.setAttribute('aria-label', 'Copy code');
  button.addEventListener('click', () => copyText(code.textContent, button));
  pre.append(button);
});

const blogFilter = document.querySelector('[data-blog-filter]');
if (blogFilter) {
  const cards = [...document.querySelectorAll('[data-blog-card]')];
  const count = document.querySelector('[data-blog-count]');
  const empty = document.querySelector('[data-blog-empty]');
  blogFilter.addEventListener('input', () => {
    const query = blogFilter.value.trim().toLowerCase();
    let visible = 0;
    cards.forEach((card) => {
      const matches = !query || card.dataset.searchable.includes(query);
      card.hidden = !matches;
      if (matches) visible += 1;
    });
    if (count) count.textContent = `${visible} field note${visible === 1 ? '' : 's'}`;
    if (empty) empty.hidden = visible !== 0;
  });
}

const dialog = document.querySelector('[data-search-dialog]');
const searchInput = document.querySelector('[data-search-input]');
const searchResults = document.querySelector('[data-search-results]');
let searchIndex;

function renderResults(query) {
  if (!searchResults || !searchIndex) return;
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) {
    searchResults.innerHTML = '<p class="search-hint">Try “pipeline”, “encoder”, or “cross-validation”.</p>';
    return;
  }
  const matches = searchIndex
    .map((entry) => {
      const title = entry.title.toLowerCase();
      const haystack = `${entry.title} ${entry.description} ${entry.headings.join(' ')}`.toLowerCase();
      const score = terms.reduce((total, term) => total + (title.includes(term) ? 3 : haystack.includes(term) ? 1 : -10), 0);
      return { entry, score };
    })
    .filter(({ score }) => score >= terms.length)
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  if (!matches.length) {
    searchResults.innerHTML = '<p class="search-empty">No page matched that search.</p>';
    return;
  }
  searchResults.replaceChildren(...matches.map(({ entry }) => {
    const link = document.createElement('a');
    link.className = 'search-result';
    link.href = entry.url;
    const type = document.createElement('small');
    type.textContent = entry.type;
    const title = document.createElement('strong');
    title.textContent = entry.title;
    const description = document.createElement('span');
    description.textContent = entry.description;
    link.append(type, title, description);
    return link;
  }));
}

async function openSearch() {
  if (!dialog) return;
  dialog.showModal();
  searchInput?.focus();
  if (!searchIndex) {
    const response = await fetch('/search.json');
    searchIndex = await response.json();
  }
  renderResults(searchInput?.value || '');
}

document.querySelector('[data-search-open]')?.addEventListener('click', openSearch);
document.querySelector('[data-search-close]')?.addEventListener('click', () => dialog?.close());
searchInput?.addEventListener('input', () => renderResults(searchInput.value));
dialog?.addEventListener('click', (event) => {
  if (event.target === dialog) dialog.close();
});
document.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    openSearch();
  }
});

const tocLinks = [...document.querySelectorAll('.page-toc a')];
if (tocLinks.length && 'IntersectionObserver' in window) {
  const linksById = new Map(tocLinks.map((link) => [decodeURIComponent(link.hash.slice(1)), link]));
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      tocLinks.forEach((link) => link.classList.remove('active'));
      linksById.get(entry.target.id)?.classList.add('active');
    });
  }, { rootMargin: '-20% 0px -70% 0px' });
  linksById.forEach((_, id) => {
    const heading = document.getElementById(id);
    if (heading) observer.observe(heading);
  });
}

function registerWebMcp() {
  const modelContext = navigator.modelContext || (window.navigator && window.navigator.modelContext);
  if (!modelContext) return;

  const tools = [
    {
      name: 'search_docs',
      description: 'Search documentation and field notes for datarust and datarust-profile.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Search term' }
        },
        required: ['query']
      },
      execute: async ({ query }) => {
        if (!searchIndex) {
          const res = await fetch('/search.json');
          searchIndex = await res.json();
        }
        const terms = (query || '').toLowerCase().split(/\s+/).filter(Boolean);
        const matches = searchIndex
          .map((entry) => {
            const title = entry.title.toLowerCase();
            const haystack = `${entry.title} ${entry.description} ${entry.headings.join(' ')}`.toLowerCase();
            const score = terms.reduce((total, term) => total + (title.includes(term) ? 3 : haystack.includes(term) ? 1 : -10), 0);
            return { entry, score };
          })
          .filter(({ score }) => score >= terms.length)
          .sort((a, b) => b.score - a.score)
          .slice(0, 5)
          .map(({ entry }) => entry);
        return { results: matches };
      }
    },
    {
      name: 'get_crate_info',
      description: 'Get details, features, and documentation links for datarust or datarust-profile crates.',
      inputSchema: {
        type: 'object',
        properties: {
          crateName: { type: 'string', enum: ['datarust', 'datarust-profile'], description: 'Crate name' }
        },
        required: ['crateName']
      },
      execute: async ({ crateName }) => {
        if (crateName === 'datarust-profile') {
          return {
            name: 'datarust-profile',
            tagline: 'One-call data profiling and quality reports',
            docs: 'https://datarust.dev/docs/profile/README/',
            cratesUrl: 'https://crates.io/crates/datarust-profile'
          };
        }
        return {
          name: 'datarust',
          tagline: 'Scikit-learn-style preprocessing and classical ML',
          docs: 'https://datarust.dev/docs/',
          cratesUrl: 'https://crates.io/crates/datarust'
        };
      }
    },
    {
      name: 'navigate_to',
      description: 'Navigate to a documentation or field note page on datarust.dev.',
      inputSchema: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Target path (e.g. /docs/, /blog/)' }
        },
        required: ['path']
      },
      execute: async ({ path }) => {
        if (path) window.location.href = path;
        return { success: true, path };
      }
    }
  ];

  if (typeof modelContext.provideContext === 'function') {
    try {
      modelContext.provideContext({ tools });
    } catch {
      // Fallback
    }
  }

  if (typeof modelContext.registerTool === 'function') {
    tools.forEach((tool) => {
      try {
        modelContext.registerTool(tool);
      } catch {
        // Fallback
      }
    });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', registerWebMcp);
} else {
  registerWebMcp();
}

