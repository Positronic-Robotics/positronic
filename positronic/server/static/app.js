// Positronic Dataset Viewer — frontend for positronic_server.py
//
// Pages
// -----
// The server renders three page types from Jinja templates:
//   index.html    — flat episode list (default home page)
//   grouped.html  — aggregated table (e.g. leaderboard), set via home_page config
//   episode.html  — single episode with Rerun viewer + static data sidebar
//
// Templates set window globals before this script's DOMContentLoaded fires:
//   window.API_ENDPOINT   — override for /api/episodes (grouped.html sets /api/groups/{name})
//   window.IS_GROUPED_TABLE — true on grouped.html, changes View link behavior
//   window.EPISODES_URL   — where View links point on grouped pages
//   window.VIEW_LABEL     — button text override
//
// Data flow
// ---------
// On DOMContentLoaded, initEpisodesTable() runs:
//   1. Polls /api/dataset_status until the server finishes loading the dataset
//   2. Calls loadEpisodes({}) → fetches from API_ENDPOINT (default /api/episodes)
//      Response shape: { columns, episodes, group_filters?, default_sort? }
//        columns:  array of {key, label, filter?, renderer?, align?, subtitle?}
//        episodes: array of [episodeId, [cell0, cell1, ...], groupFilters?]
//          Each cell is either a scalar (string/number) or [sortValue, displayValue]
//        group_filters: {filterKey: {label, values[]}} — only for grouped tables
//   3. Parses URL query params → splits into serverFilters vs clientFilters
//   4. If serverFilters found, re-fetches with those params
//   5. Renders filter dropdowns, table header, and table body
//
// Filters
// -------
// Server filters (state.serverFilters):
//   Defined by GroupTableConfig.group_filter_keys on the Python side. Changing one
//   re-fetches from the server — needed for grouped tables where aggregation must
//   happen server-side. On flat episode tables, these come from URL params that
//   match group_filter_keys (e.g. ?equipment=DROID).
//
// Client filters (state.filters):
//   Columns with filter:true in the table config. buildFiltersData() collects unique
//   values per column from loaded episodes. Changing one just re-runs populateTable()
//   which calls getFilteredEpisodes() to hide non-matching rows — no server round-trip.
//
// Both are synced to the URL via syncURL() so filtered views can be shared/bookmarked.
//
// Cell rendering
// --------------
// Each cell value can be a plain scalar or [sortValue, displayValue] tuple.
// Columns may have a renderer config (from Python RendererConfig):
//   'badge' — colored label (e.g. Pass/Fail), options map raw value → {label, variant}
//   'icon'  — image + text, options map raw value → {src, label?, tags?, class?, href?}
//            tags: string[] — rendered as chips below the name
//            class: string  — CSS class added to the cell (for row styling via tr:has)
//            href: string   — makes the name a link; {value} is replaced with the raw value
//            _tagStyles (on options root): {tagName: {bg, color, border}} for tag chip colors
// Without a renderer, the display value (or scalar) is shown as plain text.
//
// Episode detail page
// -------------------
// episode.html calls initializeSidebar(staticData) with the episode's static JSON.
// This renders a collapsible key-value tree in the sidebar panel.
// initSidebar() (called on every page) handles sidebar resize/toggle/scroll state,
// persisted in localStorage. It no-ops on pages without a .sidebar element.

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  sort: { columnIndex: null, direction: 'desc' },
  filters: {},           // client-side column filters
  serverFilters: {},     // server-side group filters (sent to /api/episodes)
  episodes: [],          // current episode data
  columns: [],           // column definitions from server
  filtersData: {},       // unique values per filterable column
};

// ---------------------------------------------------------------------------
// Data fetching
// ---------------------------------------------------------------------------

async function fetchJSON(url) {
  const response = await fetch(url);
  if (response.status === 202) return null;  // dataset still loading
  return response.json();
}

async function loadDatasetInfo() {
  const data = await fetchJSON('/api/dataset_info');
  if (!data) return;
  document.getElementById('dataset-stats').innerHTML =
    `<p><strong>${data.num_episodes}</strong> episodes.</p>`;
}

async function loadEpisodes(filters = {}) {
  const endpoint = new URL(window.API_ENDPOINT || '/api/episodes', window.location.origin);
  for (const [key, value] of Object.entries(filters)) {
    endpoint.searchParams.append(key, value);
  }
  return fetchJSON(endpoint);
}

// ---------------------------------------------------------------------------
// URL sync
// ---------------------------------------------------------------------------

function syncURL() {
  const url = new URL(window.location);
  url.search = new URLSearchParams({ ...state.serverFilters, ...state.filters }).toString();
  window.history.replaceState({}, '', url);
}

function readFiltersFromURL(serverFilterKeys) {
  const params = new URLSearchParams(window.location.search);
  for (const [key, value] of params.entries()) {
    if (serverFilterKeys.has(key)) {
      state.serverFilters[key] = value;
    } else if (state.filtersData[key]?.includes(value)) {
      state.filters[key] = value;
    }
  }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

async function pollUntilLoaded() {
  const statusEl = document.getElementById('loading-status');
  const statsEl = document.getElementById('dataset-stats');

  statusEl.classList.add('show');
  statsEl.innerHTML = 'Checking dataset status...';

  return new Promise((resolve) => {
    const interval = setInterval(async () => {
      const status = await fetchJSON('/api/dataset_status');
      if (!status || status.loading) return;
      clearInterval(interval);
      statusEl.classList.remove('show');
      resolve(status.loaded);
    }, 2000);
  });
}

async function initEpisodesTable() {
  const container = document.getElementById('episodes-container');
  if (!container) return;  // episode detail page has no table
  const table = container.querySelector('.episodes-table');
  const loadingEl = container.querySelector('.loading');

  // Check if dataset is ready
  const status = await fetchJSON('/api/dataset_status');
  if (!status) return;

  if (status.loading) {
    const loaded = await pollUntilLoaded();
    if (!loaded) {
      document.getElementById('dataset-stats').innerHTML =
        '<span class="error-message">Failed to load dataset</span>';
      container.innerHTML = '<div class="loading error-message">Dataset loading failed</div>';
      return;
    }
  } else if (!status.loaded) {
    document.getElementById('dataset-stats').innerHTML =
      '<span class="error-message">Failed to load dataset</span>';
    container.innerHTML = '<div class="loading error-message">Dataset loading failed</div>';
    return;
  }

  loadDatasetInfo();

  // First fetch discovers server-side filter keys
  const initial = await loadEpisodes({});
  if (!initial) return;

  const { columns, group_filters: groupFilters, default_sort: defaultSort } = initial;
  state.columns = columns;
  state.filtersData = buildFiltersData(initial.episodes, columns);

  // Parse URL into server/client filters
  const serverFilterKeys = groupFilters ? new Set(Object.keys(groupFilters)) : new Set();
  readFiltersFromURL(serverFilterKeys);

  // Re-fetch with server filters if any were set from URL
  if (Object.keys(state.serverFilters).length > 0) {
    const filtered = await loadEpisodes(state.serverFilters);
    state.episodes = filtered ? filtered.episodes : initial.episodes;
  } else {
    state.episodes = initial.episodes;
  }

  if (defaultSort) {
    const sortIndex = columns.findIndex((c) => c.key === defaultSort.column);
    if (sortIndex !== -1) {
      state.sort = { columnIndex: String(sortIndex), direction: defaultSort.direction || 'desc' };
    }
  }

  renderServerFilters(groupFilters);
  renderClientFilters(columns);
  renderTableHeader(columns);
  populateTable(columns);

  loadingEl.remove();
  table.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Filtering & sorting
// ---------------------------------------------------------------------------

function buildFiltersData(episodes, columns) {
  const result = {};
  for (const [index, column] of Object.entries(columns)) {
    if (!column.filter) continue;
    const values = new Set();
    for (const [, episodeData] of episodes) {
      const v = episodeData[index];
      if (v !== null && v !== undefined) values.add(String(v));
    }
    result[column.key] = Array.from(values);
  }
  return result;
}

function getFilteredEpisodes(columns) {
  const { sort: sortState, filters } = state;

  let result = state.episodes.filter(([, episodeData]) =>
    Object.entries(filters).every(([filterKey, value]) => {
      const colIdx = columns.findIndex((c) => c.key === filterKey);
      return String(episodeData[colIdx]) === value;
    })
  );

  if (sortState.columnIndex !== null) {
    result.sort((a, b) => {
      const aVal = sortableValue(a[1][sortState.columnIndex]);
      const bVal = sortableValue(b[1][sortState.columnIndex]);
      if (aVal === bVal) return 0;
      return sortState.direction === 'asc'
        ? (aVal < bVal ? -1 : 1)
        : (aVal > bVal ? -1 : 1);
    });
  }

  return result;
}

function sortableValue(entity) {
  if (entity === null || entity === undefined) return '';
  return Array.isArray(entity) ? entity[0] : entity;
}

// ---------------------------------------------------------------------------
// Cell rendering
// ---------------------------------------------------------------------------

function cellValue(entity, column) {
  if (column.renderer?.type === 'badge') {
    return renderBadge(entity, column.renderer.options?.[entity]);
  }
  if (column.renderer?.type === 'icon') {
    return renderIcon(entity, column.renderer.options);
  }
  const value = Array.isArray(entity) ? entity[1] : entity;
  return value ?? '';
}

function renderBadge(value, options) {
  if (value === null || value === undefined) return '';
  const VARIANTS = ['success', 'warning', 'danger', 'default'];
  const span = document.createElement('span');
  span.classList.add('badge');
  span.textContent = options?.label ?? String(value);
  const variant = options?.variant && VARIANTS.includes(options.variant) ? options.variant : 'default';
  span.classList.add(`badge--${variant}`);
  return span;
}

function renderIcon(value, options) {
  if (value === null || value === undefined) return '';
  const text = Array.isArray(value) ? value[1] : String(value);
  const raw = Array.isArray(value) ? value[0] : value;
  const cfg = options?.[raw];

  const nameRow = document.createElement('span');
  nameRow.classList.add('cell-with-icon');
  if (cfg?.class) nameRow.classList.add(cfg.class);
  if (cfg?.src) {
    const img = document.createElement('img');
    img.src = cfg.src;
    img.alt = '';
    img.classList.add('cell-icon');
    nameRow.appendChild(img);
  }
  const nameEl = cfg?.href ? document.createElement('a') : document.createElement('span');
  nameEl.textContent = cfg?.label ?? text;
  if (cfg?.href) nameEl.href = cfg.href.replace('{value}', encodeURIComponent(raw));
  nameRow.appendChild(nameEl);

  if (!cfg?.tags?.length) return nameRow;

  const wrapper = document.createElement('div');
  if (cfg.class) wrapper.classList.add(cfg.class);
  wrapper.appendChild(nameRow);

  const meta = document.createElement('div');
  meta.className = 'cell-tags';
  for (const tag of cfg.tags) {
    const chip = document.createElement('span');
    chip.className = 'cell-tag';
    const style = cfg.tagStyles?.[tag] || options?._tagStyles?.[tag];
    if (style) {
      chip.style.background = style.bg;
      chip.style.color = style.color;
      chip.style.borderColor = style.border;
    }
    chip.textContent = tag;
    meta.appendChild(chip);
  }
  wrapper.appendChild(meta);
  return wrapper;
}

// ---------------------------------------------------------------------------
// Table rendering
// ---------------------------------------------------------------------------

function createCell(content, isHeader = false) {
  const cell = document.createElement(isHeader ? 'th' : 'td');
  if (content instanceof HTMLElement) {
    cell.appendChild(content);
  } else {
    cell.textContent = String(content);
  }
  return cell;
}

function renderTableHeader(columns) {
  const headerRow = document.querySelector('.episodes-table thead tr');
  const headerCells = [];

  for (const [columnIndex, { label, subtitle, align, sortable }] of Object.entries(columns)) {
    let content;
    if (subtitle) {
      content = document.createElement('span');
      content.textContent = label;
      const info = document.createElement('span');
      info.className = 'column-info';
      info.textContent = '\u24D8';
      info.dataset.tooltip = subtitle;
      info.addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.column-info.active').forEach(el => {
          if (el !== info) el.classList.remove('active');
        });
        info.classList.toggle('active');
      });
      content.appendChild(info);
    }

    const th = createCell(content || label, true);
    if (sortable !== false) {
      th.classList.add('sortable');
      th.dataset.sort = columnIndex;
      th.addEventListener('click', () => onSortClick(th, columnIndex, columns));
    }
    if (align) th.style.textAlign = align;
    headerCells.push(th);
  }

  headerRow.prepend(...headerCells);

  if (state.sort.columnIndex !== null) {
    headerCells[state.sort.columnIndex]?.classList.add(`sorted-${state.sort.direction}`);
  }
}

function onSortClick(th, columnIndex, columns) {
  const headerRow = th.parentElement;
  headerRow.querySelectorAll('th.sortable').forEach((el) => {
    el.classList.remove('sorted-asc', 'sorted-desc');
  });

  if (state.sort.columnIndex === columnIndex) {
    state.sort.direction = state.sort.direction === 'desc' ? 'asc' : 'desc';
  } else {
    state.sort.columnIndex = columnIndex;
    state.sort.direction = 'desc';
  }

  th.classList.add(`sorted-${state.sort.direction}`);
  populateTable(columns);
}

function populateTable(columns) {
  const tableBody = document.querySelector('.episodes-table tbody');
  const filtered = getFilteredEpisodes(columns);

  tableBody.innerHTML = '';
  for (const [episodeIndex, episodeData, groupFilters] of filtered) {
    const row = document.createElement('tr');

    for (const [i, entity] of episodeData.entries()) {
      const td = createCell(cellValue(entity, columns[i]));
      if (columns[i].align) td.style.textAlign = columns[i].align;
      row.appendChild(td);
    }

    const viewCell = createCell('');
    const viewLink = document.createElement('a');
    viewLink.className = 'btn btn-primary btn-small';
    if (window.IS_GROUPED_TABLE) {
      const filters = { ...groupFilters, ...state.serverFilters, ...state.filters };
      const episodesUrl = window.EPISODES_URL || '/';
      viewLink.href = `${episodesUrl}?${new URLSearchParams(filters).toString()}`;
    } else {
      viewLink.href = `/episode/${episodeIndex}`;
    }
    viewLink.textContent = window.VIEW_LABEL || 'View';
    viewCell.appendChild(viewLink);
    row.appendChild(viewCell);

    tableBody.appendChild(row);
  }

  // Store filtered episode IDs for episode viewer navigation
  if (!window.IS_GROUPED_TABLE) {
    const hasFilters = Object.keys(state.serverFilters).length > 0 ||
      filtered.length < state.episodes.length;
    if (hasFilters) {
      sessionStorage.setItem('filteredEpisodeIds', JSON.stringify(filtered.map(([id]) => id)));
      sessionStorage.setItem('episodesReferrerUrl', window.location.href);
    } else {
      sessionStorage.removeItem('filteredEpisodeIds');
      sessionStorage.removeItem('episodesReferrerUrl');
    }
  }
}

// ---------------------------------------------------------------------------
// Filter controls
// ---------------------------------------------------------------------------

function createFilterDropdown({ id, label, options, onChange }) {
  const container = document.createElement('div');
  container.classList.add('control-group', 'control-group--grow');

  const labelEl = document.createElement('label');
  labelEl.htmlFor = id;
  labelEl.textContent = label;
  container.appendChild(labelEl);

  const select = document.createElement('select');
  select.id = id;
  select.addEventListener('change', onChange);
  select.append(...options);
  container.appendChild(select);

  return container;
}

function createOption(value, label) {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = label;
  return option;
}

function renderServerFilters(groupFilters) {
  if (!groupFilters) return;
  const controlsBar = document.querySelector('.controls-bar');

  for (const [filterKey, filterData] of Object.entries(groupFilters)) {
    const options = [
      createOption('-1', 'All'),
      ...filterData.values.map((v) => createOption(v, v)),
    ];

    const container = createFilterDropdown({
      id: `filter-${filterKey}`,
      label: filterData.label,
      options,
      onChange: async (event) => {
        if (event.target.value === '-1') {
          delete state.serverFilters[filterKey];
        } else {
          state.serverFilters[filterKey] = event.target.value;
        }
        syncURL();
        const result = await loadEpisodes(state.serverFilters);
        if (result) {
          state.episodes = result.episodes;
          populateTable(result.columns);
        }
      },
    });

    if (state.serverFilters[filterKey]) {
      container.querySelector('select').value = state.serverFilters[filterKey];
    }

    controlsBar.appendChild(container);
  }
}

function renderClientFilters(columns) {
  const controlsBar = document.querySelector('.controls-bar');

  for (const [filterKey, values] of Object.entries(state.filtersData)) {
    const column = columns.find((c) => c.key === filterKey);
    const options = [
      createOption('-1', 'All'),
      ...values.map((v) => createOption(v, column.renderer?.options[v]?.label ?? v)),
    ];

    const container = createFilterDropdown({
      id: `filter-${filterKey}`,
      label: column.label,
      options,
      onChange: (event) => {
        if (event.target.value === '-1') {
          delete state.filters[filterKey];
        } else {
          state.filters[filterKey] = event.target.value;
        }
        syncURL();
        populateTable(columns);
      },
    });

    // Pre-select from URL
    if (state.filters[filterKey]) {
      container.querySelector('select').value = state.filters[filterKey];
    }

    controlsBar.appendChild(container);
  }
}

// ---------------------------------------------------------------------------
// Tooltip dismiss (column info popups)
// ---------------------------------------------------------------------------

document.addEventListener('click', () => {
  document.querySelectorAll('.column-info.active').forEach(el => el.classList.remove('active'));
});

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initEpisodesTable().catch((err) => console.error('Failed to initialize:', err));
  initSidebar();
});

// ---------------------------------------------------------------------------
// Sidebar (episode detail page)
// ---------------------------------------------------------------------------

function initSidebar() {
  const sidebar = document.querySelector('.sidebar');
  if (!sidebar) return;

  const sidebarToggler = document.querySelector('.sidebar-toggler');
  const sidebarResizer = document.querySelector('.sidebar-resizer');
  const sidebarContent = document.querySelector('.sidebar-content');
  const keyColumnResizer = document.querySelector('.key-column-resizer');
  const episodeViewer = document.getElementById('viewer-container');
  const sidebarState = loadSidebarState();

  setSidebarWidth(sidebarState.sidebarWidth);
  setKeyColumnWidth(sidebarState.keyColumnWidth);
  sidebarContent.scrollTop = sidebarState.scrollTop;
  toggleSidebar(sidebarState.isExpanded);

  sidebarToggler.addEventListener('click', () => {
    toggleSidebar(!sidebar.classList.contains('expanded'));
    save();
  });

  setupResizer(sidebarResizer, (e) => {
    setSidebarWidth(document.body.clientWidth - e.clientX);
    save();
  });
  setupResizer(keyColumnResizer, (e) => {
    const offsetX = sidebarState.sidebarWidth - (document.body.clientWidth - e.clientX);
    setKeyColumnWidth(Math.min(Math.max(50, offsetX), sidebarState.sidebarWidth - 50));
    save();
  });

  sidebarContent.addEventListener('scroll', () => {
    sidebarState.scrollTop = sidebarContent.scrollTop;
    save();
  });

  function setupResizer(resizer, onMove) {
    resizer.addEventListener('mousedown', () => {
      episodeViewer.style.zIndex = -1;
      sidebar.classList.add('isResizing');
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', () => {
        document.removeEventListener('mousemove', onMove);
        episodeViewer.style.zIndex = '';
        sidebar.classList.remove('isResizing');
        document.body.style.userSelect = '';
      }, { once: true });
    });
  }

  function setSidebarWidth(width) {
    sidebarState.sidebarWidth = Math.max(100, width);
    sidebar.style.width = `${sidebarState.sidebarWidth}px`;
  }

  function toggleSidebar(expanded) {
    sidebar.classList.toggle('expanded', expanded);
    sidebar.style.width = expanded ? `${sidebarState.sidebarWidth}px` : '0px';
    sidebarState.isExpanded = expanded;
  }

  function setKeyColumnWidth(width) {
    sidebar.style.setProperty('--dataset-key-column-width', `${width}px`);
    sidebarState.keyColumnWidth = width;
  }

  function save() {
    localStorage.setItem('sidebarState', JSON.stringify(sidebarState));
  }
}

function loadSidebarState() {
  const defaults = { isExpanded: false, sidebarWidth: 300, keyColumnWidth: 150, scrollTop: 0 };
  const saved = JSON.parse(localStorage.getItem('sidebarState'));
  return { ...defaults, ...saved };
}

// ---------------------------------------------------------------------------
// Sidebar content (static episode metadata)
// ---------------------------------------------------------------------------

function initializeSidebar(staticData) {
  const tbody = document.querySelector('.sidebar-content-wrapper tbody');

  tbody.insertAdjacentHTML('beforeend', renderLevel('', staticData));
  document.querySelectorAll('.expand-button').forEach((button) => {
    button.addEventListener('click', () => {
      const table = button.parentElement.querySelector('table');
      if (table) {
        const isShown = button.dataset.expanded === 'true';
        button.dataset.expanded = !isShown;
        table.style.display = isShown ? 'none' : 'table';
      }
    });
  });
}

function isNestable(value) {
  return typeof value === 'object' && value !== null;
}

function _formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function renderValue(value, level = 0) {
  const expandButton = level === 0 ? '<button class="expand-button">▶</button>' : '';

  switch (typeof value) {
    case 'string':
      return `"${value}"`;
    case 'number':
      return value.toFixed(2).replace(/\.00$/, '');
    case 'boolean':
      return `<input type="checkbox" ${value ? 'checked' : ''} onclick="return false" />`;
    case 'object':
      if (value === null) return 'null';
      if (value.__download__) {
        const icon = value.type === 'bytes' ? '📦' : '📄';
        return `${icon} <a href="${value.__download__}" target="_blank" style="color:#8cb4ff">${_formatSize(value.size)}</a>`;
      }
      if (Array.isArray(value)) {
        return expandButton + `[${value.map((item) => renderValue(item, level + 1)).join(', ')}]`;
      }
      return expandButton + '{...}';
    default:
      return value;
  }
}

function renderLevel(key, value) {
  if (!isNestable(value)) {
    return `<tr><td>${key}</td><td>${renderValue(value)}</td></tr>`;
  }

  let html = '';
  for (const [k, val] of Object.entries(value)) {
    html += `
      <tr>
        <td>${k}</td>
        <td>
          ${renderValue(val)}
          ${isNestable(val) ? `<table><tbody>${renderLevel(k, val)}</tbody></table>` : ''}
        </td>
      </tr>
    `;
  }
  return html;
}
