// <positronic-replay>: a list of episodes, and the player of the episode a visitor opens.
//
// Usage
//   <script type="module" src="/static/replay.js"></script>
//   <positronic-replay src="replay.json"></positronic-replay>
//   <positronic-replay layout="compact" src="https://example.org/v/token/replay.json"></positronic-replay>
//   <positronic-replay mode="full" dock link-param="episode">
//     <script type="application/json">{"columns": [...], "rows": [...]}</script>
//     <div slot="details">…</div>
//   </positronic-replay>
//
// Attributes
//   src          a manifest URL. Without it, the element reads its child <script type="application/json">.
//   layout       `full` (default): the list as a table, with the player above it or in a dock.
//                `compact`: a tally, a strip of runs, a screen that opens above the list, and the list.
//   mode         `video` (default): the cameras only. `full`: a row with `rrd` also plays its 3D replay.
//                A row with no `rrd` plays as video in either mode.
//   dock         (full layout) the player opens at the bottom of the window when a visitor opens a row.
//   dock-below   (dock) a selector of the host's bar; the dock and the list share the window below it.
//   link-param   the URL search parameter that names the open row, so a link opens it.
//   camera-names a JSON object that maps a camera key to the name the visitor sees.
//
// Manifest (`src`)
//   {title, url, about: [{key, label, value}], columns: [{key, label}], episodes: [{index, values: {key: value},
//    duration_s, videos: [{camera, url, src, width, height, duration_s}], phone?, rrd?}]}
//   A video's `src`, `phone` and `rrd` are relative to the manifest. `phone` is a video object or a URL.
//   The list shows an index column, the manifest's columns, and a duration column.
//
// Inline rows
//   {title?, about?, columns: [{key, label, type?, decimals?, unit?, hover?, filter?}],
//    rows: [{id, label?, values: {key: value}, duration_s?, cameras?: [{camera, label?, src, width, height,
//    duration_s}], phone?, rrd?, href?}]}
//   URLs are relative to the document. `href` is the row's own page: the first text cell links it.
//   A column `type` is `text` (default), `number` (with `decimals` and `unit`), `badge` or `icon`.
//   A value is a scalar, or {text, key?, color?, ink?, icon?, title?}: `key` is what a filter compares,
//   `color` paints a badge, `icon` is the image of an icon cell.
//   `filter` is `true` or {all?, options?: [[value, label], …]}; without options, every value of the column.
//
// Media
//   The element loads only the URLs a row names, and builds no URL from another one. A row with no media
//   shows "not available". A file that answers 404 shows "gone"; one that answers 202 shows "building" and
//   is asked for again after its `Retry-After`. The 3D replay plays in the rerun viewer beside this file,
//   whose page adds no browser history entry. A screen under 961 px wide plays a row's `phone` video in
//   place of its cameras and its 3D replay, and the dock covers the page there.
//
// Events and methods
//   A `replay-action` event, with `detail.action`, reports each control a visitor uses:
//   open {row, from: Row|Step|Link}, close {row, via: Button|Escape}, filter {column, value},
//   more {via: Button|Scroll}, resize {via: Drag|Key}, details {open}, full {row}, play {row, playing},
//   seek {row}, speed {row, value}. A link open is reported first, so the host can show the element.
//   `show({column key: value, …})` sets the filters; a value the column does not offer shows all.
//   The `details` slot shows in the dock beside the player, behind a Details button.
//   The `actions` slot shows in the compact head.
//
// Theme
//   --replay-accent, --replay-border, --replay-muted, --replay-hover, --replay-current, --replay-surface,
//   --replay-surface-ink, --replay-success, --replay-failure (the compact runs strip), --replay-motion (the
//   compact screen's open time), --replay-sticky-top (where the column names stick), and --list-max and
//   --video-max (the most the compact list and its videos may take). Each element has a `part` to style.
import { cellKey, cellText, DURATION_KEY, INDEX_KEY, readInline, readManifest } from './replay/data.js';
import { cellContent, element } from './replay/cells.js';
import { EpisodePlayer } from './replay/media.js';
import { STYLE } from './replay/style.js';

const TAG = 'positronic-replay';
// The rerun release whose viewer sits beside this file, under `rerun/<release>/`.
export const RERUN_VERSION = '0.30.0';
const VIEWER_URL = new URL(`rerun/${RERUN_VERSION}/embedded.html`, import.meta.url).href;
const PAGE = 50;
const NEXT_PAGE_AHEAD = '0px 0px 400px 0px';
// The width under which the dock covers the page, as the styles say.
const NARROW = '(max-width: 960px)';
const STRIP_KEY = 'positronic-replay.strip';
const STRIP_ROWS = 3;
const DOCK_MIN_PX = 160;
const HOLD_OPEN_MS = 10000;
// The compact layout counts the runs whose `outcome` is `success`, and colours each run by its outcome.
const OUTCOME_KEY = 'outcome';
const SUCCESS = 'success';

const slug = (text) =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
const outcomePart = (row) => {
  const outcome = slug(cellKey(row.values[OUTCOME_KEY]));
  return outcome ? ` outcome-${outcome}` : '';
};
const plural = (count, noun) => `${count} ${noun}${count === 1 ? '' : 's'}`;
const seconds = (value) => `${Number(value).toFixed(1)} s`;

// A page the list loads with no scroll since the last one is no visitor's scroll.
const SCROLL_KEYS = new Set(['ArrowDown', 'PageDown', 'End', ' ']);
let scrolledSinceLoad = false;
const scrolled = (event) => {
  if (event.isTrusted) scrolledSinceLoad = true;
};
window.addEventListener('wheel', scrolled, { capture: true, passive: true });
window.addEventListener('touchmove', scrolled, { capture: true, passive: true });
window.addEventListener(
  'keydown',
  (event) => {
    if (SCROLL_KEYS.has(event.key) && !event.composedPath()[0].closest?.('input, select, textarea, button'))
      scrolled(event);
  },
  { capture: true, passive: true },
);

// One dock is open on a page at a time.
let openDock = null;

function savedStrip() {
  try {
    const strip = Number(window.localStorage.getItem(STRIP_KEY));
    return strip > 0 ? strip : null;
  } catch {
    return null;
  }
}

function saveStrip(strip) {
  try {
    window.localStorage.setItem(STRIP_KEY, String(strip));
  } catch {
    // A private window keeps the height for this page only.
  }
}

function setLinkParam(name, value) {
  if (!name) return;
  const url = new URL(window.location);
  if (value === null) url.searchParams.delete(name);
  else url.searchParams.set(name, value);
  window.history.replaceState(window.history.state, '', url);
}

export class PositronicReplay extends HTMLElement {
  static observedAttributes = ['src', 'layout', 'mode'];

  #list = null;
  #request = 0;
  #loaded = null;
  #waitForScript = null;
  #shown = PAGE;
  #selects = [];
  #count = null;
  #rows = new Map();
  #current = null;
  #player = null;
  #strip = null;
  #end = null;
  #narrow = window.matchMedia(NARROW);

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    window.addEventListener('resize', () => {
      if (this.#docked && this.#current) this.#resize(this.#strip);
    });
  }

  connectedCallback() {
    this.#load();
  }

  disconnectedCallback() {
    this.#player?.stop();
    if (openDock === this) openDock = null;
    document.removeEventListener('keydown', this.#onKey);
  }

  attributeChangedCallback(name) {
    if (!this.isConnected) return;
    if (name === 'src' || !this.#list) this.#load();
    else this.#render();
  }

  get #layout() {
    return this.getAttribute('layout') === 'compact' ? 'compact' : 'full';
  }

  get #fullMode() {
    return this.getAttribute('mode') === 'full';
  }

  get #docked() {
    return this.#layout === 'full' && this.hasAttribute('dock');
  }

  /** Sets the filters to `filters`, by column key. A value a filter does not offer shows all. */
  show(filters) {
    for (const select of this.#selects) {
      const value = filters[select.dataset.key];
      if (value === undefined) continue;
      select.value = [...select.options].some((option) => option.value === value) ? value : '';
    }
    this.#shown = PAGE;
    this.#refresh();
    if (this.#current && this.#docked) this.#reveal(this.#current);
  }

  #load() {
    const src = this.getAttribute('src');
    const key = src ?? 'inline';
    if (key === this.#loaded) return;
    const request = ++this.#request;
    if (!src) {
      const script = this.querySelector(':scope > script[type="application/json"]');
      if (!script) {
        // The parser can connect the element before it adds the children.
        this.#waitForScript ??= new MutationObserver(() => this.#load());
        this.#waitForScript.observe(this, { childList: true });
        return;
      }
      this.#waitForScript?.disconnect();
      this.#loaded = key;
      this.#take(() => readInline(JSON.parse(script.textContent), document.baseURI));
      return;
    }
    this.#loaded = key;
    const url = new URL(src, document.baseURI);
    this.#status('Loading the replay…');
    fetch(url)
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json();
      })
      .then((manifest) => {
        if (request === this.#request) this.#take(() => readManifest(manifest, url.href));
      })
      .catch((error) => {
        if (request === this.#request) this.#status(`The replay did not load: ${error.message}`);
      });
  }

  #take(read) {
    try {
      this.#list = read();
    } catch (error) {
      this.#status(`The replay did not load: ${error.message}`);
      return;
    }
    this.#render();
  }

  #status(message) {
    this.shadowRoot.replaceChildren(element('style', {}, [STYLE]), element('p', { part: 'status' }, [message]));
  }

  #act(action, detail = {}) {
    this.dispatchEvent(
      new CustomEvent('replay-action', { bubbles: true, composed: true, detail: { action, ...detail } }),
    );
  }

  #cameraName = (key) => {
    try {
      return JSON.parse(this.getAttribute('camera-names') ?? '{}')[key] ?? key;
    } catch {
      return key;
    }
  };

  #newPlayer(root) {
    const player = new EpisodePlayer(root, {
      viewerUrl: VIEWER_URL,
      full: this.#fullMode,
      narrow: this.#narrow,
      act: (action, detail) => this.#act(action, detail),
      frameKeys: this.#onKey,
      cameraName: this.#cameraName,
    });
    this.#narrow.addEventListener('change', () => {
      if (this.#current && this.#player === player) this.#play(this.#current, { autoplay: false });
    });
    return player;
  }

  #render() {
    this.#player?.stop();
    this.#current = null;
    this.#shown = PAGE;
    this.#selects = [];
    this.#count = null;
    this.#end?.disconnect();
    this.#end = null;
    const { rows } = this.#list;
    const header = this.#header();
    if (!rows.length) {
      this.shadowRoot.replaceChildren(
        element('style', {}, [STYLE]),
        ...header,
        element('p', { part: 'status' }, ['This replay holds no episodes.']),
      );
      return;
    }
    if (this.#layout === 'compact') this.#renderCompact();
    else this.#renderFull(header);
  }

  #header() {
    const { title, about } = this.#list;
    if (!title || this.#layout === 'compact') return [];
    const facts = about.map((fact) =>
      element('div', { 'data-key': fact.key }, [
        element('dt', {}, [fact.label]),
        element('dd', {}, [cellText(fact.value)]),
      ]),
    );
    return [
      element('header', { part: 'header' }, [
        element('h1', {}, [title]),
        ...(facts.length ? [element('dl', { part: 'about' }, facts)] : []),
      ]),
    ];
  }

  /** The text under the player: the row's label, each column's value, and the duration. */
  #summary(row) {
    const columns = this.#list.columns.filter((column) => column.key !== INDEX_KEY && column.key !== DURATION_KEY);
    const parts = [row.label, ...columns.map((column) => `${column.label}: ${cellText(row.values[column.key])}`)];
    if (row.durationS !== null) parts.push(seconds(row.durationS));
    return parts.join(' · ');
  }

  #play(row, options) {
    this.#player.show(row, this.#summary(row), options);
  }

  // ── The full layout ──────────────────────────────────────────────────────

  #renderFull(header) {
    const table = this.#table();
    const scroller = element('div', { part: 'table-scroll' }, [table]);
    const more = element('button', { type: 'button', part: 'more', hidden: '' });
    more.addEventListener('click', (event) => this.#showMore('Button', event));
    const filters = this.#filters();
    const playerRoot = element('section', { part: this.#docked ? 'dock-player' : 'player' });
    const dock = this.#docked ? this.#dock(playerRoot) : null;
    this.shadowRoot.replaceChildren(
      element('style', {}, [STYLE]),
      ...header,
      ...(this.#docked ? [] : [playerRoot]),
      ...(filters ? [filters] : []),
      scroller,
      more,
      ...(dock ? [dock] : []),
    );
    this.#stickyHead(scroller, table);
    this.#player = this.#newPlayer(playerRoot);
    if (window.IntersectionObserver) {
      this.#end = new IntersectionObserver(
        (entries) => entries.some((entry) => entry.isIntersecting) && this.#showMore('Scroll'),
        {
          rootMargin: NEXT_PAGE_AHEAD,
        },
      );
    }
    this.#refresh();
    const linked = this.#linkedRow();
    if (this.#docked) {
      if (linked) this.#open(linked, 'Link');
    } else {
      this.#select(linked ?? this.#list.rows[0]);
    }
  }

  #linkedRow() {
    const name = this.getAttribute('link-param');
    const id = name ? new URLSearchParams(window.location.search).get(name) : null;
    return id === null ? null : (this.#list.rows.find((row) => row.id === id) ?? null);
  }

  #filters() {
    const filtered = this.#list.columns.filter((column) => column.filter);
    if (!filtered.length) return null;
    this.#selects = filtered.map((column) => {
      const select = element('select', { 'data-key': column.key, 'aria-label': column.label }, [
        element('option', { value: '' }, [column.filter.all]),
        ...column.filter.options.map((option) => element('option', { value: option.value }, [option.label])),
      ]);
      select.addEventListener('change', (event) => {
        this.#shown = PAGE;
        this.#refresh();
        if (event.isTrusted) this.#act('filter', { column: column.key, value: select.value });
      });
      return select;
    });
    this.#count = element('span', { part: 'count' });
    const controls = filtered.map((column, at) =>
      element('label', { part: 'filter' }, [
        element('span', { part: 'filter-label' }, [column.label]),
        this.#selects[at],
      ]),
    );
    return element('div', { part: 'filters' }, [...controls, this.#count]);
  }

  #table() {
    const { columns, rows } = this.#list;
    const head = element(
      'tr',
      {},
      columns.map((column) =>
        element('th', { scope: 'col', part: `head-cell ${column.type}`, title: column.hover }, [column.label]),
      ),
    );
    const linkColumn = columns.find((column) => column.type === 'text');
    this.#rows = new Map();
    const body = rows.map((row) => {
      const cells = columns.map((column) => {
        const content = cellContent(column, row.values[column.key], column === linkColumn ? row.href : null);
        return element('td', { part: `cell ${column.type}`, 'data-label': column.label }, content);
      });
      const tr = element('tr', { part: 'row', 'data-id': row.id }, cells);
      tr.addEventListener('click', (event) => this.#rowClicked(event, row));
      this.#rows.set(row, tr);
      return tr;
    });
    return element('table', { part: 'table' }, [element('thead', {}, [head]), element('tbody', {}, body)]);
  }

  #rowClicked(event, row) {
    const link = event.composedPath().find((node) => node instanceof HTMLAnchorElement);
    if (link && (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)) return;
    event.preventDefault();
    if (this.#docked) this.#open(row, 'Row', event);
    else this.#select(row, event);
  }

  // A sticky row inside the sideways scroller sticks to the scroller, so the copy that stays under the host's
  // bar sits outside it.
  #stickyHead(scroller, table) {
    if (!window.ResizeObserver) return;
    const copy = element('table', { part: 'table' }, [table.tHead.cloneNode(true)]);
    const head = element('div', { part: 'sticky-head', 'aria-hidden': 'true' }, [copy]);
    scroller.before(head);
    const columns = [...table.tHead.rows[0].cells];
    const copies = [...copy.tHead.rows[0].cells];
    const fit = () => {
      copy.style.width = `${table.getBoundingClientRect().width}px`;
      columns.forEach((column, at) => {
        copies[at].style.width = `${column.getBoundingClientRect().width}px`;
      });
      const height = copy.getBoundingClientRect().height;
      head.style.height = `${height}px`;
      head.style.marginBottom = `-${height}px`;
      head.scrollLeft = scroller.scrollLeft;
    };
    const observer = new ResizeObserver(fit);
    columns.forEach((column) => observer.observe(column));
    scroller.addEventListener('scroll', () => (head.scrollLeft = scroller.scrollLeft), { passive: true });
  }

  #matches() {
    const wanted = this.#selects.filter((select) => select.value).map((select) => [select.dataset.key, select.value]);
    return this.#list.rows.filter((row) => wanted.every(([key, value]) => cellKey(row.values[key]) === value));
  }

  #refresh() {
    if (this.#layout === 'compact') return;
    const matches = this.#matches();
    const shown = new Set(matches.slice(0, this.#shown));
    for (const [row, tr] of this.#rows) tr.hidden = !shown.has(row);
    if (this.#count) this.#count.textContent = plural(matches.length, 'episode');
    const more = this.#part('more');
    const left = matches.length - this.#shown;
    more.hidden = left <= 0;
    more.textContent = `Show ${Math.min(left, PAGE)} more of ${left}`;
    this.#end?.unobserve(more);
    if (!more.hidden) this.#end?.observe(more);
    if (this.#docked && this.#current) {
      if (!matches.includes(this.#current)) this.#close();
      else this.#showPosition(matches);
    }
  }

  /** `via` names how the visitor asked: the button, or a scroll to the end of the list. */
  #showMore(via, event = null) {
    this.#shown += PAGE;
    this.#refresh();
    if (via === 'Scroll' && !scrolledSinceLoad) return;
    if (via === 'Button' && !event?.isTrusted) return;
    scrolledSinceLoad = false;
    this.#act('more', { via });
  }

  /** Shows the page of the matches that holds `row`. */
  #reveal(row) {
    const at = this.#matches().indexOf(row);
    if (at >= this.#shown) {
      this.#shown = (Math.floor(at / PAGE) + 1) * PAGE;
      this.#refresh();
    }
  }

  #markCurrent(row) {
    for (const [each, tr] of this.#rows) tr.part.toggle('current', each === row);
  }

  /** The player above the list shows `row`. */
  #select(row, event = null) {
    this.#current = row;
    this.#reveal(row);
    this.#markCurrent(row);
    this.#play(row, { autoplay: false });
    if (event?.isTrusted) {
      this.#act('open', { row: row.id, from: 'Row' });
      setLinkParam(this.getAttribute('link-param'), row.id);
    }
  }

  #part(name) {
    return this.shadowRoot.querySelector(`[part~="${name}"]`);
  }

  // ── The dock ─────────────────────────────────────────────────────────────

  #dock(playerRoot) {
    const step = (by, label, title) => {
      const button = element('button', { type: 'button', part: 'step', 'aria-label': label, title }, [
        by < 0 ? '↑' : '↓',
      ]);
      button.addEventListener('click', () => this.#step(by));
      return button;
    };
    const details = element('div', { part: 'details', hidden: '' }, [element('slot', { name: 'details' })]);
    const toggle = element('button', { type: 'button', part: 'details-toggle', 'aria-pressed': 'false', hidden: '' }, [
      'Details',
    ]);
    toggle.addEventListener('click', (event) => {
      details.hidden = !details.hidden;
      toggle.setAttribute('aria-pressed', String(!details.hidden));
      if (event.isTrusted) this.#act('details', { open: !details.hidden });
    });
    details.querySelector('slot').addEventListener('slotchange', (event) => {
      toggle.hidden = !event.target.assignedNodes().length;
    });
    const full = element('a', { part: 'full', target: '_blank', rel: 'noopener' }, ['Full page']);
    full.addEventListener('click', (event) => event.isTrusted && this.#act('full', { row: this.#current?.id }));
    const close = element(
      'button',
      { type: 'button', part: 'close', 'aria-label': 'Close the replay', title: 'Close (Esc)' },
      ['×'],
    );
    close.addEventListener('click', (event) => {
      if (event.isTrusted) this.#act('close', { row: this.#current?.id, via: 'Button' });
      this.#close();
    });
    const resize = element('div', {
      part: 'resize',
      role: 'separator',
      'aria-orientation': 'horizontal',
      'aria-label': 'Resize the replay',
      'aria-valuemin': '0',
      'aria-valuemax': '100',
      tabindex: '0',
      title: 'Drag to resize',
    });
    this.#dragResize(resize);
    const head = element('header', { part: 'dock-head' }, [
      element('div', { part: 'steps' }, [
        step(-1, 'Previous episode', 'Previous episode (Up arrow)'),
        element('span', { part: 'position' }),
        step(1, 'Next episode', 'Next episode (Down arrow)'),
      ]),
      toggle,
      full,
      close,
      element('p', { part: 'title' }),
      element('div', { part: 'facts' }),
    ]);
    return element('aside', { part: 'dock', 'aria-label': 'Replay', hidden: '' }, [
      resize,
      head,
      element('div', { part: 'dock-body' }, [playerRoot, details]),
    ]);
  }

  /** `from` names what opened it: a row, a step from the open row, or a link that names the row. */
  #open(row, from, event = null) {
    if (from !== 'Row' || event?.isTrusted) this.#act('open', { row: row.id, from });
    if (openDock && openDock !== this) openDock.#close();
    openDock = this;
    this.#current = row;
    this.#reveal(row);
    this.#markCurrent(row);
    const [first, ...rest] = this.#list.columns;
    const titled = first?.type === 'text';
    this.#part('title').textContent = titled ? cellText(row.values[first.key]) : row.label;
    this.#part('facts').replaceChildren(...this.#facts(row, titled ? rest : this.#list.columns));
    const full = this.#part('full');
    full.hidden = !row.href;
    if (row.href) full.href = row.href;
    this.#play(row, { autoplay: true });
    this.#showPosition(this.#matches());
    const dock = this.#part('dock');
    if (dock.hidden) {
      dock.hidden = false;
      this.setAttribute('open', '');
      document.addEventListener('keydown', this.#onKey);
      this.#resize(this.#strip ?? savedStrip() ?? STRIP_ROWS * this.#rows.get(row).getBoundingClientRect().height);
    }
    this.#focus(row);
    if (!this.#narrow.matches) this.#player.focusViewer();
    setLinkParam(this.getAttribute('link-param'), row.id);
    if (from === 'Link') this.#holdOpenRow();
  }

  /** Each column's value of the row in the dock's head, a number with its column's name. */
  #facts(row, columns) {
    return columns
      .filter((column) => column.key !== INDEX_KEY)
      .map((column) => {
        const content = cellContent(column, row.values[column.key]);
        if (column.type !== 'number') return element('span', { part: `fact ${column.type}` }, content);
        return element('span', { part: 'fact number' }, [
          ...content,
          element('span', { part: 'fact-label' }, [column.label]),
        ]);
      });
  }

  #close() {
    const dock = this.#part('dock');
    if (!dock || dock.hidden) return;
    dock.hidden = true;
    this.removeAttribute('open');
    this.#player.stop();
    this.#markCurrent(null);
    this.#current = null;
    if (openDock === this) openDock = null;
    document.removeEventListener('keydown', this.#onKey);
    setLinkParam(this.getAttribute('link-param'), null);
  }

  #step(by) {
    const matches = this.#matches();
    const next = matches[matches.indexOf(this.#current) + by];
    if (next) this.#open(next, 'Step');
  }

  #showPosition(matches) {
    const at = matches.indexOf(this.#current);
    this.#part('position').textContent = `${at + 1} of ${matches.length}`;
    const [previous, next] = this.shadowRoot.querySelectorAll('[part~="step"]');
    previous.disabled = at === 0;
    next.disabled = at === matches.length - 1;
  }

  /** Escape closes the dock, and the up and down arrows step, from the page or from the viewer. */
  #onKey = (event) => {
    if (openDock !== this) return;
    const target = event.composedPath()[0];
    if (target.closest?.('input, select, textarea, [part~="resize"]')) return;
    const by = { ArrowDown: 1, ArrowUp: -1 }[event.key];
    if (event.key !== 'Escape' && !by) return;
    event.preventDefault();
    event.stopPropagation();
    if (by) {
      this.#step(by);
      return;
    }
    this.#act('close', { row: this.#current?.id, via: 'Escape' });
    this.#close();
  };

  #top() {
    const selector = this.getAttribute('dock-below');
    return (selector && document.querySelector(selector)?.getBoundingClientRect().bottom) || 0;
  }

  /** The list's strip above the dock, in px: the dock takes the rest of the window under the host's bar. */
  #resize(strip) {
    const free = window.innerHeight - this.#top();
    this.#strip = Math.round(Math.min(Math.max(strip, 0), free - DOCK_MIN_PX));
    this.style.setProperty('--dock-height', `${free - this.#strip}px`);
    this.#part('resize').setAttribute('aria-valuenow', String(Math.round(100 * (1 - this.#strip / free))));
  }

  /** Scrolls the open row to the middle of the strip above the dock; a narrow screen's dock covers the page. */
  #focus(row, behavior = 'smooth') {
    const tr = this.#rows.get(row);
    if (this.#narrow.matches) {
      tr.scrollIntoView({ block: 'nearest' });
      return;
    }
    const top = this.#top();
    const bottom = this.#part('dock').getBoundingClientRect().top;
    const box = tr.getBoundingClientRect();
    window.scrollTo({ top: window.scrollY + box.top + box.height / 2 - (top + bottom) / 2, behavior });
  }

  /** Content above the list can grow after a link opens a row: keep the row in view until the visitor acts. */
  #holdOpenRow() {
    if (!window.ResizeObserver) return;
    const observer = new ResizeObserver(() => this.#current && this.#focus(this.#current, 'instant'));
    const acts = ['wheel', 'touchstart', 'pointerdown', 'keydown'];
    const stop = () => {
      observer.disconnect();
      acts.forEach((type) => window.removeEventListener(type, stop, true));
    };
    observer.observe(document.body);
    acts.forEach((type) => window.addEventListener(type, stop, { capture: true, passive: true }));
    setTimeout(stop, HOLD_OPEN_MS);
  }

  /** A drag of the dock's top edge, or an up or down arrow on it, sets the strip; the open row stays in its middle. */
  #dragResize(handle) {
    handle.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      this.setAttribute('resizing', '');
      const before = this.#strip;
      const move = (moved) => {
        this.#resize(moved.clientY - this.#top());
        if (this.#current) this.#focus(this.#current, 'instant');
      };
      handle.addEventListener('pointermove', move);
      handle.addEventListener(
        'lostpointercapture',
        () => {
          handle.removeEventListener('pointermove', move);
          this.removeAttribute('resizing');
          saveStrip(this.#strip);
          if (this.#strip !== before) this.#act('resize', { via: 'Drag' });
        },
        { once: true },
      );
    });
    handle.addEventListener('keydown', (event) => {
      const by = { ArrowUp: -1, ArrowDown: 1 }[event.key];
      if (!by || !this.#current) return;
      event.preventDefault();
      const before = this.#strip;
      this.#resize(this.#strip + by * this.#rows.get(this.#current).getBoundingClientRect().height);
      this.#focus(this.#current, 'instant');
      saveStrip(this.#strip);
      if (this.#strip !== before) this.#act('resize', { via: 'Key' });
    });
  }

  // ── The compact layout ───────────────────────────────────────────────────

  #renderCompact() {
    const { columns, rows } = this.#list;
    const shown = columns.filter((column) => column.key !== INDEX_KEY && column.key !== DURATION_KEY);
    const succeeded = rows.filter((row) => cellKey(row.values[OUTCOME_KEY]).toLowerCase() === SUCCESS).length;
    const tally = columns.some((column) => column.key === OUTCOME_KEY)
      ? `${plural(rows.length, 'episode')} · ${succeeded} succeeded`
      : plural(rows.length, 'episode');
    this.#rows = new Map();
    const items = rows.map((row) => {
      const outcome = cellText(row.values[OUTCOME_KEY]);
      const run = element('span', {
        part: `run${outcomePart(row)}`,
        title: outcome ? `${row.label} · ${outcome}` : row.label,
      });
      const values = shown.map((column) =>
        element(
          'span',
          { part: `value${column.key === OUTCOME_KEY ? outcomePart(row) : ''}` },
          cellContent(column, row.values[column.key]),
        ),
      );
      const item = element('li', { part: 'episode', 'data-id': row.id }, [
        element('span', { part: 'episode-index' }, [row.id]),
        element('button', { type: 'button', part: 'episode-play', 'aria-label': `Play ${row.label}` }),
        element('span', { part: 'episode-values' }, values),
        element('span', { part: 'episode-time' }, [row.durationS === null ? '' : seconds(row.durationS)]),
      ]);
      for (const node of [run, item]) node.addEventListener('click', (event) => this.#pick(row, event));
      this.#rows.set(row, [item, run]);
      return item;
    });
    const list = element('ol', { part: 'list' }, items);
    const close = element('button', { type: 'button', part: 'screen-close', 'aria-label': 'Close the video' }, ['×']);
    close.addEventListener('click', (event) => {
      if (event.isTrusted) this.#act('close', { row: this.#current?.id, via: 'Button' });
      this.#closeScreen();
    });
    const playerRoot = element('div', { part: 'screen-player' });
    this.shadowRoot.replaceChildren(
      element('style', {}, [STYLE]),
      element('div', { part: 'head' }, [
        element('span', { part: 'tally' }, [tally]),
        element(
          'span',
          { part: 'runs', 'aria-hidden': 'true' },
          [...this.#rows.values()].map(([, run]) => run),
        ),
        element('slot', { name: 'actions' }),
      ]),
      element('div', { part: 'screen-slot' }, [
        element('div', { part: 'screen-clip' }, [element('div', { part: 'screen' }, [close, playerRoot])]),
      ]),
      list,
    );
    this.#player = this.#newPlayer(playerRoot);
    // A list that scrolls fades out at its lower edge until its last episode shows.
    const moreBelow = () => list.part.toggle('more-below', list.scrollTop + list.clientHeight < list.scrollHeight - 1);
    list.addEventListener('scroll', moreBelow, { passive: true });
    new ResizeObserver(moreBelow).observe(list);
  }

  /** A row opens the screen and plays; the open row plays or pauses. */
  #pick(row, event) {
    const playing = this.#player.lead;
    if (this.#current === row && playing) {
      if (event.isTrusted) this.#act('play', { row: row.id, playing: playing.paused });
      if (playing.paused) playing.play().catch((error) => console.warn(`${row.label} did not play: ${error.message}`));
      else playing.pause();
      return;
    }
    if (event.isTrusted) this.#act('open', { row: row.id, from: 'Row' });
    this.#current = row;
    for (const [each, nodes] of this.#rows) nodes.forEach((node) => node.part.toggle('current', each === row));
    this.#play(row, { autoplay: true });
    this.setAttribute('screen-open', '');
    this.removeAttribute('playing');
    const lead = this.#player.lead;
    if (!lead) return;
    const showState = () => this.toggleAttribute('playing', !lead.paused);
    for (const type of ['play', 'pause', 'ended']) lead.addEventListener(type, showState);
  }

  #closeScreen() {
    this.#player.stop();
    this.removeAttribute('screen-open');
    this.removeAttribute('playing');
    this.#current = null;
    for (const nodes of this.#rows.values()) nodes.forEach((node) => node.part.remove('current'));
  }
}

if (!customElements.get(TAG)) customElements.define(TAG, PositronicReplay);
