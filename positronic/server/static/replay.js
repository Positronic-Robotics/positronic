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
//   - `src`: a manifest URL. Without it, the element reads its child <script type="application/json">.
//   - `layout`: `full` (default) or `compact`.
//     - `full`: the list as a table, with the player above it or in a dock.
//     - `compact`: a tally, a strip of runs, a screen above the list, and the list.
//   - `mode`: `video` (default) or `full`.
//     - `video`: the cameras only.
//     - `full`: a row with `rrd` also plays its 3D replay. A row with no `rrd` plays as video.
//   - `dock` (full layout): the player opens at the bottom of the window when a visitor opens a row.
//   - `dock-below` (dock): a selector of the host's bar. The dock and the list share the window below it.
//   - `link-param` (full layout): the URL search parameter that names the open row. A link opens that row.
//   - `link-param` is read once, at the first connect.
//   - `camera-names`: a JSON object that maps a camera key to the name the visitor sees.
//   - A value of the wrong shape logs an error, and the element uses the default.
//   - The defaults: `full`, `video`, the top of the window, no link parameter, the camera keys.
//   - `link-param` with `layout="compact"` logs an error, and the element uses no link parameter.
//   - A `src` that is not a URL shows that the replay did not load.
//
// Manifest (`src`)
//   {title, url, about: [{key, label, value}], columns: [{key, label}], episodes: [{index, values: {key: value},
//    duration_s, videos: [{camera, url, src, width, height, duration_s}], phone?, rrd?}]}
//   - A video's `src`, `phone` and `rrd` are relative to the manifest.
//   - `phone` is a video object or a URL.
//   - The list shows an index column, the manifest's columns, and a duration column.
//
// Inline rows
//   {title?, about?, columns: [{key, label, type?, decimals?, unit?, hover?, filter?}],
//    rows: [{id, label?, values: {key: value}, duration_s?, cameras?: [{camera, label?, src, width, height,
//    duration_s}], phone?, rrd?, href?}]}
//   - URLs are relative to the document.
//   - `href` is the row's own page. The first text cell links it.
//   - A column `type` is `text` (default), `number` (with `decimals` and `unit`), `badge` or `icon`.
//   - A value is a scalar, or {text, key?, color?, ink?, icon?, title?}.
//   - `key` is the text a filter compares. `color` paints a badge. `icon` is the image of an icon cell.
//   - `filter` is `true` or {all?, options?: [[value, label], …]}.
//   - A filter with no options offers every value of its column.
//   - A null or absent value is an empty cell.
//
// Input of the wrong shape (manifest and inline rows)
//   - A list that is not an array: the element shows that the replay did not load.
//   - A bad entry of a list: the element logs an error and drops the entry.
//   - A field of the wrong shape: the element logs an error and leaves the field empty.
//   - A column of an unknown type: the element logs an error and reads it as `text`.
//   - Any other error while the element reads the data is a fault of the element. The element does not catch it.
//
// Media
//   - The element loads only the URLs a row names. It builds no URL from another one.
//   - A row with no media shows "not available".
//   - A file that answers 404 shows "gone".
//   - A file that answers 202 shows "building". The element asks for it again after its `Retry-After`.
//   - The 3D replay plays in the rerun viewer beside this file. The viewer page adds no browser history entry.
//   - A screen under 961 px wide plays the row's `phone` video in place of its cameras and its 3D replay.
//   - On such a screen, the dock covers the page.
//   - The longest camera that is ready leads the controls. The other cameras follow it.
//
// Compact layout
//   - The tally counts the episodes, and the runs whose `outcome` value is `success`.
//   - Each run in the strip takes the part `outcome-<value>`.
//
// Events
//   - A `replay-action` event reports each use of a control by a visitor. Each use sends one event.
//   - `detail.action` names the action. The fields of each action:
//     - open {row, from: Row|Button|Run|Link|Step, via: Button|Key for a step, via: Key for a row from the keyboard}
//     - close {row, via: Button|Escape}
//     - play {row, playing, via: Button|Video|Row|Run|Native}
//     - seek {row, via?}
//     - speed {row, value, via?}
//     - filter {column, value}
//     - more {via: Button|Scroll}
//     - resize {via: Drag|Key}
//     - details {open}
//     - full {row}
//     - link {row}
//   - A table row takes the keyboard focus. Enter or Space on it opens it, as a click does.
//   - A script event sends no action.
//   - A link open is reported first, so the host can show the element.
//   - The phone video keeps the browser's controls. A use of them sends `via: Native`.
//   - The keys inside the 3D viewer step and close as on the page. The viewer's own controls send nothing.
//
// Methods and slots
//   - `show({column key: value, …})` sets the filters.
//   - In `show()`, a number or a boolean counts as its text.
//   - In `show()`, null or a value the column does not offer shows all.
//   - In `show()`, a key no filter has, or a value of another type, logs an error.
//   - The `details` slot shows in the dock beside the player, behind a Details button.
//   - The `actions` slot shows in the compact head.
//
// Theme
//   - --replay-accent: the current row and the seek bar.
//   - --replay-border, --replay-muted, --replay-hover, --replay-current: the lines, the second text, a row under
//     the pointer, the current row.
//   - --replay-surface, --replay-surface-ink: the dock and the compact screen, and their text.
//   - --replay-success, --replay-failure: the runs of the compact strip.
//   - --replay-motion: the open time of the compact screen.
//   - --replay-sticky-top: where the copy of the column names sticks.
//   - --list-max, --video-max: the most height the compact list and its videos take.
//   - Each node has a `part` to style.
import {
  cellKey,
  cellText,
  ColumnType,
  DURATION_KEY,
  INDEX_KEY,
  InputError,
  parseJson,
  readInline,
  readManifest,
  wrongInput,
} from './replay/data.js';
import { cellContent, element } from './replay/cells.js';
import { EpisodePlayer, pause, start } from './replay/media.js';
import { STYLE } from './replay/style.js';

const TAG = 'positronic-replay';
const PAGE = 50;
const NEXT_PAGE_AHEAD = '0px 0px 400px 0px';
// The width under which the dock covers the page, as the styles say.
const NARROW = '(max-width: 960px)';
const STRIP_KEY = 'positronic-replay.strip';
const STRIP_ROWS = 3;
const DOCK_MIN_PX = 160;
const HOLD_OPEN_MS = 10000;
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

// A listener or an observer that outlives the shadow DOM it serves, held with the step that frees it.
function listen(held, target, type, handler, options) {
  target.addEventListener(type, handler, options);
  held.push(() => target.removeEventListener(type, handler, options));
}

function observe(held, observer) {
  held.push(() => observer.disconnect());
  return observer;
}

const Layout = Object.freeze({ FULL: 'full', COMPACT: 'compact' });
const Mode = Object.freeze({ VIDEO: 'video', FULL: 'full' });

// Each attribute, read once where it changes: the value the element uses, and the default it falls back to.
const ATTRIBUTES = {
  layout: (value) => oneOf('layout', value, Object.values(Layout)),
  mode: (value) => oneOf('mode', value, Object.values(Mode)),
  dock: (value) => value !== null,
  'dock-below': (value) => {
    if (value === null) return null;
    try {
      document.querySelector(value);
      return value;
    } catch (error) {
      if (!(error instanceof DOMException)) throw error;
      return wrongInput('dock-below', value, null, 'the top of the window');
    }
  },
  'camera-names': (value) => {
    if (value === null) return {};
    let names;
    try {
      names = JSON.parse(value);
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      return wrongInput('camera-names', value, {}, 'the camera keys');
    }
    const isNames = names !== null && typeof names === 'object' && !Array.isArray(names);
    if (isNames && Object.values(names).every((name) => typeof name === 'string')) return names;
    return wrongInput('camera-names', value, {}, 'the camera keys');
  },
};

/** The attribute `value` where it is one of `allowed`; the first of them where it is absent or another. */
function oneOf(input, value, allowed) {
  if (value === null || allowed.includes(value)) return value ?? allowed[0];
  return wrongInput(input, value, allowed[0], allowed[0]);
}

function setLinkParam(name, value) {
  if (!name) return;
  const url = new URL(window.location);
  if (value === null) url.searchParams.delete(name);
  else url.searchParams.set(name, value);
  window.history.replaceState(window.history.state, '', url);
}

export class PositronicReplay extends HTMLElement {
  static observedAttributes = ['src', ...Object.keys(ATTRIBUTES)];

  #list = null;
  #request = 0;
  #loading = null;
  // The source of the build on screen: a URL, `inline`, or null for none. A release clears it, and each build
  // sets it after its release.
  #loaded = null;
  #listSource = null;
  // What the current build holds outside its shadow DOM, and what the open dock holds; `#release` and `#close`
  // free them.
  #held = [];
  #dockHeld = [];
  // An upgrade reports each attribute before it connects the element, and the first connect loads it.
  #connected = false;
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
  #inputs = Object.fromEntries(Object.entries(ATTRIBUTES).map(([name, read]) => [name, read(null)]));
  // Read once, at the first connect.
  #linkParam = undefined;
  #unmatchedDockBelow = null;

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this.#connected = true;
    if (this.#linkParam === undefined) this.#linkParam = this.#readLinkParam();
    if (this.#list) this.#render();
    else this.#load();
  }

  disconnectedCallback() {
    this.#connected = false;
    this.#waitForScript?.disconnect();
    this.#loading?.abort();
    this.#loading = null;
    this.#release();
  }

  /** The `link-param` attribute, for the full layout only. */
  #readLinkParam() {
    const value = this.getAttribute('link-param');
    if (value === null) return null;
    if (!value.trim()) return wrongInput('link-param', value, null, 'no link');
    if (this.#layout === Layout.COMPACT)
      return wrongInput('link-param', value, null, 'no link: the compact layout has none');
    return value;
  }

  attributeChangedCallback(name, previous, value) {
    if (name in ATTRIBUTES) this.#inputs[name] = ATTRIBUTES[name](value);
    if (!this.#connected) return;
    if (name === 'src' || !this.#list) this.#load();
    else this.#render();
  }

  get #layout() {
    return this.#inputs.layout;
  }

  get #fullMode() {
    return this.#inputs.mode === Mode.FULL;
  }

  get #docked() {
    return this.#layout === Layout.FULL && this.#inputs.dock;
  }

  /** Sets the filters to `filters`, by column key: a string, a number or a boolean as its text, and null or an empty
   * string for all. A value a filter does not offer shows all. */
  show(filters) {
    if (filters === null || typeof filters !== 'object' || Array.isArray(filters)) {
      wrongInput('show()', filters, undefined, 'no change');
      return;
    }
    const keys = new Set(this.#selects.map((select) => select.dataset.key));
    for (const key of Object.keys(filters)) {
      if (!keys.has(key))
        wrongInput(`show() key ${key}`, filters[key], undefined, 'no change to it, as no filter has it');
    }
    for (const select of this.#selects) {
      const given = filters[select.dataset.key];
      if (given === undefined) continue;
      if (given !== null && !['string', 'number', 'boolean'].includes(typeof given)) {
        wrongInput(`show() ${select.dataset.key}`, given, undefined, 'no change to it');
        continue;
      }
      const value = given === null ? '' : String(given);
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
    this.#loading?.abort();
    this.#loading = null;
    this.#list = null;
    if (!src) {
      const script = this.querySelector(':scope > script[type="application/json"]');
      if (!script) {
        // The parser can connect the element before it adds the children, and a host can remove `src` first.
        this.#status('Waiting for the replay data…');
        this.#waitForScript ??= new MutationObserver(() => this.#load());
        this.#waitForScript.observe(this, { childList: true });
        return;
      }
      this.#waitForScript?.disconnect();
      this.#take(() => readInline(parseJson('the inline rows', script.textContent), document.baseURI), key);
      return;
    }
    this.#waitForScript?.disconnect();
    let url;
    try {
      url = new URL(src, document.baseURI);
    } catch (error) {
      if (!(error instanceof TypeError)) throw error;
      wrongInput('src', src, null, 'no replay');
      this.#status('The replay did not load: its address is not a URL.', key);
      return;
    }
    const loading = new AbortController();
    this.#loading = loading;
    this.#status('Loading the replay…', key);
    this.#fetchManifest(url, loading.signal)
      .finally(() => {
        if (this.#loading === loading) this.#loading = null;
      })
      .then(
        (manifest) => {
          if (request === this.#request) this.#take(() => readManifest(manifest, url.href), key);
        },
        (error) => {
          // A new `src`, or a removal of the element, aborts the request of the old one.
          if (error.name === 'AbortError' || request !== this.#request) return;
          if (!(error instanceof InputError)) throw error;
          console.error(`The replay at ${url} did not load:`, error);
          this.#status(`The replay did not load: ${error.message}`, key);
        },
      );
  }

  /** The manifest at `url`. A failed request, an error status or text that is not JSON is an `InputError`. */
  async #fetchManifest(url, signal) {
    let response;
    try {
      response = await fetch(url, { signal });
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      throw new InputError(`the request failed: ${error.message}`);
    }
    if (!response.ok) throw new InputError(`${response.status} ${response.statusText}`);
    return parseJson('the manifest', await response.text());
  }

  /** Reads the list of `source` and builds it. Only an `InputError` shows that the replay did not load. */
  #take(read, source) {
    try {
      this.#list = read();
    } catch (error) {
      if (!(error instanceof InputError)) throw error;
      console.error('The replay data does not read as a list of episodes:', error);
      this.#status(`The replay did not load: ${error.message}`, source);
      return;
    }
    this.#listSource = source;
    this.#render();
  }

  /** Shows `message` in place of a build; `source` is the source it stands for, and null for none. */
  #status(message, source = null) {
    this.#release();
    this.#loaded = source;
    this.shadowRoot.replaceChildren(element('style', {}, [STYLE]), element('p', { part: 'status' }, [message]));
  }

  #act(action, detail = {}) {
    this.dispatchEvent(
      new CustomEvent('replay-action', { bubbles: true, composed: true, detail: { action, ...detail } }),
    );
  }

  // A camera's key is its name in the recording, so a camera the names leave out shows its key.
  #cameraName = (key) => this.#inputs['camera-names'][key] ?? key;

  #newPlayer(root) {
    const player = new EpisodePlayer(root, {
      full: this.#fullMode,
      narrow: this.#narrow,
      act: (action, detail) => this.#act(action, detail),
      frameKeys: this.#onKey,
      cameraName: this.#cameraName,
    });
    // The other width plays the other video of the row, in the whole playback state of the player.
    listen(this.#held, this.#narrow, 'change', () => {
      if (this.#current) this.#play(this.#current, this.#player.playback());
    });
    return player;
  }

  /** Frees what a build holds: the open dock or screen, the player with its waits and requests, the listeners on
   * the window, the document and the media query, and the observers. */
  #release() {
    this.#close();
    this.removeAttribute('screen-open');
    this.removeAttribute('playing');
    this.#player?.stop();
    this.#player = null;
    for (const free of this.#held.splice(0)) free();
    this.#end = null;
    this.#current = null;
    this.#loaded = null;
  }

  #render() {
    this.#release();
    this.#loaded = this.#listSource;
    this.#shown = PAGE;
    this.#selects = [];
    this.#count = null;
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
    if (this.#layout === Layout.COMPACT) this.#renderCompact();
    else this.#renderFull(header);
  }

  #header() {
    const { title, about } = this.#list;
    if (!title || this.#layout === Layout.COMPACT) return [];
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
    if (this.#docked) {
      listen(this.#held, window, 'resize', () => {
        if (this.#current) this.#resize(this.#strip);
      });
    }
    if (window.IntersectionObserver) {
      this.#end = observe(
        this.#held,
        new IntersectionObserver(
          (entries) => entries.some((entry) => entry.isIntersecting) && this.#showMore('Scroll'),
          { rootMargin: NEXT_PAGE_AHEAD },
        ),
      );
    }
    this.#refresh();
    const linked = this.#linkedRow();
    if (this.#docked) {
      if (linked) this.#open(linked, 'Link');
    } else if (linked) {
      this.#act('open', { row: linked.id, from: 'Link' });
      this.#select(linked);
    } else {
      this.#select(this.#list.rows[0]);
    }
  }

  /** The row the page's address names; null where it names none. An address that names a row the list does not
   * hold loses the parameter. */
  #linkedRow() {
    const name = this.#linkParam;
    const id = name ? new URLSearchParams(window.location.search).get(name) : null;
    if (id === null) return null;
    const row = this.#list.rows.find((each) => each.id === id);
    if (row) return row;
    // The address comes from the visitor, as an old link or an episode the list no longer holds: no host fault.
    console.warn(`The link parameter ${name}=${id} names no row of this list; the element opens none.`);
    setLinkParam(name, null);
    return null;
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
    const linkColumn = columns.find((column) => column.type === ColumnType.TEXT);
    this.#rows = new Map();
    const body = rows.map((row) => {
      const cells = columns.map((column) => {
        const content = cellContent(column, row.values[column.key], column === linkColumn ? row.href : null);
        return element('td', { part: `cell ${column.type}`, 'data-label': column.label }, content);
      });
      const tr = element('tr', { part: 'row', 'data-id': row.id, tabindex: '0', 'aria-label': row.label }, cells);
      tr.addEventListener('click', (event) => this.#rowClicked(event, row));
      tr.addEventListener('keydown', (event) => this.#rowKey(event, row));
      this.#rows.set(row, tr);
      return tr;
    });
    return element('table', { part: 'table' }, [element('thead', {}, [head]), element('tbody', {}, body)]);
  }

  /** A click on the row's link follows it; a click anywhere else in the row opens the row. */
  #rowClicked(event, row) {
    if (event.composedPath().some((node) => node instanceof HTMLAnchorElement)) {
      if (event.isTrusted) this.#act('link', { row: row.id });
      return;
    }
    if (this.#docked) this.#open(row, 'Row', event);
    else this.#select(row, event);
  }

  /** Enter or Space on the row itself opens it; a link in the row keeps its own keys. */
  #rowKey(event, row) {
    if (event.composedPath()[0] !== this.#rows.get(row) || (event.key !== 'Enter' && event.key !== ' ')) return;
    event.preventDefault();
    if (this.#docked) this.#open(row, 'Row', event, 'Key');
    else this.#select(row, event, 'Key');
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
    const observer = observe(this.#held, new ResizeObserver(fit));
    columns.forEach((column) => observer.observe(column));
    scroller.addEventListener('scroll', () => (head.scrollLeft = scroller.scrollLeft), { passive: true });
  }

  #matches() {
    const wanted = this.#selects.filter((select) => select.value).map((select) => [select.dataset.key, select.value]);
    return this.#list.rows.filter((row) => wanted.every(([key, value]) => cellKey(row.values[key]) === value));
  }

  #refresh() {
    if (this.#layout === Layout.COMPACT) return;
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
  #select(row, event = null, via = null) {
    this.#current = row;
    this.#reveal(row);
    this.#markCurrent(row);
    this.#play(row, { playing: false });
    if (event?.isTrusted) {
      this.#act('open', { row: row.id, from: 'Row', ...(via && { via }) });
      setLinkParam(this.#linkParam, row.id);
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
      button.addEventListener('click', (event) => this.#step(by, 'Button', event));
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

  /** The strip a visitor set before, kept in the browser; null where none is kept or the browser refuses. */
  static #savedStrip() {
    try {
      const strip = Number(window.localStorage.getItem(STRIP_KEY));
      return strip > 0 ? strip : null;
    } catch (error) {
      if (!(error instanceof DOMException)) throw error;
      // The dock then opens at its default strip, which only changes its first height.
      console.error(`The browser refuses to read the saved strip from its storage: ${error.message}`);
      return null;
    }
  }

  /** `from` names what opened it: a row, a step from the open row, or a link that names the row. A step names
   * its `via`, a button or a key, and a row opened from the keyboard names `via: Key`. */
  #open(row, from, event = null, via = null) {
    // A link names the row in the page's own address, so its open counts as the visitor's.
    if (from === 'Link' || event?.isTrusted) this.#act('open', { row: row.id, from, ...(via && { via }) });
    if (openDock && openDock !== this) openDock.#close();
    openDock = this;
    this.#current = row;
    this.#reveal(row);
    this.#markCurrent(row);
    const [first, ...rest] = this.#list.columns;
    const titled = first?.type === ColumnType.TEXT;
    this.#part('title').textContent = titled ? cellText(row.values[first.key]) : row.label;
    this.#part('facts').replaceChildren(...this.#facts(row, titled ? rest : this.#list.columns));
    const full = this.#part('full');
    full.hidden = !row.href;
    if (row.href) full.href = row.href;
    this.#play(row, { playing: true });
    this.#showPosition(this.#matches());
    const dock = this.#part('dock');
    if (dock.hidden) {
      dock.hidden = false;
      this.setAttribute('open', '');
      listen(this.#dockHeld, document, 'keydown', this.#onKey);
      this.#resize(
        this.#strip ??
          PositronicReplay.#savedStrip() ??
          STRIP_ROWS * this.#rows.get(row).getBoundingClientRect().height,
      );
    }
    this.#focus(row);
    if (!this.#narrow.matches) this.#player.focusViewer();
    setLinkParam(this.#linkParam, row.id);
    if (from === 'Link') this.#holdOpenRow();
  }

  /** Each column's value of the row in the dock's head, a number with its column's name. */
  #facts(row, columns) {
    return columns
      .filter((column) => column.key !== INDEX_KEY)
      .map((column) => {
        const content = cellContent(column, row.values[column.key]);
        if (column.type !== ColumnType.NUMBER) return element('span', { part: `fact ${column.type}` }, content);
        return element('span', { part: 'fact number' }, [
          ...content,
          element('span', { part: 'fact-label' }, [column.label]),
        ]);
      });
  }

  #close() {
    for (const free of this.#dockHeld.splice(0)) free();
    if (!this.hasAttribute('open')) return;
    const dock = this.#part('dock');
    if (dock) dock.hidden = true;
    this.removeAttribute('open');
    this.#player?.stop();
    this.#markCurrent(null);
    this.#current = null;
    if (openDock === this) openDock = null;
    setLinkParam(this.#linkParam, null);
  }

  #step(by, via, event) {
    const matches = this.#matches();
    const next = matches[matches.indexOf(this.#current) + by];
    if (next) this.#open(next, 'Step', event, via);
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
      this.#step(by, 'Key', event);
      return;
    }
    if (event.isTrusted) this.#act('close', { row: this.#current?.id, via: 'Escape' });
    this.#close();
  };

  #top() {
    const selector = this.#inputs['dock-below'];
    if (!selector) return 0;
    const bar = document.querySelector(selector);
    if (bar) return bar.getBoundingClientRect().bottom;
    // The host's bar can come later, so the check runs where the dock needs it, and logs each value once.
    if (this.#unmatchedDockBelow !== selector) {
      this.#unmatchedDockBelow = selector;
      wrongInput('dock-below', selector, null, 'the top of the window, as no element matches it');
    }
    return 0;
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
    const held = [];
    const stop = () => {
      for (const free of held.splice(0)) free();
    };
    const observer = observe(held, new ResizeObserver(() => this.#current && this.#focus(this.#current, 'instant')));
    observer.observe(document.body);
    for (const type of ['wheel', 'touchstart', 'pointerdown', 'keydown']) {
      listen(held, window, type, stop, { capture: true, passive: true });
    }
    const timer = setTimeout(stop, HOLD_OPEN_MS);
    held.push(() => clearTimeout(timer));
    this.#dockHeld.push(stop);
  }

  static #saveStrip(strip) {
    try {
      window.localStorage.setItem(STRIP_KEY, String(strip));
    } catch (error) {
      if (!(error instanceof DOMException)) throw error;
      // The strip then holds for this page only, which changes nothing but the height of a later visit.
      console.error(`The browser refuses to save the strip in its storage: ${error.message}`);
    }
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
          PositronicReplay.#saveStrip(this.#strip);
          if (event.isTrusted && this.#strip !== before) this.#act('resize', { via: 'Drag' });
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
      PositronicReplay.#saveStrip(this.#strip);
      if (event.isTrusted && this.#strip !== before) this.#act('resize', { via: 'Key' });
    });
  }

  // ── The compact layout ───────────────────────────────────────────────────

  #renderCompact() {
    // The tally counts the runs whose `outcome` is `success`, and each run takes a part for its outcome.
    const OUTCOME_KEY = 'outcome';
    const SUCCESS = 'success';
    const outcomePart = (row) => {
      const outcome = cellKey(row.values[OUTCOME_KEY])
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '');
      return outcome ? ` outcome-${outcome}` : '';
    };
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
      run.addEventListener('click', (event) => this.#pick(row, event, 'Run'));
      item.addEventListener('click', (event) => {
        const button = event.composedPath().some((node) => node instanceof HTMLButtonElement);
        this.#pick(row, event, button ? 'Button' : 'Row');
      });
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
    // A media event does not bubble, so the screen catches the lead's play and pause in the capture phase.
    for (const type of ['play', 'pause', 'ended', 'emptied']) {
      playerRoot.addEventListener(type, () => this.#showPlaying(), true);
    }
    // A list that scrolls fades out at its lower edge until its last episode shows.
    const moreBelow = () => list.part.toggle('more-below', list.scrollTop + list.clientHeight < list.scrollHeight - 1);
    list.addEventListener('scroll', moreBelow, { passive: true });
    observe(this.#held, new ResizeObserver(moreBelow)).observe(list);
  }

  /** A row opens the screen and plays; the open row plays or pauses. `via` names what the visitor used: the row,
   * its play button, or its run in the strip. */
  #pick(row, event, via) {
    const playing = this.#player.lead;
    if (this.#current === row && playing) {
      if (event.isTrusted) this.#act('play', { row: row.id, playing: playing.paused, via });
      if (playing.paused) start(playing);
      else pause(playing);
      return;
    }
    if (event.isTrusted) this.#act('open', { row: row.id, from: via });
    this.#current = row;
    for (const [each, nodes] of this.#rows) nodes.forEach((node) => node.part.toggle('current', each === row));
    this.#play(row, { playing: true });
    this.setAttribute('screen-open', '');
    this.#showPlaying();
  }

  /** The host's `playing` attribute and the name of each play button, by whether the open row plays. */
  #showPlaying() {
    const playing = this.#current !== null && this.#player.lead?.paused === false;
    this.toggleAttribute('playing', playing);
    for (const [row, [item]] of this.#rows) {
      const verb = playing && row === this.#current ? 'Pause' : 'Play';
      item.querySelector('[part~="episode-play"]').setAttribute('aria-label', `${verb} ${row.label}`);
    }
  }

  #closeScreen() {
    this.#player.stop();
    this.removeAttribute('screen-open');
    this.#current = null;
    for (const nodes of this.#rows.values()) nodes.forEach((node) => node.part.remove('current'));
    this.#showPlaying();
  }
}

if (!customElements.get(TAG)) customElements.define(TAG, PositronicReplay);
