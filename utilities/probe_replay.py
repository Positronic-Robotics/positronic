#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright", "av", "numpy", "rerun-sdk==0.30.0"]
# ///
"""Check `<positronic-replay>` in a browser, over fixture manifests and inline rows that a local server serves.

It checks each layout and mode: the list, filters and paging, the player and its media states, the 3D replay with
the browser history it leaves, and the `replay-action` events. In `video` mode, and on a row with no `rrd`, the page
must ask for no recording and show no viewer.

Usage
    uv run --script utilities/probe_replay.py                 # exit 1 on a FAIL or a console error
    uv run --script utilities/probe_replay.py --headed        # watch it
    uv run --with playwright playwright install chromium        # once, for the browser
"""

import argparse
import json
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import av
import numpy as np
import rerun as rr
from playwright.sync_api import Browser, ConsoleMessage, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

STATIC = Path(__file__).resolve().parents[1] / 'positronic' / 'server' / 'static'
FIXTURES = Path(__file__).resolve().parent / 'replay_fixtures'
SHARE = '/v/Zm9vYmFyMTIzNDU2'
"""The share prefix the fixture manifest's video paths sit under."""
BUILDING_ANSWERS = 2
"""The 202 answers a building file gets before the server serves it."""
RETRY_AFTER_S = 1
PAGE = 50
ROWS = 120
SETTLE_S = 6.0
"""How long a replay gets to load: the viewer changes the history only after it loads."""
WAIT_MS = 20_000
DOCK_STRIP_PX = 450
SLOW_S = 2.0
"""How long the slow manifest takes, so a newer `src` answers first."""
STRIP_KEY = 'positronic-replay.strip'
"""Where the element keeps the strip above its dock: `STRIP_KEY` in replay.js."""


@dataclass
class Media:
    video: bytes
    rrd: bytes


@dataclass
class Requests:
    paths: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, path: str) -> int:
        with self.lock:
            self.paths.append(path)
            return self.paths.count(path)

    def since(self, mark: int) -> list[str]:
        with self.lock:
            return self.paths[mark:]

    def mark(self) -> int:
        with self.lock:
            return len(self.paths)


def make_video(path: Path) -> bytes:
    """A 2 s VP8 video: Chromium plays WebM without the codecs a branded browser adds."""
    with av.open(str(path), 'w', format='webm') as container:
        stream = container.add_stream('libvpx', rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(20):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 12, dtype=np.uint8), format='rgb24')
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path.read_bytes()


def make_recording(path: Path) -> bytes:
    if rr.__version__ != rerun_release():
        raise SystemExit(f'rerun-sdk {rr.__version__} writes this recording; the viewer here is {rerun_release()}')
    stream = rr.RecordingStream('probe_replay')
    stream.save(str(path))
    for i in range(20):
        stream.set_time('t', duration=i * 0.1)
        stream.log('joint', rr.Scalars([float(np.sin(i / 3))]))
    stream.flush()
    stream.disconnect()
    return path.read_bytes()


def rerun_release() -> str:
    (release,) = [d.name for d in (STATIC / 'rerun').iterdir() if d.is_dir()]
    return release


# ── The fixture pages ────────────────────────────────────────────────────────


def video_share() -> dict:
    return json.loads((FIXTURES / 'video_share.json').read_text())


def long_share() -> dict:
    """The fixture's episodes repeated to past one page, with their own indices."""
    manifest = video_share()
    episodes = manifest['episodes']
    manifest['episodes'] = [{**episodes[i % len(episodes)], 'index': i} for i in range(PAGE + len(episodes) * 7)]
    return manifest


def inline_rows() -> dict:
    """Rows of every cell type and media state. Row 5 builds its replay, 6 lost it, 7 has none, 8 has no media,
    9 lost its videos, 10 builds them, 11 lost its longest video only, 12 builds its replay and its videos."""
    rows = []
    for n in range(1, ROWS + 1):
        cameras = [
            {'camera': camera, 'src': f'/media/ok/{n}/{camera}.webm', 'width': 64, 'height': 48, 'duration_s': 2.0}
            for camera in ('front', 'wrist')
        ]
        row = {
            'id': str(n),
            'values': {
                'task': f'Task {n % 3}',
                'model': {'text': f'Model {"ab"[n % 2]}', 'key': 'ab'[n % 2], 'icon': '/static/icons/human.svg'},
                'result': {
                    'text': 'Success' if n % 4 else 'Partial (1/2)',
                    'key': 'success' if n % 4 else 'partial',
                    'color': '#199e70' if n % 4 else '#c98500',
                },
                'reliability': 50 + n % 50,
                'pace': n / 10,
            },
            'duration_s': 2.0,
            'cameras': cameras,
            'phone': f'/media/ok/{n}/phone.webm',
            'rrd': f'/rrd/ok/{n}.rrd',
            'href': f'/episode/{n}',
        }
        if n == 5:
            row['rrd'] = '/rrd/building/5.rrd'
        if n == 6:
            row['rrd'] = '/rrd/gone/6.rrd'
        if n in (7, 9, 10):
            del row['rrd']
        if n == 8:
            for key in ('rrd', 'cameras', 'phone'):
                del row[key]
        if n in (9, 10):
            state = 'gone' if n == 9 else 'building'
            row['cameras'] = [{**camera, 'src': camera['src'].replace('/ok/', f'/{state}/')} for camera in cameras]
            del row['phone']
        if n == 11:
            longest = {**cameras[0], 'src': cameras[0]['src'].replace('/ok/', '/gone/'), 'duration_s': 3.0}
            row['cameras'] = [longest, cameras[1]]
            for key in ('rrd', 'phone'):
                del row[key]
        if n == 12:
            row['rrd'] = '/rrd/building/12.rrd'
            row['cameras'] = [{**camera, 'src': camera['src'].replace('/ok/', '/building/')} for camera in cameras]
            del row['phone']
        rows.append(row)
    columns = [
        {'key': 'task', 'label': 'Task', 'filter': True},
        {'key': 'model', 'label': 'Model', 'type': 'icon', 'filter': {'all': 'All models'}},
        {
            'key': 'result',
            'label': 'Result',
            'type': 'badge',
            'filter': {'options': [['success', 'Success'], ['partial', 'Partial']]},
        },
        {'key': 'reliability', 'label': 'Reliability', 'type': 'number', 'decimals': 0},
        {'key': 'pace', 'label': 'Pace', 'type': 'number', 'decimals': 1, 'unit': ' s'},
    ]
    return {'columns': columns, 'rows': rows}


SCRIPT = '<script type="module" src="/static/replay.js"></script>'
# The page's own record of the element's actions, and short names for what the checks read in its shadow root.
HELPERS = """<script>
  const RETRY_MS = %RETRY_MS%;
  window.actions = [];
  document.addEventListener('replay-action', (event) => window.actions.push(event.detail));
  window.host = () => document.querySelector('positronic-replay');
  window.one = (part) => host().shadowRoot.querySelector(`[part~="${part}"]`);
  window.all = (part) => [...host().shadowRoot.querySelectorAll(`[part~="${part}"]`)];
  window.shownRows = () => all('row').filter((row) => !row.hidden).length;
  window.row = (id) => host().shadowRoot.querySelector(`tr[data-id="${id}"]`);
  window.noteText = () => (one('note') && !one('note').hidden ? one('note').textContent : '');
  window.frame = () => host().shadowRoot.querySelector('iframe');
  window.frameUrl = () => (frame() && !frame().hidden ? frame().contentWindow.location.href : '');
  window.loads = (rrd) => frameUrl().includes(encodeURIComponent(rrd));
  // The media query listeners the page holds, and the waits of the length the fixture server's Retry-After asks.
  window.queryListeners = 0;
  const addToQuery = MediaQueryList.prototype.addEventListener;
  const removeFromQuery = MediaQueryList.prototype.removeEventListener;
  MediaQueryList.prototype.addEventListener = function (type, ...rest) {
    if (type === 'change') window.queryListeners += 1;
    return addToQuery.call(this, type, ...rest);
  };
  MediaQueryList.prototype.removeEventListener = function (type, ...rest) {
    if (type === 'change') window.queryListeners -= 1;
    return removeFromQuery.call(this, type, ...rest);
  };
  window.retryWaits = new Set();
  const setTimer = window.setTimeout;
  const clearTimer = window.clearTimeout;
  window.setTimeout = (callback, ms, ...rest) => {
    const timer = setTimer(() => {
      window.retryWaits.delete(timer);
      callback(...rest);
    }, ms);
    if (ms === RETRY_MS) window.retryWaits.add(timer);
    return timer;
  };
  window.clearTimeout = (timer) => {
    window.retryWaits.delete(timer);
    clearTimer(timer);
  };
  window.viewerStarted = () => !!frame()?.contentDocument?.getElementById('the_canvas_id')?.hasAttribute('tabindex');
</script>"""


def page_html(body: str) -> str:
    helpers = HELPERS.replace('%RETRY_MS%', str(1000 * RETRY_AFTER_S))
    return f'<!doctype html><html><head><meta charset="utf-8">{helpers}{SCRIPT}</head><body>{body}</body></html>'


def inline_element(attributes: str) -> str:
    data = json.dumps(inline_rows()).replace('</', '<\\/')
    return (
        f'<positronic-replay {attributes}><script type="application/json">{data}</script>'
        '<div slot="details" id="details">The details of the episode</div></positronic-replay>'
    )


def history_control(release: str) -> str:
    """Loads the viewer's own page, history on, in place in a frame."""
    viewer = f'/static/rerun/{release}/index.html'
    return page_html(f"""<iframe id="viewer" style="width:800px;height:500px"></iframe>
<script>
  window.loadInFrame = (n) => {{
    const rrd = encodeURIComponent(`${{location.origin}}/rrd/ok/${{n}}.rrd`);
    document.getElementById('viewer').contentWindow.location.replace(`{viewer}?url=${{rrd}}&hide_welcome_screen`);
  }};
</script>""")


def pages(release: str) -> dict[str, str]:
    compact = f'layout="compact" src="{SHARE}/replay.json" style="--list-max: 100px; --video-max: 130px"'
    return {
        '/page/share': page_html(f'<positronic-replay src="{SHARE}/long.json"></positronic-replay>'),
        '/page/compact': page_html(
            f'<positronic-replay {compact}><a slot="actions" href="#">Full replay</a></positronic-replay>'
        ),
        '/page/dock': page_html(inline_element('mode="full" dock link-param="episode"')),
        '/page/dock-video': page_html(inline_element('dock link-param="episode"')),
        '/page/history-control': history_control(release),
        '/page/slow': page_html(f'<positronic-replay src="{SHARE}/slow.json"></positronic-replay>'),
        '/page/small': page_html(f'<positronic-replay src="{SHARE}/replay.json"></positronic-replay>'),
        # The page the row link of row 2 opens.
        '/episode/2': '<!doctype html><title>An episode page</title>',
    }


# ── The server ───────────────────────────────────────────────────────────────

CONTENT_TYPES = {'.js': 'text/javascript', '.html': 'text/html', '.wasm': 'application/wasm', '.svg': 'image/svg+xml'}


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # The browser drops a download when the page moves on, as a visitor does.
        if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            super().handle_error(request, client_address)


def static_file(path: str) -> Path | None:
    """The file of positronic's static folder that `path` names; None for any other path."""
    file = (STATIC / path.removeprefix('/static/')).resolve()
    return file if path.startswith('/static/') and file.is_relative_to(STATIC) and file.is_file() else None


def serve(media: Media, requests: Requests, release: str) -> ThreadingHTTPServer:
    fixed = pages(release)
    slow = {**video_share(), 'title': 'The slow manifest'}
    manifests = {
        f'{SHARE}/replay.json': video_share(),
        f'{SHARE}/long.json': long_share(),
        f'{SHARE}/slow.json': slow,
        f'{SHARE}/broken.json': {'title': 'A manifest with no episodes'},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            del format, args

        def answer(self, status: int, body: bytes = b'', content_type: str = 'text/plain', headers=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def media_answer(self, path: str, body: bytes, content_type: str):
            asked = requests.add(path)
            if '/gone/' in path:
                return self.answer(404, b'gone')
            if '/building/' in path and asked <= BUILDING_ANSWERS:
                return self.answer(202, headers={'Retry-After': str(RETRY_AFTER_S)})
            return self.answer(200, body, content_type)

        def manifest_answer(self, path: str):
            if path == f'{SHARE}/slow.json':
                time.sleep(SLOW_S)
            return self.answer(200, json.dumps(manifests[path]).encode(), 'application/json')

        def do_GET(self):
            path = self.path.split('?')[0]
            if path.endswith('.rrd'):
                return self.media_answer(path, media.rrd, 'application/octet-stream')
            if path.endswith(('.mp4', '.webm')):
                return self.media_answer(path, media.video, 'video/webm')
            requests.add(path)
            if path in fixed:
                return self.answer(200, fixed[path].encode(), 'text/html; charset=utf-8')
            if path in manifests:
                return self.manifest_answer(path)
            file = static_file(path)
            if file is None:
                return self.answer(404)
            return self.answer(200, file.read_bytes(), CONTENT_TYPES.get(file.suffix, 'application/octet-stream'))

    server = QuietServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ── The checks ───────────────────────────────────────────────────────────────


class Probe:
    def __init__(self, browser: Browser, base: str, requests: Requests):
        self.browser = browser
        self.base = base
        self.requests = requests
        self.failures: list[str] = []
        self.console_errors: list[str] = []

    def check(self, name: str, passed: bool, seen: object = '') -> bool:
        print(f'  {"OK  " if passed else "FAIL"} {name}{f": {seen}" if seen != "" else ""}')
        if not passed:
            self.failures.append(name)
        return passed

    def open(self, path: str, width: int = 1400, height: int = 900, expected: tuple[str, ...] = ()) -> Page:
        """A page of `path`. A console error that holds one of `expected` is a result the check asks for."""
        page = self.browser.new_page(viewport={'width': width, 'height': height})
        # A tall strip above the dock keeps the rows next to the open one clear of the dock.
        page.add_init_script(f"localStorage.setItem('{STRIP_KEY}', '{DOCK_STRIP_PX}')")
        page.on('console', lambda message: self.on_console(path, message, expected))
        page.on('pageerror', lambda error: self.console_errors.append(f'{path}: {error}'))
        page.goto(self.base + path, wait_until='load')
        page.wait_for_function("customElements.get('positronic-replay') !== undefined", timeout=WAIT_MS)
        return page

    def on_console(self, path: str, message: ConsoleMessage, expected: tuple[str, ...]):
        if message.type != 'error' or any(text in message.text for text in expected):
            return
        source = message.location.get('url', '')
        # A gone file answers 404, and the browser says so in the console of the page that asked for it.
        if message.text.startswith('Failed to load resource') and '/gone/' in source:
            return
        self.console_errors.append(f'{path}: {message.text} ({source})')

    @staticmethod
    def actions(page: Page) -> list[dict]:
        return page.evaluate('window.actions.splice(0)')

    @staticmethod
    def until(page: Page, condition: str, timeout: int = WAIT_MS) -> bool:
        try:
            page.wait_for_function(condition, timeout=timeout)
            return True
        except PlaywrightTimeout:
            return False

    @staticmethod
    def history_length(page: Page) -> int:
        return len(page.context.new_cdp_session(page).send('Page.getNavigationHistory')['entries'])

    def scroll_to_more(self, page: Page):
        """Wheels to the end of the list, as a visitor does."""
        for _ in range(60):
            if page.evaluate("one('more').hidden || one('more').getBoundingClientRect().top < innerHeight + 300"):
                break
            page.mouse.wheel(0, 900)
            time.sleep(0.05)
        time.sleep(0.5)

    def rrd_requests(self, mark: int) -> list[str]:
        return [path for path in self.requests.since(mark) if path.endswith('.rrd')]

    def viewer_requests(self, mark: int) -> list[str]:
        return [path for path in self.requests.since(mark) if '/rerun/' in path]

    def nothing_detailed(self, page: Page, mark: int):
        self.check('the page asks for no recording', not self.rrd_requests(mark), self.rrd_requests(mark))
        self.check('the page loads no viewer page', not self.viewer_requests(mark), self.viewer_requests(mark))
        self.check('the element holds no viewer frame', page.evaluate('frame() === null'))

    # Each layout ──────────────────────────────────────────────────────────

    def share_page(self):
        print('A video share page (full layout, video mode, a manifest past one page):')
        mark = self.requests.mark()
        page = self.open('/page/share')
        manifest = long_share()
        total = len(manifest['episodes'])
        self.until(page, "all('row').length > 0")
        self.check(
            'the title shows', page.evaluate("one('header').querySelector('h1').textContent") == manifest['title']
        )
        facts = page.evaluate("one('about').querySelectorAll('dt').length")
        self.check('every fact shows', facts == len(manifest['about']), facts)
        heads = page.evaluate("all('head-cell').slice(0, 4).map((th) => th.textContent)")
        self.check(
            'the columns are the index, the manifest columns and the duration',
            heads == ['#', 'Task', 'Outcome', 'Duration'],
            heads,
        )
        self.check('the first page shows 50 rows', page.evaluate('shownRows()') == PAGE, page.evaluate('shownRows()'))
        more = page.evaluate("one('more').textContent")
        self.check('the more button counts the rest', more == f'Show {total - PAGE} more of {total - PAGE}', more)
        self.scroll_to_more(page)
        self.check(
            'a scroll to the end shows the next page',
            self.until(page, f'shownRows() === {total}'),
            page.evaluate('shownRows()'),
        )
        actions = self.actions(page)
        self.check('the scroll sends more/Scroll', {'action': 'more', 'via': 'Scroll'} in actions, actions)
        self.check('the first row is current', page.evaluate("all('row')[0].part.contains('current')"))
        cameras = page.evaluate("all('camera').length")
        self.check('the player shows each camera', cameras == len(manifest['episodes'][0]['videos']), cameras)
        page.locator('positronic-replay [part~="play"]').click()
        self.check('Play plays every camera', self.until(page, "all('camera').every((video) => !video.paused)", 8000))
        page.locator('positronic-replay [part~="row"]').nth(1).click()
        self.check('a row click makes the row current', page.evaluate("all('row')[1].part.contains('current')"))
        actions = self.actions(page)
        wanted = [{'action': 'play', 'row': '0', 'playing': True}, {'action': 'open', 'row': '1', 'from': 'Row'}]
        self.check('the controls send play and open', actions == wanted, actions)
        self.nothing_detailed(page, mark)
        page.close()

    def compact(self):
        print('A board drawer (compact layout, the fixture manifest as it is):')
        mark = self.requests.mark()
        page = self.open('/page/compact')
        self.until(page, "all('episode').length > 0")
        tally = page.evaluate("one('tally').textContent")
        self.check('the tally counts the episodes and the successes', tally == '3 episodes · 2 succeeded', tally)
        runs = page.evaluate("all('run').map((run) => run.getAttribute('part'))")
        wanted_runs = ['run outcome-success', 'run outcome-failure', 'run outcome-success']
        self.check('each run carries its outcome', runs == wanted_runs, runs)
        actions_slot = page.evaluate("host().shadowRoot.querySelector('slot[name=actions]').assignedNodes().length")
        self.check('the actions slot shows', actions_slot == 1, actions_slot)
        list_max = page.evaluate("getComputedStyle(one('list')).maxHeight")
        self.check('the list takes its height limit from --list-max', list_max == '100px', list_max)
        episode = page.locator('positronic-replay [part~="episode"]').nth(1)
        episode.click()
        self.check(
            'a row opens the screen', self.until(page, "host().hasAttribute('screen-open') && all('camera').length > 0")
        )
        self.until(page, "all('camera').every((video) => video.readyState >= 2)", 8000)
        height = page.evaluate("Math.max(...all('camera').map((video) => video.getBoundingClientRect().height))")
        self.check('the videos take their height limit from --video-max', 0 < height <= 130.5, height)
        self.check('the screen plays', self.until(page, "host().hasAttribute('playing')", 8000))
        episode.click()
        self.check('the open row pauses', self.until(page, "!host().hasAttribute('playing')", 8000))
        page.locator('positronic-replay [part~="screen-close"]').click()
        self.check('the close button closes the screen', not page.evaluate("host().hasAttribute('screen-open')"))
        actions = self.actions(page)
        wanted = [
            {'action': 'open', 'row': '1', 'from': 'Row'},
            {'action': 'play', 'row': '1', 'playing': False},
            {'action': 'close', 'row': '1', 'via': 'Button'},
        ]
        self.check('the controls send open, play and close', actions == wanted, actions)
        self.nothing_detailed(page, mark)
        page.close()

    def dock(self):
        print('The episode list with a dock (full mode):')
        page = self.open('/page/dock')
        self.until(page, "all('row').length > 0")
        cells = page.evaluate("""(() => {
          const cells = [...row('2').cells];
          return {link: cells[0].querySelector('a')?.href, icon: !!cells[1].querySelector('img'),
                  badge: cells[2].querySelector('[part~="badge"]')?.style.backgroundColor,
                  number: cells[3].textContent, unit: cells[4].textContent};
        })()""")
        wanted_cells = {
            'link': f'{self.base}/episode/2',
            'icon': True,
            'badge': 'rgb(25, 158, 112)',
            'number': '52',
            'unit': '0.2 s',
        }
        self.check('the cells render by type', cells == wanted_cells, cells)
        count = "one('count').textContent"
        self.check('the count names every row', page.evaluate(count) == f'{ROWS} episodes', page.evaluate(count))
        # A key press changes a select as a visitor does; `select_option` sends a change no visitor made.
        page.locator('positronic-replay select[data-key="result"]').press('p')
        partial = page.evaluate(count)
        self.check('a filter compares the key of a value, not its text', partial == f'{ROWS // 4} episodes', partial)
        page.locator('positronic-replay select[data-key="model"]').press('m')
        self.check(
            'two filters keep the rows that match both', page.evaluate(count) == '0 episodes', page.evaluate(count)
        )
        page.evaluate("host().show({model: 'b', result: '', task: 'Task 2'})")
        both = sum(1 for n in range(1, ROWS + 1) if n % 3 == 2 and n % 2)
        self.check('show() sets the filters', page.evaluate(count) == f'{both} episodes', page.evaluate(count))
        page.evaluate("host().show({model: 'no such model', task: ''})")
        self.check('show() with a value a filter lacks shows all', page.evaluate(count) == f'{ROWS} episodes')
        self.check('the first page shows 50 rows', page.evaluate('shownRows()') == PAGE, page.evaluate('shownRows()'))
        actions = self.actions(page)
        wanted = [
            {'action': 'filter', 'column': 'result', 'value': 'partial'},
            {'action': 'filter', 'column': 'model', 'value': 'b'},
        ]
        self.check('the selects send filter, and show() sends nothing', actions == wanted, actions)
        page.locator('positronic-replay [part~="more"]').click()
        self.check(
            'the more button shows the next page',
            page.evaluate('shownRows()') == 2 * PAGE,
            page.evaluate('shownRows()'),
        )
        self.scroll_to_more(page)
        self.check('a scroll to the end shows the rest', self.until(page, f'shownRows() === {ROWS}'))
        actions = self.actions(page)
        wanted = [{'action': 'more', 'via': 'Button'}, {'action': 'more', 'via': 'Scroll'}]
        self.check('the list sends more/Button and more/Scroll', actions == wanted, actions)
        self.media_states(page)
        self.dock_controls(page)
        page.close()

    def media_states(self, page: Page):
        def open_row(n: int):
            page.locator(f'positronic-replay tr[data-id="{n}"] td').nth(2).click()

        def note_says(text: str, timeout: int = 8000) -> bool:
            return self.until(page, f'noteText().includes({json.dumps(text)})', timeout)

        open_row(5)
        building = note_says('Preparing the 3D replay')
        self.check(
            'a building replay says so, and its cameras play meanwhile',
            building and page.evaluate("all('camera').length") == 2,
        )
        loaded = self.until(page, "loads('/rrd/building/5.rrd')") and self.until(page, 'viewerStarted()')
        asked = Counter(self.requests.paths)['/rrd/building/5.rrd']
        self.check('the replay loads after the retries', loaded and asked > BUILDING_ANSWERS, asked)
        open_row(6)
        self.check('a gone replay says so', note_says('3D replay of this episode is gone'), page.evaluate('noteText()'))
        self.check('a gone replay plays its cameras', page.evaluate("all('camera').length") == 2)
        open_row(9)
        self.check('a gone video says so', note_says('video of this episode is gone'), page.evaluate('noteText()'))
        open_row(10)
        self.check('a building video says so', note_says('building'), page.evaluate('noteText()'))
        built = self.until(page, "all('figure').every((figure) => figure.dataset.state === 'ready')", 15000)
        self.check('a building video plays once it is built', built, page.evaluate('noteText()'))
        open_row(8)
        self.check('a row with no media is not available', note_says('not available'), page.evaluate('noteText()'))

    def dock_controls(self, page: Page):
        toggle = page.locator('positronic-replay [part~="details-toggle"]')
        self.check('the details slot puts a Details button in the dock', toggle.is_visible())
        toggle.click()
        self.check('Details shows the slot', page.evaluate("!one('details').hidden"))
        page.locator('positronic-replay [part~="resize"]').focus()
        page.keyboard.press('ArrowUp')
        page.locator('positronic-replay [part~="step"]').nth(1).click()
        with page.context.expect_page() as popup:
            page.locator('positronic-replay [part~="full"]').click()
        self.check('Full page opens the row page', popup.value.url == f'{self.base}/episode/9', popup.value.url)
        popup.value.close()
        page.locator('positronic-replay [part~="close"]').click()
        actions = self.actions(page)
        wanted = [{'action': 'open', 'row': row, 'from': 'Row'} for row in ('5', '6', '9', '10', '8')] + [
            {'action': 'details', 'open': True},
            {'action': 'resize', 'via': 'Key'},
            {'action': 'open', 'row': '9', 'from': 'Step'},
            {'action': 'full', 'row': '9'},
            {'action': 'close', 'row': '9', 'via': 'Button'},
        ]
        self.check('each control sends its action', actions == wanted, actions)

    def dock_link(self, fresh_tab_entries: int):
        """`fresh_tab_entries` is the history length of a new tab with a page, from the control arm."""
        print('A link to a row on the third page (dock, full mode), and three replays:')
        page = self.open('/page/dock?episode=120')
        start = self.history_length(page)
        self.check('the link opens the dock', self.until(page, "one('dock') && !one('dock').hidden"))
        actions = self.actions(page)
        self.check(
            'the link open is the first action',
            actions[:1] == [{'action': 'open', 'row': '120', 'from': 'Link'}],
            actions,
        )
        current = page.evaluate("!row('120').hidden && row('120').part.contains('current')")
        self.check('the row is shown and current', current)
        position = page.evaluate("one('position').textContent")
        self.check('the dock counts its place', position == f'{ROWS} of {ROWS}', position)
        urls = []
        for row, key in (('120', None), ('119', 'ArrowUp'), ('118', 'ArrowUp')):
            if key:
                page.keyboard.press(key)
            loaded = self.until(page, f"loads('/rrd/ok/{row}.rrd')")
            time.sleep(SETTLE_S)
            urls.append(page.evaluate('frameUrl()') if loaded else None)
        embedded = f'/rerun/{rerun_release()}/embedded.html?url='
        self.check('each replay loads the embedded viewer page', all(url and embedded in url for url in urls), urls)
        end = self.history_length(page)
        fresh = fresh_tab_entries
        self.check('three replays add no browser history entry', start == end == fresh, (fresh, start, end))
        page.keyboard.press('Escape')
        self.check('Escape closes the dock', self.until(page, "one('dock').hidden"))
        self.check('closing removes the link parameter', 'episode=' not in page.url, page.url)
        actions = self.actions(page)
        wanted = [
            {'action': 'open', 'row': '119', 'from': 'Step'},
            {'action': 'open', 'row': '118', 'from': 'Step'},
            {'action': 'close', 'row': '118', 'via': 'Escape'},
        ]
        self.check('the steps and Escape send open/Step and close/Escape', actions == wanted, actions)
        page.close()

    def history_control(self) -> int:
        """The history length of a new tab with a page; a new tab holds about:blank before the page."""
        print('The control arm: the viewer page with history on, three times in a frame:')
        page = self.browser.new_page(viewport={'width': 1400, 'height': 900})
        page.goto(self.base + '/page/history-control', wait_until='load')
        start = self.history_length(page)
        for n in (1, 2, 3):
            page.evaluate(f'window.loadInFrame({n})')
            time.sleep(SETTLE_S)
        end = self.history_length(page)
        self.check('the viewer adds history entries, so the check below can fail', end > start, (start, end))
        page.close()
        return start

    def lost_lead_camera(self):
        print('A row whose longest camera is gone:')
        page = self.open('/page/dock?episode=11')
        survivor = "all('camera').find((video) => video.dataset.camera === 'wrist')"
        self.check('the other camera plays', self.until(page, f'!{survivor}.paused', 8000), page.evaluate('noteText()'))
        play = page.locator('positronic-replay [part~="play"]')
        play.click()
        self.check('Pause pauses the other camera', self.until(page, f'{survivor}.paused', 5000))
        play.click()
        self.check('Play plays the other camera', self.until(page, f'!{survivor}.paused', 5000))
        page.close()

    def rebuild_while_open(self):
        print('An attribute change while the dock is open:')
        page = self.open('/page/dock')
        self.until(page, "all('row').length > 0")
        page.locator('positronic-replay tr[data-id="2"] td').nth(2).click()
        self.until(page, "host().hasAttribute('open')")
        page.evaluate("host().setAttribute('mode', 'video')")
        self.check('the rebuild closes the dock', not page.evaluate("host().hasAttribute('open')"))
        self.check('the rebuild removes the link parameter', 'episode=' not in page.url, page.url)
        padding = page.evaluate('getComputedStyle(host()).paddingBottom')
        self.check('the rebuild leaves no dock padding', padding == '0px', padding)
        page.locator('positronic-replay tr[data-id="3"] td').nth(2).click()
        opened = self.until(page, "host().hasAttribute('open') && !one('dock').hidden && all('camera').length === 2")
        self.check('a new open works', opened and 'episode=3' in page.url, page.url)
        page.keyboard.press('Escape')
        self.check('Escape closes the new dock', self.until(page, "!host().hasAttribute('open')", 5000))
        page.close()

    def rebuild_frees_everything(self):
        print('Rebuilds while a row waits for its replay and its videos:')
        page = self.open('/page/dock')
        self.until(page, "all('row').length > 0")
        self.check('the element holds one media query listener', page.evaluate('queryListeners') == 1)
        page.locator('positronic-replay tr[data-id="12"] td').nth(2).click()
        waiting = self.until(page, 'retryWaits.size > 0', 8000)
        self.check('the row waits for its files', waiting, page.evaluate('retryWaits.size'))
        mark = self.requests.mark()
        page.evaluate("host().setAttribute('mode', 'video')")
        left = page.evaluate('retryWaits.size')
        self.check('a rebuild ends every wait', left == 0, left)
        for mode in ('full', 'video', 'full'):
            page.evaluate(f"host().setAttribute('mode', '{mode}')")
        listeners = page.evaluate('queryListeners')
        self.check('after four rebuilds the element holds one media query listener', listeners == 1, listeners)
        time.sleep(3 * RETRY_AFTER_S)
        asked = [path for path in self.requests.since(mark) if '/building/12' in path]
        self.check('no request for the old row follows the rebuild', not asked, asked)
        page.close()

    def stale_and_broken_sources(self):
        print('A new src while the old one loads, and a broken src after a good one:')
        page = self.open('/page/slow')
        page.evaluate(f"host().setAttribute('src', '{SHARE}/replay.json')")
        self.until(page, "all('row').length > 0")
        time.sleep(SLOW_S + 1)
        title = page.evaluate("one('header')?.querySelector('h1').textContent")
        self.check('the answer for the old src does not render', title == video_share()['title'], title)
        page.close()
        page = self.open('/page/small', expected=('does not read as a list of episodes',))
        self.until(page, "all('row').length > 0")
        page.evaluate(f"host().setAttribute('src', '{SHARE}/broken.json')")
        self.until(page, "one('status')?.textContent.includes('did not load')", 5000)
        page.evaluate("host().setAttribute('layout', 'compact')")
        time.sleep(0.5)
        status = page.evaluate("one('status')?.textContent ?? ''")
        rows = page.evaluate("all('row').length + all('episode').length")
        self.check(
            'a layout change after a broken src shows the error, not the old rows',
            'did not load' in status and rows == 0,
            (status, rows),
        )
        page.close()

    def row_link(self):
        print('The link in a row, and a click beside it:')
        page = self.open('/page/dock')
        self.until(page, "all('row').length > 0")
        with page.expect_navigation(timeout=WAIT_MS):
            page.locator('positronic-replay tr[data-id="2"] [part~="row-link"]').click()
        self.check('a plain click on the row link follows it', page.url == f'{self.base}/episode/2', page.url)
        page.close()
        page = self.open('/page/dock')
        self.until(page, "all('row').length > 0")
        page.locator('positronic-replay tr[data-id="2"] td').nth(2).click()
        self.check('a click beside the link opens the row', self.until(page, "host().hasAttribute('open')", 5000))
        page.close()

    def no_recording_row(self):
        print('A row with no recording, in full mode:')
        mark = self.requests.mark()
        page = self.open('/page/dock?episode=7')
        self.until(page, "all('camera').length > 0")
        page.locator('positronic-replay tr[data-id="8"] td').nth(2).click()
        time.sleep(2)
        self.nothing_detailed(page, mark)
        page.close()

    def video_mode(self):
        print('The same rows in video mode:')
        mark = self.requests.mark()
        page = self.open('/page/dock-video?episode=120')
        self.until(page, "all('camera').length > 0")
        for _ in range(2):
            page.keyboard.press('ArrowUp')
            time.sleep(0.5)
        for n in (5, 6):
            page.locator(f'positronic-replay tr[data-id="{n}"] td').nth(2).click()
            time.sleep(1)
        cameras = page.evaluate("all('camera').length")
        self.check('the rows play their cameras', cameras == 2, cameras)
        self.nothing_detailed(page, mark)
        page.close()

    def phone(self):
        print('A phone (full mode):')
        mark = self.requests.mark()
        page = self.open('/page/dock?episode=120', width=400, height=800)
        playing = self.until(page, "one('phone') !== null && !one('phone').paused", 8000)
        src = page.evaluate("one('phone')?.src")
        self.check('the phone video plays', playing and src == f'{self.base}/media/ok/120/phone.webm', src)
        self.nothing_detailed(page, mark)
        page.close()


def main(headed: bool) -> int:
    release = rerun_release()
    requests = Requests()
    with tempfile.TemporaryDirectory() as scratch:
        media = Media(video=make_video(Path(scratch, 'clip.webm')), rrd=make_recording(Path(scratch, 'replay.rrd')))
    server = serve(media, requests, release)
    base = f'http://127.0.0.1:{server.server_port}'
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headed, args=['--autoplay-policy=no-user-gesture-required'])
        probe = Probe(browser, base, requests)
        for check in (probe.share_page, probe.compact, probe.dock):
            check()
        probe.dock_link(probe.history_control())
        for check in (
            probe.lost_lead_camera,
            probe.rebuild_while_open,
            probe.rebuild_frees_everything,
            probe.stale_and_broken_sources,
            probe.row_link,
            probe.no_recording_row,
            probe.video_mode,
            probe.phone,
        ):
            check()
        browser.close()
    server.shutdown()
    print('Console errors:')
    for error in probe.console_errors:
        print(f'  FAIL {error}')
    if not probe.console_errors:
        print('  OK   none')
    failed = len(probe.failures) + len(probe.console_errors)
    print(f'{"FAIL" if failed else "OK"}: {failed} failures')
    return 1 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--headed', action='store_true')
    sys.exit(main(parser.parse_args().headed))
