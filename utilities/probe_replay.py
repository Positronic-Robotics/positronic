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
from playwright.sync_api import Browser, ConsoleMessage, Error, Page, sync_playwright

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
VIDEO_S = 10
LONG_VIDEO_S = 15
SHORT_VIDEO_S = 2
"""The length of every fixture video, long enough to play across a width change."""
"""How long the slow manifest takes, so a newer `src` answers first."""
STRIP_KEY = 'positronic-replay.strip'
"""Where the element keeps the strip above its dock: `STRIP_KEY` in replay.js."""


@dataclass
class Media:
    def of(self, path: str) -> bytes:
        """The video a path serves, by its length folder."""
        return self.long_video if '/long/' in path else self.short_video if '/short/' in path else self.video

    video: bytes
    long_video: bytes
    """A video of `LONG_VIDEO_S`, which a path with `/long/` serves."""
    short_video: bytes
    """A video of `SHORT_VIDEO_S`, which a path with `/short/` serves."""
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


def make_video(path: Path, seconds: int = VIDEO_S) -> bytes:
    """A VP8 video of `seconds`: Chromium plays WebM without the codecs a branded browser adds."""
    with av.open(str(path), 'w', format='webm') as container:
        stream = container.add_stream('libvpx', rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(10 * seconds):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 2 % 256, dtype=np.uint8), format='rgb24')
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


def row_media(n: int) -> dict:
    """The media of row `n`. Row 5 builds its replay, 6 lost it, 7 has none, 8 has no media, 9 lost its videos, 10
    builds them, 11 lost its longest video only, 12 builds its replay and its videos, 13 builds its videos only, 14
    has a phone video only, which builds, 15 loses its longest video only after a wait,
    16 has three cameras: the longest lost after a wait, the next built after waits, and one that is there, 17 builds
    its replay and, more slowly, its videos, 18 has a longest camera that is there and one built after waits, and 19
    names no length, 20 names no length and its second camera is the longest, 21 builds its replay, 22 builds it after
    slow answers, 23 has a phone video shorter than its cameras,
    and 24 names no length for the row or its 15 s camera, and 10 s for its other camera."""
    cameras = [
        {'camera': camera, 'src': f'/media/ok/{n}/{camera}.webm', 'width': 64, 'height': 48, 'duration_s': VIDEO_S}
        for camera in ('front', 'wrist')
    ]

    def at(state: str, videos: list[dict]) -> list[dict]:
        return [{**video, 'src': video['src'].replace('/ok/', f'/{state}/')} for video in videos]

    def three_cameras() -> list[dict]:
        front, wrist = cameras
        side = {**at('slowbuilding', [{**front, 'camera': 'side'}])[0], 'duration_s': VIDEO_S + 0.5}
        side['src'] = side['src'].replace('front', 'side')
        return [{**at('slowgone', [front])[0], 'duration_s': VIDEO_S + 1}, side, wrist]

    def lost_longest(state: str) -> list[dict]:
        return [{**at(state, cameras[:1])[0], 'duration_s': VIDEO_S + 1}, cameras[1]]

    variants = {
        5: {'rrd': '/rrd/building/5.rrd'},
        6: {'rrd': '/rrd/gone/6.rrd'},
        7: {'rrd': None},
        8: {'rrd': None, 'cameras': None, 'phone': None},
        9: {'rrd': None, 'phone': None, 'cameras': at('gone', cameras)},
        10: {'rrd': None, 'phone': None, 'cameras': at('building', cameras)},
        11: {'rrd': None, 'phone': None, 'cameras': lost_longest('gone')},
        12: {'rrd': '/rrd/building/12.rrd', 'phone': None, 'cameras': at('building', cameras)},
        13: {'rrd': None, 'phone': None, 'cameras': at('building', cameras)},
        14: {'rrd': None, 'cameras': None, 'phone': '/media/building/14/phone.webm'},
        15: {'rrd': None, 'phone': None, 'cameras': lost_longest('slowgone')},
        16: {'rrd': None, 'phone': None, 'cameras': three_cameras()},
        17: {'rrd': '/rrd/building/17.rrd', 'phone': None, 'cameras': at('slowbuilding', cameras)},
        21: {'rrd': '/rrd/building/21.rrd'},
        23: {'rrd': None, 'phone': '/media/short/23/phone.webm'},
        24: {
            'rrd': None,
            'phone': None,
            'duration_s': None,
            'cameras': [
                cameras[0],
                {key: value for key, value in at('long', cameras[1:])[0].items() if key != 'duration_s'},
            ],
        },
        22: {'rrd': '/rrd/slowbuilding/22.rrd'},
        20: {
            'rrd': None,
            'phone': None,
            'duration_s': None,
            'cameras': [
                {key: value for key, value in camera.items() if key != 'duration_s'}
                for camera in [cameras[0], at('long', cameras[1:])[0]]
            ],
        },
        19: {
            'rrd': None,
            'phone': None,
            'duration_s': None,
            'cameras': [{key: value for key, value in camera.items() if key != 'duration_s'} for camera in cameras],
        },
        18: {
            'rrd': None,
            'phone': None,
            'cameras': [{**cameras[0], 'duration_s': VIDEO_S + 1}, *at('slowbuilding', cameras[1:])],
        },
    }
    media = {'cameras': cameras, 'phone': f'/media/ok/{n}/phone.webm', 'rrd': f'/rrd/ok/{n}.rrd', 'duration_s': VIDEO_S}
    media |= variants.get(n, {})
    return {key: value for key, value in media.items() if value is not None}


def inline_rows() -> dict:
    """Rows of every cell type and every media state of `row_media`."""
    rows = [
        {
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
            'href': f'/episode/{n}',
            **row_media(n),
        }
        for n in range(1, ROWS + 1)
    ]
    columns = [
        {'key': 'task', 'label': 'Task', 'filter': True},
        {'key': 'model', 'label': 'Model', 'type': 'icon', 'filter': {'all': 'All models'}},
        {
            'key': 'result',
            'label': 'Result',
            'type': 'badge',
            'filter': {'options': [['success', 'Success'], ['partial', 'Partial']]},
        },
        {'key': 'reliability', 'label': 'Reliability', 'type': 'number', 'decimals': 0, 'filter': True},
        {'key': 'pace', 'label': 'Pace', 'type': 'number', 'decimals': 1, 'unit': ' s'},
    ]
    return {'columns': columns, 'rows': rows}


def null_cells() -> str:
    """A column of each type, and two rows whose values are null or absent."""
    columns = [
        {'key': 'task', 'label': 'Task'},
        {'key': 'reliability', 'label': 'Reliability', 'type': 'number', 'decimals': 0},
        {'key': 'result', 'label': 'Result', 'type': 'badge'},
        {'key': 'model', 'label': 'Model', 'type': 'icon'},
    ]
    rows = [{'id': '1', 'values': dict.fromkeys(('task', 'reliability', 'result', 'model'))}, {'id': '2', 'values': {}}]
    data = json.dumps({'columns': columns, 'rows': rows})
    return f'<positronic-replay><script type="application/json">{data}</script></positronic-replay>'


def zero_filter() -> str:
    """A column whose filter is 0, which is no filter the element reads."""
    data = json.dumps({
        'columns': [{'key': 'task', 'label': 'Task', 'filter': 0}],
        'rows': [{'id': '1', 'values': {'task': 'x'}}],
    })
    return f'<positronic-replay><script type="application/json">{data}</script></positronic-replay>'


def bad_option() -> str:
    """A filter whose first option is null, over rows that hold the second."""
    columns = [{'key': 'task', 'label': 'Task', 'filter': {'options': [None, ['x', 'Task x']]}}]
    data = json.dumps({'columns': columns, 'rows': [{'id': '1', 'values': {'task': 'x'}}]})
    return f'<positronic-replay><script type="application/json">{data}</script></positronic-replay>'


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


def history_control() -> str:
    """Loads the viewer's own page, history on, in place in a frame."""
    viewer = f'/static/rerun/{rerun_release()}/index.html'
    return page_html(f"""<iframe id="viewer" style="width:800px;height:500px"></iframe>
<script>
  window.loadInFrame = (n) => {{
    const rrd = encodeURIComponent(`${{location.origin}}/rrd/ok/${{n}}.rrd`);
    document.getElementById('viewer').contentWindow.location.replace(`{viewer}?url=${{rrd}}&hide_welcome_screen`);
  }};
</script>""")


def pages() -> dict[str, str]:
    compact = f'layout="compact" src="{SHARE}/replay.json" style="--list-max: 100px; --video-max: 130px"'
    return {
        '/page/share': page_html(f'<positronic-replay src="{SHARE}/long.json"></positronic-replay>'),
        '/page/compact': page_html(
            f'<positronic-replay {compact}><a slot="actions" href="#">Full replay</a></positronic-replay>'
        ),
        '/page/dock': page_html(inline_element('mode="full" dock link-param="episode"')),
        '/page/dock-video': page_html(inline_element('dock link-param="episode"')),
        '/page/list': page_html(inline_element('mode="full" link-param="episode"')),
        '/page/nulls': page_html(null_cells()),
        '/page/zero-filter': page_html(zero_filter()),
        '/page/dock-below': page_html(
            inline_element('mode="full" dock dock-below="#no-such-bar" link-param="episode"')
        ),
        '/page/bad-option': page_html(bad_option()),
        '/page/compact-link': page_html(
            f'<positronic-replay layout="compact" link-param="episode" src="{SHARE}/replay.json"></positronic-replay>'
        ),
        '/page/names': page_html(
            f'<positronic-replay src="{SHARE}/replay.json" camera-names=\'["Outside"]\'></positronic-replay>'
        ),
        '/page/history-control': history_control(),
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


def media_status(path: str, asked: int) -> int | None:
    """The status of a media file that is not there: 404 for a gone one, after a wait for a slow one, and 202 for the
    first answers of a building one. None for a file that is there."""
    if '/slowgone/' in path or '/slowbuilding/' in path:
        time.sleep(SLOW_S)
    if 'gone/' in path:
        return 404
    if 'building/' in path and asked <= BUILDING_ANSWERS:
        return 202
    return None


def byte_span(header: str | None, body: bytes) -> tuple[int, bytes, dict[str, str]]:
    """The status, the bytes and the headers of an answer to a `Range: bytes=A-B` header, or to none. A browser
    seeks a video only where its server answers ranges."""
    if not header or not header.startswith('bytes='):
        return 200, body, {'Accept-Ranges': 'bytes'}
    first, _, last = header.removeprefix('bytes=').split(',')[0].partition('-')
    start, end = int(first or 0), min(int(last) if last else len(body) - 1, len(body) - 1)
    return 206, body[start : end + 1], {'Accept-Ranges': 'bytes', 'Content-Range': f'bytes {start}-{end}/{len(body)}'}


def static_file(path: str) -> Path | None:
    """The file of positronic's static folder that `path` names; None for any other path."""
    file = (STATIC / path.removeprefix('/static/')).resolve()
    return file if path.startswith('/static/') and file.is_relative_to(STATIC) and file.is_file() else None


def serve(media: Media, requests: Requests) -> ThreadingHTTPServer:
    fixed = pages()
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
            status = media_status(path, requests.add(path))
            if status is not None:
                return self.answer(status, headers={'Retry-After': str(RETRY_AFTER_S)} if status == 202 else None)
            status, part, headers = byte_span(self.headers.get('Range'), body)
            return self.answer(status, part, content_type, headers)

        def manifest_answer(self, path: str):
            if path == f'{SHARE}/slow.json':
                time.sleep(SLOW_S)
            return self.answer(200, json.dumps(manifests[path]).encode(), 'application/json')

        def do_GET(self):
            path = self.path.split('?')[0]
            if path.endswith('.rrd'):
                return self.media_answer(path, media.rrd, 'application/octet-stream')
            if path.endswith(('.mp4', '.webm')):
                return self.media_answer(path, media.of(path), 'video/webm')
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


def is_aborted_viewer_fetch(text: str, source: str) -> bool:
    """Whether an error is the rerun viewer's own report of a recording fetch that stopped when the element replaced
    the viewer page with the next episode or with a blank page. The probe counts these apart."""
    return 'Failed to fetch' in text and '/rerun/' in source


def button_label(page: Page, at: int) -> str:
    """The accessible name of the play button of compact row `at`."""
    return page.evaluate(f"all('episode-play')[{at}].getAttribute('aria-label')")


def camera_named(name: str) -> str:
    """The script that finds the camera video `name` in the element."""
    return f"all('camera').find((video) => video.dataset.camera === '{name}')"


def part(page: Page, name: str):
    """The element's nodes that carry the part `name`."""
    return page.locator(f'positronic-replay [part~="{name}"]')


def cell(page: Page, row: int):
    """A cell of the row `row` with no link in it."""
    return page.locator(f'positronic-replay tr[data-id="{row}"] td').nth(2)


class Probe:
    def __init__(self, browser: Browser, base: str, requests: Requests):
        self.browser = browser
        self.base = base
        self.requests = requests
        self.failures: list[str] = []
        self.console_errors: list[str] = []
        self.aborted_viewer_fetches: list[str] = []
        self.expected_errors: list[str] = []

    def check(self, name: str, passed: bool, seen: object = '') -> bool:
        print(f'  {"OK  " if passed else "FAIL"} {name}{f": {seen}" if seen != "" else ""}')
        if not passed:
            self.failures.append(name)
        return passed

    def open(
        self, path: str, width: int = 1400, height: int = 900, expected: tuple[str, ...] = (), wait_until: str = 'load'
    ) -> Page:
        """A page of `path`. A console error that holds one of `expected` is a result the check asks for. The `load`
        of a page waits for its videos to answer; `domcontentloaded` does not."""
        page = self.browser.new_page(viewport={'width': width, 'height': height})
        # A check reads the expected errors of its own page only.
        self.expected_errors = []
        # A tall strip above the dock keeps the rows next to the open one clear of the dock.
        page.add_init_script(f"localStorage.setItem('{STRIP_KEY}', '{DOCK_STRIP_PX}')")
        page.on('console', lambda message: self.on_console(path, message, expected))
        page.on('pageerror', lambda error: self.on_page_error(path, error))
        page.goto(self.base + path, wait_until=wait_until)
        page.wait_for_function("customElements.get('positronic-replay') !== undefined", timeout=WAIT_MS)
        return page

    def on_console(self, path: str, message: ConsoleMessage, expected: tuple[str, ...]):
        if message.type != 'error':
            return
        if any(text in message.text for text in expected):
            self.expected_errors.append(message.text)
            return
        source = message.location.get('url', '')
        # A gone file answers 404, and the browser says so in the console of the page that asked for it.
        if message.text.startswith('Failed to load resource') and 'gone/' in source:
            return
        if is_aborted_viewer_fetch(message.text, source):
            self.aborted_viewer_fetches.append(path)
            return
        self.console_errors.append(f'{path}: {message.text} ({source})')

    def on_page_error(self, path: str, error: Error):
        if is_aborted_viewer_fetch(error.message, error.stack or ''):
            self.aborted_viewer_fetches.append(path)
            return
        self.console_errors.append(f'{path}: {error}')

    @staticmethod
    def actions(page: Page) -> list[dict]:
        return page.evaluate('window.actions?.splice(0) ?? []')

    @staticmethod
    def until(page: Page, condition: str, timeout: int = WAIT_MS) -> bool:
        """Whether `condition` holds within `timeout`. A page that moved on, as after a followed link, fails it."""
        try:
            page.wait_for_function(condition, timeout=timeout)
            return True
        except Error:
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
        wanted = [
            {'action': 'play', 'row': '0', 'playing': True, 'via': 'Button'},
            {'action': 'open', 'row': '1', 'from': 'Row'},
        ]
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
            {'action': 'play', 'row': '1', 'playing': False, 'via': 'Row'},
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
            {'action': 'open', 'row': '9', 'from': 'Step', 'via': 'Button'},
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
            {'action': 'open', 'row': '119', 'from': 'Step', 'via': 'Key'},
            {'action': 'open', 'row': '118', 'from': 'Step', 'via': 'Key'},
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
        # One wait for the replay and one for each camera, so every first request of the row has an answer.
        waiting = self.until(page, 'retryWaits.size === 3', 8000)
        self.check('the row waits for its replay and its two videos', waiting, page.evaluate('retryWaits.size'))
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

    def building_video_autoplays(self):
        print('A row that opens playing while its videos build:')
        page = self.open('/page/dock?episode=13')
        lead = "all('camera').find((video) => video.dataset.camera === 'front')"
        self.until(page, "noteText().includes('building')", 8000)
        playing = self.until(page, f'{lead} && !{lead}.paused && {lead}.currentTime > 0.2', 15000)
        self.check('the video plays once the server has built it', playing, page.evaluate('noteText()'))
        page.close()
        page = self.open('/page/dock?episode=14')
        playing = self.until(
            page, "one('phone') !== null && !one('phone').paused && one('phone').currentTime > 0.2", 15000
        )
        actions = self.actions(page)
        self.check('a building phone video plays once built', playing, page.evaluate('noteText()'))
        self.check(
            'its retry sends no play action', actions == [{'action': 'open', 'row': '14', 'from': 'Link'}], actions
        )
        page.close()

    def used(self, page: Page, control: str, use, wanted: dict):
        """Uses one control, and checks that it sends `wanted` and nothing else."""
        use()
        time.sleep(0.8)
        actions = self.actions(page)
        self.check(f'{control} sends one action', actions == [wanted], actions)

    def every_control_reports_once(self):
        print('Every control a visitor uses, one at a time:')
        page = self.open('/page/dock')
        self.until(page, "all('row').length > 0")
        self.used(
            page,
            'the task filter',
            lambda: page.locator('positronic-replay select[data-key="task"]').press('T'),
            {'action': 'filter', 'column': 'task', 'value': 'Task 1'},
        )
        page.evaluate("host().show({task: ''})")
        self.used(page, 'the more button', lambda: part(page, 'more').click(), {'action': 'more', 'via': 'Button'})
        self.used(page, 'a scroll to the end', lambda: self.scroll_to_more(page), {'action': 'more', 'via': 'Scroll'})
        self.used(page, 'a row', lambda: cell(page, 7).click(), {'action': 'open', 'row': '7', 'from': 'Row'})
        self.until(page, "all('camera').length === 2 && !all('camera')[0].paused", 8000)
        self.used(
            page,
            'the play button',
            lambda: part(page, 'play').click(),
            {'action': 'play', 'row': '7', 'playing': False, 'via': 'Button'},
        )
        self.used(
            page,
            'a camera',
            lambda: part(page, 'camera').first.click(),
            {'action': 'play', 'row': '7', 'playing': True, 'via': 'Video'},
        )
        self.used(page, 'the seek bar', lambda: part(page, 'seek').press('ArrowRight'), {'action': 'seek', 'row': '7'})
        self.used(
            page,
            'the speed',
            lambda: part(page, 'speed').press('ArrowDown'),
            {'action': 'speed', 'row': '7', 'value': '2'},
        )
        self.used(
            page,
            'the next button',
            lambda: part(page, 'step').nth(1).click(),
            {'action': 'open', 'row': '8', 'from': 'Step', 'via': 'Button'},
        )
        self.used(
            page,
            'the up arrow',
            lambda: page.keyboard.press('ArrowUp'),
            {'action': 'open', 'row': '7', 'from': 'Step', 'via': 'Key'},
        )
        self.used(page, 'Details', lambda: part(page, 'details-toggle').click(), {'action': 'details', 'open': True})
        box = part(page, 'resize').bounding_box()

        def drag():
            page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2)
            page.mouse.down()
            page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2 - 60, steps=5)
            page.mouse.up()

        self.used(page, 'a drag of the dock edge', drag, {'action': 'resize', 'via': 'Drag'})
        self.used(
            page,
            'an arrow on the dock edge',
            lambda: part(page, 'resize').press('ArrowUp'),
            {'action': 'resize', 'via': 'Key'},
        )

        def full_page():
            with page.context.expect_page() as popup:
                part(page, 'full').click()
            popup.value.close()

        self.used(page, 'Full page', full_page, {'action': 'full', 'row': '7'})
        self.used(
            page,
            'the close button',
            lambda: part(page, 'close').click(),
            {'action': 'close', 'row': '7', 'via': 'Button'},
        )
        cell(page, 7).click()
        self.actions(page)
        self.used(
            page, 'Escape', lambda: page.keyboard.press('Escape'), {'action': 'close', 'row': '7', 'via': 'Escape'}
        )

        def follow_link():
            with page.expect_navigation(timeout=WAIT_MS):
                page.locator('positronic-replay tr[data-id="2"] [part~="row-link"]').click()

        page.evaluate(
            "addEventListener('pagehide', () => sessionStorage.setItem('actions', JSON.stringify(window.actions)))"
        )
        follow_link()
        actions = json.loads(page.evaluate("sessionStorage.getItem('actions') ?? '[]'"))
        self.check('the row link sends one action', actions == [{'action': 'link', 'row': '2'}], actions)
        page.close()

        page = self.open('/page/compact')
        self.until(page, "all('episode').length > 0")
        self.used(
            page,
            'a compact row',
            lambda: part(page, 'episode-values').nth(1).click(),
            {'action': 'open', 'row': '1', 'from': 'Row'},
        )
        self.until(page, "host().hasAttribute('playing')", 8000)
        self.used(
            page,
            'its play button',
            lambda: part(page, 'episode-play').nth(1).click(),
            {'action': 'play', 'row': '1', 'playing': False, 'via': 'Button'},
        )
        self.used(
            page,
            'its run',
            lambda: part(page, 'run').nth(1).click(),
            {'action': 'play', 'row': '1', 'playing': True, 'via': 'Run'},
        )
        self.until(page, "host().hasAttribute('playing')", 8000)
        self.used(
            page,
            'a camera on the screen',
            lambda: part(page, 'camera').first.click(),
            {'action': 'play', 'row': '1', 'playing': False, 'via': 'Video'},
        )
        self.used(
            page,
            'the play button on the screen',
            lambda: part(page, 'play').click(),
            {'action': 'play', 'row': '1', 'playing': True, 'via': 'Button'},
        )
        self.used(
            page,
            'the screen close button',
            lambda: part(page, 'screen-close').click(),
            {'action': 'close', 'row': '1', 'via': 'Button'},
        )
        self.used(
            page, 'another run', lambda: part(page, 'run').first.click(), {'action': 'open', 'row': '0', 'from': 'Run'}
        )
        page.close()

        page = self.open('/page/dock?episode=7', width=400, height=800)
        self.until(page, "one('phone') !== null && !one('phone').paused", 8000)
        self.check('the phone video keeps the browser controls', page.evaluate("one('phone').hasAttribute('controls')"))
        actions = self.actions(page)
        self.check(
            'its autoplay sends no play action', actions == [{'action': 'open', 'row': '7', 'from': 'Link'}], actions
        )
        # The browser's controls change the video as a call from the page does: the element made neither call.
        native = [
            ('a pause on its controls', 'pause()', {'action': 'play', 'row': '7', 'playing': False, 'via': 'Native'}),
            ('a play on its controls', 'play()', {'action': 'play', 'row': '7', 'playing': True, 'via': 'Native'}),
            ('a seek on its controls', 'currentTime = 3', {'action': 'seek', 'row': '7', 'via': 'Native'}),
            (
                'a drag of its seek bar',
                'currentTime = 4; one("phone").currentTime = 5; one("phone").currentTime = 6',
                {'action': 'seek', 'row': '7', 'via': 'Native'},
            ),
            (
                'a speed on its controls',
                'playbackRate = 2',
                {'action': 'speed', 'row': '7', 'value': '2', 'via': 'Native'},
            ),
        ]
        for control, call, wanted in native:
            self.used(page, control, lambda call=call: page.evaluate(f"one('phone').{call}"), wanted)
        page.close()

    def null_values(self):
        print('Rows whose values are null or absent:')
        page = self.open('/page/nulls')
        rendered = self.until(page, "all('row').length === 2", 5000)
        cells = page.evaluate("all('row').map((row) => [...row.cells].map((cell) => cell.textContent))")
        self.check('each typed cell renders empty', rendered and cells == [['', '-', '-', '']] * 2, cells)
        page.close()

    def lead_lost_while(self, path: str, wanted_playing: bool, name: str):
        """The visitor presses the play button while the lead still waits for its answer, which is a 404."""
        page = self.open(path, wait_until='domcontentloaded')
        lead = "all('camera').find((video) => video.dataset.camera === 'front')"
        survivor = "all('camera').find((video) => video.dataset.camera === 'wrist')"
        self.until(page, f'{lead} !== undefined && {lead}.error === null', 8000)
        part(page, 'play').click()
        failed = page.evaluate(f'{lead}.error !== null')
        self.check('the visitor pressed play before the lead failed', not failed)
        self.until(page, "noteText().includes('gone')", 15000)
        time.sleep(1)
        playing = page.evaluate(f'!{survivor}.paused')
        self.check(name, playing == wanted_playing, f'the other camera plays: {playing}')
        page.close()

    def lead_lost_keeps_state(self):
        print('A lead camera lost after a visitor changed the play state:')
        self.lead_lost_while('/page/dock?episode=15', False, 'an autoplay row the visitor paused stays paused')
        self.lead_lost_while('/page/list?episode=15', True, 'a row the visitor played plays on')

    def script_clicks(self):
        print('Script calls on the controls:')
        page = self.open('/page/dock')
        self.until(page, "all('row').length > 0")
        cell(page, 20).click()
        self.until(page, "host().hasAttribute('open')")
        self.actions(page)
        page.evaluate("all('step')[1].click()")
        page.evaluate("document.dispatchEvent(new KeyboardEvent('keydown', {key: 'ArrowDown'}))")
        time.sleep(0.5)
        actions = self.actions(page)
        position = page.evaluate("one('position').textContent")
        self.check(
            'a script click and a script key step and send no action',
            actions == [] and position.startswith('22 '),
            (actions, position),
        )
        self.used(
            page,
            'a visitor click on the next button',
            lambda: part(page, 'step').nth(1).click(),
            {'action': 'open', 'row': '23', 'from': 'Step', 'via': 'Button'},
        )
        page.close()

    def three_cameras(self):
        print('Three cameras: the longest lost after a wait, the next built after waits, one there at once:')
        page = self.open('/page/dock?episode=16', wait_until='domcontentloaded')
        between = self.until(
            page,
            f"{camera_named('front')}.parentElement.dataset.state === 'gone' && "
            f"{camera_named('side')}.parentElement.dataset.state === 'building'",
            10000,
        )
        self.check('the longest camera is gone while the next still builds', between)
        note = page.evaluate('noteText()')
        self.check('the note names the lost and the building camera', 'is gone' in note and 'is building' in note, note)
        before = page.evaluate(f"[Number(one('seek').value), {camera_named('wrist')}.currentTime]")
        time.sleep(1)
        after = page.evaluate(
            f"[Number(one('seek').value), {camera_named('wrist')}.currentTime, !{camera_named('wrist')}.paused]"
        )
        drives = after[1] - before[1] > 0.5 and abs(after[0] - after[1]) < 0.3 and after[2]
        self.check('the camera that is there leads and plays meanwhile', drives, (before, after))
        built = self.until(page, f"{camera_named('side')}.parentElement.dataset.state === 'ready'", 15000)
        time.sleep(1)
        side = page.evaluate(
            f"[Number(one('seek').value), {camera_named('side')}.currentTime, !{camera_named('side')}.paused]"
        )
        self.check('the built camera leads once it is there', built and abs(side[0] - side[1]) < 0.3 and side[2], side)
        page.close()

    def host_inputs(self):
        print('Host inputs of the wrong shape:')
        page = self.open('/page/names', expected=('camera-names',))
        self.until(page, "all('caption').length > 0")
        captions = page.evaluate("all('caption').map((caption) => caption.textContent)")
        keys = [video['camera'] for video in video_share()['episodes'][0]['videos']]
        errors = [text for text in self.expected_errors if 'camera-names' in text]
        self.check('a camera-names that is no object logs one error', len(errors) == 1, errors)
        self.check('the cameras then show their keys', captions == keys, captions)
        page.close()
        page = self.open('/page/dock')
        self.until(page, "all('row').length > 0")
        as_number = page.evaluate("host().show({reliability: 52}), one('count').textContent")
        as_text = page.evaluate("host().show({reliability: '52'}), one('count').textContent")
        self.check(
            'show() takes a number as its text', as_number == as_text != f'{ROWS} episodes', (as_number, as_text)
        )
        page.close()

    def list_entries_and_link_param(self):
        print('A bad entry of a list, and link-param where it does not apply:')
        page = self.open('/page/bad-option', expected=('options[0]',))
        self.until(page, "all('row').length > 0", 5000)
        options = page.evaluate("[...host().shadowRoot.querySelectorAll('select option')].map((o) => o.textContent)")
        errors = [text for text in self.expected_errors if 'options[0]' in text]
        self.check('the good option renders', options == ['All', 'Task x'], options)
        self.check('the null option logs one error', len(errors) == 1, errors)
        page.close()
        page = self.open('/page/compact-link?episode=1', expected=('link-param',))
        self.until(page, "all('episode').length > 0", 5000)
        time.sleep(0.5)
        errors = [text for text in self.expected_errors if 'link-param' in text]
        opened = page.evaluate("host().hasAttribute('screen-open')")
        self.check('link-param on the compact layout logs one error', len(errors) == 1, errors)
        self.check('and opens nothing', not opened and self.actions(page) == [], opened)
        page.close()
        page = self.open('/page/list?episode=30')
        self.until(page, "row('30').part.contains('current')", 5000)
        page.evaluate("host().setAttribute('link-param', 'other')")
        time.sleep(0.3)
        self.check('a new link-param changes nothing', page.evaluate("row('30').part.contains('current')"))
        cell(page, 31).click()
        self.until(page, "row('31').part.contains('current')", 5000)
        self.check(
            'a row still writes the first link-param, and no other',
            'episode=31' in page.url and 'other=' not in page.url,
            page.url,
        )
        page.close()

    def take_over_and_join(self):
        print('A recording that takes over from building videos, and a camera built under a paused lead:')
        page = self.open('/page/dock?episode=17', wait_until='domcontentloaded')
        took_over = self.until(page, "loads('/rrd/building/17.rrd')", 15000)
        mark = self.requests.mark()
        time.sleep(3 * SLOW_S)
        asked = [path for path in self.requests.since(mark) if '/17/' in path and not path.endswith('.rrd')]
        self.check('the replaced videos ask for nothing more', took_over and not asked, asked)
        page.close()
        page = self.open('/page/dock?episode=18', wait_until='domcontentloaded')
        front, wrist = camera_named('front'), camera_named('wrist')
        self.until(page, f'{front} && !{front}.paused && {front}.currentTime > 1', 8000)
        part(page, 'play').click()
        built = self.until(page, f"{wrist}.parentElement.dataset.state === 'ready'", 20000)
        time.sleep(1)
        times = page.evaluate(f'[{front}.currentTime, {wrist}.currentTime, {front}.paused, {wrist}.paused]')
        joined = abs(times[0] - times[1]) < 0.3 and times[2] and times[3]
        self.check('a camera built under a paused lead takes its time and stays paused', built and joined, times)
        page.close()

    def dock_below_and_length(self):
        print('A dock-below that matches nothing, and a row that names no length:')
        page = self.open('/page/dock-below?episode=7', expected=('dock-below',))
        self.until(page, "host().hasAttribute('open')", 5000)
        for size in ((1300, 850), (1400, 900)):
            page.set_viewport_size({'width': size[0], 'height': size[1]})
            time.sleep(0.3)
        errors = [text for text in self.expected_errors if 'dock-below' in text]
        self.check('a dock-below that matches nothing logs one error', len(errors) == 1, errors)
        page.close()
        page = self.open('/page/dock?episode=19')
        lengthed = self.until(page, f"Math.abs(Number(one('seek').max) - {VIDEO_S}) < 0.2", 8000)
        clock = page.evaluate("[one('seek').max, one('clock').textContent]")
        self.check(
            'a row with no length takes the length of its video', lengthed and clock[1].endswith('0:10.0'), clock
        )
        page.close()

    def src_removed(self):
        print('A src removed while a row plays, then an inline script added:')
        page = self.open('/page/small')
        self.until(page, "all('camera').length > 0")
        part(page, 'play').click()
        self.until(page, "all('camera').every((video) => !video.paused)", 8000)
        page.evaluate("window.removed = all('camera'); host().removeAttribute('src')")
        time.sleep(0.5)
        state = page.evaluate("[all('camera').length, one('player') === null, removed.every((video) => video.paused)]")
        self.check('no video plays and no player shows', state == [0, True, True], state)
        status = page.evaluate("one('status')?.textContent ?? ''")
        self.check('the element says it waits for data', 'Waiting' in status, status)
        data = json.dumps({
            'columns': [{'key': 'task', 'label': 'Task'}],
            'rows': [{'id': '1', 'values': {'task': 'x'}}],
        })
        page.evaluate(
            "(data) => { const script = document.createElement('script'); script.type = 'application/json';"
            ' script.textContent = data; host().append(script); }',
            data,
        )
        self.check('an inline script then renders', self.until(page, "all('row').length === 1", 5000))
        page.close()

    def rate_and_keyboard(self):
        print('The speed under late and new leads, and a row opened from the keyboard:')
        page = self.open('/page/dock?episode=18', wait_until='domcontentloaded')
        front, wrist = camera_named('front'), camera_named('wrist')
        self.until(page, f'{front} && !{front}.paused', 8000)
        part(page, 'speed').press('ArrowDown')
        built = self.until(page, f"{wrist}.parentElement.dataset.state === 'ready'", 20000)
        rates = page.evaluate(f'[{front}.playbackRate, {wrist}.playbackRate]')
        self.check('a camera ready after the visitor chose 2x plays at 2x', built and rates == [2, 2], rates)
        page.close()
        page = self.open('/page/dock?episode=16', wait_until='domcontentloaded')
        side = camera_named('side')
        self.until(page, f'{camera_named("wrist")} && !{camera_named("wrist")}.paused', 8000)
        part(page, 'speed').press('ArrowDown')
        # At 2x the seek bar moves 0.5 s of video between two `timeupdate` events, so it may lag that much.
        led = self.until(page, f"{side}.parentElement.dataset.state === 'ready'", 20000)
        time.sleep(0.5)
        state = page.evaluate(
            f"[{side}.playbackRate, one('speed').value, Math.abs(Number(one('seek').value) - {side}.currentTime) < 0.6]"
        )
        self.check('a new lead plays at 2x and its speed control shows 2x', led and state == [2, '2', True], state)
        page.close()
        page = self.open('/page/dock')
        self.until(page, "all('row').length > 0")
        on_row = 'host().shadowRoot.activeElement?.matches(\'tr[part~="row"]\')'
        for _ in range(12):
            page.keyboard.press('Tab')
            if page.evaluate(on_row):
                break
        focused = page.evaluate(f'{on_row} ? host().shadowRoot.activeElement.dataset.id : null')
        self.actions(page)
        page.keyboard.press('Enter')
        opened = self.until(page, "host().hasAttribute('open')", 5000)
        actions = self.actions(page)
        wanted = [{'action': 'open', 'row': focused, 'from': 'Row', 'via': 'Key'}]
        self.check(
            'Tab reaches a row, and Enter opens it',
            focused is not None and opened and actions == wanted,
            (focused, actions),
        )
        page.close()

    def rate_across_widths_and_real_length(self):
        print('The speed across a width change, and the lead of a row that names no length:')
        page = self.open('/page/dock?episode=7')
        front = camera_named('front')
        self.until(page, f'{front} && !{front}.paused', 8000)
        part(page, 'speed').press('ArrowDown')
        page.set_viewport_size({'width': 400, 'height': 800})
        narrow = self.until(page, "one('phone') !== null && one('phone').playbackRate === 2", 5000)
        self.check(
            'the phone video plays at the rate the visitor chose', narrow, page.evaluate("one('phone')?.playbackRate")
        )
        page.set_viewport_size({'width': 1400, 'height': 900})
        wide = self.until(page, f"{front} && {front}.playbackRate === 2 && one('speed').value === '2'", 5000)
        self.check(
            'the cameras and their control keep that rate',
            wide,
            page.evaluate(f"[{front}?.playbackRate, one('speed')?.value]"),
        )
        page.close()
        page = self.open('/page/dock?episode=20')
        longest = self.until(page, f"Math.abs(Number(one('seek').max) - {LONG_VIDEO_S}) < 0.2", 8000)
        self.check('the longest video leads once the lengths are known', longest, page.evaluate("one('seek').max"))
        page.close()

    def width_change_around_the_viewer(self):
        print('A width change while the recording is pending, and after the viewer took over:')
        page = self.open('/page/dock?episode=22', wait_until='domcontentloaded')
        self.until(page, "host().hasAttribute('open')", 5000)
        page.set_viewport_size({'width': 400, 'height': 800})
        playing = self.until(page, "one('phone') !== null && !one('phone').paused", 8000)
        self.check('an autoplay row still waiting for its recording plays its phone video', playing)
        page.close()
        page = self.open('/page/dock?episode=21', wait_until='domcontentloaded')
        self.until(page, "one('speed') !== null", 8000)
        part(page, 'speed').press('ArrowDown')
        took_over = self.until(page, "loads('/rrd/building/21.rrd')", 15000)
        page.set_viewport_size({'width': 400, 'height': 800})
        phone = "one('phone') !== null && !one('phone').paused && one('phone').playbackRate === 2"
        self.check(
            'after the viewer took over, the phone video plays at the chosen rate',
            took_over and self.until(page, phone, 8000),
            page.evaluate("[one('phone')?.paused, one('phone')?.playbackRate]"),
        )
        page.close()

    def end_and_zero_filter(self):
        print('A phone video that ends before a width change, and a filter of 0:')
        page = self.open('/page/dock?episode=23', width=400, height=800)
        ended = self.until(page, "one('phone') !== null && one('phone').ended", 15000)
        page.set_viewport_size({'width': 1400, 'height': 900})
        front = camera_named('front')
        settled = self.until(
            page, f'{front} && {front}.readyState >= 1 && Math.abs({front}.currentTime - {SHORT_VIDEO_S}) < 0.3', 8000
        )
        time.sleep(0.5)
        state = page.evaluate(f'[{front}?.paused, {front}?.currentTime]')
        self.check('the cameras show the end of the phone video, paused', ended and settled and state[0] is True, state)
        page.close()
        page = self.open('/page/zero-filter', expected=('filter',))
        self.until(page, "all('row').length > 0", 5000)
        errors = [text for text in self.expected_errors if 'columns[0].filter' in text]
        selects = page.evaluate("host().shadowRoot.querySelectorAll('select').length")
        self.check(
            'a filter of 0 logs one error and gives no filter', len(errors) == 1 and selects == 0, (errors, selects)
        )
        page.close()

    def src_restored_and_mixed_length(self):
        print('A src removed and restored, and a row whose longest camera names no length:')
        page = self.open('/page/small')
        self.until(page, "all('row').length > 0")
        page.evaluate("window.first = host().getAttribute('src'); host().removeAttribute('src')")
        self.until(page, "one('status')?.textContent.includes('Waiting')", 5000)
        page.evaluate("host().setAttribute('src', window.first)")
        self.check('the restored src renders again', self.until(page, "all('row').length > 0", 5000))
        page.close()
        page = self.open('/page/dock?episode=24')
        reaches = self.until(page, f"Math.abs(Number(one('seek').max) - {LONG_VIDEO_S}) < 0.2", 8000)
        self.check('the seek bar reaches the end of the longest camera', reaches, page.evaluate("one('seek').max"))
        page.close()

    def link_to_no_row(self):
        print('A link that names no row of the list:')
        for path, docked in (('/page/dock?episode=999', True), ('/page/list?episode=999', False)):
            warnings = []
            page = self.browser.new_page(viewport={'width': 1400, 'height': 900})
            page.on('console', lambda message, seen=warnings: message.type == 'warning' and seen.append(message.text))
            page.goto(self.base + path, wait_until='load')
            self.until(page, "all('row').length > 0", 8000)
            time.sleep(0.3)
            named = [text for text in warnings if 'episode=999' in text]
            actions = self.actions(page)
            layout = 'dock' if docked else 'list'
            self.check(f'the {layout} logs one warning', len(named) == 1, warnings)
            self.check(f'the {layout} drops the parameter', 'episode=' not in page.url, page.url)
            self.check(f'the {layout} sends no open', not [a for a in actions if a['action'] == 'open'], actions)
            if docked:
                self.check('the dock stays closed', not page.evaluate("host().hasAttribute('open')"))
            else:
                self.check('the list selects its first row', page.evaluate("row('1').part.contains('current')"))
            page.close()

    def swap_reports_and_button_names(self):
        print('No action from a swap of the phone video, and the names of the compact play buttons:')
        page = self.open('/page/dock?episode=7', width=400, height=800)
        self.until(page, "one('phone') !== null && !one('phone').paused", 8000)
        page.evaluate("one('phone').playbackRate = 2")
        time.sleep(0.4)
        self.actions(page)
        page.set_viewport_size({'width': 1400, 'height': 900})
        self.until(page, "all('camera').length === 2", 5000)
        part(page, 'close').click()
        time.sleep(0.5)
        actions = self.actions(page)
        self.check(
            'a width swap and a close send no speed action', not [a for a in actions if a['action'] == 'speed'], actions
        )
        page.close()
        page = self.open('/page/compact')
        self.until(page, "all('episode').length > 0")
        part(page, 'episode-values').nth(1).click()
        self.until(page, "host().hasAttribute('playing')", 8000)
        self.check(
            'a playing row names its button Pause', button_label(page, 1).startswith('Pause'), button_label(page, 1)
        )
        part(page, 'episode-play').nth(1).click()
        self.until(page, "!host().hasAttribute('playing')", 5000)
        self.check(
            'a paused row names its button Play', button_label(page, 1).startswith('Play'), button_label(page, 1)
        )
        part(page, 'episode-values').nth(2).click()
        self.until(page, "host().hasAttribute('playing')", 8000)
        self.check(
            'another row names the old button Play',
            button_label(page, 1).startswith('Play') and button_label(page, 2).startswith('Pause'),
            (button_label(page, 1), button_label(page, 2)),
        )
        page.close()

    def undocked_link(self):
        print('A link to a row of a replay with no dock:')
        page = self.open('/page/list?episode=30')
        self.until(page, "row('30').part.contains('current')", 5000)
        actions = self.actions(page)
        self.check(
            'the link open sends open/Link', actions[:1] == [{'action': 'open', 'row': '30', 'from': 'Link'}], actions
        )
        page.close()

    def width_change(self):
        print('A width change while a row plays:')
        page = self.open('/page/dock?episode=7')
        lead = "all('camera').find((video) => video.dataset.camera === 'front')"
        self.until(page, f'{lead} && !{lead}.paused && {lead}.currentTime > 1', 8000)
        at = page.evaluate(f'{lead}.currentTime')
        page.set_viewport_size({'width': 400, 'height': 800})
        moved = f"one('phone') !== null && !one('phone').paused && one('phone').currentTime >= {at - 0.3}"
        playing = self.until(page, moved, 5000)
        moment = page.evaluate("one('phone')?.currentTime ?? 0")
        self.check('the phone video plays on from the same moment', playing and moment >= at - 0.3, (at, moment))
        page.set_viewport_size({'width': 1400, 'height': 900})
        playing = self.until(page, f'{lead} && !{lead}.paused && {lead}.currentTime >= {moment - 0.3}', 5000)
        back = page.evaluate(f'{lead}?.currentTime ?? 0')
        self.check('the cameras play on from the same moment', playing and back >= moment - 0.3, (moment, back))
        actions = self.actions(page)
        self.check(
            'the width swaps send no play action', actions == [{'action': 'open', 'row': '7', 'from': 'Link'}], actions
        )
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
    requests = Requests()
    with tempfile.TemporaryDirectory() as scratch:
        media = Media(
            video=make_video(Path(scratch, 'clip.webm')),
            long_video=make_video(Path(scratch, 'long.webm'), LONG_VIDEO_S),
            short_video=make_video(Path(scratch, 'short.webm'), SHORT_VIDEO_S),
            rrd=make_recording(Path(scratch, 'replay.rrd')),
        )
    server = serve(media, requests)
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
            probe.width_change,
            probe.three_cameras,
            probe.host_inputs,
            probe.undocked_link,
            probe.list_entries_and_link_param,
            probe.take_over_and_join,
            probe.dock_below_and_length,
            probe.src_removed,
            probe.rate_and_keyboard,
            probe.rate_across_widths_and_real_length,
            probe.width_change_around_the_viewer,
            probe.end_and_zero_filter,
            probe.src_restored_and_mixed_length,
            probe.link_to_no_row,
            probe.swap_reports_and_button_names,
            probe.null_values,
            probe.lead_lost_keeps_state,
            probe.script_clicks,
            probe.building_video_autoplays,
            probe.every_control_reports_once,
            probe.no_recording_row,
            probe.video_mode,
            probe.phone,
        ):
            check()
        browser.close()
    server.shutdown()
    print(f'Aborted viewer fetches, which a replaced viewer page reports: {len(probe.aborted_viewer_fetches)}')
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
