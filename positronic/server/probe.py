"""Render a dataset-viewer episode in a headless browser and report what the panels showed.

The viewer's failures are visual — a video panel that sits on its loading spinner, a plot that
never fills — and none of them reach the server log, so reading the code or curling the endpoint
answers nothing. This drives the real page, steps the time cursor, saves screenshots and prints the
browser console, which is what a report like "the video is black" has to be checked against.

Two targets:

``--url`` opens a running server's episode page::

    uv run --locked --extra probe positronic-server-probe --url=https://myhost:8400/episode/5

``--rrd`` serves a recording from disk to the viewer bundled with that server, which is how a
recording gets bisected: rebuild it with parts left out until the symptom flips::

    uv run --locked --extra probe positronic-server-probe --rrd=./episode.rrd --viewer=https://myhost:8400

Needs a Chrome (``channel='chrome'``) — the bundled Chromium ships without H.264, so video panels
stay blank there whatever the recording holds.
"""

import functools
import http.server
import logging
import socket
import threading
from pathlib import Path
from urllib.parse import quote

import configuronic as cfn

from positronic.utils.logging import init_logging

_RERUN_BUNDLE = 'static/rerun/0.30.0/index.html'


class _CorsHandler(http.server.SimpleHTTPRequestHandler):
    """The viewer fetches the recording cross-origin, so it needs the header the stdlib omits."""

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):  # noqa: A002 — the stdlib's signature
        logging.debug(format % args)


def _serve_directory(directory: Path) -> tuple[str, http.server.ThreadingHTTPServer]:
    with socket.socket() as probe_socket:
        probe_socket.bind(('127.0.0.1', 0))
        port = probe_socket.getsockname()[1]
    server = http.server.ThreadingHTTPServer(
        ('127.0.0.1', port), functools.partial(_CorsHandler, directory=str(directory))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f'http://127.0.0.1:{port}', server


def _rrd_url(rrd: Path, viewer: str) -> tuple[str, http.server.ThreadingHTTPServer]:
    origin, server = _serve_directory(rrd.parent)
    recording = quote(f'{origin}/{rrd.name}', safe='')
    return f'{viewer.rstrip("/")}/{_RERUN_BUNDLE}?url={recording}&hide_welcome_screen=', server


@cfn.config()
def main(
    output_dir: str = './probe_shots',
    url: str | None = None,
    rrd: str | None = None,
    viewer: str = 'http://127.0.0.1:8000',
    settle_s: float = 25.0,
    steps: int = 6,
    step_s: float = 1.5,
) -> None:
    """Screenshot a viewer page after it settles and after stepping the time cursor.

    Args:
        output_dir: Directory the screenshots are written to.
        url: Episode page to open. Mutually exclusive with ``rrd``.
        rrd: Recording to serve to ``viewer``'s bundled rerun. Mutually exclusive with ``url``.
        viewer: Server whose rerun bundle renders ``rrd``.
        settle_s: Seconds to wait after load before the first screenshot; the recording streams in.
        steps: Time-cursor steps to take, one screenshot each.
        step_s: Seconds to wait after each step.
    """
    from playwright.sync_api import sync_playwright  # noqa: PLC0415 — optional dependency

    if (url is None) == (rrd is None):
        raise ValueError('Pass exactly one of --url or --rrd')

    shots = Path(output_dir)
    shots.mkdir(parents=True, exist_ok=True)
    file_server = None
    if rrd is not None:
        url, file_server = _rrd_url(Path(rrd).resolve(), viewer)

    console: list[str] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel='chrome',
                headless=True,
                args=['--ignore-certificate-errors', '--autoplay-policy=no-user-gesture-required'],
            )
            page = browser.new_context(ignore_https_errors=True, viewport={'width': 1600, 'height': 1000}).new_page()
            page.on('console', lambda message: console.append(f'{message.type}: {message.text}'))
            page.on('pageerror', lambda error: console.append(f'pageerror: {error}'))
            page.goto(url, wait_until='load', timeout=120_000)
            page.wait_for_timeout(settle_s * 1000)
            page.screenshot(path=str(shots / 'settled.png'))
            page.mouse.click(800, 500)  # the cursor keys go to the viewport, so focus it first
            for step in range(1, steps + 1):
                page.keyboard.press('ArrowRight')
                page.wait_for_timeout(step_s * 1000)
                page.screenshot(path=str(shots / f'step_{step:02d}.png'))
            browser.close()
    finally:
        if file_server is not None:
            file_server.shutdown()

    logging.info(f'Wrote {steps + 1} screenshots to {shots}')
    for line in console:
        if line.startswith(('error', 'pageerror')) or 'video' in line.lower():
            logging.warning(line)


def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
