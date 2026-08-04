# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "playwright",
# ]
# ///
"""Render a dataset-viewer page in a headless browser: screenshot each step, print the console.

The PEP 723 header carries the browser dependency, so run it with ``uv run --script``.

Two targets:

``--url`` opens a running server's episode page::

    uv run --script utilities/viewer_probe.py --url https://myhost:8400/episode/5

``--rrd`` serves a recording from disk to the viewer bundled with that server::

    uv run --script utilities/viewer_probe.py --rrd ./episode.rrd --viewer https://myhost:8400

Needs a Chrome (``channel='chrome'``) — the bundled Chromium ships without H.264, so video panels
stay blank there whatever the recording holds.
"""

import argparse
import functools
import http.server
import logging
import socket
import threading
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

RERUN_BUNDLE = 'static/rerun/0.30.0/index.html'


class CorsHandler(http.server.SimpleHTTPRequestHandler):
    """The viewer fetches the recording cross-origin, so it needs the header the stdlib omits."""

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):  # noqa: A002 — the stdlib's signature
        logging.debug(format % args)


def serve_directory(directory: Path) -> tuple[str, http.server.ThreadingHTTPServer]:
    with socket.socket() as probe_socket:
        probe_socket.bind(('127.0.0.1', 0))
        port = probe_socket.getsockname()[1]
    server = http.server.ThreadingHTTPServer(
        ('127.0.0.1', port), functools.partial(CorsHandler, directory=str(directory))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f'http://127.0.0.1:{port}', server


def rrd_url(rrd: Path, viewer: str) -> tuple[str, http.server.ThreadingHTTPServer]:
    origin, server = serve_directory(rrd.parent)
    recording = quote(f'{origin}/{rrd.name}', safe='')
    return f'{viewer.rstrip("/")}/{RERUN_BUNDLE}?url={recording}&hide_welcome_screen=', server


def probe(url: str, output_dir: Path, settle_s: float, steps: int, step_s: float) -> list[str]:
    """Screenshot the page after it settles and after each step of the time cursor."""
    output_dir.mkdir(parents=True, exist_ok=True)
    console: list[str] = []
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
        page.screenshot(path=str(output_dir / 'settled.png'))
        page.mouse.click(800, 500)  # the cursor keys go to the viewport, so focus it first
        for step in range(1, steps + 1):
            page.keyboard.press('ArrowRight')
            page.wait_for_timeout(step_s * 1000)
            page.screenshot(path=str(output_dir / f'step_{step:02d}.png'))
        browser.close()
    return console


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--url', help='episode page to open')
    target.add_argument('--rrd', type=Path, help="recording to serve to the viewer's bundled rerun")
    parser.add_argument('--viewer', default='http://127.0.0.1:8000', help='server whose rerun bundle renders --rrd')
    parser.add_argument('--output_dir', type=Path, default=Path('./probe_shots'))
    parser.add_argument(
        '--settle_s', type=float, default=25.0, help='wait before the first shot; the recording streams in'
    )
    parser.add_argument('--steps', type=int, default=6, help='time-cursor steps, one screenshot each')
    parser.add_argument('--step_s', type=float, default=1.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    file_server = None
    url = args.url
    if args.rrd is not None:
        url, file_server = rrd_url(args.rrd.resolve(), args.viewer)
    try:
        console = probe(url, args.output_dir, args.settle_s, args.steps, args.step_s)
    finally:
        if file_server is not None:
            file_server.shutdown()

    logging.info(f'Wrote {args.steps + 1} screenshots to {args.output_dir}')
    for line in console:
        if line.startswith(('error', 'pageerror')) or 'video' in line.lower():
            logging.warning(line)


if __name__ == '__main__':
    main()
