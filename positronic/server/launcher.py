"""FastAPI server that manages named presets of policy-inference commands.

Presets live in a TOML file the operator edits by hand. Each preset is the full policy contract
(remote endpoint plus how observations are shaped for it). Starting a run pastes the preset through
``bash -c`` so ``$(date ...)`` and ``$ENV_VARS`` expand at launch, exactly as they do in tmux today.
The launched process serves its own robot console; the launcher only detects that console's readiness,
streams the process logs, and stops it with SIGINT (Ctrl-C) or SIGKILL.
"""

import logging
import os
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
import tomllib
from collections import deque
from datetime import datetime
from pathlib import Path

import configuronic as cfn
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from positronic.utils.logging import init_logging

_DATE_RE = re.compile(r'\$\(date \+([^)]+)\)')
_ARG_SPLIT_RE = re.compile(r'\s+(?=--)')
_ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

_LOG_BUFFER_LINES = 20000
_CONSOLE_CACHE_SECONDS = 2.0
_CONSOLE_PROBE_TIMEOUT = 0.3


def _pkg_path(*parts: str) -> str:
    return str(Path(__file__).resolve().parent.joinpath(*parts))


def _collapse(text: str) -> str:
    return ' '.join(text.split())


def _split_args(raw: str) -> list[str]:
    if not raw:
        return []
    return _ARG_SPLIT_RE.split(raw)


def _resolve_dates(text: str) -> str:
    return _DATE_RE.sub(lambda m: datetime.now().strftime(m.group(1)), text)


def _load_presets(path: str) -> dict:
    """Parse the presets TOML into the /api/presets response shape, re-reading the file on every call."""
    defaults = {
        'error': None,
        'runner': '',
        'default_task': '',
        'common': {'raw': '', 'args': []},
        'common_resolved': [],
        'presets': {},
    }
    p = Path(path)
    if not p.exists():
        return {**defaults, 'error': f'Presets file not found: {p}'}
    try:
        data = tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as e:
        return {**defaults, 'error': f'Failed to parse {p}: {e}'}

    common_raw = _collapse(str(data.get('common', '')))
    common_args = _split_args(common_raw)
    presets = {}
    for name, entry in data.get('presets', {}).items():
        raw = _collapse(str(entry.get('args', '')))
        presets[name] = {'raw': raw, 'args': _split_args(raw)}
    return {
        'error': None,
        'runner': str(data.get('runner', '')),
        'default_task': str(data.get('default_task', '')),
        'common': {'raw': common_raw, 'args': common_args},
        'common_resolved': [_resolve_dates(a) for a in common_args],
        'presets': presets,
    }


class Launcher:
    """Single-slot manager for the running inference subprocess and its logs."""

    def __init__(self, presets_path: str, console_port: int):
        self.presets_path = presets_path
        self.console_port = console_port
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._status = 'idle'
        self._run: dict | None = None
        self._last_run: dict | None = None
        self._log: deque[str] = deque(maxlen=_LOG_BUFFER_LINES)
        self._log_count = 0
        self._console_ready = False
        self._console_checked_at = 0.0

    def _build_command(self, parsed: dict, preset: str, task: str, extra_args: str) -> str:
        parts = [parsed['runner'], parsed['presets'][preset]['raw'], parsed['common']['raw']]
        if task:
            parts.append(f'--driver.task={shlex.quote(task)}')
        if extra_args:
            parts.append(extra_args)
        return ' '.join(p for p in parts if p)

    def start(self, preset: str, task: str, extra_args: str) -> tuple[int, dict]:
        parsed = _load_presets(self.presets_path)
        if parsed['error'] is not None:
            return 400, {'error': parsed['error']}
        if preset not in parsed['presets']:
            return 400, {'error': f'Unknown preset: {preset}'}
        with self._lock:
            if self._proc is not None:
                return 409, {'error': 'A run is already in progress'}
            command = self._build_command(parsed, preset, task, extra_args)
            proc = subprocess.Popen(
                ['bash', '-c', command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._proc = proc
            self._status = 'running'
            self._run = {
                'preset': preset,
                'task': task,
                'command': command,
                'started_at': time.time(),
                'stop_requested_at': None,
            }
            self._log.clear()
            self._log_count = 0
            self._console_ready = False
            self._console_checked_at = 0.0
            threading.Thread(target=self._reader, args=(proc,), daemon=True).start()
        return 200, {'ok': True}

    def stop(self, force: bool) -> tuple[int, dict]:
        with self._lock:
            proc = self._proc
            if proc is None:
                return 409, {'error': 'Nothing is running'}
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL if force else signal.SIGINT)
            self._status = 'stopping'
            if self._run['stop_requested_at'] is None:
                self._run['stop_requested_at'] = time.time()
        return 200, {'ok': True}

    def _reader(self, proc: subprocess.Popen) -> None:
        for line in iter(proc.stdout.readline, ''):
            clean = _ANSI_RE.sub('', line).rstrip('\n')
            with self._lock:
                self._log.append(clean)
                self._log_count += 1
        exit_code = proc.wait()
        with self._lock:
            if self._proc is proc:
                run = self._run
                self._last_run = {
                    'preset': run['preset'],
                    'exit_code': exit_code,
                    'started_at': run['started_at'],
                    'ended_at': time.time(),
                    'stopped': run['stop_requested_at'] is not None,
                }
                self._run = None
                self._proc = None
                self._status = 'idle'

    def _tcp_probe(self) -> bool:
        try:
            with socket.create_connection(('127.0.0.1', self.console_port), timeout=_CONSOLE_PROBE_TIMEOUT):
                return True
        except OSError:
            return False

    def _console_ready_cached(self) -> bool:
        now = time.time()
        with self._lock:
            if now - self._console_checked_at < _CONSOLE_CACHE_SECONDS:
                return self._console_ready
        ready = self._tcp_probe()
        with self._lock:
            self._console_checked_at = time.time()
            self._console_ready = ready
        return ready

    def state(self) -> dict:
        with self._lock:
            run = dict(self._run) if self._run is not None else None
            last_run = dict(self._last_run) if self._last_run is not None else None
            status = self._status
            log_length = self._log_count
        if run is not None:
            run['console_ready'] = self._console_ready_cached()
        return {
            'status': status,
            'run': run,
            'last_run': last_run,
            'console_port': self.console_port,
            'log_length': log_length,
        }

    def get_logs(self, offset: int) -> dict:
        with self._lock:
            count = self._log_count
            buf = list(self._log)
        base = count - len(buf)
        if offset < base:
            return {'next': count, 'lines': buf, 'truncated': True}
        start = min(offset, count) - base
        return {'next': count, 'lines': buf[start:], 'truncated': False}

    def shutdown(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait()


launcher: Launcher | None = None

app = FastAPI()
app.mount('/static', StaticFiles(directory=_pkg_path('static')), name='static')


class StartRequest(BaseModel):
    preset: str
    task: str = ''
    extra_args: str = ''


class StopRequest(BaseModel):
    force: bool = False


@app.get('/')
async def index():
    return FileResponse(_pkg_path('templates', 'launcher.html'))


@app.get('/api/state')
async def api_state():
    return launcher.state()


@app.get('/api/presets')
async def api_presets():
    return _load_presets(launcher.presets_path)


@app.post('/api/start')
async def api_start(req: StartRequest):
    code, body = launcher.start(req.preset, req.task, req.extra_args)
    return body if code == 200 else JSONResponse(status_code=code, content=body)


@app.post('/api/stop')
async def api_stop(req: StopRequest):
    code, body = launcher.stop(req.force)
    return body if code == 200 else JSONResponse(status_code=code, content=body)


@app.get('/api/logs')
async def api_logs(offset: int = 0):
    return launcher.get_logs(offset)


@cfn.config()
def main(presets: str = 'presets.toml', port: int = 8000, console_port: int = 8080, host: str = '0.0.0.0'):
    """Serve the launcher web app.

    Args:
        presets: Path to the presets TOML, resolved against the current working directory.
        port: Port the launcher listens on.
        console_port: Port where the launched process serves its own robot console.
        host: Interface the launcher binds to.
    """
    global launcher
    launcher = Launcher(presets_path=presets, console_port=console_port)
    logging.info(f'Starting launcher on http://{host}:{port} (child console on port {console_port})')
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        launcher.shutdown()


def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
