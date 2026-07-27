"""Positronic-free wall-clock span writer for the env-server process.

The env server runs in a benchmark's isolated interpreter (RoboLab's Isaac venv) where positronic — and so
``positronic.telemetry`` — cannot be imported. This module is stdlib-only, copied in alongside the dumb
``server``/``protocol``, and writes the same OTLP/JSON-lines shape ``positronic.telemetry.read_spans`` parses,
so the env process's spans land in its own ``env.spans.jsonl`` sidecar next to the harness's file.

It is synchronous (one span per line, flushed per line) — the server is single-client lockstep, one request in
flight — so no batch thread and no per-tick allocation churn. ``span`` nests through a module span stack: the
``env.step``/``env.reset`` span is active while the sim's physics/render calls open their own spans under it.
Unbound (no telemetry env vars), every helper is inert.
"""

import json
import os
import platform
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_SCOPE = 'positronic'
_ENV_DIR = 'POSITRONIC_ENV_TELEMETRY_DIR'
_ENV_RUN_ID = 'POSITRONIC_RUN_ID'

_lock = threading.Lock()
_file = None
_trace_id = ''
_resource_attrs: list[dict[str, Any]] = []
_stack: list[str] = []  # active span ids (hex), innermost last — a new span parents to the top


def active() -> bool:
    return _file is not None


def _otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {'boolValue': value}
    if isinstance(value, int):
        return {'intValue': str(value)}
    if isinstance(value, float):
        return {'doubleValue': value}
    if isinstance(value, str):
        return {'stringValue': value}
    if isinstance(value, (list, tuple)):
        return {'arrayValue': {'values': [_otlp_value(item) for item in value]}}
    return {'stringValue': json.dumps(value)}


def _otlp_attributes(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{'key': key, 'value': _otlp_value(value)} for key, value in attrs.items()]


@contextmanager
def bind(telemetry_dir: Path | str, run_id: str):
    """Open ``env.spans.jsonl`` under ``telemetry_dir`` and write ``env.meta.json``; a second bind is a no-op
    so a nested/duplicate call cannot reopen the file."""
    global _file, _trace_id, _resource_attrs
    if _file is not None:
        yield
        return
    directory = Path(telemetry_dir)
    directory.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    meta = {
        'schema': 1,
        'run_id': run_id,
        'host': host,
        'process': 'env',
        'pid': os.getpid(),
        'started_wall_ns': time.time_ns(),
        'started_mono_ns': time.monotonic_ns(),
        'python': platform.python_version(),
        'platform': platform.platform(),
    }
    (directory / 'env.meta.json').write_text(json.dumps(meta, indent=2))
    _trace_id = os.urandom(16).hex()
    _resource_attrs = _otlp_attributes({'run.id': run_id, 'process.name': 'env', 'host.name': host})
    _file = open(directory / 'env.spans.jsonl', 'a')
    try:
        yield
    finally:
        with _lock:
            _file.close()
        _file = None
        _stack.clear()


def bind_from_env():
    """Bind from the telemetry env vars the parent sets under ``--timing`` (the launcher forwards the whole
    environment to this subprocess); an inert context manager when they are absent."""
    directory = os.environ.get(_ENV_DIR)
    run_id = os.environ.get(_ENV_RUN_ID)
    if directory is None or run_id is None:
        return _inert()
    return bind(directory, run_id)


@contextmanager
def _inert():
    yield


@contextmanager
def span(name: str, **attrs: Any):
    """Time the enclosed block as a span parented to the span currently on the stack. Inert while unbound."""
    if _file is None:
        yield
        return
    span_id = os.urandom(8).hex()
    parent_id = _stack[-1] if _stack else None
    start_ns = time.time_ns()
    _stack.append(span_id)
    try:
        yield
    finally:
        _stack.pop()
        _write_span(name, span_id, parent_id, start_ns, time.time_ns(), attrs)


def _write_span(name: str, span_id: str, parent_id: str | None, start_ns: int, end_ns: int, attrs: dict[str, Any]):
    encoded: dict[str, Any] = {
        'traceId': _trace_id,
        'spanId': span_id,
        'name': name,
        'startTimeUnixNano': str(start_ns),
        'endTimeUnixNano': str(end_ns),
        'attributes': _otlp_attributes(attrs),
    }
    if parent_id is not None:
        encoded['parentSpanId'] = parent_id
    doc = {
        'resourceSpans': [
            {
                'resource': {'attributes': _resource_attrs},
                'scopeSpans': [{'scope': {'name': _SCOPE}, 'spans': [encoded]}],
            }
        ]
    }
    line = json.dumps(doc, separators=(',', ':')) + '\n'
    with _lock:
        if _file is not None:
            _file.write(line)
            _file.flush()
