"""Record inference observations and actions to rerun ``.rrd`` files.

Recordings are written with `rerun <https://rerun.io>`_, a logging/visualization
tool. Each episode becomes one ``.rrd`` file in the recording directory; open it in
the rerun viewer to inspect what flowed through the policy.

A :class:`Recorder` hands out lightweight ``tap(name)`` wrappers. A tap inserted
into a policy pipeline logs the observation passing *down* through it and the action
chunk coming back *up*, under entity paths prefixed by its ``name``. Placing two
taps at different points captures both ends of a remote inference round-trip in one
correlated recording::

    rec = Recorder(recording_dir)
    pipeline = rec.tap('raw') | codec | rec.tap('server')
    policy = pipeline.wrap(remote_policy)

- the ``raw`` tap (outermost) logs the observation as received and the final action
  chunk;
- the ``server`` tap (innermost, next to the remote policy) logs the observation as
  sent to the server and the chunk as received back.

Each entity is stamped on the timelines given by ``timelines`` (a mapping of rerun
timeline name to observation key; by default ``wall_time`` and ``inference_time``
read from the matching ``*_ns`` observation fields). Those values are read once per
inference at the outermost tap and reused by every inner tap, so all taps stamp the
same inference at the same time and their streams line up. A per-tap ``step``
sequence timeline is always added for ordering within a single tap.

An action chunk is logged structure-of-arrays at the single inference timestamp:
each field is stacked across the chunk into one ``rr.Tensor``. Robot commands are
encoded to plain arrays and grouped by command type so each tensor is homogeneous.

Entity paths are ``{tap_name}/{data_key}``. A tap's incoming observation keys and
outgoing action keys share that namespace; in the rare case the same key appears on
both sides, the later write overwrites at that timestamp.
"""

import itertools
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from positronic.drivers.roboarm import command as roboarm_command
from positronic.policy.base import DelegatingSession, PolicyWrapper, Session
from positronic.utils.rerun_compat import log_numeric_series, set_timeline_sequence, set_timeline_time

DEFAULT_TIMELINES = {'wall_time': 'wall_time_ns', 'inference_time': 'inference_time_ns'}

# Process-wide episode counter so files stay unique even across concurrent
# ``Recorder`` instances (e.g. one per websocket session on a server).
_EPISODE_COUNTER = itertools.count(1)


def _squeeze_batch(arr: np.ndarray) -> np.ndarray:
    """Remove leading size-1 dims from a potential image array."""
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _as_image(value: Any) -> np.ndarray | None:
    """Return squeezed RGB array if *value* looks like an image, else None."""
    if not isinstance(value, np.ndarray):
        return None
    squeezed = _squeeze_batch(value)
    if squeezed.ndim == 3 and squeezed.shape[-1] == 3:
        return squeezed
    return None


def _as_numeric(value: Any) -> Any | None:
    """Return a loggable numeric form of *value*, or None if not numeric."""
    if isinstance(value, np.ndarray | int | float | np.integer | np.floating):
        return value
    if isinstance(value, list | tuple):
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(np.float64)
    return None


def _stack_numeric(values: list) -> np.ndarray | None:
    """Stack a per-action field into one numeric array, or None if not stackable."""
    try:
        arr = np.array(values)
    except (TypeError, ValueError):
        return None
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        return None
    return arr


def _build_blueprint(image_paths: list[str], numeric_paths: list[str]) -> rrb.Blueprint | None:
    if not image_paths and not numeric_paths:
        return None
    image_views = [rrb.Spatial2DView(name=p.rsplit('/', 1)[-1], origin=p) for p in image_paths]
    numeric_views = [rrb.TimeSeriesView(name=p.rsplit('/', 1)[-1], origin=p) for p in numeric_paths]
    grid_items: list[Any] = []
    if image_views:
        grid_items.append(rrb.Grid(*image_views))
    if numeric_views:
        grid_items.append(rrb.Grid(*numeric_views))
    return rrb.Blueprint(rrb.Grid(*grid_items))


def _command_field_arrays(key: str, commands: list) -> list[tuple[str, np.ndarray]]:
    """Stack robot commands grouped by type, each group's fields as homogeneous arrays."""
    groups: dict[str, list] = {}
    for cmd in commands:
        groups.setdefault(cmd.TYPE, []).append(cmd)
    out: list[tuple[str, np.ndarray]] = []
    for type_name, group in groups.items():
        wires = [roboarm_command.to_wire(c) for c in group]
        for field in (k for k in wires[0] if k != 'type'):
            arr = _stack_numeric([w[field] for w in wires])
            if arr is not None:
                out.append((f'{key}/{type_name}/{field}', arr))
    return out


def action_chunk_arrays(actions: list[dict]) -> list[tuple[str, np.ndarray]]:
    """Turn an action chunk into structure-of-arrays: one ``(path_suffix, array)`` per field.

    Each field is stacked across the chunk into a single array. Robot commands are
    encoded to plain arrays and grouped by command type so each array is homogeneous.
    The per-action ``timestamp`` (relative seconds at this point) is stored as one
    int64 array in nanoseconds, matching the canonical time unit used elsewhere.
    """
    keys: list[str] = []
    for action in actions:
        for key in action:
            if key not in keys:
                keys.append(key)
    out: list[tuple[str, np.ndarray]] = []
    for key in keys:
        values = [a[key] for a in actions if key in a]
        if values and all(isinstance(v, roboarm_command.CommandType) for v in values):
            out.extend(_command_field_arrays(key, values))
            continue
        arr = _stack_numeric(values)
        if arr is None:
            continue
        if key == 'timestamp':
            arr = np.rint(arr * 1e9).astype(np.int64)
        out.append((key, arr))
    return out


def _log_action_chunk(prefix: str, actions: list[dict]) -> None:
    """Log an action chunk as structure-of-arrays at the current timestamp."""
    for suffix, arr in action_chunk_arrays(actions):
        rr.log(f'{prefix}/{suffix}', rr.Tensor(arr))


class Recorder:
    """Writes one rerun ``.rrd`` file per episode and hands out ``tap(name)`` wrappers.

    Taps share the recorder's current episode stream, so taps placed at different
    points in one pipeline write to the same recording. The episode boundary is
    tracked by a live-session counter: the first tap session to start (when none are
    active) opens a fresh ``.rrd``; later taps in the same episode write to it; each
    ``close()`` decrements, and the next session opened after the count returns to
    zero starts the next file. Episodes are assumed to run one at a time (sessions on
    one recorder do not overlap).

    ``timelines`` maps rerun timeline names to observation keys. The values are read
    once per inference at the outermost tap and reused by inner taps so every tap
    stamps the inference identically.
    """

    def __init__(self, recording_dir: str | Path, timelines: dict[str, str] | None = None):
        self._dir = Path(recording_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._timelines = dict(timelines) if timelines is not None else dict(DEFAULT_TIMELINES)
        self._stream: rr.RecordingStream | None = None
        self._live = 0
        self._depth = 0
        self._timeline_values: dict[str, Any] = {}
        self._image_paths: list[str] = []
        self._numeric_paths: list[str] = []

    def tap(self, name: str) -> '_RecordingTap':
        return _RecordingTap(self, name)

    def _open_stream(self) -> rr.RecordingStream:
        if self._live == 0:
            episode_num = next(_EPISODE_COUNTER)
            ts = datetime.now().strftime('%y%m%d_%H%M%S')
            self._stream = rr.RecordingStream(application_id='positronic_inference')
            self._stream.save(str(self._dir / f'{ts}_{episode_num:04d}.rrd'))
        self._live += 1
        return self._stream

    def _release_stream(self) -> None:
        self._live -= 1


class _RecordingTap(PolicyWrapper):
    """A named tap. Wraps a single session to log its observations and actions."""

    def __init__(self, rec: Recorder, name: str):
        self._rec = rec
        self._name = name

    def wrap_session(self, inner: Session, context) -> Session:
        stream = self._rec._open_stream()
        return _RecordingTapSession(inner, self._rec, self._name, stream)


class _RecordingTapSession(DelegatingSession):
    """Logs the observation flowing down and the action chunk flowing up at one point."""

    def __init__(self, inner: Session, rec: Recorder, name: str, stream: rr.RecordingStream):
        super().__init__(inner)
        self._rec = rec
        self._name = name
        self._stream = stream
        self._step = 0

    def _set_timelines(self) -> None:
        for timeline, value in self._rec._timeline_values.items():
            set_timeline_time(timeline, value)
        set_timeline_sequence('step', self._step)

    def _log(self, prefix: str, data: dict) -> None:
        """Recursively log obs *data* under *prefix*, recording entity paths on the Recorder."""
        for key, value in data.items():
            if key.endswith('_time_ns') or isinstance(value, str):
                continue
            path = f'{prefix}/{key}'
            if isinstance(value, dict):
                self._log(path, value)
            elif (img := _as_image(value)) is not None:
                rr.log(path, rr.Image(img).compress())
                self._rec._image_paths.append(path)
            elif (num := _as_numeric(value)) is not None:
                log_numeric_series(path, num)
                self._rec._numeric_paths.append(path)

    def _send_blueprint(self) -> None:
        bp = _build_blueprint(
            list(dict.fromkeys(self._rec._image_paths)), list(dict.fromkeys(self._rec._numeric_paths))
        )
        if bp is not None:
            rr.send_blueprint(bp)

    def __call__(self, obs):
        rec = self._rec
        outermost = rec._depth == 0
        if outermost:
            rec._timeline_values = {t: obs[k] for t, k in rec._timelines.items() if k in obs}
        rec._depth += 1
        try:
            with self._stream:
                self._set_timelines()
                self._log(self._name, obs)

            actions = self._inner(obs)

            if actions is not None:
                with self._stream:
                    self._set_timelines()
                    _log_action_chunk(self._name, actions)
            # Send a combined blueprint (all taps' paths) once, from the outermost
            # tap, after inner taps have logged their first obs.
            if outermost and self._step == 0:
                with self._stream:
                    self._send_blueprint()
            self._step += 1
            return actions
        finally:
            rec._depth -= 1
            if rec._depth == 0:
                rec._timeline_values = {}

    def close(self):
        super().close()
        self._rec._release_stream()
