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
timeline name to observation key; by default ``wall_time`` and ``obs_time``
read from the matching ``*_ns`` observation fields). Those values are read once per
inference at the outermost tap and reused by every inner tap, so all taps stamp the
same inference at the same time and their streams line up. A per-tap ``step``
sequence timeline is always added for ordering within a single tap.

An action chunk is shown two ways. A Cartesian end-effector command becomes one 3D
object (latest-at, so a new chunk replaces the last): a thin ``rr.LineStrips3D`` path,
and at each waypoint the gripper's two fingers as fixed-length bars (``rr.LineStrips3D``)
oriented by its pose, closed into a thin rectangle by two faint span edges. The finger
separation along the jaw axis encodes grip on a fixed scale and the color encodes horizon.
The robot's actual gripper is overlaid the same way in white for predicted-vs-realized.
Every field is *also* logged as ``rr.Scalars`` on a dedicated ``action_time`` timeline
(each action stamped at the inference-request time plus its horizon offset), so a
``TimeSeriesView`` reads commanded values with real axes. That anchor is the pre-inference
``obs_time_ns``, so it precedes the harness's true execution time (stamped after
inference by ``ChunkedSchedule``) by the inference latency. Select ``action_time`` to see them.

Entity paths are ``{tap_name}/{data_key}``. A tap's incoming observation keys and
outgoing action keys share that namespace; in the rare case the same key appears on
both sides, the later write overwrites at that timestamp.
"""

import itertools
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from positronic import geom
from positronic.drivers.roboarm import command as roboarm_command
from positronic.policy.base import DelegatingSession, PolicyWrapper, Session
from positronic.utils.rerun_compat import log_numeric_series, set_timeline_sequence, set_timeline_time

DEFAULT_TIMELINES = {'wall_time': 'wall_time_ns', 'obs_time': 'obs_time_ns'}

# Process-wide episode counter so files stay unique even across concurrent
# ``Recorder`` instances (e.g. one per websocket session on a server).
_EPISODE_COUNTER = itertools.count(1)


def _squeeze_batch(arr: np.ndarray) -> np.ndarray:
    """Remove leading size-1 dims from a potential image array."""
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _as_image(value: Any) -> np.ndarray | None:
    """Return a single RGB frame if *value* looks like an image, else None.

    A temporal stack ``(T, H, W, 3)`` (e.g. from ``TemporalStack``) records its most recent
    frame — the current observation — so the stack does not flood the recording as numeric series.
    """
    if not isinstance(value, np.ndarray):
        return None
    squeezed = _squeeze_batch(value)
    if squeezed.ndim == 4 and squeezed.shape[-1] == 3:
        squeezed = squeezed[-1]
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


# Each waypoint shows the gripper's two fingers as fixed-length bars lying in the gripper's
# local X-Y plane (perpendicular to the finger/approach axis, local Z; on a top-down grasp
# local Z points down so the bars lie parallel to the world XY plane, a footprint seen from
# above). Each bar runs along the local X (finger-depth) axis with fixed length; their
# separation along the local Y jaw axis encodes grip on a fixed absolute scale (grip 0 ->
# wide/open, grip 1 -> narrow/closed). Two span edges (along Y) close the bars into a
# rectangle; bars carry the horizon color. The viewer doesn't honor line alpha (low-alpha
# lines render black), so spans use a faint near-background gray to read as "almost there".
_GRIP_OPEN_HALF = 0.011
_GRIP_CLOSED_HALF = 0.003
_FINGER_HALF_DEPTH = 0.004
_FINGER_RADIUS = 0.0004
_SPAN_RADIUS = 0.00025
_ACTUAL_SCALE = 1.5  # the actual-EE marker reuses the glyph, drawn bolder
_SPAN_RGB = (205, 205, 210)
_POSE_LABELS = ['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz']


def _build_blueprint(
    image_paths: list[str], numeric_paths: list[str], path3d_paths: list[str] = (), series_paths: list[str] = ()
) -> rrb.Blueprint | None:
    categories = [
        (rrb.Spatial2DView, image_paths),
        (rrb.TimeSeriesView, numeric_paths),
        (rrb.Spatial3DView, path3d_paths),
        (rrb.TimeSeriesView, series_paths),
    ]
    grids = [
        rrb.Grid(*[view(name=p.rsplit('/', 1)[-1], origin=p) for p in dict.fromkeys(paths)])
        for view, paths in categories
        if paths
    ]
    return rrb.Blueprint(rrb.Grid(*grids)) if grids else None


def _command_field_arrays(key: str, commands: list, horizons: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Stack robot commands grouped by type, each group's fields and horizons as arrays."""
    groups: dict[str, list] = {}
    for cmd, h in zip(commands, horizons, strict=True):
        groups.setdefault(cmd.TYPE, []).append((cmd, h))
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for type_name, group in groups.items():
        wires = [roboarm_command.to_wire(c) for c, _ in group]
        group_horizon = np.array([h for _, h in group], dtype=np.float64)
        for field in (k for k in wires[0] if k != 'type'):
            arr = _stack_numeric([w[field] for w in wires])
            if arr is not None:
                out.append((f'{key}/{type_name}/{field}', arr, group_horizon))
    return out


def _horizon(actions: list[dict]) -> np.ndarray:
    """Relative chunk time (seconds) used as the curve x-axis, falling back to action index."""
    if actions and all('timestamp' in a for a in actions):
        ts = _stack_numeric([a['timestamp'] for a in actions])
        if ts is not None and ts.ndim == 1:
            return ts.astype(np.float64)
    return np.arange(len(actions), dtype=np.float64)


def _horizon_colors(horizon: np.ndarray) -> np.ndarray:
    """Per-waypoint color along a near->far gradient (warm = soon, cool = late)."""
    h = np.asarray(horizon, dtype=np.float64)
    rng = float(h.max() - h.min()) if h.size else 0.0
    t = (h - h.min()) / rng if rng > 0 else np.zeros(h.shape)
    near = np.array([255, 224, 64], dtype=np.float64)
    far = np.array([128, 48, 200], dtype=np.float64)
    return (near[None, :] * (1 - t)[:, None] + far[None, :] * t[:, None]).astype(np.uint8)


def _jaw_half(grip: np.ndarray | None, n: int) -> np.ndarray:
    """Per-waypoint half jaw-width from grip on a fixed absolute scale (0 -> open, 1 -> closed)."""
    if grip is None:
        return np.full(n, 0.5 * (_GRIP_OPEN_HALF + _GRIP_CLOSED_HALF))
    g = np.clip(np.asarray(grip, dtype=np.float64).reshape(-1), 0.0, 1.0)
    return _GRIP_OPEN_HALF - g * (_GRIP_OPEN_HALF - _GRIP_CLOSED_HALF)


def _gripper_rects(
    centers: np.ndarray, rotations: np.ndarray, grip: np.ndarray | None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per waypoint, the gripper rectangle split into finger bars and span edges.

    Returns ``(finger_strips, span_strips)``: two-point world-frame segments. Finger bars run
    along the local X (finger-depth) axis with fixed length; span edges run along the local Y
    jaw axis and close the rectangle, so their length (the finger separation) encodes grip on
    a fixed absolute scale (grip 0 -> wide/open, grip 1 -> narrow/closed). Two of each per
    waypoint.
    """
    n = len(centers)
    jaw = _jaw_half(grip, n)
    finger_strips: list[np.ndarray] = []
    span_strips: list[np.ndarray] = []
    for i in range(n):
        w, d = jaw[i], _FINGER_HALF_DEPTH
        corners = np.array([[-d, -w, 0.0], [d, -w, 0.0], [d, w, 0.0], [-d, w, 0.0]])
        a, b, c, e = centers[i] + corners @ rotations[i].T
        finger_strips.append(np.array([a, b]))  # -Y finger, runs along X
        finger_strips.append(np.array([e, c]))  # +Y finger, runs along X
        span_strips.append(np.array([b, c]))  # +X span, runs along Y
        span_strips.append(np.array([a, e]))  # -X span, runs along Y
    return finger_strips, span_strips


def _log_action_series(path: str, arr: np.ndarray, horizon: np.ndarray, base_ns: int, names: list[str] | None) -> None:
    """Overlay a chunk on the ``action_time`` timeline as a named multi-line time series.

    Each action is stamped at ``base_ns + horizon_i``, where ``base_ns`` is the
    inference-request time; successive chunks lay out along one clock so a ``TimeSeriesView``
    has real axes. This precedes true execution by the inference latency: the harness's
    ``ChunkedSchedule`` anchors commands at ``clock.now()`` *after* inference, which a recorder
    tap sitting inside it cannot observe.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if names:
        rr.log(path, rr.SeriesLines(names=names), static=True)
    for i in range(arr.shape[0]):
        set_timeline_time('action_time', base_ns + int(round(float(horizon[i]) * 1e9)))
        rr.log(path, rr.Scalars(arr[i]))
    # ``action_time`` is owned by this series; clear it so other entities aren't stamped on it.
    rr.disable_timeline('action_time')


def _log_ee_pose_chunk(
    path: str,
    commands: list,
    horizon: np.ndarray,
    grip: np.ndarray | None,
    actual_pos: np.ndarray | None,
    actual_grip: float | None,
) -> np.ndarray:
    """Log a Cartesian end-effector chunk as one 3D object.

    The predicted path is a thin line; each waypoint draws the gripper's two finger bars
    oriented by its pose (finger separation = grip, color = horizon), faintly closed into a
    rectangle. The robot's actual gripper (pose + grip from the obs) is overlaid in white.
    """
    translations = np.array([c.pose.translation for c in commands], dtype=np.float64)
    rotations = np.array([c.pose.rotation.as_rotation_matrix for c in commands], dtype=np.float64)
    quats_wxyz = np.array([c.pose.rotation.as_quat for c in commands], dtype=np.float64)

    rr.log(f'{path}/trajectory/path', rr.LineStrips3D([translations], radii=0.0012, colors=[120, 120, 120]))

    finger_strips, span_strips = _gripper_rects(translations, rotations, grip)
    finger_colors = np.repeat(_horizon_colors(horizon), 2, axis=0)  # (2n, 3); two fingers per waypoint
    span_colors = np.tile(np.array(_SPAN_RGB, np.uint8), (len(span_strips), 1))
    radii = np.array([_FINGER_RADIUS] * len(finger_strips) + [_SPAN_RADIUS] * len(span_strips))
    rr.log(
        f'{path}/trajectory/grippers',
        rr.LineStrips3D(finger_strips + span_strips, colors=np.concatenate([finger_colors, span_colors]), radii=radii),
    )

    if actual_pos is not None:
        ee = np.asarray(actual_pos, dtype=np.float64)
        if ee.size >= 7:
            rot = geom.Rotation.from_quat(ee[3:7]).as_rotation_matrix
            grip_a = np.array([actual_grip]) if actual_grip is not None else None
            fingers_a, spans_a = _gripper_rects(ee[:3][None, :], rot[None, :, :], grip_a)
            colors_a = [[245, 245, 245]] * len(fingers_a) + [list(_SPAN_RGB)] * len(spans_a)
            radii_a = [_FINGER_RADIUS * _ACTUAL_SCALE] * len(fingers_a) + [_SPAN_RADIUS * _ACTUAL_SCALE] * len(spans_a)
            rr.log(f'{path}/trajectory/actual', rr.LineStrips3D(fingers_a + spans_a, colors=colors_a, radii=radii_a))

    return np.concatenate([translations, quats_wxyz], axis=1)


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
    stamps the inference identically. ``blueprint``, if given, is sent as the recording's
    layout instead of the auto-built one.
    """

    def __init__(
        self, recording_dir: str | Path, timelines: dict[str, str] | None = None, blueprint: rrb.Blueprint | None = None
    ):
        self._dir = Path(recording_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._timelines = dict(timelines) if timelines is not None else dict(DEFAULT_TIMELINES)
        self._blueprint = blueprint
        self._stream: rr.RecordingStream | None = None
        self._live = 0
        self._depth = 0
        self._timeline_values: dict[str, Any] = {}
        self._image_paths: list[str] = []
        self._numeric_paths: list[str] = []
        self._path3d_paths: list[str] = []
        self._series_paths: list[str] = []

    def tap(self, name: str) -> '_RecordingTap':
        return _RecordingTap(self, name)

    @property
    def stream(self) -> rr.RecordingStream | None:
        """The active recording stream, for logging supplementary panels into the current ``.rrd``."""
        return self._stream

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

    def wrap_session(self, inner: Session, context, now) -> Session:
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

    def _log_action_chunk(self, prefix: str, actions: list[dict], obs: dict) -> None:
        """Log the action chunk as an enriched 3D trajectory + ``action_time`` time series."""
        # Drop keyless validity sentinels (timestamp-only entries a time codec appends): they carry
        # no command to plot and would flip the ``all(... in a)`` command checks below.
        actions = [a for a in actions if a.keys() - {'timestamp'}]
        horizon = _horizon(actions)
        tv = self._rec._timeline_values
        base_ns = int(tv.get('obs_time', tv.get('wall_time', next(iter(tv.values()), 0))))
        grip = _stack_numeric([a['target_grip'] for a in actions]) if all('target_grip' in a for a in actions) else None
        actual_pos = obs.get('robot_state.ee_pose') if isinstance(obs, Mapping) else None
        actual_grip = obs.get('grip') if isinstance(obs, Mapping) else None
        # Under TemporalStack these arrive as (T, 7) / (T,) stacks; the overlay draws the current pose,
        # which is the last frame (offsets end at 0 = now), mirroring the image collapse in `_as_image`.
        if actual_pos is not None:
            actual_pos = np.asarray(actual_pos)
            if actual_pos.ndim == 2:
                actual_pos = actual_pos[-1]
        actual_g = float(np.asarray(actual_grip).reshape(-1)[-1]) if actual_grip is not None else None
        keys: list[str] = []
        for action in actions:
            for key in action:
                if key not in keys and key != 'timestamp':
                    keys.append(key)
        for key in keys:
            idx = [i for i, a in enumerate(actions) if key in a]
            values = [actions[i][key] for i in idx]
            h = horizon[idx]
            path = f'{prefix}/{key}'
            series_path = f'{prefix}/series/{key}'
            if all(isinstance(v, roboarm_command.CartesianPosition) for v in values):
                grip_sub = grip[idx] if grip is not None else None
                pose = _log_ee_pose_chunk(path, values, h, grip_sub, actual_pos, actual_g)
                _log_action_series(series_path, pose, h, base_ns, names=_POSE_LABELS)
                self._rec._path3d_paths.append(f'{path}/trajectory')
                self._rec._series_paths.append(series_path)
            elif all(isinstance(v, roboarm_command.CommandType) for v in values):
                for suffix, arr, group_h in _command_field_arrays(key, values, h):
                    _log_action_series(f'{prefix}/series/{suffix}', arr, group_h, base_ns, names=None)
                    self._rec._series_paths.append(f'{prefix}/series/{suffix}')
            elif (arr := _stack_numeric(values)) is not None:
                _log_action_series(series_path, arr, h, base_ns, names=[key])
                self._rec._series_paths.append(series_path)

    def _send_blueprint(self) -> None:
        rec = self._rec
        bp = rec._blueprint or _build_blueprint(
            rec._image_paths, rec._numeric_paths, rec._path3d_paths, rec._series_paths
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
                    self._log_action_chunk(self._name, actions, obs)
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
