"""Replay one recorded observation through a live inference endpoint and save an ``.rrd``.

Point this at a recorded episode and a moment in it; the observation at that moment is
sent to a remote policy endpoint and the commands it plays back are written to a rerun
recording, with the predicted end-effector trajectory overlaid on the robot's actual
pose at that moment. Open the ``.rrd`` to see whether the predicted chunk descends
toward the object or rises away.

The recording is named after the served model (from the server's handshake metadata, or
``--label``) and that metadata is logged as a ``meta`` panel, so several probes load into
one rerun viewer as distinguishable, self-describing recordings.

Usage::

    uv run --locked positronic-probe \\
        --dataset.path=<episode-or-dataset> --episode=0 --at=3.0 \\
        --policy=.remote --policy.url=<host>:<port> \\
        --task='Pick ...' --output_dir=./probe_recordings
"""

import re
import time

import configuronic as cfn
import numpy as np
import pos3
import rerun as rr
import rerun.blueprint as rrb

import positronic.cfg.ds
import positronic.cfg.policy as policy_cfg
from pimm.logging import init_logging
from positronic import keys
from positronic.dataset.dataset import Dataset
from positronic.drivers.roboarm.command import CartesianPosition, JointDelta
from positronic.policy import Policy, Recorder
from positronic.policy.executor import Executor

# Tap name; the recorder logs each obs/action entity under ``{_TAP}/{key}`` (see recording.py).
_TAP = 'raw'
# Observation keys the endpoint expects, mirroring the inference harness, plus every image.*.
_STATE_KEYS = (keys.JOINTS, keys.JOINT_VEL, keys.EE_POSE, keys.GRIP)


def _build_wire_obs(sample: dict, task: str | None, now_ns: int, recorded_ts: int) -> dict:
    obs = {k: sample[k] for k in _STATE_KEYS if k in sample}
    obs.update({k: v for k, v in sample.items() if k.startswith(keys.IMAGE_PREFIX)})
    if task:
        obs[keys.TASK] = task
    obs[keys.WALL_TIME_NS] = now_ns  # rerun wall_time timeline
    obs[keys.OBS_TIME_NS] = recorded_ts  # rerun obs_time + action_time anchor
    return obs


def _recording_name(meta: dict) -> str:
    """A short recording name from server metadata, e.g. ``groot@110000`` / ``gyros@18500``."""
    server_type = meta.get('server.type', 'model')
    ckpt = meta.get('server.checkpoint_id')
    if not ckpt:
        path = str(meta.get('server.checkpoint_path', ''))
        match = re.search(r'step_count=0*(\d+)', path)
        ckpt = match.group(1) if match else (path.rstrip('/').rsplit('/', 1)[-1] or None)
    return f'{server_type}@{ckpt}' if ckpt else str(server_type)


def _meta_doc(name: str, meta: dict) -> str:
    rows = '\n'.join(f'- **{k.removeprefix("server.")}**: {v}' for k, v in meta.items() if k.startswith('server.'))
    return f'## {name}\n\n{rows}'


def _play(session, obs: dict, runtime: Executor) -> list[tuple[int, dict]]:
    """Every command the session emits for ``obs``, from the endpoint's answer to the end of that chunk.

    The observation is one frozen frame, so the walk moves the clock rather than the world. The first call
    asks the endpoint and the wait is what that round trip takes. The session anchors the chunk on the call
    that receives it and asks for a call at each waypoint, so the walk follows the instants it names. An
    endpoint that commands nothing gives an empty list.
    """
    session(obs, time.time_ns())
    # The endpoint's own ``infer_timeout`` bounds the wait, and a round trip that fails raises out of the
    # call below rather than here.
    runtime.wait()
    played: list[tuple[int, dict]] = []
    now_ns = time.time_ns()
    while True:
        commands, resume_at_ns = session(obs, now_ns)
        if commands:
            played.append((now_ns, dict(commands)))
        if runtime.owes_an_answer:  # the chunk has run out, and the session is asking the endpoint again
            return played
        now_ns = resume_at_ns


def _is_cartesian_chunk(played: list[tuple[int, dict]]) -> bool:
    """Whether every command is a Cartesian end-effector pose (so a 3D trajectory exists)."""
    return bool(played) and all(isinstance(c.get(keys.ROBOT_COMMAND), CartesianPosition) for _, c in played)


def _log_commands(played: list[tuple[int, dict]], obs: dict, wall_ns: int, inf_ns: int) -> None:
    """Log the commanded fields as one named time-series on the obs's live timelines, and the predicted
    end-effector path in 3D beside the pose the robot was actually at.

    Plots EE pose fields for a Cartesian chunk or joint velocities for a DROID chunk. A rerun time-series
    view plots only the active timeline, and the images live on ``wall_time`` — so each command is stamped
    on ``wall_time`` / ``obs_time`` offset by its horizon, with a relative ``chunk_time`` axis alongside.
    """
    if not played:
        return
    commands = [c for _, c in played]
    if _is_cartesian_chunk(played):
        poses = [c[keys.ROBOT_COMMAND].pose for c in commands]
        labels = ['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz']
        rows = [[*p.translation, *p.rotation.as_quat] for p in poses]
        path = np.array([p.translation for p in poses], dtype=np.float64)
        rr.log('trajectory/path', rr.LineStrips3D([path], radii=0.0012, colors=[120, 120, 120]), static=True)
        actual = np.asarray(obs[keys.EE_POSE], dtype=np.float64).reshape(-1)[:3]
        rr.log('trajectory/actual', rr.Points3D([actual], radii=0.006, colors=[245, 245, 245]), static=True)
    elif all(isinstance(c.get(keys.ROBOT_COMMAND), JointDelta) for c in commands):
        deltas = [c[keys.ROBOT_COMMAND].velocities for c in commands]
        labels = [f'dq{i}' for i in range(len(deltas[0]))]
        rows = [list(d) for d in deltas]
    else:
        return
    horizon = np.array([(at_ns - played[0][0]) / 1e9 for at_ns, _ in played])
    if all(keys.TARGET_GRIP in c for c in commands):
        labels.append(keys.TARGET_GRIP)
        rows = [row + [c[keys.TARGET_GRIP]] for row, c in zip(rows, commands, strict=True)]
    data = np.array(rows, float)

    rr.log('commands', rr.SeriesLines(names=labels), static=True)
    for i, h in enumerate(horizon):
        h_ns = int(round(h * 1e9))
        rr.set_time('wall_time', timestamp=np.datetime64(wall_ns + h_ns, 'ns'))
        rr.set_time('obs_time', timestamp=np.datetime64(inf_ns + h_ns, 'ns'))
        rr.set_time('chunk_time', duration=float(h))
        rr.log('commands', rr.Scalars(data[i]))


def _blueprint(image_keys: list[str], has_trajectory: bool) -> rrb.Blueprint:
    """Images + server meta on top; the commands time-series below, with the 3D trajectory beside it
    only for a Cartesian chunk (a velocity chunk has none, so the view is omitted)."""
    images = [rrb.Spatial2DView(origin=f'{_TAP}/{key}', name=key) for key in image_keys]
    top = rrb.Horizontal(*images, rrb.TextDocumentView(origin='meta', name='server'))
    commands = rrb.TimeSeriesView(origin='commands', name='commands')
    trajectory = rrb.Spatial3DView(origin='trajectory', name='trajectory')
    bottom = rrb.Horizontal(trajectory, commands) if has_trajectory else commands
    return rrb.Blueprint(rrb.Vertical(top, bottom))


@cfn.config(dataset=positronic.cfg.ds.local, policy=policy_cfg.remote, episode=0, at=0.0, task=None, label=None)
def main(
    dataset: Dataset, policy: Policy, episode: int, at: float, task: str | None, label: str | None, output_dir: str
):
    ep = dataset[episode]
    ts = int(np.clip(ep.start_ts + int(at * 1e9), ep.start_ts, ep.last_ts))
    sample = ep.time[ts]
    if keys.EE_POSE not in sample:
        raise ValueError('episode has no robot_state.ee_pose; cannot overlay actual pose')
    task = task or ep.static.get(keys.TASK)

    now_ns = time.time_ns()
    obs = _build_wire_obs(sample, task, now_ns, ts)
    image_keys = [k for k in obs if k.startswith(keys.IMAGE_PREFIX)]

    rec = Recorder(pos3.sync(output_dir))
    tapped = rec.tap(_TAP).wrap(policy)
    with tapped.episode({keys.TASK: task} if task else None) as fns:
        runtime = Executor(fns)
        session = tapped.new_session(runtime)
        meta = dict(tapped.meta)
        name = label or _recording_name(meta)
        try:
            played = _play(session, obs, runtime)
        finally:
            # The call that drained the chunk asked the endpoint again; closing the runtime waits that
            # answer out. It goes before the episode, whose socket the call still holds.
            runtime.close()
            session.close()

    print(f'episode {episode} @ {at:.3f}s (ts={ts}) [{name}]: {len(played)} command(s); rrd -> {output_dir}')
    stream = rec.stream
    assert stream is not None, 'the tap opened the recording when the session was made'
    with stream:
        stream.send_recording_name(name)
        rr.log('meta', rr.TextDocument(_meta_doc(name, meta), media_type=rr.MediaType.MARKDOWN), static=True)
        _log_commands(played, obs, now_ns, ts)
        # Sent here, not at Recorder construction: the layout depends on the chunk type, which is
        # only known after inference — a velocity chunk drops the 3D trajectory view.
        rr.send_blueprint(_blueprint(image_keys, _is_cartesian_chunk(played)))


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
