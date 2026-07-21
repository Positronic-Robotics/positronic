"""Replay one recorded observation through a live inference endpoint and save an ``.rrd``.

Point this at a recorded episode and a moment in it; the observation at that moment is
sent to a remote policy endpoint and the returned action chunk is written to a rerun
recording, with the predicted end-effector trajectory overlaid on the robot's actual
pose at that moment. Open the ``.rrd`` to see whether the predicted chunk descends
toward the object or rises away.

The recording is named after the served model (from the server's handshake metadata, or
``--label``) and that metadata is logged as a ``meta`` panel, so several probes load into
one rerun viewer as distinguishable, self-describing recordings.

Usage::

    uv run --locked positronic-probe \\
        --dataset.path=<episode-or-dataset> --episode=0 --at=3.0 \\
        --policy=.remote --policy.host=<host> --policy.port=<port> \\
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
from positronic.dataset.dataset import Dataset
from positronic.drivers.roboarm.command import CartesianPosition, JointDelta
from positronic.policy import Policy, Recorder, is_action
from positronic.utils.logging import init_logging

# Tap name; the recorder logs each obs/action entity under ``{_TAP}/{key}`` (see recording.py).
_TAP = 'raw'
# Observation keys the endpoint expects, mirroring the inference harness, plus every image.*.
_STATE_KEYS = ('robot_state.q', 'robot_state.dq', 'robot_state.ee_pose', 'grip')


def _build_wire_obs(sample: dict, task: str | None, now_ns: int, recorded_ts: int) -> dict:
    obs = {k: sample[k] for k in _STATE_KEYS if k in sample}
    obs.update({k: v for k, v in sample.items() if k.startswith('image.')})
    if task:
        obs['task'] = task
    obs['wall_time_ns'] = now_ns  # rerun wall_time timeline
    obs['obs_time_ns'] = recorded_ts  # rerun obs_time + action_time anchor
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


def _is_cartesian_chunk(actions: list[dict] | None) -> bool:
    """Whether every action carries a Cartesian end-effector command (so a 3D trajectory exists)."""
    return bool(actions) and all(isinstance(a.get('robot_command'), CartesianPosition) for a in actions)


def _log_commands(actions: list[dict], wall_ns: int, inf_ns: int) -> None:
    """Log the chunk's per-step fields as one named time-series on the obs's live timelines.

    Plots EE pose fields for a Cartesian chunk or joint velocities for a DROID chunk. The tap
    already logs this on ``action_time``, but a rerun time-series view plots only the active
    timeline, and the images live on ``wall_time`` — so to read the chunk on the same timeline as
    the scene we re-stamp each waypoint on ``wall_time`` / ``obs_time`` (offset by its horizon),
    with a relative ``chunk_time`` axis alongside.
    """
    if _is_cartesian_chunk(actions):
        commands = [a['robot_command'] for a in actions]
        labels = ['tx', 'ty', 'tz', 'qw', 'qx', 'qy', 'qz']
        rows = [[*c.pose.translation, *c.pose.rotation.as_quat] for c in commands]
    elif all(isinstance(a.get('robot_command'), JointDelta) for a in actions):
        deltas = [a['robot_command'].velocities for a in actions]
        labels = [f'dq{i}' for i in range(len(deltas[0]))]
        rows = [list(d) for d in deltas]
    else:
        return
    horizon = np.array([float(a.get('timestamp', i)) for i, a in enumerate(actions)])
    horizon -= horizon[0]
    if all('target_grip' in a for a in actions):
        labels.append('target_grip')
        rows = [row + [a['target_grip']] for row, a in zip(rows, actions, strict=True)]
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
    trajectory = rrb.Spatial3DView(origin=f'{_TAP}/robot_command/trajectory', name='trajectory')
    bottom = rrb.Horizontal(trajectory, commands) if has_trajectory else commands
    return rrb.Blueprint(rrb.Vertical(top, bottom))


@cfn.config(dataset=positronic.cfg.ds.local, policy=policy_cfg.remote, episode=0, at=0.0, task=None, label=None)
def main(
    dataset: Dataset, policy: Policy, episode: int, at: float, task: str | None, label: str | None, output_dir: str
):
    ep = dataset[episode]
    ts = int(np.clip(ep.start_ts + int(at * 1e9), ep.start_ts, ep.last_ts))
    sample = ep.time[ts]
    if 'robot_state.ee_pose' not in sample:
        raise ValueError('episode has no robot_state.ee_pose; cannot overlay actual pose')
    task = task or ep.static.get('task')

    now_ns = time.time_ns()
    obs = _build_wire_obs(sample, task, now_ns, ts)
    image_keys = [k for k in obs if k.startswith('image.')]

    rec = Recorder(pos3.sync(output_dir))
    session = rec.tap(_TAP).wrap(policy).new_session({'task': task} if task else None)
    meta = dict(session.meta)
    name = label or _recording_name(meta)
    try:
        actions = session(obs)
        if actions is not None:
            actions = [a for a in actions if is_action(a)]  # drop the codec's keyless validity sentinel
        n = 0 if actions is None else len(actions)
        print(f'episode {episode} @ {at:.3f}s (ts={ts}) [{name}]: {n} action(s); rrd -> {output_dir}')
        with rec.stream:
            rec.stream.send_recording_name(name)
            rr.log('meta', rr.TextDocument(_meta_doc(name, meta), media_type=rr.MediaType.MARKDOWN), static=True)
            if actions:
                _log_commands(actions, now_ns, ts)
            # Sent here, not at Recorder construction: the layout depends on the chunk type, which is
            # only known after inference — a velocity chunk drops the 3D trajectory view.
            rr.send_blueprint(_blueprint(image_keys, _is_cartesian_chunk(actions)))
    finally:
        session.close()


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
