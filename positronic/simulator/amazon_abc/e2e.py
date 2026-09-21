"""End-to-end check that the ABC env server works: drive both arms over the socket and read them back.

positronic launches the env-server subprocess (ABC's own interpreter, its own MuJoCo) and this drives the real
boundary: it commands each arm an absolute Cartesian target anchored on the pose that came back over the wire,
and asserts the arm arrives there while its partner holds still. That is a genuine oracle for the whole path —
the server's IK, the control site it solves against, the per-arm channel routing and the grip polarity all have
to agree for an arm to reach a pose measured in the same frame the command was written in. Run it on a box with
a working GL context::

    uv run --locked python -m positronic.simulator.amazon_abc.e2e --task put_plastic_bottles_in_bin
"""

import argparse

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm import command as roboarm_command
from positronic.eval import keys as eval_keys
from positronic.simulator.amazon_abc import keys as abc_keys
from positronic.simulator.amazon_abc import mapping
from positronic.simulator.amazon_abc.adapter import CAMERAS, AbcAdapter
from positronic.simulator.amazon_abc.launcher import serve_abc
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.client import EnvConnection

_RISE = np.array([0.0, 0.0, 0.02])  # metres, straight up: clear of the table and well inside both arms' reach
_SETTLE_STEPS = 24  # the joints are position-servoed, so a target is approached rather than jumped to
_ARRIVED_TOL = 0.008  # metres
_HELD_TOL = 0.008  # metres the arm nobody commanded may drift while its partner moves


def _message(payload):
    return pimm.Message(payload, ts=0, updated=True)


def _observe(adapter: AbcAdapter, frame: dict) -> dict:
    return adapter.observations(frame[protocol.FRAME_OBS])


def _eef(obs: dict, arm: str) -> geom.Transform3D:
    return obs[keys.arm_channel(keys.ROBOT_STATE, arm)].ee_pose


def _drive(conn: EnvConnection, adapter: AbcAdapter, commands: dict) -> dict:
    """Send ``commands`` once, then hold them for the settle window, and return the last observation."""
    action = adapter.action(commands)
    result: dict = {}
    for _ in range(_SETTLE_STEPS):
        result = conn.step(action)
        action = adapter.action(dict.fromkeys(commands))
    return _observe(adapter, result)


def _raise_one_arm(conn: EnvConnection, adapter: AbcAdapter, start: dict, arm: str) -> dict:
    """Send ``arm`` straight up from where ``start`` reads it, and check both arms once it settles."""
    partner = next(other for other in mapping.ARMS if other != arm)
    target = geom.Transform3D(_eef(start, arm).translation + _RISE, _eef(start, arm).rotation)
    obs = _drive(
        conn,
        adapter,
        {
            keys.arm_channel(keys.ROBOT_COMMAND, arm): _message(roboarm_command.CartesianPosition(target)),
            # Every arm's channel rides every action, so the partner is told to hold where it stands.
            keys.arm_channel(keys.ROBOT_COMMAND, partner): None,
            keys.arm_channel(keys.TARGET_GRIP, arm): _message(1.0),
            keys.arm_channel(keys.TARGET_GRIP, partner): _message(0.0),
        },
    )
    reached = np.linalg.norm(_eef(obs, arm).translation - target.translation)
    drift = np.linalg.norm(_eef(obs, partner).translation - _eef(start, partner).translation)
    closed = obs[keys.arm_channel(keys.GRIP, arm)]
    opened = obs[keys.arm_channel(keys.GRIP, partner)]
    print(f'{arm}: reached {reached * 1000:.1f} mm, {partner} drifted {drift * 1000:.1f} mm')
    assert reached < _ARRIVED_TOL, f'{arm} stopped {reached * 1000:.1f} mm short of its target'
    assert drift < _HELD_TOL, f'{partner} moved {drift * 1000:.1f} mm while only {arm} was commanded'
    assert closed > 0.9, f'{arm} was asked to close and reports {closed:.3f}'
    assert opened < 0.1, f'{partner} was asked to open and reports {opened:.3f}'
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='put_plastic_bottles_in_bin')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--camera-height', type=int, default=168)
    parser.add_argument('--camera-width', type=int, default=224)
    args = parser.parse_args()

    adapter = AbcAdapter(CAMERAS)
    with serve_abc([args.task]) as (host, port):
        conn = EnvConnection(host, port)
        records = conn.tasks({mapping.SELECT_TASKS: args.task})
        params = adapter.task_params(records)
        assert [p[eval_keys.TASK] for p in params] == [args.task], params

        frame = conn.reset(
            adapter.reset_token({
                **params[0],
                eval_keys.SEED: args.seed,
                abc_keys.CAMERA_HEIGHT: args.camera_height,
                abc_keys.CAMERA_WIDTH: args.camera_width,
            })
        )
        assert frame[protocol.FRAME_META][mapping.META_TASK], 'the env reported no instruction'
        assert frame[protocol.FRAME_CONTROL_DT] > 0.0
        start = _observe(adapter, frame)
        for logical in CAMERAS:
            assert start[logical].array.shape == (args.camera_height, args.camera_width, 3), logical
        assert adapter.privileged(frame[protocol.FRAME_OBS])[mapping.OBS_SIM_STATE].size > 0
        print(f'reset: {frame[protocol.FRAME_META][mapping.META_TASK]!r} at {frame[protocol.FRAME_CONTROL_DT]}s')

        for arm in mapping.ARMS:
            start = _raise_one_arm(conn, adapter, start, arm)

        conn.close()
    print('ABC env server e2e passed')


if __name__ == '__main__':
    main()
