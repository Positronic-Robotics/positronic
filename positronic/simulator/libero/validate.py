# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git",
#     "robosuite==1.4.1",
#     "numpy<2",
#     "msgpack",
#     "websockets",
# ]
# ///
"""On-box validation for ``LiberoEnv``'s command transform — runs inside LIBERO's own 3.10 interpreter.

positronic is pinned ``>=3.11`` and cannot import LIBERO/robosuite/mujoco, so the FK/IK/normalization in
``env.py`` cannot be unit-tested in positronic's suite. This script builds the real env and checks the transform
against robosuite's *own* control path:

- ``ee``/OSC_POSE, ``joint``/JOINT_POSITION, ``joint_delta``/JOINT_VELOCITY: drive the active controller's
  ``set_goal`` forward from a random normalized action to its goal setpoint, feed that setpoint back through
  ``_arm_action``, and assert the recovered action equals the original. This proves ``_arm_action`` is the exact
  inverse of robosuite's ``scale_action`` + ``set_goal`` — byte-identical to what a LIBERO policy drives.
- forward kinematics: ``_fk`` of the live joints reproduces the controller's eef-site read to the bit.
- inverse kinematics: ``_fk(_ik(pose))`` recovers the target pose (round-trip; LIBERO has no IK to match).

Run on a LIBERO-capable box, with the positronic-free ``server`` module on the path the launcher uses::

    PYTHONPATH=positronic/simulator/env_server \
        uv run --no-project positronic/simulator/libero/validate.py
"""

import argparse

import numpy as np
from env import LiberoEnv
from robosuite.utils.transform_utils import mat2quat, quat2axisangle

_OSC_SAMPLES = 64
_IK_SAMPLES = 16
_ATOL = 1e-9  # joint cells and FK are float64-exact, so the algebra inverts to float error
_OSC_ATOL = 1e-5  # robosuite mat2quat casts to float32, so the OSC orientation channel carries ~1e-6 error


def _osc_goal(c) -> np.ndarray:  # the OSC goal pose packed into the [pos(3), R(9)] wire vector
    return np.concatenate([c.goal_pos, np.asarray(c.goal_ori).reshape(9)])


def _goal_qpos(c) -> np.ndarray:
    return np.asarray(c.goal_qpos)


def _goal_vel(c) -> np.ndarray:
    return np.asarray(c.goal_vel)


def _check_serve(env: LiberoEnv) -> None:
    env.reset(0)
    out = None
    for _ in range(5):
        out = env.step({'command': {'type': 'hold'}, 'grip': 0.0})
    assert {'agentview_image', 'joint_pos', 'eef_pos', 'eef_quat', 'sim_state'} <= out['obs'].keys()
    print('  serve smoke: OK (5 hold steps, obs keys present)')


def _check_action_inverse(env: LiberoEnv, ctype: str, key: str, goal_payload, atol: float = _ATOL) -> None:
    """For ``_OSC_SAMPLES`` random actions: forward through the active controller's ``set_goal``, invert via
    ``_arm_action``, assert the recovered action matches. ``goal_payload`` reads the controller's resulting goal
    setpoint into the wire command field ``key``."""
    env.reset(0)
    c = env._controller
    dim = len(c.input_max)
    for _ in range(_OSC_SAMPLES):
        action = np.random.uniform(-1.0, 1.0, dim)
        c.set_goal(action.copy())
        recovered = env._arm_action({'type': ctype, key: goal_payload(c)})
        assert np.allclose(recovered, action, atol=atol), f'{ctype}: {action} -> {recovered}'
    print(f'  {ctype} inverse: OK ({_OSC_SAMPLES} random actions, atol {atol})')


def _check_fk_identity(env: LiberoEnv) -> None:
    env.reset(0)
    pos_fk, rot_fk = env._fk(env._cur_q())
    pos_live, rot_live = env._cur_pose()
    assert np.allclose(pos_fk, pos_live, atol=_ATOL), f'fk pos {pos_fk} vs live {pos_live}'
    assert np.allclose(rot_fk, rot_live, atol=_ATOL), f'fk rot {rot_fk} vs live {rot_live}'
    print(f'  fk identity: OK (matches eef-site read, atol {_ATOL})')


def _check_ik_roundtrip(env: LiberoEnv) -> None:
    env.reset(0)
    n = len(env._qpos_idx)
    for _ in range(_IK_SAMPLES):
        target_pos, target_rot = env._fk(env._cur_q() + np.random.uniform(-0.1, 0.1, n))
        pos, rot = env._fk(env._ik(target_pos, target_rot))
        ang = np.linalg.norm(quat2axisangle(mat2quat(target_rot @ rot.T)))
        assert np.allclose(pos, target_pos, atol=1e-3), f'ik pos off by {pos - target_pos}'
        assert ang < 1e-2, f'ik orientation off by {ang} rad'
    print(f'  ik round-trip: OK ({_IK_SAMPLES} reachable targets, pos<1e-3 m, ori<1e-2 rad)')


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LiberoEnv's command transform on a LIBERO box.")
    parser.add_argument('--suite', default='libero_spatial', help='LIBERO task suite')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--camera-resolution', type=int, default=128)
    args = parser.parse_args()
    np.random.seed(0)

    print('ee / OSC_POSE')
    ee = LiberoEnv(args.suite, args.task_id, args.camera_resolution, 'ee')
    _check_serve(ee)
    _check_action_inverse(ee, 'cartesian', 'pose', _osc_goal, atol=_OSC_ATOL)
    _check_fk_identity(ee)
    _check_ik_roundtrip(ee)
    ee.close()

    print('joint / JOINT_POSITION')
    jp = LiberoEnv(args.suite, args.task_id, args.camera_resolution, 'joint')
    _check_action_inverse(jp, 'joint_pos', 'q', _goal_qpos)
    jp.close()

    print('joint_delta / JOINT_VELOCITY')
    jv = LiberoEnv(args.suite, args.task_id, args.camera_resolution, 'joint_delta')
    _check_action_inverse(jv, 'joint_vel', 'dq', _goal_vel)
    jv.close()

    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
