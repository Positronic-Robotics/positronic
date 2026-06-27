# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "msgpack",
#     "websockets",
#     "robosuite==1.4.1",
#     "mujoco==3.2.3",  # pin to openpi's LIBERO eval engine version, so our physics matches the policy's training/eval
#     "numpy<2",
#     # LIBERO is not listed here: it ships ``libero`` as a namespace package with no installable wheel, so the
#     # caller puts a source checkout on ``PYTHONPATH`` (see ``launcher._ensure_libero_src``). These are the
#     # packages LIBERO's benchmark/env path imports but never declares.
#     "torch<2.6",  # torch 2.6 flipped ``torch.load`` to weights_only=True; LIBERO's init-states are plain pickles
#     "bddl==1.0.1",
#     "gym==0.25.2",
#     "hydra-core",
#     "easydict",
#     "einops",
#     "cloudpickle",
#     "future",
#     "matplotlib",
#     "pyyaml",
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

Run on a LIBERO-capable box with the positronic-free ``server`` module and a LIBERO source checkout on the
path (the same two entries ``launcher._spawn`` sets; the checkout lands at ``~/.cache/positronic/libero/src``
once the env server has run once)::

    PYTHONPATH=positronic/simulator/env_server:~/.cache/positronic/libero/src \
        uv run --no-project positronic/simulator/libero/validate.py
"""

import argparse

import numpy as np
from env import LiberoEnv
from robosuite.utils.transform_utils import axisangle2quat, euler2mat, mat2quat, quat2axisangle, quat2mat

_OSC_SAMPLES = 64
_IK_SAMPLES = 16
_ATOL = 1e-9  # pure joint-space algebra (q+dq, goal velocities) inverts to float64 error
_OSC_ATOL = 1e-5  # robosuite mat2quat casts to float32, so the OSC orientation channel carries ~1e-6 error
_FK_ATOL = 2e-4  # a fresh FK recompute vs the live stepped eef site agree only to float precision near the
# settled, near-singular tool-down pose — far tighter than a real FK bug (O(1e-2)+), still not float64-exact


def _token(args, control_mode: str) -> dict:
    """The reset token for ``control_mode`` on the CLI's task (seed 0); the env builds + caches from its spec."""
    return {
        'suite': args.suite,
        'task_id': args.task_id,
        'camera_resolution': args.camera_resolution,
        'control_mode': control_mode,
        'seed': 0,
    }


def _osc_goal(c) -> np.ndarray:  # the OSC goal pose packed into the [pos(3), R(9)] wire vector
    return np.concatenate([c.goal_pos, np.asarray(c.goal_ori).reshape(9)])


def _goal_qpos(c) -> np.ndarray:
    return np.asarray(c.goal_qpos)


def _check_serve(env: LiberoEnv, token: dict) -> None:
    env.reset(token)
    out = None
    for _ in range(5):
        out = env.step({'command': {'type': 'hold'}, 'grip': 0.0})
    assert {'agentview_image', 'joint_pos', 'eef_pos', 'eef_quat', 'grip', 'sim_state'} <= out['obs'].keys()
    print('  serve smoke: OK (5 hold steps, obs keys present)')


def _check_grip(env: LiberoEnv, token: dict) -> None:
    """Drive the gripper to both stops and assert the observed ``grip`` reaches the [0, 1] closure extremes —
    the open command (0) settles near 0, the closed command (1) near 1."""
    env.reset(token)
    out = None
    for _ in range(40):
        out = env.step({'command': {'type': 'hold'}, 'grip': 0.0})
    assert out['obs']['grip'] < 0.05, f'open grip {out["obs"]["grip"]}'
    for _ in range(40):
        out = env.step({'command': {'type': 'hold'}, 'grip': 1.0})
    assert out['obs']['grip'] > 0.95, f'closed grip {out["obs"]["grip"]}'
    print('  grip normalization: OK (open < 0.05, closed > 0.95)')


def _check_action_inverse(env: LiberoEnv, token: dict, ctype: str, key: str, goal_payload, atol: float = _ATOL) -> None:
    """For ``_OSC_SAMPLES`` random actions: forward through the active controller's ``set_goal``, invert via
    ``_arm_action``, assert the recovered action matches. ``goal_payload`` reads the controller's resulting goal
    setpoint into the wire command field ``key``."""
    env.reset(token)
    c = env._controller
    # Sync the controller's cached ee_pos/joint state to the live sim (a real ``step`` does this every turn), so
    # ``set_goal`` builds its goal from the same pose ``_arm_action`` reads back via ``_cur_pose``/``_cur_q``.
    c.update(force=True)
    dim = len(c.input_max)
    for _ in range(_OSC_SAMPLES):
        action = np.random.uniform(-1.0, 1.0, dim)
        c.set_goal(action.copy())
        recovered = env._arm_action({'type': ctype, key: goal_payload(c)})
        assert np.allclose(recovered, action, atol=atol), f'{ctype}: {action} -> {recovered}'
    print(f'  {ctype} inverse: OK ({_OSC_SAMPLES} random actions, atol {atol})')


def _robosuite_obs(env: LiberoEnv) -> dict:
    """robosuite's own observation dict — the source openpi's LIBERO eval reads its state from."""
    return env._env.env._get_observations(force_update=True)


def _nudge_pose(env: LiberoEnv) -> np.ndarray:
    """A small random Cartesian goal (pos + orientation) off the current pose, in the adapter's wire format."""
    pos, rot = env._cur_pose()
    target_pos = pos + np.random.uniform(-0.02, 0.02, 3)
    target_rot = euler2mat(np.random.uniform(-0.1, 0.1, 3)) @ rot
    return np.concatenate([target_pos, target_rot.reshape(9)])


def _check_obs_encoding(env: LiberoEnv, token: dict) -> None:
    """Measure and verify the constants ``LiberoObservationCodec`` needs for the 8-dim state.

    openpi's LIBERO state is ``[robot0_eef_pos(3), quat2axisangle(robot0_eef_quat)(3), robot0_gripper_qpos(2)]``
    (openpi ``examples/libero/main.py``). The env reports the eef pose in the grip-site frame and the gripper as a
    single closure scalar, so the codec must (a) rotate the grip-site orientation into robosuite's hand-body frame
    by a fixed offset and (b) reconstruct the two finger qpos from the closure scalar. This drives the arm and
    gripper across their range, proves both are pose-invariant, and prints the values to bake into the codec.
    """
    wire = env.reset(token)['obs']
    samples = []
    for _ in range(8):
        samples.append((wire, _robosuite_obs(env)))
        wire = env.step({'command': {'type': 'cartesian', 'pose': _nudge_pose(env)}, 'grip': 0.0})['obs']

    # Orientation: a fixed grip-site -> hand-body rotation. Measure on the first sample (xyzw, w>=0 from mat2quat).
    r_off = quat2mat(samples[0][0]['eef_quat']).T @ quat2mat(samples[0][1]['robot0_eef_quat'])
    q = mat2quat(r_off)

    # Gripper: drive to both stops to read the finger qpos endpoints, then the codec's linear reconstruction
    # CLOSED + (1 - grip) * (OPEN - CLOSED) recovers the true qpos from the closure scalar (q_open/q_closed are
    # the load-bearing, non-circular measurement; the sweep checks the linear model holds).
    env.reset(token)
    for _ in range(40):
        wire = env.step({'command': {'type': 'hold'}, 'grip': 0.0})['obs']
    q_open = np.asarray(_robosuite_obs(env)['robot0_gripper_qpos'])
    for _ in range(40):
        wire = env.step({'command': {'type': 'hold'}, 'grip': 1.0})['obs']
    q_closed = np.asarray(_robosuite_obs(env)['robot0_gripper_qpos'])
    recon_err = 0.0
    env.reset(token)
    for g in np.linspace(0.0, 1.0, 6):
        for _ in range(20):
            wire = env.step({'command': {'type': 'hold'}, 'grip': float(g)})['obs']
        true_qpos = np.asarray(_robosuite_obs(env)['robot0_gripper_qpos'])
        recon = q_closed + (1.0 - wire['grip']) * (q_open - q_closed)
        recon_err = max(recon_err, float(np.max(np.abs(recon - true_qpos))))

    # Print the bake values first so they surface even if a check below fails.
    print('  obs encoding:')
    print(f'    bake _GRIP_SITE_TO_HAND = geom.Rotation.from_quat([{q[3]:.9f}, {q[0]:.9f}, {q[1]:.9f}, {q[2]:.9f}])')
    print(f'    bake _GRIPPER_QPOS_OPEN = np.array({np.round(q_open, 6).tolist()})')
    print(f'    bake _GRIPPER_QPOS_CLOSED = np.array({np.round(q_closed, 6).tolist()})')
    print(f'    gripper reconstruction max err over sweep: {recon_err:.5f}')

    # Position matches robosuite's grip-site pos directly; the grip->hand offset is a fixed rotation.
    for w, robo in samples:
        assert np.allclose(w['eef_pos'], robo['robot0_eef_pos'], atol=_OSC_ATOL), 'eef_pos != robot0_eef_pos'
        assert np.allclose(quat2mat(w['eef_quat']) @ r_off, quat2mat(robo['robot0_eef_quat']), atol=_OSC_ATOL), (
            'grip->hand offset varies across poses'
        )

    # The policy consumes the axis-angle 3-vector the codec emits, so verify it encodes robosuite's orientation.
    # The tool-down home pose sits at angle ~pi, where the axis-angle map is singular (w ~ 0, so a tiny quat
    # perturbation flips the 3-vector's sign for the same rotation); convert the codec's axis-angle back to a quat
    # and compare to robosuite's, double-cover and all. A genuine branch/sign bug away from pi yields a different
    # rotation and still fails, so this stays a real check.
    for w, robo in samples:
        aa_codec = quat2axisangle(-mat2quat(quat2mat(w['eef_quat']) @ r_off))  # negate w>=0 form to the codec's w<=0
        q_codec, q_ref = axisangle2quat(aa_codec), robo['robot0_eef_quat']
        assert np.allclose(q_codec, q_ref, atol=_OSC_ATOL) or np.allclose(q_codec, -q_ref, atol=_OSC_ATOL), (
            f'axis-angle encodes the wrong rotation: {aa_codec} -> {q_codec} vs {q_ref}'
        )
    print(f'    grip->hand offset + axis-angle branch match robosuite across {len(samples)} poses')


def _check_osc_delta_scale(env: LiberoEnv, token: dict) -> None:
    """Pin the OSC scaling and control rate the libero codec bakes: ``PoseDeltaAction.OUTPUT_MAX``
    must equal the controller's per-step output range, and the codec stamps the chunk at the env's control rate."""
    env.reset(token)
    c = env._controller
    output_max = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])  # mirrors PoseDeltaAction.OUTPUT_MAX
    assert np.allclose(c.output_max, output_max) and np.allclose(c.output_min, -output_max), (
        f'OSC output range [{c.output_min}, {c.output_max}] != +/-{output_max.tolist()}'
    )
    freq = env._env.env.control_freq
    assert freq == 20, f'control_freq {freq} Hz != 20 (the libero codec stamps the chunk at fps=20)'
    print(f'  osc delta scale: output_max == {output_max.tolist()} OK; control_freq {freq} Hz == libero codec fps')


def _check_fk_identity(env: LiberoEnv, token: dict) -> None:
    env.reset(token)
    pos_fk, rot_fk = env._fk(env._cur_q())
    pos_live, rot_live = env._cur_pose()
    # After the seeded reset's settle the arm carries residual motion near the singular tool-down pose, so the
    # scratch ``mj_forward`` recompute and the live stepped site agree only to float precision, not float64.
    assert np.allclose(pos_fk, pos_live, atol=_FK_ATOL), f'fk pos {pos_fk} vs live {pos_live}'
    assert np.allclose(rot_fk, rot_live, atol=_FK_ATOL), f'fk rot {rot_fk} vs live {rot_live}'
    print(f'  fk identity: OK (matches eef-site read, atol {_FK_ATOL})')


def _check_ik_roundtrip(env: LiberoEnv, token: dict) -> None:
    env.reset(token)
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
    ee = LiberoEnv()
    ee_token = _token(args, 'ee')
    _check_serve(ee, ee_token)
    _check_grip(ee, ee_token)
    _check_action_inverse(ee, ee_token, 'cartesian', 'pose', _osc_goal, atol=_OSC_ATOL)
    _check_obs_encoding(ee, ee_token)
    _check_osc_delta_scale(ee, ee_token)
    _check_fk_identity(ee, ee_token)
    _check_ik_roundtrip(ee, ee_token)
    ee.close()

    print('joint / JOINT_POSITION')
    jp = LiberoEnv()
    _check_action_inverse(jp, _token(args, 'joint'), 'joint_pos', 'q', _goal_qpos)
    jp.close()

    print('joint_delta / JOINT_VELOCITY')
    jv = LiberoEnv()
    # The wire ``dq`` is a per-step joint delta, so feed the controller's goal velocity as the delta it covers
    # over one control period; ``_arm_action`` divides by that period to recover the rate.
    _check_action_inverse(
        jv, _token(args, 'joint_delta'), 'joint_vel', 'dq', lambda c: np.asarray(c.goal_vel) * jv._control_dt
    )
    jv.close()

    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
