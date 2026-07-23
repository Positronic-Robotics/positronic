"""On-box validation for ``RobolabEnv``'s command transform — runs inside RoboLab's own venv.

positronic cannot import isaaclab/robolab, so the joint-target mapping and the differential-IK path in
``env.py`` cannot be unit-tested in positronic's suite. This script builds the real env, drives it directly
(no sockets), and checks:

- the wire observation contract: keys, shapes, dtypes, quat normalization, grip range;
- grip: the binary gripper term reaches both closure extremes;
- ``joint_pos`` pass-through: the applied action is bit-identical to the commanded joints (the leaderboard
  equivalence claim) and the arm converges onto the target;
- ``joint_vel``: the joint targets anchor on the measured joints (``q + dq``), exactly;
- the eef_frame <-> base_link offset: un-offsetting the observed eef quat recovers the base_link body quat;
- Cartesian tracking, the ``examples/run_abs_ik_demo.py`` protocol: ±5 cm translations and ±20° world-axis
  rotations held 30 steps each, pass at 5 mm / 2°; the demo's own known-divergent ``translate +x`` case is
  reported but not fatal;
- ``cartesian_delta``: solves to the same joint targets as the equivalent absolute command, and the composed
  target tracks end-to-end.

Run on a RoboLab-capable box the same way the launcher runs ``env.py`` (AppLauncher flags apply)::

    PYTHONPATH=positronic/simulator/env_server \
        uv run --project <robolab clone> positronic/simulator/robolab/validate.py --headless
"""

import math
import sys

import numpy as np
import torch

# Importing ``env`` launches the Isaac app — a precondition for every isaaclab/robolab import below.
from env import RobolabEnv, simulation_app
from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul

from robolab.robots.droid import EEF_OFFSET_ROT

_TOKEN = {'task': 'BananaInBowlTask', 'instruction_type': 'default'}
_HOLD = {'command': {'type': 'hold'}, 'grip': 0.0}
_HOLD_STEPS = 30
_SETTLE_STEPS = 10
_POS_DELTA = 0.05  # m — run_abs_ik_demo's per-case translation magnitude
_ROT_DELTA = math.radians(20.0)
_POS_TOL = 0.005  # m — run_abs_ik_demo's pass tolerance
_ROT_TOL = math.radians(2.0)
# run_abs_ik_demo documents "translate +x" as currently divergent; report it, but don't fail the run on it.
_KNOWN_DIVERGENT = {'translate +x'}
_EEF_OFFSET_ROT_T = torch.tensor([EEF_OFFSET_ROT], dtype=torch.float32)

_OBS_SPECS = {
    'joint_pos': ((7,), np.float32),
    'joint_vel': ((7,), np.float32),
    'eef_pos': ((3,), np.float32),
    'eef_quat': ((4,), np.float32),
    'over_shoulder_left_camera': ((720, 1280, 3), np.uint8),
    'wrist_cam': ((720, 1280, 3), np.uint8),
    'subtask': ((4,), np.float32),
}


def _quat_about_axis(angle: float, axis: int) -> torch.Tensor:
    """Unit wxyz quaternion rotating by ``angle`` radians about world x/y/z (axis 0/1/2)."""
    q = torch.zeros(4)
    q[0] = math.cos(angle / 2)
    q[1 + axis] = math.sin(angle / 2)
    return q


def _quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> float:
    """The shortest rotation angle between two wxyz quats, in radians (double-cover safe)."""
    w = quat_mul(q1.reshape(1, 4), quat_inv(q2.reshape(1, 4)))[0, 0]
    return 2.0 * math.acos(min(1.0, abs(float(w))))


def _wire_pose(pos: torch.Tensor, quat: torch.Tensor) -> np.ndarray:
    """A flat ``[translation(3), rotation_matrix(9)]`` wire pose from an env-local position and a world quat."""
    rot = matrix_from_quat(quat.reshape(1, 4))[0]
    return np.concatenate([pos.numpy(), rot.numpy().reshape(9)]).astype(np.float32)


def _settle(env: RobolabEnv) -> dict:
    out = None
    for _ in range(_SETTLE_STEPS):
        out = env.step(_HOLD)
    return out


def _check_obs_contract(env: RobolabEnv) -> None:
    out = env.reset(_TOKEN)
    assert abs(out['control_dt'] - 1 / 15) < 1e-6, f'control_dt {out["control_dt"]} != 1/15'
    step = env.step(_HOLD)
    assert step.keys() == {'obs', 'done', 'success', 'control_dt', 'timing'}, f'step keys {sorted(step)}'
    assert step['timing'].keys() == {'physics_s', 'render_s', 'wall_s'}, f'timing keys {sorted(step["timing"])}'
    for obs in (out['obs'], step['obs']):
        for key, (shape, dtype) in _OBS_SPECS.items():
            arr = obs[key]
            assert isinstance(arr, np.ndarray) and arr.shape == shape and arr.dtype == dtype, (
                f'{key}: {type(arr).__name__} shape={getattr(arr, "shape", None)} dtype={getattr(arr, "dtype", None)}'
            )
        assert isinstance(obs['grip'], float) and 0.0 <= obs['grip'] <= 1.0, f'grip {obs["grip"]!r}'
        assert abs(float(np.linalg.norm(obs['eef_quat'])) - 1.0) < 1e-3, f'eef_quat norm {obs["eef_quat"]}'
    print('  obs contract: OK (keys, shapes, dtypes, quat norm, grip range)')


def _check_grip(env: RobolabEnv) -> None:
    env.reset(_TOKEN)
    out = None
    for _ in range(_HOLD_STEPS):
        out = env.step({'command': {'type': 'hold'}, 'grip': 1.0})
    assert out['obs']['grip'] > 0.9, f'closed grip {out["obs"]["grip"]}'
    for _ in range(_HOLD_STEPS):
        out = env.step({'command': {'type': 'hold'}, 'grip': 0.0})
    assert out['obs']['grip'] < 0.1, f'open grip {out["obs"]["grip"]}'
    print('  grip: OK (closed > 0.9, open < 0.1)')


def _check_joint_pos_passthrough(env: RobolabEnv) -> None:
    env.reset(_TOKEN)
    q0 = env._measured_q().cpu().numpy()
    target = q0 + np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1], dtype=np.float32)
    out = env.step({'command': {'type': 'joint_pos', 'q': target}, 'grip': 0.0})
    applied = env._env.action_manager.action[0, :7].cpu().numpy()
    assert np.array_equal(applied, target), f'joint_pos not passed through bit-identically: {applied} vs {target}'
    for _ in range(59):
        out = env.step({'command': {'type': 'joint_pos', 'q': target}, 'grip': 0.0})
    err = float(np.max(np.abs(out['obs']['joint_pos'] - target)))
    assert err < 0.05, f'joint_pos convergence err {err} rad'
    print(f'  joint_pos: OK (bit-identical pass-through; converged to {err:.4f} rad)')


def _check_joint_vel_anchoring(env: RobolabEnv) -> None:
    env.reset(_TOKEN)
    dq = np.full(7, 0.01, dtype=np.float32)
    expected = env._measured_q() + torch.as_tensor(dq, device=env._env.device)
    env.step({'command': {'type': 'joint_vel', 'dq': dq}, 'grip': 0.0})
    applied = env._env.action_manager.action[0, :7]
    assert torch.equal(applied, expected), f'joint_vel target {applied} != q + dq {expected}'
    print('  joint_vel: OK (targets anchor on measured q + dq, exactly)')


def _check_eef_offset(env: RobolabEnv) -> None:
    out = env.reset(_TOKEN)
    eef_quat = torch.as_tensor(out['obs']['eef_quat']).reshape(1, 4)
    base_quat = env._robot.data.body_quat_w[:1, env._body_idx].cpu()
    recovered = quat_mul(eef_quat, quat_inv(_EEF_OFFSET_ROT_T))
    err = min(float((recovered - base_quat).abs().max()), float((recovered + base_quat).abs().max()))
    assert err < 1e-3, f'eef->base_link offset round-trip err {err}'
    print('  eef offset: OK (obs eef_quat un-offsets to the base_link body quat)')


def _run_cartesian_cases(env: RobolabEnv) -> int:
    """The run_abs_ik_demo protocol over the wire ``cartesian`` path; returns the count of non-known failures."""
    env.reset(_TOKEN)
    out = _settle(env)
    init_pos = torch.as_tensor(out['obs']['eef_pos'])
    init_quat = torch.as_tensor(out['obs']['eef_quat'])
    zero = torch.zeros(3)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    d, r = _POS_DELTA, _ROT_DELTA
    cases = [
        ('hold initial', zero, identity),
        ('translate +x', torch.tensor([+d, 0.0, 0.0]), identity),
        ('translate -x', torch.tensor([-d, 0.0, 0.0]), identity),
        ('translate +y', torch.tensor([0.0, +d, 0.0]), identity),
        ('translate -y', torch.tensor([0.0, -d, 0.0]), identity),
        ('translate +z', torch.tensor([0.0, 0.0, +d]), identity),
        ('translate -z', torch.tensor([0.0, 0.0, -d]), identity),
        ('rotate +X', zero, _quat_about_axis(+r, 0)),
        ('rotate -X', zero, _quat_about_axis(-r, 0)),
        ('rotate +Y', zero, _quat_about_axis(+r, 1)),
        ('rotate -Y', zero, _quat_about_axis(-r, 1)),
        ('rotate +Z', zero, _quat_about_axis(+r, 2)),
        ('rotate -Z', zero, _quat_about_axis(-r, 2)),
    ]
    failures = 0
    tol = f'{_POS_TOL * 1000:.0f} mm / {math.degrees(_ROT_TOL):.0f} deg'
    print(f'  cartesian tracking ({_HOLD_STEPS} held steps, tol {tol}):')
    print(f'    {"case":<14} {"result":<10} {"pos_err_mm":>10} {"rot_err_deg":>11}')
    for name, dpos, dquat in cases:
        # Absolute targets built from the captured initial pose, so a diverged case doesn't bias the next.
        target_pos = init_pos + dpos
        target_quat = quat_mul(dquat.reshape(1, 4), init_quat.reshape(1, 4))[0]  # world-frame rotation on top
        command = {'type': 'cartesian', 'pose': _wire_pose(target_pos, target_quat)}
        terminated = False
        for _ in range(_HOLD_STEPS):
            out = env.step({'command': command, 'grip': 0.0})
            if out['done']:
                terminated = True
                break
        if terminated:
            env.reset(_TOKEN)
            out = _settle(env)
            init_pos = torch.as_tensor(out['obs']['eef_pos'])
            init_quat = torch.as_tensor(out['obs']['eef_quat'])
            print(f'    {name:<14} {"SKIPPED":<10} {"-":>10} {"-":>11}')
            continue
        pos_err = float(np.linalg.norm(out['obs']['eef_pos'] - target_pos.numpy()))
        rot_err = _quat_angle(torch.as_tensor(out['obs']['eef_quat']), target_quat)
        if pos_err <= _POS_TOL and rot_err <= _ROT_TOL:
            result = 'PASS'
        elif name in _KNOWN_DIVERGENT:
            result = 'KNOWN-FAIL'
        else:
            result = 'FAIL'
            failures += 1
        print(f'    {name:<14} {result:<10} {pos_err * 1000:>10.2f} {math.degrees(rot_err):>11.2f}')
    return failures


def _check_cartesian_delta(env: RobolabEnv) -> None:
    env.reset(_TOKEN)
    out = _settle(env)
    cur_pos = torch.as_tensor(out['obs']['eef_pos'])
    cur_quat = torch.as_tensor(out['obs']['eef_quat'])
    dpos = torch.tensor([0.0, 0.0, -0.05])
    dquat = _quat_about_axis(math.radians(10.0), 2)
    # The contract compose: translation adds, rotation left-multiplies onto the measured pose (world frame).
    target_pos = cur_pos + dpos
    target_quat = quat_mul(dquat.reshape(1, 4), cur_quat.reshape(1, 4))[0]
    delta_cmd = {'type': 'cartesian_delta', 'delta': _wire_pose(dpos, dquat)}
    pose_cmd = {'type': 'cartesian', 'pose': _wire_pose(target_pos, target_quat)}
    # Same sim state, no stepping: the delta must solve to the joint targets of the absolute pose it composes to.
    q_delta = env._joint_targets(delta_cmd)
    q_abs = env._joint_targets(pose_cmd)
    assert torch.allclose(q_delta, q_abs, atol=1e-4), f'delta vs absolute joint targets differ: {q_delta - q_abs}'
    # End-to-end: one delta step, then hold the absolute target it defined.
    out = env.step({'command': delta_cmd, 'grip': 0.0})
    for _ in range(_HOLD_STEPS - 1):
        out = env.step({'command': pose_cmd, 'grip': 0.0})
    pos_err = float(np.linalg.norm(out['obs']['eef_pos'] - target_pos.numpy()))
    rot_err = _quat_angle(torch.as_tensor(out['obs']['eef_quat']), target_quat)
    assert pos_err <= _POS_TOL and rot_err <= _ROT_TOL, (
        f'composed delta target missed: {pos_err * 1000:.2f} mm / {math.degrees(rot_err):.2f} deg'
    )
    print(f'  cartesian_delta: OK ({pos_err * 1000:.2f} mm / {math.degrees(rot_err):.2f} deg)')


def main() -> None:
    env = RobolabEnv()
    _check_obs_contract(env)
    _check_grip(env)
    _check_joint_pos_passthrough(env)
    _check_joint_vel_anchoring(env)
    _check_eef_offset(env)
    failures = _run_cartesian_cases(env)
    _check_cartesian_delta(env)
    env.close()
    # ``simulation_app.close()`` can end the process outright, so the verdict prints before it.
    if failures:
        print(f'{failures} cartesian case(s) FAILED')
        sys.exit(1)
    print('ALL CHECKS PASSED')
    simulation_app.close()


if __name__ == '__main__':
    main()
