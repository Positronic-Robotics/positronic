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
- eef_frame against ``models.DROID_EE_FRAME``: RoboLab's own scene puts eef_frame where the constant
  ``RobolabAdapter`` converts with says it is;
- Cartesian tracking, the ``examples/run_abs_ik_demo.py`` protocol: ±5 cm translations and ±20° world-axis
  rotations held 30 steps each, pass at 5 mm / 2°; the demo's own known-divergent ``translate +x`` case is
  reported but not fatal;
- ``cartesian_delta``: solves to the same joint targets as the equivalent absolute command, and the composed
  target tracks end-to-end;
- with ``--num-envs`` above one, the batch: every clone answers its own frame and takes its own command.

The transform checks all drive a single clone, whatever ``--num-envs`` asks for; the batch check builds its
own env, because a clone count is fixed at ``create_env``.

Run on a RoboLab-capable box the same way the launcher runs ``env.py`` (AppLauncher flags apply)::

    PYTHONPATH=positronic/simulator/env_server \
        uv run --project <robolab clone> positronic/simulator/robolab/validate.py --headless
"""

import math
import sys

import keys
import numpy as np
import protocol  # pyright: ignore[reportMissingImports]
import torch

# Importing ``env`` launches the Isaac app — a precondition for every isaaclab/robolab import below.
from env import RobolabEnv, args, simulation_app
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul

from robolab.robots.droid import EEF_OFFSET_ROT

_TOKEN = {'task': 'BananaInBowlTask', 'instruction_type': 'default'}
_HOLD = {protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: protocol.HOLD}, protocol.ACTION_GRIP: 0.0}
_HOLD_STEPS = 30
_SETTLE_STEPS = 10
_POS_DELTA = 0.05  # m — run_abs_ik_demo's per-case translation magnitude
_ROT_DELTA = math.radians(20.0)
_POS_TOL = 0.005  # m — run_abs_ik_demo's pass tolerance
_ROT_TOL = math.radians(2.0)
# run_abs_ik_demo documents "translate +x" as currently divergent; report it, but don't fail the run on it.
_KNOWN_DIVERGENT = {'translate +x'}
_EEF_OFFSET_ROT_T = torch.tensor([EEF_OFFSET_ROT], dtype=torch.float32)

# ``models.DROID_EE_FRAME`` restated against the flange, because this script runs in RoboLab's venv and cannot
# import positronic: the same transform as ``frame_transform(fr3_urdf, FLANGE_LINK, DROID_EEF_LINK)``, quat wxyz.
# It is measured from an FR3 and applied to RoboLab's Panda, so the check below is what says the two agree.
# rules-allow: hardcoded-keys — this script runs in RoboLab's venv, where positronic is not importable, so
# there is no shared constant to reach for; naming the body is the point of the check
_FLANGE_BODY = 'panda_link8'
_FLANGE_TO_EEF_POS = (0.0, 0.0, 0.018174023)
_FLANGE_TO_EEF_QUAT = (-0.707106781, 0.0, 0.0, -0.707106781)
_FRAME_POS_TOL = 1e-4  # m — float32 body poses, so well above round-off and far below a real geometry change
_FRAME_ROT_TOL = math.radians(0.05)

# The cameras follow ``--cameras``, so a run of any set checks the set that run renders.
_OBS_SPECS = {
    keys.OBS_JOINT_POS: ((7,), np.float32),
    keys.OBS_JOINT_VEL: ((7,), np.float32),
    keys.OBS_EEF_POS: ((3,), np.float32),
    keys.OBS_EEF_QUAT: ((4,), np.float32),
    **dict.fromkeys(keys.CAMERA_SETS[args.cameras], ((720, 1280, 3), np.uint8)),
    keys.OBS_SUBTASK: ((4,), np.float32),
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


def _reset(env: RobolabEnv, token: dict) -> dict:
    """Reset the env under validation and report its one slot's frame: these checks drive a single scene."""
    return protocol.one_slot(env.reset(token))


def _step(env: RobolabEnv, action: dict) -> dict:
    """Step the env under validation with one slot's action and report that slot's frame."""
    return protocol.one_slot(env.step([action]))


def _settle(env: RobolabEnv) -> dict:
    out = _step(env, _HOLD)
    for _ in range(_SETTLE_STEPS - 1):
        out = _step(env, _HOLD)
    return out


def _check_obs_contract(env: RobolabEnv) -> None:
    out = _reset(env, _TOKEN)
    control_dt = out[protocol.FRAME_CONTROL_DT]
    assert abs(control_dt - 1 / 15) < 1e-6, f'control_dt {control_dt} != 1/15'
    step = _step(env, _HOLD)
    assert step.keys() == {
        protocol.FRAME_OBS,
        protocol.FRAME_DONE,
        protocol.FRAME_SUCCESS,
        protocol.FRAME_CONTROL_DT,
    }, f'step keys {sorted(step)}'
    for obs in (out[protocol.FRAME_OBS], step[protocol.FRAME_OBS]):
        for key, (shape, dtype) in _OBS_SPECS.items():
            arr = obs[key]
            assert isinstance(arr, np.ndarray) and arr.shape == shape and arr.dtype == dtype, (
                f'{key}: {type(arr).__name__} shape={getattr(arr, "shape", None)} dtype={getattr(arr, "dtype", None)}'
            )
        rendered = {k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3}
        assert rendered == set(keys.CAMERA_SETS[args.cameras]), f'rendered cameras {sorted(rendered)}'
        grip = obs[keys.OBS_GRIP]
        assert isinstance(grip, float) and 0.0 <= grip <= 1.0, f'grip {grip!r}'
        quat = obs[keys.OBS_EEF_QUAT]
        assert abs(float(np.linalg.norm(quat)) - 1.0) < 1e-3, f'eef_quat norm {quat}'
    print('  obs contract: OK (keys, camera set, shapes, dtypes, quat norm, grip range)')


def _check_grip(env: RobolabEnv) -> None:
    out = _reset(env, _TOKEN)
    for _ in range(_HOLD_STEPS):
        out = _step(env, {protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: protocol.HOLD}, protocol.ACTION_GRIP: 1.0})
    closed = out[protocol.FRAME_OBS][keys.OBS_GRIP]
    assert closed > 0.9, f'closed grip {closed}'
    for _ in range(_HOLD_STEPS):
        out = _step(env, {protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: protocol.HOLD}, protocol.ACTION_GRIP: 0.0})
    opened = out[protocol.FRAME_OBS][keys.OBS_GRIP]
    assert opened < 0.1, f'open grip {opened}'
    print('  grip: OK (closed > 0.9, open < 0.1)')


def _check_joint_pos_passthrough(env: RobolabEnv) -> None:
    _reset(env, _TOKEN)
    q0 = env._measured_q()[0].cpu().numpy()
    target = q0 + np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1], dtype=np.float32)
    out = _step(
        env,
        {
            protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: target},
            protocol.ACTION_GRIP: 0.0,
        },
    )
    applied = env._env.action_manager.action[0, :7].cpu().numpy()
    assert np.array_equal(applied, target), f'joint_pos not passed through bit-identically: {applied} vs {target}'
    for _ in range(59):
        out = _step(
            env,
            {
                protocol.ACTION_COMMAND: {
                    protocol.COMMAND_TYPE: protocol.JOINT_POS,
                    protocol.COMMAND_JOINT_POS: target,
                },
                protocol.ACTION_GRIP: 0.0,
            },
        )
    err = float(np.max(np.abs(out[protocol.FRAME_OBS][keys.OBS_JOINT_POS] - target)))
    assert err < 0.05, f'joint_pos convergence err {err} rad'
    print(f'  joint_pos: OK (bit-identical pass-through; converged to {err:.4f} rad)')


def _check_joint_vel_anchoring(env: RobolabEnv) -> None:
    _reset(env, _TOKEN)
    dq = np.full(7, 0.01, dtype=np.float32)
    expected = env._measured_q()[0] + torch.as_tensor(dq, device=env._env.device)
    _step(
        env,
        {
            protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: protocol.JOINT_DELTA, protocol.COMMAND_JOINT_DELTA: dq},
            protocol.ACTION_GRIP: 0.0,
        },
    )
    applied = env._env.action_manager.action[0, :7]
    assert torch.equal(applied, expected), f'joint_vel target {applied} != q + dq {expected}'
    print('  joint_vel: OK (targets anchor on measured q + dq, exactly)')


def _check_eef_offset(env: RobolabEnv) -> None:
    out = _reset(env, _TOKEN)
    eef_quat = torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_QUAT]).reshape(1, 4)
    base_quat = env._robot.data.body_quat_w[:1, env._body_idx].cpu()
    recovered = quat_mul(eef_quat, quat_inv(_EEF_OFFSET_ROT_T))
    err = min(float((recovered - base_quat).abs().max()), float((recovered + base_quat).abs().max()))
    assert err < 1e-3, f'eef->base_link offset round-trip err {err}'
    print('  eef offset: OK (obs eef_quat un-offsets to the base_link body quat)')


def _check_flange_to_eef(env: RobolabEnv) -> None:
    out = _reset(env, _TOKEN)
    robot, sim = env._robot, env._env
    assert robot is not None and sim is not None, 'reset builds both'
    flange = robot.data.body_names.index(_FLANGE_BODY)
    flange_pos = (robot.data.body_pos_w[:1, flange] - sim.scene.env_origins[:1, 0:3]).cpu()
    flange_quat = robot.data.body_quat_w[:1, flange].cpu()
    eef_pos = torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_POS]).reshape(1, 3)
    eef_quat = torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_QUAT]).reshape(1, 4)

    rel_quat = quat_mul(quat_inv(flange_quat), eef_quat)
    rel_pos = quat_apply(quat_inv(flange_quat), eef_pos - flange_pos)[0]
    pos_err = float(torch.linalg.norm(rel_pos - torch.tensor(_FLANGE_TO_EEF_POS)))
    rot_err = _quat_angle(rel_quat[0], torch.tensor(_FLANGE_TO_EEF_QUAT))
    assert pos_err <= _FRAME_POS_TOL and rot_err <= _FRAME_ROT_TOL, (
        f'RoboLab measures eef_frame at {rel_pos.tolist()} / {rel_quat[0].tolist()} off {_FLANGE_BODY}, '
        f'{pos_err * 1000:.3f} mm and {math.degrees(rot_err):.3f} deg from where DROID_EE_FRAME puts it'
    )
    print(f'  flange -> eef_frame: OK ({pos_err * 1000:.4f} mm / {math.degrees(rot_err):.4f} deg)')


def _run_cartesian_cases(env: RobolabEnv) -> int:
    """The run_abs_ik_demo protocol over the wire ``cartesian`` path; returns the count of non-known failures."""
    _reset(env, _TOKEN)
    out = _settle(env)
    init_pos = torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_POS])
    init_quat = torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_QUAT])
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
        command = {
            protocol.COMMAND_TYPE: protocol.CARTESIAN,
            protocol.COMMAND_POSE: _wire_pose(target_pos, target_quat),
        }
        terminated = False
        for _ in range(_HOLD_STEPS):
            out = _step(env, {protocol.ACTION_COMMAND: command, protocol.ACTION_GRIP: 0.0})
            if out[protocol.FRAME_DONE]:
                terminated = True
                break
        if terminated:
            _reset(env, _TOKEN)
            out = _settle(env)
            init_pos = torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_POS])
            init_quat = torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_QUAT])
            print(f'    {name:<14} {"SKIPPED":<10} {"-":>10} {"-":>11}')
            continue
        pos_err = float(np.linalg.norm(out[protocol.FRAME_OBS][keys.OBS_EEF_POS] - target_pos.numpy()))
        rot_err = _quat_angle(torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_QUAT]), target_quat)
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
    _reset(env, _TOKEN)
    out = _settle(env)
    cur_pos = torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_POS])
    cur_quat = torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_QUAT])
    dpos = torch.tensor([0.0, 0.0, -0.05])
    dquat = _quat_about_axis(math.radians(10.0), 2)
    # The contract compose: translation adds, rotation left-multiplies onto the measured pose (world frame).
    target_pos = cur_pos + dpos
    target_quat = quat_mul(dquat.reshape(1, 4), cur_quat.reshape(1, 4))[0]
    delta_cmd = {protocol.COMMAND_TYPE: protocol.CARTESIAN_DELTA, protocol.COMMAND_DELTA: _wire_pose(dpos, dquat)}
    pose_cmd = {protocol.COMMAND_TYPE: protocol.CARTESIAN, protocol.COMMAND_POSE: _wire_pose(target_pos, target_quat)}
    # Same sim state, no stepping: the delta must solve to the joint targets of the absolute pose it composes to.
    q_delta = env._joint_targets([delta_cmd])
    q_abs = env._joint_targets([pose_cmd])
    assert torch.allclose(q_delta, q_abs, atol=1e-4), f'delta vs absolute joint targets differ: {q_delta - q_abs}'
    # End-to-end: one delta step, then hold the absolute target it defined.
    out = _step(env, {protocol.ACTION_COMMAND: delta_cmd, protocol.ACTION_GRIP: 0.0})
    for _ in range(_HOLD_STEPS - 1):
        out = _step(env, {protocol.ACTION_COMMAND: pose_cmd, protocol.ACTION_GRIP: 0.0})
    pos_err = float(np.linalg.norm(out[protocol.FRAME_OBS][keys.OBS_EEF_POS] - target_pos.numpy()))
    rot_err = _quat_angle(torch.as_tensor(out[protocol.FRAME_OBS][keys.OBS_EEF_QUAT]), target_quat)
    assert pos_err <= _POS_TOL and rot_err <= _ROT_TOL, (
        f'composed delta target missed: {pos_err * 1000:.2f} mm / {math.degrees(rot_err):.2f} deg'
    )
    print(f'  cartesian_delta: OK ({pos_err * 1000:.2f} mm / {math.degrees(rot_err):.2f} deg)')


def _check_batch(num_envs: int) -> None:
    """Every clone answers its own frame and takes the command addressed to it.

    Each clone is driven to a target of its own, so a frame or an action that reached the wrong clone shows
    up as a slot commanded away from its target. Builds its own env: the clone count is fixed at
    ``create_env``, so this cannot share the single-clone env the checks above drive.
    """
    env = RobolabEnv(num_envs)
    out = env.reset(_TOKEN)
    assert len(out[protocol.SLOTS]) == num_envs, (
        f'{num_envs} clones asked for, {len(out[protocol.SLOTS])} frames answered'
    )
    targets = [(env._measured_q()[slot] + 0.05 * (slot + 1)).cpu().numpy() for slot in range(num_envs)]
    out = env.step([
        {
            protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: target},
            protocol.ACTION_GRIP: 0.0,
        }
        for target in targets
    ])
    applied = env._env.action_manager.action[:, :7].cpu().numpy()
    for slot, target in enumerate(targets):
        err = float(np.max(np.abs(applied[slot] - target)))
        assert err < 1e-6, f'clone {slot} was commanded {err:.6f} rad off its own target'
    assert len(out[protocol.SLOTS]) == num_envs, (
        f'the step answered {len(out[protocol.SLOTS])} slots for {num_envs} clones'
    )
    for slot, frame in enumerate(out[protocol.SLOTS]):
        expected = {protocol.FRAME_OBS, protocol.FRAME_DONE, protocol.FRAME_SUCCESS}
        assert frame.keys() == expected, f'clone {slot} answered {sorted(frame)}'
    env.close()
    print(f'  batch: OK ({num_envs} clones answer their own frames and take their own commands)')


def main() -> None:
    env = RobolabEnv()
    _check_obs_contract(env)
    _check_grip(env)
    _check_joint_pos_passthrough(env)
    _check_joint_vel_anchoring(env)
    _check_eef_offset(env)
    _check_flange_to_eef(env)
    failures = _run_cartesian_cases(env)
    _check_cartesian_delta(env)
    env.close()
    if args.num_envs > 1:
        _check_batch(args.num_envs)
    # ``simulation_app.close()`` can end the process outright, so the verdict prints before it — and flushes,
    # because a redirected stdout is block-buffered and would otherwise lose every check line with it.
    if failures:
        print(f'{failures} cartesian case(s) FAILED', flush=True)
        sys.exit(1)
    print('ALL CHECKS PASSED', flush=True)
    simulation_app.close()


if __name__ == '__main__':
    main()
