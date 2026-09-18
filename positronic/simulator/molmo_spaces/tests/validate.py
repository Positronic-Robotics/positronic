"""Validate the MolmoSpaces rig's command transforms against the live sim.

MolmoSpaces' Franka runs a joint-position controller, so every other command reaches it only through the
conversions in ``mapping``, the Cartesian pair through MolmoSpaces' IK. The solver uses a separate robot
model, so its results are checked against the grasp site in a real benchmark scene (``mapping``'s unit tests
cover the routing with a stub solver).

Four properties. The kinematic three read the arm's grasp site (the frame the env observes in, so command and
observation share a frame); the fourth is the adoption's coverage of the command contract:

- **FK identity** — the scratch-``MjData`` recompute of the measured joints reproduces the live grasp-site read.
- **IK round-trip** — for reachable targets sampled by perturbing the measured joints, ``_fk(_ik(pose))``
  recovers the pose. This is the property a Cartesian policy depends on.
- **Cartesian hold** — commanding the pose the arm already holds resolves to the joints it already holds, which
  makes an absolute Cartesian setpoint stable when a policy re-sends it.
- **Command contract** — every canonical command type converts to joint targets through the live IK.

Runs in MolmoSpaces' venv, flat off ``PYTHONPATH`` like ``parity_native.py`` (positronic-free: ``molmo_spaces``
plus this package's ``mapping``/``env``), so positronic's interpreter cannot import it. Needs the asset packs
(``MLSPACES_ASSETS_DIR``) and a GL backend (``MUJOCO_GL``; a GPU-less box uses mesa software EGL). Launch it the
way ``parity.py`` launches the native reference — the venv python under ``launcher.molmo_subprocess_env()``::

    uv run --locked python -c "
    import subprocess
    from positronic.simulator.molmo_spaces import launcher
    subprocess.run([str(launcher.ensure_molmo_venv()),
                    'positronic/simulator/molmo_spaces/tests/validate.py',
                    '--benchmark', '<suite/scene_dataset/task_config/benchmark>'],
                   env=launcher.molmo_subprocess_env(), check=True)"
"""

# The flat ``protocol`` module resolves only inside MolmoSpaces' own venv, where this validation runs; pyright
# checks it against positronic's deps, which cannot see it. That import carries its own
# ``reportMissingImports`` suppression, so one that should resolve here still fails the check.

import argparse
import os
from pathlib import Path

# env.py installs the CGL stub before loading MolmoSpaces; import it before any other molmo_spaces import.
import env  # noqa: E402
import mapping  # noqa: E402 -- positronic-free wire mappings, on PYTHONPATH
import mujoco  # noqa: E402
import numpy as np
import protocol  # pyright: ignore[reportMissingImports] -- flat on PYTHONPATH beside ``server``, see ``launcher``

# Sampled targets perturb each measured joint by up to this much (radians): far enough that the solver has real
# work to do, near enough that every target stays reachable and away from the limits.
_JOINT_JITTER = 0.1
_IK_SAMPLES = 16
_POS_ATOL = 1e-3  # metres
_ORI_ATOL = 1e-2  # radians
# The live site is read after the sim has stepped, so it carries residual motion the scratch recompute of the
# same joints cannot reproduce exactly; float precision, not float64, is the right bar for the identity.
_FK_ATOL = 1e-5
# The step the relative commands carry: small enough that the target stays reachable from the measured
# configuration, large enough that the conversion is not the identity.
_DELTA_POS = 0.01  # metres
_DELTA_Q = 0.01  # radians


def _fk(sim_env, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The grasp-site world pose a candidate arm configuration reaches — the inverse of the rig's ``_ik``.

    Runs on a copy of the live scene's ``MjData``, so the checks below probe candidate
    joints without perturbing the sim.
    """
    arm = sim_env._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
    data = mujoco.MjData(arm.mj_model)  # pyright: ignore[reportAttributeAccessIssue]
    mujoco.mj_copyData(data, arm.mj_model, arm.mj_data)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[np.asarray(arm.joint_posadr)] = np.asarray(q, dtype=np.float64).reshape(-1)
    mujoco.mj_forward(arm.mj_model, data)  # pyright: ignore[reportAttributeAccessIssue]
    return data.site_xpos[arm.leaf_frame_id].copy(), data.site_xmat[arm.leaf_frame_id].reshape(3, 3).copy()


def _check_fk_identity(sim_env) -> None:
    pos_fk, rot_fk = _fk(sim_env, sim_env._measured_arm_q())
    pos_live, rot_live = sim_env._measured_eef_pose()
    assert np.allclose(pos_fk, pos_live, atol=_FK_ATOL), f'fk pos {pos_fk} vs live {pos_live}'
    assert np.allclose(rot_fk, rot_live, atol=_FK_ATOL), f'fk rot {rot_fk} vs live {rot_live}'
    print(f'  fk identity: OK (matches the grasp-site read, atol {_FK_ATOL})')


def _check_ik_roundtrip(sim_env) -> None:
    measured = np.asarray(sim_env._measured_arm_q(), dtype=np.float64)
    for _ in range(_IK_SAMPLES):
        jitter = np.random.uniform(-_JOINT_JITTER, _JOINT_JITTER, measured.size)
        target_pos, target_rot = _fk(sim_env, measured + jitter)
        pos, rot = _fk(sim_env, sim_env._ik(target_pos, target_rot))
        ang = float(np.arccos(np.clip((np.trace(target_rot @ rot.T) - 1) / 2, -1, 1)))
        assert np.allclose(pos, target_pos, atol=_POS_ATOL), f'ik pos off by {pos - target_pos}'
        assert ang < _ORI_ATOL, f'ik orientation off by {ang} rad'
    print(f'  ik round-trip: OK ({_IK_SAMPLES} reachable targets, pos<{_POS_ATOL} m, ori<{_ORI_ATOL} rad)')


def _check_cartesian_command_is_a_noop_at_the_measured_pose(sim_env) -> None:
    # Commanding the pose the arm already holds must resolve to (essentially) the joints it already holds —
    # the property that makes an absolute Cartesian setpoint stable when a policy re-sends it.
    pos, rot = sim_env._measured_eef_pose()
    command = {protocol.COMMAND_TYPE: protocol.CARTESIAN, protocol.COMMAND_POSE: np.concatenate([pos, rot.reshape(-1)])}
    target = env.mapping.wire_command_to_arm_action(
        command, sim_env._measured_arm_q(), ik=sim_env._ik, current_eef=(pos, rot)
    )
    drift = np.abs(np.asarray(target, dtype=np.float64) - np.asarray(sim_env._measured_arm_q(), dtype=np.float64))
    assert drift.max() < 1e-3, f'holding the measured pose moved the joints by {drift.max()} rad'
    print(f'  cartesian hold: OK (max joint drift {drift.max():.2e} rad)')


def _check_every_canonical_command_converts(sim_env) -> None:
    """Drive every canonical command type through the real conversion — the adoption's whole obligation.

    The command contract is total: MolmoSpaces' Franka natively takes joint-position targets alone, so each
    canonical type has to reach it as one. ``mapping``'s unit tests pin that routing against a stub solver;
    here each type runs through the live IK and the measured pose, so a type the rig cannot actually resolve
    fails. The iteration is over ``protocol.CANONICAL_COMMAND_TYPES`` rather than a list written here, so a
    type added to the wire fails this check until the rig converts it.
    """
    measured = np.asarray(sim_env._measured_arm_q(), dtype=np.float64)
    pos, rot = sim_env._measured_eef_pose()
    identity_rot = np.eye(3).reshape(-1)
    payloads = {
        protocol.CARTESIAN: {protocol.COMMAND_POSE: np.concatenate([pos, rot.reshape(-1)])},
        protocol.CARTESIAN_DELTA: {protocol.COMMAND_DELTA: np.concatenate([np.full(3, _DELTA_POS), identity_rot])},
        protocol.JOINT_POS: {protocol.COMMAND_JOINT_POS: measured},
        protocol.JOINT_VEL: {protocol.COMMAND_JOINT_VEL: np.full(measured.size, _DELTA_Q)},
        protocol.HOLD: {},
    }
    unmapped = [kind for kind in protocol.CANONICAL_COMMAND_TYPES if kind not in payloads]
    assert not unmapped, f'the rig has no wire payload for canonical command types {unmapped}'

    for kind in protocol.CANONICAL_COMMAND_TYPES:
        command = {protocol.COMMAND_TYPE: kind, **payloads[kind]}
        target = env.mapping.wire_command_to_arm_action(
            command, measured, ik=sim_env._ik, current_eef=sim_env._measured_eef_pose()
        )
        target = np.asarray(target, dtype=np.float64)
        assert target.shape == measured.shape, f'{kind}: joint targets {target.shape} vs measured {measured.shape}'
        assert np.all(np.isfinite(target)), f'{kind}: non-finite joint targets {target}'
    covered = ', '.join(protocol.CANONICAL_COMMAND_TYPES)
    print(f'  command contract: OK ({covered} -> {measured.size} joint targets)')


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the MolmoSpaces rig's Cartesian command transform.")
    parser.add_argument(
        '--benchmark',
        type=mapping.BenchmarkPath.parse,
        required=True,
        help='suite/scene_dataset/task_config/benchmark under the asset packs',
    )
    parser.add_argument('--episode_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    np.random.seed(0)

    sim_env = env.MolmoSpacesEnv(Path(os.environ[mapping.ASSETS_DIR_ENV]))
    token = {**args.benchmark._asdict(), mapping.TOKEN_EPISODE_INDEX: args.episode_index, mapping.TOKEN_SEED: args.seed}
    sim_env.reset(token)
    print(f'molmo_spaces episode {args.episode_index} (seed {args.seed})')
    try:
        _check_fk_identity(sim_env)
        _check_ik_roundtrip(sim_env)
        _check_cartesian_command_is_a_noop_at_the_measured_pose(sim_env)
        _check_every_canonical_command_converts(sim_env)
    finally:
        sim_env.close()
    print('all checks passed')


if __name__ == '__main__':
    main()
