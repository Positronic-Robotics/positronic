"""Check command conversions against a live MolmoSpaces scene.

Checks FK against the measured grasp-site pose, IK on reachable targets, Cartesian hold,
and conversion of every canonical command type.

Requires MolmoSpaces' environment, asset packs and a GL backend.
Run with the launcher's subprocess environment::

    uv run --locked python -c "
    import subprocess
    from positronic.simulator.molmo_spaces import launcher
    subprocess.run([str(launcher.ensure_molmo_venv()),
                    'positronic/simulator/molmo_spaces/tests/validate.py',
                    '--benchmark', '<suite/scene_dataset/task_config/benchmark>'],
                   env=launcher.molmo_subprocess_env(), check=True)"
"""

import argparse
import os
from pathlib import Path

# env.py installs the CGL stub before loading MolmoSpaces; import it before any other molmo_spaces import.
import env  # noqa: E402
import mapping  # noqa: E402 -- positronic-free wire mappings, on PYTHONPATH
import mujoco  # noqa: E402
import numpy as np
import protocol  # pyright: ignore[reportMissingImports] -- flat on PYTHONPATH beside ``server``, see ``launcher``

_JOINT_JITTER = 0.1  # radians
_IK_SAMPLES = 16
_POS_ATOL = 1e-3  # metres
_ORI_ATOL = 1e-2  # radians
_FK_ATOL = 1e-5  # Measured joints are float32; FK uses float64.
_DELTA_POS = 0.01  # metres
_DELTA_Q = 0.01  # radians


def _fk(sim_env, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The grasp-site world pose for candidate joints, computed on a copy of the scene state."""
    arm = sim_env._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
    data = mujoco.MjData(arm.mj_model)
    mujoco.mj_copyData(data, arm.mj_model, arm.mj_data)
    data.qpos[np.asarray(arm.joint_posadr)] = np.asarray(q, dtype=np.float64).reshape(-1)
    mujoco.mj_forward(arm.mj_model, data)
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
    pos, rot = sim_env._measured_eef_pose()
    command = {protocol.COMMAND_TYPE: protocol.CARTESIAN, protocol.COMMAND_POSE: np.concatenate([pos, rot.reshape(-1)])}
    target = env.mapping.wire_command_to_arm_action(
        command, sim_env._measured_arm_q(), ik=sim_env._ik, current_eef=(pos, rot)
    )
    drift = np.abs(np.asarray(target, dtype=np.float64) - np.asarray(sim_env._measured_arm_q(), dtype=np.float64))
    assert drift.max() < 1e-3, f'holding the measured pose moved the joints by {drift.max()} rad'
    print(f'  cartesian hold: OK (max joint drift {drift.max():.2e} rad)')


def _check_every_canonical_command_converts(sim_env) -> None:
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
