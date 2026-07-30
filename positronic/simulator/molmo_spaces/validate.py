"""Validate the MolmoSpaces rig's Cartesian command transform against the live sim.

MolmoSpaces' Franka runs a joint-position controller, so a Cartesian command only reaches it through the
differential IK in ``env.py``. That solver is arithmetic over the live MuJoCo model, which no unit test can
reach (``mapping``'s tests cover the routing with a stub solver, not the kinematics), so it is checked here
against a real benchmark scene — the same shape of check ``simulator/libero/validate.py`` runs for the LIBERO
rig.

Two properties, both on the arm's grasp site (the frame ``observe_payload`` reports, so command and observation
share a frame):

- **FK identity** — the scratch-``MjData`` recompute of the measured joints reproduces the live grasp-site read,
  confirming the scratch evaluation is seeded correctly and reads the same frame.
- **IK round-trip** — for reachable targets sampled by perturbing the measured joints, ``_fk(_ik(pose))``
  recovers the pose. This is the property a Cartesian policy depends on; the sampling stays near the measured
  configuration so every target is reachable and the check tests the solver, not the workspace.

Runs in MolmoSpaces' venv, flat off ``PYTHONPATH`` like ``parity_native.py`` (positronic-free: ``molmo_spaces``
plus this package's ``mapping``/``env``), so positronic's interpreter cannot import it. Needs the asset packs
(``MLSPACES_ASSETS_DIR``) and a GL backend (``MUJOCO_GL``; a GPU-less box uses mesa software EGL). Launch it the
way ``parity.py`` launches the native reference — the venv python under ``launcher.molmo_subprocess_env()``::

    uv run --locked python -c "
    import subprocess
    from positronic.simulator.molmo_spaces import launcher
    subprocess.run([str(launcher.ensure_molmo_venv()),
                    'positronic/simulator/molmo_spaces/validate.py', '--benchmark_dir', '<dir>'],
                   env=launcher.molmo_subprocess_env(), check=True)"
"""

# ``env`` and ``molmo_spaces`` resolve only inside MolmoSpaces' own venv (the flat ``env`` module off PYTHONPATH,
# and the molmo stack), which pyright checks against positronic's deps and cannot see. This module imports no
# positronic packages, so missing-import errors here are exclusively those foreign imports — suppress just that
# category file-wide; every other type check stays active.
# pyright: reportMissingImports=false

import argparse
from pathlib import Path

# env.py sets MUJOCO_GL and installs the CGL stub at import, GL-safely pulling in the molmo_spaces stack — so
# import it before any other molmo_spaces import. Reaching into its private ``_fk``/``_ik`` is the point: this
# validates that exact solver, not a re-derivation of it.
import env  # noqa: E402
import numpy as np

# Sampled targets perturb each measured joint by up to this much (radians): far enough that the solver has real
# work to do, near enough that every target stays reachable and away from the limits.
_JOINT_JITTER = 0.1
_IK_SAMPLES = 16
# The solver iterates to _IK_TOL on the 6-vector error; these are the per-component budgets that implies.
_POS_ATOL = 1e-3  # metres
_ORI_ATOL = 1e-2  # radians
# The live site is read after the sim has stepped, so it carries residual motion the scratch recompute of the
# same joints cannot reproduce exactly; float precision, not float64, is the right bar for the identity.
_FK_ATOL = 1e-5


def _ori_error(target_rot: np.ndarray, rot: np.ndarray) -> float:
    return float(np.linalg.norm(env._pose_error(np.zeros(3), target_rot, np.zeros(3), rot)[3:]))


def _check_fk_identity(sim_env) -> None:
    pos_fk, rot_fk = sim_env._fk(sim_env._measured_arm_q())
    pos_live, rot_live = sim_env._measured_eef_pose()
    assert np.allclose(pos_fk, pos_live, atol=_FK_ATOL), f'fk pos {pos_fk} vs live {pos_live}'
    assert np.allclose(rot_fk, rot_live, atol=_FK_ATOL), f'fk rot {rot_fk} vs live {rot_live}'
    print(f'  fk identity: OK (matches the grasp-site read, atol {_FK_ATOL})')


def _check_ik_roundtrip(sim_env) -> None:
    measured = np.asarray(sim_env._measured_arm_q(), dtype=np.float64)
    for _ in range(_IK_SAMPLES):
        target_pos, target_rot = sim_env._fk(measured + np.random.uniform(-_JOINT_JITTER, _JOINT_JITTER, measured.size))
        pos, rot = sim_env._fk(sim_env._ik(target_pos, target_rot))
        ang = _ori_error(target_rot, rot)
        assert np.allclose(pos, target_pos, atol=_POS_ATOL), f'ik pos off by {pos - target_pos}'
        assert ang < _ORI_ATOL, f'ik orientation off by {ang} rad'
    print(f'  ik round-trip: OK ({_IK_SAMPLES} reachable targets, pos<{_POS_ATOL} m, ori<{_ORI_ATOL} rad)')


def _check_cartesian_command_is_a_noop_at_the_measured_pose(sim_env) -> None:
    # Commanding the pose the arm already holds must resolve to (essentially) the joints it already holds —
    # the property that makes an absolute Cartesian setpoint stable when a policy re-sends it.
    pos, rot = sim_env._measured_eef_pose()
    command = {'type': 'cartesian', 'pose': np.concatenate([pos, rot.reshape(-1)])}
    target = env.mapping.wire_command_to_arm_action(command, sim_env._measured_arm_q(), ik=sim_env._ik)
    drift = np.abs(np.asarray(target, dtype=np.float64) - np.asarray(sim_env._measured_arm_q(), dtype=np.float64))
    assert drift.max() < 1e-3, f'holding the measured pose moved the joints by {drift.max()} rad'
    print(f'  cartesian hold: OK (max joint drift {drift.max():.2e} rad)')


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the MolmoSpaces rig's Cartesian command transform.")
    parser.add_argument('--benchmark_dir', required=True, help='dir containing benchmark.json')
    parser.add_argument('--episode_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    # These checks are pure kinematics — they never step the sim, so the episode horizon is irrelevant to them.
    # The override keeps a benchmark that declares no ``task_horizon_sec`` (the checked-in test benchmark is one)
    # usable as a scene here, instead of failing the build over a field this run never reads.
    parser.add_argument('--task_horizon_steps', type=int, default=1)
    args = parser.parse_args()
    np.random.seed(0)

    sim_env = env.MolmoSpacesEnv(Path(args.benchmark_dir), args.task_horizon_steps)
    sim_env.reset({'episode_index': args.episode_index, 'seed': args.seed})
    print(f'molmo_spaces episode {args.episode_index} (seed {args.seed})')
    try:
        _check_fk_identity(sim_env)
        _check_ik_roundtrip(sim_env)
        _check_cartesian_command_is_a_noop_at_the_measured_pose(sim_env)
    finally:
        sim_env.close()
    print('all checks passed')


if __name__ == '__main__':
    main()
