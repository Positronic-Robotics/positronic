"""Native-vs-positronic parity check for the MolmoSpaces integration.

The fidelity check ``docs/architecture.md`` ("Benchmarks are native; adoptions are faithful") mandates for every
sim-env integration, added here for MolmoSpaces. It drives one pinned benchmark episode twice — natively through
MolmoSpaces' own stack (``parity_native.py``: ``JsonEvalTaskSampler`` -> ``reset``/``step``/``is_done``/
``judge_success``, MolmoSpaces' native horizon) and through the positronic path (launcher -> env server -> wire ->
the raw payload the ``MolmoAdapter`` maps) — feeding the *same* scripted actions (hold the arm, gripper open) and
asserts they agree byte-for-byte.

MolmoSpaces benchmark episodes are exact-pose deterministic, so the strong fidelity form applies: identical call
sequence (same step count, same terminating step) and byte-identical outcomes modulo wire format (joint
positions/velocities, eef pose, gripper closure and success verdict equal at every step; camera frames equal by
content hash). Holding the arm never succeeds, so the episode runs out its horizon — exercising the horizon case
explicitly: both stacks terminate at exactly the native ``task_horizon`` step, via the wire ``done``, with
``success=False``. The two rollouts run in separate MuJoCo processes, so equality also confirms the render path is
deterministic across processes.

This asserts fidelity against the pinned ``_MOLMO_COMMIT`` (``launcher.py``); re-run it on any bump of that pin
before merge — a sim version change can silently shift the horizon or the rollout.

Needs the MolmoSpaces asset packs (``MLSPACES_ASSETS_DIR``) and a GL backend (``MUJOCO_GL``; a GPU-less box uses
mesa software EGL — ``EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1``), and a benchmark whose task spec carries
``task_horizon_sec`` (the horizon the sim owns). Run on a box with those::

    MLSPACES_ASSETS_DIR=... MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
        uv run --locked python -m positronic.simulator.molmo_spaces.parity --benchmark_dir <dir>
"""

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.molmo_spaces import launcher, mapping

# parity_native.py runs only in MolmoSpaces' venv (it imports the flat, positronic-free ``env``), so reference it
# by path — importing it into positronic's interpreter would fail on that import.
_PARITY_NATIVE = Path(__file__).parent / 'parity_native.py'
_HOLD = {'command': {'type': protocol.HOLD}, 'grip': 0.0}
_ARRAY_FIELDS = (mapping.OBS_JOINT_POS, mapping.OBS_JOINT_VEL, mapping.OBS_EEF_POS, mapping.OBS_EEF_QUAT)


def _is_rgb(value: object) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3 and value.dtype == np.uint8


def _drive_positronic(benchmark_dir: Path, episode_index: int, seed: int, max_steps: int) -> dict:
    """Drive one episode through launcher -> env server -> wire, holding the arm to the sim's own ``done``."""
    fields: dict[str, list] = {k: [] for k in (*_ARRAY_FIELDS, 'grip')}
    camera_names: list[str] = []
    cam_hashes: dict[str, list[str]] = {}

    def record(obs: dict) -> None:
        for key in fields:
            fields[key].append(obs[key])
        for name in camera_names:
            cam_hashes[name].append(hashlib.sha256(np.ascontiguousarray(obs[name]).tobytes()).hexdigest())

    with launcher.serve_molmo_spaces(benchmark_dir) as (host, port):
        conn = EnvConnection(host, port)
        try:
            frame = conn.reset({mapping.TOKEN_EPISODE_INDEX: episode_index, mapping.TOKEN_SEED: seed})
            reported_horizon = frame[protocol.FRAME_HORIZON]
            camera_names = [k for k, v in frame['obs'].items() if _is_rgb(v)]
            cam_hashes = {name: [] for name in camera_names}
            record(frame['obs'])
            out = {'done': False, 'success': False}
            step = 0
            while not out['done'] and step < max_steps:
                out = conn.step(_HOLD)
                step += 1
                record(out['obs'])
        finally:
            conn.close()
    return {
        **{key: np.stack(fields[key]) for key in _ARRAY_FIELDS},
        mapping.OBS_GRIP: np.array(fields[mapping.OBS_GRIP], dtype=np.float32),
        'camera_names': camera_names,
        'cam_hashes': cam_hashes,
        'reported_horizon': reported_horizon,
        'termination_step': step,
        'final_success': bool(out['success']),
    }


def _run_native(benchmark_dir: Path, episode_index: int, seed: int, max_steps: int, out_path: Path) -> dict:
    """Drive the native reference (``parity_native.py``) in MolmoSpaces' venv and load its recorded rollout."""
    python = launcher.ensure_molmo_venv()
    subprocess.run(
        [
            str(python),
            str(_PARITY_NATIVE),
            '--benchmark_dir',
            str(benchmark_dir),
            '--episode_index',
            str(episode_index),
            '--seed',
            str(seed),
            '--max_steps',
            str(max_steps),
            '--out',
            str(out_path),
        ],
        env=launcher.molmo_subprocess_env(),
        check=True,
    )
    return dict(np.load(out_path, allow_pickle=False))


def _assert_parity(native: dict, positronic: dict, max_steps: int) -> None:
    horizon = int(native['native_horizon'])
    n_term, p_term = int(native['termination_step']), positronic['termination_step']
    assert n_term < max_steps, f'native never terminated in {max_steps} steps — raise --max_steps above the horizon'
    assert p_term < max_steps, f'positronic never terminated in {max_steps} steps — raise --max_steps above the horizon'
    # The horizon case: holding the arm never succeeds, so both stacks run out the native horizon and stop there.
    assert n_term == horizon == p_term, (
        f'terminating step differs: native {n_term}, horizon {horizon}, positronic {p_term}'
    )
    assert not bool(native['final_success']) and not positronic['final_success'], 'a held arm must not score success'
    # The env reports its horizon at reset (in sim-seconds); it must match native's and equal timeout's yardstick.
    n_horizon = float(native['horizon_sec'])
    assert n_horizon == positronic['reported_horizon'], (
        f'reported horizon differs: native {n_horizon}s, positronic {positronic["reported_horizon"]}s'
    )

    assert list(native['camera_names']) == positronic['camera_names'], 'camera sets differ between the stacks'
    for field in (*_ARRAY_FIELDS, 'grip'):
        n, p = native[field], positronic[field]
        assert n.shape == p.shape, f'{field} shape differs: native {n.shape}, positronic {p.shape}'
        assert np.array_equal(n, p), f'{field} differs between native and positronic rollouts'
    for name in positronic['camera_names']:
        n_hashes, p_hashes = list(native[f'cam_hash__{name}']), positronic['cam_hashes'][name]
        assert n_hashes == p_hashes, f'camera {name} frames differ between native and positronic rollouts'


def run(benchmark_dir: Path, *, episode_index: int = 0, seed: int = 0, max_steps: int = 1200) -> None:
    """Run the same episode natively and through positronic and assert byte-identical parity."""
    with tempfile.TemporaryDirectory() as tmp:
        native = _run_native(benchmark_dir, episode_index, seed, max_steps, Path(tmp) / 'native.npz')
        positronic = _drive_positronic(benchmark_dir, episode_index, seed, max_steps)
    _assert_parity(native, positronic, max_steps)
    frames = positronic['termination_step'] + 1
    print(f'PARITY PASSED — episode {episode_index}: {frames} frames, terminated at horizon {native["native_horizon"]}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Native-vs-positronic parity check for MolmoSpaces.')
    parser.add_argument(
        '--benchmark_dir', required=True, help='dir containing benchmark.json (task_horizon_sec required)'
    )
    parser.add_argument('--episode_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=1200, help='safety cap; must exceed the benchmark horizon')
    args = parser.parse_args()
    run(Path(args.benchmark_dir), episode_index=args.episode_index, seed=args.seed, max_steps=args.max_steps)


if __name__ == '__main__':
    main()
