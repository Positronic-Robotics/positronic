"""Deterministic replay of recorded pi05 rollouts against the MolmoSpaces integration.

The parity check (``parity.py``) pins that positronic drives the sim exactly as MolmoSpaces' own stack does,
but it drives a scripted hold — no real policy behaviour ever reaches the sim. This replays the commands a
real pi05 rollout emitted, open-loop from the benchmark's own seed, and asserts the sim retraces the recorded
trajectory step for step.

It reproduces because the recorded commands are absolute joint targets, so no observation feeds back into the
stream: the replay depends only on the sim rollout and the env-server path between positronic and it. MuJoCo
CPU physics is deterministic for a pinned build, and the render nondeterminism ``exp-004`` found never reaches
an open-loop replay, which reads no images. Same-host replay of the pinned ``_MOLMO_COMMIT`` reproduces the
recorded ``sim_state`` *exactly* — zero deviation across every replayed step of both fixtures — so a
divergence here is a real change in the integration (command mapping, horizon, the wire, scene selection),
not sim noise.

The rollout stops where the recording stops pinning it: an eval episode's command signals end before its
observations do, so the last few steps of every recording apply commands that were never written down
(internal#130). Those steps are excluded, which is why this asserts trajectory fidelity and not the recorded
``eval.success`` — success lands inside that unwritten tail. Closing internal#130 is what would let the
verdict itself be replayed.

Fixtures (``replay_ep*.npz``, from ``make_replay_fixture.py``) hold only the commands, the grip and
checkpoints of the recorded ``sim_state`` — never the videos. The benchmark is a multi-hundred-MB asset pack
that cannot be committed, so the test skips unless this box has it.

Run on a box with the asset packs (a GPU-less one uses mesa software EGL)::

    MLSPACES_ASSETS_DIR=... MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
        uv run --locked pytest positronic/simulator/molmo_spaces/tests/test_replay.py --no-cov
"""

import os
from pathlib import Path

import numpy as np
import pytest

from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.molmo_spaces import launcher

FIXTURES = sorted(Path(__file__).parent.glob('replay_ep*.npz'))

# Replay on the recording's own host reproduces every checkpoint bit-for-bit, so this budget is for the float
# drift a different CPU can introduce. It sits orders of magnitude below the ~centimetre scale that decides a
# pick, so it cannot mask a real regression; a cross-host run that exceeds it is worth investigating rather
# than widening.
SIM_STATE_TOL = 1e-6


def _benchmark_dir(benchmark_path: str) -> Path:
    """The benchmark the fixture was recorded against, resolved in this box's asset packs.

    The fixture pins the path from ``benchmarks/`` down, so the lookup is exact rather than a name search:
    the same benchmark name sits under every scene dataset with different episodes, and resolving to the
    wrong one would replay these commands against a different scene.
    """
    assets = os.environ.get('MLSPACES_ASSETS_DIR')
    if not assets:
        pytest.skip('MLSPACES_ASSETS_DIR is unset — MolmoSpaces asset packs are needed to replay')
    benchmark_dir = Path(assets) / 'benchmarks' / benchmark_path
    if not (benchmark_dir / 'benchmark.json').is_file():
        pytest.skip(f'{benchmark_dir} is absent — this asset pack cannot replay the fixture')
    return benchmark_dir


def _replay(benchmark_dir: Path, episode_index: int, commands: np.ndarray, grips: np.ndarray) -> list[np.ndarray]:
    """Step the recorded commands open-loop, returning the sim state each one produced."""
    states: list[np.ndarray] = []
    with launcher.serve_molmo_spaces(benchmark_dir) as (host, port):
        conn = EnvConnection(host, port)
        try:
            # No seed: the benchmark episode carries its own, exactly as the recorded run left it unset.
            conn.reset({'episode_index': episode_index, 'seed': None})
            for command, grip in zip(commands, grips, strict=True):
                out = conn.step({'command': {'type': protocol.JOINT_POS, 'q': command}, 'grip': float(grip)})
                states.append(np.asarray(out['obs']['sim_state'], dtype=np.float64))
                if out['done']:
                    break
        finally:
            conn.close()
    return states


@pytest.mark.parametrize('fixture_path', FIXTURES, ids=lambda path: path.stem)
def test_recorded_rollout_replays_the_recorded_trajectory(fixture_path: Path):
    fixture = np.load(fixture_path, allow_pickle=False)
    commands, grips = fixture['commands'], fixture['grips']
    episode_index = int(fixture['episode_index'])
    states = _replay(_benchmark_dir(str(fixture['benchmark_path'])), episode_index, commands, grips)

    # The sim must not end the trial inside the replayed prefix: the recording ran every one of these steps,
    # so an early terminal means the integration now scores or expires the episode differently.
    assert len(states) == len(commands), (
        f'replay of episode {episode_index} terminated after {len(states)} of {len(commands)} recorded steps '
        f'(its {int(fixture["unreplayable_tail_steps"])} unrecorded tail steps are excluded)'
    )

    # Checkpoints along the way, not just the end state: drift shows up long before it would flip a verdict.
    for step, recorded in zip(fixture['checkpoint_steps'], fixture['checkpoint_sim_state'], strict=True):
        replayed = states[int(step) - 1]  # states[i] holds step i + 1; the fixture indexes steps from 1
        deviation = float(np.max(np.abs(replayed - recorded)))
        assert deviation <= SIM_STATE_TOL, (
            f'sim_state diverged by {deviation:.3e} at step {step} of episode {episode_index}'
        )
