"""Deterministic replay of recorded pi05 rollouts against the MolmoSpaces integration.

The parity check (``parity.py``) pins that positronic drives the sim exactly as MolmoSpaces' own stack does,
but it drives a scripted hold — no real policy behaviour ever reaches the sim. This replays the commands a
real pi05 rollout emitted, open-loop from the benchmark's own seed, and asserts the sim retraces the pinned
trajectory step for step.

It reproduces because the recorded commands are absolute joint targets, so no observation feeds back into the
stream: the replay depends only on the sim rollout and the env-server path between positronic and it. MuJoCo
CPU physics is deterministic for a pinned build, and the render nondeterminism ``exp-004`` found never reaches
an open-loop replay, which reads no images. Same-host replay of the pinned ``_MOLMO_COMMIT`` reproduces the
fixture's ``sim_state`` *exactly* — zero deviation across every replayed step of both fixtures — so a
divergence here is a real change in the integration (command mapping, horizon, the wire, scene selection),
not sim noise.

The rollout stops where the recording stops pinning it: an eval episode's command signals end before its
observations do, so the last few steps of every recording apply commands that were never written down
(internal#130). Those steps are excluded, which is why this asserts trajectory fidelity and not the recorded
``eval.success`` — success lands inside that unwritten tail. Closing internal#130 is what would let the
verdict itself be replayed.

Fixtures (``replay_ep*.npz``, from ``make_replay_fixture.py``) hold only the commands, the grip and
``sim_state`` checkpoints — never the videos. The commands come from the recording; the checkpoints are taken
by replaying them, so what this pins is the integration's own trajectory under a real policy's commands, and
they are regenerated together. The benchmark is a multi-hundred-MB asset pack that cannot be committed, so
the test skips unless this box has it.

Run on a box with the asset packs (a GPU-less one uses mesa software EGL)::

    MLSPACES_ASSETS_DIR=... MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
        uv run --locked pytest positronic/simulator/molmo_spaces/tests/test_replay.py --no-cov
"""

import os
from pathlib import Path

import numpy as np
import pytest

from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.tests import make_replay_fixture as fixture_fields
from positronic.simulator.molmo_spaces.tests.make_replay_fixture import replay_commands

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
    assets = os.environ.get(mapping.ASSETS_DIR_ENV)
    if not assets:
        pytest.skip(f'{mapping.ASSETS_DIR_ENV} is unset — MolmoSpaces asset packs are needed to replay')
    benchmark_dir = Path(assets) / mapping.ASSETS_BENCHMARKS_DIR / benchmark_path
    if not (benchmark_dir / mapping.MOLMO_BENCHMARK_MANIFEST).is_file():
        pytest.skip(f'{benchmark_dir} is absent — this asset pack cannot replay the fixture')
    return benchmark_dir


@pytest.mark.parametrize('fixture_path', FIXTURES, ids=lambda path: path.stem)
def test_recorded_rollout_replays_the_pinned_trajectory(fixture_path: Path):
    fixture = np.load(fixture_path, allow_pickle=False)
    commands, grips = fixture[fixture_fields.FIELD_COMMANDS], fixture[fixture_fields.FIELD_GRIPS]
    episode_index = int(fixture[fixture_fields.FIELD_EPISODE_INDEX])
    benchmark_dir = _benchmark_dir(str(fixture[fixture_fields.FIELD_BENCHMARK_PATH]))
    states = replay_commands(benchmark_dir, episode_index, commands, grips)

    # The sim must not end the trial inside the replayed prefix: the recording ran every one of these steps,
    # so an early terminal means the integration now scores or expires the episode differently.
    assert len(states) == len(commands), (
        f'replay of episode {episode_index} terminated after {len(states)} of {len(commands)} recorded steps '
        f'(its {int(fixture[fixture_fields.FIELD_UNREPLAYABLE_TAIL_STEPS])} unrecorded tail steps are excluded)'
    )

    # Checkpoints along the way, not just the end state: drift shows up long before it would flip a verdict.
    checkpoint_steps = fixture[fixture_fields.FIELD_CHECKPOINT_STEPS]
    for step, pinned in zip(checkpoint_steps, fixture[fixture_fields.FIELD_CHECKPOINT_SIM_STATE], strict=True):
        replayed = states[int(step) - 1]  # states[i] holds step i + 1; the fixture indexes steps from 1
        deviation = float(np.max(np.abs(replayed - pinned)))
        assert deviation <= SIM_STATE_TOL, (
            f'sim_state diverged by {deviation:.3e} at step {step} of episode {episode_index}'
        )
