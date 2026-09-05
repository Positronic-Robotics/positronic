# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Regenerate a deterministic-replay fixture from a recorded MolmoSpaces eval episode.

``test_replay.py`` replays a real pi05 rollout open-loop against the sim and asserts it reproduces. This
script distils one recorded episode into the fixture that replay needs: the commanded joint targets and grip
per step, taken from the recording, plus checkpoints of the ``sim_state`` those commands produce, taken by
replaying them here. The commands are what the recording pins; the checkpoints pin the integration's current
trajectory, so a later run that drifts from it fails. Regenerate them together whenever the recorded
``sim_state`` changes shape or the pinned MolmoSpaces commit moves.

Two properties make the distillation exact. The recorded commands are *absolute* joint targets, so the replay
is open-loop. And the proxy applies the last command received when it steps, so sampling the command signal at
each observation frame's timestamp reconstructs the stream the sim saw.

An episode's command signals stop before its observations do (internal#130), so the fixture keeps the prefix up
to the final recorded command and counts the rest as the recording's gap. Commands are stored as float32, the
dtype ``env.py`` casts them to.

Run (needs positronic for the dataset reader, and the MolmoSpaces assets for the replay — hence
``--locked``, not ``--no-project``)::

    MLSPACES_ASSETS_DIR=... MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
    uv run --locked python positronic/simulator/molmo_spaces/tests/make_replay_fixture.py \
        --dataset_dir ~/.cache/positronic/s3/_/inference/molmo_battle_test/2026-07-29/sweep_jp \
        --episode_index 3 --episode_index 6

Output: ``replay_ep<NN>.npz`` next to this script, one per episode (tens of KB — actions and checkpoints
only, never the videos).
"""

import argparse
import os
from pathlib import Path

import numpy as np

from positronic import keys
from positronic.dataset.local_dataset import DiskEpisode
from positronic.dataset.signal import Signal
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.molmo_spaces import launcher, mapping

# The fixture's own fields, as a distilled episode records them.
FIELD_EPISODE_INDEX = 'episode_index'
FIELD_BENCHMARK_PATH = 'benchmark_path'
FIELD_TASK = 'task'
FIELD_COMMANDS = 'commands'
FIELD_GRIPS = 'grips'
FIELD_UNREPLAYABLE_TAIL_STEPS = 'unreplayable_tail_steps'
FIELD_CHECKPOINT_STEPS = 'checkpoint_steps'
FIELD_CHECKPOINT_SIM_STATE = 'checkpoint_sim_state'
FIELD_EXPECTED_SUCCESS = 'expected_success'

# Checkpoint stride over the replayed steps: dense enough that drift is caught early rather than only at the
# end state, sparse enough to keep the fixture small. The final step is always included on top.
CHECKPOINT_STRIDE = 8


def find_episode_dir(dataset_dir: Path, episode_index: int) -> Path:
    """The recorded episode directory whose spec carries ``episode_index``."""
    for path in sorted(dataset_dir.rglob('static.json')):
        episode_dir = path.parent
        if DiskEpisode(episode_dir).static.get(molmo_keys.EPISODE_INDEX) == episode_index:
            return episode_dir
    raise SystemExit(f'no recorded episode with {molmo_keys.EPISODE_INDEX}={episode_index} under {dataset_dir}')


def benchmark_of(episode: DiskEpisode) -> mapping.BenchmarkPath:
    """The benchmark the episode was recorded against, as its trial params name it."""
    return mapping.BenchmarkPath(*(episode.static[key] for key in molmo_keys.BENCHMARK_DIMENSIONS))


def sample_at(signal: Signal, timestamps: list[int]) -> list:
    """The signal's value at each timestamp — the last one at or before it, a pimm receiver's semantics."""
    sampled = signal.time[timestamps]
    assert isinstance(sampled, Signal)  # a sequence of timestamps samples a Signal, a single one a record
    return [value for value, _ts in sampled]


def replay_commands(
    bench: mapping.BenchmarkPath, episode_index: int, commands: np.ndarray, grips: np.ndarray
) -> list[np.ndarray]:
    """Step the commands open-loop through a MolmoSpaces env server, returning the sim state each produced.

    Stops early if the sim ends the trial, so a caller can tell a full replay from a truncated one by the
    length of what comes back.
    """
    states: list[np.ndarray] = []
    with launcher.serve_molmo_spaces() as (host, port):
        conn = EnvConnection(host, port)
        try:
            # No seed: the benchmark episode carries its own, exactly as the recorded run left it unset.
            conn.reset({**bench._asdict(), mapping.TOKEN_EPISODE_INDEX: episode_index, mapping.TOKEN_SEED: None})
            for command, grip in zip(commands, grips, strict=True):
                action = {
                    protocol.ACTION_COMMAND: {
                        protocol.COMMAND_TYPE: protocol.JOINT_POS,
                        protocol.COMMAND_JOINT_POS: command,
                    },
                    protocol.ACTION_GRIP: float(grip),
                }
                out = conn.step(action)
                states.append(np.asarray(out[protocol.FRAME_OBS][mapping.OBS_SIM_STATE], dtype=np.float64))
                if out[protocol.FRAME_DONE]:
                    break
        finally:
            conn.close()
    return states


def build_fixture(episode_dir: Path) -> dict[str, np.ndarray]:
    episode = DiskEpisode(episode_dir)
    bench = benchmark_of(episode)
    states = episode[mapping.OBS_SIM_STATE]
    # rules-allow: hardcoded-keys — 'target_grip' is a canonical channel name spelled across every
    # adoption and the eval configs; it belongs in positronic.keys, as its own sweep (internal#211).
    commands, grips = episode[keys.TARGET_JOINTS], episode['target_grip']
    # Frame 0 is the reset observation; every later frame is one step.
    frame_ts = [ts for _value, ts in states]
    step_ts = frame_ts[1:]
    played = [np.asarray(value, dtype=np.float32) for value in sample_at(commands, step_ts)]
    grip = [float(np.asarray(value).reshape(-1)[0]) for value in sample_at(grips, step_ts)]

    # The recording's command signals stop before its observations do (internal#130), so only the steps up to
    # and including the first one that reads the final recorded command are pinned by the recording; past that
    # the commands the run actually applied were never written, and no substitute reproduces them. Replay that
    # prefix and report the rest as the recording's gap rather than replaying commands it does not contain.
    last_command_ts = commands[len(commands) - 1][1]
    replayable = int(np.searchsorted(step_ts, last_command_ts, side='left')) + 1

    steps = np.arange(1, replayable + 1)
    checkpoints = np.unique(np.concatenate([steps[::CHECKPOINT_STRIDE], steps[-1:]]))
    episode_index = int(episode.static[molmo_keys.EPISODE_INDEX])
    played_prefix = np.stack(played[:replayable])
    grip_prefix = np.array(grip[:replayable], dtype=np.float32)
    replayed = replay_commands(bench, episode_index, played_prefix, grip_prefix)
    if len(replayed) != replayable:
        raise SystemExit(
            f'episode {episode_index}: the sim ended after {len(replayed)} of {replayable} replayable steps, '
            'so the recording and the integration no longer agree on the trial length'
        )
    return {
        FIELD_EPISODE_INDEX: np.asarray(episode.static[molmo_keys.EPISODE_INDEX], dtype=np.int32),
        FIELD_BENCHMARK_PATH: np.asarray(str(bench.relative)),
        FIELD_TASK: np.asarray(episode.static[mapping.META_TASK]),
        FIELD_COMMANDS: played_prefix,
        FIELD_GRIPS: grip_prefix,
        FIELD_UNREPLAYABLE_TAIL_STEPS: np.asarray(len(step_ts) - replayable, dtype=np.int32),
        FIELD_CHECKPOINT_STEPS: checkpoints.astype(np.int32),
        FIELD_CHECKPOINT_SIM_STATE: np.stack([replayed[int(step) - 1] for step in checkpoints]),
        FIELD_EXPECTED_SUCCESS: np.asarray(episode.static[eval_keys.SUCCESS], dtype=bool),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Distil recorded eval episodes into replay fixtures.')
    parser.add_argument('--dataset_dir', type=Path, required=True, help='recorded eval run (holds run_metadata)')
    parser.add_argument(
        '--episode_index', type=int, action='append', required=True, help='benchmark episode to distil; repeatable'
    )
    args = parser.parse_args()

    if not os.environ.get(mapping.ASSETS_DIR_ENV):
        raise SystemExit(
            f'{mapping.ASSETS_DIR_ENV} must point at the MolmoSpaces asset packs — the checkpoints '
            'are taken by replaying the recorded commands, which needs the benchmark scene'
        )

    for episode_index in args.episode_index:
        fixture = build_fixture(find_episode_dir(args.dataset_dir, episode_index))
        if not fixture[FIELD_EXPECTED_SUCCESS]:
            raise SystemExit(f'episode {episode_index} did not succeed — replay fixtures pin successful rollouts')
        out = Path(__file__).parent / f'replay_ep{episode_index:02d}.npz'
        np.savez_compressed(out, **fixture)  # pyright: ignore[reportArgumentType] -- numpy's savez **kwds stub
        steps = len(fixture[FIELD_COMMANDS])
        print(f'Wrote {out} ({out.stat().st_size} bytes, {steps} steps, {fixture[FIELD_BENCHMARK_PATH]})')


if __name__ == '__main__':
    main()
