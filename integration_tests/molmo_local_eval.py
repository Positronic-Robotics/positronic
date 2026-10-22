"""Run a public MolmoSpaces eval through the eval CLI, as `docs/local-molmospaces-eval.md` does, and check it."""

import subprocess
import sys
from collections import Counter
from pathlib import Path

import configuronic as cfn
from positronic_evals.evals import MOLMO_EPISODE_INDEX_KEY, EvalDefinition, EvalRef, public_eval
from positronic_evals.labels import label
from positronic_evals.scoring import Scores, score

from positronic.cli.eval.run import SCORE_LINE_PREFIX, score_line
from positronic.dataset.episode import META_PATH
from positronic.dataset.local_dataset import LocalDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = 'molmo.franka_pick_mini_smoke'
LOG_NAME = 'eval.log'
RECORDING_DIR = 'episodes'


def eval_command(eval_name: str, host: str, port: int, output: Path) -> list[str]:
    """The command of the guide's run step."""
    return [
        str(Path(sys.executable).with_name('positronic')),
        'eval',
        'run',
        f'--eval={eval_name}',
        '--policy=.remote',
        f'--policy.address.host={host}',
        f'--policy.address.port={port}',
        f'--output_dir={output}',
    ]


def recorded_score(definition: EvalDefinition, recording: Path) -> Scores:
    """Check that `recording` holds each trial of `definition` once, each labelled, and return its score."""
    dataset = LocalDataset(recording)
    episodes = [Path(episode.meta[META_PATH]) for episode in dataset]
    trials = Counter(trial[MOLMO_EPISODE_INDEX_KEY] for task in definition.tasks for trial in task.trials)
    recorded = Counter(episode.static.get(MOLMO_EPISODE_INDEX_KEY) for episode in dataset)
    if recorded != trials:
        raise ValueError(f'{recording}: recorded trials {dict(recorded)}, the eval defines {dict(trials)}')
    labelled = [label(episode, definition.scoring.labelers) for episode in episodes]
    unlabelled = [episode.name for episode, given in zip(episodes, labelled, strict=True) if given is None]
    if unlabelled:
        raise ValueError(f'{recording}: no labeler labels the episodes {unlabelled}')
    return score(definition.scoring, [given[1] for given in labelled if given is not None])


def check_log(log: Path, expected: Scores) -> None:
    """Check that the run printed one score, and that it is `expected`."""
    want = score_line(expected)
    printed = [line for line in log.read_text().splitlines() if line.startswith(SCORE_LINE_PREFIX)]
    if printed != [want]:
        raise ValueError(f'{log}: the run printed {printed}, the recorded episodes score {want!r}')


def check_run(output: Path, eval_name: str) -> None:
    expected = recorded_score(public_eval(EvalRef(eval_name)), output / RECORDING_DIR)
    check_log(output / LOG_NAME, expected)
    print(f'PASS: {score_line(expected)}')


@cfn.config(eval_name=SMOKE)
def check(output_dir: str, eval_name: str):
    check_run(Path(output_dir).expanduser(), eval_name)


@cfn.config(eval_name=SMOKE, host='localhost', port=8000, wall_timeout=3600.0)
def run(output_dir: str, eval_name: str, host: str, port: int, wall_timeout: float):
    # The eval runs with `cwd=REPO_ROOT`, so a relative path would name a second directory there.
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True)
    log = output / LOG_NAME
    print(f'Running {eval_name}; log: {log}', flush=True)
    with log.open('w') as stream:
        subprocess.run(
            eval_command(eval_name, host, port, output / RECORDING_DIR),
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=wall_timeout,
        )
    check_run(output, eval_name)


def main() -> int:
    try:
        cfn.cli({'run': run, 'check': check})
    except (ValueError, OSError, LookupError, subprocess.SubprocessError) as exc:
        print(f'FAIL: {exc}\nRecordings and the log remain in the requested output_dir.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
