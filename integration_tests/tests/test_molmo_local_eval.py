import pytest
from positronic_evals.evals import MOLMO_EPISODE_INDEX_KEY, EvalRef, public_eval
from positronic_evals.scoring import Scores
from pydantic import JsonValue

from integration_tests.molmo_local_eval import LOG_NAME, RECORDING_DIR, SMOKE, check_log, check_run, recorded_score
from pimm.time import EMITTED_WALL, EMITTED_WORLD, Time
from positronic import keys
from positronic.cli.eval.run import SCORE_LINE
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.eval import keys as eval_keys

DEFINITION = public_eval(EvalRef(SMOKE))
TRIALS = [trial[MOLMO_EPISODE_INDEX_KEY] for task in DEFINITION.tasks for trial in task.trials]


def _record(recording, episodes: list[tuple[JsonValue, bool | None]]) -> None:
    """Record one episode per (trial, success) pair. A success of None records no outcome, so no labeler labels it."""
    with LocalDatasetWriter(recording) as writer:
        for trial, success in episodes:
            with writer.new_episode() as episode:
                episode.set_static(keys.TASK, 'pick up the cup.')
                episode.set_static(MOLMO_EPISODE_INDEX_KEY, trial)
                if success is not None:
                    episode.set_static(eval_keys.SUCCESS, success)
                    episode.set_static(eval_keys.TERMINATED, True)
                for world_ns in (0, 1_000_000_000):
                    episode.append(keys.GRIP, 0.0, Time(**{EMITTED_WORLD: world_ns, EMITTED_WALL: world_ns}))


def _write_log(output, *lines: str) -> None:
    (output / LOG_NAME).write_text(''.join(f'{line}\n' for line in lines))


def test_a_run_of_every_trial_scores_its_successes(tmp_path):
    _record(tmp_path, [(trial, i % 2 == 0) for i, trial in enumerate(TRIALS)])

    scores = recorded_score(DEFINITION, tmp_path)

    assert scores == Scores(primary=3 / 5, episodes=5, unscored=0)


@pytest.mark.parametrize('trials', [TRIALS[:-1], [*TRIALS, TRIALS[0]], [*TRIALS[:-1], 99]])
def test_a_run_that_misses_or_repeats_a_trial_fails(tmp_path, trials: list[JsonValue]):
    _record(tmp_path, [(trial, True) for trial in trials])

    with pytest.raises(ValueError, match='recorded trials'):
        recorded_score(DEFINITION, tmp_path)


def test_a_run_with_an_episode_that_no_labeler_labels_fails(tmp_path):
    _record(tmp_path, [(trial, None if trial == TRIALS[2] else True) for trial in TRIALS])

    with pytest.raises(ValueError, match='no labeler labels'):
        recorded_score(DEFINITION, tmp_path)


def test_a_log_that_printed_the_recorded_score_passes(tmp_path, capsys):
    _record(tmp_path / RECORDING_DIR, [(trial, True) for trial in TRIALS])
    _write_log(tmp_path, 'INFO an earlier line', SCORE_LINE.format(primary=1.0, episodes=5, unscored=0), 'INFO closed')

    check_run(tmp_path, SMOKE)

    assert capsys.readouterr().out.startswith('PASS: ')


@pytest.mark.parametrize(
    'lines',
    [
        (),
        (SCORE_LINE.format(primary=0.8, episodes=5, unscored=0),),
        (SCORE_LINE.format(primary=1.0, episodes=5, unscored=0),) * 2,
    ],
)
def test_a_log_without_the_recorded_score_fails(tmp_path, lines: tuple[str, ...]):
    _write_log(tmp_path, *lines)

    with pytest.raises(ValueError, match='the recorded episodes score'):
        check_log(tmp_path / LOG_NAME, Scores(primary=1.0, episodes=5, unscored=0))
