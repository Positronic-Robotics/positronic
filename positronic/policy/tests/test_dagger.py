from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from eval_vocabulary.timeline import recording_timeline

from pimm import Time
from positronic import keys
from positronic.dataset import transforms
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers.recording import HARNESS_STEP, HARNESS_WALL, HARNESS_WORLD
from positronic.policy.dagger import DaggerDataset, teleoperator_runs
from positronic.policy.keys import CONTROL_SOURCE, LICENSE, SHADOW, SOURCES, TELEOPERATOR
from positronic.policy.license import LicenseRule
from positronic.policy.record import RecordSamples, StepRecord

ARM = keys.ROBOT_COMMAND
GRIP = keys.TARGET_GRIP
STATE = keys.JOINTS
# The observation names of the rig. The record has ``robot_state.q`` for the observation ``robot_state``.
OBSERVATIONS = [keys.ROBOT_STATE]
# Signals of the record that are not observations of the rig: the ground truth of the simulator.
GROUND_TRUTH = ['sim_state', 'gt_state.objects.cube.pose']
POLICY = 'policy'
VLA = 'vla'
SHADOW_ARM = f'{SHADOW}.{POLICY}.{ARM}'
PERIOD_NS = 20_000_000
TASK = 'task'
ALLOWED = 'apache-2.0'
RULE = LicenseRule({ALLOWED})
# The licenses of the policy and the teleoperator, which the rule allows.
LICENSED = {f'{SOURCES}.{POLICY}.{LICENSE}': ALLOWED, f'{SOURCES}.{TELEOPERATOR}.{LICENSE}': ALLOWED}


def _at(ns: int, step: int) -> Time:
    """The time ``ns`` on the world clock at step ``step``. The wall clock moves at each step, also when the world clock
    does not."""
    return Time(**{HARNESS_WORLD: ns, HARNESS_WALL: ns + 7 + step, HARNESS_STEP: step})


def _policy() -> StepRecord:
    return StepRecord(POLICY, frozenset({POLICY}))


def _teleoperator(shadow: tuple[str, ...] = (POLICY,)) -> StepRecord:
    return StepRecord(TELEOPERATOR, frozenset({TELEOPERATOR}), shadow={name: {ARM: 0.0} for name in shadow})


def _record(
    tmp_path,
    steps: list[StepRecord],
    *,
    grip_missing_at: int | None = None,
    rig: bool = False,
    repeated_at: int | None = None,
    static: Mapping[str, Any] = LICENSED,
):
    """The record with ``steps`` and the static items ``static``, as the harness writes it, added to the dataset at
    ``tmp_path``. Step ``i`` is at ``(i + 1) * PERIOD_NS``. Its observation and ground truth are ``[i]``, its command
    is ``[10 + i]`` with the grip ``i``, and each of its shadow actions is ``[20 + i]``. Step ``grip_missing_at`` sends
    no grip. Step ``repeated_at`` is at the time of the step before it.

    On the virtual clock, each sample of a step has the time of the step. On a ``rig``, the harness reads the
    observation between the step before and the step, and the control record, the command and the grip have later
    times in that order."""
    samples = RecordSamples({ARM: None})
    read, command, grip = (-PERIOD_NS // 2, 1_000, 2_000) if rig else (0, 0, 0)
    with LocalDatasetWriter(tmp_path) as dataset, dataset.new_episode() as episode:
        episode.set_static(TASK, 'lift the cube')
        for name, value in static.items():
            episode.set_static(name, value)
        for i, step in enumerate(steps):
            ns = (i if i == repeated_at else i + 1) * PERIOD_NS
            for name in [STATE, *GROUND_TRUTH]:
                episode.append(name, np.array([float(i)]), _at(ns + read, i))
            shadow = {name: {ARM: np.array([20.0 + i])} for name in step.shadow}
            recorded = StepRecord(step.source, step.uses, events=step.events, shadow=shadow)
            for signal, value, time in samples.of(recorded, _at(ns, i)):
                episode.append(signal, value, time)
            episode.append(ARM, np.array([10.0 + i]), _at(ns + command, i))
            if i != grip_missing_at:
                episode.append(GRIP, float(i), _at(ns + grip, i))
    return LocalDataset(tmp_path)[-1]


def _values(signal) -> list[float]:
    return [float(value[0]) for value in signal.values()]


def _steps(signal) -> list[int]:
    return list(signal.timestamps(HARNESS_STEP))


def test_a_record_that_the_teleoperator_drives_from_the_start_is_one_episode_of_each_observation_and_command(tmp_path):
    record = _record(tmp_path, [_teleoperator(shadow=())] * 3)

    [run] = teleoperator_runs(record, OBSERVATIONS, [ARM])

    episode = run.episode
    assert _values(episode[STATE]) == [0.0, 1.0, 2.0]
    assert _values(episode[ARM]) == [10.0, 11.0, 12.0]
    assert _steps(episode[STATE]) == _steps(episode[ARM]) == [0, 1, 2]
    assert episode[TASK] == 'lift the cube'
    assert CONTROL_SOURCE not in episode
    assert not set(GROUND_TRUTH) & set(episode)
    assert episode.meta['uid'] == record.meta['uid']
    assert run.steps_without_shadow == {}


def test_two_teleoperator_runs_split_by_policy_steps_are_two_episodes_with_the_shadow_action_of_each_step(tmp_path):
    steps = [_teleoperator(), _teleoperator(), _policy(), _policy(), _teleoperator()]
    record = _record(tmp_path, steps)

    first, second = teleoperator_runs(record, OBSERVATIONS, [ARM])

    assert dict(DaggerDataset(LocalDataset(tmp_path), OBSERVATIONS, [ARM], RULE).steps_without_shadow) == {POLICY: 0}
    assert _values(first.episode[ARM]) == [10.0, 11.0]
    assert _values(first.episode[SHADOW_ARM]) == [20.0, 21.0]
    assert _steps(first.episode[SHADOW_ARM]) == [0, 1]
    assert _values(second.episode[STATE]) == [4.0]
    assert _values(second.episode[ARM]) == [14.0]
    assert _values(second.episode[SHADOW_ARM]) == [24.0]
    assert first.steps_without_shadow == second.steps_without_shadow == {POLICY: 0}


def test_a_step_with_no_shadow_action_stays_with_no_shadow_action_and_counts(tmp_path):
    steps = [_teleoperator(), _teleoperator(shadow=()), _teleoperator()]
    record = _record(tmp_path, steps)

    [run] = teleoperator_runs(record, OBSERVATIONS, [ARM])

    assert _values(run.episode[ARM]) == [10.0, 11.0, 12.0]
    assert _steps(run.episode[SHADOW_ARM]) == [0, 2]
    assert _values(run.episode[SHADOW_ARM]) == [20.0, 22.0]
    assert run.steps_without_shadow == {POLICY: 1}


def test_a_run_whose_first_step_has_no_shadow_action_gets_no_shadow_action_of_the_steps_before(tmp_path):
    steps = [_policy(), _teleoperator(), _teleoperator(shadow=()), _policy(), _teleoperator(shadow=())]
    record = _record(tmp_path, steps)

    first, second = teleoperator_runs(record, OBSERVATIONS, [ARM])

    assert _values(first.episode[SHADOW_ARM]) == [21.0]
    assert SHADOW_ARM not in second.episode
    assert first.steps_without_shadow == second.steps_without_shadow == {POLICY: 1}


def test_a_label_with_no_sample_at_a_step_of_a_run_fails_the_read(tmp_path):
    record = _record(tmp_path, [_teleoperator(shadow=())] * 3, grip_missing_at=1)

    with pytest.raises(ValueError, match=f"label '{GRIP}' has no sample at the step {HARNESS_STEP}=1"):
        teleoperator_runs(record, OBSERVATIONS, [ARM, GRIP])


@pytest.mark.parametrize('labels', [[ARM, GRIP], [GRIP, ARM]])
def test_a_record_of_a_rig_where_the_samples_of_a_step_have_different_times_gives_the_samples_of_each_step(
    tmp_path, labels
):
    record = _record(tmp_path, [_policy(), _teleoperator(), _teleoperator()], rig=True)

    [run] = teleoperator_runs(record, OBSERVATIONS, labels)

    episode = run.episode
    assert _values(episode[ARM]) == [11.0, 12.0]
    assert list(episode[GRIP].values()) == [1.0, 2.0]
    assert _values(episode[STATE]) == [1.0, 2.0]
    assert _values(episode[SHADOW_ARM]) == [21.0, 22.0]
    # Each signal has the time of the grip, the last sample of each step.
    for name in (ARM, GRIP, STATE, SHADOW_ARM):
        assert _steps(episode[name]) == [1, 2], name
        assert list(episode[name].timestamps(HARNESS_WORLD)) == [2 * PERIOD_NS + 2_000, 3 * PERIOD_NS + 2_000], name


# With the grip as the only label, the last step sends no label, and only `control.source` has two samples at one time.
@pytest.mark.parametrize(('labels', 'grip_missing_at'), [([ARM], None), ([GRIP], 3)])
def test_a_record_with_two_steps_at_one_time_gives_the_samples_of_each_step(tmp_path, labels, grip_missing_at):
    steps = [_policy(), _teleoperator(), _teleoperator(), _policy()]
    record = _record(tmp_path, steps, repeated_at=2, grip_missing_at=grip_missing_at)

    [run] = teleoperator_runs(record, OBSERVATIONS, labels)

    episode = run.episode
    assert _values(episode[STATE]) == [1.0, 2.0]
    assert _values(episode[SHADOW_ARM]) == [21.0, 22.0]
    assert _steps(episode[labels[0]]) == _steps(episode[STATE]) == [1, 2]
    assert list(episode[labels[0]].timestamps(HARNESS_WORLD)) == [2 * PERIOD_NS, 2 * PERIOD_NS]
    if labels == [ARM]:
        assert _values(episode[ARM]) == [11.0, 12.0]
    else:
        assert list(episode[GRIP].values()) == [1.0, 2.0]


def test_a_codec_reads_a_training_episode_on_harness_world_and_keeps_the_last_of_two_steps_at_one_time(tmp_path):
    record = _record(tmp_path, [_policy(), _teleoperator(), _teleoperator(), _policy()], repeated_at=2)
    [run] = teleoperator_runs(record, OBSERVATIONS, [ARM, GRIP])
    episode = run.episode

    signals = [episode[ARM], episode[GRIP], episode[STATE]]
    timeline = recording_timeline({name for signal in signals for name in signal.timelines})
    encoded = transforms.concat(*signals, timelines=(timeline,))

    assert timeline == HARNESS_WORLD
    assert [value.tolist() for value in encoded.values()] == [[12.0, 2.0, 2.0]]
    assert list(encoded.timestamps(HARNESS_WORLD)) == [2 * PERIOD_NS]


def test_a_record_with_no_step_numbers_fails_the_read(tmp_path):
    with LocalDatasetWriter(tmp_path) as dataset, dataset.new_episode() as episode:
        episode.append(CONTROL_SOURCE, TELEOPERATOR, Time(**{HARNESS_WORLD: PERIOD_NS}))
        episode.append(ARM, np.array([10.0]), Time(**{HARNESS_WORLD: PERIOD_NS}))

    before = 'a harness wrote it before it numbered the steps'
    with pytest.raises(ValueError, match=f'no step numbers on {HARNESS_STEP}: {before}'):
        teleoperator_runs(LocalDataset(tmp_path)[-1], OBSERVATIONS, [ARM])


def test_the_steps_without_shadow_count_for_each_source_in_shadow(tmp_path):
    steps = [_teleoperator((POLICY, VLA)), _teleoperator((POLICY,)), _teleoperator((POLICY,)), _teleoperator((VLA,))]
    record = _record(tmp_path, steps)

    [run] = teleoperator_runs(record, OBSERVATIONS, [ARM])

    assert run.steps_without_shadow == {POLICY: 1, VLA: 2}
    assert _steps(run.episode[f'{SHADOW}.{VLA}.{ARM}']) == [0, 3]


def test_a_record_where_the_teleoperator_drives_no_step_has_no_episode(tmp_path):
    records = tmp_path / 'records'
    _record(records, [])
    # The first label is a channel that only the teleoperator sends, so a record with no teleoperator step has none.
    _record(records, [_policy(), _policy()])

    assert [teleoperator_runs(record, OBSERVATIONS, [f'{ARM}.pose']) for record in LocalDataset(records)] == [[], []]


def test_the_dagger_dataset_holds_the_runs_of_each_record_and_writes_as_a_positronic_dataset(tmp_path):
    records = tmp_path / 'records'
    _record(records, [_teleoperator(), _policy(), _teleoperator(shadow=())])
    _record(records, [_policy(), _teleoperator(shadow=()), _teleoperator()])
    _record(records, [_teleoperator(), _teleoperator(shadow=())])
    data = DaggerDataset(LocalDataset(records), OBSERVATIONS, [ARM], RULE)

    assert [_values(episode[ARM]) for episode in data] == [[10.0], [12.0], [11.0, 12.0], [10.0, 11.0]]
    assert dict(data.steps_without_shadow) == {POLICY: 3}
    with LocalDatasetWriter(tmp_path / 'training') as dataset:
        for episode in data:
            with dataset.new_episode() as written:
                for name, value in episode.static.items():
                    written.set_static(name, value)
                for name, signal in episode.signals.items():
                    for value, time in signal:
                        written.append(name, value, time)
    for episode, written in zip(data, LocalDataset(tmp_path / 'training'), strict=True):
        assert set(written) == set(episode)
        for name, signal in episode.signals.items():
            for timeline in (HARNESS_STEP, HARNESS_WORLD):
                assert list(written[name].timestamps(timeline)) == list(signal.timestamps(timeline))
            np.testing.assert_array_equal(written[name].values(), signal.values())


def test_the_dagger_dataset_skips_each_record_that_the_license_rule_drops_and_counts_it(tmp_path):
    records = tmp_path / 'records'
    _record(records, [_policy(), _teleoperator()])
    _record(
        records,
        [_policy(), _teleoperator(), _teleoperator()],
        static={**LICENSED, f'{SOURCES}.{POLICY}.{LICENSE}': 'gpl-3.0'},
    )
    _record(records, [_teleoperator(), _teleoperator(), _teleoperator()], static={})
    _record(records, [_teleoperator(), _teleoperator(shadow=())])

    data = DaggerDataset(LocalDataset(records), OBSERVATIONS, [ARM], RULE)

    assert [_values(episode[ARM]) for episode in data] == [[11.0], [10.0, 11.0]]
    assert data.dropped_records == 2
    assert dict(data.steps_without_shadow) == {POLICY: 1}
