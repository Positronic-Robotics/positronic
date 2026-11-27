import json

import numpy as np
import pytest

from pimm import Time
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers.recording import HARNESS_STEP
from positronic.policy.codecs import PolicyCommandError
from positronic.policy.keys import CONTROL_EVENT, CONTROL_SOURCE, CONTROL_USES
from positronic.policy.record import Event, EventKind, RecordSamples, StepRecord, decode_uses, encode_uses

ARM = 'robot_command'
GRIP = 'target_grip'
# A serializer that splits a command into suffixed fields, as the pose serializers do.
SERIALIZERS = {ARM: lambda command: {'.q': np.asarray(command), '.empty': None}, GRIP: None}
TAKE_OVER = Event(kind=EventKind.SWITCH, from_='policy', to='teleoperator', cause='take_over')
HAND_BACK = Event(kind=EventKind.SWITCH, from_='teleoperator', to='policy', cause='hand_back')


def _at(ns: int) -> Time:
    return Time(world=ns, wall=ns + 7)


def test_the_sources_that_a_step_uses_include_its_driver():
    with pytest.raises(ValueError, match='include its driver'):
        StepRecord('teleoperator', frozenset({'policy'}))


def test_the_driver_has_no_shadow_actions():
    with pytest.raises(ValueError, match='no shadow actions'):
        StepRecord('policy', frozenset({'policy'}), shadow={'policy': {GRIP: 0.5}})


def test_an_event_encodes_as_one_json_object_with_only_the_parts_that_it_has():
    assert json.loads(TAKE_OVER.encode()) == {
        'kind': 'switch',
        'from': 'policy',
        'to': 'teleoperator',
        'cause': 'take_over',
    }
    assert json.loads(Event(kind=EventKind.SWITCH).encode()) == {'kind': 'switch'}
    assert Event.model_validate_json(TAKE_OVER.encode()) == TAKE_OVER


def test_the_uses_encode_as_one_sorted_json_list_and_decode_back():
    uses = frozenset({'teleoperator', 'policy'})
    assert json.loads(encode_uses(uses)) == ['policy', 'teleoperator']
    assert decode_uses(encode_uses(uses)) == uses


def test_a_control_value_goes_in_once_and_again_only_when_it_changes():
    samples = RecordSamples(SERIALIZERS)
    first = samples.of(StepRecord('policy', frozenset({'policy'})), _at(10))
    assert [(signal, value) for signal, value, _ in first] == [(CONTROL_SOURCE, 'policy'), (CONTROL_USES, '["policy"]')]
    assert samples.of(StepRecord('policy', frozenset({'policy'})), _at(20)) == []
    switched = samples.of(StepRecord('teleoperator', frozenset({'teleoperator'}), events=(TAKE_OVER,)), _at(30))
    assert [(signal, value) for signal, value, _ in switched] == [
        (CONTROL_SOURCE, 'teleoperator'),
        (CONTROL_USES, '["teleoperator"]'),
        (CONTROL_EVENT, TAKE_OVER.encode()),
    ]


def test_the_second_event_of_a_step_comes_1_ns_after_the_first_on_each_clock_at_the_same_step():
    samples = RecordSamples(SERIALIZERS)
    at = Time(world=100, wall=107, **{HARNESS_STEP: 4})
    step = samples.of(StepRecord('policy', frozenset({'policy'}), events=(TAKE_OVER, HAND_BACK)), at)
    events = [(value, time) for signal, value, time in step if signal == CONTROL_EVENT]
    assert events == [(TAKE_OVER.encode(), at), (HAND_BACK.encode(), Time(world=101, wall=108, **{HARNESS_STEP: 4}))]


def test_a_sample_of_a_later_step_at_the_same_world_time_comes_1_ns_after_the_last_sample_of_its_signal():
    # The harness can run the next step before the world clock moves, so only the wall time is later.
    samples = RecordSamples(SERIALIZERS)
    samples.of(StepRecord('policy', frozenset({'policy'}), events=(TAKE_OVER, HAND_BACK)), _at(100))
    step = samples.of(StepRecord('teleoperator', frozenset({'teleoperator'}), events=(TAKE_OVER,)), _at(100))
    assert [(signal, time) for signal, _, time in step] == [
        (CONTROL_SOURCE, Time(world=101, wall=108)),
        (CONTROL_USES, Time(world=101, wall=108)),
        (CONTROL_EVENT, Time(world=102, wall=109)),
    ]


def test_a_shadow_action_goes_through_the_serializer_of_its_channel():
    samples = RecordSamples(SERIALIZERS)
    record = StepRecord('teleoperator', frozenset({'teleoperator'}), shadow={'policy': {ARM: [0.1, 0.2], GRIP: 0.5}})
    shadow = {signal: value for signal, value, _ in samples.of(record, _at(10)) if signal.startswith('shadow.')}
    assert shadow.keys() == {'shadow.policy.robot_command.q', 'shadow.policy.target_grip'}
    np.testing.assert_array_equal(shadow['shadow.policy.robot_command.q'], [0.1, 0.2])
    assert shadow['shadow.policy.target_grip'] == 0.5


def test_a_shadow_action_on_a_channel_that_the_embodiment_does_not_take_fails_the_step():
    record = StepRecord('teleoperator', frozenset({'teleoperator'}), shadow={'policy': {'wheels': 1.0}})
    with pytest.raises(PolicyCommandError, match="name \\['wheels'\\]"):
        RecordSamples(SERIALIZERS).of(record, _at(10))


def test_the_record_writes_and_reads_back_the_control_signals_and_the_shadow_actions(tmp_path):
    samples = RecordSamples(SERIALIZERS)
    records = [
        (StepRecord('policy', frozenset({'policy'})), 0),
        (StepRecord('teleoperator', frozenset({'teleoperator'}), (TAKE_OVER,), {'policy': {GRIP: 0.5}}), 10),
        (StepRecord('teleoperator', frozenset({'teleoperator'}), (), {'policy': {GRIP: 0.25}}), 20),
        (StepRecord('policy', frozenset({'policy'}), (HAND_BACK,)), 30),
    ]
    with LocalDatasetWriter(tmp_path) as dataset, dataset.new_episode() as episode:
        for record, ns in records:
            for signal, value, time in samples.of(record, _at(ns)):
                episode.append(signal, value, time)
    recorded = LocalDataset(tmp_path)[0]
    assert list(recorded[CONTROL_SOURCE].values()) == ['policy', 'teleoperator', 'policy']
    assert list(recorded[CONTROL_SOURCE].timestamps('world')) == [0, 10, 30]
    events = recorded[CONTROL_EVENT]
    assert [Event.model_validate_json(event) for event in events.values()] == [TAKE_OVER, HAND_BACK]
    assert list(recorded['shadow.policy.target_grip'].values()) == [0.5, 0.25]
