import json

import numpy as np
import pytest

from pimm import Time
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers.recording import HARNESS_STEP, HARNESS_WALL, HARNESS_WORLD
from positronic.keys import JOINTS_SUFFIX, ROBOT_COMMAND, TARGET_GRIP
from positronic.policy.codecs import PolicyCommandError
from positronic.policy.keys import (
    CONTROL_EVENT,
    CONTROL_SOURCE,
    CONTROL_USES,
    HELPER_HAND_BACK,
    HELPER_TAKE_OVER,
    POLICY_SOURCE,
    SHADOW,
    TELEOPERATOR,
)
from positronic.policy.record import Event, EventKind, RecordSamples, StepRecord, decode_uses, encode_uses

# A serializer that splits a command into suffixed fields, as the pose serializers do.
SERIALIZERS = {ROBOT_COMMAND: lambda command: {JOINTS_SUFFIX: np.asarray(command), '.empty': None}, TARGET_GRIP: None}
TAKE_OVER = Event(kind=EventKind.SWITCH, from_=POLICY_SOURCE, to=TELEOPERATOR, cause=HELPER_TAKE_OVER)
HAND_BACK = Event(kind=EventKind.SWITCH, from_=TELEOPERATOR, to=POLICY_SOURCE, cause=HELPER_HAND_BACK)
POLICY_STEP = StepRecord(POLICY_SOURCE, frozenset({POLICY_SOURCE}))
SHADOW_ARM = f'{SHADOW}.{POLICY_SOURCE}.{ROBOT_COMMAND}{JOINTS_SUFFIX}'
SHADOW_GRIP = f'{SHADOW}.{POLICY_SOURCE}.{TARGET_GRIP}'


def _time(world_ns: int, wall_ns: int, step: int | None = None) -> Time:
    steps = {} if step is None else {HARNESS_STEP: step}
    return Time(**{HARNESS_WORLD: world_ns, HARNESS_WALL: wall_ns}, **steps)


def _at(ns: int) -> Time:
    return _time(ns, ns + 7)


def test_the_sources_that_a_step_uses_include_its_driver():
    with pytest.raises(ValueError, match='include its driver'):
        StepRecord(TELEOPERATOR, frozenset({POLICY_SOURCE}))


def test_the_driver_has_no_shadow_actions():
    with pytest.raises(ValueError, match='no shadow actions'):
        StepRecord(POLICY_SOURCE, frozenset({POLICY_SOURCE}), shadow={POLICY_SOURCE: {TARGET_GRIP: 0.5}})


def test_an_event_encodes_as_one_json_object_with_only_the_parts_that_it_has():
    assert json.loads(TAKE_OVER.encode()) == {
        'kind': EventKind.SWITCH,
        'from': POLICY_SOURCE,
        'to': TELEOPERATOR,
        'cause': HELPER_TAKE_OVER,
    }
    assert json.loads(Event(kind=EventKind.SWITCH).encode()) == {'kind': EventKind.SWITCH}
    assert Event.model_validate_json(TAKE_OVER.encode()) == TAKE_OVER


def test_the_uses_encode_as_one_sorted_json_list_and_decode_back():
    uses = frozenset({TELEOPERATOR, POLICY_SOURCE})
    assert json.loads(encode_uses(uses)) == [POLICY_SOURCE, TELEOPERATOR]
    assert decode_uses(encode_uses(uses)) == uses


def _by_signal(samples, part: int) -> dict[str, list]:
    """The value (``part`` 1) or the time (``part`` 2) of each of ``samples``, in order, by signal."""
    parts: dict[str, list] = {}
    for sample in samples:
        parts.setdefault(sample[0], []).append(sample[part])
    return parts


def test_a_control_value_goes_in_once_and_again_only_when_it_changes():
    samples = RecordSamples(SERIALIZERS)
    first = samples.of(POLICY_STEP, _at(10))
    assert _by_signal(first, 1) == {
        CONTROL_SOURCE: [POLICY_SOURCE],
        CONTROL_USES: [encode_uses(frozenset({POLICY_SOURCE}))],
    }
    assert samples.of(POLICY_STEP, _at(20)) == []
    switched = samples.of(StepRecord(TELEOPERATOR, frozenset({TELEOPERATOR}), events=(TAKE_OVER,)), _at(30))
    assert _by_signal(switched, 1) == {
        CONTROL_SOURCE: [TELEOPERATOR],
        CONTROL_USES: [encode_uses(frozenset({TELEOPERATOR}))],
        CONTROL_EVENT: [TAKE_OVER.encode()],
    }


def test_the_second_event_of_a_step_comes_1_ns_after_the_first_on_each_clock_at_the_same_step():
    samples = RecordSamples(SERIALIZERS)
    at = _time(100, 107, step=4)
    step = samples.of(StepRecord(POLICY_SOURCE, frozenset({POLICY_SOURCE}), events=(TAKE_OVER, HAND_BACK)), at)
    events = [(value, time) for signal, value, time in step if signal == CONTROL_EVENT]
    assert events == [(TAKE_OVER.encode(), at), (HAND_BACK.encode(), _time(101, 108, step=4))]


def test_a_sample_of_a_later_step_at_the_same_world_time_comes_1_ns_after_the_last_sample_of_its_signal():
    # The harness can run the next step before the world clock moves, so only the wall time is later.
    samples = RecordSamples(SERIALIZERS)
    samples.of(StepRecord(POLICY_SOURCE, frozenset({POLICY_SOURCE}), events=(TAKE_OVER, HAND_BACK)), _at(100))
    step = samples.of(StepRecord(TELEOPERATOR, frozenset({TELEOPERATOR}), events=(TAKE_OVER,)), _at(100))
    assert _by_signal(step, 2) == {
        CONTROL_SOURCE: [_time(101, 108)],
        CONTROL_USES: [_time(101, 108)],
        CONTROL_EVENT: [_time(102, 109)],
    }


def test_a_shadow_action_goes_through_the_serializer_of_its_channel():
    samples = RecordSamples(SERIALIZERS)
    record = StepRecord(
        TELEOPERATOR, frozenset({TELEOPERATOR}), shadow={POLICY_SOURCE: {ROBOT_COMMAND: [0.1, 0.2], TARGET_GRIP: 0.5}}
    )
    shadow = {signal: value for signal, value, _ in samples.of(record, _at(10)) if signal.startswith(f'{SHADOW}.')}
    assert shadow.keys() == {SHADOW_ARM, SHADOW_GRIP}
    np.testing.assert_array_equal(shadow[SHADOW_ARM], [0.1, 0.2])
    assert shadow[SHADOW_GRIP] == 0.5


def test_a_shadow_action_on_a_channel_that_the_embodiment_does_not_take_fails_the_step():
    record = StepRecord(TELEOPERATOR, frozenset({TELEOPERATOR}), shadow={POLICY_SOURCE: {'wheels': 1.0}})
    with pytest.raises(PolicyCommandError, match="name \\['wheels'\\]"):
        RecordSamples(SERIALIZERS).of(record, _at(10))


def test_the_record_writes_and_reads_back_the_control_signals_and_the_shadow_actions(tmp_path):
    samples = RecordSamples(SERIALIZERS)
    records = [
        (POLICY_STEP, 0),
        (StepRecord(TELEOPERATOR, frozenset({TELEOPERATOR}), (TAKE_OVER,), {POLICY_SOURCE: {TARGET_GRIP: 0.5}}), 10),
        (StepRecord(TELEOPERATOR, frozenset({TELEOPERATOR}), (), {POLICY_SOURCE: {TARGET_GRIP: 0.25}}), 20),
        (StepRecord(POLICY_SOURCE, frozenset({POLICY_SOURCE}), (HAND_BACK,)), 30),
    ]
    with LocalDatasetWriter(tmp_path) as dataset, dataset.new_episode() as episode:
        for record, ns in records:
            for signal, value, time in samples.of(record, _at(ns)):
                episode.append(signal, value, time)
    recorded = LocalDataset(tmp_path)[0]
    assert list(recorded[CONTROL_SOURCE].values()) == [POLICY_SOURCE, TELEOPERATOR, POLICY_SOURCE]
    assert list(recorded[CONTROL_SOURCE].timestamps(HARNESS_WORLD)) == [0, 10, 30]
    events = recorded[CONTROL_EVENT]
    assert [Event.model_validate_json(event) for event in events.values()] == [TAKE_OVER, HAND_BACK]
    assert list(recorded[SHADOW_GRIP].values()) == [0.5, 0.25]
