import numpy as np
import pytest

from positronic.drivers.roboarm.command import JointDelta
from positronic.keys import ROBOT_COMMAND
from positronic.policy.base import Commands
from positronic.policy.codecs import PolicyCommandError
from positronic.policy.keys import HELPER_HAND_BACK, HELPER_TAKE_OVER, POLICY_SOURCE, TELEOPERATOR
from positronic.policy.record import Event, EventKind, StepRecord
from positronic.policy.takeover import Takeover, helper_commands

MOTOR = 'motor'
ARM = ROBOT_COMMAND
POLICY = {MOTOR: 1}
HELPER = {MOTOR: 2}
TAKE_OVER = Event(kind=EventKind.SWITCH, from_=POLICY_SOURCE, to=TELEOPERATOR, cause=HELPER_TAKE_OVER)
HAND_BACK = Event(kind=EventKind.SWITCH, from_=TELEOPERATOR, to=POLICY_SOURCE, cause=HELPER_HAND_BACK)


def _element(take_over: int | None = None, hand_back: int | None = None) -> dict:
    return {**HELPER, HELPER_TAKE_OVER: take_over, HELPER_HAND_BACK: hand_back}


def _steps(takeover: Takeover, *elements: dict) -> list[tuple[Commands, StepRecord]]:
    return [takeover.step(POLICY, element) for element in elements]


def test_the_policy_drives_from_the_start_and_the_record_keeps_no_shadow_action():
    takeover = Takeover()
    commands, record = takeover.step(POLICY, _element())
    assert commands == POLICY
    assert record == StepRecord(POLICY_SOURCE, frozenset({POLICY_SOURCE}))
    assert takeover.control() == {'source': POLICY_SOURCE, 'event': None}


def test_take_over_gives_the_rig_the_helper_commands_in_the_step_of_the_press_and_shadows_the_policy():
    takeover = Takeover()
    _, (commands, record) = _steps(takeover, _element(), _element(take_over=5))
    assert commands == HELPER
    assert record == StepRecord(TELEOPERATOR, frozenset({TELEOPERATOR}), (TAKE_OVER,), {POLICY_SOURCE: POLICY})
    assert takeover.control() == {
        'source': TELEOPERATOR,
        'event': {'kind': EventKind.SWITCH, 'from': POLICY_SOURCE, 'to': TELEOPERATOR, 'cause': HELPER_TAKE_OVER},
    }


def test_hand_back_gives_the_rig_back_to_the_policy_in_the_step_of_the_press():
    takeover = Takeover()
    *_, (commands, record) = _steps(takeover, _element(), _element(take_over=5), _element(take_over=5, hand_back=9))
    assert commands == POLICY
    assert record == StepRecord(POLICY_SOURCE, frozenset({POLICY_SOURCE}), (HAND_BACK,))


def test_the_presses_of_the_first_element_are_not_new():
    takeover = Takeover()
    (commands, record), (later, _) = _steps(takeover, _element(take_over=5), _element(take_over=5))
    assert commands == later == POLICY
    assert record.events == ()


@pytest.mark.parametrize('first', [{}, HELPER], ids=['no element', 'no press field'])
def test_when_the_first_element_has_no_press_field_the_first_later_press_is_new(first):
    takeover = Takeover()
    _, (commands, record) = _steps(takeover, first, _element(take_over=5))
    assert commands == HELPER
    assert record.events == (TAKE_OVER,)


def test_an_element_with_no_press_field_keeps_the_last_presses():
    takeover = Takeover()
    steps = _steps(takeover, _element(take_over=5), HELPER, _element(take_over=5), {}, _element(take_over=7))
    assert [record.source for _, record in steps] == [
        POLICY_SOURCE,
        POLICY_SOURCE,
        POLICY_SOURCE,
        POLICY_SOURCE,
        TELEOPERATOR,
    ]


def test_while_the_helper_drives_a_step_with_no_element_gives_the_rig_the_commands_of_the_newest_element():
    takeover = Takeover()
    *_, (commands, record) = _steps(takeover, _element(), _element(take_over=5), {})
    assert commands == HELPER
    assert record.source == TELEOPERATOR
    assert record.events == ()


def test_a_step_with_no_element_repeats_the_absolute_targets_of_the_newest_element_but_not_its_deltas():
    takeover = Takeover()
    delta = JointDelta(velocities=np.ones(7))
    newest = {**_element(take_over=5), MOTOR: 2, ARM: delta}
    _, (sent, _), (repeated, _) = _steps(takeover, _element(), newest, {})
    assert sent == {MOTOR: 2, ARM: delta}
    assert repeated == {MOTOR: 2}


def test_while_the_helper_drives_an_element_with_only_press_fields_gives_the_rig_no_commands():
    takeover = Takeover()
    *_, (commands, record) = _steps(takeover, _element(), _element(take_over=5), {HELPER_TAKE_OVER: 5}, {})
    assert commands == {}
    assert record.source == TELEOPERATOR


def test_a_press_for_the_source_that_drives_switches_nothing():
    takeover = Takeover()
    steps = _steps(takeover, _element(), _element(hand_back=3), _element(take_over=5, hand_back=3))
    steps += _steps(takeover, _element(take_over=6, hand_back=3))
    assert [record.source for _, record in steps] == [POLICY_SOURCE, POLICY_SOURCE, TELEOPERATOR, TELEOPERATOR]
    assert [record.events for _, record in steps] == [(), (), (TAKE_OVER,), ()]


@pytest.mark.parametrize(
    'take_over, hand_back, source', [(5, 9, POLICY_SOURCE), (9, 5, TELEOPERATOR), (5, 5, POLICY_SOURCE)]
)
def test_when_both_presses_are_new_the_later_press_wins_and_at_one_time_hand_back_wins(take_over, hand_back, source):
    takeover = Takeover()
    _, (_, record) = _steps(takeover, _element(), _element(take_over, hand_back))
    assert record.source == source


def test_the_policy_with_no_commands_has_no_shadow_action():
    takeover = Takeover()
    takeover.step(POLICY, _element())
    _, record = takeover.step({}, _element(take_over=5))
    assert record.shadow == {}


@pytest.mark.parametrize('press', [1.5, '5', True])
def test_a_press_that_is_not_a_time_fails_the_step(press):
    with pytest.raises(PolicyCommandError, match='press fields'):
        Takeover().step(POLICY, {HELPER_TAKE_OVER: press})


def test_the_commands_of_an_element_leave_out_the_press_fields():
    assert helper_commands(_element(take_over=5)) == HELPER
