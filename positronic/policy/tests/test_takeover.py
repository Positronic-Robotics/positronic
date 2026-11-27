import pytest

from positronic.policy.base import Commands
from positronic.policy.codecs import PolicyCommandError
from positronic.policy.keys import HELPER_HAND_BACK, HELPER_TAKE_OVER
from positronic.policy.record import Event, EventKind, StepRecord
from positronic.policy.takeover import Takeover, helper_commands

MOTOR = 'motor'
POLICY = {MOTOR: 1}
HELPER = {MOTOR: 2}
TAKE_OVER = Event(kind=EventKind.SWITCH, from_='policy', to='teleoperator', cause=HELPER_TAKE_OVER)
HAND_BACK = Event(kind=EventKind.SWITCH, from_='teleoperator', to='policy', cause=HELPER_HAND_BACK)


def _element(take_over: int | None = None, hand_back: int | None = None) -> dict:
    return {**HELPER, HELPER_TAKE_OVER: take_over, HELPER_HAND_BACK: hand_back}


def _steps(takeover: Takeover, *elements: dict) -> list[tuple[Commands, StepRecord]]:
    return [takeover.step(POLICY, element) for element in elements]


def test_the_policy_drives_from_the_start_and_the_record_keeps_no_shadow_action():
    takeover = Takeover()
    commands, record = takeover.step(POLICY, _element())
    assert commands == POLICY
    assert record == StepRecord('policy', frozenset({'policy'}))
    assert takeover.control() == {'source': 'policy', 'event': None}


def test_take_over_gives_the_rig_the_helper_commands_in_the_step_of_the_press_and_shadows_the_policy():
    takeover = Takeover()
    _, (commands, record) = _steps(takeover, _element(), _element(take_over=5))
    assert commands == HELPER
    assert record == StepRecord('teleoperator', frozenset({'teleoperator'}), (TAKE_OVER,), {'policy': POLICY})
    assert takeover.control() == {
        'source': 'teleoperator',
        'event': {'kind': 'switch', 'from': 'policy', 'to': 'teleoperator', 'cause': 'take_over'},
    }


def test_hand_back_gives_the_rig_back_to_the_policy_in_the_step_of_the_press():
    takeover = Takeover()
    *_, (commands, record) = _steps(takeover, _element(), _element(take_over=5), _element(take_over=5, hand_back=9))
    assert commands == POLICY
    assert record == StepRecord('policy', frozenset({'policy'}), (HAND_BACK,))


def test_the_presses_of_the_first_element_are_not_new():
    takeover = Takeover()
    (commands, record), (later, _) = _steps(takeover, _element(take_over=5), _element(take_over=5))
    assert commands == later == POLICY
    assert record.events == ()


def test_an_element_with_no_press_field_keeps_the_last_presses():
    takeover = Takeover()
    steps = _steps(takeover, _element(take_over=5), HELPER, _element(take_over=5), {}, _element(take_over=7))
    assert [record.source for _, record in steps] == ['policy', 'policy', 'policy', 'policy', 'teleoperator']


def test_while_the_helper_drives_an_element_with_no_commands_gives_the_rig_none():
    takeover = Takeover()
    *_, (commands, record) = _steps(takeover, _element(), _element(take_over=5), {})
    assert commands == {}
    assert record.source == 'teleoperator'
    assert record.events == ()


def test_a_press_for_the_source_that_drives_switches_nothing():
    takeover = Takeover()
    steps = _steps(takeover, _element(), _element(hand_back=3), _element(take_over=5, hand_back=3))
    steps += _steps(takeover, _element(take_over=6, hand_back=3))
    assert [record.source for _, record in steps] == ['policy', 'policy', 'teleoperator', 'teleoperator']
    assert [record.events for _, record in steps] == [(), (), (TAKE_OVER,), ()]


@pytest.mark.parametrize('take_over, hand_back, source', [(5, 9, 'policy'), (9, 5, 'teleoperator'), (5, 5, 'policy')])
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
