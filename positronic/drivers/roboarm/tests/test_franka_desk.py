import json
from types import TracebackType

import pytest

from positronic.drivers.roboarm.franka_desk import (
    DeskState,
    acknowledge_errors,
    desk_from_env,
    read_state,
    reboot,
    state_as_json,
)

LOCKED = ('Locked',) * 7
UNLOCKED = ('Unlocked',) * 7


def safety_status(
    brakes: tuple[str, ...] = LOCKED, status: str = 'Work', recoverable: dict[str, bool] | None = None
) -> dict:
    """A `Desk.safety_status()` reading, shaped as the Desk web API returns it."""
    return {
        'brakeState': list(brakes),
        'safetyControllerStatus': status,
        'recoverableErrors': {'td2Timeout': False, 'genericJointError': False, **(recoverable or {})},
    }


class FakeDesk:
    """A Desk that answers from canned readings and records what was asked of it.

    Entering it stands for taking the robot control token, as the real context manager does, so a
    test can assert an operation left control alone.
    """

    def __init__(self, status: dict | None = None, active_token: dict | None = None, fci_active: bool = False):
        self._status = status if status is not None else safety_status()
        self._token = {'activeToken': active_token, 'fciActive': fci_active}
        self.calls: list[str] = []
        self.self_test_error: Exception | None = None
        self.control_held = False

    def safety_status(self) -> dict:
        self.calls.append('safety_status')
        return self._status

    def reboot(self, wait: bool = False) -> None:
        self.calls.append(f'reboot(wait={wait})')

    def run_self_test(self) -> None:
        self.calls.append('run_self_test')
        if self.self_test_error is not None:
            raise self.self_test_error

    def _authenticate(self) -> None:
        self.calls.append('_authenticate')

    def _token_state(self) -> dict:
        self.calls.append('_token_state')
        return self._token

    def __enter__(self) -> 'FakeDesk':
        self.calls.append('take_control')
        self.control_held = True
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.calls.append('release_control')
        self.control_held = False


def test_read_state_maps_a_full_reading():
    desk = FakeDesk(
        status=safety_status(brakes=UNLOCKED, status='Recovery', recoverable={'td2Timeout': True}),
        active_token={'id': 7},
        fci_active=True,
    )

    state = read_state(desk)

    assert state == DeskState(
        brakes=UNLOCKED,
        safety_status='Recovery',
        control_held=True,
        fci_active=True,
        recoverable_errors=('td2Timeout',),
    )
    assert not state.parked


def test_read_state_reports_an_idle_box():
    state = read_state(FakeDesk())

    assert state == DeskState(
        brakes=LOCKED, safety_status='Work', control_held=False, fci_active=False, recoverable_errors=()
    )
    assert state.parked


def test_read_state_never_takes_control():
    desk = FakeDesk(active_token={'id': 3})

    read_state(desk)

    assert 'take_control' not in desk.calls
    assert 'release_control' not in desk.calls
    assert desk.control_held is False


@pytest.mark.parametrize('brakes', [(), ('Locked',), ('Locked',) * 6, ('Locked',) * 8])
def test_parked_is_false_on_a_partial_reading(brakes: tuple[str, ...]):
    """An `all()` over an empty or truncated brake list is vacuously true; `parked` must not be."""
    assert not read_state(FakeDesk(safety_status(brakes=brakes))).parked


@pytest.mark.parametrize('unlocked_joint', [0, 3, 6])
def test_parked_is_false_when_a_joint_is_unlocked(unlocked_joint: int):
    brakes: list[str] = list(LOCKED)
    brakes[unlocked_joint] = 'Unlocked'

    assert not read_state(FakeDesk(safety_status(brakes=tuple(brakes)))).parked


def test_recoverable_errors_lists_only_the_flagged_ones():
    desk = FakeDesk(safety_status(recoverable={'td2Timeout': True, 'genericJointError': False, 'other': True}))

    assert read_state(desk).recoverable_errors == ('other', 'td2Timeout')


def test_acknowledge_errors_runs_the_self_test_under_control():
    desk = FakeDesk()

    acknowledge_errors(desk)

    assert desk.calls == ['take_control', 'run_self_test', 'release_control']
    assert desk.control_held is False


def test_acknowledge_errors_releases_control_when_the_self_test_fails():
    desk = FakeDesk()
    desk.self_test_error = TimeoutError('TD2 self-test did not complete')

    with pytest.raises(TimeoutError):
        acknowledge_errors(desk)

    assert desk.calls == ['take_control', 'run_self_test', 'release_control']
    assert desk.control_held is False


def test_reboot_waits_for_the_box_to_come_back():
    desk = FakeDesk()

    reboot(desk)

    assert desk.calls == ['reboot(wait=True)']


def test_state_as_json_carries_parked_alongside_the_fields():
    payload = json.loads(state_as_json(read_state(FakeDesk())))

    assert payload == {
        'brakes': list(LOCKED),
        'safety_status': 'Work',
        'control_held': False,
        'fci_active': False,
        'recoverable_errors': [],
        'parked': True,
    }


@pytest.mark.parametrize(
    ('user', 'password', 'missing'),
    [(None, 'pw', 'DESK_USER'), ('user', None, 'DESK_PASSWORD'), (None, None, 'DESK_USER and DESK_PASSWORD')],
)
def test_desk_from_env_names_the_missing_variables(
    monkeypatch: pytest.MonkeyPatch, user: str | None, password: str | None, missing: str
):
    for name, value in (('DESK_USER', user), ('DESK_PASSWORD', password)):
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=missing):
        desk_from_env('desk.invalid', user_env='DESK_USER', password_env='DESK_PASSWORD')
