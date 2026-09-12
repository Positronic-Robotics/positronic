"""Franka Desk operations: read the control box's state, reboot it, clear recoverable safety errors.

Desk's web API is the only way to reach the brakes, FCI, the robot control token and the TD2 safety
self-test; `positronic_franka.desk.Desk` speaks that API, and the arm driver (`franka.py`) drives it
inline for the duration of a run. The three operations here are the out-of-band half — the ones an
operator or a tool runs on its own, when the control box needs attention rather than the arm needs
driving:

- `read_state` is passive. It authenticates and reads. It never takes the control token, so it is
  safe to call while another process is driving the arm.
- `reboot` restarts the control box. The reboot deactivates FCI, locks the brakes and drops the
  control token, which is what makes it the recovery path for a token stranded by a crashed session
  and for a `SafetyError` Desk refuses to clear any other way. It commands no arm motion.
- `acknowledge_errors` takes the control token, acknowledges the recoverable safety errors Desk
  reports and runs the TD2 self-test, then releases the token on the way out — including when the
  self-test fails. It needs the token, so nothing else may be holding control while it runs.

Credentials reach this module only as arguments or through the environment (`desk_from_env`), and
never appear in a returned value, a log line or an exception message.

From the shell::

    positronic-franka-desk state --host=<desk-host>       # JSON snapshot on stdout
    positronic-franka-desk reboot --host=<desk-host>
    positronic-franka-desk ack_errors --host=<desk-host>

`FRANKA_DESK_USER` and `FRANKA_DESK_PASSWORD` carry the Desk login for all three.
"""

import dataclasses
import json
import os
from types import TracebackType
from typing import Protocol

import configuronic as cfn
import pos3

from positronic.drivers import vendor_import
from positronic.utils.logging import init_logging

# Brake state Desk reports per joint, and the number of joints a Franka arm has. `parked` requires a
# reading of exactly that length: `all()` over an empty or truncated read is vacuously true, which
# would report a moving arm as mechanically held.
_BRAKE_LOCKED = 'Locked'
_JOINT_COUNT = 7


class DeskClient(Protocol):
    """The `positronic_franka.desk.Desk` surface these operations drive.

    Typed structurally rather than against the class itself so this module — and its tests — import
    without `positronic-franka`, which ships only in the Linux-only `hardware` extra.

    `_authenticate` and `_token_state` are private API of that pinned dependency, declared here and
    called in `read_state` alone, so a version bump that renames them breaks in one obvious place.
    Desk exposes no public equivalent: authentication is otherwise done only by the context manager,
    which also takes the control token, and the token state is where FCI and the active token live.
    """

    def safety_status(self) -> dict: ...

    def reboot(self, wait: bool = False) -> None: ...

    def run_self_test(self) -> None: ...

    def _authenticate(self) -> None: ...

    def _token_state(self) -> dict: ...

    def __enter__(self) -> 'DeskClient': ...

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None: ...


@dataclasses.dataclass(frozen=True)
class DeskState:
    """What the control box reports about itself at one moment.

    :param brakes: Per-joint brake state, in the order Desk reports it (`'Locked'` / `'Unlocked'`).
    :param safety_status: Safety controller state, e.g. `'Work'`, `'Recovery'`, `'SafetyError'`.
    :param control_held: Whether a session holds the robot control token. While it is held, taking
        control elsewhere is refused, so `acknowledge_errors` cannot run.
    :param fci_active: Whether FCI is up, i.e. whether libfranka can connect and drive the arm.
    :param recoverable_errors: Errors Desk flags as recoverable, which `acknowledge_errors` clears.
        An unrecoverable fault shows up as a `safety_status` of `'SafetyError'` instead, and needs a
        reboot.
    """

    brakes: tuple[str, ...]
    safety_status: str
    control_held: bool
    fci_active: bool
    recoverable_errors: tuple[str, ...]

    @property
    def parked(self) -> bool:
        """Whether every joint brake is locked, so the arm is mechanically held and cannot move.

        False for a reading that does not cover the whole arm: a truncated or empty brake list is
        an incomplete answer, not evidence that the arm is safe.
        """
        return len(self.brakes) == _JOINT_COUNT and all(brake == _BRAKE_LOCKED for brake in self.brakes)


def read_state(desk: DeskClient) -> DeskState:
    """Read the control box's state, without taking the robot control token.

    Passive: it authenticates and reads, and commands nothing. Safe to call while another session
    holds control and drives the arm — that session shows up as `control_held`.

    :param desk: Desk client for the control box to read.
    """
    # `_authenticate` and `_token_state` are private API of positronic-franka; see `DeskClient`.
    # Authenticating here, rather than through the context manager, is what keeps this read passive:
    # entering the context manager would also take the control token away from whoever holds it.
    desk._authenticate()
    status = desk.safety_status()
    token = desk._token_state()
    return DeskState(
        brakes=tuple(status['brakeState']),
        safety_status=status['safetyControllerStatus'],
        control_held=token['activeToken'] is not None,
        fci_active=bool(token['fciActive']),
        recoverable_errors=tuple(sorted(flag for flag, active in status['recoverableErrors'].items() if active)),
    )


def reboot(desk: DeskClient) -> None:
    """Reboot the control box and block until it is back and its safety controller has settled.

    The reboot deactivates FCI, locks the brakes and drops the control token, which recovers a token
    stranded by a crashed session and a `SafetyError` Desk refuses to clear while it stands. It
    commands no arm motion. The box is unreachable for roughly 40 seconds in the middle; this raises
    `TimeoutError` if it never goes down or never settles afterwards.

    :param desk: Desk client for the control box to reboot.
    """
    desk.reboot(wait=True)


def acknowledge_errors(desk: DeskClient) -> None:
    """Acknowledge the recoverable safety errors Desk reports and run the TD2 safety self-test.

    Takes the robot control token for the duration and releases it on the way out, including when
    the self-test fails, so a failure never strands control. Nothing else may be holding control:
    Desk grants the token to one session at a time and refuses to hand it over while it is held.

    :param desk: Desk client for the control box to clear.
    """
    with desk:
        desk.run_self_test()


def desk_from_env(
    host: str, user_env: str = 'FRANKA_DESK_USER', password_env: str = 'FRANKA_DESK_PASSWORD'
) -> DeskClient:
    """Build a Desk client for `host`, reading the login and password from the environment.

    Credentials stay in the environment so they never reach a command line, which processes on the
    same machine can read and which tooling tends to record verbatim.

    :param host: Hostname or address of the control box's Desk web interface.
    :param user_env: Environment variable holding the Desk login.
    :param password_env: Environment variable holding the Desk password.
    """
    login, password = os.environ.get(user_env), os.environ.get(password_env)
    if not (login and password):
        missing = ' and '.join(name for name, value in ((user_env, login), (password_env, password)) if not value)
        raise RuntimeError(f'{missing} not set in the environment; Desk operations need Desk credentials.')
    # Deferred: positronic-franka lives in the Linux-only `hardware` extra and only this constructor
    # needs the concrete class, everything else being typed against `DeskClient`. Importing it at
    # module level would make the module, and its tests, unimportable without that extra.
    with vendor_import('positronic_franka', 'Franka support', platforms=('linux',)):
        from positronic_franka.desk import Desk  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    return Desk(host, login, password)


def state_as_json(state: DeskState) -> str:
    """Render a `DeskState` as JSON, with `parked` alongside the fields it is derived from."""
    return json.dumps({**dataclasses.asdict(state), 'parked': state.parked}, indent=2, sort_keys=True)


def print_state(host: str) -> None:
    """Print the control box's state as JSON on stdout. Takes no control, so it is safe during a run."""
    print(state_as_json(read_state(desk_from_env(host))))


def reboot_control_box(host: str) -> None:
    """Reboot the control box, waiting for it to come back. Locks the brakes; commands no arm motion."""
    reboot(desk_from_env(host))


def acknowledge_safety_errors(host: str) -> None:
    """Clear recoverable safety errors by running the TD2 self-test. Nothing else may hold control."""
    acknowledge_errors(desk_from_env(host))


# Console entry point for [project.scripts]. Every command fails loudly, so a failed operation
# leaves a non-zero exit status behind.
@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({
        'state': cfn.Config(print_state),
        'reboot': cfn.Config(reboot_control_box),
        'ack_errors': cfn.Config(acknowledge_safety_errors),
    })


if __name__ == '__main__':
    _internal_main()
