"""Driver for a Trossen WidowX AI arm the operator holds, read as a teleoperation source.

The operator moves the arm and the controller does nothing but hold its weight: it runs in
``external_effort`` mode, where every joint is back-drivable. What it reads is what a follower is asked to
stand at, joint for joint, with no kinematics in between — which is the whole reason to prefer a leader to
a hand tracked in space. A pose has to be solved for, and near the workspace boundary a solution may not
exist at all; joints always do.

The arm carries the leader end effector, whose fingers are shorter than the follower's. The gripper's
position is computed from the motor angle through that geometry, so a leader read as a follower reports
5.9 mm less than it holds. That is past the 4 mm the controller tolerates below the gripper's range, and
it then refuses position mode outright with ``Joint limit exceeded ... Setting to idle``. Measured on both
leaders of the station: the same arm reads -0.0004 m as a leader and -0.0063 m as a follower.

Force feedback is what the follower is holding, pushed back into the operator's hand. It stays off until
the follower driver publishes its external efforts, which it does not yet; ``force_feedback_gain`` is 0
and ``follower_efforts`` goes unconnected, which leaves plain gravity compensation.

The trigger takes the operator's hand to move, because the gripper joint carries the most friction of any
joint on the arm. What the controller cancels of it is that joint's ``friction_constant_term``, which
``gripper_friction_constant`` stands in for.
"""

import contextlib
import logging
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

import pimm
from positronic.drivers import vendor_import

from . import command

# trossen_arm lives in the `trossen` extra, which the type-check environment does not install.
with vendor_import(
    'trossen_arm', 'Trossen arm support', hint='Re-run with the trossen extra:\n  uv run --locked --extra trossen ...\n'
):
    import trossen_arm  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

_ARM_JOINTS = 6
_GRIPPER_JOINT = 6
# The leader is read at the rate the follower is driven at: a reading the follower never uses is latency
# the operator paid for and got nothing back.
_HZ = 100
_CONNECT_TIMEOUT_S = 20.0
# How often a failure that stands is worth saying again. A tick rate of complaints buries every other line.
_COMPLAIN_EVERY_S = 5.0
# How often the arm is asked back into the mode it should be in, once something has knocked it out.
_RECOVER_EVERY_S = 1.0
# The pace a leader is driven at, and the least time any such move takes. The operator has a hand on the
# arm while it travels, so it goes no faster than the follower does.
_MOVE_SPEED = 0.6  # rad/s
_MIN_MOVE_TIME_S = 1.0


def _connect(ip: str) -> Any:
    """Open the controller of an arm held as a leader, and take ownership of it."""
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_leader, ip, True, _CONNECT_TIMEOUT_S
    )
    return driver


@contextlib.contextmanager
def _gripper_freed(driver: Any, constant: float | None) -> Iterator[None]:
    """The gripper with ``constant`` as its friction constant term, and the arm's own value back after.

    The controller adds this term to the effort it puts on the joint, in the direction the joint moves, so
    a larger one leaves less of the gripper's own friction for the operator's hand to overcome.
    """
    if constant is None:
        yield
        return
    characteristics = driver.get_joint_characteristics()
    stood_at = characteristics[_GRIPPER_JOINT].friction_constant_term
    characteristics[_GRIPPER_JOINT].friction_constant_term = constant
    driver.set_joint_characteristics(characteristics)
    logger.info(f'The gripper friction constant is {constant}, where the arm carries {stood_at}')
    try:
        yield
    finally:
        characteristics[_GRIPPER_JOINT].friction_constant_term = stood_at
        try:
            driver.set_joint_characteristics(characteristics)
        # rules-allow: swallowed-error — an arm that cannot be reached keeps the term until it is
        # configured again, and the handle still has to go back
        except Exception as exc:
            logger.error(f'The gripper keeps the friction constant {constant} until it is configured again: {exc}')


@contextlib.contextmanager
def _opened(connect: Callable[[str], Any], ip: str) -> Iterator[Any]:
    """The arm, left idle and its handle given back however the run ends — including one that never starts."""
    driver = connect(ip)
    try:
        yield driver
    finally:
        try:
            driver.set_all_modes(trossen_arm.Mode.idle)
        # rules-allow: swallowed-error — an arm that cannot be reached cannot be set idle either, and the
        # handle still has to go back
        except trossen_arm.RuntimeError as exc:
            logger.error(f'The leader at {ip} was not set idle: {exc}')
        finally:  # an arm that will not go idle still has a handle to give back
            driver.cleanup()


class Leader(pimm.ControlSystem):
    """One WidowX AI arm the operator holds, publishing the joints and the grip it is being moved to.

    The arm is left free the whole time it runs: there is no tracking to switch on, because nothing is
    driven from here. Whoever reads these ports decides when they reach a follower.
    """

    def __init__(
        self,
        ip: str,
        *,
        force_feedback_gain: float = 0.0,
        gripper_friction_constant: float | None = None,
        connect: Callable[[str], Any] = _connect,
    ) -> None:
        """
        :param ip: Address of the leader arm's controller.
        :param force_feedback_gain: Share of the follower's external effort pushed back into the
            operator's hand. 0 leaves the arm in plain gravity compensation.
        :param gripper_friction_constant: What the controller cancels of the gripper's own friction, in N.
            None keeps the value the arm was calibrated with. A larger one takes the trigger's stiffness
            off the operator's hand; too large and the trigger moves on its own.
        :param connect: ``ip -> TrossenArmDriver`` factory; a test injects its own.
        """
        self._ip = ip
        self._gain = force_feedback_gain
        self._gripper_friction = gripper_friction_constant
        self._connect = connect

        self.joints = pimm.ControlSystemEmitter[np.ndarray](self)
        self.grip = pimm.ControlSystemEmitter[float](self)
        self.follower_efforts = pimm.ControlSystemReceiver[np.ndarray](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)

    def _pushed_back(self) -> list[float]:
        """What the leader pushes back with: the follower's own external effort, reversed and scaled, so
        the operator feels what the follower is holding.

        Zero while nothing reports it, which is gravity compensation and nothing else.
        """
        zero = [0.0] * (_ARM_JOINTS + 1)
        if not self._gain:
            return zero
        felt = self.follower_efforts.read()
        if felt is None:
            return zero
        return (-self._gain * np.asarray(felt.data, dtype=np.float64)).tolist()

    def _go_to(self, driver: Any, asked: command.CommandType, closed: float, travel: float) -> None:
        """Drive the arm to the joints ``asked`` names, then give it back to the operator's hand.

        The controller plans the whole move, at the pace the follower travels at. The gripper stays where
        the operator holds it: ``set_all_positions`` takes every joint, and this move is the arm's.
        """
        if not isinstance(asked, command.JointPosition):
            raise NotImplementedError(f'A leader is driven to joints, and {asked} names none')
        target = np.asarray(asked.positions, dtype=np.float64)
        positions = np.asarray(driver.get_all_positions(), dtype=np.float64)
        seconds = max(_MIN_MOVE_TIME_S, float(np.max(np.abs(target - positions[:_ARM_JOINTS]))) / _MOVE_SPEED)
        held = float(np.clip(positions[_GRIPPER_JOINT], closed, closed + travel))
        logger.info(f'The leader at {self._ip} travels to the pose the session asked for, in {seconds:.1f} s')
        driver.set_all_modes(trossen_arm.Mode.position)
        try:
            driver.set_all_positions([*target, held], seconds, True)
        finally:
            driver.set_all_modes(trossen_arm.Mode.external_effort)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        with _opened(self._connect, self._ip) as driver, _gripper_freed(driver, self._gripper_friction):
            limit = driver.get_joint_limits()[_GRIPPER_JOINT]
            closed, travel = limit.position_min, limit.position_max - limit.position_min
            driver.set_all_modes(trossen_arm.Mode.external_effort)
            logger.info(f'The leader at {self._ip} is free to move; the controller holds its weight')

            limiter = pimm.RateLimiter(clock, hz=_HZ)
            complained_at: float = -_COMPLAIN_EVERY_S
            failed_at: float | None = None
            while not should_stop.value:
                now = clock.now()
                for call in self.sync_move.incoming():
                    with pimm.calls.raise_to(call):
                        self._go_to(driver, call.request, closed, travel)
                        call.set_result(None)
                if failed_at is None or now - failed_at >= _RECOVER_EVERY_S:
                    try:
                        if failed_at is not None:
                            # TODO: a fault the controller latches is cleared by a new session, not by
                            # asking for the mode again; reopen one, the way the follower's `recover` does.
                            driver.set_all_modes(trossen_arm.Mode.external_effort)
                            failed_at = None
                            logger.info(f'The leader at {self._ip} is free to move again')
                        driver.set_all_external_efforts(self._pushed_back(), 0.0, False)
                        positions = np.asarray(driver.get_all_positions(), dtype=np.float64)
                        ts = clock.now_ns()
                        self.joints.emit(positions[:_ARM_JOINTS], ts)
                        self.grip.emit(self._grip_of(positions[_GRIPPER_JOINT], closed, travel), ts)
                    # rules-allow: swallowed-error — a leader that stops being heard from is one arm of a
                    # session, and the rest of the rig goes on recording without it
                    except Exception as exc:
                        failed_at = now
                        if now - complained_at >= _COMPLAIN_EVERY_S:
                            complained_at = now
                            logger.error(f'The leader at {self._ip} is not being read: {exc}')

                yield limiter.wait()

    @staticmethod
    def _grip_of(position: float, closed: float, travel: float) -> float:
        """How closed the operator is holding the trigger, as the 1-is-closed grip positronic speaks.

        The reading sits a little outside the joint range at either end, so it saturates to 0..1.
        """
        return float(np.clip(1.0 - (position - closed) / travel, 0.0, 1.0))
