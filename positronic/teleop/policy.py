from collections.abc import Mapping
from typing import Any

import numpy as np
from positronic_wire import registry
from positronic_wire.wire import ConnectRefused, SessionAddress

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.policy import keys as policy_keys
from positronic.policy.base import Policy, PolicyEnd, PolicyRun, Runtime, Step

from .clutch import OperatorPosition, Tracker
from .session import ATTEMPT_STARTED, CONTROL, TASK, Event, Frame, TeleopSession

# A frame answer wakes the run. This is the latest time at which the run asks for its next step without one.
_IDLE_NS = 1_000_000_000


def _robot_pose(obs: Mapping[str, Any]) -> geom.Transform3D:
    return geom.Transform3D.from_vector(np.asarray(obs[keys.EE_POSE], dtype=float), geom.Rotation.Representation.QUAT)


class TeleopPolicy(Policy):
    """A person controls the robot through the teleop service. Each run is a client of one service session.

    The policy holds configuration only, so a driver can pickle it. It has no wire spec, so a model server cannot
    declare it.
    """

    # TODO: Set the ready and frame timeouts from the limits of the teleop service on a rig.
    def __init__(
        self,
        wire: str,
        address: SessionAddress,
        operator_position: OperatorPosition = OperatorPosition.FRONT,
        open_timeout: float = 10.0,
        ready_timeout: float = 130.0,
        frame_timeout: float = 5.0,
    ):
        self._wire = wire
        self._address = address
        self._operator_position = operator_position
        self._open_timeout = open_timeout
        self._ready_timeout = ready_timeout
        self._frame_timeout = frame_timeout
        self._probed = False

    def meta(self) -> dict[str, Any]:
        """Probe the service on the first call only. A later call is inside a recording, and must not fail it."""
        if not self._probed:
            refusal = registry.client_wire(self._wire).probe(self._address, None, self._open_timeout)
            if refusal is not None:
                raise ConnectRefused(refusal, f'the teleop service at {self._address} does not answer: {refusal.value}')
            self._probed = True
        return {policy_keys.TYPE: 'teleop', 'operator_position': self._operator_position.name}

    def run(self, runtime: Runtime) -> PolicyRun:
        connection = registry.client_wire(self._wire).dial(self._address, None, self._open_timeout)
        session = TeleopSession(connection, self._ready_timeout, self._frame_timeout)
        try:
            ready = runtime.submit(session.ready)
            obs = yield
            ready.result()
            tracker = Tracker(self._operator_position.value)
            attempt_started = False

            def display(obs: Mapping[str, Any]) -> dict[str, Any]:
                return {TASK: obs[keys.TASK], CONTROL: tracker.on, ATTEMPT_STARTED: attempt_started}

            # A stay command, so that every command signal starts at the episode start. It asks for the pose that
            # the arm has, so the arm does not move.
            stay = {keys.ROBOT_COMMAND: CartesianPosition(_robot_pose(obs)), keys.TARGET_GRIP: float(obs[keys.GRIP])}
            pending = runtime.submit(session.frame, display(obs))
            obs = yield Step(stay, runtime.time_ns + _IDLE_NS)
            while True:
                if not pending.done():
                    obs = yield Step({}, runtime.time_ns + _IDLE_NS)
                    continue
                frame: Frame = pending.result()
                target = None if frame.pose is None else tracker.update(frame.pose)
                for event in frame.events:
                    match event:
                        case Event.CLUTCH if tracker.on:
                            tracker.turn_off()
                        case Event.CLUTCH if frame.pose is not None:
                            tracker.turn_on(_robot_pose(obs))
                            target = tracker.update(frame.pose)
                        case Event.START_TASK if not attempt_started:
                            runtime.set_metadata(policy_keys.ATTEMPT_START_NS, runtime.time_ns)
                            attempt_started = True
                        case Event.COMPLETE:
                            return PolicyEnd.COMPLETE
                        case Event.GIVE_UP:
                            return PolicyEnd.GIVE_UP
                if target is None and tracker.on:
                    tracker.turn_off()
                commands = {}
                if tracker.on and target is not None:
                    commands = {keys.ROBOT_COMMAND: CartesianPosition(target), keys.TARGET_GRIP: frame.grip}
                pending = runtime.submit(session.frame, display(obs))
                obs = yield Step(commands, runtime.time_ns + _IDLE_NS)
        finally:
            session.close()
