import time
from collections.abc import Generator, Mapping
from contextlib import closing
from typing import Any

import numpy as np
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY
from positronic_wire import registry
from positronic_wire.wire import ConnectRefused, SessionAddress

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.policy import keys as policy_keys
from positronic.policy.base import Answer, Obs, Policy, PolicyEnd, PolicyRun, Runtime, Step

from .clutch import OperatorPosition, Tracker
from .session import ATTEMPT_STARTED, CONTROL, TASK, Event, Frame, TeleopSession

# A frame answer wakes the run. This is the latest time at which the run asks for its next step without one.
_IDLE_NS = 1_000_000_000


# HACK: The run reads wall time, so it learns whether its clock is simulated. A person needs the episode clock to run
#  with the wall clock, as a charged sim clock does.
# TODO: Let a policy state that it needs a charged clock, and let `eval run` refuse an uncharged run for it.
class _MovingClock:
    """Raise when the episode clock stands still for ``still_limit_sec`` of wall time."""

    def __init__(self, still_limit_sec: float):
        self._still_limit_ns = round(still_limit_sec * 1e9)
        self._time_ns: int | None = None
        self._still_since_ns = time.monotonic_ns()

    def check(self, time_ns: int) -> None:
        now_ns = time.monotonic_ns()
        if time_ns != self._time_ns:
            self._time_ns, self._still_since_ns = time_ns, now_ns
        elif now_ns - self._still_since_ns > self._still_limit_ns:
            raise RuntimeError(
                'the episode clock stands still while a person waits: run the sim with charged inference time'
            )


class TeleopPolicy(Policy):
    """A person controls the robot through the teleop service. Each run is a client of one service session.

    The policy holds configuration only, so a driver can pickle it. It has no wire spec, so a model server cannot
    declare it. A run waits for the teleoperator to be ready before its first ``yield``. A Cancel, an error of the
    service or the ready timeout fails the run there, so the episode does not start.
    """

    # TODO: Set the ready and frame timeouts from the limits of the teleop service on a rig.
    def __init__(
        self,
        wire: str,
        address: SessionAddress,
        operator_position: OperatorPosition = OperatorPosition.FRONT,
        cameras: tuple[str, ...] | None = None,
        image_fps: float = 15.0,
        open_timeout: float = 10.0,
        ready_timeout: float = 130.0,
        frame_timeout: float = 5.0,
        still_clock_timeout: float = 1.0,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ):
        self._wire = wire
        self._address = address
        self._operator_position = operator_position
        self._cameras = cameras
        self._image_period_ns = round(1e9 / image_fps)
        self._open_timeout = open_timeout
        self._ready_timeout = ready_timeout
        self._frame_timeout = frame_timeout
        self._still_clock_timeout = still_clock_timeout
        self._jpeg_quality = jpeg_quality
        self._probed = False

    def meta(self) -> dict[str, Any]:
        """Probe the service once, unless a run already opened a session with it. Raise when it does not answer."""
        if not self._probed:
            refusal = registry.client_wire(self._wire).probe(self._address, None, self._open_timeout)
            if refusal is not None:
                raise ConnectRefused(refusal, f'the teleop service at {self._address} does not answer: {refusal.value}')
            self._probed = True
        return {policy_keys.TYPE: 'teleop', 'operator_position': self._operator_position.name}

    def run(self, runtime: Runtime) -> PolicyRun:
        connection = registry.client_wire(self._wire).dial(self._address, None, self._open_timeout)
        self._probed = True
        with closing(
            TeleopSession(connection, self._ready_timeout, self._frame_timeout, self._jpeg_quality)
        ) as session:
            # Before the first `yield`, so the wait is outside the recording and the time limit. It blocks the control
            # thread of the World, so the ready timeout bounds it.
            session.ready()
            obs = yield
            return (yield from self._control(runtime, session, obs))

    def _camera_names(self, obs: Mapping[str, Any]) -> tuple[str, ...]:
        """The cameras that the run sends: ``cameras``, or every image of ``Obs``, in its order."""
        if self._cameras is not None:
            return self._cameras
        return tuple(name for name in obs if name.startswith(keys.IMAGE_PREFIX))

    @staticmethod
    def _robot_pose(obs: Mapping[str, Any]) -> geom.Transform3D:
        return geom.Transform3D.from_vector(
            np.asarray(obs[keys.EE_POSE], dtype=float), geom.Rotation.Representation.QUAT
        )

    def _control(self, runtime: Runtime, session: TeleopSession, obs: Obs) -> Generator[Step, Obs, PolicyEnd]:
        """Hold the robot, then move it with the hand while control is on, until the teleoperator ends the run."""
        tracker = Tracker(self._operator_position.value)
        attempt_started = False

        clock = _MovingClock(self._still_clock_timeout)

        images_due_ns = runtime.time_ns

        def display(obs: Mapping[str, Any]) -> dict[str, Any]:
            return {TASK: obs[keys.TASK], CONTROL: tracker.on, ATTEMPT_STARTED: attempt_started}

        def frame_request(obs: Mapping[str, Any]) -> Answer[Frame]:
            """The next frame request, with copies of the camera images when they are due. The worker encodes them."""
            nonlocal images_due_ns
            images = None
            if runtime.time_ns >= images_due_ns:
                images = {name: np.array(obs[name]) for name in self._camera_names(obs)}
                images_due_ns = runtime.time_ns + self._image_period_ns
            return runtime.submit(session.frame, display(obs), images)

        # A stay command, so that every command signal starts at the episode start. It asks for the pose that the arm
        # has, so the arm does not move.
        stay = {keys.ROBOT_COMMAND: CartesianPosition(self._robot_pose(obs)), keys.TARGET_GRIP: float(obs[keys.GRIP])}
        pending = frame_request(obs)
        obs = yield Step(stay, runtime.time_ns + _IDLE_NS)
        while True:
            clock.check(runtime.time_ns)
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
                        tracker.turn_on(self._robot_pose(obs))
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
            pending = frame_request(obs)
            obs = yield Step(commands, runtime.time_ns + _IDLE_NS)
