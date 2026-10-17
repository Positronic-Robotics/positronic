import time
from collections.abc import Callable, Generator, Sequence
from typing import Any

import numpy as np
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic import geom, keys
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs, Policy, PolicyEnd, PolicyRun, Runtime, Step
from positronic.teleop.protocol import EE_POSE, END, GRIP, IMAGES, KIND, START_TASK, TARGET, TASK, Reply, Request

QUAT = geom.Rotation.Representation.QUAT


class TrialNotStarted(Exception):
    """The teleoperator cancelled the trial, or was not ready within the ready limit of the service."""


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


class TeleopClient(Policy):
    """The client half of a teleop session: a person controls the robot through the teleop service.

    The teleop service declares it, with its arguments, in the stack of each session. A run asks the service for
    Ready before its first ``yield``, so the wait is outside the recording and the time limit. Then it keeps one frame
    call pending, and turns each reply into commands. The service keeps the trial state and the clutch.
    """

    WIRE_NAME = 'teleop_client'
    WIRE_VERSION = 1
    # A frame answer wakes the run. This is the latest time at which the run asks for its next step without one.
    _IDLE_NS = 1_000_000_000

    def __init__(self, ready_backstop_sec: float, cameras: Sequence[str], image_fps: float, still_clock_timeout: float):
        self._ready_backstop_sec = ready_backstop_sec
        self._cameras = tuple(cameras)
        self._image_fps = image_fps
        self._still_clock_timeout = still_clock_timeout

    def run(self, runtime: Runtime, infer: Callable[[dict], Any]) -> PolicyRun:
        self._await_ready(infer)
        obs = yield
        return (yield from self._control(runtime, infer, obs))

    def _await_ready(self, infer: Callable[[dict], Any]) -> None:
        """Wait until the teleoperator is ready. The backstop ends a wait of WAITING answers with no end."""
        deadline = time.monotonic() + self._ready_backstop_sec
        while True:
            match Reply(infer({KIND: Request.READY})[KIND]):
                case Reply.READY:
                    return
                case Reply.CANCELLED:
                    raise TrialNotStarted('the teleoperator cancelled the trial')
                case Reply.EXPIRED:
                    raise TrialNotStarted('the teleoperator was not ready within the ready limit of the service')
                case Reply.WAITING if time.monotonic() > deadline:
                    raise TimeoutError(f'the teleop service gave no Ready within {self._ready_backstop_sec} s')
                case Reply.WAITING:
                    pass
                case kind:
                    raise ValueError(f'the teleop service answered {kind} to a {Request.READY} request')

    def _control(self, runtime: Runtime, infer: Callable[[dict], Any], obs: Obs) -> Generator[Step, Obs, PolicyEnd]:
        """Hold the robot, then send each command of the service, until the teleoperator ends the run."""
        clock = _MovingClock(self._still_clock_timeout)
        image_period_ns = round(1e9 / self._image_fps)
        images_due_ns = runtime.time_ns
        attempt_started = False

        def frame_request(obs: Obs) -> dict[str, Any]:
            """The next frame request, with copies of the camera images when they are due."""
            nonlocal images_due_ns
            request = {KIND: Request.FRAME, TASK: obs[keys.TASK], EE_POSE: np.asarray(obs[keys.EE_POSE], dtype=float)}
            if runtime.time_ns >= images_due_ns:
                request[IMAGES] = {name: np.array(obs[name]) for name in self._cameras}
                images_due_ns = runtime.time_ns + image_period_ns
            return request

        # `Episode.bounds` start at the latest first record of all signals. A stay command starts every command signal
        # at the episode start, so the bounds keep the time before the first Control. It asks for the pose that the arm
        # has, so the arm does not move.
        pose = geom.Transform3D.from_vector(np.asarray(obs[keys.EE_POSE], dtype=float), QUAT)
        stay = {keys.ROBOT_COMMAND: CartesianPosition(pose), keys.TARGET_GRIP: float(obs[keys.GRIP])}
        pending = runtime.submit(infer, frame_request(obs))
        obs = yield Step(stay, runtime.time_ns + self._IDLE_NS)
        while True:
            clock.check(runtime.time_ns)
            if not pending.done():
                obs = yield Step({}, runtime.time_ns + self._IDLE_NS)
                continue
            reply = pending.result()
            if reply[START_TASK] and not attempt_started:
                runtime.set_metadata(policy_keys.ATTEMPT_START_NS, runtime.time_ns)
                attempt_started = True
            if END in reply:
                return PolicyEnd(reply[END])
            commands = {}
            if TARGET in reply:
                target = geom.Transform3D.from_vector(np.asarray(reply[TARGET], dtype=float), QUAT)
                commands = {keys.ROBOT_COMMAND: CartesianPosition(target), keys.TARGET_GRIP: float(reply[GRIP])}
            pending = runtime.submit(infer, frame_request(obs))
            obs = yield Step(commands, runtime.time_ns + self._IDLE_NS)

    def to_spec(self) -> dict[str, Any]:
        args = {
            'ready_backstop_sec': self._ready_backstop_sec,
            'cameras': list(self._cameras),
            'image_fps': self._image_fps,
            'still_clock_timeout': self._still_clock_timeout,
        }
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: args}
