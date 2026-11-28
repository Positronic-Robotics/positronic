"""The teleoperator at the rig, the helper of a run: a policy whose elements carry the commands and the press fields
of its console (``positronic.policy.takeover``)::

    {'robot_command': CartesianPosition(...), 'take_over': 1_200_000_000, 'hand_back': None}

A console that sends or reads poses states the frame of each pose, as a policy does::

    Teleoperator(console, license='internal', frames={TARGET_EE_POSE: EEFrame(site=GRIPPER_SITE)})
"""

from collections.abc import Mapping
from typing import Any

import pimm
from positronic.drivers.roboarm.models import EEFrame, ee_frames_entry
from positronic.policy.base import Policy, PolicyRun, Runtime, Step
from positronic.policy.keys import LICENSE, TELEOPERATOR, TYPE


class Teleoperator(Policy):
    """The teleoperator at the rig. ``inputs`` gets each message of the console. The first step of a run carries the
    last message of the console, so a press from before the episode is not new. Each later step carries the message
    that came since the step before, or no element when none came. ``license`` is the license of the commands of the
    person. ``frames`` gives the frame of each pose that the console sends or reads."""

    def __init__(
        self, inputs: pimm.SignalReceiver[dict[str, Any]], license: str, frames: Mapping[str, EEFrame] | None = None
    ):
        self._inputs = inputs
        self._license = license
        self._frames = dict(frames or {})

    def meta(self) -> dict[str, Any]:
        stated = ee_frames_entry(self._frames) if self._frames else {}
        return {TYPE: TELEOPERATOR, LICENSE: self._license, **stated}

    def run(self, runtime: Runtime) -> PolicyRun:
        yield
        message = self._inputs.read()
        while True:
            yield Step(dict(message.data) if message is not None else {}, runtime.time_ns)
            message = pimm.read_updated(self._inputs)
