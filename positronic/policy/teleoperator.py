"""The teleoperator at the rig, the helper of a run: a policy whose elements carry the commands and the press fields
of its console (``positronic.policy.takeover``)::

    {'robot_command': CartesianPosition(...), 'take_over': 1_200_000_000, 'hand_back': None}
"""

from typing import Any

import pimm
from positronic.policy.base import Policy, PolicyRun, Runtime, Step
from positronic.policy.keys import LICENSE, TELEOPERATOR, TYPE


class Teleoperator(Policy):
    """The teleoperator at the rig. ``inputs`` gets each message of the console. A step carries the message that came
    since the last step, or no element when none came. ``license`` is the license of the commands of the person."""

    def __init__(self, inputs: pimm.SignalReceiver[dict[str, Any]], license: str):
        self._inputs = inputs
        self._license = license

    def meta(self) -> dict[str, Any]:
        return {TYPE: TELEOPERATOR, LICENSE: self._license}

    def run(self, runtime: Runtime) -> PolicyRun:
        yield
        while True:
            message = pimm.read_updated(self._inputs)
            yield Step(dict(message.data) if message is not None else {}, runtime.time_ns)
