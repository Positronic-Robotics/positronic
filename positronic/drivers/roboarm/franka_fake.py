from collections.abc import Iterator

import pimm

from . import State, command


class FakeFranka(pimm.ControlSystem):
    """The ports an embodiment reads off a Franka arm, with no device behind them. It emits nothing."""

    def __init__(self) -> None:
        self.commands = pimm.ControlSystemReceiver[command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[command.CommandType, None](self)
        self.state = pimm.ControlSystemEmitter[State](self)
        self.robot_meta = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        while not should_stop.value:
            yield pimm.Sleep(0.1)
