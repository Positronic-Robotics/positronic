from collections.abc import Iterator

import pimm


class FakeRobotiq2F(pimm.ControlSystem):
    """The ports an embodiment reads off a Robotiq gripper, with no device behind them. It emits nothing."""

    def __init__(self) -> None:
        self.grip = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.sync_move = pimm.calls.ControlSystemHandler[float, None](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        while not should_stop.value:
            yield pimm.Sleep(0.1)
