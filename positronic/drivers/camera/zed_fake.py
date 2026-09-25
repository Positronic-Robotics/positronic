from collections.abc import Iterator

import pimm


class FakeSLCamera(pimm.ControlSystem):
    """The port an embodiment reads off a ZED camera, with no device behind it. It emits nothing."""

    def __init__(self) -> None:
        self.frame = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        while not should_stop.value:
            yield pimm.Sleep(0.1)
