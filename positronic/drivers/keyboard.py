from typing import Iterator
from pynput.keyboard import Events

import pimm


class Keyboard:
    buttons: pimm.SignalEmitter = pimm.NoOpEmitter()

    def __init__(self) -> None:
        pass

    def run(self, should_stop: pimm.SignalReader, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        button_state = {}

        with Events() as events:
            while not should_stop.value:
                event = events.get()

                if isinstance(event, events.Press):
                    button_state[str(event.key)] = True
                elif isinstance(event, events.Release):
                    del button_state[event.key]

                self.buttons.emit(button_state)

                yield clock.sleep(0)  # no need for additional sleep, since waiting is done in .get()
