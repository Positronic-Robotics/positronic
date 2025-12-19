import logging
import select
import sys
import termios
import tty

import pimm


class KeyboardControl(pimm.ControlSystem):
    def __init__(self, quit_key: str | None = None):
        self.quit_key = quit_key
        self.keyboard_inputs = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        # Check if stdin is a TTY before attempting terminal operations
        if not sys.stdin.isatty():
            print('WARNING: KeyboardControl cannot read input - stdin is not a terminal', file=sys.stderr)
            logging.warning('KeyboardControl cannot read input - stdin is not a terminal')
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while not should_stop.value:
                r, _, _ = select.select([sys.stdin], [], [], 0.0)
                if r:
                    key = sys.stdin.read(1)
                    if key == self.quit_key:
                        return
                    self.keyboard_inputs.emit(key)
                yield pimm.Sleep(0.01)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
