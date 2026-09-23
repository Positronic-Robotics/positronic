"""A World with one protected background child, run as its own process by the signal tests.

Usage
    python -m pimm.tests.sigterm_probe <directory>

The child writes ``child.pid`` when it starts and ``shut_down`` when its shutdown ends.
"""

import os
import pathlib
import sys
import time

from pimm.core import ControlSystem, ShutdownPolicy, Sleep
from pimm.world import World


class Marker(ControlSystem):
    shutdown_policy = ShutdownPolicy.WAIT_FOR_COMPLETION

    def __init__(self, directory: str):
        self.directory = pathlib.Path(directory)

    def run(self, should_stop, clock):
        (self.directory / 'child.pid').write_text(str(os.getpid()))
        while not should_stop.value:
            yield Sleep(0.01)
        (self.directory / 'shut_down').write_text('')


if __name__ == '__main__':
    with World() as world:
        world.start([], Marker(sys.argv[1]))
        while not world.should_stop:
            time.sleep(0.05)
