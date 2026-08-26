from typing import reveal_type
from dataclasses import dataclass

import pimm
from pimm.methods import ControlSystemCaller, ControlSystemHandler


@dataclass(frozen=True)
class Reset:
    home: bool
    speed: float = 1.0


class Robot(pimm.ControlSystem):
    def __init__(self):
        self.reset = ControlSystemHandler[Reset, bool](self)
        self.seen: list[Reset] = []

    def run(self, should_stop, clock):
        waiting = []
        while not should_stop.value:
            for call in self.reset.incoming():
                reveal_type(call.request)
                reveal_type(call.request.home)
                self.seen.append(call.request)
                waiting.append((2, call))  # answered two ticks later
            still = []
            for ticks, call in waiting:
                if ticks:
                    still.append((ticks - 1, call))
                elif call.request.speed > 2.0:
                    call.set_exception(ValueError('too fast'))
                else:
                    call.set_result(call.request.home)
            waiting = still
            yield pimm.Sleep(0.001)


class Policy(pimm.ControlSystem):
    def __init__(self):
        self.reset = ControlSystemCaller[Reset, bool](self)
        self.results = []

    def run(self, should_stop, clock):
        futures = [self.reset(Reset(home=True)), self.reset(Reset(home=False, speed=9.0))]
        while not all(f.done() for f in futures):
            yield pimm.Sleep(0.001)
        reveal_type(futures[0].result())
        self.results = [f.exception() or f.result() for f in futures]


if __name__ == '__main__':
    robot, policy = Robot(), Policy()
    with pimm.World(virtual_time=True) as world:
        world.connect(policy.reset, robot.reset)
        world.run([policy, robot])
    print('seen    :', robot.seen)
    print('results :', policy.results)
