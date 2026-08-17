import multiprocessing as mp
from concurrent.futures import Future, InvalidStateError

import pytest

from pimm.core import ControlSystem, Sleep
from pimm.methods import ControlSystemCaller, ControlSystemHandler
from pimm.world import World


class Adder(ControlSystem):
    """Serves `add(a, b)`, failing on a negative operand; `defer` holds every answer that many ticks."""

    def __init__(self, defer: int = 0):
        self.add = ControlSystemHandler[tuple[int, int], int](self)
        self._defer = defer
        self.seen = []

    def run(self, should_stop, clock):
        waiting = []
        while not should_stop.value:
            for call in self.add.incoming():
                self.seen.append(call.request)
                waiting.append((self._defer, call))
            still_waiting = []
            for ticks, call in waiting:
                if ticks > 0:
                    still_waiting.append((ticks - 1, call))
                else:
                    self._answer(call)
            waiting = still_waiting
            yield Sleep(0.001)

    @staticmethod
    def _answer(call):
        a, b = call.request
        if a < 0 or b < 0:
            call.set_exception(ValueError('negative operand'))
        else:
            call.set_result(a + b)


class Client(ControlSystem):
    """Makes every call up front, runs until all are answered, then returns — which stops the world."""

    def __init__(self, calls: list[tuple[int, int]], total=None):
        self.add = ControlSystemCaller[tuple[int, int], int](self)
        self._calls = calls
        self._total = total
        self.results = []

    def run(self, should_stop, clock):
        futures = [self.add(pair) for pair in self._calls]
        while not all(f.done() for f in futures):
            yield Sleep(0.001)
        self.results = [f.exception() or f.result() for f in futures]
        if self._total is not None:
            self._total.value = sum(r for r in self.results if isinstance(r, int))


class Passive(ControlSystem):
    def run(self, should_stop, clock):
        while not should_stop.value:
            yield Sleep(0.001)


@pytest.fixture
def bound():
    """A caller wired to a handler in-process, without scheduling either owner."""
    caller, handler = ControlSystemCaller(Passive()), ControlSystemHandler(Passive())
    with World() as world:
        for emitter, receiver in ((caller.requests, handler.requests), (handler.replies, caller.replies)):
            physical_emitter, physical_receiver = world.local_pipe(maxsize=0)
            emitter._bind(physical_emitter)
            receiver._bind(physical_receiver)
        yield caller, handler


class TestCallAndAnswer:
    def test_call_returns_a_future_completed_by_the_answer(self, bound):
        caller, handler = bound
        future = caller((1, 2))
        assert isinstance(future, Future)
        assert not future.done()

        (call,) = handler.incoming()
        assert call.request == (1, 2)
        call.set_result(3)
        assert future.done()
        assert future.result() == 3

    def test_calls_arrive_in_order_and_each_once(self, bound):
        caller, handler = bound
        for i in range(20):
            caller(i)
        assert [call.request for call in handler.incoming()] == list(range(20))
        assert list(handler.incoming()) == []

    def test_replies_may_return_out_of_order(self, bound):
        caller, handler = bound
        first, second = caller(1), caller(2)
        call_1, call_2 = handler.incoming()
        call_2.set_result('two')
        assert second.result() == 'two'
        assert not first.done()
        call_1.set_result('one')
        assert first.result() == 'one'

    def test_exception_set_by_handler_is_raised_by_result(self, bound):
        caller, handler = bound
        future = caller(None)
        (call,) = handler.incoming()
        call.set_exception(ValueError('boom'))
        assert isinstance(future.exception(), ValueError)
        with pytest.raises(ValueError, match='boom'):
            future.result()

    def test_answering_twice_raises(self, bound):
        caller, handler = bound
        caller(None)
        (call,) = handler.incoming()
        call.set_result(1)
        with pytest.raises(InvalidStateError):
            call.set_result(2)
        with pytest.raises(InvalidStateError):
            call.set_exception(ValueError())

    def test_unanswered_future_does_not_wait(self, bound):
        caller, handler = bound
        future = caller(None)
        with pytest.raises(TimeoutError):
            future.result()
        with pytest.raises(TimeoutError):
            future.exception()
        with pytest.raises(NotImplementedError):
            future.result(timeout=0.01)
        with pytest.raises(NotImplementedError):
            future.exception(timeout=0.01)

    def test_cancel_raises(self, bound):
        caller, handler = bound
        with pytest.raises(NotImplementedError):
            caller(None).cancel()

    def test_unbound_caller_raises(self):
        with pytest.raises(RuntimeError):
            ControlSystemCaller(Passive())(None)

    def test_unbound_handler_yields_nothing(self):
        assert list(ControlSystemHandler(Passive()).incoming()) == []


class TestWorldConnect:
    def test_handler_serves_one_caller(self):
        handler = ControlSystemHandler(Passive())
        with World() as world:
            world.connect(ControlSystemCaller(Passive()), handler)
            with pytest.raises(AssertionError):
                world.connect(ControlSystemCaller(Passive()), handler)

    def test_caller_reaches_one_handler(self):
        caller = ControlSystemCaller(Passive())
        with World() as world:
            world.connect(caller, ControlSystemHandler(Passive()))
            with pytest.raises(AssertionError):
                world.connect(caller, ControlSystemHandler(Passive()))

    def test_wrappers_do_not_apply_to_methods(self):
        caller, handler = ControlSystemCaller(Passive()), ControlSystemHandler(Passive())
        with World() as world:
            with pytest.raises(AssertionError):
                world.connect(caller, handler, emitter_wrapper=lambda e: e)

    def test_in_process(self):
        client, adder = Client([(1, 2), (-1, 2), (3, 4)]), Adder(defer=2)
        with World(virtual_time=True) as world:
            world.connect(client.add, adder.add)
            world.run([client, adder])
        assert client.results[0] == 3 and client.results[2] == 7
        assert isinstance(client.results[1], ValueError)
        assert adder.seen == [(1, 2), (-1, 2), (3, 4)]

    def test_handler_in_background_process(self):
        client, adder = Client([(1, 2), (3, 4)]), Adder()
        with World() as world:
            world.connect(client.add, adder.add)
            world.run(client, background=adder)
        assert client.results == [3, 7]

    def test_caller_in_background_process(self):
        total = mp.get_context('spawn').Value('i', 0)
        client, adder = Client([(1, 2), (3, 4)], total), Adder()
        with World() as world:
            world.connect(client.add, adder.add)
            world.run(adder, background=client)
        assert total.value == 10
