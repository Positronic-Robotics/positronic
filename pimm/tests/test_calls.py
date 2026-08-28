import multiprocessing as mp
import pickle
from itertools import islice

import pytest

from pimm.calls import Answer, CallerDict, ControlSystemCaller, ControlSystemHandler, HandlerStopped, all_of, raise_to
from pimm.core import ControlSystem, NoValueException, Sleep
from pimm.tests.testing import Passive, wire_call
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
        answers = [self.add(pair) for pair in self._calls]
        while not all(a.done() for a in answers) and not should_stop.value:
            yield Sleep(0.001)
        for answer in answers:
            try:
                self.results.append(answer.result())
            except Exception as e:
                self.results.append(e)
        if self._total is not None:
            self._total.value = sum(r for r in self.results if isinstance(r, int))


@pytest.fixture
def bound():
    """A caller wired to a handler in-process, without scheduling either owner."""
    caller, handler = ControlSystemCaller(Passive()), ControlSystemHandler(Passive())
    with World() as world:
        wire_call(world, caller, handler)
        yield caller, handler


class TestCallAndAnswer:
    def test_call_returns_an_answer_completed_by_the_handler(self, bound):
        caller, handler = bound
        answer = caller((1, 2))
        assert isinstance(answer, Answer)
        assert not answer.done()

        (call,) = handler.incoming()
        assert call.request == (1, 2)
        call.set_result(3)
        assert answer.done()
        assert answer.result() == 3

    def test_calls_arrive_in_order_and_unreached_ones_wait_for_the_next_incoming(self, bound):
        caller, handler = bound
        for i in range(20):
            caller(i)
        assert [call.request for call in islice(handler.incoming(), 5)] == list(range(5))
        assert [call.request for call in handler.incoming()] == list(range(5, 20))
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
        answer = caller(None)
        (call,) = handler.incoming()
        call.set_exception(ValueError('boom'))
        assert answer.done()
        with pytest.raises(ValueError, match='boom'):
            answer.result()

    def test_a_block_that_raises_answers_the_call_with_it(self, bound):
        caller, handler = bound
        answer = caller(None)
        (call,) = handler.incoming()

        with raise_to(call):
            raise ValueError('boom')

        with pytest.raises(ValueError, match='boom'):
            answer.result()

    def test_a_block_that_returns_leaves_the_answer_to_it(self, bound):
        """The block sets its own result, so a handler that answers later — or not at all — still can."""
        caller, handler = bound
        answer = caller(None)
        (call,) = handler.incoming()

        with raise_to(call):
            pass

        assert not answer.done()

    def test_answering_twice_raises(self, bound):
        caller, handler = bound
        caller(None)
        (call,) = handler.incoming()
        call.set_result(1)
        with pytest.raises(AssertionError):
            call.set_result(2)
        with pytest.raises(AssertionError):
            call.set_exception(ValueError())

    def test_result_of_an_unanswered_call_raises(self, bound):
        caller, handler = bound
        with pytest.raises(NoValueException):
            caller(None).result()

    def test_unbound_caller_raises(self):
        with pytest.raises(RuntimeError):
            ControlSystemCaller(Passive())(None)

    def test_a_caller_says_whether_anything_will_answer_it(self, bound):
        """A caller with no handler bound says so, and calling it raises rather than waiting for an answer
        that cannot come."""
        caller, _ = bound
        assert caller.connected
        assert not ControlSystemCaller(Passive()).connected

    def test_unbound_handler_yields_nothing(self):
        assert list(ControlSystemHandler(Passive()).incoming()) == []


class Deaf(ControlSystem):
    """Ends without reading, so every call made to it is one it never reached."""

    def __init__(self):
        self.add = ControlSystemHandler[tuple[int, int], int](self)

    def run(self, should_stop, clock):
        yield Sleep(0.001)


class Interrupted(ControlSystem):
    """Ends on the interrupt an operator sends, which reaches every process of a run at once."""

    def __init__(self):
        self.add = ControlSystemHandler[tuple[int, int], int](self)

    def run(self, should_stop, clock):
        yield Sleep(0.001)
        raise KeyboardInterrupt


class TestAllOf:
    def test_one_answer_stands_for_many(self, bound):
        caller, handler = bound
        both = all_of([caller(1), caller(2)])
        first, second = handler.incoming()

        first.set_result('one')
        assert not both.done(), 'a caller waits for the slowest of them'

        second.set_result('two')
        assert both.done()
        assert both.result() == ('one', 'two')

    def test_a_single_failure_is_what_the_caller_hears(self, bound):
        caller, handler = bound
        both = all_of([caller(1), caller(2)])
        first, second = handler.incoming()
        first.set_result('one')
        second.set_exception(ValueError('boom'))

        assert both.done()
        with pytest.raises(ValueError, match='boom'):
            both.result()

    def test_nothing_to_wait_for_is_answered(self):
        """An ``all_of`` over no answers is done at once, so asking for nothing goes straight through."""
        assert all_of([]).done()
        assert all_of([]).result() == ()


class TestCallerDict:
    def test_names_fix_the_ports_the_dict_has(self):
        """A control system that knows who it calls up front has no use for a key it was never built with."""
        callers = CallerDict(Passive(), names=['a', 'b'])

        assert sorted(callers) == ['a', 'b']
        assert all(isinstance(caller, ControlSystemCaller) for caller in callers.values())
        with pytest.raises(KeyError):
            callers['c']

    def test_without_names_a_key_allocates_on_first_use(self):
        callers = CallerDict(Passive())

        assert callers['a'] is callers['a']
        assert callers['a'].owner is not None


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

    def test_wrappers_do_not_apply_to_calls(self):
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

    def test_a_handler_whose_system_ends_answers_the_calls_it_never_reached(self):
        """A caller blocked on an answer would wait for the rest of the run, and the answer is never coming."""
        client, deaf = Client([(1, 2)]), Deaf()
        with World(virtual_time=True) as world:
            world.connect(client.add, deaf.add)
            world.run([client, deaf])
        assert isinstance(client.results[0], HandlerStopped)

    def test_a_handler_the_operator_interrupts_answers_nothing(self):
        """An interrupt can land inside a manager call, and the connection then carries half a message.

        Reading it again returns the tail of another, so an interrupted system says nothing at all.
        """
        client, interrupted = Client([(1, 2)]), Interrupted()
        with World(virtual_time=True) as world:
            world.connect(client.add, interrupted.add)
            with pytest.raises(KeyboardInterrupt):
                world.run([client, interrupted])
        assert client.results == []

    def test_a_stopped_handler_survives_the_trip_to_another_process(self):
        """A reply crosses a pipe as pickle, and an exception is rebuilt by calling its class with its args."""
        assert isinstance(pickle.loads(pickle.dumps(HandlerStopped())), HandlerStopped)

    def test_caller_in_background_process(self):
        total = mp.get_context('spawn').Value('i', 0)
        client, adder = Client([(1, 2), (3, 4)], total), Adder()
        with World() as world:
            world.connect(client.add, adder.add)
            world.run(adder, background=client)
        assert total.value == 10
