"""Composition order, metadata, and execution of mixed processor and codec stacks."""

import threading

import pytest

from positronic.policy.base import Policy, Step
from positronic.policy.codec import Codec, RestrictImageSize
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.layers import ChunkedSchedule, StopOnFault
from positronic.policy.sequential import Sequential


@pytest.mark.parametrize('nested', [False, True])
def test_codec_work_is_inside_submit(nested):
    caller_thread = threading.get_ident()
    threads = []

    class ThreadCodec(Codec):
        def encode(self, data):
            threads.append(threading.get_ident())
            return {**data, 'encoded': 42}

        def _decode_single(self, data):
            threads.append(threading.get_ident())
            return {'value': data['value'] + 10}

    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    conversion = Sequential(ThreadCodec(), RestrictImageSize()) if nested else ThreadCodec()
    run = runtime.start(Sequential(ChunkedSchedule(fps=10), conversion), lambda obs: [{'value': obs['encoded']}])
    try:
        first = run.send({})
        assert isinstance(first, Step)
        assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
        completed = run.send({})
        assert isinstance(completed, Step)
        assert dict(first.commands) | dict(completed.commands) == {'value': 52}
        assert completed.resume_at_ns == 100_000_000
        assert len(threads) == 2
        assert all(thread != caller_thread for thread in threads)
    finally:
        runtime.close()
        run.close()


def test_sequential_combines_component_metadata():
    class NamedSchedule(ChunkedSchedule):
        def meta(self):
            return {'config': {'fps': 10}}

    class NamedStop(StopOnFault):
        def meta(self):
            return {'config': {'fault_handling': True, 'fps': 20}}

    assert Sequential(NamedStop(), NamedSchedule(fps=10)).meta() == {'config.fault_handling': True, 'config.fps': 10}


def test_mixed_sequence_preserves_order_step_timing_and_empty_commands():
    events = []
    caller_thread = threading.get_ident()

    class TraceCodec(Codec):
        def __init__(self, name, factor):
            self.name = name
            self.factor = factor

        def encode(self, data):
            assert threading.get_ident() == caller_thread
            events.append(f'{self.name}.encode')
            return {**data, 'value': data['value'] * self.factor}

        def _decode_single(self, data):
            events.append(f'{self.name}.decode')
            return {'value': data['value'] + self.factor}

        @property
        def meta(self):
            return {'codec': {self.name: self.factor}}

    class Forward(Policy):
        def run(self, runtime, infer):
            obs = yield
            try:
                while True:
                    events.append('processor')
                    obs = yield Step(infer(obs) if not obs.get('skip') else {}, 123)
            finally:
                events.append('processor.close')

    def infer(obs):
        events.append('infer')
        return {'value': obs['value']}

    stack = Sequential(TraceCodec('outer', 2), Forward(), TraceCodec('inner', 3))
    assert stack.meta() == {'codec.outer': 2, 'codec.inner': 3}
    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    run = runtime.start(stack, infer)
    try:
        assert run.send({'value': 1}) == Step({'value': 11}, 123)
        assert events == ['outer.encode', 'processor', 'inner.encode', 'infer', 'inner.decode', 'outer.decode']
        events.clear()
        assert run.send({'value': 1, 'skip': True}) == Step({}, 123)
        assert events == ['outer.encode', 'processor']
    finally:
        runtime.close()
        run.close()
    assert events[-1] == 'processor.close'
