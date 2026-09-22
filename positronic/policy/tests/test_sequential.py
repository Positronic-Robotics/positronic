"""Composition order, metadata, and execution of mixed processor and codec stacks."""

import threading
from contextlib import nullcontext

import pytest

from positronic import telemetry, telemetry_keys
from positronic.policy.base import Policy, Step
from positronic.policy.codec import ChangeEEFrame, Codec, RestrictImageSize
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


@pytest.mark.parametrize('with_codec', [False, True])
@pytest.mark.parametrize('timed', [False, True])
def test_sequence_preserves_normal_processor_completion(with_codec, timed):
    closed = []

    class Finite(Policy):
        def run(self, runtime):
            try:
                yield
                yield Step({'value': 1}, 123)
            finally:
                closed.append(True)

    stack = Sequential(RestrictImageSize(), Finite()) if with_codec else Sequential(Finite())
    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    with telemetry.timings_to(lambda *_: None) if timed else nullcontext():
        run = runtime.start(stack)
        try:
            assert run.send({}) == Step({'value': 1}, 123)
            with pytest.raises(StopIteration):
                run.send({})
            assert closed == [True]
        finally:
            runtime.close()
            run.close()


@pytest.mark.parametrize('translation', [0.1, 0.2])
def test_sequential_rejects_two_frame_conversions(translation):
    outer = ChangeEEFrame([0.1, 0, 0, 1, 0, 0, 0])
    inner = ChangeEEFrame([translation, 0, 0, 1, 0, 0, 0]) | RestrictImageSize()
    schedule = ChunkedSchedule(fps=10)
    stack = Sequential(outer, schedule)
    assert stack.meta() == outer.meta | schedule.meta()
    with pytest.raises(ValueError, match='Only one component'):
        Sequential(stack, inner).meta()


def test_mixed_sequence_preserves_order_step_timing_and_empty_commands(tmp_path):
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
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'mixed-stack'):
        run = runtime.start(stack, infer)
        try:
            assert run.send({'value': 1}) == Step({'value': 11}, 123)
            assert events == ['outer.encode', 'processor', 'inner.encode', 'infer', 'inner.decode', 'outer.decode']
            with telemetry.span('between_calls'):
                events.clear()
            assert run.send({'value': 1, 'skip': True}) == Step({}, 123)
            assert events == ['outer.encode', 'processor']
        finally:
            runtime.close()
            run.close()
    assert events[-1] == 'processor.close'
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    roots = sorted((s for s in spans if s.parent_id is None), key=lambda s: s.start_ns)
    assert [s.name for s in roots] == ['sequential', 'between_calls', 'sequential']
    assert roots[0].end_ns <= roots[1].start_ns <= roots[1].end_ns <= roots[2].start_ns
    parent = roots[0]
    for name in ('trace_codec', 'forward', 'trace_codec'):
        [child] = [s for s in spans if s.parent_id == parent.span_id and s.name == name]
        assert parent.start_ns <= child.start_ns <= child.end_ns <= parent.end_ns
        parent = child


@pytest.mark.parametrize('finish', ['close', 'return', 'error'])
@pytest.mark.parametrize('timed', [False, True])
def test_generator_forwards_exceptions_and_closes(finish, timed):
    closed = []

    class Recover(Policy):
        def run(self, runtime):
            try:
                yield
                try:
                    yield Step({}, 1)
                except ValueError:
                    yield Step({}, 2)
            finally:
                closed.append(True)

    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    with telemetry.timings_to(lambda *_: None) if timed else nullcontext():
        run = runtime.start(Recover())
        try:
            if finish != 'close':
                assert run.send({}) == Step({}, 1)
                assert run.throw(ValueError('recover')) == Step({}, 2)
                if finish == 'return':
                    with pytest.raises(StopIteration):
                        run.send({})
                else:
                    with pytest.raises(RuntimeError, match='failed'):
                        run.throw(RuntimeError('failed'))
        finally:
            runtime.close()
            run.close()
    assert closed == [True]
