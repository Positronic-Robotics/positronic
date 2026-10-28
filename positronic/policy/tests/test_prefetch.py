"""Prefetch selection, action cadence, recovery, and its plain-data declaration."""

import asyncio
import json
from concurrent.futures import Future

import numpy as np
import pytest
from positronic_model_server.spec import ARGS, LATEST, VERSION

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.offboard.cache import CACHE, MISSING, REFERENCES, SessionCache
from positronic.policy import keys as policy_keys
from positronic.policy.base import Policy, PolicyEnd, Processor, Step
from positronic.policy.executor import Executor, _UnchargedAnswer
from positronic.policy.processors import ChunkedSchedule, ConstantDuration, PauseOnUnavailable, Prefetch
from positronic.policy.sequential import Sequential
from positronic.policy.spec import from_spec

VALUE = 'value'
MOTOR = 'motor'
OFFSETS = (-0.4, -0.3, -0.2, -0.1, 0.0)
TICKS = [*range(0, 601, 50), 601, *range(650, 951, 50), 1003]


class Rig:
    def __init__(self, *, offsets=OFFSETS, lead=0.25, inference_lead=0.0, chunk_size=10, pad_start=True):
        self.now = 0
        self.runtime = Executor(lambda: self.now, simulated=True, charge_inference_time=False)
        self.calls = []
        self.requests = []
        self.cache = SessionCache()
        self.chunk_size = chunk_size
        self.upload: Future | None = None
        self.upload_result = None
        self.skip = False
        self.fail = False
        self.definition = Prefetch(
            ChunkedSchedule(
                fps=10, timing_key=policy_keys.NEXT_INFERENCE_NS, inference_lead=ConstantDuration(inference_lead)
            ),
            (VALUE,),
            offsets,
            lead,
            pad_start=pad_start,
        )
        self.run = self.runtime.start(self.definition, self.infer)
        self.runtime.submit = self.submit

    async def model(self, obs):
        self.calls.append((self.now // 1_000_000, False, obs[VALUE][:, 0].tolist()))
        return [{MOTOR: i} for i in range(self.chunk_size)]

    def infer(self, obs):
        self.requests.append(obs)
        if CACHE in obs:
            self.calls.append((self.now // 1_000_000, True, obs[CACHE][VALUE][:, 0].tolist()))
            if self.fail:
                raise RuntimeError('upload failed')
            if self.skip:
                return {}
        return asyncio.run(self.cache.call(obs, self.model))

    def submit(self, function, obs):
        if CACHE in obs and self.upload is not None:
            self.upload_result = function(obs)
            return _UnchargedAnswer(self.upload)
        answer = Future()
        answer.set_result(function(obs))
        return _UnchargedAnswer(answer)

    def step(self, ms, **extra):
        self.now = ms * 1_000_000
        step = self.run.send({VALUE: np.array([ms]), **extra})
        assert isinstance(step, Step)
        return step

    def drive(self, ticks=TICKS):
        return {ms: self.step(ms) for ms in ticks}

    def close(self):
        self.run.close()
        self.runtime.close()


@pytest.fixture
def rig():
    active = Rig()
    try:
        yield active
    finally:
        active.close()


def test_settled_prefix_and_fresh_tail_match_the_original_schedule(rig):
    steps = rig.drive()
    assert rig.calls == [(0, False, [0] * 5), (750, True, [600, 700]), (1003, False, [600, 700, 800, 900, 1003])]
    assert steps[700].resume_at_ns == 750_000_000
    assert rig.requests[-1][VALUE][:, 0].tolist() == [800, 900, 1003]
    assert set(rig.requests[-1][REFERENCES]) == {VALUE}
    assert [steps[t].commands[MOTOR] for t in range(0, 1000, 100)] == list(range(10))


def test_nested_definition_survives_json_and_has_no_shared_episode_state():
    rig = Rig(inference_lead=0.2)
    try:
        spec = json.loads(json.dumps(rig.definition.to_spec()))
        rebuilt = from_spec(spec)
        assert isinstance(rebuilt, Processor)
        assert rebuilt.to_spec() == spec
        rig.run.close()
        rig.run = rig.runtime.start(rebuilt, rig.infer)
        steps = rig.drive([*range(0, 401, 50), 401, *range(450, 751, 50), 803])
        assert steps[500].resume_at_ns == 550_000_000
        assert rig.calls == [(0, False, [0] * 5), (550, True, [400, 500]), (803, False, [400, 500, 600, 700, 803])]
        rig.run.close()
        rig.run = rig.runtime.start(rebuilt, rig.infer)
        rig.step(2000)
        assert rig.calls[-1] == (2000, False, [2000] * 5)
    finally:
        rig.close()


def test_skipped_upload_uses_fresh_history(rig):
    rig.skip = True
    rig.drive()
    assert rig.calls[-1] == (1003, False, [601, 700, 800, 900, 1003])


@pytest.mark.parametrize(
    'finished, frames', [(1403, [600, 700, 1200, 1300, 1403]), (2503, [2100, 2200, 2300, 2400, 2503])]
)
def test_slow_upload_keeps_sampling_and_playing_actions(rig, finished, frames):
    rig.upload = Future()
    steps = rig.drive(range(0, finished, 50))
    assert len(rig.calls) == 2
    assert steps[800].commands == {MOTOR: 8}
    assert steps[900].commands == {MOTOR: 9}
    rig.upload.set_result(rig.upload_result)
    rig.step(finished)
    assert rig.calls[-1] == (finished, False, frames)


def test_replanning_uses_fresh_history_even_after_a_successful_upload(rig):
    rig.drive([0, 400, 500, 600, 750])
    assert rig.step(800, **{keys.ROBOT_STATUS: RobotStatus.ERROR}).commands == {MOTOR: 0}
    assert rig.calls[-1] == (800, False, [400, 500, 600, 600, 800])


def test_no_settled_offset_sends_no_upload():
    rig = Rig(offsets=(-0.1, 0.0))
    try:
        rig.drive()
        assert rig.calls == [(0, False, [0, 0]), (1003, False, [900, 1003])]
    finally:
        rig.close()


def test_short_chunks_request_an_immediate_upload_wakeup():
    rig = Rig(offsets=(-0.4, 0.0), chunk_size=1)
    try:
        assert rig.step(0) == Step({MOTOR: 0}, 0)
        assert rig.step(0).resume_at_ns == 100_000_000
        rig.step(100)
        assert rig.calls == [(0, False, [0, 0]), (0, True, [0]), (100, False, [0, 100])]
    finally:
        rig.close()


def test_upload_failure_surfaces(rig):
    rig.fail = True
    with pytest.raises(RuntimeError, match='upload failed'):
        rig.drive()


def test_lost_cache_retries_the_same_selection_once(rig):
    rig.drive(TICKS[:-1])
    rig.cache = SessionCache()
    rig.step(1003)
    assert rig.calls[-1] == (1003, False, [600, 700, 800, 900, 1003])
    assert REFERENCES in rig.requests[-2] and REFERENCES not in rig.requests[-1]


def test_outer_fault_guard_remains_effective_during_upload(rig):
    rig.run.close()
    rig.run = rig.runtime.start(Sequential(PauseOnUnavailable(), rig.definition), rig.infer)
    rig.upload = Future()
    rig.step(0, **{keys.ROBOT_STATUS: RobotStatus.AVAILABLE})
    rig.step(750, **{keys.ROBOT_STATUS: RobotStatus.AVAILABLE})
    assert rig.step(800, **{keys.ROBOT_STATUS: RobotStatus.ERROR}) == Step({}, 801_000_000)
    rig.upload.set_result(rig.upload_result)
    assert rig.step(1000, **{keys.ROBOT_STATUS: RobotStatus.AVAILABLE}).commands == {MOTOR: 0}


def test_unpadded_history_omits_samples_before_episode_start():
    rig = Rig(offsets=(-2.0, -0.4, 0.0), pad_start=False)
    try:
        rig.drive()
        assert rig.calls == [(0, False, [0]), (750, True, [600]), (1003, False, [600, 1003])]
    finally:
        rig.close()


def test_wrapper_preserves_commands_other_outputs_and_child_completion(rig):
    class Done(Policy):
        def run(self, runtime, infer):
            yield
            yield Step({MOTOR: 7}, 123, outputs={MOTOR: 99, policy_keys.NEXT_INFERENCE_NS: None})
            return PolicyEnd.COMPLETE

    run = rig.runtime.start(Prefetch(Done(), (VALUE,), (0.0,), 0.2), rig.infer)
    assert run.send({VALUE: np.array([0])}) == Step({MOTOR: 7}, 123, outputs={MOTOR: 99})
    with pytest.raises(StopIteration) as end:
        run.send({VALUE: np.array([0])})
    assert end.value.value is PolicyEnd.COMPLETE


def test_server_cache_is_session_local_and_replaces_batches():
    async def exercise():
        first, second = SessionCache(), SessionCache()

        async def model(obs):
            return obs[VALUE].tolist()

        refs = await first.call({CACHE: {VALUE: np.array([[1]])}}, model)
        request = {REFERENCES: refs, VALUE: np.array([[2]])}
        assert await first.call(request, model) == [[1], [2]]
        assert await second.call(request, model) == {MISSING: True}
        await first.call({CACHE: {VALUE: np.array([[3]])}}, model)
        assert await first.call(request, model) == {MISSING: True}

    asyncio.run(exercise())


def test_close_cancels_an_upload(rig):
    rig.upload = Future()
    rig.drive(range(0, 751, 50))
    rig.run.close()
    assert rig.upload.cancelled()


def test_upstream_readiness_defers_inference(rig):
    rig.step(0, **{policy_keys.INFERENCE_READY: False})
    assert rig.calls == []
    rig.step(50, **{policy_keys.INFERENCE_READY: True})
    assert rig.calls == [(50, False, [0, 0, 0, 0, 50])]


def test_cached_frames_survive_source_mutation_and_history_eviction(rig):
    rig.upload = Future()
    frame = np.array([0])
    for ms in range(0, 751, 50):
        rig.now = ms * 1_000_000
        frame[0] = ms
        rig.run.send({VALUE: frame})
    frame[0] = -1
    rig.drive(range(800, 1401, 50))
    rig.upload.set_result(rig.upload_result)
    rig.step(1403)
    assert rig.calls[-1] == (1403, False, [600, 700, 1200, 1300, 1403])


def test_custom_control_keys_and_multiple_fields(rig):
    deadline_key, ready_key, second = 'deadline', 'ready', 'second'
    definition = Prefetch(
        ChunkedSchedule(fps=10, timing_key=deadline_key, ready_key=ready_key),
        (VALUE, second),
        OFFSETS,
        0.25,
        timing_key=deadline_key,
        ready_key=ready_key,
    )
    observed = []

    async def model(obs):
        observed.append(obs)
        return [{MOTOR: i} for i in range(10)]

    def infer(obs):
        return asyncio.run(rig.cache.call(obs, model))

    rig.run.close()
    rig.run = rig.runtime.start(definition, infer)
    for ms in TICKS:
        rig.step(ms, **{second: np.array([ms + 1]), 'state': ms})
    assert len(observed) == 2
    assert set(observed[-1]) == {VALUE, second, 'state'}
    assert observed[-1][VALUE][:, 0].tolist() == [600, 700, 800, 900, 1003]
    assert observed[-1][second][:, 0].tolist() == [601, 701, 801, 901, 1004]
    assert observed[-1]['state'] == 1003


def test_nested_latest_versions_resolve(rig):
    node = rig.definition.to_spec()
    node[VERSION] = node[ARGS][Prefetch.CHILD_ARG][VERSION] = LATEST
    assert from_spec(node).to_spec() == rig.definition.to_spec()


@pytest.mark.parametrize('offsets', [(), (0.1,), (0.0, -0.1), (-0.1, -0.1, 0.0), (float('nan'), 0.0)])
def test_invalid_offsets_are_rejected(offsets):
    with pytest.raises(ValueError, match='offsets'):
        Prefetch(ChunkedSchedule(fps=10), (VALUE,), offsets, 0.25)


def test_child_without_a_deadline_fails_clearly(rig):
    run = rig.runtime.start(Prefetch(ChunkedSchedule(fps=10), (VALUE,), OFFSETS, 0.25), rig.infer)
    with pytest.raises(ValueError, match='must publish'):
        run.send({VALUE: np.array([0]), policy_keys.INFERENCE_READY: False})


def test_a_second_cache_miss_fails_without_another_retry(rig):
    rig.drive(TICKS[:-1])
    requests = []

    async def missing(obs, infer):
        requests.append(obs)
        return {MISSING: True}

    rig.cache.call = missing
    with pytest.raises(RuntimeError, match='no cached references'):
        rig.step(1003)
    assert len(requests) == 2
    assert REFERENCES in requests[0] and REFERENCES not in requests[1]
    assert requests[1][VALUE][:, 0].tolist() == [600, 700, 800, 900, 1003]


def test_all_offsets_can_be_cached_when_no_current_frame_is_requested():
    rig = Rig(offsets=(-0.4, -0.3))
    try:
        rig.drive()
        assert rig.calls == [(0, False, [0, 0]), (750, True, [600, 700]), (1003, False, [600, 700])]
        assert rig.requests[-1][VALUE].shape == (0, 1)
    finally:
        rig.close()


def test_scheduler_publishes_deadline_separately_even_with_a_command_of_the_same_name(rig):
    schedule = ChunkedSchedule(fps=10, timing_key=MOTOR)
    run = rig.runtime.start(schedule, lambda obs: [{MOTOR: 7}])
    try:
        assert run.send({}) == Step({MOTOR: 7}, 100_000_000, outputs={MOTOR: 100_000_000})
    finally:
        run.close()
