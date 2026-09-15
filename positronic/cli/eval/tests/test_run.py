import importlib
import os
import sys
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
from typing import cast

import pos3
import pytest

import pimm
from positronic import telemetry, telemetry_keys
from positronic.cfg.eval import number_trials, spec
from positronic.cli.eval.run import TaskDriver, _pass_span, main, prepare_output_dir, timed_pass
from positronic.eval import Embodiment, Eval, Task
from positronic.eval import keys as eval_keys
from positronic.policy import Policy, Session
from positronic.policy.harness import Rollout
from positronic.simulator.env_server.telemetry import ENV_TELEMETRY_DIR
from positronic.tests.testing_coutils import IdleSession, drive_scheduler


def _eval(simulated: bool) -> Eval:
    return Eval(embodiment=cast(Embodiment, SimpleNamespace(simulated=simulated)), tasks=partial(iter, ()))


def test_timed_sweep_rejects_real_embodiment(tmp_path):
    """``--timing`` with a real embodiment anywhere in the sweep fails up front: everything under the bound
    tracer enters the report, so a real eval's spans and wall time would silently corrupt it."""
    with pytest.raises(ValueError, match='all-simulated'):
        main(policy=object(), evals=[_eval(True), _eval(False)], output_dir=tmp_path, timing=True)


class _IdlePolicy(Policy):
    """Enough policy for ``main`` to warm up and close; it is never asked for an action."""

    def __init__(self):
        self.observations: list[dict] = []

    def new_session(self, *_args, **_kwargs) -> Session:
        return IdleSession(self)

    def close(self):
        pass


@pytest.mark.timeout(30.0)
def test_an_exhausted_trial_plan_ends_the_sweep():
    """How an unattended run finishes: the driver runs out of tasks, the world stops, ``main`` returns."""
    embodiment = Embodiment(
        descriptor='stub',
        observations={},
        commands={},
        prepare_handlers={},
        static_meta={},
        meta_source=None,
        simulated=True,
    )
    main(policy=_IdlePolicy(), evals=[Eval(embodiment=embodiment, tasks=partial(iter, ()))])


class _EpisodeStub(pimm.ControlSystem):
    """Stands in for the harness: records the task it was asked for, and answers a round later."""

    def __init__(self):
        self.asked: list[Task] = []
        self.perform_task = pimm.calls.ControlSystemHandler[Rollout, dict](self)

    def run(self, should_stop, clock):
        while not should_stop.value:
            for call in self.perform_task.incoming():
                self.asked.append(call.request.task)
                yield pimm.Sleep(0.01)  # an episode takes a round to run
                assert not list(self.perform_task.incoming()), 'a task was asked for while one was running'
                call.set_result({})
            yield pimm.Sleep(0.01)


@pytest.mark.timeout(3.0)
def test_the_driver_asks_for_its_tasks_one_at_a_time():
    """The plan belongs to the driver: it asks for each task in turn, and only once the running episode has
    answered."""
    tasks = [Task(instruction_source='stack', timeout_sec=0.05, meta={eval_keys.TRIAL_INDEX: i}) for i in range(2)]
    stub = _EpisodeStub()
    driver = TaskDriver(partial(iter, tasks), _IdlePolicy(), None)
    with pimm.World(virtual_time=True) as world:
        world.connect(driver.perform_task, stub.perform_task)
        drive_scheduler(world.start([driver, stub]), steps=200)

    assert stub.asked == tasks


# `positronic.cli.eval` exports a command named `run`, which takes the attribute path to this module.
run_module = importlib.import_module('positronic.cli.eval.run')


@pytest.mark.parametrize('states, charged', [({}, True), ({'charge_inference_time': False}, False)])
def test_a_local_run_stamps_its_charge_on_every_task(run_command, monkeypatch, states: dict, charged: bool):
    """A local run charges inference time unless it states otherwise, and stamps that on every task the
    eval makes."""
    made = [Task(instruction_source='t', timeout_sec=1.0) for _ in range(2)]
    seen: list[Task] = []

    def capture(policy, evals, output_dir, timing):
        seen.extend(evals[0].tasks())

    monkeypatch.setattr(run_module, 'main', capture)
    eval_cfg = Eval(embodiment=cast(Embodiment, SimpleNamespace(simulated=True)), tasks=partial(iter, made))

    run_command(run_module.run, eval=eval_cfg, policy='a policy', **states)

    assert [task.charge_inference_time for task in seen] == [charged, charged]


def test_a_spec_carries_only_what_the_eval_binds():
    """An eval leaves an axis unbound to run every value of it, and the env reads an absent key as that."""
    assert spec(suite='libero_spatial', task_id=None) == {'suite': 'libero_spatial'}
    assert spec(task_id=0) == {'task_id': 0}, 'zero is a bound task, not an unbound axis'
    assert spec(task=None) == {}


def test_a_sweep_numbers_its_trials_across_every_task():
    """Trials of tasks that differ are numbered once over the whole plan."""
    quick = Task(instruction_source='quick', timeout_sec=1.0)
    slow = Task(instruction_source='slow', timeout_sec=90.0)
    pairs = [(quick, {eval_keys.TASK: 'quick'}), (slow, {eval_keys.TASK: 'slow'}), (slow, {eval_keys.TASK: 'slow'})]
    trials = number_trials(pairs)

    assert [t.timeout_sec for t in trials] == [1.0, 90.0, 90.0]
    assert [t.meta[eval_keys.TRIAL_INDEX] for t in trials] == [0, 1, 2]
    assert [t.meta[eval_keys.TRIAL_COUNT] for t in trials] == [3, 3, 3]
    assert [t.meta[eval_keys.TASK] for t in trials] == ['quick', 'slow', 'slow']
    assert [t.prepare_args[eval_keys.SCENE] for t in trials] == [params for _, params in pairs]


def test_timed_sweep_needs_an_output_dir():
    """There is nowhere to write the sidecars without one, so the sweep is rejected before it spends anything."""
    with pytest.raises(ValueError, match='output_dir'):
        main(policy=object(), evals=[_eval(True)], timing=True)


def test_failed_pass_exported_and_stamped(tmp_path):
    """A sweep that dies mid-pass still exports its pass span — the partial window is real recorded data —
    stamped ``pass.failed`` so the reduce can name the mix instead of silently folding it in."""
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-fail'):
        with pytest.raises(RuntimeError):
            with _pass_span(**{telemetry.ATTR_RUN_ID: 'run-fail'}):
                raise RuntimeError('sim died')

    spans = {s.name: s for s in telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS))}
    assert spans[telemetry_keys.SPAN_EVAL_PASS].attrs.get(telemetry_keys.ATTR_PASS_FAILED) is True


def test_the_stats_sampler_runs_inside_the_pass_span(tmp_path, monkeypatch):
    """Sampling is bounded by the pass span, so every sample the reduce sees falls in the window it counts.
    Sampling around the span instead leaves its first and last outside — and a run shorter than the sampling
    interval loses its only sample, reporting a GPU box as CPU-only. Asserted as nesting rather than as
    timestamps because the overlap is a thread race: it is the order that makes it impossible.

    Construction sits outside the span for the opposite reason: NVML init and counter priming are setup, not
    eval wall, and charging them to W_pass depresses the real-time factor."""
    order = []

    class _RecordingSampler:
        def __init__(self, path):
            order.append('sampler built')

        def __enter__(self):
            order.append('sampler in')
            return self

        def __exit__(self, *exc):
            order.append('sampler out')

    @contextmanager
    def _recording_pass(**attrs):
        order.append('pass in')
        yield
        order.append('pass out')

    monkeypatch.setattr(telemetry, 'StatsSampler', _RecordingSampler)
    # `positronic.cli.eval` binds `run` to the config, shadowing the submodule of that name, so the module
    # object whose global is being replaced comes from `sys.modules`.
    monkeypatch.setattr(sys.modules['positronic.cli.eval.run'], '_pass_span', _recording_pass)

    with timed_pass(tmp_path, True, object()):
        pass

    assert order == ['sampler built', 'pass in', 'sampler in', 'sampler out', 'pass out']


def test_the_spans_sidecar_lands_where_the_episodes_upload_from(tmp_path, monkeypatch):
    """`pos3.sync` mirrors the whole local directory, so a sidecar inside it uploads with the episodes.
    Every binary that records resolves its output directory here, so each one gets that without being
    told: the eval sweep, `positronic.inference`, and the rig and sim binaries that import this."""
    monkeypatch.delenv(ENV_TELEMETRY_DIR, raising=False)
    with pos3.mirror(cache_root=str(tmp_path / 'mirror'), show_progress=False):
        local_dir = prepare_output_dir(tmp_path / 'episodes')

    assert local_dir is not None
    assert os.environ[ENV_TELEMETRY_DIR] == str(local_dir / telemetry.TELEMETRY_SUBDIR)


def test_a_run_that_records_nothing_leaves_no_telemetry_directory(tmp_path, monkeypatch):
    """The directory has one owner, in both directions: a run with no output dir writes no spans, and never
    into the directory an earlier run in this process named."""
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path / 'an-earlier-run' / 'telemetry'))

    assert prepare_output_dir(None) is None
    assert ENV_TELEMETRY_DIR not in os.environ
