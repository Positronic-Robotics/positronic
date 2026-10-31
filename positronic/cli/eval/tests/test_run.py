import importlib
import inspect
import logging
import os
import re
import sys
from contextlib import contextmanager
from functools import partial
from itertools import islice
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import pos3
import pytest
from eval_vocabulary.episode import ABC_REWARD_SIGNAL, SIGNAL_FILE_SUFFIX, SUBTASK_SIGNAL
from eval_vocabulary.timeline import HARNESS_WORLD
from positronic_evals.evals import MOLMO_EPISODE_INDEX_KEY, MOLMO_SCORING, PUBLIC_EVALS, EvalRef, public_eval
from positronic_evals.labels import (
    ABC_LABELER,
    MOLMO_LABELER,
    ROBOLAB_LABELER,
    Fail,
    Label,
    SubtaskAchieved,
    Success,
    TryWindow,
    abc_max_progress,
    label,
    recorded_try_window,
)
from positronic_evals.scoring import Scores, score

import pimm
from pimm.time import EMITTED_WALL, EMITTED_WORLD, Time
from positronic import keys, telemetry, telemetry_keys
from positronic.cfg.eval import number_trials, spec
from positronic.cfg.eval.real import droid as real_droid
from positronic.cfg.video_encoder import jetson_h264
from positronic.cli.account.gateway import CommandFailed, Fault
from positronic.cli.eval.run import TaskDriver, _pass_span, main, prepare_output_dir, scoped_env_var, timed_pass
from positronic.dataset.episode import META_UID, META_WRITER_VIDEO_ENCODER
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.vector import SimpleSignalWriter
from positronic.dataset.video import LibavEncoder
from positronic.eval import Embodiment, Eval, Task
from positronic.eval import keys as eval_keys
from positronic.policy import Policy, PolicyRun, Runtime, Step
from positronic.policy.harness import Rollout
from positronic.simulator.env_server.telemetry import ENV_TELEMETRY_DIR
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.tests.testing_inference import make_embodiment


def _eval(simulated: bool) -> Eval:
    return Eval(embodiment=cast(Embodiment, SimpleNamespace(simulated=simulated)), tasks=partial(iter, ()))


def test_timed_sweep_rejects_real_embodiment(tmp_path):
    """``--timing`` with a real embodiment anywhere in the sweep fails up front: everything under the bound
    tracer enters the report, so a real eval's spans and wall time would silently corrupt it."""
    with pytest.raises(CommandFailed, match='all-simulated') as exc:
        main(policy=object(), evals=[_eval(True), _eval(False)], output_dir=tmp_path, timing=True)
    assert exc.value.fault is Fault.USAGE


class _IdlePolicy(Policy):
    """A policy that asks for regular calls without emitting commands."""

    def run(self, runtime: Runtime) -> PolicyRun:
        yield
        while True:
            yield Step({}, runtime.time_ns + 100_000_000)


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


@pytest.mark.parametrize('record', [False, True])
def test_eval_uses_the_supplied_writer_only_when_recording(tmp_path, record):
    embodiment = make_embodiment(simulated=True)
    tasks = partial(iter, [Task('wait', timeout_sec=0.05)])
    encoder = LibavEncoder(options=(('preset', 'ultrafast'),))
    factory = Mock(wraps=partial(LocalDatasetWriter, video_encoder=encoder))
    output = tmp_path / 'recording'
    with pos3.mirror(cache_root=str(tmp_path / 'mirror'), show_progress=False):
        main(
            _IdlePolicy(),
            evals=[Eval(embodiment=embodiment, tasks=tasks)],
            output_dir=output if record else None,
            dataset_factory=factory,
        )
    if record:
        factory.assert_called_once_with(output)
        assert LocalDataset(output)[0].meta['writer'][META_WRITER_VIDEO_ENCODER] == repr(encoder)
    else:
        factory.assert_not_called()


def _record(output_dir, successes: list[bool]) -> list:
    """Record one MolmoSpaces episode per entry of `successes`, and return their directories."""
    paths = []
    with LocalDatasetWriter(output_dir) as writer:
        for success in successes:
            with writer.new_episode() as episode:
                episode.set_static(keys.TASK, 'pick up the cup.')
                episode.set_static(eval_keys.SUCCESS, success)
                episode.set_static(eval_keys.TERMINATED, True)
                for world_ns in (0, 1_000_000_000):
                    episode.append(keys.GRIP, 0.0, Time(**{EMITTED_WORLD: world_ns, EMITTED_WALL: world_ns}))
            paths.append(episode.path)
    return paths


def test_a_sweep_returns_the_episodes_it_recorded_and_not_the_earlier_ones(tmp_path, monkeypatch):
    dataset_dir = tmp_path / 'run'
    earlier = _record(dataset_dir, [False])
    swept: list = []
    monkeypatch.setattr(
        run_module, 'run_world', lambda *args, **kwargs: swept.extend(_record(dataset_dir, [True, True]))
    )

    with pos3.mirror(cache_root=str(tmp_path / 'mirror'), show_progress=False):
        recorded = main(policy=_IdlePolicy(), evals=[_eval(True)], output_dir=dataset_dir)

    assert recorded == swept
    assert not set(earlier) & set(recorded)


class _ArmThatStopsShort(pimm.ControlSystem):
    """An arm whose every move stops short. It counts the moves it is asked for."""

    def __init__(self):
        self.move = pimm.calls.ControlSystemHandler[object, None](self)
        self.asked = 0

    def run(self, should_stop, clock):
        while not should_stop.value:
            for call in self.move.incoming():
                self.asked += 1
                call.set_exception(RuntimeError('the arm stopped short of its target'))
            yield pimm.Sleep(0.01)


@pytest.mark.timeout(30.0)
def test_a_home_that_stops_short_ends_an_unattended_run():
    arm = _ArmThatStopsShort()
    embodiment = Embodiment(
        descriptor='stub',
        observations={},
        commands={},
        prepare_handlers={eval_keys.ARM: arm.move},
        static_meta={},
        meta_source=None,
        control_systems=(arm,),
        simulated=True,
    )
    tasks = [Task(instruction_source='stack', timeout_sec=0.05, prepare_args={eval_keys.ARM: 'home'})] * 2
    with pytest.raises(RuntimeError, match='stopped short'):
        main(policy=_IdlePolicy(), evals=[Eval(embodiment=embodiment, tasks=partial(iter, tasks))])
    assert arm.asked == 1


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
        for _ in islice(world.start([driver, stub]), 200):
            pass

    assert stub.asked == tasks


class _ScriptedEpisodes(pimm.ControlSystem):
    """Stands in for the harness: records each task it is asked for, and answers with the next outcome."""

    def __init__(self, outcomes: list[dict | Exception]):
        self.asked: list[Task] = []
        self._outcomes = list(outcomes)
        self.perform_task = pimm.calls.ControlSystemHandler[Rollout, dict](self)

    def run(self, should_stop, clock):
        while not should_stop.value:
            for call in self.perform_task.incoming():
                self.asked.append(call.request.task)
                outcome = self._outcomes.pop(0)
                if isinstance(outcome, Exception):
                    call.set_exception(outcome)
                else:
                    call.set_result(outcome)
            yield pimm.Sleep(0.01)


def _drive(tasks: list[Task], episodes: _ScriptedEpisodes) -> None:
    driver = TaskDriver(partial(iter, tasks), _IdlePolicy(), None)
    with pimm.World(virtual_time=True) as world:
        world.connect(driver.perform_task, episodes.perform_task)
        for _ in islice(world.start([driver, episodes]), 200):
            pass


def _tasks(count: int) -> list[Task]:
    return [Task(instruction_source='stack', timeout_sec=0.05, meta={eval_keys.TRIAL_INDEX: i}) for i in range(count)]


@pytest.mark.timeout(3.0)
@pytest.mark.parametrize(
    'failure', [pimm.SignalError('camera lost'), pimm.calls.HandlerStopped(), RuntimeError('the arm holds an error')]
)
def test_a_failed_episode_ends_the_run_and_no_task_runs_twice(failure):
    tasks = _tasks(2)
    episodes = _ScriptedEpisodes([failure])
    with pytest.raises(type(failure)):
        _drive(tasks, episodes)

    assert episodes.asked == [tasks[0]]


# `positronic.cli.eval` exports a command named `run`, which takes the attribute path to this module.
run_module = importlib.import_module('positronic.cli.eval.run')


@pytest.mark.parametrize('states, charged', [({}, True), ({'charge_inference_time': False}, False)])
def test_a_local_run_stamps_its_charge_on_every_task(run_command, monkeypatch, states: dict, charged: bool):
    """A local run charges inference time unless it states otherwise, and stamps that on every task the
    eval makes."""
    made = [Task(instruction_source='t', timeout_sec=1.0) for _ in range(2)]
    seen: list[Task] = []

    def capture(policy, evals, output_dir, timing, dataset_factory):
        seen.extend(evals[0].tasks())

    monkeypatch.setattr(run_module, 'main', capture)
    eval_cfg = Eval(embodiment=cast(Embodiment, SimpleNamespace(simulated=True)), tasks=partial(iter, made))

    run_command(run_module.run, eval=eval_cfg, policy='a policy', **states)

    assert [task.charge_inference_time for task in seen] == [charged, charged]


SMOKE = 'molmo.franka_pick_mini_smoke'


def test_local_eval_accepts_a_writer_preset_from_the_cli(run_command, monkeypatch):
    launch = Mock()
    monkeypatch.setattr(run_module, 'main', launch)
    run_command(run_module.run, eval=SMOKE, policy='a policy', dataset_factory='@positronic.cfg.dataset_writer.jetson')
    factory = launch.call_args.kwargs['dataset_factory']
    assert factory.func is LocalDatasetWriter
    assert factory.keywords == {'video_encoder': jetson_h264.instantiate()}


def test_a_local_run_of_a_public_eval_runs_the_config_its_definition_names(run_command, monkeypatch):
    seen: list[Eval] = []
    monkeypatch.setattr(
        run_module, 'main', lambda policy, evals, output_dir, timing, dataset_factory: seen.extend(evals)
    )

    run_command(run_module.run, eval=SMOKE, policy='a policy')

    assert [ev.embodiment.descriptor for ev in seen] == ['remote.molmo_spaces.droid']


def test_a_local_run_of_a_public_eval_prints_the_score_of_the_episodes_it_recorded(
    run_command, monkeypatch, capsys, tmp_path
):
    _record(tmp_path, [False, False])  # an earlier run's episodes, which the sweep does not return
    monkeypatch.setattr(
        run_module,
        'main',
        lambda policy, evals, output_dir, timing, dataset_factory: _record(output_dir, [True, False, True, True]),
    )

    run_command(run_module.run, eval=SMOKE, policy='a policy', output_dir=tmp_path)

    expected = run_module.score_line(Scores(primary=0.75, episodes=4, unscored=0))
    assert capsys.readouterr().out.splitlines()[-1] == expected == 'score 0.750 over 4 episodes, 0 unscored'


def test_a_local_run_of_a_public_eval_without_an_output_dir_prints_no_score(run_command, monkeypatch, capsys, caplog):
    monkeypatch.setattr(run_module, 'main', lambda policy, evals, output_dir, timing, dataset_factory: [])

    with caplog.at_level(logging.WARNING, logger=run_module.logger.name):
        run_command(run_module.run, eval=SMOKE, policy='a policy')

    assert capsys.readouterr().out == ''
    assert 'pass --output_dir' in caplog.text


@pytest.mark.parametrize('name', PUBLIC_EVALS)
def test_a_public_eval_binds_its_arguments_on_the_installed_positronic(name: str):
    definition = public_eval(EvalRef(name))

    config = run_module._public_eval_config(definition)

    eval_config = config.kwargs['eval']
    assert set(definition.args) <= set(inspect.signature(eval_config.target).parameters)
    assert eval_config.kwargs.items() >= definition.args.items()
    assert isinstance(config.instantiate(), Eval)


def test_the_molmo_trial_key_of_a_public_eval_is_the_one_molmo_spaces_reads():
    assert MOLMO_EPISODE_INDEX_KEY == molmo_keys.EPISODE_INDEX


def test_the_molmo_labeler_reads_the_outcome_that_positronic_records(tmp_path):
    with LocalDatasetWriter(tmp_path) as writer, writer.new_episode() as episode:
        episode.set_static(keys.TASK, 'pick up the cup.')
        episode.set_static(eval_keys.SUCCESS, True)
        episode.set_static(eval_keys.TERMINATED, True)
        # Ten seconds pass on the wall clock for each second on the world clock, as in a sim slower than real time.
        for world_ns in (2_000_000_000, 4_000_000_000, 6_500_000_000):
            episode.append(keys.GRIP, 0.0, Time(**{EMITTED_WORLD: world_ns, EMITTED_WALL: 10 * world_ns}))

    given = label(episode.path, MOLMO_SCORING.labelers)

    assert given is not None
    labeler, given_label = given
    assert labeler == MOLMO_LABELER
    assert given_label == Label(
        uid=episode.meta[META_UID],
        task='pick up the cup.',
        annotator=MOLMO_LABELER,
        seconds_to_end=4.5,
        events=(Success(seconds=4.5),),
    )
    assert score(MOLMO_SCORING, [given_label]).model_dump() == {'primary': 1.0, 'episodes': 1, 'unscored': 0}


def test_the_try_window_reads_an_episode_that_positronic_records(tmp_path):
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    with LocalDatasetWriter(tmp_path) as writer, writer.new_episode() as episode:
        # The first grip sample is a held reading, 400 s before the try.
        for world_ns in (0, 400_000_000_000, 400_020_000_000, 460_000_000_000):
            episode.append(keys.GRIP, 0.0, Time(**{HARNESS_WORLD: world_ns}))
        for world_ns in (400_010_000_000, 400_043_000_000, 461_000_000_000):
            episode.append(keys.EXTERIOR_IMAGE, frame, Time(**{HARNESS_WORLD: world_ns}))
    with SimpleSignalWriter(episode.path / f'{keys.TARGET_GRIP}{SIGNAL_FILE_SUFFIX}'):
        pass

    assert recorded_try_window(episode.path) == TryWindow(400_000_000_000, 461_000_000_000)


def test_the_robolab_labeler_reads_the_progress_that_positronic_records(tmp_path):
    with LocalDatasetWriter(tmp_path) as writer, writer.new_episode() as episode:
        episode.set_static(keys.TASK, 'put all the cubes and blocks in the bin')
        episode.set_static(eval_keys.TASK, 'CubesAndBlocksInBinTask')
        episode.set_static(eval_keys.SUCCESS, False)
        episode.set_static(eval_keys.TERMINATED, True)
        for world_ns, progress in ((0, [0, 0, 2, 0]), (1_000_000_000, [0, 1, 2, 0.5]), (2_000_000_000, [0, 1, 2, 0.5])):
            subtask = np.array(progress, dtype=np.float32)
            episode.append(SUBTASK_SIGNAL, subtask, Time(**{EMITTED_WORLD: world_ns, EMITTED_WALL: 10 * world_ns}))

    assert label(episode.path, (ROBOLAB_LABELER,)) == (
        ROBOLAB_LABELER,
        Label(
            uid=episode.meta[META_UID],
            task='CubesAndBlocksInBinTask',
            annotator=ROBOLAB_LABELER,
            seconds_to_end=2.0,
            total_subtasks=2,
            events=(SubtaskAchieved(seconds=1.0, achieved=1, score=0.5), Fail(seconds=2.0)),
        ),
    )


def test_the_abc_labeler_reads_the_outcome_and_the_progress_that_positronic_records(tmp_path):
    with LocalDatasetWriter(tmp_path) as writer, writer.new_episode() as episode:
        episode.set_static(keys.TASK, 'put the plastic bottles in the bin')
        episode.set_static(eval_keys.TASK, 'put_plastic_bottles_in_bin')
        episode.set_static(eval_keys.TERMINATED, False)
        for world_ns, reward in ((0, 0.0), (1_000_000_000, 0.5), (2_000_000_000, 0.25)):
            episode.append(ABC_REWARD_SIGNAL, reward, Time(**{EMITTED_WORLD: world_ns, EMITTED_WALL: 10 * world_ns}))

    assert label(episode.path, (ABC_LABELER,)) == (
        ABC_LABELER,
        Label(uid=episode.meta[META_UID], task='put_plastic_bottles_in_bin', annotator=ABC_LABELER, seconds_to_end=2.0),
    )
    assert abc_max_progress(episode.path) == 0.5


@pytest.mark.parametrize('name', ['molmo.held_out', ''])
def test_a_local_run_refuses_a_name_the_public_code_does_not_define(run_command, name: str):
    with pytest.raises(SystemExit, match=f'--eval={name!r}: .*--policy-image runs an eval on the platform'):
        run_command(run_module.run, eval=name, policy='a policy')


@pytest.mark.parametrize('value', [None, 1])
def test_a_local_run_refuses_an_eval_that_is_neither_a_config_nor_a_name(run_command, value: object):
    with pytest.raises(CommandFailed, match=f'--eval={value!r} names nothing to run') as exit_info:
        run_command(run_module.run, eval=value, policy='a policy')
    assert exit_info.value.fault is Fault.USAGE


PINNED = public_eval(EvalRef(SMOKE)).positronic_revision


@pytest.mark.parametrize(
    'installed, warns',
    [
        ({'commit': PINNED, 'dirty': False}, False),
        ({'commit': PINNED, 'dirty': True}, True),
        ({'commit': '0' * 40, 'dirty': False}, True),
        (None, True),
    ],
)
def test_a_public_eval_warns_unless_the_installed_positronic_is_the_platforms(
    monkeypatch, caplog, installed: dict | None, warns: bool
):
    monkeypatch.setattr(run_module, 'get_package_git_state', lambda: installed)

    with caplog.at_level(logging.WARNING, logger=run_module.logger.name):
        run_module._public_eval_definition(SMOKE)

    assert (PINNED in caplog.text) is warns


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


def test_a_real_droid_trial_homes_the_arm_with_no_control_law():
    """The arm's own controller runs the home, on the attended path and on a planned sweep."""
    attended = real_droid.attended_trials.instantiate()()
    planned = real_droid.pick_place.override(embodiment=None, trial_count=2).instantiate().tasks()
    for task in [attended, *planned]:
        assert task.prepare_args[eval_keys.ARM].mode is None


def test_timed_sweep_needs_an_output_dir():
    """There is nowhere to write the sidecars without one, so the sweep is rejected before it spends anything."""
    with pytest.raises(CommandFailed, match='output_dir') as exc:
        main(policy=object(), evals=[_eval(True)], timing=True)
    assert exc.value.fault is Fault.USAGE


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
    """Every binary that records resolves its output directory here, so each one writes its spans where
    `pos3.sync` mirrors them without being told."""
    monkeypatch.delenv(ENV_TELEMETRY_DIR, raising=False)
    with pos3.mirror(cache_root=str(tmp_path / 'mirror'), show_progress=False):
        with scoped_env_var(ENV_TELEMETRY_DIR):
            local_dir = prepare_output_dir(tmp_path / 'episodes')
            pointed_at = os.environ[ENV_TELEMETRY_DIR]

    assert local_dir is not None
    assert pointed_at == str(local_dir / telemetry.TELEMETRY_SUBDIR)


def test_a_run_that_records_nothing_leaves_no_telemetry_directory(tmp_path, monkeypatch):
    """A run recording nowhere has nowhere to put spans, and takes no earlier run's directory."""
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path / 'an-earlier-run' / 'telemetry'))

    assert prepare_output_dir(None) is None
    assert ENV_TELEMETRY_DIR not in os.environ


def test_an_output_dir_it_cannot_create_is_a_usage_fault_naming_it(tmp_path):
    (tmp_path / 'a-file').write_text('')
    output_dir = tmp_path / 'a-file' / 'episodes'

    with pos3.mirror(cache_root=str(tmp_path / 'mirror'), show_progress=False), scoped_env_var(ENV_TELEMETRY_DIR):
        with pytest.raises(CommandFailed, match=re.escape(str(output_dir))) as exc:
            prepare_output_dir(output_dir)

    assert exc.value.fault is Fault.USAGE


def test_a_failure_outside_the_output_dir_is_not_reported_as_one(tmp_path, monkeypatch):
    # Recording the run reads files of its own, such as `uv.lock`; a failure there is a defect, not a bad argument.
    def unreadable_lockfile(*_args, **_kwargs):
        raise PermissionError(13, 'Permission denied', str(tmp_path / 'elsewhere' / 'uv.lock'))

    monkeypatch.setattr('positronic.utils.save_run_metadata', unreadable_lockfile)

    with pos3.mirror(cache_root=str(tmp_path / 'mirror'), show_progress=False), scoped_env_var(ENV_TELEMETRY_DIR):
        with pytest.raises(PermissionError):
            prepare_output_dir(tmp_path / 'episodes')


def test_a_recorded_run_leaves_no_telemetry_directory_behind(tmp_path, monkeypatch):
    """A destination that outlives its run reaches a later `Harness` in the same process."""
    monkeypatch.delenv(ENV_TELEMETRY_DIR, raising=False)

    with pos3.mirror(cache_root=str(tmp_path / 'mirror'), show_progress=False):
        with scoped_env_var(ENV_TELEMETRY_DIR):
            prepare_output_dir(tmp_path / 'episodes')

    assert ENV_TELEMETRY_DIR not in os.environ


def test_the_scope_gives_back_a_directory_its_caller_set(tmp_path, monkeypatch):
    """The scope restores rather than clears, so it cannot reach past the value it replaced."""
    monkeypatch.setenv(ENV_TELEMETRY_DIR, str(tmp_path / 'mine'))

    with pos3.mirror(cache_root=str(tmp_path / 'mirror'), show_progress=False):
        with scoped_env_var(ENV_TELEMETRY_DIR):
            prepare_output_dir(tmp_path / 'episodes')

    assert os.environ[ENV_TELEMETRY_DIR] == str(tmp_path / 'mine')
