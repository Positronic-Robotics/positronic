import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest

from positronic import telemetry, telemetry_keys
from positronic.cli.eval.finish_request import FinishRequest
from positronic.cli.eval.run import Driver, _pass_span, _timed_pass, main
from positronic.eval import Embodiment, Eval, Task

# `positronic.cli.eval` binds `run` to the config, shadowing the submodule of that name, so the module
# object whose globals these tests patch comes from `sys.modules`.
run_module = sys.modules['positronic.cli.eval.run']


def _eval(simulated: bool) -> Eval:
    return Eval(embodiment=cast(Embodiment, SimpleNamespace(simulated=simulated)), task=cast(Task, SimpleNamespace()))


def test_timed_sweep_rejects_real_embodiment(tmp_path):
    """``--timing`` with a real embodiment anywhere in the sweep fails up front: everything under the bound
    tracer enters the report, so a real eval's spans and wall time would silently corrupt it."""
    with pytest.raises(ValueError, match='all-simulated'):
        main(policy=object(), evals=[_eval(True), _eval(False)], output_dir=tmp_path, timing=True)


def test_timed_attended_run_rejects_real_embodiment(tmp_path):
    """The attended (driver) path runs one embodiment rather than a sweep, and reaches the same check."""
    real = cast(Embodiment, SimpleNamespace(simulated=False))
    with pytest.raises(ValueError, match='all-simulated'):
        main(
            policy=object(),
            embodiment=real,
            driver=lambda _: cast(Driver, SimpleNamespace()),
            output_dir=tmp_path,
            timing=True,
        )


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
    monkeypatch.setattr(run_module, '_pass_span', _recording_pass)

    with _timed_pass(tmp_path, True, object()):
        pass

    assert order == ['sampler built', 'pass in', 'sampler in', 'sampler out', 'pass out']


@pytest.mark.parametrize('simulated', [True, False])
def test_the_driver_is_scheduled_ahead_of_the_watcher_and_the_harness(monkeypatch, simulated):
    """The driver is scheduled before the watcher and the harness, on both embodiments."""
    scheduled = []

    class _FakeWorld:
        def __init__(self, virtual_time=False):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def connect(self, *args, **kwargs):
            pass

        def run(self, foreground, background=None):
            scheduled.extend(foreground)

    keyboard = SimpleNamespace(name='keyboard')
    harness = SimpleNamespace(
        directive=None, finish_requested=None, manual_command=None, ds_command=None, name='harness'
    )
    finish = SimpleNamespace(requested=None, name='watcher')
    monkeypatch.setattr(run_module.pimm, 'World', _FakeWorld)
    monkeypatch.setattr(run_module, 'Harness', lambda *args, **kwargs: harness)
    monkeypatch.setattr(run_module.wire, 'wire_embodiment', lambda *args, **kwargs: None)

    run_module._run_world(
        policy=object(),
        embodiment=cast(Embodiment, SimpleNamespace(simulated=simulated, control_systems=[], observations={})),
        task=None,
        trials=None,
        driver=cast(
            Driver,
            SimpleNamespace(
                gui=None, directives=None, directive_wrapper=None, control_systems=[keyboard], manual_commands=None
            ),
        ),
        output_dir=None,
        show_gui=False,
        on_complete=None,
        finish=cast(FinishRequest, finish),
    )

    assert scheduled[:3] == [keyboard, finish, harness]


def test_a_misconfigured_run_is_rejected_before_the_policy_is_warmed(monkeypatch):
    """`finish_request.from_env` raises on a run nothing can address. Reached after the warmup, that
    raise would leave a warmed policy — and whatever remote session it holds — with nothing to close
    it, since `main`'s `finally` has not been entered yet. So it is checked with the other up-front
    validation, and the warmup never runs."""
    warmed = []

    def new_session():
        warmed.append(True)
        return SimpleNamespace(close=lambda: None)

    policy = cast(object, SimpleNamespace(new_session=new_session, close=lambda: None))
    monkeypatch.setenv('ROLLOUT_RUN_ID', 'a/b')

    with pytest.raises(ValueError, match='single path segment'):
        main(policy=policy, evals=[_eval(True)])

    assert not warmed, 'the policy was warmed before the run was known to be addressable'
