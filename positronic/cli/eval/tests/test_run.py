import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest

import pimm
from positronic import telemetry, telemetry_keys
from positronic.cli.eval.run import Operator, _pass_span, _timed_pass, main
from positronic.eval import Embodiment, Eval, Task


def _eval(simulated: bool) -> Eval:
    return Eval(embodiment=cast(Embodiment, SimpleNamespace(simulated=simulated)), task=cast(Task, SimpleNamespace()))


def test_timed_sweep_rejects_real_embodiment(tmp_path):
    """``--timing`` with a real embodiment anywhere in the sweep fails up front: everything under the bound
    tracer enters the report, so a real eval's spans and wall time would silently corrupt it."""
    with pytest.raises(ValueError, match='all-simulated'):
        main(policy=object(), evals=[_eval(True), _eval(False)], output_dir=tmp_path, timing=True)


def test_timed_attended_run_rejects_real_embodiment(tmp_path):
    """The attended path runs one embodiment rather than a sweep, and reaches the same check."""
    real = cast(Embodiment, SimpleNamespace(simulated=False))
    with pytest.raises(ValueError, match='all-simulated'):
        main(
            policy=object(),
            embodiment=real,
            operator=lambda _: cast(Operator, SimpleNamespace()),
            output_dir=tmp_path,
            timing=True,
        )


class _IdlePolicy:
    """Enough policy for ``main`` to warm up and close; it is never asked for an action."""

    def new_session(self, *_args, **_kwargs):
        return SimpleNamespace(close=lambda: None)

    def close(self):
        pass


class _FinishingOperator(pimm.ControlSystem):
    """An operator surface that paces a few rounds and then returns, emitting nothing."""

    def __init__(self, rounds: int = 3):
        self._rounds = rounds
        self.directive = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        for _ in range(self._rounds):
            yield pimm.Sleep(0.01)


@pytest.mark.timeout(30.0)
def test_an_operator_control_system_returning_ends_the_run():
    """How an attended run finishes: an operator's foreground loop returns, the world stops, ``main`` returns.

    An operator that never returns can only be ended from outside the run, so this is the seam's whole
    finish contract — anything the surface must do first happens before its loop returns.
    """
    embodiment = Embodiment(
        descriptor='stub', observations={}, commands={}, static_meta={}, meta_source=None, simulated=True
    )
    surface = _FinishingOperator()
    main(
        policy=_IdlePolicy(),
        embodiment=embodiment,
        operator=lambda _: Operator(None, surface.directive, pimm.utils.identity, [surface]),
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
    # `positronic.cli.eval` binds `run` to the config, shadowing the submodule of that name, so the module
    # object whose global is being replaced comes from `sys.modules`.
    monkeypatch.setattr(sys.modules['positronic.cli.eval.run'], '_pass_span', _recording_pass)

    with _timed_pass(tmp_path, True, object()):
        pass

    assert order == ['sampler built', 'pass in', 'sampler in', 'sampler out', 'pass out']
