"""The DAgger reader: the training episodes of the steps that the teleoperator drove in the records of a DAgger round.

``DaggerDataset`` skips each record that the license rule (``positronic.policy.license``) drops, and counts it in
``dropped_records``.

The reader finds the samples of each step of a record by the step number on ``harness.step``. The steps are the
numbers from the step number of the first ``control.source`` sample to the highest step number of a label sample. Each
run of consecutive steps whose ``control.source`` names the teleoperator and whose instruction (``eval.instruction``) is
the same becomes one training episode. Its ``task`` is that instruction. Each label and each shadow action
``shadow.<name>.<channel>`` of that episode has its samples at the steps of the run. Each observation has one sample at
each step: the last value at or before the step. On ``harness.step`` and
``harness.world``, the samples of one step have one time: the step number, and the latest ``harness.world`` time of the
samples that the training episode used up to that step. Each other timeline keeps the time of its sample.
``steps_without_shadow`` counts the steps where a source has no shadow action::

    data = DaggerDataset(
        LocalDataset(round_dir), embodiment.observations, ('robot_command.pose',), LicenseRule({'apache-2.0'})
    )
    TransformedDataset(data, codec.training_encoder)
"""

import logging
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby

import numpy as np

from pimm import Time
from positronic import keys
from positronic.dataset.dataset import Dataset
from positronic.dataset.episode import META_UID, Episode, EpisodeContainer
from positronic.dataset.signal import Signal
from positronic.drivers.recording import HARNESS_STEP, HARNESS_WORLD
from positronic.eval import keys as eval_keys
from positronic.policy.keys import CONTROL_SOURCE, SHADOW, TELEOPERATOR
from positronic.policy.license import LicenseRule

_SHADOW_PREFIX = f'{SHADOW}.'


@dataclass(frozen=True)
class TeleoperatorRun:
    """One training episode: the steps of one teleoperator run. ``steps_without_shadow`` counts, for each source with
    a shadow action in the record, the steps of the run where it has none."""

    episode: Episode
    steps_without_shadow: dict[str, int]


def _steps(signal: Signal) -> np.ndarray:
    return np.asarray(signal.timestamps(HARNESS_STEP))


def _last_at(signal: Signal, steps: np.ndarray) -> np.ndarray:
    """The index of the last sample of ``signal`` at or before each of ``steps``, or -1 when it has none."""
    return np.searchsorted(_steps(signal), steps, side='right') - 1


def _step_times(used: Collection[tuple[Signal, np.ndarray]], steps: np.ndarray) -> list[Time]:
    """The time of each of ``steps`` in a training episode: the step number, and the latest ``harness.world`` time of
    the samples up to that step. ``used`` holds each signal with the mask of the steps that use it. The signal has a
    sample at or before each such step."""
    latest = np.full(len(steps), np.iinfo(np.int64).min)
    for signal, uses in used:
        world = np.asarray(signal.timestamps(HARNESS_WORLD))[_last_at(signal, steps)]
        latest = np.where(uses, np.maximum(latest, world), latest)
    return [
        Time(**{HARNESS_STEP: int(step), HARNESS_WORLD: int(ns)})
        for step, ns in zip(steps, np.maximum.accumulate(latest), strict=True)
    ]


def _is_observation(name: str, observations: Collection[str]) -> bool:
    return any(name == base or name.startswith(f'{base}.') for base in observations)


def _shadow_source(name: str) -> str:
    return name.removeprefix(_SHADOW_PREFIX).split('.', 1)[0]


def _instructions(record: Episode, steps: np.ndarray) -> list[str | None]:
    """The instruction at each of ``steps``: the last ``eval.instruction`` sample at or before the step, or ``None``
    when it has none."""
    if eval_keys.INSTRUCTION not in record:
        return [None] * len(steps)
    signal = record[eval_keys.INSTRUCTION]
    values = signal.values()
    return [values[i] if i >= 0 else None for i in _last_at(signal, steps)]


def teleoperator_runs(record: Episode, observations: Collection[str], labels: Sequence[str]) -> list[TeleoperatorRun]:
    """The training episode of each teleoperator run in ``record``, in order. A signal is an observation when its
    name is one of ``observations``, or one of them with a suffix that starts with ``.``."""
    if not labels:
        raise ValueError('The reader takes at least one label')
    uid = record.meta.get(META_UID)
    if CONTROL_SOURCE not in record or TELEOPERATOR not in record[CONTROL_SOURCE].values():
        return []
    if HARNESS_STEP not in record[CONTROL_SOURCE].timelines:
        raise ValueError(
            f'The record {uid} has no step numbers on {HARNESS_STEP}: a harness wrote it before it numbered the steps. '
            'Leave the record out of the round, or record it again'
        )
    signals = record.signals
    label_steps = {label: _steps(record[label]) for label in labels}
    last = max((int(sent[-1]) for sent in label_steps.values() if len(sent)), default=None)
    if last is None:
        return []
    control = record[CONTROL_SOURCE]
    steps = np.arange(_steps(control)[0], last + 1)
    values = control.values()
    sources = [values[i] for i in _last_at(control, steps)]
    instructions = _instructions(record, steps)
    sent = {label: np.isin(steps, label_steps[label]) for label in labels}
    shadowed = {
        name: np.isin(steps, _steps(signal)) for name, signal in signals.items() if name.startswith(_SHADOW_PREFIX)
    }
    never = np.zeros(len(steps), dtype=bool)
    shadowed_by_source: dict[str, np.ndarray] = {}
    for name, has in shadowed.items():
        source = _shadow_source(name)
        shadowed_by_source[source] = shadowed_by_source.get(source, never) | has
    observed = {
        name: signal
        for name, signal in signals.items()
        if name not in sent and _is_observation(name, observations) and len(signal)
    }
    runs = []
    for (source, instruction), group in groupby(range(len(steps)), key=lambda i: (sources[i], instructions[i])):
        if source != TELEOPERATOR:
            continue
        run = np.fromiter(group, dtype=int)
        for label, has in sent.items():
            if not has[run].all():
                missing = steps[run][~has[run]][0]
                raise ValueError(
                    f'In the record {uid}, the label {label!r} has no sample at the step {HARNESS_STEP}={missing}'
                )
        every = np.ones(len(run), dtype=bool)
        used = {name: every for name, signal in observed.items() if _steps(signal)[0] <= steps[run[0]]}
        used.update(dict.fromkeys(labels, every))
        used.update({name: has[run] for name, has in shadowed.items() if has[run].any()})
        times = _step_times([(signals[name], uses) for name, uses in used.items()], steps[run])
        data = dict(record.static)
        if instruction is not None:
            data[keys.TASK] = instruction
        data.update({
            name: signals[name].time[[time for time, use in zip(times, uses, strict=True) if use]]
            for name, uses in used.items()
        })
        without = {name: int(np.count_nonzero(~has[run])) for name, has in shadowed_by_source.items()}
        runs.append(TeleoperatorRun(EpisodeContainer(data, record.meta), without))
    return runs


@dataclass
class _Read:
    runs: list[TeleoperatorRun]
    dropped_records: int


class DaggerDataset(Dataset):
    """The training episodes of the teleoperator runs in ``records``, in order, as ``teleoperator_runs`` makes them,
    from each record that ``license_rule`` keeps."""

    def __init__(
        self, records: Dataset, observations: Collection[str], labels: Sequence[str], license_rule: LicenseRule
    ):
        self._records = records
        self._observations = frozenset(observations)
        self._labels = tuple(labels)
        self._license_rule = license_rule

    @cached_property
    def _read(self) -> _Read:
        read = _Read([], 0)
        for record in self._records:
            if refused := self._license_rule.refused(record):
                logging.warning('The license rule drops the record %s: %s', record.meta.get(META_UID), refused)
                read.dropped_records += 1
            else:
                read.runs.extend(teleoperator_runs(record, self._observations, self._labels))
        return read

    @property
    def runs(self) -> list[TeleoperatorRun]:
        return self._read.runs

    @property
    def dropped_records(self) -> int:
        """The number of records that the license rule drops."""
        return self._read.dropped_records

    @property
    def steps_without_shadow(self) -> Counter[str]:
        """For each source with a shadow action, the teleoperator steps of all runs where it has none."""
        total = Counter()
        for run in self.runs:
            # ``update`` keeps a source whose count is 0.
            total.update(run.steps_without_shadow)
        return total

    def __len__(self) -> int:
        return len(self.runs)

    def _get_episode(self, index: int) -> Episode:
        return self.runs[index].episode
