"""What the harness records for each step of a run with a helper: the ``control.*`` signals and the shadow actions.

The takeover gives a ``StepRecord`` for each step. ``RecordSamples`` turns it into the samples that the harness
writes: each ``control.*`` value when it changes, each event, and each shadow action::

    switch = Event(kind=EventKind.SWITCH, from_='policy', to='teleoperator')
    StepRecord('teleoperator', frozenset({'teleoperator'}), events=(switch,), shadow={'policy': commands})
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pimm import Time
from positronic.dataset.serializers import Serializer, expand_suffixed
from positronic.drivers.recording import HARNESS_STEP
from positronic.policy.base import Commands
from positronic.policy.codecs import PolicyCommandError
from positronic.policy.keys import CONTROL_EVENT, CONTROL_SOURCE, CONTROL_USES, SHADOW


class EventKind(StrEnum):
    # The driver changed: a press of the helper.
    SWITCH = 'switch'


class Event(BaseModel):
    """An event of the takeover. ``from`` and ``to`` are source names. ``None`` is a part that the event does not
    have."""

    model_config = ConfigDict(frozen=True, extra='forbid', validate_by_name=True)

    kind: EventKind
    from_: str | None = Field(default=None, validation_alias='from', serialization_alias='from')
    to: str | None = None
    cause: str | None = None

    def encode(self) -> str:
        """The event as the JSON string that ``control.event`` records."""
        return json.dumps(self.model_dump(mode='json', by_alias=True, exclude_none=True))


def encode_uses(uses: frozenset[str]) -> str:
    """The source names as the JSON string that ``control.uses`` records: a sorted list."""
    return json.dumps(sorted(uses))


def decode_uses(value: str) -> frozenset[str]:
    """The source names of a ``control.uses`` value that ``encode_uses`` gives."""
    return frozenset(json.loads(value))


@dataclass(frozen=True)
class StepRecord:
    """The control record of one step.

    ``source`` is the name of the driver. ``uses`` holds the names of the sources that the step used, the driver
    included. ``events`` holds the events of the step, in order. ``shadow`` holds the shadow actions of the step, by
    source name.
    """

    source: str
    uses: frozenset[str]
    events: tuple[Event, ...] = ()
    shadow: Mapping[str, Commands] = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in self.uses:
            raise ValueError(f'The sources that a step uses include its driver {self.source!r}: {sorted(self.uses)}')
        if self.source in self.shadow:
            raise ValueError(f'The driver {self.source!r} has no shadow actions')


class RecordSamples:
    """The samples that the record of one episode gets from the ``StepRecord`` of each step, as
    ``(signal, value, time)``.

    A ``control.*`` value goes in when it changes. A shadow action goes through the serializer of its channel in
    ``serializers``. The times of each signal rise: a sample that would not come after the last sample of its signal
    moves to 1 ns after it on each clock where it is not later, and keeps its step number.
    """

    def __init__(self, serializers: Mapping[str, Serializer | None]):
        self._serializers = serializers
        self._values: dict[str, Any] = {}
        self._times: dict[str, Time] = {}

    def of(self, record: StepRecord, time: Time) -> list[tuple[str, Any, Time]]:
        channels = {channel for commands in record.shadow.values() for channel in commands}
        if unknown := sorted(channels - self._serializers.keys()):
            raise PolicyCommandError(
                f'The shadow actions name {unknown}, and the embodiment takes only {sorted(self._serializers)}'
            )
        values = {CONTROL_SOURCE: record.source, CONTROL_USES: encode_uses(record.uses)}
        changed = {name: value for name, value in values.items() if self._values.get(name) != value}
        self._values.update(changed)
        samples = [*changed.items(), *((CONTROL_EVENT, event.encode()) for event in record.events)]
        for name, commands in record.shadow.items():
            for channel, command in commands.items():
                serializer = self._serializers[channel]
                value = serializer(command) if serializer is not None else command
                signal = f'{SHADOW}.{name}.{channel}'
                samples.extend((full, entry) for full, entry in expand_suffixed(signal, value) if entry is not None)
        return [(signal, value, self._rising(signal, time)) for signal, value in samples]

    def _rising(self, signal: str, time: Time) -> Time:
        last = self._times.get(signal)
        if last is not None and not last < time:
            time = Time(**{
                timeline: ns if timeline == HARNESS_STEP else max(ns, last[timeline] + 1)
                for timeline, ns in time.items()
            })
        self._times[signal] = time
        return time
