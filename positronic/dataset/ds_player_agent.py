from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass, field

import pimm
from positronic.dataset import Episode, Time
from positronic.dataset.episode import select_timeline


@dataclass
class DsPlayerStartCommand:
    episode: Episode
    start_ts: int | None = None
    end_ts: int | None = None
    timeline: str | None = None


@dataclass
class DsPlayerAbortCommand:
    pass


DsPlayerCommand = DsPlayerStartCommand | DsPlayerAbortCommand


class DsPlayerAgent(pimm.ControlSystem):
    def __init__(self, poll_hz: float = 100.0):
        self.command = pimm.ControlSystemReceiver[DsPlayerCommand](self)
        self.outputs = pimm.EmitterDict(self)
        self.finished = pimm.ControlSystemEmitter[DsPlayerStartCommand](self)
        self._poll_hz = float(poll_hz)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        playback: _Playback | None = None
        limiter = pimm.utils.RateLimiter(clock, hz=self._poll_hz)

        while not should_stop.value:
            if (cmd := pimm.value_updated(self.command)) is not None:
                playback = self._apply_command(cmd, clock)

            if playback is None:
                yield limiter.wait()
                continue

            if playback.empty:
                self.finished.emit(playback.command)
                playback = None
                yield pimm.Yield()
                continue

            wait_ns = playback.wait_ns(clock.now_ns())
            if wait_ns > 0:
                yield pimm.Sleep(wait_ns / 1e9)
                continue

            scheduled_ts, name, value = playback.pop()
            self.outputs[name].emit(value, time=Time(**{'playback.scheduled': scheduled_ts}))

            yield pimm.Yield()

    def _apply_command(self, cmd: DsPlayerCommand, clock: pimm.Clock) -> _Playback | None:
        match cmd:
            case DsPlayerStartCommand() as start_cmd:
                new_playback = _Playback.start(start_cmd, list(self.outputs.keys()), clock.now_ns())
                if new_playback is None:
                    self.finished.emit(start_cmd)
                return new_playback
            case DsPlayerAbortCommand():
                return None
        raise ValueError(f'Unknown command: {cmd}')


@dataclass
class _Playback:
    command: DsPlayerStartCommand
    start_clock_ns: int
    timeline: str
    heap: list[tuple[int, int, str, object]] = field(default_factory=list)
    streams: dict[str, Iterator[tuple[object, Time]]] = field(default_factory=dict)
    start_ts: int | None = None
    counter: int = 0

    @property
    def empty(self) -> bool:
        return not self.heap

    def wait_ns(self, now_ns: int) -> int:
        if self.start_ts is None:
            return 0

        next_ts = self.heap[0][0] - self.start_ts + self.start_clock_ns
        return max(0, next_ts - now_ns)

    def pop(self) -> tuple[int, str, object]:
        scheduled_ts, _, name, value = heapq.heappop(self.heap)
        if self.start_ts is None:
            self.start_ts = scheduled_ts
        self.schedule_next(name)
        return scheduled_ts - self.start_ts + self.start_clock_ns, name, value

    def schedule_next(self, name: str):
        stream = self.streams[name]
        try:
            value, ts = next(stream)
        except StopIteration:
            self.streams.pop(name)
            return
        self._push(name, ts[self.timeline], value)

    def _push(self, name: str, ts: int, value: object) -> None:
        heapq.heappush(self.heap, (ts, self.counter, name, value))
        self.counter += 1

    @classmethod
    def start(cls, command: DsPlayerStartCommand, output_names: list[str], start_clock_ns: int) -> _Playback | None:
        assert output_names, 'No output names provided'

        episode = command.episode
        selected = []
        for name in output_names:
            if name in episode.static:
                raise ValueError(f'Requested output {name!r} is static and cannot be emitted')
            selected.append(episode.signals[name])
        timeline = select_timeline(
            set.intersection(*(set(signal.timelines) for signal in selected)), timeline=command.timeline
        )
        playback = cls(command, start_clock_ns, timeline)
        start = Time(**{timeline: command.start_ts}) if command.start_ts is not None else None
        end = Time(**{timeline: command.end_ts}) if command.end_ts is not None else None

        for name, signal in zip(output_names, selected, strict=True):
            playback.streams[name] = iter(signal if start is None and end is None else signal.time[start:end])
            playback.schedule_next(name)

        return playback if playback.heap else None
