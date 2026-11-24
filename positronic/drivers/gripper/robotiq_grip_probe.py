"""Record how a Robotiq 2F gripper reports a close on an object that an operator holds between the fingers.

The probe moves only the gripper. It sends no command to an arm.

Usage
  uv run --locked --extra hardware python -m positronic.drivers.gripper.robotiq_grip_probe \\
      --port /dev/ttyUSB0 --label pen-middle --repeats 5 --out <dir>
  ... --speed 255 --force 128   # the speed and the force of the fingers, 0..255
  ... --label nothing           # a close with no object between the fingers

Stop each program that holds the port before you start the probe, for example a station or a console. At the start,
the probe activates the gripper: the fingers open and close fully. Hold nothing between them then. Before each close,
the probe asks for Enter. Hold the object between the fingers, push Enter, and keep the object still. The probe
records each read for ``--record-s`` seconds, then opens the fingers.

The probe writes ``<out>/<label>.json`` (the outcome and the reads of each close), ``<out>/<label>.csv`` (one row
for each read) and ``<out>/<label>.txt`` (the table that it prints). It writes them again after each close. In these
files, ``fingers`` is ``gOBJ`` (0..3, as in the manual), ``requested`` is ``gPR``, ``position`` is ``gPO`` and
``current`` is ``gCU``.
"""

import argparse
import csv
import io
import logging
import math
import os
import re
import select
import sys
from collections.abc import Generator, Iterator
from pathlib import Path

from pydantic import BaseModel

import pimm
from positronic.drivers.gripper.robotiq import _BUS_FAULTS, TUNING_2F85, Fingers, Reading, _Gripper, _serial_client
from positronic.drivers.utils import RecoveryOutcome

logger = logging.getLogger(__name__)

CLOSED = 1.0
OPEN = 0.0


class Sample(BaseModel, frozen=True):
    """One read of the gripper after a command."""

    t_s: float  # from the command
    fingers: Fingers
    requested: int
    position: int
    current: int

    @classmethod
    def of(cls, t_s: float, reading: Reading) -> 'Sample':
        return cls(
            t_s=t_s,
            fingers=reading.status.fingers,
            requested=reading.requested,
            position=reading.position,
            current=reading.current,
        )


class Outcome(BaseModel, frozen=True):
    """How one close ended."""

    stopped_s: float | None  # the first read with the close request taken and the fingers not MOVING
    still_s: float  # the first read from which the position does not change
    fingers: Fingers
    position: int
    current: int
    fingers_seen: list[Fingers]  # each change of gOBJ, in order

    @property
    def detected(self) -> bool:
        """The gripper reports an object at the end of the close."""
        return self.fingers is Fingers.CONTACT_CLOSING

    @classmethod
    def of(cls, samples: list[Sample]) -> 'Outcome':
        request = int(CLOSED * 255)
        stopped = next((s for s in samples if s.requested == request and s.fingers is not Fingers.MOVING), None)
        final = samples[-1]
        still = samples[-1]
        for s in reversed(samples):
            if s.position != final.position:
                break
            still = s
        seen = [s.fingers for i, s in enumerate(samples) if i == 0 or s.fingers is not samples[i - 1].fingers]
        return cls(
            stopped_s=None if stopped is None else stopped.t_s,
            still_s=still.t_s,
            fingers=final.fingers,
            position=final.position,
            current=final.current,
            fingers_seen=seen,
        )


class Close(BaseModel, frozen=True):
    """One close: how it ended, and each read of it."""

    repeat: int  # from 1
    outcome: Outcome
    samples: list[Sample]


class GripRecord(BaseModel):
    """What the probe recorded for one label."""

    label: str
    speed: int
    force: int
    record_s: float
    closes: list[Close] = []


HEADER = f'{"label":<20} {"repeat":>6} {"stopped_s":>9} {"still_s":>7} {"gOBJ":<15} {"gPO":>3} {"gCU":>3}  gOBJ seen'


def row(label: str, close: Close) -> str:
    """The line of ``close`` in the table."""
    o = close.outcome
    stopped = '-' if o.stopped_s is None else f'{o.stopped_s:.3f}'
    seen = '>'.join(f.name for f in o.fingers_seen)
    return (
        f'{label:<20} {close.repeat:>6} {stopped:>9} {o.still_s:>7.3f} {o.fingers.name:<15} '
        f'{o.position:>3} {o.current:>3}  {seen}'
    )


def total(record: GripRecord) -> str:
    """The line that counts the closes with an object detected."""
    detected = sum(close.outcome.detected for close in record.closes)
    return f'{record.label}: gOBJ is CONTACT_CLOSING at the end of {detected} of {len(record.closes)} closes'


def table(record: GripRecord) -> str:
    """A line for each close, and the total line."""
    return '\n'.join([HEADER, *(row(record.label, close) for close in record.closes), total(record)]) + '\n'


def _replace(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file in the same directory. A failed write keeps the old file."""
    temp = path.with_name(f'.{path.name}.tmp')
    temp.write_text(text)
    os.replace(temp, path)


def outputs(out: Path, label: str) -> tuple[Path, Path, Path]:
    """The JSON, the CSV and the table file that the probe writes in ``out`` for ``label``."""
    return out / f'{label}.json', out / f'{label}.csv', out / f'{label}.txt'


def write(record: GripRecord, out: Path) -> None:
    """Write the JSON, the CSV and the table of ``record`` to ``out``."""
    json_path, csv_path, table_path = outputs(out, record.label)
    _replace(json_path, record.model_dump_json(indent=1))
    rows = io.StringIO()
    writer = csv.writer(rows)
    writer.writerow(['repeat', *Sample.model_fields])
    for close in record.closes:
        for sample in close.samples:
            writer.writerow([close.repeat, *sample.model_dump(mode='json').values()])
    _replace(csv_path, rows.getvalue())
    _replace(table_path, table(record))


class RobotiqGripProbe(pimm.ControlSystem):
    """Closes the gripper ``repeats`` times on what the operator holds, and records each close to ``out``.

    The operator pushes Enter on the file descriptor ``operator`` before each close.
    """

    def __init__(self, port: str, record: GripRecord, repeats: int, out: Path, operator: int):
        self._port = port
        self.record = record
        self._repeats = repeats
        self._out = out
        self._operator = operator
        # _Gripper serves the moves of a driver. The probe sends it no move.
        self._moves = pimm.calls.ControlSystemHandler[float, None](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        client = _serial_client(self._port)
        if not client.connect():
            raise SystemExit(f'Cannot open {self._port}. A station or a console can hold the port: stop it first.')
        gripper = _Gripper(
            client, self._port, self._moves, pimm.NoOpReceiver(), pimm.NoOpEmitter(), TUNING_2F85, should_stop, clock
        )
        try:
            yield from self._probe(gripper, clock)
        finally:
            client.close()

    def _probe(self, gripper: _Gripper, clock: pimm.Clock) -> Generator[pimm.Command, None, None]:
        """Activate the gripper, then prompt, close, record and open once for each repeat, until the world stops."""
        outcome = yield from gripper.activate()
        if gripper.should_stop.value:
            return
        if outcome is not RecoveryOutcome.CLEARED:
            raise SystemExit(f'The gripper on {self._port} did not activate: {gripper.observe()}')
        # A move command sets rACT, so a move before the activation completes starts an activation stroke.
        try:
            yield from self._repeat(gripper, clock)
        finally:
            self._release(gripper)

    def _repeat(self, gripper: _Gripper, clock: pimm.Clock) -> Generator[pimm.Command, None, None]:
        """Prompt, close, record and open once for each repeat, until the world stops."""
        yield from self._move(gripper, OPEN, clock)
        print(HEADER)
        for repeat in range(1, self._repeats + 1):
            print(f'[{self.record.label} {repeat}/{self._repeats}] Hold the object, then push Enter to close: ', end='')
            if not (yield from self._enter_pushed(gripper.should_stop)):
                return
            samples = yield from self._move(gripper, CLOSED, clock)
            if gripper.should_stop.value:
                return
            close = Close(repeat=repeat, outcome=Outcome.of(samples), samples=samples)
            self.record.closes.append(close)
            yield from self._move(gripper, OPEN, clock)
            write(self.record, self._out)
            print(row(self.record.label, close))

    def _enter_pushed(self, should_stop: pimm.SignalReceiver) -> Generator[pimm.Command, None, bool]:
        """Yield until the operator pushes Enter. False when the world stops or the input ends first."""
        sys.stdout.flush()
        while not should_stop.value:
            if not select.select([self._operator], [], [], 0)[0]:
                yield pimm.Sleep(0.05)
                continue
            char = os.read(self._operator, 1)
            if char in (b'\n', b''):
                return char == b'\n'
        return False

    def _release(self, gripper: _Gripper) -> None:
        """Ask the fingers open, so that they do not keep their grip on the object after the probe ends."""
        try:
            gripper.go_to(OPEN, self.record.speed, self.record.force)
        # rules-allow: swallowed-error — a cleanup: the error that ended the probe is the one to propagate
        except _BUS_FAULTS as exc:
            logger.error('Could not ask the fingers of the gripper on %s to open: %s', self._port, exc)

    def _move(self, gripper: _Gripper, grip: float, clock: pimm.Clock) -> Generator[pimm.Command, None, list[Sample]]:
        """Put the fingers at ``grip`` and read the gripper for ``record_s`` seconds."""
        r = self.record
        gripper.go_to(grip, r.speed, r.force)
        start = clock.now()
        samples: list[Sample] = []
        while clock.now() - start < r.record_s and not gripper.should_stop.value:
            yield gripper.limiter.wait()
            samples.append(Sample.of(clock.now() - start, gripper.observe()))
        return samples


def _label(text: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', text):
        raise argparse.ArgumentTypeError(f'{text!r} is not a file name of letters, digits, "_", "." and "-"')
    return text


def _count(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'{value} is not a positive count')
    return value


def _seconds(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f'{value} is not a positive number of seconds')
    return value


def _byte(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 255:
        raise argparse.ArgumentTypeError(f'{value} is not in 0..255')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', required=True, help='the serial port of the gripper, for example /dev/ttyUSB0')
    parser.add_argument(
        '--label', type=_label, required=True, help='the object and the grip, for example pen-fingertip'
    )
    parser.add_argument('--repeats', type=_count, default=5)
    parser.add_argument('--speed', type=_byte, default=255)
    parser.add_argument('--force', type=_byte, default=255)
    parser.add_argument('--record-s', type=_seconds, default=3.0, help='how long to read the gripper after a command')
    parser.add_argument('--out', type=Path, required=True, help='the directory for the files')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if existing := [path for path in outputs(args.out, args.label) if path.exists()]:
        raise SystemExit(f'{", ".join(map(str, existing))} exists. Give a different --label or --out.')
    record = GripRecord(label=args.label, speed=args.speed, force=args.force, record_s=args.record_s)
    with pimm.World() as world:
        world.run(RobotiqGripProbe(args.port, record, args.repeats, args.out, sys.stdin.fileno()))
    print(total(record))


if __name__ == '__main__':
    main()
