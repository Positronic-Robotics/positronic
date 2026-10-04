"""Replay a journal of ``record.py`` offline: verify it, then branch it at the first inference.

    uv run --locked positronic/cli/examples/policy_journal/replay.py JOURNAL BRANCHES

The script does not start a world or a motor. It:

1. Verifies JOURNAL with an inference that raises if anything calls it.
2. Refuses a rerun that does not allow execution, before it writes anything.
3. Branches with a given result in place of the first inference.
4. Branches with a second inference that runs on the input the journal retained.
5. Stops a branch whose inference version the journal does not record.

Each branch is a new journal under BRANCHES. It publishes its result at the turn and time at which the
source published the first inference. The source journal does not change.
"""

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from positronic.cli.examples.policy_journal.move import MOTOR, Move
from positronic.policy.base import Obs
from positronic.policy.journal import Event, Journal, StepReturned, TurnStarted
from positronic.policy.replay import (
    Branch,
    ExecutionRefused,
    MissingResult,
    ReplaceResult,
    RerunActivity,
    branch,
    verify,
)


def unreachable_inference(obs: Obs) -> list[dict[str, int]]:
    raise AssertionError('A replay publishes recorded results and never runs the inference')


def infer_v2(obs: Obs) -> list[dict[str, int]]:
    return [{MOTOR: -1}, {MOTOR: -2}]


def fingerprint(journal: Journal) -> str:
    files = sorted(path for path in journal.path.rglob('*') if path.is_file())
    return hashlib.sha256(b''.join(path.read_bytes() for path in files)).hexdigest()


def commands(journal: Journal, events: Sequence[Event]) -> list[Any]:
    recording = journal.read()
    return [journal.commands.decode(recording.payload(e.commands)) for e in events if isinstance(e, StepReturned)]


def report(name: str, source: Journal, result: Branch) -> None:
    print(f'{name}: {result.journal.path}')
    for difference in result.differences:
        turn = next(e for e in difference.branch if isinstance(e, TurnStarted))
        before, after = commands(source, difference.source), commands(result.journal, difference.branch)
        print(f'  turn {difference.invocation} at {turn.time_ns / 1e6:g} ms: commands {before} -> {after}')


def main(source_path: Path, branches: Path) -> None:
    source = Journal(source_path)
    before = fingerprint(source)

    verified = verify(Move(unreachable_inference), source)
    print(f'verified {verified.turns} turns without running the inference, complete: {verified.complete}')

    submissions = source.read().submissions()
    for s in submissions:
        print(f'submission {s.submission}: {s.operation} v{s.version} at turn {s.invocation}, input {s.input[:12]}')
    chosen = submissions[0].submission

    try:
        branch(Move(unreachable_inference), source, branches / 'refused', [RerunActivity(chosen, infer_v2, 2)])
    except ExecutionRefused as exc:
        print(f'refused: {exc}')

    replaced = ReplaceResult(chosen, [{MOTOR: 0}, {MOTOR: 0}])
    report('replaced result', source, branch(Move(unreachable_inference), source, branches / 'replaced', [replaced]))

    rerun = RerunActivity(chosen, infer_v2, version=2, allow_execution=True)
    report('rerun inference', source, branch(Move(unreachable_inference), source, branches / 'rerun', [rerun]))

    try:
        branch(Move(unreachable_inference, version=2), source, branches / 'changed', [])
    except MissingResult:
        print('changed request: stopped, because the source records no step_plan v2')

    print(f'source unchanged: {fingerprint(source) == before}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('journal', type=Path, help='The journal that record.py wrote')
    parser.add_argument('branches', type=Path, help='A directory for the branch journals')
    args = parser.parse_args()
    main(args.journal, args.branches)
