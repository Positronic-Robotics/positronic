"""The report of a held-out evaluation, and the check that two evaluations give the same result.

A report is the JSON of `dataclasses.asdict`: the FLUX 3 Action image has no pydantic.

Usage
  python -m droid_finetune.report compare <report.json> <report.json> [--rtol=1e-4]
"""

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScoredWindow:
    episode_index: int
    start: int
    action_mse: float


@dataclass(frozen=True)
class Report:
    """`action_mse` is the mean over windows of each window's mean squared action error: radians² for the seven
    joints, closed fraction² for the gripper."""

    framework: str
    checkpoint: str
    heldout: str
    seed: int
    windows_per_episode: int
    episodes: int
    action_mse: float
    action_mse_median: float
    action_mse_per_channel: list[float]
    seconds: float
    windows: list[ScoredWindow]

    def summary(self) -> dict:
        return {k: v for k, v in dataclasses.asdict(self).items() if k != 'windows'}


@dataclass(frozen=True)
class Comparison:
    same: bool
    windows: int
    max_relative_difference: float


def build_report(
    windows: Sequence[ScoredWindow],
    per_channel: Sequence[float],
    *,
    framework: str,
    checkpoint: str,
    heldout: str,
    seed: int,
    windows_per_episode: int,
    episodes: int,
    seconds: float,
) -> Report:
    errors = np.array([w.action_mse for w in windows])
    return Report(
        framework=framework,
        checkpoint=checkpoint,
        heldout=heldout,
        seed=seed,
        windows_per_episode=windows_per_episode,
        episodes=episodes,
        action_mse=float(errors.mean()),
        action_mse_median=float(np.median(errors)),
        action_mse_per_channel=[float(x) for x in per_channel],
        seconds=seconds,
        windows=list(windows),
    )


def write_report(report: Report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(report), indent=1) + '\n')
    print(json.dumps(report.summary()))


def read_report(path: Path) -> Report:
    data = json.loads(path.read_text())
    return Report(**{**data, 'windows': [ScoredWindow(**w) for w in data['windows']]})


def compare(a: Report, b: Report, rtol: float) -> Comparison:
    """Whether each window's error in `a` equals the same window's error in `b` within `rtol`."""
    errors_a = {(w.episode_index, w.start): w.action_mse for w in a.windows}
    errors_b = {(w.episode_index, w.start): w.action_mse for w in b.windows}
    if errors_a.keys() != errors_b.keys():
        raise ValueError('the two reports score different windows')
    difference = max(abs(errors_a[k] - errors_b[k]) / max(abs(errors_b[k]), 1e-12) for k in errors_a)
    return Comparison(same=difference <= rtol, windows=len(errors_a), max_relative_difference=difference)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('compare', help='exit 1 unless two reports give the same error on every window')
    check.add_argument('first', type=Path)
    check.add_argument('second', type=Path)
    check.add_argument('--rtol', type=float, default=1e-4)
    args = parser.parse_args()
    result = compare(read_report(args.first), read_report(args.second), args.rtol)
    print(json.dumps(dataclasses.asdict(result)))
    return 0 if result.same else 1


if __name__ == '__main__':
    sys.exit(main())
