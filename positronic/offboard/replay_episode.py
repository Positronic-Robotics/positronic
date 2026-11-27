"""Play the observations of a recorded episode to a policy server, and print the commands that come back.

The client stack that the server declares runs as on a rig, on the episode's clock, which stops while a call is open.
Each command prints beside the command that the episode recorded at the same time.

Usage
    uv run python -m positronic.offboard.replay_episode --dataset.path=<dataset> \
        --policy.address.host=<host> --policy.address.port=8000
    ... --start_sec=10 --calls=1            # one call on the observation 10 s into the episode
    ... --end_sec=20                        # stop 20 s into the episode
    ... --episode=3                         # the fourth episode of the dataset
    ... --policy=@positronic.cfg.policy.authed_remote --policy.wire=websocket_tls --policy.address.port=443
                                            # sends the token in AUTH_TOKEN as a bearer token
"""

import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Self

import configuronic as cfn
import numpy as np
import pos3

import positronic.cfg.ds
import positronic.cfg.policy
from pimm.logging import init_logging
from pimm.time import RECEIVED_WORLD
from positronic import keys
from positronic.dataset import Dataset, Episode
from positronic.dataset.time import Time
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD
from positronic.drivers.roboarm.ik import ee_frames
from positronic.drivers.roboarm.models import EEFrame
from positronic.eval import keys as eval_keys
from positronic.policy import Policy, PolicyCommandError
from positronic.policy.base import PolicyRun, Step
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.harness import policy_frames, refuse_unstated_pose_commands, stated_pose_signals

TIMELINES = (HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME)
"""The timelines to play an episode on, in order of preference. An older recording carries only the later ones."""
WAIT_SEC = 0.1
MIN_STEP_SEC = 0.005


@dataclass
class _Rig:
    """What the rig of an episode takes from a policy and sends it, as the harness decides it for that policy."""

    sent: list[str]
    """The observations sent: a pose only where the policy states its frame at the site the rig reports it at."""
    recorded: list[str]
    """The recorded commands to print beside the policy's."""
    channels: frozenset[str]
    """The command channels the episode records, which a step may name."""
    stated_poses: frozenset[str]
    """The pose signals whose frame the policy states. A pose command names only their channels."""

    @classmethod
    def of(cls, ep: Episode, timeline: str, observed: list[str], declared: dict[str, EEFrame]) -> Self:
        rig_frames = ee_frames(ep.static)
        stated = stated_pose_signals(declared, rig_frames)
        # An older recording states no frames, and its poses carry only the suffix.
        poses = rig_frames.keys() | {name for name in observed if name.endswith(keys.EE_POSE_SUFFIX)}
        return cls(
            sent=sorted(name for name in observed if name not in poses - stated),
            recorded=sorted(
                name
                for name, signal in ep.signals.items()
                if keys.command_channel(name) is not None
                and not keys.is_control_mode(name)
                and timeline in signal.timelines
            ),
            channels=frozenset(channel for name in ep.signals if (channel := keys.command_channel(name)) is not None),
            stated_poses=stated,
        )


@dataclass
class _Recording:
    """An episode read on one timeline, and the sink that each line of the play goes to."""

    ep: Episode
    timeline: str
    first_ns: int
    out: Callable[[str], None]

    def at(self, names: list[str], time_ns: int) -> dict[str, Any]:
        """The value of each signal in ``names`` at ``time_ns``, for each signal that has a sample by then."""
        at = Time(**{self.timeline: time_ns})
        values = {}
        for name in names:
            try:
                values[name] = self.ep.signals[name].time[at][0]
            except KeyError:
                continue
        return values

    def instruction(self, time_ns: int) -> str:
        """The instruction the policy got at ``time_ns``, or the episode's task where the episode records none."""
        signal = self.ep.signals.get(eval_keys.INSTRUCTION)
        names = [eval_keys.INSTRUCTION] if signal is not None and self.timeline in signal.timelines else []
        return str(self.at(names, time_ns).get(eval_keys.INSTRUCTION, self.ep.static[keys.TASK]))

    def say(self, time_ns: int, line: str) -> None:
        self.out(f'{(time_ns - self.first_ns) / 1e9:8.3f} s  {line}')

    def step(self, run: PolicyRun, rig: _Rig, time_ns: int) -> Step | None:
        """Send the observation at ``time_ns`` to ``run``, and write each command it returns beside the recorded ones.

        Return ``None`` when the policy ends the run. Raise ``PolicyCommandError`` for a step that the harness refuses.
        """
        obs = {
            **self.at(rig.sent, time_ns),
            keys.TASK: self.instruction(time_ns),
            keys.DESCRIPTOR: self.ep.static.get(eval_keys.EMBODIMENT, ''),
        }
        try:
            step = run.send(obs)
        except StopIteration as end:
            self.say(time_ns, f'the policy ended the run: {end.value}')
            return None
        assert step is not None, 'a policy must yield a Step for each observation'
        if unknown := sorted(step.commands.keys() - rig.channels):
            raise PolicyCommandError(
                f'The step names {unknown}, and the episode records commands only on {sorted(rig.channels)}'
            )
        refuse_unstated_pose_commands(step.commands, rig.stated_poses)
        if step.commands:
            for name, value in step.commands.items():
                self.say(time_ns, f'{name} = {value}')
            for name, value in self.at(rig.recorded, time_ns).items():
                self.say(time_ns, f'recorded {name} = {value}')
        return step


def _wait_for_answers(runtime: Executor) -> int:
    """Wait until the clock may advance, and return how many calls answered."""
    while (result := runtime.wait(WAIT_SEC)).status is WaitStatus.TIMED_OUT:
        pass
    return len(result.completed)


def play(
    ep: Episode,
    policy: Policy,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    calls: int | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """Play ``ep`` from ``start_sec`` to ``end_sec`` to ``policy``, write each line to ``out``, and return the calls.

    ``calls`` stops the play when the client asks for one more call than it. Raise as the harness does: ``ValueError``
    for a policy that states a pose frame the rig does not report, and ``PolicyCommandError`` for a step it refuses.
    """
    observed = [name for name in ep.signals if keys.is_observation(name)]
    timeline = next((t for t in TIMELINES if all(t in ep.signals[name].timelines for name in observed)), None)
    if not observed or timeline is None:
        raise ValueError(f'The episode has no observation on one of the timelines {TIMELINES}')
    first_ns = max(ep.signals[name].bounds(timeline).start for name in observed)
    last_ns = min(ep.signals[name].bounds(timeline).finish for name in observed)
    stop_ns = last_ns if end_sec is None else min(last_ns, first_ns + round(end_sec * 1e9))
    now = [first_ns + round(start_sec * 1e9)]
    recording = _Recording(ep, timeline, first_ns, out)
    out(f'Episode of {(last_ns - first_ns) / 1e9:.1f} s on {timeline}: {recording.instruction(now[0])!r}')

    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    run = runtime.start(policy)
    answered = 0
    # The runtime drains its calls before the run closes, as on a rig.
    with closing(run), closing(runtime):
        rig = _Rig.of(ep, timeline, observed, policy_frames(policy, runtime))
        out(f'Observations sent: {rig.sent}')
        while now[0] <= stop_ns:
            runtime.start_tick(now[0])
            called_ns = time.monotonic_ns()
            step = recording.step(run, rig, now[0])
            if step is None or (calls is not None and answered >= calls and runtime.has_pending):
                break
            if completed := _wait_for_answers(runtime):
                answered += completed
                recording.say(now[0], f'call {answered} answered in {(time.monotonic_ns() - called_ns) / 1e6:.0f} ms')
                continue
            now[0] = max(step.resume_at_ns, now[0] + round(MIN_STEP_SEC * 1e9))
    out(f'{answered} calls answered')
    return answered


@cfn.config(
    dataset=positronic.cfg.ds.local_all,
    policy=positronic.cfg.policy.remote,
    episode=0,
    start_sec=0.0,
    end_sec=None,
    calls=None,
)
def replay(
    dataset: Dataset, policy: Policy, episode: int, start_sec: float, end_sec: float | None, calls: int | None
) -> None:
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    play(dataset[episode], policy, start_sec, end_sec, calls)


if __name__ == '__main__':
    with pos3.mirror():
        init_logging()
        cfn.cli(replay)
