"""Play the observations of a recorded episode to a policy server, and print the commands that come back.

The client stack that the server declares runs as on a rig, on the episode's clock. The clock stops while a call
is open, so the latency of the server does not change which observations it gets. Each command prints beside the
command that the episode recorded at the same time.

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
from typing import Any

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
from positronic.eval import keys as eval_keys
from positronic.policy import Policy, PolicyCommandError
from positronic.policy.base import PolicyRun, Step
from positronic.policy.executor import Executor, WaitStatus

TIMELINES = (HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME)
"""The timelines to play an episode on, in order of preference. An older recording carries only the later ones."""
WAIT_SEC = 0.1
MIN_STEP_SEC = 0.005


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

    def step(self, run: PolicyRun, sent: list[str], recorded: list[str], time_ns: int) -> Step | None:
        """Send the observation at ``time_ns`` to ``run``, and write each command it returns beside the recorded ones.

        Return ``None`` when the policy ends the run. Raise ``PolicyCommandError`` for a step that names a channel a
        rig does not take.
        """
        obs = {
            **self.at(sent, time_ns),
            keys.TASK: self.instruction(time_ns),
            keys.DESCRIPTOR: self.ep.static.get(eval_keys.EMBODIMENT, ''),
        }
        try:
            step = run.send(obs)
        except StopIteration as end:
            self.say(time_ns, f'the policy ended the run: {end.value}')
            return None
        assert step is not None, 'a policy must yield a Step for each observation'
        if unknown := sorted(name for name in step.commands if not keys.is_command_channel(name)):
            raise PolicyCommandError(f'The step names {unknown}, and a rig takes only command and grip channels')
        if step.commands:
            for name, value in step.commands.items():
                self.say(time_ns, f'{name} = {value}')
            for name, value in self.at(recorded, time_ns).items():
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

    ``calls`` stops the play when the client asks for one more call than it. Raise ``PolicyCommandError`` for a step
    that names a channel a rig does not take.
    """
    observed = [name for name in ep.signals if keys.is_observation(name)]
    timeline = next((t for t in TIMELINES if all(t in ep.signals[name].timelines for name in observed)), None)
    if not observed or timeline is None:
        raise ValueError(f'The episode has no observation on one of the timelines {TIMELINES}')
    recorded = [
        name
        for name, signal in ep.signals.items()
        if keys.is_command_channel(name) and not keys.is_control_mode(name) and timeline in signal.timelines
    ]
    first_ns = max(ep.signals[name].bounds(timeline).start for name in observed)
    last_ns = min(ep.signals[name].bounds(timeline).finish for name in observed)
    stop_ns = last_ns if end_sec is None else min(last_ns, first_ns + round(end_sec * 1e9))
    now = [first_ns + round(start_sec * 1e9)]
    recording = _Recording(ep, timeline, first_ns, out)
    out(f'Episode of {(last_ns - first_ns) / 1e9:.1f} s on {timeline}: {recording.instruction(now[0])!r}')

    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    run = runtime.start(policy)
    frames = ee_frames(policy.meta()) | ee_frames(runtime.metadata_snapshot())
    # The harness sends a pose only to a policy that states the frame of that pose.
    sent = sorted(name for name in observed if not name.endswith(keys.EE_POSE_SUFFIX) or name in frames)
    out(f'Observations sent: {sent}')
    answered = 0
    with closing(runtime), closing(run):
        while now[0] <= stop_ns:
            runtime.start_tick(now[0])
            called_ns = time.monotonic_ns()
            step = recording.step(run, sent, recorded, now[0])
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
