"""A policy that plays a recorded episode's commands back, in place of a model."""

import logging
from typing import Any, cast

import numpy as np
import pos3

from positronic import geom, keys
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.signal import Signal
from positronic.drivers.roboarm import command as roboarm_command

from .base import Policy, Session
from .wrappers import ChunkedSchedule

logger = logging.getLogger(__name__)

# The serializer suffix each replayable arm command is written under, most faithful first; pose targets
# go back through the rig's IK. The delta forms are absent: a delta means something only against the
# state it was issued from.
# Taken off the single-arm keys rather than spelled again — a multi-arm channel carries the same suffix.
_ARM_SUFFIXES = (
    (keys.TARGET_JOINTS.removeprefix(keys.ROBOT_COMMAND), roboarm_command.JointPosition),
    (keys.TARGET_EE_POSE.removeprefix(keys.ROBOT_COMMAND), roboarm_command.CartesianPosition),
    (keys.TARGET_RESET.removeprefix(keys.ROBOT_COMMAND), roboarm_command.Reset),
)


def _rebuild(command_type: Any, value: np.ndarray) -> Any:
    if command_type is roboarm_command.Reset:
        return roboarm_command.Reset()  # recorded as a sentinel; the command carries nothing to restore
    if command_type is roboarm_command.CartesianPosition:
        # The quaternion representation ``Serializers.transform_3d`` wrote it in.
        pose = geom.Transform3D.from_vector(np.asarray(value), geom.Rotation.Representation.QUAT)
        return roboarm_command.CartesianPosition(pose)
    return roboarm_command.JointPosition(np.asarray(value))


def _arm_commands(episode: Episode) -> dict[str, dict[int, Any]]:
    """Every recorded arm command as ``channel -> {instant: command}``.

    An embodiment names its own command channels, and a multi-arm rig has several
    (``robot_command.left``, ``robot_command.right``), so the set is read off the recording rather
    than assumed. ``Serializers.robot_command`` writes one signal per command type, so a recording
    whose action space changed mid-episode carries more than one per channel.
    """
    commands: dict[str, dict[int, Any]] = {}
    for suffix, command_type in _ARM_SUFFIXES:  # most faithful first
        for name in sorted(episode.signals):
            if not (name.startswith(keys.ROBOT_COMMAND) and name.endswith(suffix)):
                continue
            signal = episode.signals[name]
            if len(signal) == 0:
                continue
            per_channel = commands.setdefault(name[: -len(suffix)], {})
            for value, ts in signal:
                per_channel.setdefault(ts, _rebuild(command_type, value))  # an instant keeps its first
    return commands


def _grip_signals(episode: Episode) -> dict[str, Any]:
    """Every recorded grip channel, by the name the embodiment commands it under."""
    return {
        name: episode.signals[name]
        for name in sorted(episode.signals)
        if (name == keys.TARGET_GRIP or name.startswith(f'{keys.TARGET_GRIP}.')) and len(episode.signals[name]) > 0
    }


def _unreplayable_arm_signals(episode: Episode) -> list[str]:
    """Arm-command signals the recording carries that this cannot reissue — the delta forms."""
    replayable = tuple(suffix for suffix, _ in _ARM_SUFFIXES)
    return sorted(n for n in episode.signals if n.startswith(f'{keys.ROBOT_COMMAND}.') and not n.endswith(replayable))


def load_actions(episode: Episode) -> list[dict[str, Any]]:
    """The episode's commands as an action list: one entry per instant any channel was commanded at.

    - Every command channel the recording carries replays, whatever the embodiment named them.
    - Each channel keeps its own recorded timing, so a grip command between two arm waypoints falls
      due between them.
    - ``keys.ACTION_TIMESTAMP`` is seconds from the earliest instant, so the list replays at the
      cadence it was recorded at.
    - An action carries a channel only where the recording commanded it; omitting one emits nothing
      there, so the rig holds. Any channel alone replays alone.
    """
    arm = _arm_commands(episode)
    grips = _grip_signals(episode)
    # Asked whether or not absolutes were found: a recording that switched part-way carries both, and
    # replaying only the reissuable stretch holds the arm still through motion the recording made.
    if unreplayable := _unreplayable_arm_signals(episode):
        raise ValueError(
            f'Episode carries arm commands this cannot reissue: {unreplayable}. Only '
            f'{[suffix for suffix, _ in _ARM_SUFFIXES]} can be replayed — a delta means something only '
            f'against the state it was issued from, so the stretch it covers cannot be reconstructed '
            f'and replaying around it would present a partial trajectory as a faithful one.'
        )
    if not arm and not grips:
        raise ValueError(f'Episode records nothing replayable: it carries {sorted(episode.signals)}.')

    grip_stamps = {name: {ts for _, ts in signal} for name, signal in grips.items()}
    stamps: set[int] = set()
    for per_channel in arm.values():
        stamps |= per_channel.keys()
    for name_stamps in grip_stamps.values():
        stamps |= name_stamps
    first_ts = min(stamps)
    actions = []
    for ts in sorted(stamps):
        action: dict[str, Any] = {keys.ACTION_TIMESTAMP: (ts - first_ts) / 1e9}
        for channel, per_channel in arm.items():
            if ts in per_channel:
                action[channel] = per_channel[ts]
        for name, signal in grips.items():
            # A grip holds until changed, so it carries from its first sample on, sampled at or before.
            if ts >= min(grip_stamps[name]):
                action[name] = float(cast(tuple[Any, int], cast(Signal[np.ndarray], signal).time[ts])[0])
        actions.append(action)
    return actions


class ReplaySession(Session):
    """Hands out the recording in chunks, in order, and holds once it is spent.

    One session is one playback: the harness opens a session per episode, so every episode replays the
    recording from its first waypoint.
    """

    def __init__(self, actions: list[dict[str, Any]], chunk_sec: float):
        self._actions = actions
        self._chunk_sec = chunk_sec
        self._cursor = 0

    def __call__(self, obs: dict[str, Any]) -> list[dict[str, Any]] | None:
        if self._cursor >= len(self._actions):
            return None  # spent: no new trajectory, so the rig holds where the recording left it
        start = self._actions[self._cursor][keys.ACTION_TIMESTAMP]
        chunk = []
        while self._cursor < len(self._actions):
            action = self._actions[self._cursor]
            # A chunk keeps its final waypoint for the next one: the wrapper re-queries when it falls due,
            # and the new trajectory replaces the playing one before it is applied, so it would be lost.
            if len(chunk) > 1 and action[keys.ACTION_TIMESTAMP] - start >= self._chunk_sec:
                self._cursor -= 1
                break
            chunk.append({**action, keys.ACTION_TIMESTAMP: action[keys.ACTION_TIMESTAMP] - start})
            self._cursor += 1
        return chunk

    @property
    def meta(self) -> dict[str, Any]:
        return {keys.TYPE: 'replay'}


# The meta fields a replay reports beside ``keys.TYPE``.
META_DATASET_PATH = 'replay.dataset_path'
META_EPISODE = 'replay.episode'

# What a replay's ``sampling_key`` is marked with, so a recorded key says which kind of policy it named.
SAMPLING_KEY_PREFIX = 'replay:'


class ReplayPolicy(Policy):
    """Plays a recorded episode back through the policy interface, with no model and no server.

    It answers the question a rig asks of a policy — "what should the arm do next?" — from a recording
    instead of from inference, so an operator surface, a recorder and an embodiment can be exercised
    end to end with nothing served. Each episode replays the recording from the start; when the
    recording runs out the rig holds, and the operator ends the episode as they would any other.

    The dataset must be one the replaying embodiment can execute: the commands are re-issued verbatim,
    so a recording made against a different action space (joint targets for an arm driven in cartesian
    space, or the reverse) reaches the driver as commands it will interpret in its own terms. A
    recording made in the same sim is the faithful case.

    ``chunk_sec`` is how much of the recording each inference hands over. It only changes how the
    playback is parcelled up — the waypoints keep their recorded spacing either way — so it trades the
    re-query rate against how long a cancelled episode keeps executing.
    """

    def __init__(self, dataset_path: str, episode: int = 0, chunk_sec: float = 1.0):
        self._dataset_path = dataset_path
        self._episode_index = episode
        self._chunk_sec = chunk_sec
        self._actions: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._actions is None:
            dataset = LocalDataset(pos3.sync(self._dataset_path))
            if not 0 <= self._episode_index < len(dataset):
                raise IndexError(
                    f'{self._dataset_path} holds {len(dataset)} episode(s); no episode {self._episode_index}'
                )
            self._actions = load_actions(cast(Episode, dataset[self._episode_index]))
            span = self._actions[-1][keys.ACTION_TIMESTAMP] if self._actions else 0.0
            logger.info(
                'Replaying %s episode %d: %d waypoints over %.1fs',
                self._dataset_path,
                self._episode_index,
                len(self._actions),
                span,
            )
        return self._actions

    def new_session(self, context=None, now=None) -> Session:
        session = ReplaySession(self._load(), self._chunk_sec)
        # The same scheduling wrapper a served policy runs under, so the harness re-queries at the end
        # of each chunk and the chunk's relative timestamps are anchored to the live clock.
        return ChunkedSchedule().wrap_session(session, context, now)

    @property
    def meta(self) -> dict[str, Any]:
        """What this policy is. Reading it fetches the recording and checks the episode exists."""
        self._load()
        return {keys.TYPE: 'replay', META_DATASET_PATH: self._dataset_path, META_EPISODE: self._episode_index}

    @property
    def sampling_key(self) -> str:
        """The recording it plays: the whole of what makes one replay distinct from another.

        Answered from the constructor arguments, so unlike ``meta`` it fetches nothing.
        """
        return f'{SAMPLING_KEY_PREFIX}{self._dataset_path}#{self._episode_index}'
