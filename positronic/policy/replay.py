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

# The recorded arm-command signals this reads back, most faithful first. Joint targets replay exactly;
# pose targets go back through the driver's IK, so they land on the joint solution of the replaying rig
# rather than the recorded one. The delta forms (``.pose_delta``, ``.joint_deltas``) are deliberately
# absent: a delta means something only against the state it was issued from, so replaying one onto a
# different scene produces a different trajectory while looking like a faithful one.
_ARM_SIGNALS = (
    (keys.TARGET_JOINTS, roboarm_command.JointPosition),
    (keys.TARGET_EE_POSE, roboarm_command.CartesianPosition),
)


def _rebuild(command_type: Any, value: np.ndarray) -> Any:
    if command_type is roboarm_command.CartesianPosition:
        # The quaternion representation ``Serializers.transform_3d`` wrote it in.
        pose = geom.Transform3D.from_vector(np.asarray(value), geom.Rotation.Representation.QUAT)
        return roboarm_command.CartesianPosition(pose)
    return roboarm_command.JointPosition(np.asarray(value))


def _arm_commands(episode: Episode) -> dict[int, Any]:
    """Every recorded arm command, keyed by the instant it was issued at, empty where there is none.

    ``Serializers.robot_command`` maps each command type to its own suffix, so one command writes one
    signal, and a recording whose action space changed mid-episode carries commands in more than one —
    each covering its own stretch of the timeline. Every supported signal contributes its waypoints; a
    recording carrying any unsupported one is refused by ``load_actions`` rather than played in part.
    """
    commands: dict[int, Any] = {}
    for name, command_type in _ARM_SIGNALS:
        signal = episode.signals.get(name)
        if signal is None or len(signal) == 0:
            continue
        for value, ts in signal:
            # ``_ARM_SIGNALS`` is most-faithful-first, so an instant already claimed keeps its command.
            commands.setdefault(ts, _rebuild(command_type, value))
    return commands


def _unreplayable_arm_signals(episode: Episode) -> list[str]:
    """Arm-command signals the recording carries that this cannot reissue — the delta forms."""
    supported = {name for name, _ in _ARM_SIGNALS}
    return sorted(n for n in episode.signals if n.startswith(f'{keys.ROBOT_COMMAND}.') and n not in supported)


def load_actions(episode: Episode) -> list[dict[str, Any]]:
    """The episode's commands as an action list: one entry per instant either channel was commanded at.

    Each channel keeps the timing it was recorded with rather than the other's cadence, so a grip
    command issued between two arm waypoints falls due between them, and one issued after the last of
    them still falls due.

    ``keys.ACTION_TIMESTAMP`` is seconds from the earliest instant, so the list replays at the cadence
    it was recorded at. An action carries a channel only where the recording commanded it: the arm
    where a command was issued at that instant, the grip from its first sample onwards (sampled at or
    before the instant, since a grip holds until changed). An action omitting a channel emits nothing
    on it, so the rig holds what it has there, which is what commanding nothing means. In practice the
    two channels are emitted together and land on the same instants; an episode with no grip channel
    replays the arm alone, and one that only ever commanded the grip replays the grip alone.
    """
    arm = _arm_commands(episode)
    grip = episode.signals.get(keys.TARGET_GRIP)
    grip_stamps = {ts for _, ts in grip} if grip is not None and len(grip) > 0 else set()
    # Asked whether or not absolutes were found. A recording that switched action space part-way carries
    # both, and replaying the stretches this can reissue while dropping the rest holds the arm still
    # through motion the recording made — a partial trajectory presented as faithful playback.
    if unreplayable := _unreplayable_arm_signals(episode):
        raise ValueError(
            f'Episode carries arm commands this cannot reissue: {unreplayable}. Only '
            f'{[name for name, _ in _ARM_SIGNALS]} can be replayed — a delta means something only '
            f'against the state it was issued from, so the stretch it covers cannot be reconstructed '
            f'and replaying around it would present a partial trajectory as a faithful one.'
        )
    if not arm and not grip_stamps:
        raise ValueError(f'Episode records nothing replayable: it carries {sorted(episode.signals)}.')
    stamps = sorted(arm.keys() | grip_stamps)
    first_ts = stamps[0]
    grip_start = min(grip_stamps) if grip_stamps else 0
    actions = []
    for ts in stamps:
        action: dict[str, Any] = {keys.ACTION_TIMESTAMP: (ts - first_ts) / 1e9}
        if ts in arm:
            action[keys.ROBOT_COMMAND] = arm[ts]
        if grip_stamps and ts >= grip_start:
            sample = cast(tuple[Any, int], cast(Signal[np.ndarray], grip).time[ts])
            action[keys.TARGET_GRIP] = float(sample[0])
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
            # A chunk keeps its final waypoint for the next one, which re-issues it at the same instant.
            # The scheduling wrapper re-queries the moment that waypoint falls due, and in a rig that runs
            # both in one process the new trajectory replaces the playing one before the waypoint is
            # applied — so a chunk that ended on it would lose it. Re-issuing an absolute target the rig
            # has already reached commands nothing new.
            if len(chunk) > 1 and action[keys.ACTION_TIMESTAMP] - start >= self._chunk_sec:
                self._cursor -= 1
                break
            chunk.append({**action, keys.ACTION_TIMESTAMP: action[keys.ACTION_TIMESTAMP] - start})
            self._cursor += 1
        return chunk

    @property
    def meta(self) -> dict[str, Any]:
        return {keys.TYPE: 'replay'}


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
        return {keys.TYPE: 'replay', 'replay.dataset_path': self._dataset_path, 'replay.episode': self._episode_index}
