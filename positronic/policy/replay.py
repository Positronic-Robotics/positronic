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


def _arm_signal(episode: Episode) -> tuple[Signal[np.ndarray], Any]:
    """The episode's arm-command signal and the command type that rebuilds it."""
    for name, command_type in _ARM_SIGNALS:
        signal = episode.signals.get(name)
        if signal is not None and len(signal) > 0:
            return signal, command_type
    recorded = sorted(episode.signals)
    raise ValueError(
        f'Episode records no replayable arm command: expected one of '
        f'{[name for name, _ in _ARM_SIGNALS]}, and it carries {recorded}. An episode recorded from '
        f'delta commands cannot be replayed — deltas only mean anything against the state they were '
        f'issued from.'
    )


def _rebuild(command_type: Any, value: np.ndarray) -> Any:
    if command_type is roboarm_command.CartesianPosition:
        # The quaternion representation ``Serializers.transform_3d`` wrote it in.
        pose = geom.Transform3D.from_vector(np.asarray(value), geom.Rotation.Representation.QUAT)
        return roboarm_command.CartesianPosition(pose)
    return roboarm_command.JointPosition(np.asarray(value))


def load_actions(episode: Episode) -> list[dict[str, Any]]:
    """The episode's commands as an action list: one entry per recorded arm waypoint.

    ``keys.ACTION_TIMESTAMP`` is seconds from the first waypoint, so the list replays at the cadence it
    was recorded at. The grip is sampled at or before each waypoint's time (the two channels are emitted
    together, so in practice they land on the same instants); a waypoint earlier than the grip's first
    sample carries no grip at all, and an episode with no grip channel replays the arm alone.
    """
    arm, command_type = _arm_signal(episode)
    grip = episode.signals.get(keys.TARGET_GRIP)
    first_ts = arm.start_ts
    actions = []
    for value, ts in arm:
        action = {keys.ACTION_TIMESTAMP: (ts - first_ts) / 1e9, keys.ROBOT_COMMAND: _rebuild(command_type, value)}
        if grip is not None and len(grip) > 0 and ts >= grip.start_ts:
            # Sampled at or before the waypoint. An arm waypoint that precedes the grip's first sample
            # gets no grip field: the recording commanded none there, so the only grip available is a
            # future one, and attaching it would close the gripper earlier than the recording did. An
            # action omitting the channel emits nothing on it, so the rig holds the grip it has.
            sample = cast(tuple[Any, int], grip.time[ts])
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
