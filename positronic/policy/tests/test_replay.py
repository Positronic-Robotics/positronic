from typing import Any, cast

import numpy as np
import pos3
import pytest

from positronic import keys
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers.roboarm import command as roboarm_command
from positronic.policy.replay import GRIP_COMMAND, ReplayPolicy, ReplaySession, load_actions

HZ = 10  # waypoints per second in the fixtures below


@pytest.fixture(autouse=True)
def _mirror():
    """``ReplayPolicy`` resolves its dataset path through pos3, which needs a live mirror."""
    with pos3.mirror():
        yield


def _write(root, signals: dict[str, list[tuple[Any, int]]]) -> Episode:
    """One episode holding the given ``name -> [(value, ts_ns)]`` streams."""
    with LocalDatasetWriter(root) as writer, writer.new_episode() as episode:
        for name, samples in signals.items():
            for value, ts_ns in samples:
                episode.append(name, value, ts_ns=ts_ns)
    return cast(Episode, LocalDataset(root)[0])


def _joint_fixture(root, count: int = 5, start_ts: int = 1_000_000_000):
    """``count`` joint waypoints at ``HZ``, joint 0 sweeping, grip closing alongside."""
    step = int(1e9 / HZ)
    joints, grips = [], []
    for i in range(count):
        q = np.full(7, 0.1, dtype=np.float32)
        q[0] = np.float32(i)
        joints.append((q, start_ts + i * step))
        grips.append((np.float32(i / count), start_ts + i * step))
    return _write(root, {keys.TARGET_JOINTS: joints, GRIP_COMMAND: grips})


def test_load_actions_rebuilds_joint_commands_at_recorded_cadence(tmp_path):
    actions = load_actions(_joint_fixture(tmp_path))

    assert len(actions) == 5
    assert [a['timestamp'] for a in actions] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    for i, action in enumerate(actions):
        command = action[keys.ROBOT_COMMAND]
        assert isinstance(command, roboarm_command.JointPosition)
        assert command.positions[0] == pytest.approx(i)
        assert action[GRIP_COMMAND] == pytest.approx(i / 5)


def test_load_actions_rebuilds_pose_commands(tmp_path):
    pose = np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    episode = _write(tmp_path, {keys.TARGET_EE_POSE: [(pose, 1_000_000_000), (pose + 0.1, 1_100_000_000)]})

    actions = load_actions(episode)

    assert len(actions) == 2
    command = actions[0][keys.ROBOT_COMMAND]
    assert isinstance(command, roboarm_command.CartesianPosition)
    assert command.pose.translation == pytest.approx([0.3, 0.0, 0.5], abs=1e-6)
    assert GRIP_COMMAND not in actions[0]  # the fixture records no grip channel


def test_load_actions_prefers_joints_over_pose(tmp_path):
    """Joint targets replay exactly; pose targets go back through the driver's IK."""
    q = np.zeros(7, dtype=np.float32)
    pose = np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    episode = _write(tmp_path, {keys.TARGET_JOINTS: [(q, 1_000_000_000)], keys.TARGET_EE_POSE: [(pose, 1_000_000_000)]})

    command = load_actions(episode)[0][keys.ROBOT_COMMAND]

    assert isinstance(command, roboarm_command.JointPosition)


def test_load_actions_rejects_an_episode_with_no_replayable_arm_command(tmp_path):
    episode = _write(tmp_path, {f'{keys.ROBOT_COMMAND}.pose_delta': [(np.zeros(7, dtype=np.float32), 1_000_000_000)]})

    with pytest.raises(ValueError, match='no replayable arm command'):
        load_actions(episode)


def test_load_actions_holds_the_grip_recorded_before_the_first_waypoint(tmp_path):
    """The two channels can start a beat apart; the grip in force is the last one commanded."""
    q = np.zeros(7, dtype=np.float32)
    episode = _write(
        tmp_path,
        {
            keys.TARGET_JOINTS: [(q, 2_000_000_000)],
            GRIP_COMMAND: [(np.float32(0.25), 1_000_000_000), (np.float32(0.75), 3_000_000_000)],
        },
    )

    actions = load_actions(episode)

    assert actions[0][GRIP_COMMAND] == pytest.approx(0.25)


def test_session_hands_out_the_recording_in_chunks_then_holds(tmp_path):
    # The bare session, without the scheduling wrapper: one call, one chunk.
    session = ReplaySession(load_actions(_joint_fixture(tmp_path, count=5)), chunk_sec=0.25)

    chunks = [session({'obs_time_ns': 0}) for _ in range(4)]

    assert chunks[2] is None and chunks[3] is None  # spent: no new trajectory, so the rig holds
    assert chunks[0] is not None and chunks[1] is not None
    assert [len(chunks[0]), len(chunks[1])] == [3, 2]
    played = [action for chunk in chunks[:2] if chunk for action in chunk]
    assert [a[keys.ROBOT_COMMAND].positions[0] for a in played] == pytest.approx([0, 1, 2, 3, 4])
    # Each chunk's timestamps restart at its own first waypoint — the scheduling wrapper anchors them.
    assert [a['timestamp'] for a in chunks[0]] == pytest.approx([0.0, 0.1, 0.2])
    assert [a['timestamp'] for a in chunks[1]] == pytest.approx([0.0, 0.1])


def test_a_chunk_longer_than_the_recording_plays_it_in_one_go(tmp_path):
    session = ReplaySession(load_actions(_joint_fixture(tmp_path, count=5)), chunk_sec=10.0)

    chunk = session({'obs_time_ns': 0})

    assert chunk is not None and len(chunk) == 5
    assert session({'obs_time_ns': 0}) is None


def test_each_session_replays_the_recording_from_the_start(tmp_path):
    """The harness opens a session per episode, so every episode gets the whole recording."""
    _joint_fixture(tmp_path, count=3)
    policy = ReplayPolicy(str(tmp_path), chunk_sec=10.0)
    now = 100.0

    first = policy.new_session(now=lambda: now)({'obs_time_ns': 0})
    second = policy.new_session(now=lambda: now)({'obs_time_ns': 0})

    assert first is not None and second is not None
    assert len(first) == len(second) == 3
    assert first[0][keys.ROBOT_COMMAND].positions[0] == second[0][keys.ROBOT_COMMAND].positions[0]


def test_the_scheduling_wrapper_paces_playback_against_the_clock(tmp_path):
    """Wrapped as the harness runs it: a chunk is handed out once, then withheld until it has played."""
    policy = ReplayPolicy(str(tmp_path), chunk_sec=0.25)
    _joint_fixture(tmp_path, count=5)
    now = 100.0
    session = policy.new_session(now=lambda: now)

    first = session({'obs_time_ns': 0})
    assert first is not None
    assert [a['timestamp'] for a in first] == pytest.approx([100.0, 100.1, 100.2])
    assert session({'obs_time_ns': 0}) is None  # the chunk is still playing

    now = 100.2
    second = session({'obs_time_ns': 0})
    assert second is not None
    assert [a['timestamp'] for a in second] == pytest.approx([100.2, 100.3])


def test_missing_episode_names_what_the_dataset_holds(tmp_path):
    _joint_fixture(tmp_path, count=2)
    policy = ReplayPolicy(str(tmp_path), episode=7)

    with pytest.raises(IndexError, match='holds 1 episode'):
        policy.new_session()


def test_meta_names_the_recording_it_plays(tmp_path):
    _joint_fixture(tmp_path)
    policy = ReplayPolicy(str(tmp_path), episode=0)

    assert policy.meta[keys.TYPE] == 'replay'
    assert policy.meta['replay.dataset_path'] == str(tmp_path)
