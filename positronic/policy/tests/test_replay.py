import shutil
from typing import Any, cast

import numpy as np
import pos3
import pytest

from positronic import keys
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers.roboarm import command as roboarm_command
from positronic.policy.base import SampledPolicy
from positronic.policy.replay import ReplayPolicy, ReplaySession, load_actions

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
    return _write(root, {keys.TARGET_JOINTS: joints, keys.TARGET_GRIP: grips})


def test_load_actions_rebuilds_joint_commands_at_recorded_cadence(tmp_path):
    actions = load_actions(_joint_fixture(tmp_path))

    assert len(actions) == 5
    assert [a[keys.ACTION_TIMESTAMP] for a in actions] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    for i, action in enumerate(actions):
        command = action[keys.ROBOT_COMMAND]
        assert isinstance(command, roboarm_command.JointPosition)
        assert command.positions[0] == pytest.approx(i)
        assert action[keys.TARGET_GRIP] == pytest.approx(i / 5)


def test_load_actions_omits_the_grip_on_waypoints_before_the_grip_channel_starts(tmp_path):
    """A grip command is only ever applied at or after the instant it was recorded at.

    The arm can start streaming before the gripper does. The only grip sample available to those early
    waypoints is a future one, and attaching it would move the gripper earlier than the recording did.
    """
    step = int(1e9 / HZ)
    start = 1_000_000_000
    q = np.zeros(7, dtype=np.float32)
    episode = _write(
        tmp_path,
        {
            keys.TARGET_JOINTS: [(q, start + i * step) for i in range(4)],
            # The gripper closes at the third arm waypoint and not before.
            keys.TARGET_GRIP: [(np.float32(1.0), start + 2 * step)],
        },
    )

    actions = load_actions(episode)

    assert [keys.TARGET_GRIP in a for a in actions] == [False, False, True, True]
    assert actions[2][keys.TARGET_GRIP] == pytest.approx(1.0)
    assert actions[3][keys.TARGET_GRIP] == pytest.approx(1.0)


def test_load_actions_keeps_a_grip_command_issued_between_or_after_the_arm_waypoints(tmp_path):
    """The grip keeps the timing it was recorded with, rather than the arm's cadence.

    Keying actions on the arm alone delays a grip command issued between two arm waypoints and drops
    one issued after the last of them — so a recording that ends by closing the gripper replays
    without ever closing it.
    """
    step = int(1e9 / HZ)
    start = 1_000_000_000
    q = np.zeros(7, dtype=np.float32)
    episode = _write(
        tmp_path,
        {
            keys.TARGET_JOINTS: [(q, start), (q, start + 2 * step)],
            # One grip command halfway between the two arm waypoints, one after the last of them.
            keys.TARGET_GRIP: [(np.float32(0.5), start + step), (np.float32(1.0), start + 3 * step)],
        },
    )

    actions = load_actions(episode)

    assert [a[keys.ACTION_TIMESTAMP] for a in actions] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    # The grip-only instants carry no arm command: the recording issued none there.
    assert [keys.ROBOT_COMMAND in a for a in actions] == [True, False, True, False]
    assert actions[1][keys.TARGET_GRIP] == pytest.approx(0.5)
    assert actions[3][keys.TARGET_GRIP] == pytest.approx(1.0)


def test_load_actions_replays_both_arm_streams_when_the_action_space_changed(tmp_path):
    """A command writes one signal, so a recording that switched action space carries both.

    Reading one and dropping the other would replay part of the recording as though it were all of it.
    """
    step = int(1e9 / HZ)
    start = 1_000_000_000
    q = np.full(7, 0.25, dtype=np.float32)
    pose = np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    episode = _write(
        tmp_path,
        {
            keys.TARGET_JOINTS: [(q, start), (q, start + step)],
            keys.TARGET_EE_POSE: [(pose, start + 2 * step), (pose, start + 3 * step)],
        },
    )

    actions = load_actions(episode)

    assert len(actions) == 4
    assert [type(a[keys.ROBOT_COMMAND]) for a in actions] == [
        roboarm_command.JointPosition,
        roboarm_command.JointPosition,
        roboarm_command.CartesianPosition,
        roboarm_command.CartesianPosition,
    ]
    assert [a[keys.ACTION_TIMESTAMP] for a in actions] == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_load_actions_rebuilds_pose_commands(tmp_path):
    pose = np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    episode = _write(tmp_path, {keys.TARGET_EE_POSE: [(pose, 1_000_000_000), (pose + 0.1, 1_100_000_000)]})

    actions = load_actions(episode)

    assert len(actions) == 2
    command = actions[0][keys.ROBOT_COMMAND]
    assert isinstance(command, roboarm_command.CartesianPosition)
    assert command.pose.translation == pytest.approx([0.3, 0.0, 0.5], abs=1e-6)
    assert keys.TARGET_GRIP not in actions[0]  # the fixture records no grip channel


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
            keys.TARGET_GRIP: [(np.float32(0.25), 1_000_000_000), (np.float32(0.75), 3_000_000_000)],
        },
    )

    actions = load_actions(episode)

    assert actions[0][keys.TARGET_GRIP] == pytest.approx(0.25)


def test_session_hands_out_the_recording_in_chunks_then_holds(tmp_path):
    # The bare session, without the scheduling wrapper: one call, one chunk.
    session = ReplaySession(load_actions(_joint_fixture(tmp_path, count=5)), chunk_sec=0.25)

    chunks = [session({}) for _ in range(4)]

    assert chunks[2] is None and chunks[3] is None  # spent: no new trajectory, so the rig holds
    assert chunks[0] is not None and chunks[1] is not None
    assert [len(chunks[0]), len(chunks[1])] == [3, 3]
    played = [action for chunk in chunks[:2] if chunk for action in chunk]
    # Waypoint 2 opens the second chunk as well as closing the first: the boundary is handed over, not spent.
    assert [a[keys.ROBOT_COMMAND].positions[0] for a in played] == pytest.approx([0, 1, 2, 2, 3, 4])
    # Each chunk's timestamps restart at its own first waypoint — the scheduling wrapper anchors them.
    assert [a[keys.ACTION_TIMESTAMP] for a in chunks[0]] == pytest.approx([0.0, 0.1, 0.2])
    assert [a[keys.ACTION_TIMESTAMP] for a in chunks[1]] == pytest.approx([0.0, 0.1, 0.2])


def test_a_chunk_longer_than_the_recording_plays_it_in_one_go(tmp_path):
    session = ReplaySession(load_actions(_joint_fixture(tmp_path, count=5)), chunk_sec=10.0)

    chunk = session({})

    assert chunk is not None and len(chunk) == 5
    assert session({}) is None


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
    assert [a[keys.ACTION_TIMESTAMP] for a in first] == pytest.approx([100.0, 100.1, 100.2])
    assert session({'obs_time_ns': 0}) is None  # the chunk is still playing

    now = 100.2
    second = session({'obs_time_ns': 0})
    assert second is not None
    # The handed-over waypoint keeps the instant it had in the first chunk, so the re-issue changes no timing.
    assert [a[keys.ACTION_TIMESTAMP] for a in second] == pytest.approx([100.2, 100.3, 100.4])
    assert second[0][keys.ROBOT_COMMAND].positions[0] == first[-1][keys.ROBOT_COMMAND].positions[0]


def test_every_waypoint_plays_when_a_new_chunk_lands_before_the_last_one_is_applied(tmp_path):
    """The order a rig running both in one process gives them: the harness hands the player a new trajectory
    in the round the previous chunk's last waypoint falls due, before the player applies it."""
    _joint_fixture(tmp_path, count=9)
    clock = 100.0
    session = ReplayPolicy(str(tmp_path), chunk_sec=0.25).new_session(now=lambda: clock)
    player = roboarm_command.TrajectoryPlayer()

    played = []
    for tick in range(40):  # half a waypoint apart, so no tick collapses two of them
        clock = 100.0 + tick * (0.5 / HZ)
        chunk = session({'obs_time_ns': int(clock * 1e9)})
        if chunk is not None:
            player.set([(int(a[keys.ACTION_TIMESTAMP] * 1e9), a[keys.ROBOT_COMMAND]) for a in chunk])
        command = player.advance(int(clock * 1e9))
        if command is not None:
            played.append(float(command.positions[0]))

    assert sorted(set(played)) == pytest.approx(list(range(9)))  # a re-issued waypoint may play twice
    assert played == sorted(played)


def test_missing_episode_names_what_the_dataset_holds(tmp_path):
    _joint_fixture(tmp_path, count=2)
    policy = ReplayPolicy(str(tmp_path), episode=7)

    with pytest.raises(IndexError, match='holds 1 episode'):
        policy.new_session()
    # From `meta` too, which a run resolves before hardware, so a missing episode is named at startup.
    with pytest.raises(IndexError, match='holds 1 episode'):
        _ = ReplayPolicy(str(tmp_path), episode=7).meta


def test_reading_meta_fetches_the_recording_so_a_sampled_set_warms_together(tmp_path):
    """A warm-up opens a session on the one endpoint the sampler picks and reaches the others only
    through the `meta` it reads to key them, so `meta` is where a replay does its fetching."""
    a, b = tmp_path / 'a', tmp_path / 'b'
    _joint_fixture(a, count=3)
    _joint_fixture(b, count=3)
    policies = [ReplayPolicy(str(a), chunk_sec=10.0), ReplayPolicy(str(b), chunk_sec=10.0)]

    SampledPolicy(*policies).new_session(now=lambda: 100.0).close()

    # With the datasets gone, a policy that still had fetching to do could not play.
    shutil.rmtree(a)
    shutil.rmtree(b)
    for policy in policies:
        chunk = policy.new_session(now=lambda: 100.0)({'obs_time_ns': 0})
        assert chunk is not None and len(chunk) == 3


def test_meta_names_the_recording_it_plays(tmp_path):
    _joint_fixture(tmp_path)
    policy = ReplayPolicy(str(tmp_path), episode=0)

    assert policy.meta[keys.TYPE] == 'replay'
    assert policy.meta['replay.dataset_path'] == str(tmp_path)
