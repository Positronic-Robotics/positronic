"""What a roboarena policy sends a server, and what it makes of the answer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from positronic_wire.roboarena import RoboarenaAddress

from positronic import keys as rig
from positronic.drivers.roboarm import RobotStatus
from positronic.offboard.roboarena import RoboarenaClient
from positronic.policy.codec import ACTION
from positronic.policy.executor import Executor, WaitStatus
from positronic.utils.serialization import serialize
from positronic.vendors.dreamzero import roboarena as wire
from positronic.vendors.dreamzero import roboarena_policy

ANNOUNCED = {
    wire.RESOLUTION: [180, 320],
    wire.NEEDS_WRIST_CAMERA: True,
    wire.NUM_EXTERIOR_CAMERAS: 2,
    wire.NEEDS_STEREO_CAMERA: False,
    wire.NEEDS_SESSION_ID: False,
    wire.ACTION_SPACE: roboarena_policy.JOINT_POSITION_SPACE,
}

# rules-allow: hardcoded-keys — a roboarena server's own spelling, held independent of the code under test:
# a set built from the module's own names would compare that module against itself.
REQUIRED = {
    'observation/joint_position',
    'observation/gripper_position',
    'observation/wrist_image_left',
    'observation/exterior_image_1_left',
    'observation/exterior_image_2_left',
    'prompt',
}

ADDRESS = RoboarenaAddress('a-server-host', 8000)

# The episode clock's reading when a test starts: a machine up for about fourteen days, in `time.monotonic()`
# seconds. The harness schedules against that clock.
UPTIME_NS = 1_191_360 * 1_000_000_000

# How long a server takes to answer one observation, and the cadence the codec plays each chunk at.
ROUND_TRIP_NS = 1_500_000_000
FPS = 15


def announced(**fields):
    """The config a server announces on connect, with `fields` changed."""
    return {**ANNOUNCED, **fields}


def encoded():
    """One observation as the codec writes it, with its exterior cameras counted from 0."""
    return {
        wire.JOINT_POSITION: np.zeros(7, dtype=np.float32),
        wire.GRIPPER_POSITION: np.zeros(1, dtype=np.float32),
        wire.WRIST_IMAGE: np.zeros((180, 320, 3), dtype=np.uint8),
        wire.exterior_image(0): np.zeros((180, 320, 3), dtype=np.uint8),
        wire.exterior_image(1): np.zeros((180, 320, 3), dtype=np.uint8),
        wire.PROMPT: 'pick up the red cube',
    }


def rig_observation(**images):
    """One observation as the harness hands it to a policy, with `images` changed."""
    return {
        rig.JOINTS: np.zeros(7, dtype=np.float32),
        rig.GRIP: np.float32(0.0),
        rig.WRIST_IMAGE: np.zeros((480, 640, 3), dtype=np.uint8),
        rig.EXTERIOR_IMAGE: np.zeros((720, 1280, 3), dtype=np.uint8),
        rig.EXTERIOR_IMAGE_2: np.zeros((720, 1280, 3), dtype=np.uint8),
        rig.TASK: 'pick up the red cube',
        **images,
    }


class FakeClient(RoboarenaClient):
    """A roboarena server that records what it was sent and answers a fixed chunk."""

    def __init__(self, chunk):
        super().__init__(ADDRESS.host, ADDRESS.port)
        self.chunk = chunk
        self.sent = []

    def infer(self, observation):
        self.sent.append(observation)
        return {roboarena_policy.ACTIONS_FIELD: self.chunk}


class Clock:
    """The episode clock a test moves by hand, starting at `UPTIME_NS`."""

    def __init__(self):
        self.now_ns = UPTIME_NS

    def __call__(self):
        return self.now_ns


class SlowClient(FakeClient):
    """A server whose answer arrives `ROUND_TRIP_NS` after the observation was taken."""

    def __init__(self, chunk, clock: Clock):
        super().__init__(chunk)
        self._clock = clock

    def infer(self, observation):
        answer = super().infer(observation)
        self._clock.now_ns += ROUND_TRIP_NS
        return answer


def endpoint_over(client, config=None):
    """The inference one episode submits, built the way the policy builds it."""
    return roboarena_policy.RoboarenaEndpoint(client, announced() if config is None else config)


def start_stack(client, clock: Clock):
    """The stack a run executes, around `client`, on a simulated runtime that reads `clock`.

    Built through `local_stack` rather than here, so the test runs the stack a run executes.
    """
    runtime = Executor(clock, simulated=True, charge_inference_time=False)
    run = runtime.start(roboarena_policy.local_stack(ANNOUNCED), endpoint_over(client, ANNOUNCED))
    return runtime, run


def wait_for_answer(runtime):
    while runtime.wait(timeout_sec=1.0).status is not WaitStatus.ANSWERS_READY:
        pass


def one_inference(client, obs):
    """Send `obs` through the stack a run executes, and wait for the inference it starts."""
    runtime, run = start_stack(client, Clock())
    try:
        run.send(obs)
        wait_for_answer(runtime)
    finally:
        runtime.close()
        run.close()


def play(chunk_rows: int) -> list[tuple[int, dict]]:
    """Drive the stack a run executes through one chunk, as the harness does.

    Returns each step that emitted commands, as (time on the episode clock, commands).
    """
    clock = Clock()
    runtime, run = start_stack(SlowClient(np.zeros((chunk_rows, 8), dtype=np.float32), clock), clock)
    emitted = []
    try:
        run.send(rig_observation())
        wait_for_answer(runtime)
        # Bounded: a stack that plays several rows in one step never emits `chunk_rows` times.
        for _ in range(chunk_rows):
            step = run.send(rig_observation())
            if step.commands:
                emitted.append((clock.now_ns, dict(step.commands)))
            clock.now_ns = max(clock.now_ns, step.resume_at_ns)
    finally:
        runtime.close()
        run.close()
    return emitted


def websocket_answering(*replies):
    """A websocket that gives each of `replies` in turn."""
    return MagicMock(**{'recv.side_effect': list(replies)})


class TestWhatTheServerAsksFor:
    def test_the_announced_config_names_exactly_the_keys_that_server_requires(self):
        assert roboarena_policy.wanted_keys(ANNOUNCED) == frozenset(REQUIRED)

    def test_a_stateless_server_is_sent_no_session_id(self):
        assert wire.SESSION_ID not in roboarena_policy.wanted_keys(ANNOUNCED)

    def test_a_server_tracking_sessions_is_sent_one(self):
        stateful = announced(**{wire.NEEDS_SESSION_ID: True})
        assert wire.SESSION_ID in roboarena_policy.wanted_keys(stateful)

    def test_a_server_wanting_one_exterior_camera_gets_the_first(self):
        keys = roboarena_policy.wanted_keys(announced(**{wire.NUM_EXTERIOR_CAMERAS: 1}))
        # rules-allow: hardcoded-keys — the wire's own spelling, held independent of the code under test
        assert 'observation/exterior_image_1_left' in keys
        assert 'observation/exterior_image_2_left' not in keys

    def test_a_server_wanting_more_exterior_cameras_than_the_codec_writes_is_refused(self):
        with pytest.raises(ValueError, match='exterior cameras'):
            roboarena_policy.wanted_keys(announced(**{wire.NUM_EXTERIOR_CAMERAS: 3}))

    def test_a_server_wanting_no_wrist_camera_gets_none(self):
        keys = roboarena_policy.wanted_keys(announced(**{wire.NEEDS_WRIST_CAMERA: False}))
        assert wire.WRIST_IMAGE not in keys

    def test_a_server_asking_for_stereo_is_refused(self):
        with pytest.raises(ValueError, match='stereo'):
            roboarena_policy.wanted_keys(announced(**{wire.NEEDS_STEREO_CAMERA: True}))

    def test_another_action_space_is_refused_rather_than_decoded_as_joints(self):
        with pytest.raises(ValueError, match='moves the arm wrongly'):
            roboarena_policy.wanted_keys(announced(**{wire.ACTION_SPACE: 'cartesian_position'}))

    def test_the_wire_counts_exterior_cameras_from_one(self):
        # rules-allow: hardcoded-keys — the wire's own spelling, held independent of the code under test
        assert roboarena_policy.renaming() == {
            wire.exterior_image(0): 'observation/exterior_image_1_left',
            wire.exterior_image(1): 'observation/exterior_image_2_left',
        }

    def test_the_announced_resolution_is_read_as_height_then_width(self):
        assert roboarena_policy.image_size(ANNOUNCED) == (320, 180)

    def test_a_server_announcing_no_resolution_is_refused(self):
        with pytest.raises(ValueError, match='no image resolution'):
            roboarena_policy.image_size(announced(**{wire.RESOLUTION: None}))


class TestOneInference:
    def test_the_message_holds_exactly_the_keys_that_server_requires(self):
        client = FakeClient(np.zeros((32, 8), dtype=np.float32))
        endpoint_over(client)({**encoded(), 'observation/exterior_image_2_left_extra': 1})
        assert set(client.sent[0]) == REQUIRED

    def test_an_observation_missing_a_required_key_is_refused_before_it_is_sent(self):
        client = FakeClient(np.zeros((32, 8), dtype=np.float32))
        short = {key: value for key, value in encoded().items() if key != wire.PROMPT}
        with pytest.raises(ValueError, match='prompt'):
            endpoint_over(client)(short)
        assert client.sent == []

    def test_a_one_camera_server_is_sent_the_first_view_and_not_the_second(self):
        client = FakeClient(np.zeros((32, 8), dtype=np.float32))
        obs = {**encoded(), wire.exterior_image(1): np.full((180, 320, 3), 255, dtype=np.uint8)}
        endpoint_over(client, announced(**{wire.NUM_EXTERIOR_CAMERAS: 1}))(obs)

        sent = client.sent[0]
        # rules-allow: hardcoded-keys — the wire's own spelling, held independent of the code under test
        assert sent['observation/exterior_image_1_left'].max() == 0
        assert 'observation/exterior_image_2_left' not in sent

    def test_a_server_tracking_sessions_is_sent_a_new_id_each_episode(self):
        stateful = announced(**{wire.NEEDS_SESSION_ID: True})
        client = FakeClient(np.zeros((32, 8), dtype=np.float32))

        for _ in range(2):
            endpoint_over(client, stateful)(encoded())

        first, second = (sent[wire.SESSION_ID] for sent in client.sent)
        assert first and second and first != second

    def test_a_chunk_row_of_another_width_is_refused_and_named(self):
        with pytest.raises(ValueError, match='reaches the arm as joint positions') as refusal:
            endpoint_over(FakeClient(np.zeros((8, 32), dtype=np.float32)))(encoded())
        assert '(8, 32)' in str(refusal.value)

    @pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
    def test_a_chunk_carrying_a_non_finite_value_is_refused(self, bad):
        chunk = np.zeros((32, 8), dtype=np.float32)
        chunk[7, 3] = bad
        with pytest.raises(ValueError, match='not a finite number'):
            endpoint_over(FakeClient(chunk))(encoded())

    def test_a_chunk_of_no_number_at_all_is_refused_by_the_same_gate(self):
        """`isfinite` raises on a dtype it cannot read, so the gate reads the dtype first."""
        with pytest.raises(ValueError, match='not a finite number'):
            endpoint_over(FakeClient(np.full((32, 8), 'x')))(encoded())

    def test_a_large_finite_action_still_reaches_the_codec(self):
        """The joint limits are the arm's to hold, so the gate refuses only what is not a number."""
        chunk = np.full((32, 8), 1e30, dtype=np.float32)
        assert len(endpoint_over(FakeClient(chunk))(encoded())) == 32

    def test_the_chunk_decodes_into_one_entry_per_row_in_order(self):
        chunk = np.arange(32 * 8, dtype=np.float32).reshape(32, 8)
        answered = endpoint_over(FakeClient(chunk))(encoded())
        assert [entry[ACTION].tolist() for entry in answered] == chunk.tolist()

    def test_a_single_action_is_one_entry(self):
        one = np.arange(8, dtype=np.float32)
        answered = endpoint_over(FakeClient(one))(encoded())
        assert [entry[ACTION].tolist() for entry in answered] == [one.tolist()]


class TestTheWire:
    def test_the_chunk_is_read_out_of_the_reply_s_own_field(self):
        chunk = np.zeros((32, 8), dtype=np.float32)
        websocket = websocket_answering(serialize(ANNOUNCED), serialize({roboarena_policy.ACTIONS_FIELD: chunk}))
        client = RoboarenaClient(ADDRESS.host, ADDRESS.port)
        with patch('positronic_wire.roboarena.connect', return_value=websocket):
            client.connect()
            assert len(endpoint_over(client)(encoded())) == 32

    def test_a_server_error_names_the_endpoint_and_the_server_s_own_words(self):
        websocket = websocket_answering(serialize(ANNOUNCED), 'CUDA out of memory')
        client = RoboarenaClient(ADDRESS.host, ADDRESS.port)
        with patch('positronic_wire.roboarena.connect', return_value=websocket):
            client.connect()
            with pytest.raises(RuntimeError) as raised:
                endpoint_over(client)(encoded())

        assert 'ws://a-server-host:8000' in str(raised.value)
        assert 'CUDA out of memory' in str(raised.value)
        websocket.close.assert_called_once()

    def test_each_episode_opens_its_own_connection_and_closes_it(self):
        sockets = [websocket_answering(serialize(ANNOUNCED)) for _ in range(2)]
        policy = roboarena_policy.RoboarenaPolicy(ADDRESS)

        with patch('positronic_wire.roboarena.connect', side_effect=sockets):
            for _ in range(2):
                runtime = Executor(Clock(), simulated=True, charge_inference_time=False)
                run = runtime.start(policy)
                runtime.close()
                run.close()

        for socket in sockets:
            socket.close.assert_called_once()

    def test_each_episode_builds_its_stack_from_the_config_its_own_connection_announced(self, monkeypatch):
        """A server restarted between two episodes may announce another resolution."""
        second = announced(**{wire.RESOLUTION: [144, 256]})
        sockets = [websocket_answering(serialize(config)) for config in (ANNOUNCED, second)]
        built = []

        def record(config):
            built.append(config)
            raise StopRun

        monkeypatch.setattr(roboarena_policy, 'local_stack', record)
        policy = roboarena_policy.RoboarenaPolicy(ADDRESS)

        with patch('positronic_wire.roboarena.connect', side_effect=sockets):
            for _ in range(2):
                with pytest.raises(StopRun):
                    Executor(Clock(), simulated=True, charge_inference_time=False).start(policy)

        assert [config[wire.RESOLUTION] for config in built] == [[180, 320], [144, 256]]


class StopRun(Exception):
    """Ends a run as it is built, once the config it was built from is recorded."""


class TestTheStack:
    def test_the_rig_s_own_observation_reaches_the_wire_as_that_server_requires_it(self):
        client = FakeClient(np.zeros((32, 8), dtype=np.float32))
        one_inference(client, rig_observation())

        sent = client.sent[0]
        assert set(sent) == REQUIRED
        assert (sent[wire.JOINT_POSITION].shape, sent[wire.JOINT_POSITION].dtype) == ((7,), np.float32)
        assert (sent[wire.GRIPPER_POSITION].shape, sent[wire.GRIPPER_POSITION].dtype) == ((1,), np.float32)
        for camera in REQUIRED - {wire.JOINT_POSITION, wire.GRIPPER_POSITION, wire.PROMPT}:
            assert (sent[camera].shape, sent[camera].dtype) == ((180, 320, 3), np.uint8)
        assert isinstance(sent[wire.PROMPT], str)

    def test_each_over_shoulder_view_reaches_a_slot_of_its_own(self):
        """Black in the first camera and white in the second, so one camera in both slots sends two black frames."""
        client = FakeClient(np.zeros((32, 8), dtype=np.float32))
        one_inference(client, rig_observation(**{rig.EXTERIOR_IMAGE_2: np.full((720, 1280, 3), 255, np.uint8)}))

        sent = client.sent[0]
        # rules-allow: hardcoded-keys — the wire's own spelling, held independent of the code under test
        assert sent['observation/exterior_image_1_left'].max() == 0
        assert sent['observation/exterior_image_2_left'].min() == 255

    def test_a_chunk_plays_from_where_it_arrived_and_not_from_where_it_was_asked_for(self):
        """32 actions at 15 Hz cover 2.13 s. Anchored where it was asked for, most of the chunk is past due
        when a 1.5 s round trip ends, and the arm holds still between chunks."""
        emitted = play(32)

        arrived_ns = UPTIME_NS + ROUND_TRIP_NS
        assert [at for at, _ in emitted] == pytest.approx([arrived_ns + i * 1e9 / FPS for i in range(32)], abs=1)

    def test_every_action_of_the_chunk_reaches_the_arm(self):
        emitted = play(32)

        assert len(emitted) == 32
        assert all(rig.ROBOT_COMMAND in commands for _, commands in emitted)

    @pytest.mark.parametrize('status', [RobotStatus.ERROR, RobotStatus.BUSY])
    def test_an_unavailable_arm_is_sent_no_command_and_the_server_no_observation(self, status):
        client = FakeClient(np.zeros((32, 8), dtype=np.float32))
        runtime, run = start_stack(client, Clock())
        try:
            steps = [run.send(rig_observation(**{rig.ROBOT_STATUS: status})) for _ in range(3)]
        finally:
            runtime.close()
            run.close()

        assert [step.commands for step in steps] == [{}, {}, {}]
        assert client.sent == []

    def test_an_available_arm_is_served(self):
        client = FakeClient(np.zeros((32, 8), dtype=np.float32))
        one_inference(client, rig_observation(**{rig.ROBOT_STATUS: RobotStatus.AVAILABLE}))

        assert len(client.sent) == 1
