import ctypes
import ctypes.util
import errno
import mmap
import os
import socket
from collections.abc import Generator
from types import MappingProxyType
from unittest.mock import ANY

import numpy as np
import pytest

from positronic import keys
from positronic.drivers.roboarm.command import (
    CartesianPosition,
    Impedance,
    JointDelta,
    JointPosition,
    PositionControl,
    to_wire,
)
from positronic.geom import Rotation, Transform3D
from positronic.offboard import frame_ring
from positronic.offboard.client import InferenceClient
from positronic.offboard.protocol import deserialise, serialise, typed_commands
from positronic.offboard.server import PolicyServer
from positronic.utils.serialization import encode_jpeg


def test_inference_client_connect_and_infer(inference_server, mock_policy):
    """Test standard client connection and inference flow."""
    host, port = inference_server
    client = InferenceClient(f'{host}:{port}')

    session = client.new_session()
    try:
        # 1. Verify Metadata Handshake
        assert session.metadata['model_name'] == 'test_model'

        # 2. Verify Inference
        obs = {'image': 'test'}
        action = session.infer(obs)

        assert action['action_data'] == [1, 2, 3]
        mock_policy._mock_session.assert_called_with(obs, ANY)
    finally:
        session.close()


def test_inference_client_new_session(inference_server, mock_policy):
    """Test that starting a new session calls new_session on the policy."""
    host, port = inference_server
    client = InferenceClient(f'{host}:{port}')

    # First session
    session = client.new_session()
    session.close()

    # Second session
    session = client.new_session()
    session.close()

    assert mock_policy.new_session.call_count == 2


def test_session_url_selects_the_model(multi_policy_server):
    host, port, policies = multi_policy_server
    endpoint = f'{host}:{port}'

    default_session = InferenceClient(endpoint).new_session()
    try:
        assert default_session.metadata['model_name'] == 'alpha'
        action = default_session.infer({'obs': 'default'})
        assert action['action_data'] == ['alpha']
    finally:
        default_session.close()

    alpha_session = InferenceClient(f'{endpoint}/api/v1/session/alpha').new_session()
    try:
        assert alpha_session.metadata['model_name'] == 'alpha'
        action = alpha_session.infer({'obs': 'alpha'})
        assert action['action_data'] == ['alpha']
    finally:
        alpha_session.close()

    beta_session = InferenceClient(f'{endpoint}/api/v1/session/beta').new_session()
    try:
        assert beta_session.metadata['model_name'] == 'beta'
        action = beta_session.infer({'obs': 'beta'})
        assert action['action_data'] == ['beta']
    finally:
        beta_session.close()

    policies['alpha']._mock_session.assert_any_call({'obs': 'alpha'}, ANY)
    policies['beta']._mock_session.assert_any_call({'obs': 'beta'}, ANY)
    policies['alpha']._mock_session.assert_any_call({'obs': 'default'}, ANY)


def test_wire_serialisation_accepts_mappingproxy():
    backing = {'a': 1, 'b': {'c': 2}}
    frozen = MappingProxyType(backing)
    payload = {'obs': frozen}

    round_trip = deserialise(serialise(payload))

    # mappingproxy is normalized to a plain dict for the wire.
    assert round_trip == {'obs': {'a': 1, 'b': {'c': 2}}}


def test_jpeg_round_trips_single_image_and_stack():
    """An ``encode_jpeg`` marker survives the wire and decodes back to the original shape and order."""
    single = np.full((16, 24, 3), 90, dtype=np.uint8)
    restored_single = deserialise(serialise({keys.WRIST_IMAGE: encode_jpeg(single)}))[keys.WRIST_IMAGE]
    assert isinstance(restored_single, np.ndarray)
    assert restored_single.shape == (16, 24, 3)
    assert restored_single.dtype == np.uint8
    np.testing.assert_allclose(restored_single, single, atol=4)

    stack = np.stack([np.full((16, 24, 3), (t + 1) * 60, dtype=np.uint8) for t in range(3)])
    restored_stack = deserialise(serialise({keys.WRIST_IMAGE: encode_jpeg(stack)}))[keys.WRIST_IMAGE]
    assert restored_stack.shape == (3, 16, 24, 3)
    # q90 JPEG on solid colors is near-lossless; this also verifies per-frame order is preserved.
    np.testing.assert_allclose(restored_stack, stack, atol=4)


class TestCommandEnvelope:
    """A ``CommandType`` sitting anywhere in the payload crosses as the ``__cmd__`` envelope and comes back
    typed, without the receiver having to know which channel carries it."""

    def test_cartesian_position(self):
        pose = Transform3D(translation=np.array([0.1, 0.2, 0.3], dtype=np.float32), rotation=Rotation.identity)
        cmd = CartesianPosition(pose=pose)
        result = deserialise(serialise(cmd))
        assert isinstance(result, CartesianPosition)
        np.testing.assert_allclose(result.pose.translation, [0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(result.pose.rotation.as_quat, Rotation.identity.as_quat, atol=1e-6)

    def test_joint_position(self):
        positions = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float32)
        result = deserialise(serialise(JointPosition(positions=positions)))
        assert isinstance(result, JointPosition)
        np.testing.assert_allclose(result.positions, positions)

    def test_joint_delta(self):
        velocities = np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07], dtype=np.float32)
        result = deserialise(serialise(JointDelta(velocities=velocities)))
        assert isinstance(result, JointDelta)
        np.testing.assert_allclose(result.velocities, velocities)

    def test_control_modes(self):
        zeros = np.zeros(3)
        impedance = Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
        assert deserialise(serialise(JointDelta(velocities=zeros))).mode is None
        pinned = PositionControl()
        assert deserialise(serialise(JointDelta(velocities=zeros, mode=pinned))).mode == pinned
        assert deserialise(serialise(JointDelta(velocities=zeros, mode=impedance))).mode == impedance
        custom = Impedance(kq=(20.0,) * 7, kqd=(2.0,) * 7, kx=(0.0,) * 6, kxd=(0.0,) * 6)
        assert deserialise(serialise(JointDelta(velocities=zeros, mode=custom))).mode == custom
        stiff = PositionControl(stiffness=(100.0,) * 7)
        assert deserialise(serialise(JointDelta(velocities=zeros, mode=stiff))).mode == stiff

    def test_a_stiffness_the_mode_does_not_name_is_absent_from_the_wire(self):
        """Naming no gains is `None`, as pinning no mode is: neither is a value with a width."""
        with pytest.raises(ValueError, match='at least one joint'):
            PositionControl(stiffness=())
        assert 'stiffness' not in to_wire(PositionControl())
        assert deserialise(serialise(JointDelta(velocities=np.zeros(3), mode=PositionControl()))).mode.stiffness is None

    def test_an_impedance_half_is_disabled_by_zeroing_it(self):
        """An empty half would record as a zero-width vector, which no later gain vector could follow."""
        with pytest.raises(ValueError, match='at least one axis'):
            Impedance(kq=(), kqd=(), kx=(750.0,) * 6, kxd=(37.0,) * 6)
        cartesian_only = Impedance(kq=(0.0,) * 7, kqd=(0.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
        assert deserialise(serialise(JointDelta(velocities=np.zeros(3), mode=cartesian_only))).mode == cartesian_only

    def test_a_bare_wire_mapping_carries_a_nested_mode(self):
        """A server built on another stack sends plain data, mode included, and it types like any other."""
        impedance = Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
        wire = to_wire(JointDelta(velocities=np.zeros(7), mode=impedance))
        typed = typed_commands({keys.ROBOT_COMMAND: wire})[keys.ROBOT_COMMAND]
        assert isinstance(typed, JointDelta)
        assert typed.mode == impedance

    def test_action_trajectory_payload(self):
        """The actual server→client payload: a list of action dicts with embedded Commands."""
        pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
        joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        actions = [
            {keys.ROBOT_COMMAND: CartesianPosition(pose=pose), 'target_grip': 0.5, 'timestamp': 0.0},
            # An action need not carry every channel: this one moves the arm and leaves the gripper be.
            {keys.ROBOT_COMMAND: JointPosition(positions=joints), 'timestamp': 0.1},
        ]
        result = deserialise(serialise({'result': actions}))['result']
        assert len(result) == 2
        assert isinstance(result[0][keys.ROBOT_COMMAND], CartesianPosition)
        assert isinstance(result[1][keys.ROBOT_COMMAND], JointPosition)
        assert result[0]['target_grip'] == 0.5
        assert result[1]['timestamp'] == 0.1
        np.testing.assert_allclose(result[1][keys.ROBOT_COMMAND].positions, joints)

    def test_plain_dict_passthrough(self):
        """Dicts without Commands round-trip unchanged."""
        payload = {'action_data': [1, 2, 3], 'meta': {'k': 'v'}}
        assert deserialise(serialise(payload)) == payload


_WIRE_POSE = [0.4, 0.0, 0.6, 1, 0, 0, 0, 1, 0, 0, 0, 1]  # translation + a 3x3 rotation, the wire's own layout


# rules-allow: hardcoded-keys — every wire name below is spelled the way a server sends it: the command
# mapping, and the `target_grip` / `timestamp` fields an action carries. Reading the decoder's own constants
# would make test and decoder agree whatever those names became, leaving the wire itself unpinned.
class TestServedCommandDecode:
    """An endpoint that does not speak the ``__cmd__`` envelope answers with the command as a bare mapping;
    ``typed_commands`` types it off the channel it sits at."""

    @staticmethod
    def _wire_command(pose=_WIRE_POSE) -> dict:
        """One served ``cartesian_pos`` command, in the wire's own layout."""
        return {'type': 'cartesian_pos', 'pose': pose}

    def test_a_served_command_arrives_typed(self):
        served = typed_commands([{keys.ROBOT_COMMAND: self._wire_command(), 'target_grip': 0.5, 'timestamp': 0.0}])[0]

        decoded = served[keys.ROBOT_COMMAND]
        assert isinstance(decoded, CartesianPosition), f'the driver would be handed {decoded!r}'
        np.testing.assert_allclose(decoded.pose.translation, [0.4, 0.0, 0.6], atol=1e-6)
        assert served['target_grip'] == 0.5 and served['timestamp'] == 0.0  # the rest of the action survives

    def test_the_vector_decodes_from_a_plain_sequence_as_from_an_array(self):
        """A transport may hand the vector back as a list rather than an array; either decodes the same."""
        pose = np.asarray(_WIRE_POSE, dtype=np.float32)

        from_array = typed_commands({keys.ROBOT_COMMAND: self._wire_command(pose)})
        from_list = typed_commands({keys.ROBOT_COMMAND: self._wire_command(pose.tolist())})

        np.testing.assert_allclose(
            from_list[keys.ROBOT_COMMAND].pose.translation, from_array[keys.ROBOT_COMMAND].pose.translation, atol=1e-6
        )

    def test_every_arm_of_a_multi_arm_action_decodes(self):
        """A bimanual embodiment names its channels ``robot_command.{side}``; every one of them decodes."""
        wire = self._wire_command()
        served = typed_commands({f'{keys.ROBOT_COMMAND}.left': wire, f'{keys.ROBOT_COMMAND}.right': wire})

        for side in ('left', 'right'):
            got = served[f'{keys.ROBOT_COMMAND}.{side}']
            assert isinstance(got, CartesianPosition), f'the {side} driver would be handed {got!r}'

    def test_a_command_channel_carrying_a_vector_is_left_alone(self):
        """``robot_command.pose`` / ``.joints`` share the prefix but carry a vector, so a prefix match alone
        would hand them to ``from_wire``; reading the value is what tells them apart."""
        pose = np.zeros(7, dtype=np.float32)

        served = typed_commands({keys.TARGET_EE_POSE: pose, keys.TARGET_JOINTS: pose})

        np.testing.assert_array_equal(served[keys.TARGET_EE_POSE], pose)
        np.testing.assert_array_equal(served[keys.TARGET_JOINTS], pose)

    def test_a_typed_command_and_a_sentinel_are_left_alone(self):
        """A command the ``__cmd__`` envelope already typed, and a result carrying no command channel at
        all, both come back unchanged."""
        typed = CartesianPosition(pose=Transform3D(translation=np.zeros(3), rotation=Rotation.identity))
        assert typed_commands({keys.ROBOT_COMMAND: typed})[keys.ROBOT_COMMAND] is typed
        assert typed_commands({'timestamp': 1.6}) == {'timestamp': 1.6}
        assert typed_commands(None) is None

    def test_a_command_crossing_the_wire_arrives_typed_from_either_shape(self):
        """The two shapes converge: the envelope a positronic server writes, and the bare mapping a partner
        endpoint sends, reach the rig as the same typed command."""
        pose = Transform3D(translation=np.array([0.4, 0.0, 0.6], dtype=np.float32), rotation=Rotation.identity)
        enveloped = {keys.ROBOT_COMMAND: CartesianPosition(pose=pose)}
        bare = {keys.ROBOT_COMMAND: self._wire_command()}

        from_envelope = typed_commands(deserialise(serialise(enveloped)))[keys.ROBOT_COMMAND]
        from_bare = typed_commands(deserialise(serialise(bare)))[keys.ROBOT_COMMAND]

        assert isinstance(from_envelope, CartesianPosition) and isinstance(from_bare, CartesianPosition)
        np.testing.assert_allclose(from_bare.pose.translation, from_envelope.pose.translation, atol=1e-6)


SESSION_ID = 'session-under-test'

pytestmark_ring = pytest.mark.skipif(
    not frame_ring.SUPPORTED, reason='a frame ring needs memfd_create, which Linux has and macOS has not'
)


@pytest.fixture
def frame_channel(socket_path: str) -> Generator[frame_ring.FrameChannel, None, None]:
    channel = frame_ring.FrameChannel(frame_ring.channel_path(socket_path))
    channel.start(PolicyServer.claim_socket_path(channel.path, socket.SOCK_SEQPACKET))
    channel.open_session(SESSION_ID)
    yield channel
    channel.close()


@pytest.fixture
def frame_writer(socket_path: str, frame_channel) -> Generator[frame_ring.FrameWriter, None, None]:
    writer = frame_ring.FrameWriter(frame_ring.channel_path(socket_path), SESSION_ID)
    yield writer
    writer.close()


_PIXELS = np.random.default_rng(0)


def _image(height: int = 48, width: int = 64) -> np.ndarray:
    return _PIXELS.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _over_the_wire(obs):
    """``obs`` as the server reads it, so a reference is tested through msgpack rather than beside it."""
    return deserialise(serialise(obs))


@pytestmark_ring
def test_a_ring_carries_every_image_of_an_observation_byte_identical(frame_channel, frame_writer):
    image, stack = _image(), np.stack([_image(), _image()])
    obs = {'image.left': image, keys.GRIP: 0.5, 'nested': {'frames': [image, stack]}}

    served = frame_channel.resolve(SESSION_ID, _over_the_wire(frame_writer.pack(obs)))

    np.testing.assert_array_equal(served['image.left'], image)
    np.testing.assert_array_equal(served['nested']['frames'][0], image)
    np.testing.assert_array_equal(served['nested']['frames'][1], stack)
    assert served[keys.GRIP] == 0.5


@pytestmark_ring
def test_a_packed_observation_leaves_the_pixels_out_of_the_message(frame_writer):
    image = _image(720, 1280)

    message = serialise(frame_writer.pack({'image.left': image}))

    assert len(message) < image.nbytes // 100


@pytestmark_ring
def test_an_observation_with_no_image_creates_no_ring(frame_writer):
    assert frame_writer.pack({keys.GRIP: 0.5}) == {keys.GRIP: 0.5}
    assert frame_writer._ring is None


@pytestmark_ring
def test_a_sealed_ring_refuses_a_write_a_resize_and_a_hole():
    ring = frame_ring.FrameRing(1024, slots=2)
    try:
        with pytest.raises(PermissionError):
            mmap.mmap(ring.fd, 1024, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        with pytest.raises(PermissionError):
            os.write(ring.fd, b'x')
        with pytest.raises(PermissionError):
            os.ftruncate(ring.fd, 1 << 20)
        libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
        # FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE, from linux/falloc.h.
        assert libc.fallocate(ring.fd, 0x01 | 0x02, ctypes.c_int64(0), ctypes.c_int64(1024)) == -1
        assert ctypes.get_errno() == errno.EPERM
    finally:
        ring.close()


@pytestmark_ring
def test_a_slot_written_over_before_it_is_read_is_refused(frame_channel, frame_writer):
    stale = _over_the_wire(frame_writer.pack({'image.left': _image()}))
    for _ in range(frame_ring.SLOTS):
        frame_writer.pack({'image.left': _image()})

    with pytest.raises(frame_ring.TornFrame):
        frame_channel.resolve(SESSION_ID, stale)


@pytestmark_ring
def test_a_reference_that_points_outside_the_ring_is_refused(frame_channel, frame_writer):
    packed = _over_the_wire(frame_writer.pack({'image.left': _image()}))
    packed['image.left'][frame_ring._OFFSET] = 1 << 30

    with pytest.raises(ValueError, match='outside the ring'):
        frame_channel.resolve(SESSION_ID, packed)


@pytestmark_ring
def test_a_reference_from_a_session_that_handed_over_no_ring_is_refused(frame_channel):
    frame_channel.open_session('another-session')

    with pytest.raises(RuntimeError, match='before it handed over a ring'):
        frame_channel.resolve('another-session', {frame_ring._RING: True})


@pytestmark_ring
def test_a_larger_frame_grows_the_ring_and_leaves_the_earlier_view_readable(frame_channel, frame_writer):
    small, large = _image(8, 8), _image(64, 64)

    first = frame_channel.resolve(SESSION_ID, _over_the_wire(frame_writer.pack({'image.left': small})))
    second = frame_channel.resolve(SESSION_ID, _over_the_wire(frame_writer.pack({'image.left': large})))

    np.testing.assert_array_equal(second['image.left'], large)
    np.testing.assert_array_equal(first['image.left'], small)


@pytestmark_ring
def test_closing_a_channel_that_waits_for_a_handover_ends_its_thread(socket_path):
    channel = frame_ring.FrameChannel(frame_ring.channel_path(socket_path))
    channel.start(PolicyServer.claim_socket_path(channel.path, socket.SOCK_SEQPACKET))
    accepting = channel._thread

    channel.close()

    assert accepting is not None and not accepting.is_alive()
