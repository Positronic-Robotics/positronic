import asyncio
import ctypes
import ctypes.util
import errno
import fcntl
import logging
import mmap
import os
import socket
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple
from unittest.mock import AsyncMock

import numpy as np
import pytest
from positronic_model_server import server_wire, websocket_wire
from positronic_model_server.serialization import encode_jpeg
from positronic_wire import websocket, wire

from positronic import keys
from positronic.drivers.roboarm.command import (
    FRAME_FIELD,
    CartesianPosition,
    Impedance,
    JointDelta,
    JointPosition,
    PositionControl,
    from_wire,
    to_wire,
)
from positronic.geom import Rotation, Transform3D
from positronic.offboard import frame_ring
from positronic.offboard.client import InferenceClient
from positronic.offboard.protocol import deserialise, serialise, typed_commands


def test_inference_client_connect_and_infer(inference_server, mock_model):
    """Test standard client connection and inference flow."""
    host, port = inference_server
    client = InferenceClient(websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.SESSION_PATH, ''))

    session = client.new_session()
    try:
        # 1. Verify Metadata Handshake
        assert session.metadata['model_name'] == 'test_model'

        # 2. Verify Inference
        obs = {'image': 'test'}
        action = session.infer(obs)

        assert action['action_data'] == [1, 2, 3]
        mock_model.assert_called_with(obs, session_id=session.session_id)
    finally:
        session.close()


def test_connections_reuse_the_loaded_model(inference_server, mock_model):
    host, port = inference_server
    client = InferenceClient(websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.SESSION_PATH, ''))
    for index in range(2):
        session = client.new_session()
        try:
            session.infer({'episode': index})
        finally:
            session.close()
    assert mock_model.call_count == 2
    mock_model.close.assert_not_called()


def _server_connection(send: AsyncMock, receive: AsyncMock) -> websocket_wire.WebsocketServerConnection:
    return websocket_wire.WebsocketServerConnection(
        {
            'type': 'websocket',
            'asgi': {'version': '3.0', 'spec_version': '2.4'},
            'http_version': '1.1',
            'scheme': 'ws',
            'path': wire.SESSION_PATH,
            'raw_path': wire.SESSION_PATH.encode(),
            'query_string': b'',
            'root_path': '',
            'headers': [],
            'client': ('10.0.0.1', 4321),
            'server': ('localhost', 8000),
            'subprotocols': [],
        },
        receive,
        send,
        server_wire.ServedHostPort('localhost', 8000),
        close_timeout=0.05,
    )


async def _no_event() -> None:
    await asyncio.sleep(3600)


def test_a_send_after_the_session_closed_says_the_peer_is_gone():
    sent = AsyncMock()
    conn = _server_connection(sent, AsyncMock(side_effect=_no_event))
    asyncio.run(conn.refuse('Session refused'))

    with pytest.raises(wire.PeerDisconnected):
        asyncio.run(conn.send(b'a frame nobody is there to read'))
    sent.assert_awaited_once_with({'type': 'websocket.close', 'code': 1008, 'reason': 'Session refused'})


def test_a_refusal_waits_for_the_client_to_close_first():
    sent = AsyncMock()
    received = AsyncMock(
        side_effect=[{'type': 'websocket.receive', 'bytes': b'a frame'}, {'type': 'websocket.disconnect'}]
    )
    conn = _server_connection(sent, received)
    asyncio.run(conn.refuse('Session refused'))

    sent.assert_not_awaited()
    with pytest.raises(wire.PeerDisconnected):
        asyncio.run(conn.send(b'a frame nobody is there to read'))


def test_a_send_on_a_live_session_goes_out():
    sent = AsyncMock()
    conn = _server_connection(sent, AsyncMock())

    asyncio.run(conn.send(b'a frame'))

    sent.assert_awaited_once_with({'type': 'websocket.send', 'bytes': b'a frame'})


def test_a_transport_send_failure_marks_the_session_closed():
    sent = AsyncMock(side_effect=OSError('connection lost'))
    conn = _server_connection(sent, AsyncMock())
    for _ in range(2):
        with pytest.raises(wire.PeerDisconnected):
            asyncio.run(conn.send(b'a frame'))
    assert sent.await_count == 1


def test_a_disconnect_event_closes_the_session():
    conn = _server_connection(AsyncMock(), AsyncMock(return_value={'type': 'websocket.disconnect', 'code': 1000}))
    with pytest.raises(wire.PeerDisconnected):
        asyncio.run(conn.receive())
    with pytest.raises(wire.PeerDisconnected):
        asyncio.run(conn.send(b'a frame'))


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


def test_a_lower_jpeg_quality_gives_smaller_frames():
    image = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    low, high = (len(serialise({keys.WRIST_IMAGE: encode_jpeg(image, quality)})) for quality in (30, 90))
    assert low < high


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

    def test_an_impedance_frame_travels_on_the_wire(self):
        frame = Transform3D(np.array([0.0, 0.0, 0.1034]), Rotation.from_quat([0.9239, 0.0, 0.0, 0.3827]))
        impedance = Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6, frame=frame)
        assert deserialise(serialise(JointDelta(velocities=np.zeros(3), mode=impedance))).mode == impedance
        assert from_wire(to_wire(impedance)) == impedance
        assert from_wire(to_wire(impedance)) != replace(impedance, frame=Transform3D.identity)

    def test_an_impedance_wire_with_no_frame_runs_at_the_receivers_control_frame(self):
        """A sender that names no frame runs its stiffness at the receiver's control frame."""
        impedance = Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
        wire = to_wire(impedance)
        assert wire[FRAME_FIELD] == Transform3D.identity.as_vector(Rotation.Representation.ROTATION_MATRIX).tolist()
        del wire[FRAME_FIELD]
        assert from_wire(wire) == impedance

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


def _bind_frame_channel(socket_path: str) -> socket.socket:
    """The frame channel a websocket wire binds beside the Unix socket at ``socket_path``."""
    channel = wire.frame_channel_path(Path(socket_path))
    assert channel is not None
    return websocket_wire.claim_socket_path(channel, socket.SOCK_SEQPACKET)


def _frame_channel_dialer(socket_path: str) -> Callable[[float], socket.socket]:
    """How the Unix websocket wire reaches the frame channel beside ``socket_path``."""
    address = wire.UnixSocketAddress(Path(socket_path), wire.SESSION_PATH, '')
    dial = websocket.WebsocketUnixClientWire().frame_channel_dialer(address)
    assert dial is not None
    return dial


@pytest.fixture
def frame_channel(socket_path: str) -> Generator[frame_ring.FrameChannel, None, None]:
    channel = frame_ring.FrameChannel()
    channel.start(_bind_frame_channel(socket_path))
    channel.open_session(SESSION_ID)
    yield channel
    channel.close()


@pytest.fixture
def frame_writer(socket_path: str, frame_channel) -> Generator[frame_ring.FrameWriter, None, None]:
    writer = frame_ring.FrameWriter(_frame_channel_dialer(socket_path), SESSION_ID)
    yield writer
    writer.close()


_PIXELS = np.random.default_rng(0)


def _image(height: int = 48, width: int = 64) -> np.ndarray:
    return _PIXELS.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _packed(writer: frame_ring.FrameWriter, obs) -> dict:
    """``obs`` as ``writer`` packs it, with its images written into the ring."""
    packed, write = writer.pack(obs)
    if write is not None:
        write()
    return packed


def _over_the_wire(obs):
    """``obs`` as the server reads it, after a msgpack round trip."""
    return deserialise(serialise(obs))


@pytestmark_ring
def test_a_ring_carries_every_image_of_an_observation_byte_identical(frame_channel, frame_writer):
    image, stack = _image(), np.stack([_image(), _image()])
    obs = {'image.left': image, keys.GRIP: 0.5, 'nested': {'frames': [image, stack]}}

    served = frame_channel.resolve(SESSION_ID, _over_the_wire(_packed(frame_writer, obs)))

    np.testing.assert_array_equal(served['image.left'], image)
    np.testing.assert_array_equal(served['nested']['frames'][0], image)
    np.testing.assert_array_equal(served['nested']['frames'][1], stack)
    assert served[keys.GRIP] == 0.5


class _Pair(NamedTuple):
    image: np.ndarray
    grip: float


class _Frames(list):
    pass


@pytestmark_ring
def test_every_container_the_wire_carries_reaches_the_server_as_the_message_path_delivers_it(
    frame_channel, frame_writer
):
    """The wire carries a tuple, a list and any subclass of either as a plain array, and a mapping as a dict."""
    image = _image()
    obs = {
        'pair': _Pair(image, 0.5),
        'tuple': (image, 0.5),
        'frames': _Frames([image]),
        'view': MappingProxyType({'image': image}),
        keys.TASK: 'pick',
    }

    served = frame_channel.resolve(SESSION_ID, _over_the_wire(_packed(frame_writer, obs)))

    for key in ('pair', 'tuple', 'frames'):
        assert type(served[key]) is list, key
        np.testing.assert_array_equal(served[key][0], image)
    assert served['pair'][1] == served['tuple'][1] == 0.5
    np.testing.assert_array_equal(served['view']['image'], image)
    assert served[keys.TASK] == 'pick'


@pytestmark_ring
def test_a_packed_observation_leaves_the_pixels_out_of_the_message(frame_writer):
    image = _image(720, 1280)

    message = serialise(_packed(frame_writer, {'image.left': image}))

    assert len(message) < image.nbytes // 100


@pytestmark_ring
def test_an_observation_with_no_image_creates_no_ring_and_no_write(frame_writer):
    assert frame_writer.pack({keys.GRIP: 0.5}) == ({keys.GRIP: 0.5}, None)
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
def test_a_channel_that_answers_and_refuses_the_ring_still_raises(socket_path, frame_channel):
    """Only a path that reaches no channel turns the ring off. A channel that refuses the handover raises."""
    writer = frame_ring.FrameWriter(_frame_channel_dialer(socket_path), 'a-session-not-open-here')
    try:
        with pytest.raises(RuntimeError, match='did not map the frame ring'):
            _packed(writer, {'image.left': _image()})
    finally:
        writer.close()


@pytestmark_ring
def test_a_frame_view_keeps_its_pixels_for_the_calls_a_frame_stays_valid_and_no_longer(frame_channel, frame_writer):
    first = _image()
    view = frame_channel.resolve(SESSION_ID, _over_the_wire(_packed(frame_writer, {'image.left': first})))['image.left']
    for _ in range(frame_ring.CALLS_A_FRAME_STAYS_VALID - 1):
        _packed(frame_writer, {'image.left': _image()})

    np.testing.assert_array_equal(view, first)
    _packed(frame_writer, {'image.left': _image()})
    assert not np.array_equal(view, first)


@pytestmark_ring
def test_a_slot_written_over_before_it_is_read_is_refused(frame_channel, frame_writer):
    stale = _over_the_wire(_packed(frame_writer, {'image.left': _image()}))
    for _ in range(frame_ring.CALLS_A_FRAME_STAYS_VALID):
        _packed(frame_writer, {'image.left': _image()})

    with pytest.raises(frame_ring.TornFrame):
        frame_channel.resolve(SESSION_ID, stale)


@pytestmark_ring
def test_a_reference_that_points_outside_the_ring_is_refused(frame_channel, frame_writer):
    packed = _over_the_wire(_packed(frame_writer, {'image.left': _image()}))
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

    first = frame_channel.resolve(SESSION_ID, _over_the_wire(_packed(frame_writer, {'image.left': small})))
    second = frame_channel.resolve(SESSION_ID, _over_the_wire(_packed(frame_writer, {'image.left': large})))

    np.testing.assert_array_equal(second['image.left'], large)
    np.testing.assert_array_equal(first['image.left'], small)


@pytestmark_ring
def test_a_growth_whose_answer_is_lost_after_the_server_took_the_ring_leaves_the_next_frame_readable(
    frame_channel, frame_writer, monkeypatch
):
    _packed(frame_writer, {'image.left': _image(8, 8)})
    hand_over = frame_ring.FrameWriter._hand_over

    def answer_lost(writer, sock, ring):
        hand_over(writer, sock, ring)
        raise TimeoutError('the answer came after the writer stopped waiting')

    monkeypatch.setattr(frame_ring.FrameWriter, '_hand_over', answer_lost)
    with pytest.raises(TimeoutError):
        _packed(frame_writer, {'image.left': _image(64, 64)})
    monkeypatch.undo()

    small = _image(8, 8)
    served = frame_channel.resolve(SESSION_ID, _over_the_wire(_packed(frame_writer, {'image.left': small})))
    np.testing.assert_array_equal(served['image.left'], small)


@pytestmark_ring
def test_an_unsealed_descriptor_is_refused_before_it_is_mapped():
    """An unsealed ring may shrink under the mapping, and the next read of a dropped page is a SIGBUS."""
    fd = os.memfd_create('unsealed', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    os.ftruncate(fd, 1 << 16)
    try:
        with pytest.raises(ValueError, match='without the seals'):
            frame_ring.MappedRing(fd, slots=2, slot_bytes=1024)
    finally:
        os.close(fd)


@pytestmark_ring
def test_a_descriptor_sealed_beyond_what_the_server_needs_is_accepted():
    """The server needs its seals present, not alone: a peer may seal the ring against further seals."""
    ring = frame_ring.FrameRing(1024, slots=2)
    try:
        # F_SEAL_SEAL, from linux/fcntl.h: no further seal may be added.
        fcntl.fcntl(ring.fd, frame_ring._F_ADD_SEALS, 0x0001)

        assert frame_ring.MappedRing(ring.fd, slots=2, slot_bytes=1024).array is not None
    finally:
        ring.close()


@pytestmark_ring
def test_a_reference_naming_a_dtype_the_wire_refuses_is_refused(frame_channel, frame_writer):
    """An object dtype reads the ring's bytes as pointers into this process, and touching one segfaults."""
    packed = _over_the_wire(_packed(frame_writer, {'image.left': _image()}))
    packed['image.left'][frame_ring._DTYPE] = '|O'

    with pytest.raises(ValueError, match='which the wire does not carry'):
        frame_channel.resolve(SESSION_ID, packed)


@pytestmark_ring
def test_an_image_whose_dtype_the_wire_refuses_never_reaches_the_ring(frame_writer):
    """``is_image`` asks about shape alone, so the ring keeps the serializer's dtype policy itself."""
    forgeable = np.empty((4, 4, 3), dtype=object)
    forgeable[:] = 'not pixels'

    packed = _packed(frame_writer, {'image.left': forgeable})

    assert packed['image.left'] is forgeable
    assert frame_writer._ring is None
    with pytest.raises(ValueError, match='Unsupported dtype'):
        serialise(packed)


@pytestmark_ring
def test_a_float_image_travels_through_the_ring(frame_channel, frame_writer):
    """The ring carries every dtype the wire carries, not the uint8 frames alone."""
    depth = _PIXELS.random((8, 8, 3)).astype(np.float32)

    packed = _packed(frame_writer, {'image.left': depth})

    # The message path carries a float image too, so only the reference proves the ring took this one.
    assert isinstance(packed['image.left'], dict) and frame_ring._RING in packed['image.left']
    served = frame_channel.resolve(SESSION_ID, _over_the_wire(packed))
    np.testing.assert_array_equal(served['image.left'], depth)
    assert served['image.left'].dtype == np.float32


@pytestmark_ring
@pytest.mark.parametrize('refused', ['memfd_create', 'seals'])
def test_a_host_that_builds_no_ring_says_so_at_error_level(monkeypatch, caplog, refused):
    def refuse(*_args, **_kwargs):
        raise OSError(errno.EPERM, 'refused under test')

    if refused == 'memfd_create':
        monkeypatch.setattr(os, 'memfd_create', refuse)
    else:
        monkeypatch.setattr(frame_ring.fcntl, 'fcntl', refuse)

    with caplog.at_level(logging.WARNING, logger=frame_ring.logger.name):
        assert not frame_ring._ring_is_supported()

    assert [record.levelno for record in caplog.records] == [logging.ERROR]
    assert 'No frame ring' in caplog.records[0].message


@pytestmark_ring
def test_closing_a_channel_that_waits_for_a_handover_ends_its_thread(socket_path):
    channel = frame_ring.FrameChannel()
    channel.start(_bind_frame_channel(socket_path))
    accepting = channel._thread

    channel.close()

    assert accepting is not None and not accepting.is_alive()


@pytestmark_ring
def test_closing_a_channel_before_its_accept_thread_runs_raises_nothing_in_that_thread(socket_path, monkeypatch):
    released = threading.Event()
    accept_forever = frame_ring.FrameChannel._accept_forever

    def held_until_released(self, *args):
        released.wait()
        accept_forever(self, *args)

    monkeypatch.setattr(frame_ring.FrameChannel, '_accept_forever', held_until_released)
    raised: list[BaseException | None] = []
    monkeypatch.setattr(threading, 'excepthook', lambda failure: raised.append(failure.exc_value))
    channel = frame_ring.FrameChannel()
    listener = _bind_frame_channel(socket_path)
    channel.start(listener)
    accepting = channel._thread
    closing = threading.Thread(target=channel.close)
    closing.start()
    while listener.fileno() != -1:
        time.sleep(0.01)

    released.set()
    closing.join(timeout=frame_ring.HANDOVER_TIMEOUT_SEC)

    assert accepting is not None and not accepting.is_alive()
    assert raised == []
