"""A shared-memory ring that carries an observation's frames to a server on the same host.

The client creates the ring, seals it, and hands its descriptor over an ``AF_UNIX`` socket beside the
session socket. Each inference writes the images into a slot and sends a reference for each, which
the server reads through a mapping the seals let it neither write nor resize.
"""

import fcntl
import logging
import mmap
import os
import socket
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from positronic.utils import serialization

logger = logging.getLogger(__name__)

# A ring needs ``memfd_create``, which Linux has and macOS does not. A server without it declares no
# ring, and every image stays in the message.
SUPPORTED = hasattr(os, 'memfd_create')

# The suffix of the descriptor socket, which each side derives from its own session socket path.
SOCKET_SUFFIX = '.frames'

# Slots per ring, so the writer never takes a slot the server may still read.
SLOTS = 4

# The slot header: two sequence numbers, then padding that puts every payload on a 64-byte boundary.
_HEADER_BYTES = 64
_ALIGN = 64

# The envelope one image travels in when the ring carries its bytes: the slot and the sequence number
# that say which write it belongs to, and where the array sits inside that slot.
_RING = b'__ring__'
_SLOT = b'slot'
_SEQ = b'seq'
_OFFSET = b'offset'
_SHAPE = b'shape'
_DTYPE = b'dtype'

# The fields of a handover, which the client sends with the descriptor.
_SESSION = 'session'
_SLOTS = 'slots'
_SLOT_BYTES = 'slot_bytes'

_ACK = b'\x01'
_HANDOVER_BYTES = 4096


class TornFrame(RuntimeError):
    """The slot a reference names holds another write, so the pixels under it belong elsewhere."""


def channel_path(socket_path: str) -> str:
    """The descriptor socket that belongs to the session socket at ``socket_path``."""
    return socket_path + SOCKET_SUFFIX


def _aligned(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


# The seal numbers from ``linux/fcntl.h``: a Python built against other headers exports none of them.
# ``F_SEAL_FUTURE_WRITE`` (Linux 5.1) spares the writer's own mapping, which ``F_SEAL_WRITE`` cannot.
_F_ADD_SEALS = 1033
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_FUTURE_WRITE = 0x0010
_SEALS = _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_FUTURE_WRITE


class FrameRing:
    """One sealed ring of ``slots`` slots, each holding ``slot_bytes`` of image data.

    The constructor maps it writable before it seals it, which is the only order the seals allow.
    """

    def __init__(self, slot_bytes: int, slots: int = SLOTS):
        self.slots = slots
        self.slot_bytes = slot_bytes
        self._stride = _HEADER_BYTES + _aligned(slot_bytes)
        self.fd = os.memfd_create('positronic-frames', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        os.ftruncate(self.fd, self._stride * slots)
        self._map = mmap.mmap(self.fd, self._stride * slots, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        fcntl.fcntl(self.fd, _F_ADD_SEALS, _SEALS)
        self._seq = 0

    def write(self, arrays: Sequence[np.ndarray]) -> list[dict[bytes, Any]]:
        """Copy ``arrays`` into the next slot and return one reference each."""
        self._seq += 1
        slot = self._seq % self.slots
        counters = np.ndarray(2, dtype=np.uint64, buffer=self._map, offset=slot * self._stride)
        counters[0] = self._seq
        payload = slot * self._stride + _HEADER_BYTES
        offset = 0
        references = []
        for array in arrays:
            destination = np.ndarray(array.shape, dtype=array.dtype, buffer=self._map, offset=payload + offset)
            np.copyto(destination, array)
            references.append({
                _RING: True,
                _SLOT: slot,
                _SEQ: self._seq,
                _OFFSET: offset,
                _SHAPE: list(array.shape),
                _DTYPE: array.dtype.str,
            })
            offset += _aligned(array.nbytes)
        counters[1] = self._seq
        return references

    def close(self) -> None:
        self._map.close()
        os.close(self.fd)


def _detach_images(value: Any, found: list[tuple[dict[bytes, Any], np.ndarray]]) -> Any:
    """``value`` with an empty reference in place of every image, each paired with its array in ``found``.

    The caller fills each reference in once the ring says where its image landed.
    """
    if serialization.is_image(value):
        reference: dict[bytes, Any] = {}
        found.append((reference, value))
        return reference
    if isinstance(value, Mapping):
        return {key: _detach_images(item, found) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_detach_images(item, found) for item in value)
    return value


# How long a handover waits for the server to map the ring.
HANDOVER_TIMEOUT_SEC = 10.0


class FrameWriter:
    """The client's half: one ring per session, handed to the server and grown when a frame outgrows it.

    ``pack`` returns the observation with a reference in place of every image.
    """

    def __init__(self, channel: str, session_id: str):
        self._channel = channel
        self._session_id = session_id
        self._ring: FrameRing | None = None

    def pack(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        found: list[tuple[dict[bytes, Any], np.ndarray]] = []
        packed = _detach_images(obs, found)
        if not found:
            return packed
        arrays = [array for _reference, array in found]
        ring = self._ring_for(sum(_aligned(array.nbytes) for array in arrays))
        for (reference, _array), written in zip(found, ring.write(arrays), strict=True):
            reference.update(written)
        return packed

    def _ring_for(self, slot_bytes: int) -> FrameRing:
        if self._ring is not None and self._ring.slot_bytes >= slot_bytes:
            return self._ring
        ring = FrameRing(slot_bytes)
        try:
            self._hand_over(ring)
        except Exception:
            ring.close()
            raise
        if self._ring is not None:
            # A mapping outlives the descriptor it was made from, so this frees the ring only here.
            self._ring.close()
        self._ring = ring
        return ring

    def _hand_over(self, ring: FrameRing) -> None:
        header = serialization.serialise({_SESSION: self._session_id, _SLOTS: ring.slots, _SLOT_BYTES: ring.slot_bytes})
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
            sock.settimeout(HANDOVER_TIMEOUT_SEC)
            sock.connect(self._channel)
            socket.send_fds(sock, [header], [ring.fd])
            if sock.recv(len(_ACK)) != _ACK:
                raise RuntimeError(f'The server at {self._channel} did not map the frame ring it was handed')

    def close(self) -> None:
        if self._ring is not None:
            self._ring.close()
            self._ring = None


class MappedRing:
    """The server's half: the ring mapped read-only, and the array a reference names.

    Nothing unmaps the ring. Every view holds a reference to the mapping, so it lives until the last
    view is gone; ``mmap.close`` under a live view leaves that view pointing at memory the process no
    longer owns.
    """

    def __init__(self, fd: int, slots: int, slot_bytes: int):
        self._slots = slots
        self._slot_bytes = slot_bytes
        self._stride = _HEADER_BYTES + _aligned(slot_bytes)
        self._map = mmap.mmap(fd, self._stride * slots, mmap.MAP_SHARED, mmap.PROT_READ)

    def array(self, reference: Mapping[bytes, Any]) -> np.ndarray:
        """A read-only view of the image ``reference`` names, over the ring's own pages."""
        slot, seq, offset = reference[_SLOT], reference[_SEQ], reference[_OFFSET]
        dtype = np.dtype(reference[_DTYPE])
        shape = tuple(reference[_SHAPE])
        nbytes = dtype.itemsize * int(np.prod(shape))
        if not 0 <= slot < self._slots or offset < 0 or offset + nbytes > self._slot_bytes:
            raise ValueError(f'A frame reference names slot {slot} at {offset}+{nbytes}, outside the ring')
        counters = np.ndarray(2, dtype=np.uint64, buffer=self._map, offset=slot * self._stride)
        if counters[0] != seq or counters[1] != seq:
            raise TornFrame(f'Slot {slot} holds write {counters[0]}..{counters[1]}, and the reference names {seq}')
        return np.ndarray(shape, dtype=dtype, buffer=self._map, offset=slot * self._stride + _HEADER_BYTES + offset)


class FrameChannel:
    """The socket that carries ring descriptors to this server, and the rings each session holds.

    A thread accepts each handover, maps the ring read-only and answers, under the session id the
    server put in its ready handshake. The caller claims ``path`` and hands the socket to ``start``.
    """

    def __init__(self, path: str):
        self.path = path
        self._rings: dict[str, list[MappedRing]] = {}
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._closing = False

    def start(self, sock: socket.socket) -> None:
        """Serve handovers on ``sock``, which the caller already bound to ``path`` and listened on."""
        self._socket = sock
        self._thread = threading.Thread(target=self._accept_forever, name='frame-channel', daemon=True)
        self._thread.start()
        logger.info('Frame ring channel listening on %s', self.path)

    def open_session(self, session_id: str) -> None:
        with self._lock:
            self._rings[session_id] = []

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._rings.pop(session_id, None)

    def resolve(self, session_id: str, obs: Any) -> Any:
        """``obs`` with a read-only view in place of every frame reference it carries."""
        if isinstance(obs, Mapping):
            if _RING in obs:
                return self._ring(session_id).array(obs)
            return {key: self.resolve(session_id, item) for key, item in obs.items()}
        if isinstance(obs, list | tuple):
            return type(obs)(self.resolve(session_id, item) for item in obs)
        return obs

    def _ring(self, session_id: str) -> MappedRing:
        with self._lock:
            rings = self._rings.get(session_id, [])
        if not rings:
            raise RuntimeError(f'Session {session_id} sent a frame reference before it handed over a ring')
        return rings[-1]

    def _accept_forever(self) -> None:
        assert self._socket is not None
        while not self._closing:
            try:
                connection, _address = self._socket.accept()
            except OSError:
                if not self._closing:
                    logger.exception('The frame ring channel stopped accepting')
                return
            with connection:
                try:
                    self._take_ring(connection)
                # One bad handover must not take the channel down.
                except Exception:
                    logger.exception('A frame ring handover failed')

    def _take_ring(self, connection: socket.socket) -> None:
        message, fds, _flags, _address = socket.recv_fds(connection, _HANDOVER_BYTES, 1)
        if not fds:
            raise RuntimeError('A frame ring handover carried no descriptor')
        try:
            header = serialization.deserialise(message)
            ring = MappedRing(fds[0], header[_SLOTS], header[_SLOT_BYTES])
        finally:
            os.close(fds[0])
        session_id = header[_SESSION]
        with self._lock:
            rings = self._rings.get(session_id)
            if rings is not None:
                rings.append(ring)
        if rings is None:
            raise RuntimeError(f'A frame ring arrived for session {session_id}, which is not open here')
        connection.send(_ACK)

    def close(self) -> None:
        self._closing = True
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            self._rings.clear()
