"""A shared-memory ring that carries an observation's frames to a server on the same host.

The client seals the ring and hands its descriptor over the wire's frame channel, and each inference sends
a reference in place of each image. The server reads a frame through a mapping, not a copy, so a frame stays
valid for ``CALLS_A_FRAME_STAYS_VALID`` calls: a server that keeps it longer reads other pixels, with no error.
"""

import fcntl
import logging
import mmap
import os
import socket
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from positronic_model_server import serialization

logger = logging.getLogger(__name__)

# The ready handshake's metadata key, true when the server takes this session's images through a ring.
HANDSHAKE_KEY = 'frame_ring'

# A frame stays valid for the call that delivers it and the calls after it, up to this count. The ring has one
# slot per call, so the call after the last of them writes the frame's slot again.
CALLS_A_FRAME_STAYS_VALID = 4

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


class TornFrame(RuntimeError):
    """The slot a reference names holds another write, so the pixels under it belong elsewhere."""


def _aligned(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


# The seal numbers from ``linux/fcntl.h``: a Python built against other headers exports none of them.
# ``F_SEAL_FUTURE_WRITE`` (Linux 5.1) spares the writer's own mapping, which ``F_SEAL_WRITE`` cannot.
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_FUTURE_WRITE = 0x0010
_SEALS = _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_FUTURE_WRITE


def _ring_is_supported() -> bool:
    """Whether this host builds a sealed ring: macOS has no ``memfd_create``, Linux before 5.1 no seal."""
    if not hasattr(os, 'memfd_create'):
        return False
    try:
        fd = os.memfd_create('positronic-frames-probe', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    except OSError as refused:
        logger.error('No frame ring: this host refuses memfd_create (%s). Images stay in the message.', refused)
        return False
    try:
        fcntl.fcntl(fd, _F_ADD_SEALS, _SEALS)
    except OSError as refused:
        logger.error('No frame ring: this host refuses the memfd seals (%s). Images stay in the message.', refused)
        return False
    finally:
        os.close(fd)
    return True


SUPPORTED = _ring_is_supported()


class FrameRing:
    """One sealed ring of ``slots`` slots, each holding ``slot_bytes`` of image data.

    The constructor maps it writable before it seals it, which is the only order the seals allow.
    """

    def __init__(self, slot_bytes: int, slots: int = CALLS_A_FRAME_STAYS_VALID):
        self.slots = slots
        self.slot_bytes = slot_bytes
        self._stride = _HEADER_BYTES + _aligned(slot_bytes)
        self.fd = os.memfd_create('positronic-frames', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            os.ftruncate(self.fd, self._stride * slots)
            self._map = mmap.mmap(self.fd, self._stride * slots, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            fcntl.fcntl(self.fd, _F_ADD_SEALS, _SEALS)
        except BaseException:
            os.close(self.fd)
            raise
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


def _as_wire_sequence(sequence: list | tuple, items: Iterable[Any]) -> list | tuple:
    """``items`` as a plain list or tuple: the wire carries any subclass of either as a plain array."""
    return list(items) if isinstance(sequence, list) else tuple(items)


def _detach_images(value: Any, found: list[tuple[dict[bytes, Any], np.ndarray]]) -> Any:
    """``value`` with an empty reference in place of every image, each paired with its array in ``found``."""
    if serialization.is_image(value) and not serialization.is_unsupported_dtype(value.dtype):
        reference: dict[bytes, Any] = {}
        found.append((reference, value))
        return reference
    if isinstance(value, Mapping):
        return {key: _detach_images(item, found) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return _as_wire_sequence(value, (_detach_images(item, found) for item in value))
    return value


# How long a handover waits for the server to map the ring.
HANDOVER_TIMEOUT_SEC = 10.0


class FrameWriter:
    """The client's half: one ring per session, handed to the server and grown when a frame outgrows it.

    ``pack`` returns the observation with a reference in place of every image. ``dial_channel`` is the
    wire's ``frame_channel_dialer``. Where no channel answers, the session takes no ring, and ``pack``
    returns every image whole from then on.
    """

    def __init__(self, dial_channel: Callable[[float], socket.socket], session_id: str):
        self._dial_channel = dial_channel
        self._session_id = session_id
        self._ring: FrameRing | None = None
        self._channel_answers = True

    @property
    def channel_answers(self) -> bool:
        """False once a dial found no frame channel, after which every image stays in the message."""
        return self._channel_answers

    def hand_over_empty_ring(self) -> bool:
        """Hand the server a ring with empty slots, and return whether the channel answered.

        A session calls this when it opens, so it knows from its first call whether its images take the ring.
        The first image grows the ring to its size.
        """
        return self._ring_for(0) is not None

    def pack(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        if not self._channel_answers:
            return dict(obs)
        found: list[tuple[dict[bytes, Any], np.ndarray]] = []
        packed = _detach_images(obs, found)
        if not found:
            return packed
        arrays = [array for _reference, array in found]
        ring = self._ring_for(sum(_aligned(array.nbytes) for array in arrays))
        if ring is None:
            return dict(obs)
        for (reference, _array), written in zip(found, ring.write(arrays), strict=True):
            reference.update(written)
        return packed

    def _ring_for(self, slot_bytes: int) -> FrameRing | None:
        """A ring with slots of at least ``slot_bytes`` the server has mapped, or None where no channel answers."""
        if self._ring is not None and self._ring.slot_bytes >= slot_bytes:
            return self._ring
        sock = self._dial()
        if sock is None:
            return None
        ring = FrameRing(slot_bytes)
        try:
            with sock:
                self._hand_over(sock, ring)
        except Exception:
            ring.close()
            raise
        if self._ring is not None:
            # A mapping outlives the descriptor it was made from, so this frees the ring only here.
            self._ring.close()
        self._ring = ring
        return ring

    def _dial(self) -> socket.socket | None:
        """A socket connected to the channel, or None where no channel answers.

        A hard link to the session socket, or a bind mount of the socket file alone, reaches no channel.
        """
        try:
            return self._dial_channel(HANDOVER_TIMEOUT_SEC)
        # The ring is optional: the session sends whole images, as on a server that declares none.
        except OSError as unreachable:
            self._channel_answers = False
            logger.error('No frame ring: %s', unreachable)
            return None

    def _hand_over(self, sock: socket.socket, ring: FrameRing) -> None:
        header = serialization.serialise({_SESSION: self._session_id, _SLOTS: ring.slots, _SLOT_BYTES: ring.slot_bytes})
        socket.send_fds(sock, [header], [ring.fd])
        if sock.recv(len(_ACK)) != _ACK:
            raise RuntimeError('The server did not map the frame ring it was handed')

    def close(self) -> None:
        if self._ring is not None:
            self._ring.close()
            self._ring = None


def _is_sealed(fd: int) -> bool:
    """Whether ``fd`` carries every seal the server's mapping needs, and is therefore a memfd."""
    try:
        seals = fcntl.fcntl(fd, _F_GET_SEALS)
    except OSError:
        # An ordinary file answers EINVAL, and a mapping over one shrinks under the reader.
        return False
    return (seals & _SEALS) == _SEALS


class MappedRing:
    """The server's half: the ring mapped read-only, and the array a reference names.

    Nothing unmaps the ring. Every view holds a reference to the mapping, so it lives until the last
    view is gone; ``mmap.close`` under a live view leaves that view pointing at memory the process no
    longer owns.
    """

    def __init__(self, fd: int, slots: int, slot_bytes: int):
        # Without the seals the peer can shrink the ring under this mapping. A read of a dropped page
        # then raises SIGBUS, which kills this process.
        if not _is_sealed(fd):
            raise ValueError('A frame ring descriptor arrived without the seals that hold its pages in place')
        self._slots = slots
        self._slot_bytes = slot_bytes
        self._stride = _HEADER_BYTES + _aligned(slot_bytes)
        self._map = mmap.mmap(fd, self._stride * slots, mmap.MAP_SHARED, mmap.PROT_READ)

    def array(self, reference: Mapping[bytes, Any]) -> np.ndarray:
        """A read-only view of the image ``reference`` names, over the ring's own pages.

        The view is valid for ``CALLS_A_FRAME_STAYS_VALID`` calls. After that the client writes the slot
        again, and the view shows the new frame with no error. Copy the array to keep it longer.
        """
        slot, seq, offset = reference[_SLOT], reference[_SEQ], reference[_OFFSET]
        dtype = np.dtype(reference[_DTYPE])
        # The peer names the dtype, and an object dtype reads the ring's bytes as pointers into this
        # process. A read of one kills the process.
        if serialization.is_unsupported_dtype(dtype):
            raise ValueError(f'A frame reference names dtype {dtype}, which the wire does not carry')
        shape = tuple(reference[_SHAPE])
        nbytes = dtype.itemsize * int(np.prod(shape))
        if not 0 <= slot < self._slots or offset < 0 or offset + nbytes > self._slot_bytes:
            raise ValueError(f'A frame reference names slot {slot} at {offset}+{nbytes}, outside the ring')
        counters = np.ndarray(2, dtype=np.uint64, buffer=self._map, offset=slot * self._stride)
        if counters[0] != seq or counters[1] != seq:
            raise TornFrame(f'Slot {slot} holds write {counters[0]}..{counters[1]}, and the reference names {seq}')
        return np.ndarray(shape, dtype=dtype, buffer=self._map, offset=slot * self._stride + _HEADER_BYTES + offset)


# How long ``accept`` blocks before the thread re-reads whether the channel is closing.
_ACCEPT_POLL_SEC = 0.2
# The largest handover header the channel reads.
_HANDOVER_BYTES = 4096


class FrameChannel:
    """The socket that carries ring descriptors to this server, and the rings each session holds.

    A thread accepts each handover, maps the ring read-only and answers, under the session id the
    server put in its ready handshake. The caller binds the socket, with the wire's ``bind_frame_channel``.
    """

    def __init__(self):
        self._rings: dict[str, MappedRing | None] = {}
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._closing = False

    def start(self, sock: socket.socket) -> None:
        """Serve handovers on ``sock``, which the caller already bound and listened on."""
        sock.settimeout(_ACCEPT_POLL_SEC)
        self._socket = sock
        self._thread = threading.Thread(target=self._accept_forever, args=(sock,), name='frame-channel', daemon=True)
        self._thread.start()
        logger.info('Frame ring channel listening on %s', sock.getsockname())

    def open_session(self, session_id: str) -> None:
        with self._lock:
            self._rings[session_id] = None

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._rings.pop(session_id, None)

    def resolve(self, session_id: str, obs: Any) -> Any:
        """``obs`` with a read-only view in place of every frame reference it carries.

        Each view is valid for ``CALLS_A_FRAME_STAYS_VALID`` calls, the call that delivers it included. A
        server that keeps a frame longer reads the pixels of a later call, with no error, so it copies the frame.
        """
        if isinstance(obs, Mapping):
            if _RING in obs:
                return self._ring(session_id).array(obs)
            return {key: self.resolve(session_id, item) for key, item in obs.items()}
        if isinstance(obs, list | tuple):
            return _as_wire_sequence(obs, (self.resolve(session_id, item) for item in obs))
        return obs

    def _ring(self, session_id: str) -> MappedRing:
        with self._lock:
            ring = self._rings.get(session_id)
        if ring is None:
            raise RuntimeError(f'Session {session_id} sent a frame reference before it handed over a ring')
        return ring

    def _accept_forever(self, sock: socket.socket) -> None:
        """Take handovers on ``sock`` until the channel closes."""
        while not self._closing:
            try:
                connection, _address = sock.accept()
            except TimeoutError:
                continue
            except OSError:
                if not self._closing:
                    logger.exception('The frame ring channel stopped accepting')
                return
            with connection:
                if self._closing:
                    return
                try:
                    self._take_ring(connection)
                # One bad handover must not take the channel down.
                except Exception:
                    logger.exception('A frame ring handover failed')

    def _take_ring(self, connection: socket.socket) -> None:
        connection.settimeout(HANDOVER_TIMEOUT_SEC)
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
            # A ring this replaces stays mapped while any view over it lives, so dropping it here frees
            # only what nobody reads.
            open_here = session_id in self._rings
            if open_here:
                self._rings[session_id] = ring
        if not open_here:
            raise RuntimeError(f'A frame ring arrived for session {session_id}, which is not open here')
        connection.send(_ACK)

    def close(self) -> None:
        self._closing = True
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._thread is not None:
            # A handover under way holds the thread for its own deadline, so the join waits that long.
            self._thread.join(timeout=HANDOVER_TIMEOUT_SEC)
            self._thread = None
        with self._lock:
            self._rings.clear()
