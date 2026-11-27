"""The WebRTC peer of a page: it sends a stream of frames to the page as one video track.

The page offers and the peer answers. The setup messages go over a channel that the caller owns: ``VideoPeer.signal``
takes each message of the page, and the ``send`` of the peer carries each message to the page. The page opens a data
channel, ``TIMING_CHANNEL``, on which it measures the age of each frame, and on which it reports what it measured.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, Any, Literal

import av
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCDataChannel,
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.exceptions import InvalidAccessError, InvalidStateError
from aiortc.mediastreams import VIDEO_TIME_BASE, MediaStreamError
from aiortc.sdp import candidate_from_sdp
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import pimm

logger = logging.getLogger(__name__)

# The label of the data channel that the page opens for the timing of the frames.
TIMING_CHANNEL = 'timing'


class Offer(BaseModel):
    """The offer of the page. ``id`` counts the offers of the page, so that the page knows the answer to each."""

    type: Literal['offer'] = 'offer'
    id: int
    sdp: str


class Candidate(BaseModel):
    """An ICE candidate of the page for the connection of the offer ``id``. An empty ``candidate`` ends them."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal['candidate'] = 'candidate'
    id: int
    candidate: str
    sdp_mid: str | None = Field(None, alias='sdpMid')
    sdp_mline_index: int | None = Field(None, alias='sdpMLineIndex')


class Answer(BaseModel):
    """The answer of the peer to the offer ``id``, with all the ICE candidates of the peer."""

    type: Literal['answer'] = 'answer'
    id: int
    sdp: str


_PAGE_SIGNAL: TypeAdapter[Offer | Candidate] = TypeAdapter(Annotated[Offer | Candidate, Field(discriminator='type')])


class Ping(BaseModel):
    """A ping of the page at ``ping`` ms on the clock of the page."""

    ping: float


class Pong(BaseModel):
    """The answer to the ping at ``pong``, sent at ``service_ms`` on the clock of the connection."""

    pong: float
    service_ms: float


class FrameSent(BaseModel):
    """The time of a frame that the track gives to the encoder, in ticks of the RTP clock from the connection start.

    The RTP timestamp of the frame is this time plus an offset that only the sender knows. The page finds the offset
    from the frames that it shows."""

    frame: int


class VideoReport(BaseModel):
    """What the page measured on the video track in ``seconds``: the number of frames that it showed, the age of each
    frame whose age it knows, and the gap of each stall. An age goes from the time the track gave the frame to the
    encoder to the time the page showed it. ``packets_lost``, ``nacks`` and ``plis`` count from the connection start.
    """

    seconds: float
    frames: int
    ages_ms: list[float]
    stalls_ms: list[float]
    packets_lost: int
    nacks: int
    plis: int
    rtt_ms: float | None
    codec: str | None

    def summary(self) -> str:
        if self.ages_ms:
            median, p95 = np.percentile(self.ages_ms, [50, 95])
            age = f'age {median:.0f}/{p95:.0f}/{max(self.ages_ms):.0f} ms (median/95th/max)'
        else:
            age = 'age unknown'
        stalls = f'{len(self.stalls_ms)} stalls ({sum(self.stalls_ms):.0f} ms)'
        rtt = 'RTT unknown' if self.rtt_ms is None else f'RTT {self.rtt_ms:.0f} ms'
        return (
            f'{self.frames} frames in {self.seconds:.1f} s, {age}, {stalls}; since the connect {self.packets_lost} '
            f'packets lost, {self.nacks} NACK, {self.plis} PLI; {rtt}; {self.codec or "codec unknown"}'
        )


class _Report(BaseModel):
    report: VideoReport


_PAGE_TIMING: TypeAdapter[Ping | _Report] = TypeAdapter(Ping | _Report)


class _ConnectionClock:
    """The time from the start of one peer connection: in ms for the page, and in ticks of the RTP clock of video."""

    def __init__(self) -> None:
        self._start_ns = time.monotonic_ns()

    def ms(self) -> float:
        return (time.monotonic_ns() - self._start_ns) / 1e6

    def ticks(self) -> int:
        return (time.monotonic_ns() - self._start_ns) * VIDEO_TIME_BASE.denominator // 1_000_000_000


class FrameTrack(MediaStreamTrack):
    """The video track of the RGB frames that ``next_frame`` gives, each one time. ``next_frame`` takes the time of the
    last frame, and gives ``None`` until a newer frame comes. ``on_frame`` takes the pts of each frame of the track."""

    kind = 'video'

    def __init__(
        self,
        next_frame: Callable[[pimm.Time | None], pimm.Message | None],
        clock: _ConnectionClock,
        on_frame: Callable[[int], None],
        poll_sec: float,
    ):
        super().__init__()
        self._next_frame = next_frame
        self._clock = clock
        self._on_frame = on_frame
        self._poll_sec = poll_sec
        self._last_time: pimm.Time | None = None

    async def recv(self) -> av.VideoFrame:
        while (message := self._next_frame(self._last_time)) is None:
            if self.readyState != 'live':
                raise MediaStreamError
            await asyncio.sleep(self._poll_sec)
        self._last_time = message.time
        rgb = message.data
        # The encoder takes the frame as YUV 4:2:0, which needs an even width and height
        h, w = rgb.shape[:2]
        even = np.pad(rgb, ((0, h % 2), (0, w % 2), (0, 0)), mode='edge')
        frame = av.VideoFrame.from_ndarray(even, format='rgb24')
        frame.pts = self._clock.ticks()
        frame.time_base = VIDEO_TIME_BASE
        self._on_frame(frame.pts)
        return frame


class _Timing:
    """The service end of the ``TIMING_CHANNEL`` of one connection: it answers each ping, sends the time of each frame,
    and logs each report of the page."""

    def __init__(self, clock: _ConnectionClock) -> None:
        self._clock = clock
        self._channel: RTCDataChannel | None = None

    def attach(self, channel: RTCDataChannel) -> None:
        if channel.label != TIMING_CHANNEL:
            logger.warning('The page opened a data channel %r, which the video peer does not use', channel.label)
            return
        self._channel = channel
        channel.on('message', self._take)

    def _send(self, message: BaseModel) -> None:
        if self._channel is not None and self._channel.readyState == 'open':
            self._channel.send(message.model_dump_json())

    def frame_sent(self, pts: int) -> None:
        self._send(FrameSent(frame=pts))

    def _take(self, data: str | bytes) -> None:
        try:
            message = _PAGE_TIMING.validate_json(data)
        except ValidationError as error:
            # The timing only measures the video, so the video goes on without it.
            logger.error('The video peer cannot use a timing message of the page: %s', error)
            return
        if isinstance(message, Ping):
            self._send(Pong(pong=message.ping, service_ms=self._clock.ms()))
        else:
            logger.info('Video over WebRTC: %s', message.report.summary())


class VideoPeer:
    """The WebRTC peer of one page. It answers each offer of the page with a new peer connection, which sends the frames
    of ``next_frame`` (as ``FrameTrack`` takes them) as one video track. ``send`` carries a setup message to the page.

    The peer has no ICE servers, so it offers only the addresses of this computer: the page must reach one of them."""

    # How often a track looks for a new frame.
    POLL_SEC = 0.002

    def __init__(
        self,
        next_frame: Callable[[pimm.Time | None], pimm.Message | None],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ):
        self._next_frame = next_frame
        self._send = send
        self._connection: RTCPeerConnection | None = None
        self._offer_id: int | None = None
        self._signals: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        self._task = asyncio.ensure_future(self._take_signals())

    def signal(self, message: Mapping[str, Any]) -> None:
        """Take a setup message of the page. The peer acts on each message after the ones that came before it."""
        self._signals.put_nowait(message)

    async def close(self) -> None:
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        await self._close_connection()

    async def _take_signals(self) -> None:
        while True:
            message = await self._signals.get()
            try:
                signal = _PAGE_SIGNAL.validate_python(message)
                if isinstance(signal, Offer):
                    await self._answer(signal)
                elif signal.id == self._offer_id and self._connection is not None:
                    await self._connection.addIceCandidate(self._ice_candidate(signal))
            except (ValueError, InvalidStateError, InvalidAccessError) as error:
                # The next offer of the page starts a new connection, so one bad message does not end the video.
                logger.error('The video peer cannot use a setup message of the page: %s', error)

    @staticmethod
    def _ice_candidate(signal: Candidate) -> RTCIceCandidate | None:
        if not signal.candidate:
            return None
        candidate = candidate_from_sdp(signal.candidate.removeprefix('candidate:'))
        candidate.sdpMid = signal.sdp_mid
        candidate.sdpMLineIndex = signal.sdp_mline_index
        return candidate

    async def _answer(self, offer: Offer) -> None:
        await self._close_connection()
        connection = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self._connection, self._offer_id = connection, offer.id
        clock = _ConnectionClock()
        timing = _Timing(clock)
        connection.on('datachannel', timing.attach)

        @connection.on('connectionstatechange')
        def log_state() -> None:
            logger.info('The video connection %d of the page is %s', offer.id, connection.connectionState)

        await connection.setRemoteDescription(RTCSessionDescription(offer.sdp, 'offer'))
        connection.addTrack(FrameTrack(self._next_frame, clock, timing.frame_sent, self.POLL_SEC))
        await connection.setLocalDescription(await connection.createAnswer())
        await self._send(Answer(id=offer.id, sdp=connection.localDescription.sdp).model_dump())

    async def _close_connection(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
