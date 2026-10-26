"""The frame streamer: which rows it sends and when, and how a request refers to them."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import av
import numpy as np
import pytest
from positronic_model_server import serialization

from positronic import keys
from positronic.offboard import protocol as legacy_protocol
from positronic.offboard.frame_stream import H264_FRAME, REF, ROW, FrameCodec, FrameStreamer, declared_codecs
from positronic.policy.codecs import ResizeMode, TransformImages

KEY = keys.WRIST_IMAGE
# Targets fall between the 0.1 s ticks below, so each row has one frame that TemporalStack samples.
OFFSETS = (-0.25, -0.15, 0.0)


class _Clock:
    """An episode clock that only the test moves."""

    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        return self.ns


def _frame(index: int) -> np.ndarray:
    return np.full((8, 12, 3), 10 * index, dtype=np.uint8)


@contextmanager
def _streamer(**overrides: Any) -> Iterator[tuple[FrameStreamer, _Clock, list[dict[str, Any]]]]:
    """A JPEG streamer of ``OFFSETS`` at 10 actions per second, its clock, and the messages it sent."""
    clock, sent = _Clock(), []
    settings: dict[str, Any] = {'codec': FrameCodec.JPEG, 'keys': (KEY,), 'offsets_sec': OFFSETS, 'pad_start': True}
    streamer = FrameStreamer(sent.append, clock, fps=10.0, **(settings | overrides))
    try:
        yield streamer, clock, sent
    finally:
        streamer.close()


def _tick(streamer: FrameStreamer, clock: _Clock, *indices: int) -> None:
    """Send the frame of each tick index, one tick every 0.1 s."""
    for index in indices:
        clock.ns = index * 100_000_000
        streamer.on_tick({KEY: _frame(index), 'task': 'pick'})


def _rows(sent: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """``(call, back, tick index)`` of each JPEG row that went out, in send order."""
    rows = []
    for message in sent:
        row = serialization.deserialise(serialization.serialise(message))[ROW]
        rows.append((row['call'], row['back'], round(float(row['images'][KEY].mean()) / 10)))
    return rows


def _decoded(value: dict[bytes, Any]) -> np.ndarray:
    return serialization.deserialise(serialization.serialise(value))


@contextmanager
def _streamed_call_1(**overrides: Any) -> Iterator[tuple[FrameStreamer, _Clock, list[dict[str, Any]]]]:
    """A streamer that read a 5-row chunk at 1.0 s and sent the rows of the request it expects at 1.5 s."""
    with _streamer(**overrides) as (streamer, clock, sent):
        _tick(streamer, clock, *range(11))
        streamer.on_answer([{}] * 5)
        _tick(streamer, clock, 11, 12, 13, 14, 15)
        streamer.flush()
        yield streamer, clock, sent


@pytest.mark.parametrize(
    'meta, codecs',
    [
        ({}, set()),
        ({'frame_stream': 'yes'}, set()),
        ({'frame_stream': {'version': 2, 'codecs': ['jpeg']}}, set()),
        ({'frame_stream': {'version': 1, 'codecs': ['jpeg', 'vp9']}}, {FrameCodec.JPEG}),
        ({'frame_stream': {'version': 1, 'codecs': ['jpeg', 'h264']}}, {FrameCodec.JPEG, FrameCodec.H264}),
    ],
)
def test_declared_codecs_are_the_known_codecs_of_a_version_1_declaration(meta, codecs):
    assert declared_codecs(meta) == codecs


def test_a_stack_whose_last_offset_is_not_the_current_frame_is_refused():
    with pytest.raises(ValueError, match='current frame'), _streamer(offsets_sec=(-0.2, -0.1)):
        pass


def test_each_row_goes_out_with_the_frame_temporal_stack_samples_at_the_predicted_request_time():
    with _streamer() as (streamer, clock, sent):
        _tick(streamer, clock, *range(11))
        # Read at 1.0 s, 5 rows at 10 per second: the next request is due at 1.5 s, its rows at 1.25 s and 1.35 s.
        streamer.on_answer([{}] * 5)
        assert streamer.call == 1
        _tick(streamer, clock, 11, 12)
        streamer.flush()
        assert _rows(sent) == []
        _tick(streamer, clock, 13)
        streamer.flush()
        assert _rows(sent) == [(1, 2, 12)]
        _tick(streamer, clock, 14)
        streamer.flush()
        assert _rows(sent) == [(1, 2, 12), (1, 1, 13)]


def test_the_horizon_bounds_the_predicted_request_time():
    with _streamer(horizon_sec=0.3) as (streamer, clock, sent):
        _tick(streamer, clock, *range(11))
        streamer.on_answer([{}] * 5)  # due at 1.3 s, not 1.5 s
        _tick(streamer, clock, 11, 12)
        streamer.flush()
        assert _rows(sent) == [(1, 2, 10), (1, 1, 11)]


def test_a_row_message_is_a_one_key_map_that_a_server_finds_before_it_decodes_the_message():
    with _streamed_call_1() as (_, _, sent):
        message = sent[0]
        assert set(message) == {ROW}
        assert set(message[ROW]) == {'call', 'back', 't_ns', 'images'}
        assert legacy_protocol.serialise(message).startswith(bytes([0x81, 0xA0 | len(ROW)]) + ROW.encode())


def test_a_request_refers_to_the_streamed_rows_and_carries_the_current_frame():
    with _streamed_call_1() as (streamer, _, _):
        stack = np.stack([_frame(12), _frame(13), _frame(15)])
        request = streamer.request({KEY: stack, 'task': 'pick'})
        value = request[KEY]
        assert value[REF] is True
        assert (value[b'call'], value[b'rows']) == (1, [2, 1])
        np.testing.assert_allclose(_decoded(value[b'tail']), stack[2:], atol=2)
        assert request['task'] == 'pick'


def test_a_request_refers_only_to_the_rows_before_the_first_one_that_differs():
    with _streamed_call_1() as (streamer, _, _):
        stack = np.stack([_frame(12), np.full_like(_frame(0), 7), _frame(15)])
        value = streamer.request({KEY: stack})[KEY]
        assert value[b'rows'] == [2]
        np.testing.assert_allclose(_decoded(value[b'tail']), stack[1:], atol=2)


def test_a_request_of_a_new_call_does_not_refer_to_the_rows_of_the_last_call():
    with _streamed_call_1() as (streamer, _, _):
        streamer.on_answer([{}] * 5)
        value = streamer.request({KEY: np.stack([_frame(12), _frame(13), _frame(15)])})[KEY]
        assert (value[b'call'], value[b'rows']) == (2, [])


def test_rows_go_out_sized_as_the_declared_stack_sizes_its_frames():
    resize = TransformImages(6, 4, mode=ResizeMode.BOUND)
    with _streamed_call_1(resize=resize.encode) as (streamer, _, sent):
        row = serialization.deserialise(serialization.serialise(sent[0]))[ROW]
        assert row['images'][KEY].shape == (4, 6, 3)
        stack = resize.encode({KEY: np.stack([_frame(12), _frame(13), _frame(15)])})[KEY]
        assert streamer.request({KEY: stack})[KEY][b'rows'] == [2, 1]


def test_a_row_that_is_not_due_when_the_request_goes_out_never_goes_out():
    with _streamer() as (streamer, clock, sent):
        _tick(streamer, clock, *range(11))
        streamer.on_answer([{}] * 5)
        _tick(streamer, clock, 11, 12, 13)
        value = streamer.request({KEY: np.stack([_frame(12), _frame(13), _frame(13)])})[KEY]
        assert value[b'rows'] == [2]
        _tick(streamer, clock, 14, 15, 16)
        streamer.flush()
        assert _rows(sent) == [(1, 2, 12)]


def test_the_timer_sends_a_due_row_between_ticks():
    with _streamer() as (streamer, clock, sent):
        _tick(streamer, clock, *range(11))
        streamer.on_answer([{}] * 5)
        clock.ns = 1_300_000_000
        deadline = time.monotonic() + 5
        while not sent and time.monotonic() < deadline:
            time.sleep(0.01)
        streamer.flush()
        assert _rows(sent) == [(1, 2, 10)]


def test_a_failed_send_reaches_the_next_flush():
    def send(message: dict[str, Any]) -> None:
        raise ConnectionResetError('the connection dropped')

    clock = _Clock()
    streamer = FrameStreamer(
        send, clock, codec=FrameCodec.JPEG, keys=(KEY,), offsets_sec=OFFSETS, pad_start=True, fps=10
    )
    try:
        _tick(streamer, clock, *range(11))
        streamer.on_answer([{}] * 5)
        _tick(streamer, clock, 11, 12, 13)
        with pytest.raises(RuntimeError, match='frame streamer failed') as raised:
            streamer.flush()
        assert isinstance(raised.value.__cause__, ConnectionResetError)
    finally:
        streamer.close()


def _h264_frames(units: list[dict[bytes, Any]]) -> list[np.ndarray]:
    """Decode H.264 access units in send order, as the server's one decoder for a camera does."""
    decoder = av.CodecContext.create('h264', 'r')
    decoder.thread_type = 'NONE'  # one frame out for each unit in
    frames = []
    for unit in units:
        assert unit[H264_FRAME] is True
        (decoded,) = decoder.decode(av.Packet(unit[b'data']))
        h, w, _ = unit[b'shape']
        frames.append(decoded.to_ndarray(format='rgb24')[:h, :w])
    return frames


def test_h264_rows_and_the_current_frame_make_one_stream_in_send_order():
    with _streamed_call_1(codec=FrameCodec.H264) as (streamer, _, sent):
        stack = np.stack([_frame(12), _frame(13), _frame(15)])
        value = streamer.request({KEY: stack})[KEY]
        assert value[b'rows'] == [2, 1]
        units = [message[ROW]['images'][KEY] for message in sent] + [value[b'tail']]
        for decoded, expected in zip(_h264_frames(units), stack, strict=True):
            np.testing.assert_allclose(decoded, expected, atol=4)


def test_h264_carries_the_whole_stack_as_jpeg_when_a_row_differs():
    with _streamed_call_1(codec=FrameCodec.H264) as (streamer, _, _):
        stack = np.stack([_frame(12), np.full_like(_frame(0), 7), _frame(15)])
        value = streamer.request({KEY: stack})[KEY]
        assert value[b'rows'] == []
        np.testing.assert_allclose(_decoded(value[b'tail']), stack, atol=2)
