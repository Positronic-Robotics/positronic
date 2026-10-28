"""The frame streamer: which rows it sends and when, and how a request refers to them."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import av
import numpy as np
import pytest
from positronic_model_server import serialization
from positronic_model_server.serialization import encode_jpeg

from positronic import keys
from positronic.offboard import protocol as legacy_protocol
from positronic.offboard.frame_stream import (
    H264_FRAME,
    REF,
    ROW,
    FrameCodec,
    FrameStreamer,
    RowsMissing,
    StackRows,
    StreamedRows,
    _VideoEncoder,
    declared_codecs,
)
from positronic.policy.codecs import ResizeMode, TransformImages
from positronic.policy.processors import ConstantDuration, DelayEstimate

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


def test_the_inference_lead_brings_the_next_request_forward_and_a_passed_row_goes_at_once():
    with _streamer(lead=ConstantDuration(0.3).run().update) as (streamer, clock, sent):
        _tick(streamer, clock, *range(11))
        streamer.on_answer([{}] * 5)  # read at 1.0 s: the chunk ends at 1.5 s, the next request goes at 1.2 s
        streamer.flush()
        assert _rows(sent) == [(1, 2, 9)]  # its target, 0.95 s, has passed
        _tick(streamer, clock, 11)
        streamer.flush()
        assert _rows(sent) == [(1, 2, 9), (1, 1, 10)]


def test_an_answer_read_before_the_chunk_ends_starts_its_chunk_at_that_end():
    with _streamer(lead=ConstantDuration(0.3).run().update) as (streamer, clock, sent):
        _tick(streamer, clock, *range(11))
        streamer.on_answer([{}] * 5)  # read at 1.0 s: the chunk runs to 1.5 s
        _tick(streamer, clock, 11, 12, 13)
        streamer.on_answer([{}] * 5)  # read at 1.3 s: the chunk runs from 1.5 s to 2.0 s, the next request at 1.7 s
        _tick(streamer, clock, *range(14, 18))
        streamer.flush()
        assert _rows(sent)[2:] == [(2, 2, 14), (2, 1, 15)]


def test_the_lead_estimate_follows_the_delay_from_the_submit_to_the_read():
    with _streamer(lead=DelayEstimate.max(last=5, max_sec=1.0).run().update) as (streamer, clock, sent):
        _tick(streamer, clock, *range(8))
        streamer.on_submit()  # at 0.7 s
        _tick(streamer, clock, 8, 9, 10)
        streamer.on_answer([{}] * 5)  # read at 1.0 s, 0.3 s after its submit: the next request goes at 1.2 s
        _tick(streamer, clock, 11)
        streamer.flush()
        assert _rows(sent) == [(1, 2, 9), (1, 1, 10)]


def test_a_replan_starts_the_next_chunk_when_its_answer_is_read():
    with _streamer(lead=ConstantDuration(0.3).run().update) as (streamer, clock, sent):
        _tick(streamer, clock, *range(11))
        streamer.on_answer([{}] * 5)  # read at 1.0 s: the chunk runs to 1.5 s
        _tick(streamer, clock, 11, 12)
        streamer.request({KEY: np.stack([_frame(9), _frame(10), _frame(12)])})  # the call at 1.2 s
        streamer.on_replan(1_250_000_000)  # an arm error at 1.25 s clears the plan
        _tick(streamer, clock, 13)
        # Read at 1.3 s: the chunk runs from 1.3 s, not from 1.5 s, and the next request goes at 1.5 s.
        streamer.on_answer([{}] * 5)
        _tick(streamer, clock, 14)
        streamer.flush()
        assert _rows(sent)[2:] == [(2, 2, 12), (2, 1, 13)]


def test_an_answer_whose_next_request_is_due_at_once_plans_no_rows():
    with _streamer() as (streamer, clock, sent):
        _tick(streamer, clock, *range(11))
        streamer.on_answer([])
        _tick(streamer, clock, 11, 12)
        streamer.flush()
        assert sent == []
        assert streamer.request({KEY: np.stack([_frame(9), _frame(10), _frame(12)])})[KEY][b'rows'] == []


def _wire(message: Any) -> Any:
    """``message`` as the server reads it: serialised, then deserialised with its JPEG frames decoded."""
    return legacy_protocol.deserialise(legacy_protocol.serialise(message))


def _row(call: int, back: int, image: Any) -> dict[str, Any]:
    return _wire({ROW: {'call': call, 'back': back, 't_ns': 0, 'images': {KEY: image}}})[ROW]


def _gradient(index: int) -> np.ndarray:
    """A smooth 16 x 24 frame that H.264 keeps close."""
    ramp = np.add.outer(np.arange(16), np.arange(24)).astype(np.uint8) * 4
    return np.repeat(ramp[..., None], 3, axis=2) + np.uint8(10 * index)


def test_a_server_builds_a_stack_from_the_streamed_rows_and_the_tail():
    rows = StreamedRows()
    frames = [_gradient(i) for i in range(3)]
    for back, frame in ((2, frames[0]), (1, frames[1])):
        rows.add(_row(1, back, encode_jpeg(frame)))
    request = _wire({KEY: {REF: True, b'call': 1, b'rows': [2, 1], b'tail': encode_jpeg(frames[2][None])}, 'task': 'x'})
    resolved = rows.resolve(request)
    full = _wire({KEY: encode_jpeg(np.stack(frames)), 'task': 'x'})
    assert resolved.keys() == full.keys() and resolved['task'] == 'x'
    np.testing.assert_array_equal(resolved[KEY], full[KEY])


def test_a_server_answers_the_rows_it_does_not_have():
    rows = StreamedRows()
    rows.add(_row(1, 2, encode_jpeg(_gradient(0))))
    request = _wire({KEY: {REF: True, b'call': 1, b'rows': [2, 1], b'tail': encode_jpeg(_gradient(2)[None])}})
    with pytest.raises(RowsMissing) as raised:
        rows.resolve(request)
    assert raised.value.missing == [[KEY, 1]]


def test_a_row_of_a_newer_call_drops_the_rows_of_the_older_ones():
    rows = StreamedRows()
    rows.add(_row(1, 2, encode_jpeg(_gradient(0))))
    rows.add(_row(2, 1, encode_jpeg(_gradient(1))))
    rows.add(_row(1, 1, encode_jpeg(_gradient(1))))  # a row of an answered call
    tail = encode_jpeg(_gradient(2)[None])
    with pytest.raises(RowsMissing) as raised:
        rows.resolve(_wire({KEY: {REF: True, b'call': 1, b'rows': [2], b'tail': tail}}))
    assert raised.value.missing == [[KEY, 2]]
    assert rows.resolve(_wire({KEY: {REF: True, b'call': 2, b'rows': [1], b'tail': tail}}))[KEY].shape == (2, 16, 24, 3)


def test_a_server_decodes_each_h264_unit_in_order_also_the_tail_of_a_request_it_misses():
    encoder = _VideoEncoder(24, 16, crf=18, fps=10.0)
    frames = [_gradient(i) for i in range(6)]
    rows = StreamedRows()
    rows.add(_row(1, 2, encoder.encode(frames[0])))
    rows.add(_row(1, 1, encoder.encode(frames[1])))
    missed = _wire({KEY: {REF: True, b'call': 1, b'rows': [2, 1, 3], b'tail': encoder.encode(frames[2])}})
    with pytest.raises(RowsMissing):
        rows.resolve(missed)
    rows.add(_row(2, 2, encoder.encode(frames[3])))
    rows.add(_row(2, 1, encoder.encode(frames[4])))
    request = _wire({KEY: {REF: True, b'call': 2, b'rows': [2, 1], b'tail': encoder.encode(frames[5])}})
    np.testing.assert_allclose(rows.resolve(request)[KEY], np.stack(frames[3:]), atol=6)


@pytest.mark.parametrize('fps', [0.4, 0.5, 2.5, 29.97])
def test_an_h264_stream_of_any_positive_rate_carries_each_frame(fps):
    encoder = _VideoEncoder(24, 16, crf=18, fps=fps)
    frames = [_gradient(i) for i in range(3)]
    rows = StreamedRows()
    rows.add(_row(1, 2, encoder.encode(frames[0])))
    rows.add(_row(1, 1, encoder.encode(frames[1])))
    request = _wire({KEY: {REF: True, b'call': 1, b'rows': [2, 1], b'tail': encoder.encode(frames[2])}})
    np.testing.assert_allclose(rows.resolve(request)[KEY], np.stack(frames), atol=6)


def test_a_server_decodes_an_h264_frame_of_odd_sides_to_its_declared_shape():
    encoder = _VideoEncoder(23, 15, crf=18, fps=10.0)
    rows = StreamedRows()
    rows.add(_row(1, 1, encoder.encode(_gradient(0)[:15, :23])))
    request = _wire({KEY: {REF: True, b'call': 1, b'rows': [1], b'tail': encoder.encode(_gradient(1)[:15, :23])}})
    assert rows.resolve(request)[KEY].shape == (2, 15, 23, 3)


@pytest.mark.parametrize(
    'row, match',
    [
        ({'call': 1, 'back': 0, 'images': {}}, 'frames back'),
        ({'call': 1, 'back': StreamedRows.MAX_BACK + 1, 'images': {}}, 'frames back'),
        ({'call': 1, 'back': 1, 'images': {str(i): None for i in range(StreamedRows.MAX_IMAGES + 1)}}, 'images'),
        ({'call': 1, 'back': 1, 'images': {KEY: 'pixels'}}, 'JPEG frame or an H.264 unit'),
        ({'call': 1, 'back': 1, 'images': {KEY: {H264_FRAME: True, b'data': b'', b'shape': [0, 4, 3]}}}, 'outside'),
        # A JPEG frame, as the message decodes it.
        ({'call': 1, 'back': 1, 'images': {KEY: np.zeros((StreamedRows.MAX_SIDE + 1, 4, 3), np.uint8)}}, 'outside'),
    ],
)
def test_a_server_refuses_a_malformed_row(row, match):
    with pytest.raises(ValueError, match=match):
        StreamedRows().add(row)


def test_a_stack_that_refers_to_a_row_twice_is_refused():
    rows = StreamedRows()
    rows.add(_row(1, 1, encode_jpeg(_gradient(0))))
    with pytest.raises(ValueError, match='once'):
        rows.resolve(_wire({KEY: {REF: True, b'call': 1, b'rows': [1, 1], b'tail': encode_jpeg(_gradient(1)[None])}}))


def test_a_row_past_the_held_bytes_is_dropped_and_its_request_misses_it(monkeypatch):
    monkeypatch.setattr(StreamedRows, 'MAX_HELD_BYTES', _gradient(0).nbytes)
    rows = StreamedRows()
    rows.add(_row(1, 2, encode_jpeg(_gradient(0))))
    rows.add(_row(1, 1, encode_jpeg(_gradient(1))))
    with pytest.raises(RowsMissing) as raised:
        rows.resolve(_wire({KEY: {REF: True, b'call': 1, b'rows': [2, 1], b'tail': encode_jpeg(_gradient(2)[None])}}))
    assert raised.value.missing == [[KEY, 1]]


def test_an_rtc_stream_targets_the_rows_from_the_submit_time_plus_call_after():
    # OFFSETS (-0.25, -0.15, 0.0): the next call is due at submit + 0.8 s, its rows at +0.55 s and +0.65 s.
    with _streamer(call_after_sec=0.8) as (streamer, clock, sent):
        _tick(streamer, clock, *range(3))
        streamer.on_answer([{}] * 24)  # the first answer, read at 0.2 s: the schedule calls again at 1.0 s
        _tick(streamer, clock, *range(3, 11))
        streamer.on_submit()  # the call on tick 10 at 1.0 s, due again at 1.8 s
        streamer.flush()
        assert _rows(sent) == [(1, 2, 7), (1, 1, 8)]
        _tick(streamer, clock, 11, 12, 13)
        streamer.on_answer([{}] * 24)  # read at 1.3 s: rows at 1.55 s and 1.65 s
        _tick(streamer, clock, 14, 15, 16, 17)
        streamer.flush()
        assert _rows(sent)[2:] == [(2, 2, 15), (2, 1, 16)]


def test_an_rtc_answer_read_in_time_sends_the_rows_whose_targets_have_passed_at_once():
    with _streamer(call_after_sec=0.3) as (streamer, clock, sent):
        _tick(streamer, clock, 0)
        streamer.on_answer([{}] * 24)
        _tick(streamer, clock, 1, 2)
        streamer.on_submit()  # at 0.2 s: due again at 0.5 s, its rows at 0.25 s and 0.35 s
        _tick(streamer, clock, 3, 4)
        streamer.on_answer([{}] * 24)  # read at 0.4 s, before 0.5 s: both rows are due already
        streamer.flush()
        assert _rows(sent)[-2:] == [(2, 2, 2), (2, 1, 3)]


def test_an_rtc_answer_read_after_the_predicted_call_plans_no_rows():
    with _streamer(call_after_sec=0.3) as (streamer, clock, sent):
        _tick(streamer, clock, 0)
        streamer.on_answer([{}] * 24)
        _tick(streamer, clock, 1, 2)
        streamer.on_submit()  # at 0.2 s: due again at 0.5 s
        _tick(streamer, clock, 3, 4, 5, 6)
        streamer.flush()
        before = len(sent)
        streamer.on_answer([{}] * 24)  # read at 0.6 s: the schedule calls at once
        _tick(streamer, clock, 7, 8)
        streamer.flush()
        assert len(sent) == before
        assert streamer.counts['unplanned'] == 1
        request = streamer.request({KEY: np.stack([_frame(5), _frame(7), _frame(8)])})
        assert request[KEY][b'rows'] == []


def test_snap_refers_to_every_streamed_row_near_the_predicted_time():
    with _streamed_call_1(snap_sec=0.05) as (streamer, clock, _):
        clock.ns = 1_520_000_000  # 20 ms after the predicted 1.5 s
        stack = np.stack([_frame(11), _frame(13), _frame(15)])  # the stack picked tick 11 where the stream sent 12
        value = streamer.request({KEY: stack})[KEY]
        assert value[b'rows'] == [2, 1]
        assert streamer.counts['snapped'] == 1
        np.testing.assert_allclose(_decoded(value[b'tail']), stack[2:], atol=2)


def test_snap_does_not_apply_past_its_tolerance_nor_without_it():
    for snap, at_ns in ((0.05, 1_560_000_000), (0.0, 1_520_000_000)):
        with _streamed_call_1(snap_sec=snap) as (streamer, clock, _):
            clock.ns = at_ns
            stack = np.stack([_frame(11), _frame(13), _frame(15)])
            assert streamer.request({KEY: stack})[KEY][b'rows'] == []
            assert streamer.counts['snapped'] == 0


def test_snap_refers_only_to_rows_of_the_stack_s_shape():
    with _streamed_call_1(snap_sec=0.05) as (streamer, clock, _):
        clock.ns = 1_500_000_000
        stack = np.zeros((3, 4, 4, 3), np.uint8)
        assert streamer.request({KEY: stack})[KEY][b'rows'] == []


def test_snap_reads_the_submit_time_not_the_time_the_request_is_built():
    with _streamed_call_1(snap_sec=0.03) as (streamer, clock, _):
        clock.ns = 1_510_000_000
        streamer.on_submit()  # the call at 1.51 s; its codecs run until 1.56 s
        clock.ns = 1_560_000_000
        stack = np.stack([_frame(11), _frame(13), _frame(15)])
        assert streamer.request({KEY: stack})[KEY][b'rows'] == [2, 1]


@contextmanager
def _fed_call_1(**overrides: Any) -> Iterator[tuple[FrameStreamer, list[dict[str, np.ndarray]]]]:
    """``_streamed_call_1`` with the ticks given as the entries of a temporal stack's buffer (``on_entry``)."""
    with _streamer(**overrides) as (streamer, clock, _):
        entries = []
        for index in range(16):
            clock.ns = index * 100_000_000
            entries.append({KEY: _frame(index)})
            assert streamer.on_entry(clock.ns / 1e9, entries[-1]) == (KEY,)
            if index == 10:
                streamer.on_answer([{}] * 5)
        streamer.flush()
        yield streamer, entries


def test_a_stack_entry_feed_streams_the_entries_that_temporal_stack_samples():
    with _fed_call_1() as (streamer, entries):
        rows = streamer._streamed_entries
        assert rows[2] is entries[12] and rows[1] is entries[13]


def test_a_request_refers_to_the_rows_of_stack_rows_that_sample_the_streamed_entries():
    with _fed_call_1() as (streamer, entries):
        value = streamer.request({KEY: StackRows(KEY, [entries[12], entries[13], entries[15]])})[KEY]
        assert (value[b'call'], value[b'rows']) == (1, [2, 1])
        # Equal pixels in another entry: not the streamed row, so it goes inline.
        value = streamer.request({KEY: StackRows(KEY, [entries[12], {KEY: _frame(13)}, entries[15]])})[KEY]
        assert value[b'rows'] == [2]


@pytest.mark.parametrize('codec', [FrameCodec.JPEG, FrameCodec.H264])
@pytest.mark.parametrize('picks', [(12, 13, 15), (12, 14, 15), (11, 13, 15)], ids=['all', 'one', 'none'])
def test_a_request_of_stack_rows_carries_the_bytes_that_the_stacked_array_carries(codec, picks):
    resize = TransformImages(6, 4, mode=ResizeMode.BOUND)
    with (
        _fed_call_1(codec=codec, resize=resize.encode) as (fed, entries),
        _streamed_call_1(codec=codec, resize=resize.encode) as (stacked, _, _),
    ):
        rows = resize.encode({KEY: StackRows(KEY, [entries[i] for i in picks])})[KEY]
        array = resize.encode({KEY: np.stack([_frame(i) for i in picks])})[KEY]
        expected = stacked.request({KEY: array})[KEY]
        assert serialization.serialise(fed.request({KEY: rows})[KEY]) == serialization.serialise(expected)


def test_a_request_sizes_only_the_rows_of_stack_rows_that_it_carries():
    sized = []
    resize = TransformImages(6, 4, mode=ResizeMode.BOUND)

    def size(frames: list[np.ndarray]) -> list[np.ndarray]:
        sized.append(len(frames))
        return [resize.encode({KEY: frame})[KEY] for frame in frames]

    with _fed_call_1(resize=resize.encode) as (streamer, entries):
        rows = StackRows(KEY, [entries[12], entries[13], entries[15]]).sized_by(size, resize.transformed_shape)
        value = streamer.request({KEY: rows})[KEY]
        assert value[b'rows'] == [2, 1]
        assert sized == [1]
        assert rows.row_shape(0) == (4, 6, 3)


def test_snap_refers_to_the_streamed_rows_of_stack_rows_by_shape():
    with _fed_call_1(snap_sec=0.05) as (streamer, entries):
        rows = StackRows(KEY, [entries[11], entries[13], entries[15]])  # tick 11 where the stream sent 12
        assert streamer.request({KEY: rows})[KEY][b'rows'] == [2, 1]
        assert streamer.counts['snapped'] == 1


def test_a_row_whose_entry_has_left_the_feed_is_not_streamed():
    with _streamer(call_after_sec=0.3) as (streamer, clock, sent):
        streamer.on_entry(0.0, {KEY: _frame(0)})
        streamer.on_answer([{}] * 24)  # the first answer at 0.0 s: the next call at 0.3 s, its rows at 0.05 s, 0.15 s
        with streamer._lock:
            streamer._targets = [(2, 0.05), (1, 0.15)]
        for index in range(1, 20):  # the rows were not due until the entry of 0.05 s had left
            streamer._record(index * 0.1, {KEY: _frame(index)}, keep_sec=0.25)
        streamer._dispatch_due(1.9)
        streamer.flush()
        assert sent == []
