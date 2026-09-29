import stat
from pathlib import Path

import av
import numpy as np
import pytest

from positronic.cfg.video_encoder import jetson_h264
from positronic.dataset.gst_video import GST_INSPECT, GST_LAUNCH, GstH264Encoder, RawFormat
from positronic.dataset.video import VideoSignal, VideoSignalWriter

# The software stand-in for the Jetson chain, with the same frame budget
SOFTWARE_H264 = GstH264Encoder(
    encode=(
        *('videoconvert', '!', 'video/x-raw,format=I420', '!'),
        *('x264enc', 'bitrate={bitrate_kbps}', 'key-int-max={gop}', 'bframes=0', 'speed-preset=ultrafast'),
    ),
    elements=('videoconvert', 'x264enc'),
    frame_bits=33_333,
)


def _software_chain_gap() -> str | None:
    try:
        SOFTWARE_H264.ensure_available()
    except RuntimeError as e:
        return str(e)
    return None


_GAP = _software_chain_gap()
needs_gstreamer = pytest.mark.skipif(
    _GAP is not None, reason=f'needs gst-launch-1.0 with the base, good, bad and ugly plugins: {_GAP}'
)


def _textured_frame(index: int, height: int = 120, width: int = 160) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    frame = np.stack([(x * 255) // width, (y * 255) // height, np.full_like(x, 128)], axis=-1).astype(np.uint8)
    top, left = (index * 3) % (height - 20), (index * 5) % (width - 20)
    frame[top : top + 20, left : left + 20] = (index * 37) % 256
    return frame


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255**2 / mse))


def _write(tmp_path: Path, encoder: GstH264Encoder, frames: list[np.ndarray]) -> VideoSignal:
    video, index = tmp_path / 'cam.mp4', tmp_path / 'cam.frames.parquet'
    with VideoSignalWriter(video, index, encoder) as w:
        for i, frame in enumerate(frames):
            w.append(frame, 1_000_000_000 + i * 33_333_333)
    return VideoSignal(video, index)


def _fake_tools(directory: Path, absent: set[str]) -> None:
    directory.mkdir()
    inspect = f'case "$1" in {"|".join(absent)}) exit 1;; esac\nexit 0' if absent else 'exit 0'
    for tool, script in ((GST_LAUNCH, 'exit 0'), (GST_INSPECT, inspect)):
        path = directory / tool
        path.write_text(f'#!/bin/sh\n{script}\n')
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_jetson_command_pins_the_elements_the_caps_and_the_bitrate():
    encoder = jetson_h264.instantiate()
    command = encoder.launch_command(Path('/data/ep/cam.mp4'), 800, 600, 100, 30)

    assert ' '.join(command) == (
        'gst-launch-1.0 -q fdsrc fd=0 ! rawvideoparse width=800 height=600 format=bgrx framerate=100/1 ! '
        'nvvidconv ! video/x-raw(memory:NVMM),format=I420 ! nvv4l2h264enc control-rate=1 bitrate=3333300 '
        'iframeinterval=30 idrinterval=30 insert-sps-pps=true num-B-Frames=0 ! h264parse ! '
        'video/x-h264,stream-format=avc,alignment=au ! mp4mux ! filesink location=/data/ep/cam.mp4'
    )


def test_the_bitrate_scales_with_the_nominal_rate_so_the_frame_budget_holds():
    encoder = jetson_h264.instantiate()
    for fps in (30, 100):
        (bitrate,) = [t for t in encoder.launch_command(Path('x.mp4'), 8, 8, fps, 30) if t.startswith('bitrate=')]
        assert bitrate == f'bitrate={encoder.frame_bits * fps}'


def test_a_host_without_gst_launch_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(tmp_path))
    with pytest.raises(RuntimeError, match=f'{GST_LAUNCH}, {GST_INSPECT} not on PATH'):
        jetson_h264.instantiate().ensure_available()


def test_a_host_without_the_encoder_is_refused_with_its_name(tmp_path, monkeypatch):
    _fake_tools(tmp_path / 'bin', absent={'nvv4l2h264enc'})
    monkeypatch.setenv('PATH', str(tmp_path / 'bin'))
    with pytest.raises(RuntimeError, match=r'GStreamer elements nvv4l2h264enc are absent') as refused:
        jetson_h264.instantiate().ensure_available()
    assert 'libx264_veryfast' in str(refused.value)


def test_a_host_with_every_element_passes(tmp_path, monkeypatch):
    _fake_tools(tmp_path / 'bin', absent=set())
    monkeypatch.setenv('PATH', str(tmp_path / 'bin'))
    jetson_h264.instantiate().ensure_available()


@needs_gstreamer
def test_every_frame_reads_back_at_its_index_and_timestamp(tmp_path):
    frames = [_textured_frame(i) for i in range(45)]
    signal = _write(tmp_path, SOFTWARE_H264, frames)

    assert len(signal) == len(frames)
    np.testing.assert_array_equal(signal.keys(), [1_000_000_000 + i * 33_333_333 for i in range(45)])
    decoded = signal.values()
    for i in (0, 44, 3, 31, 30, 29, 12):
        assert _psnr(decoded[i], frames[i]) > 30, f'frame {i}'


@needs_gstreamer
def test_the_file_has_no_b_frames_a_keyframe_every_gop_and_a_nominal_rate_of_100(tmp_path):
    _write(tmp_path, SOFTWARE_H264, [_textured_frame(i) for i in range(65)])

    with av.open(str(tmp_path / 'cam.mp4')) as container:
        (stream,) = container.streams.video
        assert stream.average_rate == 100
        assert not stream.codec_context.has_b_frames
        keyframes = [p.pts for p in container.demux(stream) if p.is_keyframe and p.pts is not None]
        assert stream.time_base is not None
        ticks_per_frame = round(1 / (100 * stream.time_base))
    assert [pts // ticks_per_frame for pts in keyframes] == [0, 30, 60]


@needs_gstreamer
@pytest.mark.parametrize('raw_format', list(RawFormat))
def test_flat_colours_keep_their_channels(tmp_path, raw_format):
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (40, 40, 40), (200, 120, 30)]
    encoder = GstH264Encoder(SOFTWARE_H264.encode, SOFTWARE_H264.elements, SOFTWARE_H264.frame_bits, raw_format)
    signal = _write(tmp_path, encoder, [np.full((64, 64, 3), c, dtype=np.uint8) for c in colours])

    for frame, colour in zip(signal.values(), colours, strict=True):
        np.testing.assert_allclose(frame.reshape(-1, 3).mean(axis=0), colour, atol=6, err_msg=f'colour {colour}')


@needs_gstreamer
def test_a_pipeline_that_fails_surfaces_its_error(tmp_path):
    broken = GstH264Encoder(('nosuchelement',), (), SOFTWARE_H264.frame_bits)
    with pytest.raises(RuntimeError, match='Video encoding failed') as failed:
        with VideoSignalWriter(tmp_path / 'cam.mp4', tmp_path / 'cam.frames.parquet', broken) as w:
            for i in range(20):
                w.append(_textured_frame(i), i + 1)
    assert 'nosuchelement' in str(failed.value.__cause__)


@needs_gstreamer
def test_abort_stops_the_process_and_deletes_the_files(tmp_path):
    video, index = tmp_path / 'cam.mp4', tmp_path / 'cam.frames.parquet'
    w = VideoSignalWriter(video, index, SOFTWARE_H264)
    for i in range(5):
        w.append(_textured_frame(i), i + 1)
    w.abort()

    assert not video.exists()
    assert not index.exists()
