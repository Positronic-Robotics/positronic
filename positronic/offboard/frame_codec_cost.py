"""Measure what one observation window costs per image codec: bytes, encode time, decode time.

Replays a recorded episode's cameras through the rig-side bound, then encodes each temporal-stack
window as one JPEG per frame, which the wire carries, and as one h264 GOP. JPEG encodes
single-threaded through ``encode_jpeg``; h264 encodes with x264's own frame threading.

Usage
  python -m positronic.offboard.frame_codec_cost --episode <dir holding <camera>.mp4>
  python -m positronic.offboard.frame_codec_cost --episode <dir> --bound 640x180 --json rows.json
"""

import argparse
import dataclasses
import io
import json
import pathlib
import statistics
import time

import av
import numpy as np

from positronic.policy.codec import RestrictImageSize
from positronic.utils.serialization import FRAMES, encode_jpeg, unpack


@dataclasses.dataclass(frozen=True)
class Jpeg:
    """One JPEG per frame, at the quality the wire itself encodes at."""

    name = 'jpeg'

    def cost(self, window: np.ndarray) -> tuple[int, float, float]:
        """Bytes, encode ms and decode ms for one window."""
        start = time.perf_counter()
        marker = encode_jpeg(window)
        encode_ms = 1000 * (time.perf_counter() - start)

        start = time.perf_counter()
        unpack(marker)
        decode_ms = 1000 * (time.perf_counter() - start)
        return sum(len(buf) for buf in marker[FRAMES]), encode_ms, decode_ms


@dataclasses.dataclass(frozen=True)
class H264:
    """One x264 setting, over the whole window as a single GOP."""

    preset: str
    crf: int

    @property
    def name(self) -> str:
        return f'h264 {self.preset} crf{self.crf}'

    def cost(self, window: np.ndarray) -> tuple[int, float, float]:
        """Bytes, encode ms and decode ms for one window."""
        height, width = window.shape[1:3]
        buffer = io.BytesIO()
        start = time.perf_counter()
        with av.open(buffer, 'w', format='mp4') as container:
            stream = container.add_stream('libx264', rate=15)
            stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
            # One self-contained GOP with no lookahead: a request carries its own window and waits on it.
            stream.options = {'preset': self.preset, 'crf': str(self.crf), 'tune': 'zerolatency', 'g': str(len(window))}
            for frame in window:
                container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')))
            container.mux(stream.encode(None))
        encode_ms = 1000 * (time.perf_counter() - start)
        payload = buffer.getvalue()

        start = time.perf_counter()
        with av.open(io.BytesIO(payload), 'r') as container:
            for frame in container.decode(video=0):
                frame.to_ndarray(format='rgb24')
        return len(payload), encode_ms, 1000 * (time.perf_counter() - start)


Codec = Jpeg | H264


@dataclasses.dataclass(frozen=True)
class Cost:
    """What one window of one camera cost under one codec."""

    codec: str
    camera: str
    window: int
    kib: float
    encode_ms: float
    decode_ms: float


def bounded_frames(mp4: pathlib.Path, bound: RestrictImageSize, rate_hz: float) -> list[np.ndarray]:
    """Every frame the stack samples, through the rig's own bound."""
    key = 'image'
    with av.open(str(mp4), 'r') as container:
        recorded_rate = container.streams.video[0].average_rate
        if recorded_rate is None:
            raise ValueError(f'{mp4} declares no frame rate, so the sampled frames cannot be chosen')
        step = max(1, round(float(recorded_rate) / rate_hz))
        frames = []
        for index, frame in enumerate(container.decode(video=0)):
            if index % step:
                continue
            frames.append(bound.encode({key: frame.to_ndarray(format='rgb24')})[key])
    return frames


def costs(frames: list[np.ndarray], camera: str, depth: int, codecs: list[Codec], limit: int) -> list[Cost]:
    """One row per codec per window, over the windows the episode holds."""
    starts = list(range(0, len(frames) - depth + 1))[: limit or None]
    rows = []
    for window_index, start in enumerate(starts):
        window = np.stack(frames[start : start + depth])
        for codec in codecs:
            size, encode_ms, decode_ms = codec.cost(window)
            rows.append(Cost(codec.name, camera, window_index, size / 1024, encode_ms, decode_ms))
    return rows


def _row(label: str, group: list[Cost], windows: list[int]) -> str:
    """One printed line: the median over windows of what the group cost in each window."""
    per_window = [[row for row in group if row.window == window] for window in windows]
    return (
        f'{label:>16}  '
        f'{statistics.median(sum(row.kib for row in w) for w in per_window):8.0f}  '
        f'{statistics.median(sum(row.encode_ms for row in w) for w in per_window):10.1f}  '
        f'{statistics.median(sum(row.decode_ms for row in w) for w in per_window):10.1f}'
    )


def report(rows: list[Cost], depth: int) -> None:
    """Median cost per codec, per camera and summed over the cameras one request carries."""
    names = list(dict.fromkeys(row.codec for row in rows))
    cameras = list(dict.fromkeys(row.camera for row in rows))
    windows = sorted({row.window for row in rows})
    print(f'\n{depth} frames, {len(windows)} windows, {len(cameras)} cameras\n')
    print(f'{"codec":>22}  {"camera":>16}  {"KiB":>8}  {"encode ms":>10}  {"decode ms":>10}')
    for name in names:
        of_codec = [row for row in rows if row.codec == name]
        for camera in cameras:
            print(f'{name:>22}  {_row(camera, [row for row in of_codec if row.camera == camera], windows)}')
        print(f'{name:>22}  {_row("every camera", of_codec, windows)}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', required=True, type=pathlib.Path, help='directory holding <camera>.mp4')
    parser.add_argument('--cameras', default='image.exterior,image.wrist')
    parser.add_argument('--frames', type=int, default=25, help='temporal-stack depth')
    parser.add_argument('--rate', type=float, default=15.0, help='stack sampling rate, Hz')
    parser.add_argument('--bound', default='1024x288', help='WxH rig-side bound')
    parser.add_argument('--x264', default='ultrafast:20,veryfast:20', help='comma-separated preset:crf')
    parser.add_argument('--windows', type=int, default=100, help='windows per camera, 0 for every one')
    parser.add_argument('--json', type=pathlib.Path, help='write every row here')
    args = parser.parse_args()

    bound = RestrictImageSize(*(int(side) for side in args.bound.lower().split('x')))
    codecs: list[Codec] = [Jpeg()]
    codecs += [H264(spec.split(':')[0], int(spec.split(':')[1])) for spec in args.x264.split(',')]
    rows: list[Cost] = []
    for camera in args.cameras.split(','):
        frames = bounded_frames(args.episode / f'{camera}.mp4', bound, args.rate)
        print(f'{camera}: {len(frames)} sampled frames at {frames[0].shape[1]}x{frames[0].shape[0]}')
        rows += costs(frames, camera, args.frames, codecs, args.windows)

    report(rows, args.frames)
    if args.json:
        args.json.write_text(json.dumps([dataclasses.asdict(row) for row in rows]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
