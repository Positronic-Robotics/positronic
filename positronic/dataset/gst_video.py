"""H.264 into MP4 through one ``gst-launch-1.0`` child process per file, fed raw frames on its stdin."""

import enum
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .video import VideoEncoderSession

GST_LAUNCH = 'gst-launch-1.0'
GST_INSPECT = 'gst-inspect-1.0'
# The elements around ``GstH264Encoder.encode``: raw frames in from stdin, MP4 out to the file
_FRAME_ELEMENTS = ('fdsrc', 'rawvideoparse', 'h264parse', 'mp4mux', 'filesink')


class RawFormat(enum.Enum):
    """The order of the four bytes of each pixel that the child process reads."""

    RGBA = 'rgba'
    BGRX = 'bgrx'


@dataclass(frozen=True)
class GstH264Encoder:
    """Encodes H.264 into MP4 in a ``gst-launch-1.0`` child process, one process per file."""

    # gst-launch tokens that turn raw pixels into an H.264 stream. ``str.format`` fills ``{bitrate_bps}``,
    # ``{bitrate_kbps}`` and ``{gop}`` (in frames).
    encode: tuple[str, ...]
    # The elements that ``encode`` names, which ``ensure_available`` looks for
    elements: tuple[str, ...]
    # The rate-control budget for one frame. The bitrate is this times the nominal frame rate of the file.
    frame_bits: int
    raw_format: RawFormat = RawFormat.RGBA
    finish_timeout_s: float = 60.0

    def ensure_available(self) -> None:
        tools = [tool for tool in (GST_LAUNCH, GST_INSPECT) if shutil.which(tool) is None]
        if tools:
            raise RuntimeError(f'{", ".join(tools)} not on PATH, so this host cannot run {self!r}')
        missing = [
            element
            for element in (*_FRAME_ELEMENTS, *self.elements)
            if subprocess.run([GST_INSPECT, element], capture_output=True).returncode != 0
        ]
        if missing:
            raise RuntimeError(
                f'GStreamer elements {", ".join(missing)} are absent on this host. Set a software video_encoder, '
                'for example @positronic.cfg.video_encoder.libx264_veryfast'
            )

    def launch_command(self, path: Path, width: int, height: int, fps: int, gop: int) -> list[str]:
        bitrate = self.frame_bits * fps
        values = {'bitrate_bps': bitrate, 'bitrate_kbps': bitrate // 1000, 'gop': gop}
        return [
            GST_LAUNCH,
            '-q',
            *('fdsrc', 'fd=0', '!'),
            *('rawvideoparse', f'width={width}', f'height={height}', f'format={self.raw_format.value}'),
            *(f'framerate={fps}/1', '!'),
            *(token.format(**values) for token in self.encode),
            *('!', 'h264parse', '!', 'video/x-h264,stream-format=avc,alignment=au'),
            *('!', 'mp4mux', '!', 'filesink', f'location={path}'),
        ]

    def open(self, path: Path, width: int, height: int, fps: int, gop: int) -> VideoEncoderSession:
        return _GstSession(self.launch_command(path, width, height, fps, gop), width, height, self)


class _GstSession:
    def __init__(self, command: list[str], width: int, height: int, encoder: GstH264Encoder):
        self._timeout_s = encoder.finish_timeout_s
        self._swap_red_blue = encoder.raw_format is RawFormat.BGRX
        self._pixels = np.full((height, width, 4), 255, dtype=np.uint8)
        # A file, not a pipe: nothing reads stderr before the process exits, and a full pipe would stall it.
        self._stderr = tempfile.TemporaryFile()
        # Unbuffered, so a killed process leaves no pending bytes for ``close`` to flush.
        self._process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr, bufsize=0
        )

    def write(self, frame: np.ndarray, index: int) -> None:
        self._pixels[..., :3] = frame[..., ::-1] if self._swap_red_blue else frame
        stdin = self._process.stdin
        assert stdin is not None
        pending = self._pixels.data.cast('B')
        try:
            while pending:
                pending = pending[stdin.write(pending) :]
        except BrokenPipeError as e:
            raise self._failure(self._process.wait(self._timeout_s)) from e

    def finish(self) -> None:
        stdin = self._process.stdin
        assert stdin is not None
        stdin.close()
        try:
            code = self._process.wait(self._timeout_s)
        except subprocess.TimeoutExpired as e:
            self.abort()
            raise RuntimeError(f'{GST_LAUNCH} did not finish the file in {self._timeout_s} s') from e
        if code != 0:
            raise self._failure(code)
        self._stderr.close()

    def abort(self) -> None:
        self._process.kill()
        self._process.wait()
        assert self._process.stdin is not None
        self._process.stdin.close()
        self._stderr.close()

    def _failure(self, code: int) -> RuntimeError:
        self._stderr.seek(0)
        message = self._stderr.read().decode(errors='replace').strip()
        self._stderr.close()
        return RuntimeError(f'{GST_LAUNCH} exited with code {code}: {message}')
